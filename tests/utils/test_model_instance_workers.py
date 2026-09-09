"""Placing a worker within a model instance.

Two call sites ask "which part of this instance does this worker run" when a
worker goes away, and answer it by scanning every instance in the cluster and
walking its embedded subordinate list. A workload row already is that answer,
and it is indexed by worker.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.utils.model_instance_workers import (
    ModelInstanceWorkerMatch,
    _binding_of,
    instance_placements,
    placements_from_workloads,
    get_worker_matches_from_workloads,
    report_match_disagreement,
)


def _row(worker_id, owner_id, group_index):
    return SimpleNamespace(
        worker_id=worker_id, owner_id=owner_id, group_index=group_index
    )


async def _matches(rows, worker_ids):
    with patch(
        "gpustack.utils.model_instance_workers.Workload.all_by_fields",
        AsyncMock(return_value=rows),
    ):
        return await get_worker_matches_from_workloads(MagicMock(), worker_ids)


@pytest.mark.asyncio
async def test_group_index_zero_is_the_instances_own_worker():
    matches = await _matches([_row(7, 1, 0)], [7])

    assert matches[7][1] == ModelInstanceWorkerMatch(
        is_main_worker=True, subordinate_worker_indexes=()
    )


@pytest.mark.asyncio
async def test_a_follower_maps_back_to_its_position_in_the_embedded_list():
    """group_index counts the leader, the embedded list does not, so follower
    i sits at group_index i+1. Getting this off by one would mark the wrong
    subordinate unreachable."""
    matches = await _matches([_row(7, 1, 2)], [7])

    assert matches[7][1] == ModelInstanceWorkerMatch(
        is_main_worker=False, subordinate_worker_indexes=(1,)
    )


@pytest.mark.asyncio
async def test_a_worker_can_hold_both_roles_of_one_instance():
    """A follower can be scheduled onto the leader's own worker, which is why
    the unique constraint includes the group position."""
    matches = await _matches([_row(7, 1, 0), _row(7, 1, 1)], [7])

    assert matches[7][1] == ModelInstanceWorkerMatch(
        is_main_worker=True, subordinate_worker_indexes=(0,)
    )


@pytest.mark.asyncio
async def test_workers_are_resolved_in_one_pass():
    """The callers ask about every offline worker at once; a query each would
    be worse than the scan this replaces."""
    rows = [_row(7, 1, 0), _row(8, 1, 1), _row(8, 2, 0)]

    matches = await _matches(rows, [7, 8])

    assert set(matches) == {7, 8}
    assert set(matches[8]) == {1, 2}


@pytest.mark.asyncio
async def test_a_worker_with_nothing_on_it_is_absent():
    assert await _matches([_row(7, 1, 0)], [9]) == {}


def test_agreement_is_counted(caplog):
    """Not silent: an empty log would read the same as the comparison never
    having run, which is what "no rows compiled yet" looks like."""
    embedded = ModelInstanceWorkerMatch(is_main_worker=True)

    with caplog.at_level(logging.INFO):
        report_match_disagreement(
            SimpleNamespace(name="mi", id=1), 7, embedded, embedded
        )

    assert "differs on" not in caplog.text
    assert "agreed=" in caplog.text


def test_a_missing_row_is_not_a_disagreement(caplog):
    """An instance whose rows have not been compiled yet says nothing about
    whether the rows can be read instead; the caller falls back to the
    embedded list."""
    report_match_disagreement(
        SimpleNamespace(name="mi", id=1),
        7,
        ModelInstanceWorkerMatch(is_main_worker=True),
        None,
    )

    assert caplog.text == ""


def test_a_different_placement_is_reported(caplog):
    caplog.set_level(logging.INFO)
    report_match_disagreement(
        SimpleNamespace(name="mi", id=1),
        7,
        ModelInstanceWorkerMatch(is_main_worker=True),
        ModelInstanceWorkerMatch(subordinate_worker_indexes=(0,)),
    )

    assert "Workload placement differs on worker 7" in caplog.text


# ---------------------------------------------------------------------------
# The same placements, read off the rows
# ---------------------------------------------------------------------------


def _instance_with_follower():
    from gpustack.schemas.models import (
        ComputedResourceClaim,
        DistributedServerCoordinateModeEnum,
        DistributedServers,
        ModelInstance,
        ModelInstanceSubordinateWorker,
    )

    return ModelInstance(
        id=1,
        name="mi",
        worker_id=1,
        worker_name="w1",
        worker_ip="10.0.0.1",
        worker_ifname="eth0",
        gpu_type="cuda",
        gpu_indexes=[0],
        gpu_addresses=["0000:01:00.0"],
        computed_resource_claim=ComputedResourceClaim(vram={0: 100}),
        port=8000,
        ports=[8000, 8001],
        pid=42,
        distributed_servers=DistributedServers(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
            subordinate_workers=[
                ModelInstanceSubordinateWorker(
                    worker_id=2,
                    worker_name="w2",
                    worker_ip="10.0.0.2",
                    worker_ifname="eth1",
                    gpu_type="cuda",
                    gpu_indexes=[1],
                    gpu_addresses=["0000:02:00.0"],
                    computed_resource_claim=ComputedResourceClaim(vram={1: 200}),
                    ports=[9000],
                    pid=43,
                )
            ],
        ),
    )


def test_the_rows_place_the_workers_the_same_way():
    """The binding is what the readers moved onto this actually use, and it
    has to survive being read off a row -- where the claim is JSON and the
    ports are a name to port map rather than a list."""
    from gpustack.server.model_instance_workloads import compile_model_instance

    instance = _instance_with_follower()

    from_instance = instance_placements(instance)
    from_rows = placements_from_workloads(compile_model_instance(instance))

    assert _binding_of(from_instance) == _binding_of(from_rows)


def test_the_comparison_is_silent_when_the_two_agree(caplog):
    from gpustack.server.model_instance_workloads import compile_model_instance
    from gpustack.utils.comparison import tally

    # Process-global, so the reporting cadence carries over from whatever ran
    # before this.
    tally("Workload placements").reset()
    instance = _instance_with_follower()

    with caplog.at_level(logging.INFO):
        instance_placements(instance, compile_model_instance(instance))

    assert "differs on" not in caplog.text
    assert "agreed=" in caplog.text


def test_a_row_bound_elsewhere_is_reported(caplog):
    """Placing a worker wrongly is how the wrong node gets told to run
    something, so it has to be loud rather than absorbed."""
    from gpustack.server.model_instance_workloads import compile_model_instance

    instance = _instance_with_follower()
    rows = compile_model_instance(instance)
    rows[1].worker_id = 99

    with caplog.at_level(logging.INFO):
        instance_placements(instance, rows)

    assert "Workload placements differs on model instance mi" in caplog.text
