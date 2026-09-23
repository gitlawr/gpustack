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
    subordinate_placements,
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


# ---------------------------------------------------------------------------
# A follower's state, counted apart from its binding
# ---------------------------------------------------------------------------


def _followers_agreeing():
    from gpustack.schemas.models import ModelInstanceStateEnum
    from gpustack.server.model_instance_workloads import compile_model_instance
    from gpustack.utils.comparison import tally

    tally("Workload follower state").reset()
    instance = _instance_with_follower()
    instance.distributed_servers.subordinate_workers[0].state = (
        ModelInstanceStateEnum.RUNNING
    )
    return instance, compile_model_instance(instance)


def test_a_followers_state_is_counted(caplog):
    """Apart from the binding: the two are different claims about the rows,
    and one tally would let either pass on the other's evidence."""
    instance, rows = _followers_agreeing()

    with caplog.at_level(logging.INFO):
        subordinate_placements(instance, rows)

    assert "Workload follower state: agreed=1" in caplog.text


def test_a_followers_state_read_back_wrongly_is_reported(caplog):
    from gpustack.schemas.workloads import WorkloadStateEnum

    instance, rows = _followers_agreeing()
    rows[1].state = WorkloadStateEnum.ERROR

    with caplog.at_level(logging.INFO):
        subordinate_placements(instance, rows)

    assert "Workload follower state differs on" in caplog.text


def test_an_instance_with_no_followers_says_nothing_about_state(caplog):
    """Silence has to mean the rows had nothing to say, not that they
    agreed."""
    from gpustack.schemas.models import ModelInstance
    from gpustack.server.model_instance_workloads import compile_model_instance
    from gpustack.utils.comparison import tally

    tally("Workload follower state").reset()
    instance = ModelInstance(id=1, name="mi", worker_id=1)

    with caplog.at_level(logging.INFO):
        instance_placements(instance, compile_model_instance(instance))

    assert "Workload follower state" not in caplog.text


def test_the_leaders_state_is_never_read_off_its_row():
    """Five instance states mirror onto a pending workload, so reading one
    back would be a guess. Only the follower rows carry a state here."""
    from gpustack.utils.model_instance_workers import placements_from_workloads
    from gpustack.server.model_instance_workloads import compile_model_instance

    instance, _ = _followers_agreeing()
    from_rows = placements_from_workloads(compile_model_instance(instance))

    assert from_rows[0].state is None
    assert from_rows[1].state is not None


def test_an_instance_with_no_subordinates_is_not_counted_as_agreement(caplog):
    """Comparing nothing against nothing says nothing about whether a
    subordinate can be read off a row, and the count is what decides whether
    distributed has been covered at all."""
    from gpustack.schemas.models import ModelInstance
    from gpustack.server.model_instance_workloads import compile_model_instance
    from gpustack.utils.comparison import tally

    tally("Workload placements").reset()
    instance = ModelInstance(id=1, name="mi", worker_id=1)

    with caplog.at_level(logging.INFO):
        subordinate_placements(instance, compile_model_instance(instance))

    assert caplog.text == ""


def test_rows_claiming_a_subordinate_the_instance_does_not_have_are_reported(caplog):
    """Skipping the empty case must not swallow this: one side empty is a
    disagreement, not an absence."""
    from gpustack.schemas.models import ModelInstance
    from gpustack.server.model_instance_workloads import compile_model_instance

    rows = compile_model_instance(_instance_with_follower())
    bare = ModelInstance(id=1, name="mi", worker_id=1)

    with caplog.at_level(logging.INFO):
        subordinate_placements(bare, rows)

    assert "Workload placements differs on" in caplog.text


# ---------------------------------------------------------------------------
# Empty collections, in the spelling the callers take
# ---------------------------------------------------------------------------


def test_an_unscheduled_instance_reads_the_same_both_ways():
    """The first reading of every instance is of one with no binding yet, so
    a difference here is reported for each one and the tally never clears."""
    from gpustack.schemas.models import ModelInstance
    from gpustack.server.model_instance_workloads import compile_model_instance

    instance = ModelInstance(id=1, name="mi")

    assert _binding_of(instance_placements(instance)) == _binding_of(
        placements_from_workloads(compile_model_instance(instance))
    )


@pytest.mark.parametrize("field", ["gpu_indexes", "gpu_addresses", "ports"])
def test_a_collection_the_row_stores_as_null_reads_back_as_empty(field):
    """A row compiles an empty collection to NULL, the column's default.
    Several callers iterate or count these without a guard -- the binpack
    scorer over a subordinate's gpu_indexes, the backends sizing their
    ranks -- so None would raise where the list merely yields nothing."""
    from gpustack.schemas.models import ModelInstance
    from gpustack.server.model_instance_workloads import compile_model_instance

    rows = compile_model_instance(ModelInstance(id=1, name="mi"))
    assert getattr(rows[0], field) is None

    placement = placements_from_workloads(rows)[0]
    assert getattr(placement, field) == []
    # What the callers actually do with it.
    assert len(getattr(placement, field)) == 0
    assert list(getattr(placement, field)) == []
