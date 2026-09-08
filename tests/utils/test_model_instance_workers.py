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


def test_agreement_is_silent(caplog):
    embedded = ModelInstanceWorkerMatch(is_main_worker=True)
    report_match_disagreement(SimpleNamespace(name="mi", id=1), 7, embedded, embedded)

    assert caplog.text == ""


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

    assert "place worker 7 differently" in caplog.text
