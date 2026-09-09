"""
The model instance controller's workload sync.

Stage 3 step 1: the rows are compiled and written, and nothing reads them yet.
That is what makes this safe to land -- a wrong row is a wrong row rather than
a stopped container -- and what these tests hold it to.
"""

import contextlib
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.schemas.workloads import (
    Workload,
    WorkloadOwnerKindEnum,
    WorkloadStateEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack import envs
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.server import controllers as controllers_module
from gpustack.server.model_instance_workloads import FoldDeclineReason
from gpustack.server.controllers import (
    ModelInstanceController,
    ModelInstanceWorkloadStateController,
)

_ANY_INSTANCE = SimpleNamespace(id=3)


class _FakeSessionCtx:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *exc):
        return False


def _compiled(group_index):
    """A real Workload, so the spec/state split is exercised rather than
    stubbed: the update path has to build a WorkloadUpdate from it."""
    return Workload(
        name=f"mi-{group_index}",
        owner_kind=WorkloadOwnerKindEnum.MODEL_INSTANCE,
        owner_id=3,
        worker_id=1,
        group_index=group_index,
        state=WorkloadStateEnum.RUNNING,
    )


def _existing_workload(group_index):
    return SimpleNamespace(
        group_index=group_index,
        update=AsyncMock(),
        delete=AsyncMock(),
    )


@contextlib.contextmanager
def _workload_sync(monkeypatch, existing, compiled, instance=_ANY_INSTANCE):
    monkeypatch.setattr(
        "gpustack.server.controllers.async_session", lambda: _FakeSessionCtx()
    )
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.all_by_fields",
        AsyncMock(return_value=existing),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Workload.all_by_fields",
        AsyncMock(return_value=existing),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.ModelInstance.one_by_id",
        AsyncMock(return_value=instance),
    )
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.compile_model_instance",
        lambda mi: compiled,
    )
    create = AsyncMock()
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.create", create
    )
    yield create


def _event(event_type=EventType.UPDATED, instance_id=3):
    return Event(type=event_type, data={"id": instance_id})


@pytest.mark.asyncio
async def test_sync_creates_the_rows_that_are_missing(monkeypatch):
    with _workload_sync(
        monkeypatch, existing=[], compiled=[_compiled(0), _compiled(1)]
    ) as create:
        await ModelInstanceController(MagicMock())._sync_workloads(_event())

    assert create.await_count == 2


@pytest.mark.asyncio
async def test_sync_updates_rather_than_recreates(monkeypatch):
    """Idempotent: an instance that reports state on every health check must
    not churn its rows, or the id its logs are keyed by keeps moving."""
    existing = [_existing_workload(0)]

    with _workload_sync(
        monkeypatch, existing=existing, compiled=[_compiled(0)]
    ) as create:
        await ModelInstanceController(MagicMock())._sync_workloads(_event())

    create.assert_not_awaited()
    existing[0].update.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_drops_rows_the_instance_no_longer_has(monkeypatch):
    """A distributed instance that lost subordinate workers, or a backend that
    started delegating, leaves followers behind."""
    existing = [_existing_workload(0), _existing_workload(1), _existing_workload(2)]

    with _workload_sync(monkeypatch, existing=existing, compiled=[_compiled(0)]):
        await ModelInstanceController(MagicMock())._sync_workloads(_event())

    existing[0].update.assert_awaited_once()
    existing[1].delete.assert_awaited_once()
    existing[2].delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_deletes_every_row_with_the_instance(monkeypatch):
    existing = [_existing_workload(0), _existing_workload(1)]

    with _workload_sync(monkeypatch, existing=existing, compiled=[]):
        await ModelInstanceController(MagicMock())._sync_workloads(
            _event(EventType.DELETED)
        )

    for workload in existing:
        workload.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_never_fails_the_instances_own_reconcile(monkeypatch, caplog):
    """Nothing reads these rows yet, so a compile that goes wrong must not
    take the reconcile that does matter down with it."""
    monkeypatch.setattr(
        "gpustack.server.controllers.async_session", lambda: _FakeSessionCtx()
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Workload.all_by_fields",
        AsyncMock(side_effect=RuntimeError("db down")),
    )

    with caplog.at_level(logging.ERROR):
        await ModelInstanceController(MagicMock())._sync_workloads(_event())

    assert "Failed to sync workloads of model instance 3" in caplog.text


@pytest.mark.asyncio
async def test_sync_does_not_write_over_what_the_worker_reported(monkeypatch):
    """The worker mirrors execution state onto these rows as it goes.
    Recompiling it from the instance on every event would undo that, so an
    update carries the spec only."""
    existing = [_existing_workload(0)]

    with _workload_sync(monkeypatch, existing=existing, compiled=[_compiled(0)]):
        await ModelInstanceController(MagicMock())._sync_workloads(_event())

    update = existing[0].update.await_args.args[1]
    assert update.worker_id == 1
    assert "state" not in update.model_fields_set
    assert "state_message" not in update.model_fields_set
    assert "ports" not in update.model_fields_set
    assert "pid" not in update.model_fields_set


# ---------------------------------------------------------------------------
# Folding workload state back onto the instance (stage 3 step 2b)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _fold(monkeypatch, instance, folded, authoritative=False, workloads=None):
    monkeypatch.setattr(
        "gpustack.server.controllers.async_session", lambda: _FakeSessionCtx()
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.ModelInstance.one_by_id",
        AsyncMock(return_value=instance),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Workload.all_by_fields",
        AsyncMock(return_value=workloads if workloads is not None else []),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Worker.all_by_fields",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.aggregate_instance_state",
        lambda workloads, worker_ips=None: folded,
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.envs.MODEL_INSTANCE_STATE_FROM_WORKLOADS",
        authoritative,
    )
    yield


def _instance(**overrides):
    fields = dict(
        id=3, name="mi", state="running", state_message="", update=AsyncMock()
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.asyncio
async def test_fold_writes_nothing_while_it_is_only_being_compared(monkeypatch):
    """Off by default: the fold runs so it can be shown correct against real
    instances, without being the thing that decides them. A difference is not
    reported on sight either -- it is queued for confirmation."""
    instance = _instance(state="starting")
    controller = ModelInstanceWorkloadStateController()

    with _fold(monkeypatch, instance, folded={"state": "running"}):
        await controller._reconcile(3)

    instance.update.assert_not_awaited()
    assert controller._confirming == {3}
    assert controller._disagreed == 0
    # Nothing may outlive the test: the confirm re-reads through a session the
    # fixtures have taken away by then.
    for task in list(controller._confirm_tasks):
        task.cancel()


@pytest.mark.asyncio
async def test_a_difference_the_instance_comes_round_to_is_not_a_disagreement(
    monkeypatch, caplog
):
    """The fold read an outage the moment the server recorded it, before the
    pass that puts it on the instance. It proposed what the instance went on
    to say, so it was right and merely earlier."""
    instance = _instance(state="running")
    controller = ModelInstanceWorkloadStateController()
    monkeypatch.setattr(controllers_module, "_FOLD_CONFIRM_SECONDS", 0)

    with _fold(monkeypatch, instance, folded={"state": "running"}):
        with caplog.at_level(logging.INFO):
            await controller._confirm(3, {"state": ("starting", "running")})

    assert (controller._converged, controller._overtaken) == (1, 0)
    assert controller._disagreed == 0
    assert "disagrees" not in caplog.text


@pytest.mark.asyncio
async def test_a_proposal_that_never_came_true_is_reported(monkeypatch, caplog):
    """The difference is gone, but the instance never reached what the fold
    proposed -- the fold changed its mind instead. Judging by "do they still
    differ" would call this settled, and an authoritative fold would have
    written the proposal."""
    instance = _instance(state="error")
    controller = ModelInstanceWorkloadStateController()
    monkeypatch.setattr(controllers_module, "_FOLD_CONFIRM_SECONDS", 0)

    with _fold(monkeypatch, instance, folded={"state": "error"}):
        with caplog.at_level(logging.INFO):
            await controller._confirm(3, {"state": ("error", "running")})

    assert (controller._converged, controller._overtaken) == (0, 1)
    assert "was overtaken" in caplog.text


@pytest.mark.asyncio
async def test_a_difference_that_persists_is_reported(monkeypatch, caplog):
    instance = _instance(state="starting")
    controller = ModelInstanceWorkloadStateController()
    monkeypatch.setattr(controllers_module, "_FOLD_CONFIRM_SECONDS", 0)

    with _fold(monkeypatch, instance, folded={"state": "running"}):
        with caplog.at_level(logging.INFO):
            await controller._confirm(3, {"state": ("starting", "running")})

    assert controller._disagreed == 1
    assert "disagrees with model instance mi" in caplog.text


@pytest.mark.asyncio
async def test_a_rescheduled_instance_is_left_alone(monkeypatch):
    """Its workloads still carry the failure that caused the restart. Folding
    that back would put the instance into ERROR again and undo it."""
    instance = _instance(state="scheduled")
    controller = ModelInstanceWorkloadStateController()

    with _fold(monkeypatch, instance, folded={"state": "error"}, authoritative=True):
        await controller._reconcile(3)

    instance.update.assert_not_awaited()
    assert controller._declined == {FoldDeclineReason.INSTANCE_NOT_EXECUTING: 1}


@pytest.mark.asyncio
async def test_agreement_reports_no_disagreement(monkeypatch, caplog):
    instance = _instance(state="running")

    with _fold(monkeypatch, instance, folded={"state": "running"}):
        with caplog.at_level(logging.INFO):
            await ModelInstanceWorkloadStateController()._reconcile(3)

    assert "disagrees" not in caplog.text


@pytest.mark.asyncio
async def test_agreement_is_counted_so_an_empty_log_can_be_told_from_a_dead_one(
    monkeypatch, caplog
):
    """The gate for making the fold authoritative is "no disagreements", which
    a controller that never ran also satisfies. The tally is what separates the
    two, so agreement has to leave a trace."""
    instance = _instance(state="running")
    controller = ModelInstanceWorkloadStateController()

    with _fold(monkeypatch, instance, folded={"state": "running"}):
        with caplog.at_level(logging.INFO):
            await controller._reconcile(3)

    assert controller._agreed == {"running": 1}
    assert controller._disagreed == 0
    assert "agreed={running=1}" in caplog.text


@pytest.mark.asyncio
async def test_a_declined_group_is_counted_apart_from_agreement(monkeypatch):
    """A leader still pending is the fold declining to speak, not the fold
    agreeing; counting them together would read as coverage it does not have."""
    instance = _instance()
    controller = ModelInstanceWorkloadStateController()

    with _fold(monkeypatch, instance, folded=None):
        await controller._reconcile(3)

    assert (controller._agreed, controller._disagreed) == ({}, 0)
    assert sum(controller._declined.values()) == 1


@pytest.mark.asyncio
async def test_a_group_with_no_leader_is_reported_rather_than_just_counted(
    monkeypatch, caplog
):
    """Every group is compiled with a leader at group_index 0, so its absence
    is not a point in a normal start -- and it declines forever, which looks
    identical to a start that is merely slow."""
    instance = _instance()
    controller = ModelInstanceWorkloadStateController()

    with _fold(monkeypatch, instance, folded=None):
        with caplog.at_level(logging.WARNING):
            await controller._reconcile(3)

    assert controller._declined == {FoldDeclineReason.NO_LEADER: 1}
    assert "found no leader" in caplog.text


@pytest.mark.asyncio
async def test_fold_writes_once_it_is_authoritative(monkeypatch):
    instance = _instance(state="starting")

    with _fold(monkeypatch, instance, folded={"state": "running"}, authoritative=True):
        await ModelInstanceWorkloadStateController()._reconcile(3)

    instance.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_authoritative_fold_that_changes_nothing_does_not_write(monkeypatch):
    """The worker reports state on every health check; rewriting an unchanged
    row would publish an event per check to everything watching instances."""
    instance = _instance(state="running")

    with _fold(monkeypatch, instance, folded={"state": "running"}, authoritative=True):
        await ModelInstanceWorkloadStateController()._reconcile(3)

    instance.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_group_that_says_nothing_leaves_the_instance_alone(monkeypatch):
    instance = _instance()

    with _fold(monkeypatch, instance, folded=None, authoritative=True):
        await ModelInstanceWorkloadStateController()._reconcile(3)

    instance.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_run_of_declines_still_reports(monkeypatch, caplog):
    """Declines print nothing of their own, and an instance coming up declines
    on every event. Without the tally that stretch is indistinguishable from a
    controller that has stopped consuming events -- which is the reading the
    gate for flipping the fold depends on being able to rule out."""
    instance = _instance()
    controller = ModelInstanceWorkloadStateController()

    with _fold(monkeypatch, instance, folded=None):
        with caplog.at_level(logging.INFO):
            await controller._reconcile(3)

    assert "Workload fold [" in caplog.text
    assert "declined=" in caplog.text


@pytest.mark.asyncio
async def test_the_tally_names_the_states_it_agreed_about(monkeypatch, caplog):
    """The gate is coverage, not volume: a run that only ever agreed about
    RUNNING has said nothing about the states an instance passes through on
    the way there, and a single total cannot tell those apart."""
    controller = ModelInstanceWorkloadStateController()

    for state in ("error", "running", "running"):
        instance = _instance(state=state)
        with _fold(monkeypatch, instance, folded={"state": state}):
            with caplog.at_level(logging.INFO):
                await controller._reconcile(3)

    assert controller._agreed == {"error": 1, "running": 2}
    # The cadence widens, so the last line printed is the one at two events;
    # what matters is that a line names the states rather than a total.
    assert "agreed={error=1, running=1}" in caplog.text


@pytest.mark.asyncio
async def test_the_tally_separates_agreements_about_a_distributed_group(monkeypatch):
    """A single-worker instance never reaches _distributed_override, and both
    shapes agree about "running", so one counter cannot say whether the part
    of the fold most likely to be wrong was exercised at all."""
    controller = ModelInstanceWorkloadStateController()

    for followers in (0, 1):
        instance = _instance(state="running")
        with _fold(
            monkeypatch,
            instance,
            folded={"state": "running"},
            workloads=_group(followers),
        ):
            await controller._reconcile(3)

    assert controller._agreed == {"running": 2}
    assert controller._agreed_distributed == {"running": 1}


def _group(followers: int):
    return [
        SimpleNamespace(group_index=i, worker_id=100 + i) for i in range(followers + 1)
    ]


# ---------------------------------------------------------------------------
# An outage the worker cannot report
# ---------------------------------------------------------------------------


def _row(state, group_index=0):
    return SimpleNamespace(state=state, group_index=group_index, update=AsyncMock())


async def _mark(monkeypatch, rows):
    monkeypatch.setattr(
        controllers_module.Workload, "all_by_fields", AsyncMock(return_value=rows)
    )
    await controllers_module._mark_workloads_unreachable(MagicMock(), 3, 7)


@pytest.mark.asyncio
async def test_a_lost_workers_running_leader_is_marked_unreachable(monkeypatch):
    """The server marks an instance unreachable when it loses the worker, and
    the worker is precisely what cannot mirror that onto the workload rows.
    Left reporting RUNNING they would have the fold put the instance back to
    RUNNING and undo the outage."""
    leader = _row(WorkloadStateEnum.RUNNING)

    await _mark(monkeypatch, [leader])

    applied = leader.update.await_args[0][1]
    assert applied["state"] == WorkloadStateEnum.UNREACHABLE
    assert applied["state_message"] == "Worker is unreachable from the server"


@pytest.mark.asyncio
async def test_a_leader_still_coming_up_is_left_alone(monkeypatch):
    """The instance-side rule moves the main worker's entry only out of
    RUNNING, so marking a leader that is still starting says something the
    instance never will -- and the fold reading it reports a disagreement."""
    leader = _row(WorkloadStateEnum.STARTING)

    await _mark(monkeypatch, [leader])

    leader.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_follower_is_marked_from_any_state(monkeypatch):
    """The rule for subordinates is the broader one, and the fold has to
    reproduce what the instance says rather than what is tidier."""
    follower = _row(WorkloadStateEnum.STARTING, group_index=1)

    await _mark(monkeypatch, [follower])

    assert follower.update.await_args[0][1]["state"] == WorkloadStateEnum.UNREACHABLE


@pytest.mark.asyncio
async def test_an_outage_is_not_written_twice(monkeypatch):
    already = _row(WorkloadStateEnum.UNREACHABLE, group_index=1)

    await _mark(monkeypatch, [already])

    already.update.assert_not_awaited()


def test_the_confirm_wait_outlasts_the_pass_it_is_compared_against():
    """A distributed outage does not reach the instance on the event that
    caused it: the main worker writes it on its next sync pass. Waiting one
    interval races that pass and reports a difference where the fold had
    proposed exactly the value the worker went on to write."""
    assert (
        controllers_module._FOLD_CONFIRM_SECONDS
        > envs.MODEL_INSTANCE_HEALTH_CHECK_INTERVAL
    )


@pytest.mark.asyncio
async def test_an_authoritative_fold_reports_what_it_overruled(monkeypatch, caplog):
    """Turning the flag on stops the comparison being reported, at the point
    where it matters most: the worker still writes the instance, so every
    change the fold makes is it overruling the writer it is replacing."""
    instance = _instance(state="starting")
    controller = ModelInstanceWorkloadStateController()

    with _fold(monkeypatch, instance, folded={"state": "running"}, authoritative=True):
        with caplog.at_level(logging.INFO):
            await controller._reconcile(3)

    instance.update.assert_awaited_once()
    assert controller._corrected == 1
    assert "corrected model instance mi" in caplog.text


@pytest.mark.asyncio
async def test_an_authoritative_fold_that_changes_nothing_is_still_counted(
    monkeypatch,
):
    """Agreement has to stay visible once the flag is on, or the tally goes
    quiet exactly when the fold starts deciding things."""
    instance = _instance(state="running")
    controller = ModelInstanceWorkloadStateController()

    with _fold(monkeypatch, instance, folded={"state": "running"}, authoritative=True):
        await controller._reconcile(3)

    instance.update.assert_not_awaited()
    assert controller._agreed == {"running": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authoritative,expected",
    [(False, "comparing"), (True, "authoritative")],
)
async def test_the_tally_says_which_mode_it_is_in(
    monkeypatch, caplog, authoritative, expected
):
    """Both modes produce the same counters when everything agrees, so a
    healthy tally cannot otherwise say whether the fold is deciding anything
    or only watching -- which is the first thing to establish after a flip."""
    instance = _instance(state="running")
    controller = ModelInstanceWorkloadStateController()

    with _fold(
        monkeypatch,
        instance,
        folded={"state": "running"},
        authoritative=authoritative,
    ):
        with caplog.at_level(logging.INFO):
            await controller._reconcile(3)

    assert f"Workload fold [{expected}" in caplog.text


# ---------------------------------------------------------------------------
# The state the worker will stop writing
# ---------------------------------------------------------------------------


def _leader(state):
    return SimpleNamespace(group_index=0, state=state, worker_id=1)


@pytest.mark.parametrize(
    "instance_state",
    ["pending", "analyzing", "scheduled", "downloading"],
)
def test_a_spawned_container_reports_initializing(instance_state):
    """The worker writes INITIALIZING today and stops at step 4. Nothing else
    produces it -- the fold is silent through the whole of coming up -- so
    without this the state disappears from what a user sees."""
    instance = _instance(state=instance_state)

    folded = controllers_module._spawned_but_not_reported(
        instance, [_leader(WorkloadStateEnum.STARTING)]
    )

    assert folded == {"state": ModelInstanceStateEnum.INITIALIZING}


def test_a_rescheduled_instance_is_not_dragged_forward_by_a_stale_row():
    """After a failure the rows keep the ERROR that caused the restart while
    the instance is rescheduled. Only the run that actually spawned turns a
    row starting, which is what makes that the discriminator."""
    instance = _instance(state="scheduled")

    assert (
        controllers_module._spawned_but_not_reported(
            instance, [_leader(WorkloadStateEnum.ERROR)]
        )
        is None
    )


def test_the_instances_own_starting_is_not_walked_backwards():
    """The server writes STARTING after INITIALIZING, so reporting
    INITIALIZING from there would move the lifecycle back a step."""
    instance = _instance(state="starting")

    assert (
        controllers_module._spawned_but_not_reported(
            instance, [_leader(WorkloadStateEnum.STARTING)]
        )
        is None
    )
