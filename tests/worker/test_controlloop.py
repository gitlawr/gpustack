import asyncio
import logging
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from gpustack_runtime.deployer import WorkloadStatusStateEnum

from gpustack.api.exceptions import NotFoundException
from gpustack.worker.controlloop import (
    RestartActionEnum,
    RestartBudget,
    WorkloadPhase,
    classify_workload,
    needs_restart,
    update_resource,
    watch_forever,
)
from gpustack.worker.controlloop.workload_state import RestartPolicy

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watch_reconnects_after_a_dropped_stream():
    """A dropped watch is routine — a server restart, a proxy timeout — and
    must not stop the worker reacting to a whole resource kind."""
    attempts = []

    async def awatch(callback=None):
        attempts.append(callback)
        if len(attempts) < 3:
            raise RuntimeError("stream closed")
        raise asyncio.CancelledError()

    callback = object()
    with patch("gpustack.worker.controlloop.watcher.asyncio.sleep") as sleep:
        await watch_forever("things", awatch, callback=callback, retry_interval=5)

    assert attempts == [callback] * 3
    assert sleep.await_count == 2
    assert sleep.await_args[0][0] == 5


@pytest.mark.asyncio
async def test_watch_stops_on_cancellation():
    async def awatch(callback=None):
        raise asyncio.CancelledError()

    await watch_forever("things", awatch)


# ---------------------------------------------------------------------------
# Workload phase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,expected",
    [
        (WorkloadStatusStateEnum.PENDING, WorkloadPhase.LAUNCHING),
        (WorkloadStatusStateEnum.INITIALIZING, WorkloadPhase.LAUNCHING),
        (WorkloadStatusStateEnum.RUNNING, WorkloadPhase.RUNNING),
        (WorkloadStatusStateEnum.INACTIVE, WorkloadPhase.EXITED),
        (WorkloadStatusStateEnum.FAILED, WorkloadPhase.FAILED),
        (WorkloadStatusStateEnum.UNHEALTHY, WorkloadPhase.FAILED),
        (WorkloadStatusStateEnum.UNKNOWN, WorkloadPhase.FAILED),
    ],
)
def test_classify_covers_every_runtime_state(state, expected):
    assert classify_workload(SimpleNamespace(state=state)) == expected


def test_classify_reports_a_vanished_workload_as_missing():
    """Distinct from FAILED: nothing ever ran, or something outside gpustack
    removed it, which some policies treat differently from a crash."""
    assert classify_workload(None) == WorkloadPhase.MISSING


@pytest.mark.parametrize(
    "phase,policy,expected",
    [
        # A clean exit is the whole point of a task and the death of a service.
        (WorkloadPhase.EXITED, RestartPolicy.ALWAYS, True),
        (WorkloadPhase.EXITED, RestartPolicy.ON_FAILURE, False),
        (WorkloadPhase.EXITED, RestartPolicy.NEVER, False),
        (WorkloadPhase.FAILED, RestartPolicy.ALWAYS, True),
        (WorkloadPhase.FAILED, RestartPolicy.ON_FAILURE, True),
        (WorkloadPhase.FAILED, RestartPolicy.NEVER, False),
        (WorkloadPhase.MISSING, RestartPolicy.ALWAYS, True),
        (WorkloadPhase.MISSING, RestartPolicy.ON_FAILURE, True),
        # Nothing that is still coming up or serving is restarted.
        (WorkloadPhase.LAUNCHING, RestartPolicy.ALWAYS, False),
        (WorkloadPhase.RUNNING, RestartPolicy.ALWAYS, False),
    ],
)
def test_restart_policy_splits_service_and_task_semantics(phase, policy, expected):
    assert needs_restart(phase, policy) is expected


# ---------------------------------------------------------------------------
# Write-back
# ---------------------------------------------------------------------------


class _Update:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _client(current):
    client = MagicMock()
    client.get.return_value = SimpleNamespace(model_dump=lambda: current)
    return client


def test_update_applies_only_the_given_fields():
    client = _client({"name": "a", "state": "pending", "port": None})

    assert update_resource(client, 7, _Update, "Thing", state="running") is True

    sent = client.update.call_args[1]["model_update"]
    assert (sent.state, sent.name, sent.port) == ("running", "a", None)
    assert client.update.call_args[1]["id"] == 7


def test_update_reports_a_deleted_resource_without_raising():
    """The row can go away between the reconcile pass reading it and writing
    it back; the caller has no exception handler to reach."""
    client = MagicMock()
    client.get.side_effect = NotFoundException(message="gone")

    assert update_resource(client, 7, _Update, "Thing", state="running") is False


def test_update_reports_a_failed_writeback_without_raising(caplog):
    client = MagicMock()
    client.get.side_effect = RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        assert update_resource(client, 7, _Update, "Thing", state="running") is False

    assert "Failed to update thing 7" in caplog.text


# ---------------------------------------------------------------------------
# Restart budget
# ---------------------------------------------------------------------------


CACHE_SERVICE_BUDGET = RestartBudget(
    base_delay_seconds=30,
    max_delay_seconds=300,
    max_attempts=5,
    reset_after_seconds=600,
)

MODEL_INSTANCE_BUDGET = RestartBudget(
    base_delay_seconds=10,
    max_delay_seconds=300,
    max_attempts=None,
    first_attempt_immediate=True,
)


def test_cache_service_delay_curve_is_unchanged():
    """min(30 * 2**n, 300), backed off from the first crash."""
    curve = [CACHE_SERVICE_BUDGET.delay_for(n) for n in range(6)]

    assert curve == [30, 60, 120, 240, 300, 300]


def test_model_instance_delay_curve_is_unchanged():
    """First crash restarts immediately, then min(10 * 2**(n-1), 300)."""
    curve = [MODEL_INSTANCE_BUDGET.delay_for(n) for n in range(8)]

    assert curve == [0, 10, 20, 40, 80, 160, 300, 300]


def test_a_workload_never_restarted_owes_nothing():
    """No last attempt means no window to sit out, whatever the delay says."""
    decision = CACHE_SERVICE_BUDGET.decide(0, None, _NOW)

    assert decision.action == RestartActionEnum.RESTART
    assert decision.attempt == 1


def test_restart_waits_out_the_window_then_goes():
    last = _NOW - timedelta(seconds=20)

    waiting = CACHE_SERVICE_BUDGET.decide(1, last, _NOW)
    assert waiting.action == RestartActionEnum.WAIT
    assert waiting.delay_remaining_seconds == pytest.approx(40)

    ready = CACHE_SERVICE_BUDGET.decide(1, _NOW - timedelta(seconds=61), _NOW)
    assert ready.action == RestartActionEnum.RESTART
    assert ready.attempt == 2


def test_budget_with_no_cap_never_gives_up():
    """A model instance keeps retrying: the failure is often outside it, and
    the user turns off restart_on_error to stop it."""
    decision = MODEL_INSTANCE_BUDGET.decide(999, _NOW - timedelta(hours=1), _NOW)

    assert decision.action == RestartActionEnum.RESTART


def test_budget_gives_up_once_the_attempts_are_spent():
    decision = CACHE_SERVICE_BUDGET.decide(5, _NOW - timedelta(hours=1), _NOW)

    assert decision.action == RestartActionEnum.GIVE_UP


def test_forgiveness_needs_the_full_stable_window():
    assert not CACHE_SERVICE_BUDGET.should_forgive(
        3, _NOW - timedelta(seconds=599), _NOW
    )
    assert CACHE_SERVICE_BUDGET.should_forgive(3, _NOW - timedelta(seconds=600), _NOW)


def test_nothing_to_forgive_without_attempts_or_a_configured_window():
    assert not CACHE_SERVICE_BUDGET.should_forgive(0, _NOW - timedelta(days=1), _NOW)
    assert not MODEL_INSTANCE_BUDGET.should_forgive(3, _NOW - timedelta(days=1), _NOW)
