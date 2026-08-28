import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from gpustack_runtime.deployer import WorkloadStatusStateEnum

from gpustack.api.exceptions import NotFoundException
from gpustack.worker.controlloop import (
    WorkloadPhase,
    classify_workload,
    needs_restart,
    update_resource,
    watch_forever,
)
from gpustack.worker.controlloop.workload_state import RestartPolicy

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
