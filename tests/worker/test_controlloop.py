import asyncio
import logging
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from gpustack_runtime.deployer import WorkloadStatusStateEnum

from gpustack.api.exceptions import NotFoundException
from gpustack.worker.controlloop import (
    ProvisionRunner,
    RestartActionEnum,
    RestartBudget,
    WorkloadPhase,
    classify_workload,
    needs_restart,
    run_provisioning,
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


# ---------------------------------------------------------------------------
# Provisioning subprocesses
# ---------------------------------------------------------------------------


def _runner():
    cfg = SimpleNamespace(debug=False, get_server_url=lambda: "http://127.0.0.1")
    return ProvisionRunner(cfg, lambda: {"Authorization": "Bearer t"})


def test_start_tracks_the_process_and_passes_the_preamble_its_inputs():
    runner = _runner()

    with patch(
        "gpustack.worker.controlloop.launcher.multiprocessing.Process"
    ) as process_cls:
        returned = runner.start(
            7, "thing 7", "gpustack_thing_7", "/tmp/7.log", provision=_noop
        )

    kwargs = process_cls.call_args[1]
    assert kwargs["target"] is run_provisioning
    proctitle, log_path, cfg, headers, provision, description = kwargs["args"]
    assert (proctitle, log_path, description) == (
        "gpustack_thing_7",
        "/tmp/7.log",
        "thing 7",
    )
    assert headers == {"Authorization": "Bearer t"}
    assert provision is _noop
    assert cfg is runner._cfg

    assert returned is process_cls.return_value
    assert runner.is_running(7) is True


def test_is_running_reaps_an_exited_child():
    """Left untouched it lingers as a zombie until the workload is stopped."""
    runner = _runner()
    process = MagicMock()
    process.is_alive.return_value = False
    runner._processes[7] = process

    assert runner.is_running(7) is False

    process.join.assert_called_once_with(timeout=0)
    assert 7 not in runner._processes


def test_is_running_is_false_for_an_unknown_key():
    assert _runner().is_running(7) is False


def test_terminate_kills_the_tree_and_forgets_the_key():
    runner = _runner()
    process = MagicMock()
    process.is_alive.return_value = True
    process.pid = 4242
    runner._processes[7] = process

    with patch(
        "gpustack.worker.controlloop.launcher.terminate_process_tree"
    ) as terminate:
        runner.terminate(7)

    terminate.assert_called_once_with(4242)
    assert 7 not in runner._processes


def test_terminate_leaves_an_already_dead_child_alone():
    runner = _runner()
    process = MagicMock()
    process.is_alive.return_value = False
    runner._processes[7] = process

    with patch(
        "gpustack.worker.controlloop.launcher.terminate_process_tree"
    ) as terminate:
        runner.terminate(7)

    terminate.assert_not_called()
    assert 7 not in runner._processes


def test_provisioning_preamble_redirects_output_and_rebuilds_the_clientset(tmp_path):
    """The child inherits nothing under spawn, so the preamble is what gives it
    a clientset and a root logger bound to its own log file."""
    log_path = tmp_path / "7.log"
    seen = {}

    def provision(clientset, cfg):
        seen["clientset"] = clientset
        seen["cfg"] = cfg
        logging.getLogger("probe").info("from-the-logger")
        print("from-stdout")

    cfg = SimpleNamespace(debug=False, get_server_url=lambda: "http://127.0.0.1")
    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    root.handlers[:] = []
    try:
        with patch("gpustack.worker.controlloop.launcher.ClientSet") as clientset_cls:
            run_provisioning(
                "gpustack_thing_7", str(log_path), cfg, {"h": "v"}, provision, "thing 7"
            )
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)

    clientset_cls.assert_called_once_with(
        base_url="http://127.0.0.1", headers={"h": "v"}
    )
    assert seen["clientset"] is clientset_cls.return_value
    assert seen["cfg"] is cfg

    contents = log_path.read_text()
    assert "from-the-logger" in contents
    assert "from-stdout" in contents
    assert "Provisioning thing 7" in contents


def test_provisioning_preamble_reports_a_failure_and_reraises(tmp_path):
    """The subprocess dies either way; what matters is that the reason lands in
    the log file rather than only in the exit status."""
    log_path = tmp_path / "7.log"

    def provision(clientset, cfg):
        raise RuntimeError("boom")

    cfg = SimpleNamespace(debug=False, get_server_url=lambda: "http://127.0.0.1")
    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    root.handlers[:] = []
    try:
        with patch("gpustack.worker.controlloop.launcher.ClientSet"):
            with pytest.raises(RuntimeError):
                run_provisioning(
                    "gpustack_thing_7", str(log_path), cfg, {}, provision, "thing 7"
                )
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)

    contents = log_path.read_text()
    assert "Error provisioning thing 7: boom" in contents
    assert "RuntimeError" in contents


def _noop(clientset, cfg):
    pass
