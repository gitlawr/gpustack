"""
The container log persistence loop, which every kind of workload relies on to
keep the output of a container that is already gone.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gpustack_runtime.deployer import WorkloadStatusStateEnum

from gpustack.worker.controlloop import ContainerLogPersistence


def _fake_stop_event(max_waits: int = 100):
    """A stop event whose wait() returns instantly (tests aren't driven by real
    time) and stays unset, so the log persistence loop is driven purely by the
    get_workload state sequence. It auto-sets after max_waits waits so a
    mis-sized mock or a runaway loop fails the test fast instead of hanging CI."""
    state = {"waits": 0, "stopped": False}

    def is_set():
        return state["stopped"]

    def wait(timeout=None):
        state["waits"] += 1
        if state["waits"] >= max_waits:
            state["stopped"] = True
        return state["stopped"]

    stop_event = MagicMock()
    stop_event.is_set.side_effect = is_set
    stop_event.wait.side_effect = wait
    return stop_event


def _get_workload_sequence(states):
    """side_effect for a patched get_workload. The recovery grace-poll queries
    get_workload several times per stream EOF, so once the sequence reaches its
    terminal state it must keep returning it: a list that runs dry would raise
    IndexError, which _still_alive treats as "still alive", spinning
    the reconnect loop forever."""
    remaining = list(states)

    def next_state(name):
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return next_state


def test_persist_container_logs_reconnects_and_dedupes(tmp_path: Path):
    """On stream EOF while the workload is still running, reconnect and resume
    by skipping already-written history (anchor), appending only new lines."""
    persistence = ContainerLogPersistence(reconnect_grace_seconds=2)
    log_path = str(tmp_path / "1.container.0.log")

    # First stream: initial history. Reconnect: full history replay + new line.
    streams = [iter(["a\n", "b\n"]), iter(["a\n", "b\n", "c\n"])]
    tails = []

    def fake_logs_workload(**kwargs):
        tails.append(kwargs["tail"])
        return streams.pop(0)

    # First EOF -> still RUNNING (reconnect); second EOF -> FAILED (exit).
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch(
            "gpustack.worker.controlloop.container_logs.logs_workload",
            side_effect=fake_logs_workload,
        ),
        patch(
            "gpustack.worker.controlloop.container_logs.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        persistence._persist("wl", log_path, _fake_stop_event())

    assert tails == [-1, -1]
    assert Path(log_path).read_text(encoding="utf-8") == "a\nb\nc\n"


def test_persist_container_logs_exits_when_workload_gone(tmp_path: Path):
    """EOF while the workload no longer exists -> exit immediately, no reconnect."""
    persistence = ContainerLogPersistence(reconnect_grace_seconds=2)
    log_path = str(tmp_path / "1.container.0.log")
    tails = []

    def fake_logs_workload(**kwargs):
        tails.append(kwargs["tail"])
        return iter(["a\n"])

    with (
        patch(
            "gpustack.worker.controlloop.container_logs.logs_workload",
            side_effect=fake_logs_workload,
        ),
        patch(
            "gpustack.worker.controlloop.container_logs.get_workload", return_value=None
        ),
    ):
        persistence._persist("wl", log_path, _fake_stop_event())

    assert tails == [-1]  # only one connection, no reconnect
    assert Path(log_path).read_text(encoding="utf-8") == "a\n"


def test_persist_container_logs_resets_when_anchor_rotated(tmp_path: Path):
    """If the anchor line was rotated out of the reconnect logs, restart from
    scratch (full rewrite) instead of skipping new lines forever."""
    persistence = ContainerLogPersistence(reconnect_grace_seconds=2)
    log_path = str(tmp_path / "1.container.0.log")

    streams = [
        iter(["a\n", "b\n"]),  # round1: write a,b (anchor=b)
        iter(["x\n", "c\n"]),  # round2: anchor 'b' rotated out -> skip all, reset
        iter(["x\n", "c\n", "d\n"]),  # round3: fresh rewrite recovers
    ]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch(
            "gpustack.worker.controlloop.container_logs.logs_workload",
            side_effect=lambda **kwargs: streams.pop(0),
        ),
        patch(
            "gpustack.worker.controlloop.container_logs.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        persistence._persist("wl", log_path, _fake_stop_event())

    assert Path(log_path).read_text(encoding="utf-8") == "x\nc\nd\n"


def test_persist_container_logs_empty_reconnect_keeps_history(tmp_path: Path):
    """An empty reconnect (0 lines) must not reset first_connect; otherwise the
    next reconnect reopens in 'w' and truncates already-persisted logs."""
    persistence = ContainerLogPersistence(reconnect_grace_seconds=2)
    log_path = str(tmp_path / "1.container.0.log")

    streams = [
        iter(["a\n", "b\n"]),  # round1: write a,b
        iter([]),  # round2: empty reconnect (0 lines) -> must NOT reset
        iter(["b\n"]),  # round3: suffix replay; a,b already persisted survive
    ]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch(
            "gpustack.worker.controlloop.container_logs.logs_workload",
            side_effect=lambda **kwargs: streams.pop(0),
        ),
        patch(
            "gpustack.worker.controlloop.container_logs.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        persistence._persist("wl", log_path, _fake_stop_event())

    # Had the empty round2 reset first_connect, round3 would reopen in 'w' and
    # truncate 'a'; a,b surviving proves it did not.
    assert Path(log_path).read_text(encoding="utf-8") == "a\nb\n"


def test_persist_container_logs_window_anchor_ignores_repeated_line(
    tmp_path: Path,
):
    """The multi-line anchor window only matches the true tail: a single-line
    anchor would false-match an earlier identical line and duplicate history."""
    persistence = ContainerLogPersistence(reconnect_grace_seconds=2)
    log_path = str(tmp_path / "1.container.0.log")

    streams = [
        iter(["A\n", "B\n", "A\n", "B\n"]),  # round1: last line B repeats earlier
        iter(["A\n", "B\n", "A\n", "B\n", "C\n"]),  # round2: full replay + new C
    ]
    states = [
        SimpleNamespace(state=WorkloadStatusStateEnum.RUNNING),
        SimpleNamespace(state=WorkloadStatusStateEnum.FAILED),
    ]

    with (
        patch(
            "gpustack.worker.controlloop.container_logs.logs_workload",
            side_effect=lambda **kwargs: streams.pop(0),
        ),
        patch(
            "gpustack.worker.controlloop.container_logs.get_workload",
            side_effect=_get_workload_sequence(states),
        ),
    ):
        persistence._persist("wl", log_path, _fake_stop_event())

    # Window [A,B,A,B] matches only at the end; single-line 'B' would match
    # index 1 and duplicate A,B.
    assert Path(log_path).read_text(encoding="utf-8") == "A\nB\nA\nB\nC\n"


# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------


def test_start_without_sidecars_runs_one_thread():
    """A workload that only ever has one container gets no discovery thread
    polling for siblings that will never appear."""
    persistence = ContainerLogPersistence(reconnect_grace_seconds=2)

    with patch.object(ContainerLogPersistence, "_spawn") as spawn:
        persistence.start(7, "wl", "/tmp/wl.log")

    assert spawn.call_count == 1
    assert persistence._threads[7] == [spawn.return_value]


def test_start_with_sidecars_adds_a_discovery_thread():
    persistence = ContainerLogPersistence(reconnect_grace_seconds=2)

    with patch.object(ContainerLogPersistence, "_spawn") as spawn:
        persistence.start(
            7, "wl", "/tmp/wl.log", sidecar_log_path=lambda n: f"/tmp/{n}"
        )

    assert spawn.call_count == 2


def test_starting_again_replaces_the_previous_threads():
    """A restart must not leave the old stream writing to the old generation's
    file alongside the new one."""
    persistence = ContainerLogPersistence(reconnect_grace_seconds=2)

    with patch.object(ContainerLogPersistence, "_spawn"):
        persistence.start(7, "wl", "/tmp/a.log")
        first_stop_event = persistence._stop_events[7][0]
        persistence.start(7, "wl", "/tmp/b.log")

    assert first_stop_event.is_set()
    assert len(persistence._threads[7]) == 1


def test_stop_signals_every_thread_of_the_key():
    persistence = ContainerLogPersistence(reconnect_grace_seconds=2)
    events = [MagicMock(), MagicMock()]
    persistence._stop_events[7] = events
    persistence._threads[7] = []

    persistence.stop(7)

    for event in events:
        event.set.assert_called_once()
    assert 7 not in persistence._stop_events
