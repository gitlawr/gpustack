"""
Copying a workload's container output into files.

The container runtime only streams a live container's logs, so a crash-looped
workload's previous output is gone by the time anyone asks for it. Persisting
it is a horizontal concern -- every kind of workload wants it -- so the stream
handling lives here and the callers only say where the files go.
"""

import logging
import threading
from collections import deque
from typing import Any, Callable, Dict, List, Optional

from gpustack_runtime.deployer import get_workload, logs_workload

from gpustack.worker.controlloop.workload_state import (
    WorkloadPhase,
    classify_workload,
)

logger = logging.getLogger(__name__)

DEFAULT_CONTAINER_NAME = "default"
"""The workload's own container. Anything else in ``loggable`` is a sidecar."""


class ContainerLogPersistence:
    """
    Streams each container of a workload into a file, one thread per
    container, keyed by whatever the caller uses to identify the workload.

    Args:
        reconnect_grace_seconds: How long a stream that hit EOF waits for its
            container to come back before the thread gives up. A restart looks
            briefly indistinguishable from a termination.
    """

    def __init__(self, reconnect_grace_seconds: float):
        self._reconnect_grace_seconds = reconnect_grace_seconds
        self._threads: Dict[Any, List[threading.Thread]] = {}
        self._stop_events: Dict[Any, List[threading.Event]] = {}

    def start(
        self,
        key: Any,
        workload_name: str,
        log_path: str,
        sidecar_log_path: Optional[Callable[[str], str]] = None,
    ):
        """
        Begin persisting the workload's logs, replacing anything already
        running for this key.

        Args:
            key: Identifies the owner, for stop() and bookkeeping.
            workload_name: The workload to read from.
            log_path: Where the workload's own container goes.
            sidecar_log_path: Given a sidecar container's name, where its logs
                go. None for a workload that only ever has one container, in
                which case no discovery thread is started.
        """
        self.stop(key)

        stop_event = threading.Event()
        threads = [
            self._spawn(
                f"log-persist-{workload_name}",
                self._persist,
                (workload_name, log_path, stop_event, None),
            )
        ]

        if sidecar_log_path is not None:
            # Sidecars only exist once the workload does, so discovery polls
            # rather than reading the plan.
            threads.append(
                self._spawn(
                    f"log-discover-{workload_name}",
                    self._discover_sidecars,
                    (key, workload_name, sidecar_log_path, stop_event),
                )
            )

        self._threads[key] = threads
        self._stop_events[key] = [stop_event]
        logger.debug(f"Started container log persistence for {workload_name}")

    def stop(self, key: Any, timeout: float = 2.0):
        """Signal every thread for this key and wait for them."""
        for stop_event in self._stop_events.pop(key, []):
            stop_event.set()

        for thread in self._threads.pop(key, []):
            if thread and thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(
                        f"Log persistence thread {thread.name} for {key} did "
                        f"not stop within {timeout}s"
                    )

    @staticmethod
    def _spawn(name: str, target: Callable, args: tuple) -> threading.Thread:
        thread = threading.Thread(target=target, args=args, daemon=True, name=name)
        thread.start()
        return thread

    def _discover_sidecars(
        self,
        key: Any,
        workload_name: str,
        sidecar_log_path: Callable[[str], str],
        stop_event: threading.Event,
    ):
        """
        Wait for sidecar containers to appear, then persist each of them too.

        Exits once they are found, or when asked to stop.
        """
        while not stop_event.is_set():
            try:
                workload = get_workload(workload_name)
                if workload and workload.loggable:
                    sidecars = [
                        op
                        for op in workload.loggable
                        if op.name != DEFAULT_CONTAINER_NAME
                    ]
                    if sidecars:
                        self._start_sidecar_threads(
                            key, workload_name, sidecars, sidecar_log_path
                        )
                        logger.debug(f"Sidecar discovery for {workload_name} complete")
                        return
            except Exception:
                pass
            stop_event.wait(timeout=2)

    def _start_sidecar_threads(
        self,
        key: Any,
        workload_name: str,
        sidecars: list,
        sidecar_log_path: Callable[[str], str],
    ):
        names = []
        for op in sidecars:
            stop_event = threading.Event()
            thread = self._spawn(
                f"log-persist-{workload_name}-{op.name}",
                self._persist,
                (workload_name, sidecar_log_path(op.name), stop_event, op.token),
            )
            self._threads.setdefault(key, []).append(thread)
            self._stop_events.setdefault(key, []).append(stop_event)
            names.append(op.name)

        if names:
            logger.debug(
                f"Started sidecar log persistence threads for {workload_name}: "
                f"{names}"
            )

    def _persist(
        self,
        workload_name: str,
        log_path: str,
        stop_event: threading.Event,
        token: Optional[str] = None,
    ):
        """Persist container logs to local file (runs in a separate thread).

        Reconnects on stream EOF while the workload is still alive, resuming by
        skipping already-written history (matched by an anchor window of the
        last lines written). A manual/runtime restart briefly looks terminated
        at EOF, so it waits a grace window for the container to return before
        giving up. Exits only if the container stays terminated for that whole
        window or the thread is asked to stop.

        Args:
            workload_name: Name of the container workload
            log_path: Path to save container logs
            stop_event: Event to signal thread to stop
            token: Operation token identifying a specific container in the workload.
                If None, logs from the default (index=0) container are fetched.
        """
        retry_count = 0
        first_connect = True
        # Anchor: a window of the last lines written. Matching a run of lines
        # (not one) avoids false-matching a repeated line during replay.
        anchor_window = deque(maxlen=5)

        while not stop_event.is_set():
            try:
                log_stream = logs_workload(
                    name=workload_name,
                    token=token,
                    tail=-1,
                    follow=True,
                )

                if hasattr(log_stream, '__iter__'):
                    # On reconnect the runtime replays history from the start;
                    # skip it until the anchor window matches.
                    anchor = list(anchor_window)
                    skip_until_anchor = not first_connect and bool(anchor)
                    replayed = deque(maxlen=len(anchor)) if anchor else None
                    received_lines = False
                    with open(
                        log_path,
                        'w' if first_connect else 'a',
                        buffering=1,
                        encoding='utf-8',
                    ) as f:
                        first_connect = False
                        for line in log_stream:
                            received_lines = True
                            if stop_event.is_set():
                                break

                            if isinstance(line, bytes):
                                line = line.decode('utf-8', errors='replace')
                            else:
                                line = str(line)

                            if skip_until_anchor:
                                replayed.append(line)
                                if list(replayed) == anchor:
                                    skip_until_anchor = False
                                continue

                            f.write(line)
                            f.flush()
                            anchor_window.append(line)
                    retry_count = 0

                    # Anchor never matched -> rotated out; restart fresh. An
                    # empty reconnect must NOT reset, or the next round reopens
                    # in 'w' and truncates the saved log.
                    if skip_until_anchor and received_lines:
                        first_connect = True
                        anchor_window.clear()

                # A restart briefly looks terminated at EOF; wait for the
                # container to return before giving up, so logs aren't dropped.
                if stop_event.is_set() or not self._wait_for_recovery(
                    workload_name, stop_event
                ):
                    break
                logger.debug(
                    f"Log stream for {workload_name} ended while workload still "
                    f"running; reconnecting"
                )
                stop_event.wait(timeout=1)

            except Exception as e:
                if stop_event.is_set():
                    break
                retry_count += 1
                logger.debug(
                    f"Container not ready for {workload_name}, retrying "
                    f"(attempt {retry_count}): {e}"
                )
                stop_event.wait(timeout=2)

        logger.debug(f"Log persistence thread for {workload_name} exiting")

    def _still_alive(self, workload_name: str) -> bool:
        """Whether the workload is still alive (a dead stream should reconnect
        rather than exit)."""
        try:
            workload = get_workload(workload_name)
        except Exception:
            return True  # transient query failure: reconnect, don't drop logs
        return classify_workload(workload) in (
            WorkloadPhase.LAUNCHING,
            WorkloadPhase.RUNNING,
        )

    def _wait_for_recovery(
        self,
        workload_name: str,
        stop_event: threading.Event,
        grace_seconds: Optional[float] = None,
        poll_interval: float = 1.0,
    ) -> bool:
        """Poll until the workload is alive again (True -> reconnect) or the
        grace window elapses / stop_event fires (False -> give up). A restart
        momentarily looks terminated at EOF, which a single check can't tell
        apart from a real termination.
        """
        if grace_seconds is None:
            grace_seconds = self._reconnect_grace_seconds
        attempts = max(1, int(grace_seconds / poll_interval))
        for _ in range(attempts):
            if stop_event.is_set():
                return False
            if self._still_alive(workload_name):
                return True
            stop_event.wait(timeout=poll_interval)
        return False
