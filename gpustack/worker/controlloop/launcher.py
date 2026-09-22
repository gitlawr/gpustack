"""
Running a workload's provisioning step in a subprocess.

Provisioning is the slow, failure-prone half of a start -- resolving images,
pulling them, rendering the plan -- so it runs out of process, with its output
redirected to a log file the user can read while it happens. The preamble that
sets that up is the same for every kind of workload, and so is the bookkeeping
that tells the control loop whether a start is still under way.

The start method is spawn (set in ``cmd/start.py``), so the child inherits
nothing: it rebuilds its clientset and its logging from what it is handed, and
everything passed to ``start`` has to be picklable.
"""

import logging
import multiprocessing
from typing import Any, Callable, Dict, Optional, Set

import setproctitle

from gpustack_runtime.logging import setup_logging as setup_runtime_logging

from gpustack.client import ClientSet
from gpustack.config.config import Config
from gpustack.logging import RedirectStdoutStderr, setup_logging
from gpustack.utils.process import add_signal_handlers, terminate_process_tree

logger = logging.getLogger(__name__)


def run_provisioning(
    proctitle: str,
    log_path: str,
    cfg: Config,
    client_headers: dict,
    provision: Callable[[ClientSet, Config], None],
    description: str,
):
    """
    The subprocess entry point: set the process up, then provision.

    Module level and not a closure, because spawn pickles the target by
    reference.

    Args:
        proctitle: What the process is called in a process listing.
        log_path: Where stdout, stderr and log records go.
        cfg: The worker configuration, rebuilt in the child by spawn.
        client_headers: Auth headers the child's own clientset is built with.
        provision: Does the work, given a clientset and the config.
        description: Names the workload in this file's log lines.
    """
    setproctitle.setproctitle(proctitle)
    add_signal_handlers()

    clientset = ClientSet(base_url=cfg.get_server_url(), headers=client_headers)

    with open(log_path, "w", buffering=1, encoding="utf-8") as log_file:
        with RedirectStdoutStderr(log_file):
            # Both loggers are configured inside the redirect so their handlers
            # bind to this file: the container runtime reports the image pull
            # -- the slowest part of a start -- through its own logger.
            setup_logging(debug=cfg.debug)
            setup_runtime_logging()
            try:
                logger.info(f"Provisioning {description}")
                provision(clientset, cfg)
                logger.info(f"Finished provisioning {description}")
            except Exception as e:
                logger.exception(f"Error provisioning {description}: {e}")
                raise


class ProvisionRunner:
    """
    Tracks the provisioning subprocess of each workload this worker starts.

    Keyed by whatever the caller identifies a workload with. One process per
    key: starting a second while the first is alive is the caller's mistake to
    avoid, which ``is_running`` is there for.
    """

    def __init__(self, cfg: Config, headers_getter: Callable[[], dict]):
        self._cfg = cfg
        self._headers_getter = headers_getter
        self._processes: Dict[Any, multiprocessing.Process] = {}

    def start(
        self,
        key: Any,
        description: str,
        proctitle: str,
        log_path: str,
        provision: Callable[[ClientSet, Config], None],
    ) -> multiprocessing.Process:
        """
        Spawn the provisioning subprocess and track it.

        ``provision`` and everything it closes over must be picklable; a
        ``functools.partial`` of a module-level function is the usual shape.
        """
        process = multiprocessing.Process(
            target=run_provisioning,
            args=(
                proctitle,
                log_path,
                self._cfg,
                self._headers_getter(),
                provision,
                description,
            ),
        )
        process.daemon = False
        process.start()
        self._processes[key] = process
        return process

    def is_running(self, key: Any) -> bool:
        """
        Whether this workload is still being provisioned.

        Reaps an exited child on the way, so it does not linger as a zombie
        until the workload is stopped.
        """
        process = self._processes.get(key)
        if process is None:
            return False
        process.join(timeout=0)
        if process.is_alive():
            return True
        self._processes.pop(key, None)
        return False

    def terminate(self, key: Any):
        """Kill the subprocess and its children, if it is still running."""
        process = self._processes.pop(key, None)
        if process is not None and process.is_alive():
            terminate_process_tree(process.pid)

    def forget(self, key: Any):
        """Drop the entry without touching the process."""
        self._processes.pop(key, None)

    def pid(self, key: Any) -> Optional[int]:
        process = self._processes.get(key)
        return process.pid if process is not None else None

    def keys(self) -> Set[Any]:
        return set(self._processes)
