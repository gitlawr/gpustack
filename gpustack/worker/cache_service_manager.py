import asyncio
import logging
import multiprocessing
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple

import httpx
import setproctitle

from gpustack_runtime.deployer import (
    delete_workload,
    get_workload,
)
from gpustack_runtime.logging import setup_logging as setup_runtime_logging

from gpustack.api.exceptions import NotFoundException
from gpustack.client import ClientSet
from gpustack.config.config import Config
from gpustack.logging import RedirectStdoutStderr, setup_logging
from gpustack.schemas.cache_providers import CacheProviderHealthCheck
from gpustack.schemas.cache_services import (
    CacheServiceInstance,
    CacheServicePublic,
    CacheServiceStateEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.server.cache_provider_catalog import get_cache_provider
from gpustack.utils import network
from gpustack.utils.process import add_signal_handlers, terminate_process_tree
from gpustack.worker.cache_service.provisioner import (
    CacheServiceProvisioner,
    resolve_fallback_registry,
)
from gpustack.worker.cache_service.state import update_cache_service_instance
from gpustack.worker.controlloop import (
    ContainerLogPersistence,
    RestartActionEnum,
    RestartBudget,
    WorkloadPhase,
    classify_workload,
    watch_forever,
)

logger = logging.getLogger(__name__)

HEALTH_PROBE_TIMEOUT_SECONDS = 2

MAX_CONSECUTIVE_RESTARTS = 5
"""Consecutive crashes tolerated before the instance is parked in ERROR."""

RESTART_BACKOFF_BASE_SECONDS = 30
RESTART_BACKOFF_MAX_SECONDS = 300

RESTART_COUNT_RESET_SECONDS = 600
"""How long an instance must stay healthy after its last restart before the
consecutive-restart budget is cleared."""

RESTART_BUDGET = RestartBudget(
    base_delay_seconds=RESTART_BACKOFF_BASE_SECONDS,
    max_delay_seconds=RESTART_BACKOFF_MAX_SECONDS,
    max_attempts=MAX_CONSECUTIVE_RESTARTS,
    reset_after_seconds=RESTART_COUNT_RESET_SECONDS,
)
"""A cache server is a service with no successful end, so its first crash is
already backed off; the count lives on the instance row and is forgiven once
it has stayed healthy for the reset window."""

LOG_RECONNECT_GRACE_SECONDS = 17
"""How long a container log stream that hit EOF waits for its container to
come back before the thread gives up. A restart looks briefly the same as a
termination."""

PENDING_START_GRACE_SECONDS = 60
"""How long an instance may stay PENDING before the sync pass re-drives its
start. The start is normally driven by the instance's PENDING event; this
window bounds how long a start that never took effect — a missed event, a
worker restart mid-start, a state write-back that did not reach the server —
keeps the instance stuck."""


def _running_on(loop: asyncio.AbstractEventLoop) -> bool:
    """Whether the caller is already on that loop's thread."""
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


class CacheServiceManager:
    """
    Runs managed cache service instances on this worker: launches the
    provider's container for each instance scheduled here, and keeps the
    instance's state in sync with the workload and its health probe.
    Rendering inputs (provider, version, config) come from the instance's
    parent cache service.
    """

    _port_lock = threading.Lock()
    _start_lock = threading.Lock()

    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    """
    The ID of current worker.
    """
    _config: Config
    """
    Global configuration.
    """

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    """
    The clientset to access the API server.
    """

    _starting: Set[int]
    """
    IDs of instances whose launch is in flight in this process — from the
    moment the start is claimed until the provisioning subprocess is spawned.
    Guarded by _start_lock.
    """

    _provisioning_processes: Dict[int, multiprocessing.Process]
    """
    The mapping of cache service instance ID to provisioning (sub)process.
    While the (sub)process is alive the instance is still being provisioned —
    pulling the provider image can take minutes — so the sync pass neither
    re-drives its start nor treats the missing workload as a crash.
    """

    _loop: Optional[asyncio.AbstractEventLoop]
    """The worker's event loop, captured when the watch starts. Launches are
    posted here from other threads so every fork happens on the same thread,
    as it does in the model instance and benchmark managers."""

    _provision_log_dir: str
    """
    The directory holding the provisioning logs of cache service instances
    (written by the subprocess).
    """

    _last_start_attempt: Dict[int, datetime]
    """
    When this process last started each instance, so an instance that stays
    PENDING because its start keeps failing to take effect is retried on the
    PENDING_START_GRACE_SECONDS cadence rather than on every sync pass.
    Guarded by _start_lock.
    """

    _assigned_ports: Dict[int, Tuple[int, int]]
    """
    (port, metrics_port) pairs allocated in this process, keyed by cache
    service instance ID. Guarded by _port_lock so concurrent starts can't
    hand out the same port.
    """

    _clientset_getter: Callable[[], ClientSet]
    _worker_id_getter: Callable[[], int]

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
        cfg: Config,
    ):
        self._worker_id_getter = worker_id_getter
        self._clientset_getter = clientset_getter
        self._config = cfg
        self._provision_log_dir = f"{cfg.log_dir}/cache-services"

        self._assigned_ports = {}
        self._starting = set()
        self._last_start_attempt = {}
        self._provisioning_processes = {}
        self._loop = None
        self._container_logs = ContainerLogPersistence(LOG_RECONNECT_GRACE_SECONDS)

        # Consecutive crashes per instance, which drives the backoff and the
        # give-up budget. Kept out of the row because restart_count is the
        # provisioning log's generation number and has to keep increasing.
        # In-process only, like the model instance manager's equivalent: a
        # worker restart forgets the depth of a crash loop, but not the
        # verdict -- an instance already parked in ERROR is skipped by the
        # sync pass whatever this says.
        self._restart_attempts: Dict[int, int] = {}

        os.makedirs(self._provision_log_dir, exist_ok=True)

    async def watch_cache_service_instances_event(self):
        """
        Loop to watch cache service instances' event and handle.
        """
        self._loop = asyncio.get_running_loop()
        await watch_forever(
            "cache service instances",
            self._clientset.cache_service_instances.awatch,
            callback=self._handle_cache_service_instance_event,
        )

    def _handle_cache_service_instance_event(self, event: Event):
        """
        Handle cache service instance events.

        Args:
            event: The cache service instance event to handle.
        """
        instance = CacheServiceInstance.model_validate(event.data)
        if instance.worker_id != self._worker_id:
            return

        logger.trace(
            f"Received event: {str(event.type)}, instance id: {instance.id}, "
            f"cache service id: {instance.cache_service_id}, "
            f"state: {str(instance.state)}"
        )

        if event.type == EventType.DELETED:
            self._stop_cache_service_instance(instance)
            return

        if instance.state == CacheServiceStateEnum.PENDING:
            self._schedule_start(instance)

    def _schedule_start(self, instance: CacheServiceInstance):
        """
        Launch on the event loop thread, wherever the caller runs.

        The launch forks the provisioning subprocess, and fork keeps only the
        calling thread: a lock held by any other thread at that moment stays
        locked in the child. That is not made better or worse by which thread
        forks, but having all three managers fork from the same one leaves a
        single place to change if the start method is ever revisited. The
        model instance and benchmark managers fork from the loop, so this does
        too, including on the stale-PENDING recovery path that runs on the
        sync thread.

        A start already in flight for this instance is not started a second
        time.
        """
        if not self._claim_start(instance.id):
            logger.debug(
                f"Skipped starting cache service instance {instance.id}: "
                "a start is already in flight"
            )
            return

        loop = self._loop
        if loop is None or _running_on(loop):
            self._start_cache_service_instance(instance)
            return

        try:
            loop.call_soon_threadsafe(self._start_cache_service_instance, instance)
        except RuntimeError as e:
            # The loop is closing; the claim would otherwise block the retry
            # that a restarted worker makes.
            self._release_start(instance.id)
            logger.debug(f"Skipped starting cache service instance {instance.id}: {e}")

    def _claim_start(self, instance_id: int) -> bool:
        """Mark a start as in flight; False when one already is."""
        with CacheServiceManager._start_lock:
            if instance_id in self._starting:
                return False
            self._starting.add(instance_id)
            self._last_start_attempt[instance_id] = datetime.now(timezone.utc)
            return True

    def _release_start(self, instance_id: int):
        with CacheServiceManager._start_lock:
            self._starting.discard(instance_id)

    def _forget_start(self, instance_id: int):
        with CacheServiceManager._start_lock:
            self._starting.discard(instance_id)
            self._last_start_attempt.pop(instance_id, None)

    def _is_provisioning(self, instance_id: int) -> bool:
        """
        Whether a start for this instance is still under way in this process,
        either claimed but not yet spawned, or running in a subprocess.
        """
        with CacheServiceManager._start_lock:
            if instance_id in self._starting:
                return True
        process = self._provisioning_processes.get(instance_id)
        if process is None:
            return False
        # Reap an exited child here so it does not linger as a zombie until
        # the instance is stopped, and so is_alive() reports the truth.
        process.join(timeout=0)
        if process.is_alive():
            return True
        self._provisioning_processes.pop(instance_id, None)
        return False

    def _provision_log_path(self, instance_id: int, restart_count: int) -> str:
        """One provisioning log per start, numbered like a model instance's
        ({id}.{restart_count}.log), so the previous start's log survives the
        next one and the log viewer can offer it."""
        return f"{self._provision_log_dir}/{instance_id}.{restart_count}.log"

    def _container_log_path(self, instance_id: int, restart_count: int) -> str:
        """The container's own output for one start, numbered alongside that
        start's provisioning log."""
        return f"{self._provision_log_dir}/{instance_id}.container.{restart_count}.log"

    def _cleanup_old_logs(self, instance_id: int, restart_count: int):
        """Keep the current and previous start's logs, drop the rest.

        restart_count 0 is a fresh lifecycle, so anything present belongs to a
        previous owner of a reused id.
        """
        keep = (
            {restart_count}
            if restart_count == 0
            else {restart_count, restart_count - 1}
        )
        for path in Path(self._provision_log_dir).glob(f"{instance_id}.*.log"):
            try:
                if int(path.stem.rsplit(".", 1)[-1]) not in keep:
                    path.unlink(missing_ok=True)
            except (ValueError, OSError) as e:
                logger.debug(f"Skipped pruning log {path}: {e}")

    def _start_cache_service_instance(self, instance: CacheServiceInstance):
        """
        Launch the provisioning subprocess for a cache service instance.

        Resolving the provider declaration and pulling its image runs in a
        subprocess whose output lands in the instance's provisioning log, so a
        start that is slow or fails before the container exists is readable
        while it happens. Ports are allocated here, in the process that owns
        the worker-wide allocation, and passed in.

        Args:
            instance: The cache service instance to start.
        """
        try:
            if self._is_provisioning(instance.id):
                logger.debug(
                    f"Skipped starting cache service instance {instance.id}: "
                    "a start is already in flight"
                )
                return

            self._release_ports(instance.id)
            port, metrics_port = self._allocate_ports(instance)

            restart_count = instance.restart_count or 0
            self._cleanup_old_logs(instance.id, restart_count)
            log_file_path = self._provision_log_path(instance.id, restart_count)
            process = multiprocessing.Process(
                target=CacheServiceManager._provision_cache_service_instance,
                args=(
                    instance,
                    port,
                    metrics_port,
                    self._clientset.headers,
                    log_file_path,
                    self._config,
                    # Resolved here so the child does not re-probe the default
                    # registry, which may cost a network round trip.
                    resolve_fallback_registry(self._config),
                ),
            )
            process.daemon = False
            process.start()
            self._provisioning_processes[instance.id] = process

            # Started now rather than once the container exists: the stream
            # loop retries until it does, so a container that dies seconds
            # after starting still leaves its output behind.
            self._container_logs.start(
                instance.id,
                instance.get_deployment_metadata().name,
                self._container_log_path(instance.id, restart_count),
            )
            logger.info(
                f"Provisioning cache service instance {instance.id} "
                f"(service id={instance.cache_service_id}) on port {port}"
            )
        except Exception as e:
            self._release_ports(instance.id)
            self._update_cache_service_instance(
                instance.id,
                state=CacheServiceStateEnum.ERROR,
                state_message=str(e),
            )
            logger.error(
                f"Failed to start cache service instance {instance.id} "
                f"(service id={instance.cache_service_id}): {e}"
            )
        finally:
            self._release_start(instance.id)

    @staticmethod
    def _provision_cache_service_instance(
        instance: CacheServiceInstance,
        port: int,
        metrics_port: int,
        client_headers: dict,
        log_file_path: str,
        cfg: Config,
        fallback_registry: Optional[str] = None,
    ):
        """
        Provision a cache service instance in a subprocess.
        Exits the subprocess once the workload has been created.

        Args:
            instance: The cache service instance to provision.
            port: The service port allocated for the instance.
            metrics_port: The metrics port allocated for the instance.
            client_headers: The headers for the clientset.
            log_file_path: The path to the provisioning log file.
            cfg: The configuration.
            fallback_registry: The container registry to fall back to when a
                provider image carries none.
        """
        setproctitle.setproctitle(f"gpustack_cache_service_instance_{instance.id}")
        add_signal_handlers()

        clientset = ClientSet(
            base_url=cfg.get_server_url(),
            headers=client_headers,
        )

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with RedirectStdoutStderr(log_file):
                # The container runtime logs the image pull through its own
                # logger; without this it has no handler and the pull — the
                # slowest part of a start — leaves no trace at all.
                setup_logging(debug=cfg.debug)
                setup_runtime_logging()
                CacheServiceProvisioner(
                    clientset=clientset,
                    instance=instance,
                    cfg=cfg,
                    port=port,
                    metrics_port=metrics_port,
                    fallback_registry=fallback_registry,
                ).start()

    def _allocate_ports(self, instance: CacheServiceInstance) -> Tuple[int, int]:
        """
        Allocate the instance's (port, metrics_port) pair on this worker.

        Ports already handed out by this process and ports recorded on other
        cache service instances of this worker are both treated as
        unavailable, so a restarted worker can't re-issue a port an existing
        instance holds. The metrics port additionally excludes the service
        port picked just before it.
        """
        with CacheServiceManager._port_lock:
            unavailable_ports = {
                port for pair in self._assigned_ports.values() for port in pair
            }
            try:
                instances_page = self._clientset.cache_service_instances.list(
                    # page=-1 disables pagination: a truncated page would
                    # blind the conflict check to the ports it dropped.
                    params={"worker_id": self._worker_id, "page": -1}
                )
                for existing in instances_page.items or []:
                    if existing.id == instance.id:
                        continue
                    if existing.port:
                        unavailable_ports.add(existing.port)
                    if existing.metrics_port:
                        unavailable_ports.add(existing.metrics_port)
            except Exception as e:
                logger.warning(
                    f"Failed to list cache service instances for port "
                    f"allocation: {e}"
                )

            # Prefer the ports already recorded on the instance: engines
            # attached to this cache server carry them in denormalized
            # snapshots that nothing refreshes, so a restart that changed
            # ports would strand every running deployment on a dead
            # endpoint until its model instances are recreated.
            if (
                instance.port
                and instance.metrics_port
                and instance.port not in unavailable_ports
                and instance.metrics_port not in unavailable_ports
                and network.is_port_available(instance.port)
                and network.is_port_available(instance.metrics_port)
            ):
                self._assigned_ports[instance.id] = (
                    instance.port,
                    instance.metrics_port,
                )
                return instance.port, instance.metrics_port

            port = network.get_free_port(
                port_range=self._config.service_port_range,
                unavailable_ports=unavailable_ports,
            )
            unavailable_ports.add(port)
            metrics_port = network.get_free_port(
                port_range=self._config.service_port_range,
                unavailable_ports=unavailable_ports,
            )
            self._assigned_ports[instance.id] = (port, metrics_port)
            return port, metrics_port

    def _release_ports(self, instance_id: int):
        with CacheServiceManager._port_lock:
            self._assigned_ports.pop(instance_id, None)

    def sync_cache_service_instances_state(self):
        """
        Synchronize managed cache service instances' state on this worker:
        - PENDING past PENDING_START_GRACE_SECONDS -> start (the start path
          is idempotent), recovering starts that never took effect.
        - Workload missing, failed, unhealthy or exited -> restart with
          exponential backoff; after MAX_CONSECUTIVE_RESTARTS crashes -> ERROR.
        - Health probe passes -> RUNNING (healthy).
        - Health probe fails after RUNNING -> UNREACHABLE.
        - STARTING with a failing probe is left alone (still booting).
        """
        instances_page = self._clientset.cache_service_instances.list(
            # page=-1 disables pagination: instances beyond a page would
            # never be synced or restarted.
            params={"worker_id": self._worker_id, "page": -1}
        )
        # Prune start bookkeeping for rows that no longer exist (a missed
        # DELETED event would otherwise accumulate entries forever).
        listed_ids = {instance.id for instance in instances_page.items or []}
        with CacheServiceManager._start_lock:
            for stale_id in set(self._last_start_attempt) - listed_ids:
                self._last_start_attempt.pop(stale_id, None)
        for stale_id in set(self._provisioning_processes) - listed_ids:
            process = self._provisioning_processes.pop(stale_id, None)
            if process is not None and process.is_alive():
                terminate_process_tree(process.pid)
        if not instances_page.items:
            return

        # Parent services are fetched once per sync pass: siblings of a
        # per-node service share the same parent row.
        parent_services: Dict[int, Optional[CacheServicePublic]] = {}
        for instance in instances_page.items:
            if instance.worker_id != self._worker_id:
                continue
            if instance.state == CacheServiceStateEnum.PENDING:
                self._start_stale_pending_instance(instance)
                continue
            if instance.state not in (
                CacheServiceStateEnum.STARTING,
                CacheServiceStateEnum.RUNNING,
                CacheServiceStateEnum.UNREACHABLE,
            ):
                continue
            if self._is_provisioning(instance.id):
                # A restart is re-creating the workload; the gap between the
                # old one being deleted and the new one existing must not read
                # as a crash.
                continue
            try:
                cache_service = self._get_parent_service(
                    parent_services, instance.cache_service_id
                )
                if cache_service is None:
                    # The parent is gone; the instance row is about to be
                    # cascade-deleted, so there is nothing to sync against.
                    logger.debug(
                        f"Skipped syncing cache service instance {instance.id}: "
                        f"parent service {instance.cache_service_id} not found"
                    )
                    continue
                self._sync_single_cache_service_instance_state(instance, cache_service)
            except Exception as e:
                logger.error(
                    f"Failed to sync cache service instance {instance.id} "
                    f"(service id={instance.cache_service_id}) state: {e}"
                )

    def _start_stale_pending_instance(self, instance: CacheServiceInstance):
        """
        Start a PENDING instance whose start never took effect.

        Starts are triggered by the instance's PENDING event, so a start that
        was lost — event never delivered, worker restarted mid-start, the
        STARTING write-back never reached the server — would otherwise leave
        the instance PENDING forever, with or without a running container.
        Instances PENDING for less than PENDING_START_GRACE_SECONDS are left
        to the event path, starts still in flight are not duplicated, and one
        that already ran that recently is not repeated — a start whose
        write-back never lands would otherwise recreate the container on every
        sync pass.
        """
        now = datetime.now(timezone.utc)
        updated_at = instance.updated_at
        if (
            updated_at is not None
            and (now - updated_at).total_seconds() < PENDING_START_GRACE_SECONDS
        ):
            return

        if self._is_provisioning(instance.id):
            return

        with CacheServiceManager._start_lock:
            last_attempt = self._last_start_attempt.get(instance.id)
        if (
            last_attempt is not None
            and (now - last_attempt).total_seconds() < PENDING_START_GRACE_SECONDS
        ):
            return

        logger.info(
            f"Starting cache service instance {instance.id} "
            f"(service id={instance.cache_service_id}): still pending "
            f"after {PENDING_START_GRACE_SECONDS}s"
        )
        self._schedule_start(instance)

    def _get_parent_service(
        self,
        cache: Dict[int, Optional[CacheServicePublic]],
        cache_service_id: int,
    ) -> Optional[CacheServicePublic]:
        if cache_service_id not in cache:
            try:
                cache[cache_service_id] = self._clientset.cache_services.get(
                    id=cache_service_id
                )
            except NotFoundException:
                cache[cache_service_id] = None
        return cache[cache_service_id]

    def _sync_single_cache_service_instance_state(
        self,
        instance: CacheServiceInstance,
        cache_service: CacheServicePublic,
    ):
        """Synchronize a single cache service instance's state."""
        deployment_metadata = instance.get_deployment_metadata()
        workload = get_workload(deployment_metadata.name)

        # A cache server is a service: it has no successful end, so a clean
        # exit counts as a crash alongside a failure or a vanished workload.
        if classify_workload(workload) not in (
            WorkloadPhase.LAUNCHING,
            WorkloadPhase.RUNNING,
        ):
            self._restart_crashed_cache_service_instance(
                instance, cache_service, deployment_metadata.name
            )
            return

        ready = self._probe_ready(instance, cache_service.provider_name)
        now = datetime.now(timezone.utc)
        if ready:
            updates = {}
            if (
                instance.state != CacheServiceStateEnum.RUNNING
                or instance.healthy is not True
            ):
                updates.update(
                    state=CacheServiceStateEnum.RUNNING,
                    healthy=True,
                    last_check_at=now,
                    state_message="",
                )
            # An instance that has stayed healthy past the reset window has
            # broken out of its crash loop; clear the consecutive-restart
            # budget so a much later crash gets a fresh set of attempts.
            if RESTART_BUDGET.should_forgive(
                self._restart_attempts.get(instance.id, 0),
                instance.last_restart_time,
                now,
            ):
                self._restart_attempts.pop(instance.id, None)
            if updates:
                self._update_cache_service_instance(instance.id, **updates)
            return

        if instance.state == CacheServiceStateEnum.RUNNING:
            self._update_cache_service_instance(
                instance.id,
                state=CacheServiceStateEnum.UNREACHABLE,
                healthy=False,
                last_check_at=now,
            )
        # STARTING with a failing probe: the server is still booting, leave it
        # for the next sync round.

    def _restart_crashed_cache_service_instance(
        self,
        instance: CacheServiceInstance,
        cache_service: CacheServicePublic,
        workload_name: str,
    ):
        """
        Recover a cache service instance whose workload is missing or dead by
        re-entering PENDING (which retriggers the normal start path) with
        exponential backoff. After MAX_CONSECUTIVE_RESTARTS consecutive
        crashes — or immediately when the parent service disables
        restart_on_error — the instance is parked in ERROR until a manual
        restart.
        """
        if cache_service.restart_on_error is False:
            if instance.state != CacheServiceStateEnum.ERROR:
                self._update_cache_service_instance(
                    instance.id,
                    state=CacheServiceStateEnum.ERROR,
                    state_message=(
                        "Cache server exited. Automatic restart is disabled "
                        "for this service; restart it manually."
                    ),
                    healthy=False,
                )
            return

        now = datetime.now(timezone.utc)
        attempts = self._restart_attempts.get(instance.id, 0)
        decision = RESTART_BUDGET.decide(attempts, instance.last_restart_time, now)
        if decision.action == RestartActionEnum.GIVE_UP:
            self._update_cache_service_instance(
                instance.id,
                state=CacheServiceStateEnum.ERROR,
                state_message=(
                    f"Cache server keeps crashing "
                    f"({MAX_CONSECUTIVE_RESTARTS} restarts attempted). "
                    "Check the service logs for the failure cause."
                ),
                healthy=False,
            )
            return
        if decision.action == RestartActionEnum.WAIT:
            # Within the backoff window; retry on a later sync round.
            return

        try:
            delete_workload(workload_name)
        except Exception as e:
            # The workload may already be gone.
            logger.debug(
                f"Skipped deleting crashed cache service workload "
                f"{workload_name}: {e}"
            )

        attempt = decision.attempt
        self._restart_attempts[instance.id] = attempt
        logger.info(
            f"Restarting crashed cache service {cache_service.name} instance "
            f"(id={instance.id}), attempt {attempt}/{MAX_CONSECUTIVE_RESTARTS}"
        )
        self._update_cache_service_instance(
            instance.id,
            state=CacheServiceStateEnum.PENDING,
            # Monotonic: this numbers the provisioning log of the start it is
            # about to trigger, so it must not be reset the way the
            # consecutive-crash count above is.
            restart_count=(instance.restart_count or 0) + 1,
            last_restart_time=now,
            state_message=(
                f"Cache server exited; restarting "
                f"(attempt {attempt}/{MAX_CONSECUTIVE_RESTARTS})."
            ),
            healthy=False,
        )

    def _probe_ready(self, instance: CacheServiceInstance, provider_name: str) -> bool:
        """
        Probe the cache server per the provider's health check declaration.
        Managed cache servers run with host networking on this worker, so
        loopback reaches them directly.
        """
        provider = get_cache_provider(provider_name)
        health_check = provider.health_check if provider else CacheProviderHealthCheck()
        host = "127.0.0.1"
        port = (
            instance.metrics_port if health_check.target == "metrics" else instance.port
        )
        if not port:
            return False

        if health_check.scheme == "http":
            path = health_check.path or "/"
            if not path.startswith("/"):
                path = "/" + path
            try:
                resp = httpx.get(
                    f"http://{host}:{port}{path}",
                    timeout=HEALTH_PROBE_TIMEOUT_SECONDS,
                )
                return resp.status_code < 400
            except Exception:
                return False

        try:
            with socket.create_connection(
                (host, port), timeout=HEALTH_PROBE_TIMEOUT_SECONDS
            ):
                return True
        except Exception:
            return False

    def _stop_cache_service_instance(self, instance: CacheServiceInstance):
        """
        Stop the instance's workload and free its tracked ports.

        Args:
            instance: The cache service instance to stop.
        """
        # Tear down the provisioning subprocess first: one that is mid-pull
        # would otherwise create the workload again right after it is deleted.
        self._container_logs.stop(instance.id)

        process = self._provisioning_processes.pop(instance.id, None)
        if process is not None and process.is_alive():
            terminate_process_tree(process.pid)

        deployment_metadata = instance.get_deployment_metadata()
        try:
            delete_workload(deployment_metadata.name)
        except Exception as e:
            # The workload may already be gone (never created or cleaned up).
            logger.warning(
                f"Failed to delete cache service workload "
                f"{deployment_metadata.name}: {e}"
            )
        self._release_ports(instance.id)
        self._forget_start(instance.id)
        self._restart_attempts.pop(instance.id, None)
        self._purge_provision_logs(instance.id)
        logger.info(
            f"Stopped cache service instance {instance.id} "
            f"(service id={instance.cache_service_id})"
        )

    def _purge_provision_logs(self, instance_id: int):
        """Remove every generation of the instance's provisioning log. Only on
        teardown: a restart keeps them so the log viewer can still show the
        start that failed."""
        try:
            for path in Path(self._provision_log_dir).glob(f"{instance_id}.*.log"):
                path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(
                f"Failed to remove provisioning logs of cache service "
                f"instance {instance_id}: {e}"
            )

    def _update_cache_service_instance(self, id: int, **kwargs) -> bool:
        """
        Update cache service instance with given fields.

        Args:
            id: The ID of the cache service instance to update.
            **kwargs: The fields to update, group by field name and value.

        Returns:
            Whether the update was applied.
        """
        return update_cache_service_instance(self._clientset, id, **kwargs)
