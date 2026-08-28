import asyncio
import contextlib
from functools import partial
from datetime import datetime, timezone
import json
import re
import threading
import time

import requests
import os
from typing import Dict, Optional, Set, List, Callable
from pathlib import Path
import logging

from gpustack_runtime.deployer import (
    get_workload,
    delete_workload,
)
from gpustack_runtime.deployer.__utils__ import compare_versions

from gpustack import envs
from gpustack.api.exceptions import NotFoundException
from gpustack.config.config import Config
from gpustack.config import registration
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    is_built_in_backend,
    is_custom_backend,
)
from gpustack.utils import network
from gpustack.utils.convert import safe_int
from gpustack.utils.command import find_int_parameter
from gpustack.worker.backends.ascend_mindie import AscendMindIEServer
from gpustack.worker.backends.sglang import SGLangServer
from gpustack.utils.command import resolve_executor_backend
from gpustack.worker.backends.vllm import VLLMServer
from gpustack.worker.backends.vox_box import VoxBoxServer
from gpustack.worker.backends.custom import CustomServer
from gpustack.routes.worker.logs import (
    extract_container_restart_count,
    extract_restart_count,
)
from gpustack.worker.model_meta import get_meta_from_running_instance
from gpustack.client import ClientSet
from gpustack.schemas.models import (
    BackendEnum,
    Model,
    ModelUpdate,
    ModelInstance,
    ModelInstanceUpdate,
    ModelInstanceStateEnum,
    get_backend,
    DistributedServerCoordinateModeEnum,
    ModelInstanceSubordinateWorker,
    CategoryEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker.controlloop import (
    ContainerLogPersistence,
    ProvisionRunner,
    RestartActionEnum,
    RestartBudget,
    WorkloadPhase,
    classify_workload,
    describe_workload_failure,
    update_resource,
    watch_forever,
)
from gpustack.worker.inference_backend_manager import InferenceBackendManager

logger = logging.getLogger(__name__)

# Inference health check error message
_INFERENCE_HEALTH_CHECK_FAILED_MESSAGE = "Inference health check failed."

# Last-resort message for a workload that stopped serving without saying why.
_WORKLOAD_FAILED_MESSAGE = "Inference server exited or unhealthy."

# One health-check cycle (+2s margin) to let a container return after a stream
# EOF; beyond that gpustack marks it ERROR and takes over recovery.
LOG_RECONNECT_GRACE_SECONDS = envs.MODEL_INSTANCE_HEALTH_CHECK_INTERVAL + 2

RESTART_BUDGET = RestartBudget(
    base_delay_seconds=10,
    max_delay_seconds=300,
    # Unlike a cache server, an inference server is not given up on: the
    # failure is often outside it (a registry outage, a node coming back), and
    # a user who wants it to stop retrying turns off the model's
    # restart_on_error. See _restart_backoff_counts for what is not persisted.
    max_attempts=None,
    first_attempt_immediate=True,
)

# Global lock for port assignment to avoid pickle serialization issues
_port_lock = threading.Lock()

_SERVER_CLASS_MAPPING = {
    BackendEnum.VLLM: VLLMServer,
    BackendEnum.SGLANG: SGLangServer,
    BackendEnum.VOX_BOX: VoxBoxServer,
    BackendEnum.ASCEND_MINDIE: AscendMindIEServer,
}

# Annotation the operator device plugin writes onto the Pod after allocation,
# e.g. {"<container>": {"devices": {"groups": [{"accelerators": [{"id": "<GPU UUID>",
# "index": 0, "mode": 3, "allocated": 640000}]}]}, "deviceIDs": [...]}}.
_ALLOCATED_ACCELERATORS_ANNOTATION = "device.gpustack.ai/accelerator.allocated"


def _parse_allocated_accelerators(annotations: Optional[Dict[str, str]]) -> List[dict]:
    """
    Parse the allocated-accelerators annotation into a flat accelerator list.
    Tolerates missing/malformed payloads (device-plugin version skew):
    any parse problem means "allocation unknown", never a sync failure.
    """
    raw = (annotations or {}).get(_ALLOCATED_ACCELERATORS_ANNOTATION)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, dict):
        return []

    accelerators = []
    for container_allocation in parsed.values():
        if not isinstance(container_allocation, dict):
            continue
        devices = container_allocation.get("devices")
        if not isinstance(devices, dict):
            continue
        groups = devices.get("groups")
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_accelerators = group.get("accelerators")
            if not isinstance(group_accelerators, list):
                continue
            for accelerator in group_accelerators:
                if isinstance(accelerator, dict) and accelerator.get("id"):
                    accelerators.append(accelerator)
    return accelerators


def provision_model_instance(
    mi: ModelInstance,
    backend: BackendEnum,
    worker_id: int,
    inference_backend: InferenceBackend,
    fallback_registry: Optional[str],
    clientset: ClientSet,
    cfg: Config,
):
    """
    The domain half of provisioning, run by the shared subprocess entry point.

    Module level so spawn can pickle it by reference; the leading arguments are
    bound with functools.partial at the call site.
    """
    server_cls = _SERVER_CLASS_MAPPING.get(backend, CustomServer)
    server_cls(
        clientset, mi, cfg, worker_id, inference_backend, fallback_registry
    ).start()


class ServeManager:
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
    _serve_log_dir: str
    """
    The directory to store logs of serving model instances(in subprocess).
    """

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    """
    The clientset to access the API server.
    """
    _inference_backend_manager: InferenceBackendManager
    """
    The inference backend manager.
    """
    _error_model_instances: Dict[int, ModelInstance]
    """
    The mapping of model instance ID to error model instances.
    Used to restart error model instances.
    """
    _model_cache_by_instance: Dict[int, Model]
    """
    The cache of models by model instance ID.
    Used to avoid redundant API calls to get model information.
    """
    _model_instance_by_instance_id: Dict[int, ModelInstance]

    _clientset_getter: Callable[[], ClientSet]
    _worker_id_getter: Callable[[], int]

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
        cfg: Config,
    ):
        self._worker_id_getter = worker_id_getter
        self._config = cfg
        self._serve_log_dir = f"{cfg.log_dir}/serve"
        self._clientset_getter = clientset_getter

        self._provisioning = ProvisionRunner(cfg, lambda: self._clientset.headers)
        self._container_logs = ContainerLogPersistence(LOG_RECONNECT_GRACE_SECONDS)
        self._error_model_instances = {}
        self._model_cache_by_instance = {}
        self._model_instance_by_instance_id = {}

        # Instance-level port tracking to avoid conflicts
        self._assigned_ports: Dict[int, Set[int]] = {}

        # Consecutive restarts per instance, cleared once it serves again, so
        # a crash long after a recovery meets a fresh set of attempts.
        #
        # In-process only, so a worker restart forgets the backoff and the
        # instances still crash-looping get one immediate retry each. The
        # instance row cannot hold it: restart_count is the log-file
        # generation number ({id}.{restart_count}.log, with _cleanup_old_logs
        # reading 0 as "fresh lifecycle, purge everything"), so it has to keep
        # increasing and cannot double as a consecutive-crash counter the way
        # a cache service instance's does. Persisting this needs a column of
        # its own.
        self._restart_backoff_counts: Dict[int, int] = {}

        # Inference health check failure tracking
        # {model_instance_id: failure_count}
        self._inference_health_check_failures: Dict[int, int] = {}

        # Track last successful inference per port (set by worker proxy)
        self._last_successful_inference: Dict[int, float] = {}
        # Track last health check time per model instance
        self._last_health_check_time: Dict[int, float] = {}

        # Timestamp of the last authoritative (uncached) DB reconciliation in
        # the state sync, for the optional periodic backstop. Starts "now" so
        # the first forced reconciliation is one full period in, not immediate.
        self._last_state_reconcile_time: float = time.time()

        os.makedirs(self._serve_log_dir, exist_ok=True)

    def record_successful_inference(self, instance_id: int):
        """Called by worker proxy on successful inference response."""
        self._last_successful_inference[instance_id] = time.time()

    async def watch_models(self):
        """
        Loop to watch models to keep the cache updated.

        """

        # No callback: the stream is consumed only to keep the client cache warm.
        await watch_forever("models", self._clientset.models.awatch)

    async def watch_model_instances_event(self):
        """
        Loop to watch model instances' event and handle.

        """

        await watch_forever(
            "model instances",
            self._clientset.model_instances.awatch,
            callback=self._handle_model_instance_event,
        )

    async def watch_model_instances(self):
        """
        Loop to post process model instances, for example, restarting error instances.

        """

        logger.debug("Watching model instances.")

        while True:
            try:
                for mi in list(self._error_model_instances.values()):
                    self._restart_error_model_instance(mi)
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Error restarting model instances: {e}")
                await asyncio.sleep(5)

    def sync_model_instances_state(self):  # noqa: C901
        """
        Synchronize model instances' state.

        - If the model instance is scheduled but not initialized, skip.
        - If the provision process is still alive, skip.
        - If the workload is still launching, skip.
        - If the workload is not existed, unhealthy, inactive or failed, update the model instance state to ERROR.
        - If everything is fine, update the model instance state to RUNNING.
        """

        # Snapshot local state BEFORE the list call. Reversing this order
        # races with CREATED events and reaps freshly-assigned instances.
        local_assigned_ids = {
            mid
            for mid, mi in self._model_instance_by_instance_id.items()
            if mi.get_deployment_metadata(self._worker_id) is not None
        }

        # Read from the watch-backed cache. It holds the full set, so the
        # common (nothing-to-reap) path stays O(1) with no server/DB round
        # trip and scales independently of worker count — a direct per-worker
        # poll here is one SELECT over model_instances per worker every few
        # seconds, which does not scale to hundreds of workers.
        response = self._clientset.model_instances.list()
        all_items = response.items or []
        reap_ids = local_assigned_ids - {
            mi.id
            for mi in all_items
            if mi.get_deployment_metadata(self._worker_id) is not None
        }

        # Optional periodic reconciliation against DB truth even when the cache
        # reports nothing reapable. The cache and local serving state are both
        # fed by the same awatch stream, so a ghost re-seeded into the cache
        # sits in both and `local - cache` never flags it; an independent DB
        # read is the only thing that catches that. Disabled by default (the
        # server-side cached_all fix plus reconnect-driven reaping cover the
        # realistic cases); opt in via the
        # GPUSTACK_MODEL_INSTANCE_STATE_RECONCILE_INTERVAL env var when a
        # coordinator may drop DELETEDs on a live stream.
        reconcile_interval = envs.MODEL_INSTANCE_STATE_RECONCILE_INTERVAL
        now = time.time()
        force_authoritative = (
            reconcile_interval > 0
            and now - self._last_state_reconcile_time >= reconcile_interval
        )

        # Reaping tears down live workloads, so it must act on an authoritative
        # snapshot. The cache is not one at every instant: awatch clears it and
        # flips _watch_started before the replay snapshot arrives, so a read
        # during a reconnect window can see an empty-but-"authoritative" cache
        # and surface healthy instances as stale. So confirm reap candidates
        # against a fresh, unpaginated, uncached fetch — this only touches the
        # DB when something looks reapable or on the periodic reconciliation.
        if reap_ids or force_authoritative:
            response = self._clientset.model_instances.list(
                params={"page": -1},
                use_cache=False,
            )
            # Any successful authoritative fetch counts as a reconciliation —
            # whether triggered by the timer or by a reap candidate — so reset
            # the timer here (list() raises on API failure, so a failed fetch
            # doesn't advance it and the next tick retries). Skipped when the
            # backstop is disabled to keep the timestamp meaningless-but-inert.
            if reconcile_interval > 0:
                self._last_state_reconcile_time = now
            all_items = response.items or []
            reap_ids = local_assigned_ids - {
                mi.id
                for mi in all_items
                if mi.get_deployment_metadata(self._worker_id) is not None
            }

        # An empty authoritative result is legitimate (user stopped every
        # model); list() raises on API failure rather than returning empty, so
        # local_assigned_ids - ∅ correctly reaps everything still tracked.
        for stale_id in reap_ids:
            stale = self._model_instance_by_instance_id.get(stale_id)
            if stale is None:
                continue
            logger.info(
                f"Reaping stale model instance {stale.name} (id={stale_id}); "
                f"server no longer reports it assigned to this worker."
            )
            try:
                # Equivalent to a dropped DELETED event, so purge logs too —
                # otherwise a reused id would inherit this instance's stale logs.
                self._stop_model_instance(stale, delete_logs=True)
            except Exception as e:
                logger.warning(f"Failed to reap stale model instance {stale.name}: {e}")

        if not all_items:
            # Nothing left to sync; reap pass above already handled stale
            # local state.
            return

        model_instances: List[ModelInstance] = []
        for model_instance in all_items:
            # if the model instance is assigned to this worker, it must be scheduled.
            # But we don't need to sync the scheduled model when it is not initialized yet.
            if (
                model_instance.worker_id == self._worker_id
                and model_instance.state != ModelInstanceStateEnum.SCHEDULED
            ):
                model_instances.append(model_instance)
            if (
                model_instance.distributed_servers
                and model_instance.distributed_servers.subordinate_workers
            ):
                for sw in model_instance.distributed_servers.subordinate_workers:
                    if sw.worker_id == self._worker_id:
                        model_instances.append(model_instance)
                        break

        for model_instance in model_instances:
            # Skip if the provision process has not exited yet.
            if self._is_provisioning(model_instance):
                logger.trace(
                    f"Model instance {model_instance.name} is provisioning. Skipping sync."
                )
                continue

            is_main_worker = model_instance.worker_id == self._worker_id

            # Skip if the workload is still launching.
            # Use deployment metadata name for subordinate workers (e.g., "model-f0")
            # since their workload name differs from the model instance name.
            if is_main_worker:
                workload = get_workload(model_instance.name)
            else:
                deployment_metadata = model_instance.get_deployment_metadata(
                    self._worker_id
                )
                workload_name = (
                    deployment_metadata.name
                    if deployment_metadata
                    else model_instance.name
                )
                workload = get_workload(workload_name)

            phase = classify_workload(workload)
            if phase == WorkloadPhase.LAUNCHING:
                logger.trace(
                    f"Model instance {model_instance.name} workload is still launching. Skipping sync."
                )
                continue

            # An inference server is a service: it has no successful end, so a
            # clean exit is an error alongside a failure or a vanished workload.
            if phase != WorkloadPhase.RUNNING:
                # Only if not in ERROR state yet.
                if model_instance.state != ModelInstanceStateEnum.ERROR:
                    # Surface the workload's own diagnosis (e.g. a device-plugin
                    # admission rejection, an image-pull failure, an exit code)
                    # when available.
                    failure_message = describe_workload_failure(
                        workload, _WORKLOAD_FAILED_MESSAGE
                    )
                    with contextlib.suppress(NotFoundException):
                        # Get patch dict for main worker.
                        if is_main_worker:
                            patch_dict = {
                                "state": ModelInstanceStateEnum.ERROR,
                                "state_message": failure_message,
                            }
                        # Get patch dict for subordinate worker.
                        else:
                            sw_pos = next(
                                (
                                    i
                                    for i, sw in enumerate(
                                        model_instance.distributed_servers.subordinate_workers
                                    )
                                    if sw.worker_id == self._worker_id
                                ),
                            )
                            sw = model_instance.distributed_servers.subordinate_workers[
                                sw_pos
                            ]
                            sw.state = ModelInstanceStateEnum.ERROR
                            sw.state_message = failure_message
                            patch_dict = {
                                f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                            }
                        # Update model instance.
                        self._update_model_instance(model_instance.id, **patch_dict)
                continue

            # Otherwise, update model instance state to RUNNING if everything is fine.
            model = self._get_model(model_instance)
            if model.gpu_type_selector:
                # vGPU: read back the real device allocation the operator
                # device plugin wrote onto the workload's annotations.
                self._sync_vgpu_allocation(model_instance, workload, is_main_worker)
            if not model.backend_version:
                # backend version may be empty on initialization.
                # try to refresh to get updated model info on syncs.
                model = self._refresh_model(model_instance)

            backend = get_backend(model)
            health_check_path = self._get_health_check_path(
                backend, model.owner_principal_id
            )
            if model.env and 'GPUSTACK_MODEL_HEALTH_CHECK_PATH' in model.env:
                # NOTE: There is no known use case for now. Keep this in case the built-in backends
                # introduce breaking changes and the default health check path no longer works.
                health_check_path = model.env['GPUSTACK_MODEL_HEALTH_CHECK_PATH']

            with contextlib.suppress(NotFoundException):
                # Get patch dict for main worker.
                if is_main_worker:
                    subordinate_state = self._get_main_worker_distributed_state(
                        model_instance
                    )
                    if subordinate_state is None:
                        if model_instance.state == ModelInstanceStateEnum.RUNNING:
                            self._restart_backoff_counts.pop(model_instance.id, None)
                            continue

                        if (
                            model_instance.state == ModelInstanceStateEnum.ERROR
                            or not is_ready(
                                backend, model_instance, health_check_path, model
                            )
                        ):
                            continue

                        self._restart_backoff_counts.pop(model_instance.id, None)
                        patch_dict = {
                            "state": ModelInstanceStateEnum.RUNNING,
                            "state_message": "",
                        }

                        # Fetch model meta once running.
                        meta = get_meta_from_running_instance(
                            model_instance, backend, model
                        )
                        if meta:
                            # Some meta is set in server evaluation and should be preserved, so we update meta instead of overwrite.
                            merged_meta = dict(model.meta or {})
                            merged_meta.update(meta)
                            if merged_meta != model.meta:
                                self._update_model(model.id, meta=merged_meta)
                    elif subordinate_state["should_update"]:
                        patch_dict = {
                            "state": subordinate_state["state"],
                            "state_message": subordinate_state["state_message"],
                        }
                    else:
                        continue
                # Get patch dict for subordinate worker.
                else:
                    # For initialize later mode, the state is set to RUNNING directly,
                    # which means the subordinate worker doesn't need to wait for the main worker to be healthy.
                    if (
                        model_instance.distributed_servers.mode
                        == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                    ):
                        continue
                    # Otherwise, update subordinate worker state to RUNNING.
                    sw_pos = next(
                        (
                            i
                            for i, sw in enumerate(
                                model_instance.distributed_servers.subordinate_workers
                            )
                            if sw.worker_id == self._worker_id
                        ),
                    )
                    sw = model_instance.distributed_servers.subordinate_workers[sw_pos]
                    if sw.state == ModelInstanceStateEnum.RUNNING:
                        continue
                    sw.state = ModelInstanceStateEnum.RUNNING
                    sw.state_message = ""
                    patch_dict = {
                        f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                    }
                # Update model instance.
                self._update_model_instance(model_instance.id, **patch_dict)

    @staticmethod
    def _get_main_worker_distributed_state(
        model_instance: ModelInstance,
    ) -> Optional[dict]:
        subordinate_workers = (
            model_instance.distributed_servers.subordinate_workers
            if (
                model_instance.distributed_servers
                and model_instance.distributed_servers.subordinate_workers
            )
            else []
        )

        if not subordinate_workers:
            return None

        error_sw = None
        unreachable_sw = None
        all_running = True
        for sw in subordinate_workers:
            if sw.state == ModelInstanceStateEnum.ERROR:
                error_sw = sw
                break
            if (
                sw.state == ModelInstanceStateEnum.UNREACHABLE
                and unreachable_sw is None
            ):
                unreachable_sw = sw
            if sw.state != ModelInstanceStateEnum.RUNNING:
                all_running = False

        if error_sw:
            return {
                "should_update": model_instance.state != ModelInstanceStateEnum.ERROR,
                "state": ModelInstanceStateEnum.ERROR,
                "state_message": (
                    f"Distributed serving error in subordinate worker "
                    f"{error_sw.worker_ip}: {error_sw.state_message}."
                ),
            }

        if unreachable_sw:
            return {
                "should_update": model_instance.state
                != ModelInstanceStateEnum.UNREACHABLE,
                "state": ModelInstanceStateEnum.UNREACHABLE,
                "state_message": (
                    f"Distributed serving unreachable in subordinate worker "
                    f"{unreachable_sw.worker_ip}: {unreachable_sw.state_message}."
                ),
            }

        if not all_running:
            return {"should_update": False}

        return None

    def _sync_vgpu_allocation(  # noqa: C901
        self,
        model_instance: ModelInstance,
        workload,
        is_main_worker: bool,
    ):
        """
        Patch the instance with the real device allocation read back from the
        workload annotations: GPU UUID(s) into gpu_addresses, the allocated
        card's index into gpu_indexes (resolved via this worker's reported
        gpu_devices), and the claim's vram re-keyed from the placeholder to
        that index so worker_allocated_cache charges the partial card to the
        right index. No-op until the device plugin has allocated; deferred
        (not patched) while the allocated card is missing from the worker's
        reported devices.
        """
        accelerators = _parse_allocated_accelerators(
            getattr(workload, "annotations", None)
        )
        if not accelerators:
            return

        uuids = [a["id"] for a in accelerators]

        # Resolve the patch target first (local lookups only).
        sw_pos = None
        if not is_main_worker:
            if (
                not model_instance.distributed_servers
                or not model_instance.distributed_servers.subordinate_workers
            ):
                return
            sw_pos = next(
                (
                    i
                    for i, sw in enumerate(
                        model_instance.distributed_servers.subordinate_workers
                    )
                    if sw.worker_id == self._worker_id
                ),
                None,
            )
            if sw_pos is None:
                return
        target = (
            model_instance
            if is_main_worker
            else model_instance.distributed_servers.subordinate_workers[sw_pos]
        )

        # Steady-state no-op guard: once addresses, indexes and the re-keyed
        # claim all agree with the annotation, skip the workers API call that
        # index resolution needs — it would otherwise fire every sync cycle
        # for every vGPU instance.
        target_claim = target.computed_resource_claim
        if (
            target.gpu_addresses == uuids
            and target.gpu_indexes
            and target_claim is not None
            and target_claim.vram
            and set(target_claim.vram.keys()) == set(target.gpu_indexes)
        ):
            return

        # Resolve the allocated card's index from this worker's reported
        # devices (the detected set, not the static config, which is empty
        # for auto-detected workers).
        gpu_devices = []
        with contextlib.suppress(Exception):
            worker = self._clientset.workers.get(self._worker_id)
            gpu_devices = (worker.status and worker.status.gpu_devices) or []
        index_by_uuid = {d.uuid: d.index for d in gpu_devices if d.uuid}
        allocated_index = next(
            (index_by_uuid[u] for u in uuids if u in index_by_uuid), None
        )
        if allocated_index is None:
            # The card isn't in the worker's reported devices (yet): patching
            # now would leave the claim charged to a wrong placeholder index.
            # Retry on the next sync cycle.
            logger.debug(
                f"vgpu allocation UUIDs {uuids} not found in worker "
                f"{self._worker_id} reported devices, deferring sync"
            )
            return

        def rekey_claim(claim):
            if claim is None or not claim.vram:
                return None
            if set(claim.vram.keys()) == {allocated_index}:
                return None
            return claim.model_copy(
                update={"vram": {allocated_index: sum(claim.vram.values())}}
            )

        with contextlib.suppress(NotFoundException):
            if is_main_worker:
                patch_dict = {}
                if model_instance.gpu_addresses != uuids:
                    patch_dict["gpu_addresses"] = uuids
                if model_instance.gpu_indexes != [allocated_index]:
                    patch_dict["gpu_indexes"] = [allocated_index]
                new_claim = rekey_claim(model_instance.computed_resource_claim)
                if new_claim is not None:
                    patch_dict["computed_resource_claim"] = new_claim
                if patch_dict:
                    self._update_model_instance(model_instance.id, **patch_dict)
                return

            sw = target
            changed = False
            if sw.gpu_addresses != uuids:
                sw.gpu_addresses = uuids
                changed = True
            if sw.gpu_indexes != [allocated_index]:
                sw.gpu_indexes = [allocated_index]
                changed = True
            new_claim = rekey_claim(sw.computed_resource_claim)
            if new_claim is not None:
                sw.computed_resource_claim = new_claim
                changed = True
            if changed:
                self._update_model_instance(
                    model_instance.id,
                    **{f"distributed_servers.subordinate_workers.{sw_pos}": sw},
                )

    def sync_model_instances_inference_health(self):
        """
        Synchronize model instances' inference health by sending actual inference requests.

        Per-model configuration is read from model.env:
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED: "true"/"false" (default: false)
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL: seconds (default: global env)
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT: seconds (default: 15)
        - GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD: count (default: global env)

        If the model has received successful inference traffic recently
        (within the configured interval), the active health check is skipped.
        """

        # Use the event-driven local cache instead of an API call.
        model_instances = [
            mi
            for mi in self._model_instance_by_instance_id.values()
            if mi.state == ModelInstanceStateEnum.RUNNING
        ]
        if not model_instances:
            return

        now = time.time()

        for model_instance in model_instances:
            model = self._get_model(model_instance)
            if not model:
                continue

            # Read per-model config from model.env.
            config = _get_inference_health_check_config(model)
            if not config["enabled"]:
                continue

            interval = config["interval"]
            timeout = config["timeout"]
            threshold = config["threshold"]

            # Skip if the model is still provisioning.
            if self._is_provisioning(model_instance):
                continue

            # Skip if not enough time has passed since last check.
            last_check = self._last_health_check_time.get(model_instance.id, 0)
            if now - last_check < interval:
                continue

            self._last_health_check_time[model_instance.id] = now

            # Skip if recent successful inference was observed for this instance.
            last_success = self._last_successful_inference.get(model_instance.id, 0)
            if last_success > now - interval:
                logger.debug(
                    f"Model instance {model_instance.name} had recent successful "
                    f"inference, skipping health check."
                )
                # Reset failure count since real traffic is succeeding.
                self._inference_health_check_failures.pop(model_instance.id, None)
                continue

            # Perform inference health check.
            if not is_inference_ready(model_instance, model, timeout=timeout):
                failure_count = self._inference_health_check_failures.get(
                    model_instance.id, 0
                )
                failure_count += 1
                self._inference_health_check_failures[model_instance.id] = failure_count

                if failure_count >= threshold:
                    logger.warning(
                        f"Model instance {model_instance.name} inference health check failed "
                        f"{failure_count} times, updating state to ERROR."
                    )
                    patch_dict = {
                        "state": ModelInstanceStateEnum.ERROR,
                        "state_message": _INFERENCE_HEALTH_CHECK_FAILED_MESSAGE,
                    }
                    self._update_model_instance(model_instance.id, **patch_dict)
                    # Reset failure count after marking as error.
                    del self._inference_health_check_failures[model_instance.id]
                else:
                    logger.debug(
                        f"Model instance {model_instance.name} inference health check failed "
                        f"{failure_count}/{threshold} times."
                    )
            else:
                # Reset failure count on success.
                self._inference_health_check_failures.pop(model_instance.id, None)

    def _handle_model_instance_event(self, event: Event):
        """Handle a model instance event without ever crashing the watch stream.

        The awatch callback runs inline in the watch loop, so any exception
        that escapes here tears the stream down and forces a full reconnect
        plus cache reload. Swallow and log (with traceback) instead; the next
        event or the periodic state sync recovers the instance.
        """
        try:
            self._dispatch_model_instance_event(event)
        except Exception:
            logger.exception(
                f"Failed to handle {event.type} event for model instance "
                f"{getattr(event, 'id', None)}"
            )

    def _dispatch_model_instance_event(self, event: Event):  # noqa: C901
        """
        Handle model instance events.

        Args:
            event: The model instance event to handle.

        """
        mi = ModelInstance.model_validate(event.data)

        logger.trace(
            f"Received event: {str(event.type)}, id: {mi.id}, name: {mi.name}, state: {str(mi.state)}"
        )

        is_main_worker = mi.worker_id == self._worker_id

        if is_main_worker:
            self._model_instance_by_instance_id[mi.id] = mi
            # Return if all subordinate workers aren't running.
            if (
                mi.distributed_servers
                and mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.RUN_FIRST
                and mi.distributed_servers.subordinate_workers
            ):
                ready = all(
                    sw.state == ModelInstanceStateEnum.RUNNING
                    for sw in mi.distributed_servers.subordinate_workers
                )
                if not ready:
                    logger.info(
                        f"Model instance {mi.name} waits for all subordinate workers to be ready."
                    )
                    return
        else:
            # Return if it isn't a distribution serving.
            if not mi.distributed_servers:
                return
            # Return if it's a delegated distribution,
            # which means the main worker is responsible for serving.
            if (
                mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.DELEGATED
            ):
                return
            # Return if it isn't the member of the distribution serving.
            joined = any(
                sw.worker_id == self._worker_id
                for sw in mi.distributed_servers.subordinate_workers or []
            )
            if not joined:
                return
            # Return if the main worker isn't initialized.
            if (
                mi.distributed_servers.mode
                == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                and (
                    mi.state
                    not in [
                        ModelInstanceStateEnum.STARTING,
                        ModelInstanceStateEnum.RUNNING,
                        ModelInstanceStateEnum.ERROR,
                    ]
                )
            ):
                logger.info(
                    f"Model instance {mi.name} waits for main worker {mi.worker_ip} to be initialized."
                )
                return
            # FIXME: This is a temporary solution to prevent the main worker from being unable to start due to phantom reads.
            #        We confirm whether the operation should be performed by checking the state of the earlier subordinate worker.
            for sw in mi.distributed_servers.subordinate_workers:
                if sw.worker_id == self._worker_id:
                    break
                if sw.state not in [
                    ModelInstanceStateEnum.RUNNING,
                    ModelInstanceStateEnum.ERROR,
                ]:
                    logger.info(
                        f"Model instance {mi.name} waits for previous subordinate worker {sw.worker_ip} to be ready."
                    )
                    return

        if event.type == EventType.DELETED:
            # Teardown is left to the periodic reap in sync_model_instances_state,
            # which is the authoritative reconciler and must run anyway to catch
            # DELETEDs missed during a watch disconnect. Tearing down here too
            # would give a second concurrent caller racing the reap on
            # delete_workload, so just let the reap reap it (within one tick).
            logger.trace(
                f"DELETED event for model instance {mi.name}; "
                "teardown deferred to reap."
            )
            return

        if event.type == EventType.UPDATED:
            # Caching matched ERROR instances for restart handling.
            if mi.state == ModelInstanceStateEnum.ERROR:
                model = self._get_model(mi)
                if model.restart_on_error:
                    self._error_model_instances[mi.id] = mi
                    logger.trace(
                        f"UPDATED event: cached error model instance {mi.name} for restart."
                    )
                return

            # Restart if scheduled and this is the assigned worker.
            if is_main_worker and mi.state == ModelInstanceStateEnum.SCHEDULED:
                self._restart_model_instance(mi)
                logger.trace(
                    f"UPDATED event: restarted scheduled model instance {mi.name}."
                )

            # Start on subordinate worker if not started yet, or restart if failed.
            if not is_main_worker:
                deployment_metadata = mi.get_deployment_metadata(self._worker_id)
                workload_name = (
                    deployment_metadata.name if deployment_metadata else mi.name
                )
                workload = get_workload(workload_name)
                if not workload:
                    self._start_model_instance(mi)
                    logger.trace(
                        f"UPDATED event: started model instance {mi.name} on subordinate worker."
                    )
                elif classify_workload(workload) in (
                    WorkloadPhase.EXITED,
                    WorkloadPhase.FAILED,
                ):
                    self._stop_model_instance(mi, clear_restart_backoff=False)
                    self._start_model_instance(mi)
                    logger.trace(
                        f"UPDATED event: restarted failed model instance {mi.name} on subordinate worker."
                    )

            return

        if event.type == EventType.CREATED:
            # Only handle CREATED if this is the assigned worker
            if not is_main_worker:
                return
            if mi.state == ModelInstanceStateEnum.RUNNING:
                logger.warning(
                    f"Model instance {mi.name} is already running. Skipping start."
                )
                return
            self._start_model_instance(mi)
            logger.trace(f"CREATED event: started created model instance {mi.name}.")

    def _get_numbered_log_path(self, mi: ModelInstance) -> str:
        """Get log file path with restart count.

        Args:
            mi: The model instance.

        Returns:
            Log file path with format: {log_dir}/{model_instance_id}.{restart_count}.log
        """
        restart_count = mi.restart_count or 0
        return f"{self._serve_log_dir}/{mi.id}.{restart_count}.log"

    def _start_container_log_persistence(self, mi: ModelInstance):
        """Persist this instance's container logs, sidecars included.

        Args:
            mi: The model instance.
        """
        # The workload name differs for subordinate workers (e.g. "model-f0").
        deployment_metadata = mi.get_deployment_metadata(self._worker_id)
        workload_name = deployment_metadata.name if deployment_metadata else mi.name
        restart_count = mi.restart_count or 0

        self._container_logs.start(
            mi.id,
            workload_name,
            f"{self._serve_log_dir}/{mi.id}.container.{restart_count}.log",
            sidecar_log_path=lambda name: (
                f"{self._serve_log_dir}/{mi.id}.container.{name}.{restart_count}.log"
            ),
        )

    def _stop_container_log_persistence(
        self, model_instance_id: int, timeout: float = 2.0
    ):
        self._container_logs.stop(model_instance_id, timeout=timeout)

    def _cleanup_old_logs(self, model_instance_id: int, current_restart_count: int):
        """Keep serve logs for restart_count in {R, R-1}.

        R==0 is a fresh lifecycle start, so all existing files are purged (any
        present belong to a previous owner of a reused id).
        """
        if current_restart_count == 0:
            self._purge_instance_logs(model_instance_id)
            return

        try:
            log_dir = Path(self._serve_log_dir)

            # Separate main logs, container logs, and sidecar container logs
            main_log_pattern = f"{model_instance_id}.*.log"
            all_main_logs = [
                f for f in log_dir.glob(main_log_pattern) if '.container.' not in f.name
            ]

            container_log_pattern = f"{model_instance_id}.container.*.log"
            all_container_files = list(log_dir.glob(container_log_pattern))

            # Split into default container logs (e.g., 42.container.0.log)
            # and sidecar container logs (e.g., 42.container.ray-head.0.log)
            default_container_logs = [
                f
                for f in all_container_files
                if extract_container_restart_count(f.name) > 0
                or re.match(rf'{model_instance_id}\.container\.\d+\.log', f.name)
            ]
            sidecar_container_logs = [
                f for f in all_container_files if f not in default_container_logs
            ]

            self._cleanup_log_type(all_main_logs, current_restart_count, "main")
            self._cleanup_log_type(
                default_container_logs, current_restart_count, "container"
            )
            self._cleanup_log_type(
                sidecar_container_logs, current_restart_count, "sidecar_container"
            )

        except Exception as e:
            logger.error(f"Failed to cleanup old logs for {model_instance_id}: {e}")

    def _cleanup_log_type(
        self,
        log_files: List[Path],
        current_restart_count: int,
        log_type: str,
    ):
        """Delete log files whose restart_count is not current or previous."""

        keep = {current_restart_count}
        if current_restart_count > 0:
            keep.add(current_restart_count - 1)

        def _extract_sidecar_restart_count(filename: str) -> int:
            """Extract restart count from {id}.container.{name}.{restart_count}.log"""
            match = re.match(r'\d+\.container\.[^.]+\.(\d+)\.log', filename)
            return int(match.group(1)) if match else 0

        extract_fns = {
            "main": extract_restart_count,
            "container": extract_container_restart_count,
            "sidecar_container": _extract_sidecar_restart_count,
        }
        extract_fn = extract_fns.get(log_type, extract_container_restart_count)

        for f in log_files:
            rc = extract_fn(f.name)
            if rc in keep:
                continue
            try:
                f.unlink()
                logger.info(f"Deleted old {log_type} log file: {f}")
            except Exception as e:
                logger.warning(f"Failed to delete {log_type} log file {f}: {e}")

    def _purge_instance_logs(self, model_instance_id: int):
        """Delete all serve logs (main/container/sidecar) for a model instance id."""
        try:
            log_dir = Path(self._serve_log_dir)
            for f in log_dir.glob(f"{model_instance_id}.*.log"):
                try:
                    f.unlink()
                    logger.info(f"Deleted serve log file: {f}")
                except Exception as e:
                    logger.warning(f"Failed to delete serve log file {f}: {e}")
        except Exception as e:
            logger.error(
                f"Failed to purge logs for model instance {model_instance_id}: {e}"
            )

    def _start_model_instance(self, mi: ModelInstance):  # noqa: C901
        """
        Start model instance through a subprocess.

        Args:
            mi: The model instance to start.

        """
        if self._is_provisioning(mi):
            logger.warning(f"Model instance {mi.name} is provisioning. Skipping start.")
            return

        # Clean up old log files before starting
        self._cleanup_old_logs(mi.id, mi.restart_count or 0)

        is_main_worker = mi.worker_id == self._worker_id

        log_file_path = self._get_numbered_log_path(mi)

        sw_pos: Optional[int] = None
        sw: Optional[ModelInstanceSubordinateWorker] = None
        if not is_main_worker:
            sw_pos = next(
                (
                    i
                    for i, sw in enumerate(mi.distributed_servers.subordinate_workers)
                    if sw.worker_id == self._worker_id
                ),
            )
            sw = mi.distributed_servers.subordinate_workers[sw_pos]

        try:
            model = self._get_model(mi)
            backend = get_backend(model)

            self._assign_ports(mi, model, backend)

            logger.debug(
                f"Starting model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

            fallback_registry = (
                registration.determine_default_registry(
                    self._config.system_default_container_registry
                )
                if is_built_in_backend(backend)
                else None
            )

            process = self._provisioning.start(
                mi.id,
                description=f"model instance {mi.name}",
                proctitle=f"gpustack_model_instance_{mi.id}",
                log_path=log_file_path,
                provision=partial(
                    provision_model_instance,
                    mi,
                    backend,
                    self._worker_id,
                    self._inference_backend_manager.get_backend_by_name(
                        backend, model.owner_principal_id
                    ),
                    fallback_registry,
                ),
            )

            # Start container log persistence for containerized backends
            self._start_container_log_persistence(mi)

            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.INITIALIZING,
                    "port": mi.port,
                    "ports": mi.ports,
                    "pid": process.pid,
                }
            # Get patch dict for subordinate worker.
            else:
                sw.state = ModelInstanceStateEnum.INITIALIZING
                # For initialize later mode, the state is set to RUNNING directly,
                # which means the subordinate worker doesn't need to wait for the main worker to be healthy.
                if (
                    mi.distributed_servers.mode
                    == DistributedServerCoordinateModeEnum.INITIALIZE_LATER
                ):
                    sw.state = ModelInstanceStateEnum.RUNNING
                sw.pid = process.pid
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.info(
                f"Started model instance {mi.name}"
                f"{'' if not is_main_worker else f' on ports {mi.ports if mi.ports else [mi.port]}'}"
            )

        except Exception as e:
            # Clean up provisioning process if started.
            if self._provisioning.is_running(mi.id):
                self._stop_model_instance(mi)

            # Get patch dict for main worker.
            if is_main_worker:
                patch_dict = {
                    "state": ModelInstanceStateEnum.ERROR,
                    "state_message": f"Failed to start model instance: {e}",
                }
            # Get patch dict for subordinate worker.
            else:
                sw.state = ModelInstanceStateEnum.ERROR
                sw.state_message = f"Failed to start model instance: {e}"
                patch_dict = {
                    f"distributed_servers.subordinate_workers.{sw_pos}": sw,
                }

            self._update_model_instance(mi.id, **patch_dict)
            logger.error(f"Failed to start model instance {mi.name}: {e}")

    def _assign_ports(
        self,
        mi: ModelInstance,
        model: Model,
        backend: BackendEnum,
    ) -> None:
        """
        Assign ports to the model instance.

        This method is thread-safe and allocates ports for:
        - Main serving port
        - RPC port for vLLM DP communication (if applicable)
        - Connecting port for subordinate workers (if applicable)

        Args:
            mi: The model instance to assign ports to.
            model: The model associated with the instance.
            backend: The backend type (e.g., vLLM, SGLang).
        """
        if mi.port:
            # Port already assigned, skip.
            return

        with _port_lock:
            if mi.port:
                # Port already assigned, skip.
                return

            if self._assigned_ports:
                unavailable_ports = set.union(*self._assigned_ports.values())
            else:
                unavailable_ports = set()

            # Main serving port
            mi.port = network.get_free_port(
                port_range=self._config.service_port_range,
                unavailable_ports=unavailable_ports,
                host=mi.worker_ip,
            )
            mi.ports = [mi.port]
            unavailable_ports.add(mi.port)

            # Additional ports for distributed servers (mp path allocates all):
            #   ports[0]: HTTP API (always)
            #   ports[1]: --data-parallel-rpc-port (DP coordinator ZMQ)
            #   ports[2]: --master-port (PyTorch distributed TCP store)
            #   ports[3]: env VLLM_PORT (dp_only only; reserved but unused otherwise)
            #   ports[-1]: connecting port (= VLLM_DP_MASTER_PORT for dp_only/nested)
            # Ray path: only ports[1] (DP RPC), when user dp > 1.
            if mi.distributed_servers and mi.distributed_servers.subordinate_workers:
                # Allocate first so we can fence off the 10-port band vLLM reserves
                # around VLLM_DP_MASTER_PORT (= connecting port), keeping the cross
                # ports (incl. VLLM_PORT) outside it.
                connecting_port = network.get_free_port(
                    port_range=self._config.service_port_range,
                    unavailable_ports=unavailable_ports,
                    host=mi.worker_ip,
                )
                unavailable_ports.add(connecting_port)

                cross_ports: List[int] = []
                if backend == BackendEnum.VLLM:
                    executor_backend = resolve_executor_backend(
                        model.backend_parameters, model.backend_version
                    )
                    if executor_backend == "mp":
                        # DP RPC + PyTorch master + VLLM_PORT. Clamp the band to
                        # service_port_range; out-of-range ports would inflate
                        # get_free_port's exhaustion count.
                        _, end_port = network.parse_port_range(
                            self._config.service_port_range
                        )
                        unavailable_ports |= set(
                            range(
                                connecting_port, min(connecting_port + 10, end_port + 1)
                            )
                        )
                        cross_port_count = 3
                    else:
                        dps = find_int_parameter(
                            model.backend_parameters,
                            ["data-parallel-size", "dp"],
                        )
                        cross_port_count = 1 if dps and dps > 1 else 0
                    for _ in range(cross_port_count):
                        cross_port = network.get_free_port(
                            port_range=self._config.service_port_range,
                            unavailable_ports=unavailable_ports,
                            host=mi.worker_ip,
                        )
                        cross_ports.append(cross_port)
                        unavailable_ports.add(cross_port)

                mi.ports.extend(cross_ports)
                mi.ports.append(connecting_port)

            self._assigned_ports[mi.id] = set(mi.ports)

    def _restart_model_instance(self, mi: ModelInstance):
        """
        Restart model instance.

        Args:
            mi: The model instance to restart.
        """

        self._stop_model_instance(mi, clear_restart_backoff=False)
        self._start_model_instance(mi)

    def _update_model(self, id: int, **kwargs) -> bool:
        """
        Update model with given fields.

        Args:
            id: The ID of the model to update.
            **kwargs: The fields to update, group by field name and value.
        """

        return update_resource(
            self._clientset.models, id, ModelUpdate, "Model", **kwargs
        )

    def _update_model_instance(self, id: int, **kwargs) -> bool:
        """
        Update model instance with given fields.

        Args:
            id: The ID of the model instance to update.
            **kwargs: The fields to update, group by field name and value.
        """

        return update_resource(
            self._clientset.model_instances,
            id,
            ModelInstanceUpdate,
            "Model instance",
            **kwargs,
        )

    def _stop_model_instance(
        self,
        mi: ModelInstance,
        clear_restart_backoff: bool = True,
        delete_logs: bool = False,
    ):
        """
        Stop model instance and clean up.

        Args:
            mi: The model instance to stop.
            clear_restart_backoff: Whether to clear transient restart backoff state.
            delete_logs: Whether to remove the instance's serve log files. Only set
                on permanent teardown (DELETED); a restart must keep them so the
                log viewer can still show the previous run.
        """

        logger.debug(f"Stopping model instance {mi.name or mi.id}")

        # Stop container log persistence thread
        self._stop_container_log_persistence(mi.id)

        if delete_logs:
            self._purge_instance_logs(mi.id)

        # Teardown provisioning process if still alive.
        self._provisioning.terminate(mi.id)

        # Delete workload.
        deployment_metadata = mi.get_deployment_metadata(self._worker_id)
        if deployment_metadata:
            delete_workload(deployment_metadata.name)

        # Cleanup internal states.
        self._provisioning.forget(mi.id)
        self._assigned_ports.pop(mi.id, None)
        self._error_model_instances.pop(mi.id, None)
        self._model_cache_by_instance.pop(mi.id, None)
        self._model_instance_by_instance_id.pop(mi.id, None)
        if clear_restart_backoff:
            self._restart_backoff_counts.pop(mi.id, None)
        self._inference_health_check_failures.pop(mi.id, None)
        self._last_health_check_time.pop(mi.id, None)
        self._last_successful_inference.pop(mi.id, None)

        logger.info(f"Stopped model instance {mi.name or mi.id}")

    def _restart_error_model_instance(self, mi: ModelInstance):
        """
        Restart error model instance with exponential backoff,
        maximum delay 5 minutes.

        Args:
            mi: The model instance to restart.
        """
        if self._is_provisioning(mi):
            logger.debug(f"Model instance {mi.name} is provisioning. Skipping restart.")
            return

        restart_count = mi.restart_count or 0
        backoff_count = self._restart_backoff_counts.get(mi.id, 0)
        last_restart_time = mi.last_restart_time or mi.updated_at

        current_time = datetime.now(timezone.utc)
        decision = RESTART_BUDGET.decide(backoff_count, last_restart_time, current_time)
        if decision.action == RestartActionEnum.WAIT:
            logger.trace(
                f"Delaying restart of {mi.name} for "
                f"{decision.delay_remaining_seconds:.2f} seconds."
            )
            return

        logger.info(
            f"Restarting model instance {mi.name} "
            f"(attempt {decision.attempt}) after "
            f"{RESTART_BUDGET.delay_for(backoff_count)} seconds delay."
        )

        with contextlib.suppress(NotFoundException):
            self._restart_backoff_counts[mi.id] = decision.attempt
            self._update_model_instance(
                mi.id,
                restart_count=restart_count + 1,
                last_restart_time=current_time,
                state=ModelInstanceStateEnum.SCHEDULED,
                state_message="",
            )

        # Pop from error model instances,
        # if failed to restart next time, it will be added again in watch_model_instance_events().
        self._error_model_instances.pop(mi.id, None)

    def _get_model(self, mi: ModelInstance) -> Model:
        """
        Efficiently get model related to the model instance with caching.

        Args:
            mi: The model instance whose model to get.
        """
        if model := self._model_cache_by_instance.get(mi.id):
            return model

        model = self._clientset.models.get(mi.model_id)
        self._model_cache_by_instance[mi.id] = model
        return model

    def _refresh_model(self, mi: ModelInstance) -> Model:
        """
        Refresh the model information from the server.

        Args:
            mi: The model instance whose model to refresh.

        Returns:
            The refreshed model.
        """
        logger.debug(f"Refreshing model {mi.model_name} information from server.")
        refreshed_model = self._clientset.models.get(mi.model_id)
        self._model_cache_by_instance[mi.id] = refreshed_model
        return refreshed_model

    def _is_provisioning(self, mi: ModelInstance) -> bool:
        """
        Check if the model instance is still provisioning.

        Args:
            mi: The model instance to check.
        """
        return self._provisioning.is_running(mi.id)

    def _get_health_check_path(
        self, backend: str, owner_principal_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Get health check path for the given backend.

        Args:
            backend: The backend name.
            owner_principal_id: Owner of the model being served, used to
                resolve an Org-scoped backend row over the Platform one.
        Returns:
            The health check path if exists, else None.
        """
        inference_backend = self._inference_backend_manager.get_backend_by_name(
            backend, owner_principal_id
        )

        return inference_backend.health_check_path if inference_backend else None

    def get_instance_port_by_model_instance_id(
        self, model_instance_id: int
    ) -> Optional[int]:
        """
        Get the port of the model instance related to the given model instance ID.

        Args:
            model_instance_id: The model instance ID to get the port for.

        Returns:
            The port of the model instance if it exists and is running, else None.
        """
        instance = self._model_instance_by_instance_id.get(
            model_instance_id
        )  # Ensure the model instance is cached.
        return (
            instance.ports[0]
            if instance and instance.state == ModelInstanceStateEnum.RUNNING
            else None
        )


def is_ready(
    backend: str,
    mi: ModelInstance,
    health_check_path: Optional[str] = None,
    model: Model = None,
) -> bool:
    """
    Access the health endpoint of the given model instance to check if it is servable.
    """
    is_built_in = is_built_in_backend(backend)
    if (not is_built_in or backend == BackendEnum.CUSTOM) and (not health_check_path):
        # If custom backend does not have health check path, consider it always ready.
        return True

    if backend == BackendEnum.ASCEND_MINDIE and not health_check_path:
        # Ref: https://www.hiascend.com/document/detail/zh/mindie/21RC2/mindieservice/servicedev/mindie_service0066.html
        # /info provides metadata information and requires more time to respond. Use it for health check.
        health_check_path = "/info"
    elif (
        backend == BackendEnum.SGLANG
        and model
        and CategoryEnum.IMAGE in model.categories
    ):
        if not model.backend_version:
            # version may be empty at initialization, consider it not ready.
            return False
        elif compare_versions(model.backend_version, "0.5.5.post3") >= 0:
            # SGLang Diffusion supported health check path at v0.5.5.post3
            health_check_path = "/health"
        else:
            # Older versions do not support health check, consider it always ready.
            return True
    elif is_built_in and backend != BackendEnum.CUSTOM and not health_check_path:
        # Built-in backends (vLLM, SGLang, vox-box) except (Custom, MindIE) use /v1/models as health check path.
        health_check_path = "/v1/models"

    try:
        # Use the worker IP instead of localhost for health check.
        # Reasons:
        # 1. Connectivity to the loopback address does not work with Ascend MindIE.
        # 2. More adaptable to container networks.
        health_check_url = f"http://{mi.worker_ip}:{mi.port}{health_check_path}"
        response = requests.get(health_check_url, timeout=1)
        if response.status_code == 200:
            return True
    except Exception as e:
        logger.debug(f"Error checking model instance {mi.name} health: {e}")
        pass
    return False


def _get_inference_endpoint_and_payload(model: Model) -> tuple[str, dict] | None:
    """
    Get inference endpoint and payload for the model.
    Returns None if the model type should skip health check.
    """
    skip_categories = {
        CategoryEnum.IMAGE,
        CategoryEnum.SPEECH_TO_TEXT,
        CategoryEnum.TEXT_TO_SPEECH,
        CategoryEnum.UNKNOWN,
    }
    if not skip_categories.isdisjoint(model.categories):
        return None

    # Return endpoint and payload based on model type (priority order)
    if CategoryEnum.EMBEDDING in model.categories:
        return "/v1/embeddings", {"model": model.name, "input": "test"}

    if CategoryEnum.RERANKER in model.categories:
        return "/v1/rerank", {
            "model": model.name,
            "query": "test",
            "documents": ["test"],
        }

    return "/v1/chat/completions", {
        "model": model.name,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "max_completion_tokens": 1,
    }


def _get_inference_health_check_config(model: Model) -> dict:
    """Read per-model inference health check config from model.env."""
    env = model.env or {}
    enabled = env.get(
        "GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED", "false"
    ).lower() in (
        "true",
        "1",
    )
    interval = safe_int(
        env.get("GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL"),
        300,
    )
    timeout = safe_int(
        env.get("GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT"),
        15,
    )
    threshold = safe_int(
        env.get("GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD"),
        3,
    )
    return {
        "enabled": enabled,
        "interval": interval,
        "timeout": timeout,
        "threshold": threshold,
    }


def is_inference_ready(mi: ModelInstance, model: Model, timeout: int = 15) -> bool:
    """
    Send a minimal inference request to verify the inference capability is working.
    """
    # Check Custom backend (no standard inference API)
    if is_custom_backend(model.backend):
        return True

    # Check port assignment
    if not mi.port:
        logger.debug(f"Model instance {mi.name} does not have port assigned yet.")
        return False

    # Get endpoint and payload, None means skip health check
    result = _get_inference_endpoint_and_payload(model)
    if not result:
        logger.debug(f"Skipping inference check for {mi.name}")
        return True

    endpoint_path, payload = result
    inference_url = f"http://{mi.worker_ip}:{mi.port}{endpoint_path}"

    try:
        response = requests.post(inference_url, json=payload, timeout=timeout)
        if response.status_code == 200:
            return True
        else:
            logger.warning(
                f"Model instance {mi.name} inference health check failed "
                f"with status {response.status_code} for endpoint {endpoint_path}"
            )
    except Exception as e:
        logger.debug(
            f"Error checking model instance {mi.name} inference at {endpoint_path}: {e}"
        )

    return False
