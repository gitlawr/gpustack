"""
Provisioning of a managed cache service instance.

The manager runs this in a subprocess whose stdout/stderr are redirected to a
per-instance log file, the same shape as the model instance (``InferenceServer``)
and benchmark (``BenchmarkRunner``) provisioners. The slow part of a start —
resolving the provider declaration and pulling the container image — therefore
leaves a readable trace while it happens, instead of only through its outcome.

Ports are allocated by the manager and passed in: they are worker-wide state
that the parent process owns.
"""

import logging
import shlex
from typing import Any, Dict, List, Optional, Tuple

from gpustack_runtime.deployer import (
    Container,
    ContainerEnv,
    ContainerExecution,
    ContainerProfileEnum,
    ContainerResources,
    WorkloadPlan,
    create_workload,
    delete_workload,
)
from gpustack_runtime.detector import detect_backend, detect_devices
from gpustack_runtime.envs import (
    GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY,
    to_bool,
)

from gpustack import envs
from gpustack.api.exceptions import NotFoundException
from gpustack.client import ClientSet
from gpustack.config import registration
from gpustack.config.config import Config
from gpustack.schemas.cache_providers import (
    CUSTOM_VERSION,
    CacheProvider,
    CacheProviderVersionConfig,
    render_l2_adapter,
    render_template,
)
from gpustack.schemas.cache_services import (
    CacheServiceInstance,
    CacheServicePublic,
    CacheServiceStateEnum,
)
from gpustack.server.cache_provider_catalog import get_cache_provider
from gpustack.utils.command import (
    drop_empty_flag_values,
    extract_flag_arguments,
    merge_flag_arguments,
)
from gpustack.utils.config import apply_registry_override_to_image
from gpustack.utils.runtime import transform_workload_plan
from gpustack.worker.cache_service.state import update_cache_service_instance

logger = logging.getLogger(__name__)


class CacheServiceProvisioner:
    """
    Compiles a cache service instance into a workload plan and creates it.

    Everything here is derived from the instance's parent cache service, the
    provider declaration, and this worker's own facts (accelerator backend and
    runtime version). Nothing in this class touches the manager's in-process
    bookkeeping, so it is safe to run in a subprocess.
    """

    def __init__(
        self,
        clientset: ClientSet,
        instance: CacheServiceInstance,
        cfg: Config,
        port: int,
        metrics_port: int,
        fallback_registry: Optional[str] = None,
    ):
        self._clientset = clientset
        self._instance = instance
        self._config = cfg
        self._port = port
        self._metrics_port = metrics_port
        self._fallback_registry = fallback_registry

    def start(self):
        """
        Start the managed cache server container for the instance.

        Reports failure by parking the instance in ERROR rather than raising:
        the caller is a subprocess entry point whose exception nothing would
        surface to the user.
        """
        instance = self._instance
        try:
            self._start()
        except Exception as e:
            update_cache_service_instance(
                self._clientset,
                instance.id,
                state=CacheServiceStateEnum.ERROR,
                state_message=str(e),
            )
            logger.error(
                f"Failed to start cache service instance {instance.id} "
                f"(service id={instance.cache_service_id}): {e}"
            )

    def _start(self):
        instance = self._instance
        try:
            cache_service = self._clientset.cache_services.get(
                id=instance.cache_service_id
            )
        except NotFoundException:
            update_cache_service_instance(
                self._clientset,
                instance.id,
                state=CacheServiceStateEnum.ERROR,
                state_message=(
                    f"Parent cache service {instance.cache_service_id} not found."
                ),
            )
            return

        provider = get_cache_provider(cache_service.provider_name)
        if provider is None:
            update_cache_service_instance(
                self._clientset,
                instance.id,
                state=CacheServiceStateEnum.ERROR,
                state_message=f"Unknown cache provider: {cache_service.provider_name}",
            )
            return

        workload_plan, image = self._compile(cache_service, provider)

        # Starting is idempotent: a stale workload left over from a previous run
        # of this instance (crash, manual restart) is removed first, so restart
        # and first start share this code path.
        try:
            delete_workload(workload_plan.name)
        except Exception as e:
            # The workload may not exist yet.
            logger.debug(
                f"Skipped deleting workload {workload_plan.name} before start: {e}"
            )

        logger.info(
            f"Creating cache service workload {workload_plan.name} "
            f"with image {image} on port {self._port}"
        )
        create_workload(
            transform_workload_plan(
                self._config, workload_plan, self._fallback_registry
            )
        )

        if update_cache_service_instance(
            self._clientset,
            instance.id,
            state=CacheServiceStateEnum.STARTING,
            port=self._port,
            metrics_port=self._metrics_port,
            state_message="",
        ):
            logger.info(
                f"Started cache service {cache_service.name} instance "
                f"(id={instance.id}) on port {self._port}"
            )
        else:
            # The container is up but the server still sees the instance as
            # PENDING; the sync pass re-drives the start rather than leaving a
            # running cache server nothing points at.
            logger.error(
                f"Started cache service workload {workload_plan.name} "
                f"but failed to mark instance {instance.id} as starting"
            )

    def _compile(
        self,
        cache_service: CacheServicePublic,
        provider: CacheProvider,
    ) -> Tuple[WorkloadPlan, str]:
        """
        Render the instance into the workload plan it runs as, plus the
        resolved image. Pure with respect to the API server: the only inputs
        are the arguments and this worker's detected accelerator.
        """
        version_config, resolved_version, source_image = self._resolve_version_config(
            cache_service, provider
        )
        params = self._build_template_params(
            cache_service, provider, self._port, self._metrics_port
        )

        # A declared run command is the whole argument vector and takes the
        # image's ENTRYPOINT slot; run_args instead keeps the image's own
        # entrypoint and rides as the CMD arguments appended to it (container
        # semantics: args alone append, a command replaces). The user
        # parameters and L2 flags below join whichever vector the version
        # declared.
        overrides_entrypoint = bool(version_config.run_command)
        launch_template = version_config.run_command or version_config.run_args
        argv: Optional[List[str]] = None
        if launch_template:
            # Render per token so an optional placeholder resolving to None
            # yields an empty token that is dropped together with the flag it
            # belongs to.
            rendered_tokens = [
                render_template(token, params) for token in shlex.split(launch_template)
            ]
            argv = drop_empty_flag_values(rendered_tokens)
        user_parameters = (
            cache_service.config.parameters if cache_service.config else None
        )
        if user_parameters:
            argv = (
                merge_flag_arguments(argv, user_parameters)
                if argv
                else list(user_parameters)
            )

        # L2 storage config renders after the user-parameters merge so the
        # structured config always wins over a hand-written flag.
        argv, l2_env = self._apply_l2_storage(cache_service, provider, argv)

        # Provider env templates render first; entries rendering empty are
        # dropped so unset optional parameters don't produce invalid config.
        # Service-level env overrides provider defaults, and the L2 storage
        # credentials override both.
        env: Dict[str, str] = {}
        for key, value in (version_config.env or {}).items():
            rendered = render_template(value, params)
            if rendered:
                env[key] = rendered
        if cache_service.config and cache_service.config.env:
            env.update(cache_service.config.env)
        env.update(l2_env)

        image = apply_registry_override_to_image(
            self._config, source_image, self._fallback_registry
        )
        if not image:
            raise ValueError(
                f"Failed to resolve image for cache provider "
                f"{cache_service.provider_name} version {resolved_version}"
            )

        run_container = Container(
            image=image,
            name="default",
            profile=ContainerProfileEnum.RUN,
            execution=ContainerExecution(
                privileged=False,
                command=argv if overrides_entrypoint else None,
                args=None if overrides_entrypoint else argv,
            ),
            envs=[ContainerEnv(name=name, value=value) for name, value in env.items()],
            resources=self._gpu_resources(),
        )
        deployment_metadata = self._instance.get_deployment_metadata()
        workload_plan = WorkloadPlan(
            name=deployment_metadata.name,
            host_network=True,
            # Shares the host IPC namespace with the engine containers so the
            # cache server can import their KV buffers by CUDA IPC handle (the
            # lmcache_driven zero-copy path). Same escape hatch as the engine
            # side: service env, then the worker-global GPUSTACK_HOST_IPC,
            # overrides the default — e.g. Kubernetes PodSecurity baseline
            # rejects hostIPC pods, and the CPU host-copy path works without it.
            host_ipc=self._host_ipc_enabled(cache_service),
            containers=[run_container],
            labels=deployment_metadata.labels,
        )
        return workload_plan, image

    def _gpu_resources(self) -> ContainerResources:
        """
        Expose every local GPU to the cache server so the CUDA-IPC transfer
        path (LMCache's lmcache_driven mode) can map the KV buffers of the
        co-located engines: importing an IPC handle needs a CUDA context on
        the same device, and a per-node server attaches to engines on any of
        the node's GPUs. Empty on a worker with no detected accelerator, so
        the server stays CPU-only there (auto mode falls back to a host-copy
        transfer).
        """
        resources = ContainerResources()
        backend = detect_backend()
        if isinstance(backend, str) and backend:
            key = GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY.get(backend)
            if key:
                resources[key] = "all"
        return resources

    def _resolve_version_config(
        self,
        cache_service: CacheServicePublic,
        provider: CacheProvider,
    ) -> Tuple[CacheProviderVersionConfig, Optional[str], str]:
        """
        Resolve the (version config, version identifier, container image)
        the instance runs with. The reserved "custom" version keeps the
        default version's run command and env templates but takes the
        image from the service config, so the image must be
        command-compatible with the default declaration. Raises ValueError
        when the catalog or the service config cannot serve the request.
        """
        if cache_service.provider_version == CUSTOM_VERSION:
            if not provider.custom_version:
                raise ValueError(
                    f"Cache provider {cache_service.provider_name} does not "
                    f"allow the custom version"
                )
            version_config, _ = provider.get_version_config(None)
            if version_config is None:
                raise ValueError(
                    f"Cache provider {cache_service.provider_name} has no "
                    f"default version to template the custom version"
                )
            image = cache_service.config.image if cache_service.config else None
            if not image:
                raise ValueError(
                    f"config.image is required when provider_version is "
                    f"'{CUSTOM_VERSION}'"
                )
            return version_config, CUSTOM_VERSION, image

        version_config, resolved_version = provider.get_version_config(
            cache_service.provider_version
        )
        if version_config is None:
            raise ValueError(
                f"Unknown version '{resolved_version}' for cache provider "
                f"{cache_service.provider_name}"
            )
        backend, runtime_version = self._detect_runtime()
        # Fail fast on an unsupported accelerator: falling back to the
        # plain image (built for another accelerator family) would only
        # crash-loop the container without ever naming the real cause.
        if not version_config.supports_runtime(backend):
            raise ValueError(
                f"Cache provider {cache_service.provider_name} "
                f"({resolved_version}) has no image for {backend} workers; "
                f"scope the service to supported workers via the worker "
                f"selector"
            )
        return (
            version_config,
            resolved_version,
            version_config.resolve_image(backend, runtime_version),
        )

    def _detect_runtime(self) -> Tuple[Optional[str], Optional[str]]:
        """
        This node's (accelerator backend, runtime version), e.g.
        ("cuda", "13.0") — the key into a version's runtime_images.
        (None, None) on accelerator-less workers, where the plain image
        serves.
        """
        backend = detect_backend()
        if not (isinstance(backend, str) and backend):
            return None, None
        version = None
        try:
            version = next(
                (
                    device.runtime_version
                    for device in detect_devices()
                    if device.runtime_version
                ),
                None,
            )
        except Exception as e:
            logger.warning(f"Failed to detect accelerator runtime version: {e}")
        return backend, version

    @staticmethod
    def _host_ipc_enabled(cache_service: CacheServicePublic) -> bool:
        """Host IPC defaults on for cache servers (the CUDA-IPC transfer
        path needs it) but stays overridable: the service's env, then the
        worker-global GPUSTACK_HOST_IPC, wins over the default."""
        service_env = (cache_service.config.env if cache_service.config else None) or {}
        if envs.HOST_IPC_ENV in service_env:
            return to_bool(service_env[envs.HOST_IPC_ENV])
        if envs.HOST_IPC is not None:
            return to_bool(envs.HOST_IPC)
        return True

    @staticmethod
    def _build_template_params(
        cache_service: CacheServicePublic,
        provider: CacheProvider,
        port: int,
        metrics_port: int,
    ) -> Dict[str, Any]:
        """
        Build the placeholder namespace the version templates render
        against: the reserved platform keys, extended by the provider's
        declared fields carrying configured values (falling back to
        declared defaults). The platform keys win over a same-named field.
        """
        params: Dict[str, Any] = {
            "host": "0.0.0.0",
            "port": port,
            "metrics_port": metrics_port,
            "ram_size": (
                cache_service.config.ram_size if cache_service.config else None
            ),
            "chunk_size": (
                cache_service.config.chunk_size if cache_service.config else None
            ),
        }
        field_values = (
            cache_service.config.fields if cache_service.config else None
        ) or {}
        for field in provider.managed_fields:
            value = field_values.get(field.name, field.default)
            if isinstance(value, bool):
                value = str(value).lower()
            params.setdefault(field.name, value)
        return params

    def _apply_l2_storage(
        self,
        cache_service: CacheServicePublic,
        provider: CacheProvider,
        argv: Optional[List[str]],
    ) -> Tuple[Optional[List[str]], Dict[str, str]]:
        """
        Attach the service's L2 storage config to the cache server argument
        vector: each entry renders as one occurrence of the provider-declared
        flag carrying its adapter JSON, appended in declared order — the cache
        server prefers the earliest tier for reads and writes to all of them.
        A version running the image's own entrypoint has no vector of its
        own; the flags become its arguments.
        Secret-bearing fields go to the returned env; because env vars are
        process-global, two entries delivering a value through the same env
        var cannot coexist. Hand-written occurrences of the flag in the
        user parameters stay usable as an escape hatch for adapter types
        the declaration doesn't cover: they re-append after the structured
        entries, so the UI-visible order keeps the higher read priority.
        Raises ValueError when the provider can't serve the config.
        """
        l2_storages = cache_service.config.l2_storages if cache_service.config else None
        if not l2_storages:
            return argv, {}

        l2_args: List[str] = []
        l2_env: Dict[str, str] = {}
        env_sources: Dict[str, str] = {}
        for l2_storage in l2_storages:
            entry_args, entry_env = render_l2_adapter(
                provider, l2_storage.backend, l2_storage.params or {}
            )
            for env_name, value in entry_env.items():
                if env_name in env_sources:
                    raise ValueError(
                        f"L2 storage entries '{env_sources[env_name]}' and "
                        f"'{l2_storage.backend}' both deliver the env var "
                        f"'{env_name}'; only one entry may set it"
                    )
                env_sources[env_name] = l2_storage.backend
                l2_env[env_name] = value
            l2_args.extend(entry_args)

        remaining, hand_written = extract_flag_arguments(
            argv or [], provider.l2_adapter_flag
        )
        if hand_written:
            logger.info(
                f"Cache service {cache_service.name}"
                f"(id={cache_service.id}) also passes "
                f"{provider.l2_adapter_flag} via parameters; appending those "
                f"adapters after the structured entries"
            )
        return remaining + l2_args + hand_written, l2_env


def resolve_fallback_registry(cfg: Config) -> Optional[str]:
    """The registry a provider image falls back to when it carries none."""
    return registration.determine_default_registry(
        cfg.system_default_container_registry
    )
