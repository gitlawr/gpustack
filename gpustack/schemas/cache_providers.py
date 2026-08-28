import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, model_validator

from gpustack.utils.version import pick_runtime_version

CUSTOM_VERSION = "custom"
"""Reserved provider_version identifier: the service pins a user-supplied
container image (config.image) instead of a declared version."""


class CacheProviderSourceEnum(str, Enum):
    BUILT_IN = "built_in"
    COMMUNITY = "community"
    PARTNER = "partner"


class CacheProviderLink(BaseModel):
    """A brand link (docs, product page, support) rendered on the
    provider's catalog card."""

    label: str
    url: str


class CacheProviderHealthCheck(BaseModel):
    scheme: str = "tcp"
    """Probe scheme: "tcp" (connect check) or "http" (GET on path)."""

    path: Optional[str] = None
    """HTTP path for scheme "http". Ignored for "tcp"."""

    target: str = "port"
    """Which of the service's ports the probe hits: "port" (the service
    port) or "metrics" (the metrics port) — e.g. LMCache's /healthcheck
    lives on the HTTP frontend, not the ZMQ control port."""


class CacheProviderVersionConfig(BaseModel):
    image: Optional[str] = None
    """Container image for the managed cache server; the fallback when
    runtime_images has no entry for the node's accelerator runtime. A
    version resolving to no image at all is a catalog error."""

    runtime_images: Dict[str, Dict[str, str]] = {}
    """Images keyed by accelerator backend (e.g. "cuda") then runtime
    version (a major like "12" or a full "12.6"). The worker resolves per
    node at instance start with the platform-wide runtime-match rule
    (see pick_runtime_version), so a heterogeneous per_node fleet mixes
    images correctly. All entries must be command-compatible with
    run_command.

    Together with image this forms the version's image layout, inherited
    from the provider's defaults as a unit: a version declaring either
    field owns both."""

    run_command: Optional[str] = None
    """Argument-vector template with {{placeholder}} substitution, taking
    the image's ENTRYPOINT slot — it names the executable, not just its
    flags. For an image whose entrypoint already starts the cache server
    (and may do setup around it) declare run_args instead, which keeps
    that entrypoint."""

    run_args: Optional[str] = None
    """Argument template appended to the image's own entrypoint, for
    images that start the cache server themselves. Same substitution as
    run_command; the two are alternatives — a version declares at most
    one, since a command and its arguments concatenate into the same
    vector either way."""

    env: Optional[Dict[str, str]] = None
    """Env template for the managed container. Values support {{placeholder}}."""

    metrics: Optional["CacheProviderMetrics"] = None
    """Overrides the provider-level metrics declaration for this version,
    whole — declared when the version's exposition renames metrics (an
    exporter change typically renames a family at once, so per-key
    inheritance would hide half the picture). Undeclared versions read
    the provider default."""

    def supports_runtime(self, backend: Optional[str]) -> bool:
        """Whether the version can run on the node's accelerator.
        runtime_images doubles as the support matrix: a node with a
        detected accelerator is only served when its backend has an
        entry — the plain image targets one accelerator family and
        would just crash-loop elsewhere. Accelerator-less nodes always
        pass (the cache server runs CPU-only there), as does a version
        declaring no runtime_images."""
        return not backend or not self.runtime_images or backend in self.runtime_images

    def resolve_image(
        self,
        backend: Optional[str] = None,
        runtime_version: Optional[str] = None,
    ) -> str:
        """The image for a node's accelerator runtime, matched with the
        same rule inference-backend runners use (newest declared version
        <= the host runtime, with same-major and oldest fallbacks).
        Accelerator-less nodes and backends without entries get the
        plain image."""
        if not backend:
            return self.image
        by_version = self.runtime_images.get(backend) or {}
        picked = pick_runtime_version(list(by_version), runtime_version)
        if picked is None:
            return self.image
        return by_version[picked]


class CacheProviderKVTransferConfig(BaseModel):
    """Structured form of a vLLM-style single-slot connector argument.

    The engine accepts exactly one value for this flag while several
    parties may want to write it (this cache integration, PD transfer
    connectors, the user); a slot like that must be assembled by one
    owner. Declaring the payload structured — instead of a pre-rendered
    JSON string in args — keeps the platform able to compose it (e.g.
    into a MultiConnector) and to detect user takeover of the flag."""

    flag: str = "--kv-transfer-config"
    """Engine argument that carries the serialized payload."""

    kv_connector: str
    kv_role: str = "kv_both"
    kv_connector_extra_config: Dict[str, Any] = {}
    """Connector-specific settings. String values support {{placeholder}};
    a value that is exactly one placeholder keeps the parameter's type
    (e.g. a port renders as a JSON number, not a string)."""


class CacheProviderInjection(BaseModel):
    """Connector config injected into an inference engine that attaches to a cache service."""

    env: Dict[str, str] = {}
    """Env template. Values support {{placeholder}}; entries rendering empty are dropped."""

    kv_transfer_config: Optional[CacheProviderKVTransferConfig] = None
    """The engine's connector slot, rendered ahead of args as
    "<flag> <compact JSON>"."""

    args: List[str] = []
    """Extra command args for the inference engine. Items support {{placeholder}}."""

    files: Dict[str, str] = {}
    """Config files written inside the engine container before it starts,
    keyed by absolute path; contents support {{placeholder}}. For
    connectors that read a config file instead of env/args (e.g.
    Mooncake's MOONCAKE_CONFIG_PATH JSON)."""

    locality_params: Dict[str, Dict[str, Any]] = {}
    """Placeholder defaults keyed by the engine-to-instance placement the
    resolver derives ("node_local" | "remote"). Lets a declaration vary
    connector config by placement (e.g. a same-node zero-copy transfer
    mode) in its own vocabulary; the platform only supplies the fact."""


class CacheProviderIntegration(BaseModel):
    backend: str
    """Inference backend name this provider can attach to (e.g. "vLLM")."""

    frameworks: Optional[List[str]] = None
    """Accelerator frameworks this entry is scoped to, in the
    runtime_images key vocabulary (e.g. "cuda", "cann"). None makes the
    entry generic: it serves every framework no scoped entry claims.
    Lets a provider vary the attach contract per accelerator (e.g.
    vllm-ascend trails vLLM releases and may need a different connector
    config). Entries are selected whole — the chosen entry's versions
    AND injection apply; scoped entries do not merge with the generic
    one."""

    versions: Optional[str] = None
    """Compatible backend version range (e.g. ">=0.25.0"). Enforced when
    the engine version is known: model validation rejects a pinned
    backend_version outside the range, and the injection resolver
    degrades instead of injecting args the engine may not accept.
    Unparseable values fail open."""

    injection: CacheProviderInjection = CacheProviderInjection()


class CacheProviderResourceProfile(BaseModel):
    """How capacity config maps to host resource claims. Informational in v1."""

    ram_gib: Optional[str] = None
    cpu: Optional[float] = None


class CacheProviderMetricValue(BaseModel):
    """How to extract one semantic metric value from the provider's
    Prometheus exposition. At most one of the forms is set (validated —
    the query builder would otherwise silently pick one of several).
    Consumed by the cache-service metrics endpoint, which translates the
    form into a PromQL query over the service's scrape series."""

    gauge: Optional[str] = None
    """Gauge metric name; charted as-is."""

    rate: Optional[str] = None
    """Counter metric name; charted as its per-second rate over the
    chart's rate window (e.g. lookup traffic in tokens per second)."""

    ratio: Optional[Dict[str, str]] = None
    """{"numerator": counter, "denominator": counter}: the ratio of the
    two counters' increases over the chart's rate window."""

    gauge_ratio: Optional[Dict[str, str]] = None
    """{"numerator": gauge, "denominator": gauge}: the instantaneous ratio
    of two gauges (e.g. allocated / capacity)."""

    histogram_avg: Optional[str] = None
    """Histogram base name: increase(_sum) / increase(_count) over the
    rate window, i.e. the average observed value."""

    aggregate: Optional[str] = None
    """How gauge values combine into the service-level series: "sum"
    (default — capacities, byte counts) or "avg" (ratios). Only valid
    with the gauge form: the other forms aggregate naturally (operands
    sum before dividing, weighting instances by their actual traffic)."""

    @model_validator(mode="after")
    def _validate_forms(self):
        forms = [
            name
            for name in ("gauge", "rate", "ratio", "gauge_ratio", "histogram_avg")
            if getattr(self, name)
        ]
        if len(forms) > 1:
            raise ValueError(
                f"metric rule sets multiple extraction forms: {', '.join(forms)}"
            )
        if self.aggregate is not None:
            if self.aggregate not in ("sum", "avg"):
                raise ValueError(
                    f"aggregate must be 'sum' or 'avg', got '{self.aggregate}'"
                )
            if not self.gauge:
                raise ValueError("aggregate applies only to the gauge form")
        return self


class CacheProviderMetrics(BaseModel):
    """Where a cache service's Prometheus exposition is scraped, and how
    its semantic metrics are extracted from it."""

    path: str = "/metrics"
    """HTTP path of the Prometheus exposition on the metrics port."""

    default_port: Optional[int] = None
    """The engine's conventional metrics port (external mode: seeds the
    registration form's metrics-port field)."""

    mappings: Dict[str, CacheProviderMetricValue] = {}
    """Semantic key -> extraction rule. Keys use the platform's tier
    vocabulary — L1 is the memory (near) tier, L2 the capacity tier
    (disk/remote) — regardless of the provider's own naming: hit_rate,
    l1_usage_bytes, l1_usage_ratio, l2_usage_bytes. A provider with
    several L2 backends keeps them apart by series label, not by key."""

    throughput: Dict[str, CacheProviderMetricValue] = {}
    """Named throughput series (unit: GB/s) -> extraction rule."""


# CacheProviderVersionConfig.metrics forward-references this module's
# tail; resolve it now that the metrics classes exist.
CacheProviderVersionConfig.model_rebuild()


class CacheProviderL2Field(BaseModel):
    """One configurable parameter of an L2 storage backend."""

    name: str
    label: Optional[str] = None
    """UI label; defaults to name."""

    type: str = "string"
    """Value type: "string" | "number" | "boolean" | "password"."""

    required: bool = False
    default: Optional[Any] = None

    env_name: Optional[str] = None
    """When set, the value is delivered to the managed container via this
    env var instead of the adapter JSON (keeps secrets off the command line)."""

    metrics_target: bool = False
    """When set, the value is an additional Prometheus scrape address
    (host:port or URL) for this storage backend, added to the service's
    scrape targets. Never rendered into the adapter JSON or env."""


class CacheProviderL2Backend(BaseModel):
    """A storage backend the provider's L2 adapter can spill KV cache to."""

    display_name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    """Logo URL for brand display; the UI falls back to a generic icon."""

    fields: List[CacheProviderL2Field] = []


class CacheProviderField(BaseModel):
    """A managed-mode configuration value promoted to a structured field
    in the service form's advanced section. The field carries no
    destination of its own: it adds a {{name}} placeholder to the
    template namespace, and the version's run_command/env templates
    decide where the value lands. A flag whose placeholder renders empty
    is dropped with its value, and user-supplied free-form parameters
    still override any flag the templates produce."""

    name: str
    """Placeholder name; must not collide with the reserved platform
    placeholders (host/port/metrics_port/ram_size/chunk_size)."""

    label: Optional[str] = None
    description: Optional[str] = None

    type: str = "string"
    """Value type: "string" | "number" | "boolean" (booleans render as
    "true"/"false")."""

    default: Optional[Any] = None
    options: Optional[List[str]] = None
    """When set, the UI offers a fixed choice."""

    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    """Numeric bounds and stepper increment for number-typed fields;
    the UI control and the API validation both honor them."""


class CacheProviderExternalField(BaseModel):
    """One connection parameter a user supplies when registering an external
    service of this provider (e.g. Mooncake's metadata_server, protocol). The
    value is rendered into the provider's injection templates via the
    {{name}} placeholder; the primary service address lives on the endpoint
    (host/port) instead, not here."""

    name: str
    label: Optional[str] = None
    """UI label; defaults to name."""

    description: Optional[str] = None

    type: str = "string"
    """Value type: "string" | "number" | "boolean" | "password"."""

    required: bool = False
    default: Optional[Any] = None

    options: Optional[List[str]] = None
    """When set, the UI offers a fixed choice (e.g. protocol tcp/rdma)."""

    metrics_target: bool = False
    """When set, the value is an additional Prometheus scrape address
    (host:port or URL) added to the service's scrape targets. It is
    observability config, not connector config: the value never enters
    the injection placeholder namespace."""


class CacheProviderComponent(BaseModel):
    """One process role of a multi-component managed provider (e.g.
    Mooncake's coordinating master and its per-node memory-segment
    stores). A provider without ``components`` is single-component: the
    provider-level topology and the version launch templates describe
    its one process, and nothing changes for it."""

    topology: str = "replicas"
    """Instance layout of this component: "replicas" runs ``replicas``
    scheduler-placed instances spread across matching workers (sticky to
    the workers they already run on), "per_node" runs one per matching
    cluster worker."""

    replicas: int = 1
    """Instance count for the "replicas" topology. Fewer matching
    workers than replicas deploys what fits (a smaller pool beats
    parking the service). Ignored by "per_node"."""

    depends_on: Optional[str] = None
    """Name of a component whose instances must be RUNNING (with ports
    known) before this component's instances are created — e.g. stores
    need the master's address. The dependency must be a single-replica
    component, the only kind with one addressable endpoint (HA lifts
    this by addressing the leader through the HA backend URI instead)."""

    run_command: Optional[str] = None
    run_args: Optional[str] = None
    """Launch template of this component, same semantics as the version
    slots (a command takes the entrypoint, args ride the image's own).
    Components own their launch: version-level launch templates apply
    only to single-component providers, since one template cannot serve
    two roles. {{component.<name>.address}} resolves to a
    single-replica component's host:port."""

    env: Dict[str, str] = {}
    """Env template for this component's container; values support
    {{placeholder}} including cross-component addresses."""

    health_check: Optional[CacheProviderHealthCheck] = None
    """Probe for this component's instances; None inherits the
    provider-level health_check."""

    serves_metrics: bool = False
    """Whether this component's instances expose the exposition the
    provider's metrics declaration describes (scrape targets are built
    from these instances only — e.g. the Mooncake master, not the
    stores)."""

    attach_endpoint: bool = False
    """Whether engines attach to this component's address (exactly one
    component of a multi-component provider declares it — e.g. the
    Mooncake master; stores are internal). It must be a single-replica
    component and cannot be gated by enabled_by."""

    enabled_by: Optional[str] = None
    """Name of a boolean managed field that turns this component on;
    None means always on. A disabled component keeps no instances —
    e.g. Mooncake's optional standalone stores."""

    gpu_access: bool = True
    """Whether this component's container mounts the node's GPUs. LMCache
    needs a CUDA context for its IPC transport; a pure-RAM component
    (Mooncake master/store) opts out and saves the per-GPU context
    memory."""

    @model_validator(mode="after")
    def _one_launch_slot(self):
        if self.run_command and self.run_args:
            raise ValueError("a component declares run_command or run_args, not both")
        return self


class CacheProvider(BaseModel):
    name: str
    display_name: Optional[str] = None
    source: CacheProviderSourceEnum = CacheProviderSourceEnum.BUILT_IN
    description: Optional[str] = None
    icon: Optional[str] = None

    links: List[CacheProviderLink] = []
    """Brand links (docs, product page) rendered on the catalog card."""

    dashboard_uid: Optional[str] = None
    """UID of a provider-specific Grafana dashboard provisioned alongside
    the generic cache-service one; the service's Grafana entry points
    redirect to it. None falls back to the generic dashboard."""

    supported_modes: List[str] = []
    """Deployment modes the provider supports: "managed" and/or "external"."""

    topology: str = "replicas"
    """Managed-mode instance layout of a single-component provider:
    "replicas" runs one scheduler-placed instance (pinned when the
    service names a worker_id); "per_node" runs one instance per active
    worker of the service's cluster, following workers as they join and
    leave. Multi-component providers declare topology per component
    instead."""

    attach_locality: str = "cluster"
    """Where an engine may attach from: "node_local" means the connector
    only works against a cache server on the engine's own node (e.g.
    LMCache MP's CUDA-IPC transport), so remote fallback and multi-worker
    instances degrade; "cluster" (default) means the endpoint is
    network-reachable from any worker. Deliberately separate from
    ``topology``: placement and attach contract only coincide for
    LMCache-style providers — a distributed pool may run per-node data
    components while engines attach its cluster-wide endpoint."""

    components: Dict[str, CacheProviderComponent] = {}
    """Managed-mode process roles, keyed by component name. Empty means
    single-component (the provider-level topology and version launch
    templates describe the one process). Declared components each own
    their topology, launch and env; the shared image layout still comes
    from the version."""

    default_version: Optional[str] = None
    versions: Dict[str, CacheProviderVersionConfig] = {}

    default_run_command: Optional[str] = None
    default_run_args: Optional[str] = None
    """Launch template shared by versions that declare none of their own.
    A provider whose CLI is stable across its release line states it once
    here; a version departing from it declares its own run_command or
    run_args, and one that must run the image entrypoint bare opts out
    with an empty run_command. The pair is inherited as a unit — a
    version declaring either owns its launch — and resolved into each
    version at model construction, so every consumer (including the
    catalog API) reads the effective launch off the version config."""

    default_image: Optional[str] = None
    default_runtime_images: Dict[str, Dict[str, str]] = {}
    """Image layout shared by versions that declare none of their own, in
    the same shape as a version's image / runtime_images. {{version}}
    stands for the version key, so a provider whose tags embed the
    version declares the layout once and each version is just its key —
    the version string is stated once instead of copied into every tag.
    The placeholder is optional: a provider whose versions all share one
    image states it here without it. A version whose images depart from
    the layout (a one-off registry, a build only it has) declares its
    own, which takes over the layout whole."""

    custom_version: bool = False
    """Whether a service may pin a user-supplied container image instead of
    a declared version; the default version's run command and env templates
    still apply, so the image must be command-compatible."""

    external_fields: List[CacheProviderExternalField] = []
    """External-mode connection parameters the user fills at registration.
    Rendered into the connector injection via {{name}} alongside the
    endpoint address; empty for managed-only providers."""

    managed_fields: List[CacheProviderField] = []
    """Managed-mode configuration values promoted to structured form
    fields (e.g. the eviction policy), wired into the runtime config by
    the version templates via {{name}}; everything else stays reachable
    through the free-form parameters editor."""

    resource_profile: Optional[CacheProviderResourceProfile] = None
    health_check: CacheProviderHealthCheck = CacheProviderHealthCheck()
    default_metrics: Optional[CacheProviderMetrics] = None
    """The all-version default declaration, named like the other
    provider-level defaults (default_image, default_run_command). Do not
    read it directly for a service — a version may carry its own metrics
    block; metrics_for() resolves the effective one."""

    inference_backend_integrations: List[CacheProviderIntegration] = []

    common_parameters: List[str] = []
    """Flags the UI offers as completion hints in the extra-parameters
    editor. Excludes flags GPUStack injects itself (host/ports/capacity/
    L2 adapter), which would conflict with the structured config."""

    l2_adapter_flag: Optional[str] = None
    """Command-line flag that carries the L2 adapter JSON
    (e.g. "--l2-adapter"); None means the provider has no L2 support."""

    l2_backends: Dict[str, CacheProviderL2Backend] = {}
    """Adapter type identifier (the "type" value in the adapter JSON)
    -> backend declaration."""

    def component_layouts(self) -> Dict[str, str]:
        """Component name -> topology. A single-component provider maps
        {"": topology} — the empty string is the stored component value
        of its instance rows (a real column value, not NULL, so the
        (service, worker, component) uniqueness holds on every database:
        NULLs compare distinct inside unique constraints)."""
        if self.components:
            return {name: c.topology for name, c in self.components.items()}
        return {"": self.topology}

    def get_component(self, name: str) -> Optional[CacheProviderComponent]:
        if not name:
            return None
        return self.components.get(name)

    @model_validator(mode="after")
    def _validate_components(self) -> "CacheProvider":
        for name, component in self.components.items():
            if component.topology not in ("replicas", "per_node"):
                raise ValueError(
                    f"component '{name}' declares unknown topology "
                    f"'{component.topology}'"
                )
            if component.replicas < 1:
                raise ValueError(
                    f"component '{name}' declares replicas "
                    f"{component.replicas}; at least one is required"
                )
            dep_name = component.depends_on
            if dep_name is None:
                continue
            dependency = self.components.get(dep_name)
            if dependency is None or dep_name == name:
                raise ValueError(
                    f"component '{name}' depends on unknown component " f"'{dep_name}'"
                )
            if dependency.topology != "replicas" or dependency.replicas != 1:
                raise ValueError(
                    f"component '{name}' depends on '{dep_name}', which is "
                    "not a single-replica component: only those have one "
                    "addressable endpoint"
                )
            if dependency.depends_on:
                raise ValueError(
                    f"component '{name}' depends on '{dep_name}', which has "
                    "its own dependency: chains are not supported"
                )
        if self.components:
            attach = [
                name
                for name, component in self.components.items()
                if component.attach_endpoint
            ]
            if len(attach) != 1:
                raise ValueError(
                    "a multi-component provider declares exactly one "
                    "attach_endpoint component; engines need one address"
                )
            spec = self.components[attach[0]]
            if spec.topology != "replicas" or spec.replicas != 1:
                raise ValueError(
                    f"attach_endpoint component '{attach[0]}' must be a "
                    "single-replica component"
                )
            if spec.enabled_by:
                raise ValueError(
                    f"attach_endpoint component '{attach[0]}' cannot be "
                    "gated by enabled_by: the attach address must always "
                    "exist"
                )
            declared_fields = {field.name for field in self.managed_fields}
            for name, component in self.components.items():
                if component.enabled_by and component.enabled_by not in declared_fields:
                    raise ValueError(
                        f"component '{name}' is enabled by undeclared "
                        f"managed field '{component.enabled_by}'"
                    )
        return self

    def attach_component(self) -> str:
        """Name of the component engines attach to ("" for
        single-component providers)."""
        for name, component in self.components.items():
            if component.attach_endpoint:
                return name
        return ""

    def component_enabled(
        self, name: str, config_fields: Optional[Dict[str, Any]]
    ) -> bool:
        """Whether the component should have instances for a service
        configured with ``config_fields`` (the managed field values;
        the field declaration's default fills an unset value)."""
        spec = self.get_component(name)
        if spec is None or not spec.enabled_by:
            return True
        fields = config_fields or {}
        if spec.enabled_by in fields:
            return bool(fields[spec.enabled_by])
        declared = next(
            (field for field in self.managed_fields if field.name == spec.enabled_by),
            None,
        )
        return bool(declared.default) if declared else False

    def metrics_for(self, version: Optional[str]) -> Optional[CacheProviderMetrics]:
        """The effective metrics declaration for a service pinned to
        ``version``. Resolution rides get_version_config, so None falls
        back to the default version like everywhere else (a managed
        service created without an explicit version stores None). A
        resolved version owns its block whole; versions without one —
        and the custom version, whose image ships unknown metrics — fall
        back to the provider default (best effort, and a mismatch
        surfaces as a reasoned all-queries-failed degradation rather
        than silently empty charts)."""
        config, _ = self.get_version_config(version)
        if config is not None and config.metrics is not None:
            return config.metrics
        return self.default_metrics

    @model_validator(mode="after")
    def resolve_version_defaults(self) -> "CacheProvider":
        """Fold the provider-level templates into each version so that a
        version config is self-contained: the worker, the validators and
        the catalog API all read one effective image and command off it,
        with no second lookup on the provider."""
        for version, config in self.versions.items():
            # run_command and run_args are two slots of one launch, so a
            # version opts out of both together: a version declaring args
            # for its own entrypoint must not also inherit a command that
            # replaces that entrypoint.
            if config.run_command is None and config.run_args is None:
                config.run_command = self.default_run_command
                config.run_args = self.default_run_args
            if config.run_command and config.run_args:
                raise ValueError(
                    f"Cache provider '{self.name}' version '{version}' "
                    "declares both run_command and run_args: a command and "
                    "its arguments form one vector, so state it as whichever "
                    "one the image's entrypoint calls for"
                )
            # image and runtime_images describe one image layout, so a
            # version opts out of both together: inheriting half a layout
            # would serve some accelerators from the version's own
            # registry and the rest from the provider's.
            if config.image is None and not config.runtime_images:
                config.image = self.default_image
                config.runtime_images = {
                    backend: dict(images)
                    for backend, images in self.default_runtime_images.items()
                }
            params = {"version": version}
            if config.image:
                config.image = render_template(config.image, params)
            config.runtime_images = {
                backend: {
                    runtime: render_template(image, params)
                    for runtime, image in images.items()
                }
                for backend, images in config.runtime_images.items()
            }
            if not config.image:
                raise ValueError(
                    f"Cache provider '{self.name}' version '{version}' "
                    "resolves to no image: it must declare one, or declare "
                    "neither image nor runtime_images and inherit the "
                    "provider's default_image"
                )
        return self

    def get_version_config(
        self, version: Optional[str] = None
    ) -> Tuple[Optional[CacheProviderVersionConfig], Optional[str]]:
        """Resolve a version config, falling back to the default version."""
        target = version or self.default_version
        if target and target in self.versions:
            return self.versions[target], target
        return None, target

    def integration_for(
        self, backend_name: str, framework: Optional[str] = None
    ) -> Optional[CacheProviderIntegration]:
        """
        Pick the integration entry for an inference backend, preferring
        one scoped to the engine worker's accelerator ``framework`` over
        the generic (unscoped) entry. With framework unknown (None, e.g.
        validation before scheduling) a scoped-only declaration still
        answers "attachable" through any entry for the backend.
        """
        matches = [
            entry
            for entry in self.inference_backend_integrations
            if entry.backend.lower() == (backend_name or "").lower()
        ]
        if framework:
            for entry in matches:
                if entry.frameworks and framework in entry.frameworks:
                    return entry
        generic = next((c for c in matches if not c.frameworks), None)
        if generic is not None or framework:
            return generic
        return matches[0] if matches else None


# Dots namespace cross-component placeholders (component.master.address).
_TEMPLATE_PATTERN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_.]*)\}\}")


def render_template(value: str, params: Dict[str, Any]) -> str:
    """
    Substitute {{placeholder}} occurrences with values from params.
    Placeholders whose value is None render as an empty string; unknown
    placeholders are left unchanged.
    """

    def replace_var(match):
        var_name = match.group(1)
        if var_name in params:
            resolved = params[var_name]
            return "" if resolved is None else str(resolved)
        return match.group(0)

    return _TEMPLATE_PATTERN.sub(replace_var, value)


def _coerce_l2_field_value(field: CacheProviderL2Field, value: Any) -> Any:
    """
    Normalize a field value for the adapter JSON. Number fields render as
    JSON integers when integral (a port must serialize as 6379, not 6379.0);
    boolean fields render as JSON booleans.
    """
    if field.type == "number":
        number = float(value)
        return int(number) if number.is_integer() else number
    if field.type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return value


def render_l2_adapter(
    provider: CacheProvider, backend: str, params: Dict[str, Any]
) -> Tuple[List[str], Dict[str, str]]:
    """
    Build the (command args, container env) that configure a managed cache
    server's L2 storage backend. Fields declaring env_name are delivered via
    env so secrets stay off the command line; the rest form the adapter JSON
    together with the backend's type identifier. Unset optional fields are
    omitted from both. Raises ValueError when the provider has no L2 support
    or does not declare the backend.
    """
    if not provider.l2_adapter_flag:
        raise ValueError(
            f"Cache provider '{provider.name}' does not support L2 storage"
        )
    backend_spec = provider.l2_backends.get(backend)
    if backend_spec is None:
        raise ValueError(
            f"Cache provider '{provider.name}' has no L2 storage "
            f"backend '{backend}'"
        )

    adapter: Dict[str, Any] = {"type": backend}
    env: Dict[str, str] = {}
    for field in backend_spec.fields:
        if field.metrics_target:
            # Scrape-address fields configure GPUStack's metrics
            # collection, not the cache server.
            continue
        value = params.get(field.name, field.default)
        if value is None or value == "":
            continue
        value = _coerce_l2_field_value(field, value)
        if field.env_name:
            env[field.env_name] = str(value)
        else:
            adapter[field.name] = value

    args = [provider.l2_adapter_flag, json.dumps(adapter, separators=(",", ":"))]
    return args, env


RESERVED_INJECTION_PLACEHOLDERS = frozenset(
    {
        "host",
        "port",
        "metrics_port",
        "ram_size",
        "chunk_size",
        "local_hostname",
        "master_server_address",
        "locality",
    }
)
"""Placeholders the platform itself supplies to injection rendering."""


def validate_injection_templates(provider: "CacheProvider") -> List[str]:
    """
    Check a provider's injection templates against the placeholder
    contract; returns human-readable violations (empty when clean).

    Two invariants are enforced at load time because their failure modes
    are silent at runtime: a placeholder that nothing resolves renders
    literally into connector config (corrupting it), and a password- or
    metrics_target-typed field rendered into injection would ride into
    the cache_config snapshot on the model instance row, outside the
    cache-service redaction's reach. Every referenced placeholder must
    be a reserved platform placeholder, a declared (non-secret,
    non-scrape) field, or a key present in every locality bucket.
    """
    errors: List[str] = []
    declared = {
        field.name
        for field in provider.external_fields
        if field.type != "password" and not field.metrics_target
    }
    declared |= {field.name for field in provider.managed_fields}
    excluded = {
        field.name
        for field in provider.external_fields
        if field.type == "password" or field.metrics_target
    }
    for integration in provider.inference_backend_integrations:
        injection = integration.injection
        buckets = [set(bucket) for bucket in injection.locality_params.values()]
        locality_common = set.intersection(*buckets) if buckets else set()
        templates: List[str] = []
        templates.extend(injection.env.values())
        templates.extend(injection.args)
        templates.extend(injection.files.values())
        if injection.kv_transfer_config:
            templates.extend(
                value
                for value in injection.kv_transfer_config.kv_connector_extra_config.values()
                if isinstance(value, str)
            )
        referenced = {
            name
            for template in templates
            for name in _TEMPLATE_PATTERN.findall(template)
        }
        prefix = f"'{provider.name}' integration '{integration.backend}'"
        for name in sorted(referenced & excluded):
            errors.append(
                f"{prefix} references field '{name}': password and "
                "metrics_target values never enter injection"
            )
        allowed = RESERVED_INJECTION_PLACEHOLDERS | declared | locality_common
        for name in sorted(referenced - allowed - excluded):
            errors.append(
                f"{prefix} references placeholder '{name}', which is not a "
                "reserved placeholder, a declared field, or a key present "
                "in every locality bucket"
            )
    return errors


def render_typed_template(value: Any, params: Dict[str, Any]) -> Any:
    """
    Render a template value preserving parameter types: a string that is
    exactly one known placeholder substitutes to the parameter's value
    as-is (an int stays an int), anything else renders as a string.
    Non-string values pass through untouched.
    """
    if not isinstance(value, str):
        return value
    match = _TEMPLATE_PATTERN.fullmatch(value)
    if match and match.group(1) in params:
        return params[match.group(1)]
    return render_template(value, params)


def render_kv_transfer_config(
    config: CacheProviderKVTransferConfig, params: Dict[str, Any]
) -> List[str]:
    """Serialize a structured connector slot into its argument pair:
    [flag, compact JSON payload]."""
    payload: Dict[str, Any] = {
        "kv_connector": config.kv_connector,
        "kv_role": config.kv_role,
    }
    if config.kv_connector_extra_config:
        payload["kv_connector_extra_config"] = {
            key: render_typed_template(value, params)
            for key, value in config.kv_connector_extra_config.items()
        }
    return [config.flag, json.dumps(payload, separators=(",", ":"))]


def render_injection(
    integration: CacheProviderIntegration, params: Dict[str, Any]
) -> Tuple[Dict[str, str], List[str], Dict[str, str]]:
    """
    Render an integration entry's injection templates into
    (env, args, files). The structured connector slot (if declared)
    renders ahead of the free-form args. Env entries whose rendered
    value is empty are dropped so that unset optional parameters (e.g.
    chunk_size) don't produce invalid engine config; file contents keep
    empty renderings — a config file's schema decides what an empty
    field means.
    """
    env: Dict[str, str] = {}
    for key, value in (integration.injection.env or {}).items():
        rendered = render_template(value, params)
        if rendered:
            env[key] = rendered
    args: List[str] = []
    if integration.injection.kv_transfer_config is not None:
        args.extend(
            render_kv_transfer_config(integration.injection.kv_transfer_config, params)
        )
    args.extend(
        render_template(arg, params) for arg in (integration.injection.args or [])
    )
    files = {
        path: render_template(content, params)
        for path, content in (integration.injection.files or {}).items()
    }
    return env, args, files
