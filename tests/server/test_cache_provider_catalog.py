import json

import pytest
from pydantic import ValidationError

from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderVersionConfig,
    render_argument,
    render_l2_adapter,
    render_optional_template,
    render_template,
    render_typed_template,
    resolved_field_values,
    validate_injection_templates,
)
from gpustack.schemas.cache_services import CacheServiceModeEnum
from gpustack.server import cache_provider_catalog
from gpustack.server.cache_provider_catalog import (
    get_cache_provider,
    load_cache_providers,
    render_injection,
)


def test_catalog_asset_loads():
    providers = load_cache_providers(reload=True)
    assert providers, "bundled cache-providers.yaml should yield at least one provider"


def test_malformed_entry_costs_only_its_own_provider(monkeypatch):
    """A declaration the model rejects — including one that is not a
    mapping at all — is skipped on its own; the rest of the catalog still
    serves, so a bad edit degrades one provider instead of every cache
    service in the deployment."""
    asset = (
        "- just a string\n"
        "- name: Broken\n"
        "  versions:\n"
        "    \"v1.0\": {}\n"  # resolves to no image
        "- name: Good\n"
        "  default_image: \"repo/cache:{{version}}\"\n"
        "  versions:\n"
        "    \"v1.0\": {}\n"
    )

    class _Asset:
        def is_file(self):
            return True

        def read_text(self, encoding=None):
            return asset

    try:
        monkeypatch.setattr(cache_provider_catalog, "files", lambda _package: _Asset())
        monkeypatch.setattr(_Asset, "joinpath", lambda self, _name: self, raising=False)
        providers = load_cache_providers(reload=True)
        assert [provider.name for provider in providers] == ["Good"]
    finally:
        # The loader caches for the process lifetime; leave the bundled
        # catalog in place for the tests that read it.
        monkeypatch.undo()
        load_cache_providers(reload=True)


def test_provider_defaults_fold_into_every_version():
    """The provider-level templates are resolved at construction, so a
    version config is self-contained: consumers read one effective image
    and command off it, and {{version}} keeps the version string stated
    once instead of copied into every tag."""
    provider = CacheProvider(
        name="Templated",
        default_image="registry/cache:{{version}}",
        default_runtime_images={"cuda": {"12": "registry/cache:{{version}}-cu12"}},
        default_run_command="cache serve --port {{port}}",
        versions={
            "v1.0": {},
            # A version departing from the layout keeps its own images,
            # and its declared map replaces the default whole.
            "v2.0": {
                "image": "other/cache:2.0",
                "runtime_images": {"cann": {"8": "other/cache:2.0-cann"}},
            },
        },
    )

    templated = provider.versions["v1.0"]
    assert templated.image == "registry/cache:v1.0"
    assert templated.runtime_images == {"cuda": {"12": "registry/cache:v1.0-cu12"}}
    assert templated.run_command == "cache serve --port {{port}}"

    explicit = provider.versions["v2.0"]
    assert explicit.image == "other/cache:2.0"
    assert explicit.runtime_images == {"cann": {"8": "other/cache:2.0-cann"}}
    # runtime_images doubles as the support matrix, so a replaced map
    # narrows the accelerators the version serves.
    assert explicit.supports_runtime("cann") is True
    assert explicit.supports_runtime("cuda") is False

    # Resolution is per version: the default map is copied, never shared.
    assert templated.runtime_images is not provider.default_runtime_images


def test_own_image_takes_over_the_layout_whole():
    """image and runtime_images are one layout: a version off the
    provider's tag scheme must not serve some accelerators from its own
    registry and the rest from the provider's template."""
    provider = CacheProvider(
        name="Templated",
        default_image="registry/cache:{{version}}",
        default_runtime_images={"cuda": {"12": "registry/cache:{{version}}-cu12"}},
        versions={"v1.0": {"image": "vendor/cache:one-off"}},
    )

    version = provider.versions["v1.0"]
    assert version.image == "vendor/cache:one-off"
    assert version.runtime_images == {}
    # With no layout of its own, every node runs the declared image.
    assert version.resolve_image("cuda", "12.8") == "vendor/cache:one-off"


def test_version_without_any_image_is_rejected():
    """An image is the one thing a managed version cannot do without;
    silently declaring none would only surface as a container that never
    starts."""
    with pytest.raises(ValidationError):
        CacheProvider(name="Imageless", versions={"v1.0": {}})


def test_own_launch_takes_over_the_pair_whole():
    """run_command and run_args are two slots of one launch: a version
    supplying arguments for the image's own entrypoint must not also
    inherit a command that replaces that entrypoint."""
    provider = CacheProvider(
        name="Launched",
        default_image="registry/cache:{{version}}",
        default_run_command="cache serve --port {{port}}",
        versions={
            "v1.0": {},
            "v2.0": {"run_args": "--port {{port}}"},
        },
    )

    inherited = provider.versions["v1.0"]
    assert inherited.run_command == "cache serve --port {{port}}"
    assert inherited.run_args is None

    own = provider.versions["v2.0"]
    assert own.run_command is None
    assert own.run_args == "--port {{port}}"


def test_version_declaring_both_launch_slots_is_rejected():
    """A command and its arguments concatenate into one vector either
    way, so declaring both states the same launch twice — and only one of
    them can decide whether the image's entrypoint survives."""
    with pytest.raises(ValidationError):
        CacheProvider(
            name="Ambiguous",
            default_image="registry/cache:v1",
            versions={"v1.0": {"run_command": "cache serve", "run_args": "--port 1"}},
        )


def test_component_declarations_validate():
    """Multi-component providers declare per-role topology and launch;
    dependencies must point at single-replica components (the only ones
    with one addressable endpoint) and cannot chain."""
    from gpustack.schemas.cache_providers import CacheProviderComponent

    provider = CacheProvider(
        name="Pool",
        default_image="repo/pool:{{version}}",
        versions={"v1.0": {}},
        components={
            "master": CacheProviderComponent(
                topology="replicas",
                run_command="pool-master --port {{port}}",
                serves_metrics=True,
                attach_endpoint=True,
                gpu_access=False,
            ),
            "store": CacheProviderComponent(
                topology="per_node",
                depends_on="master",
                run_command="pool-store --port {{port}}",
                enabled_by="standalone_store",
                gpu_access=False,
            ),
        },
        managed_fields=[
            {"name": "standalone_store", "type": "boolean", "default": False}
        ],
    )
    assert provider.component_layouts() == {
        "master": "replicas",
        "store": "per_node",
    }
    assert provider.get_component("store").depends_on == "master"
    assert provider.attach_component() == "master"
    # enabled_by gates on the field value, falling back to its declared
    # default
    assert provider.component_enabled("master", None) is True
    assert provider.component_enabled("store", None) is False
    assert provider.component_enabled("store", {"standalone_store": True}) is True

    # single-component providers map the "" component — the stored
    # column value of their instance rows
    single = CacheProvider(
        name="Solo",
        topology="per_node",
        default_image="repo/solo:{{version}}",
        versions={"v1.0": {}},
    )
    assert single.component_layouts() == {"": "per_node"}
    assert single.get_component("") is None

    with pytest.raises(ValidationError):
        CacheProvider(
            name="Dangling",
            default_image="repo/x:{{version}}",
            versions={"v1.0": {}},
            components={
                "store": CacheProviderComponent(
                    topology="per_node", depends_on="ghost"
                ),
                "master": CacheProviderComponent(attach_endpoint=True),
            },
        )
    with pytest.raises(ValidationError):
        # a per_node dependency has no single address to hand out
        CacheProvider(
            name="FanDep",
            default_image="repo/x:{{version}}",
            versions={"v1.0": {}},
            components={
                "a": CacheProviderComponent(topology="per_node"),
                "b": CacheProviderComponent(
                    topology="replicas", depends_on="a", attach_endpoint=True
                ),
            },
        )
    with pytest.raises(ValidationError):
        # neither does a multi-replica one (HA addresses the leader
        # through the backend URI instead)
        CacheProvider(
            name="WideDep",
            default_image="repo/x:{{version}}",
            versions={"v1.0": {}},
            components={
                "a": CacheProviderComponent(topology="replicas", replicas=3),
                "b": CacheProviderComponent(
                    topology="replicas", depends_on="a", attach_endpoint=True
                ),
            },
        )
    with pytest.raises(ValidationError):
        CacheProviderComponent(run_command="x", run_args="y")


def test_lmcache_provider_declaration():
    provider = get_cache_provider("LMCache")
    assert provider is not None
    # Managed only: LMCache is the single-container engine GPUStack runs
    # itself; reference-only distributed caches are what external is for.
    assert provider.supported_modes == [CacheServiceModeEnum.MANAGED.value]
    # The MP server keeps KV transfers node-local, so managed deployments
    # run one instance per worker of the cluster; attach_locality is the
    # declared contract the resolver's node-local rules key on —
    # deliberately separate from topology (a distributed pool may run
    # per-node data components while engines attach its cluster-wide
    # endpoint).
    assert provider.topology == "per_node"
    assert provider.attach_locality == "node_local"
    # /healthcheck verifies engine readiness (503 until initialized) and
    # lives on the HTTP frontend — the metrics port in our port model.
    assert provider.health_check.scheme == "http"
    assert provider.health_check.path == "/healthcheck"
    assert provider.health_check.target == "metrics"

    # The declared version pins a verified image tag; a service may also
    # pin its own image via the reserved "custom" version.
    assert provider.default_version == "v0.5.4"
    assert set(provider.versions) == {"v0.5.2", "v0.5.3", "v0.5.4"}
    assert provider.custom_version is True

    version_config, version = provider.get_version_config()
    assert version_config is not None
    assert version == provider.default_version
    # Upstream's tag layout, asserted against the resolved version rather
    # than a copy of it: the bare tag is the CUDA 13 build and cu129
    # serves CUDA 12 nodes. The worker resolves per node, so a
    # heterogeneous per_node fleet mixes images; unknown runtimes and
    # accelerator-less workers get the plain image.
    assert version_config.image == f"lmcache/vllm-openai:{version}"
    assert (
        version_config.resolve_image("cuda", "13.0") == f"lmcache/vllm-openai:{version}"
    )
    assert (
        version_config.resolve_image("cuda", "12.8")
        == f"lmcache/vllm-openai:{version}-cu129"
    )
    assert version_config.resolve_image(None, None) == f"lmcache/vllm-openai:{version}"
    # Two components: the cache servers engines attach to, and the peer
    # registry P2P needs. A component owns its launch, so the version
    # slots carry none.
    assert set(provider.components) == {"server", "coordinator"}
    for declared in provider.versions.values():
        assert declared.run_command is None
        assert declared.run_args is None
    server = provider.components["server"]
    coordinator = provider.components["coordinator"]
    # The full CLI entry: the HTTP frontend on --http-port serves
    # /metrics (same registry as the standalone exposition) plus
    # /healthcheck and the admin APIs; --prometheus-port is ignored
    # there, so the frontend port doubles as the metrics port.
    assert server.run_command == (
        "lmcache server --host {{host}} --port {{port}} "
        "--l1-size-gb {{ram_size}} --chunk-size {{chunk_size}} "
        "--http-host {{host}} --http-port {{metrics_port}} "
        "--supported-transfer-mode auto --worker-reap-timeout-seconds 60 "
        "--eviction-policy {{eviction_policy}} "
        "--eviction-trigger-watermark {{eviction_trigger_watermark}} "
        "--eviction-ratio {{eviction_ratio}} --l1-align-bytes 65536 "
        "--coordinator-url http://{{component.coordinator.address}} "
        "--p2p-advertise-url {{ports.p2p.url}}"
    )
    # Engines attach per node, and the servers hold the capacity.
    assert server.topology == "per_node"
    assert server.attach_endpoint is True
    assert server.serves_metrics is True
    assert server.resource_profile.ram_gib == "{{ram_size}}"
    # The registry exists only with P2P, holds no cache and takes no GPU;
    # so does the port its peers dial, which is why both P2P flags above
    # render empty and drop while the feature is off.
    assert coordinator.enabled_by == "enable_p2p"
    assert coordinator.gpu_access is False
    assert server.depends_on == "coordinator"
    assert server.enabled_ports({}) == []
    assert server.enabled_ports({"enable_p2p": True}) == ["p2p"]
    # The coordinator speaks HTTP, so its flag carries a scheme the
    # stamped address does not — and the whole token has to vanish with
    # the address, not leave a bare scheme behind.
    assert render_argument("http://{{component.coordinator.address}}", {}) == (
        "http://{{component.coordinator.address}}"
    )
    assert (
        render_argument(
            "http://{{component.coordinator.address}}",
            {"component.coordinator.address": None},
        )
        == ""
    )
    # Capacity, chunking and the eviction knobs are all ordinary declared
    # fields wired into the run command through their placeholders; the
    # platform reserves only host/port/metrics_port for itself.
    fields = {field.name: field for field in provider.managed_fields}
    assert set(fields) == {
        "ram_size",
        "chunk_size",
        "eviction_policy",
        "eviction_trigger_watermark",
        "eviction_ratio",
        "enable_p2p",
    }
    assert fields["enable_p2p"].default is False
    # capacity always renders (required guards a cleared value, the
    # default seeds the form); chunking may fall through to the engine
    assert fields["ram_size"].required and fields["ram_size"].default == 20
    assert not fields["chunk_size"].required
    # the pre-flight sizes an instance by the capacity field
    assert provider.resource_profile.ram_gib == "{{ram_size}}"
    assert fields["eviction_policy"].default == "LRU"
    assert fields["eviction_policy"].options == ["LRU", "IsolatedLRU", "noop"]
    assert fields["eviction_trigger_watermark"].type == "number"
    # Curated defaults from the upstream deployment recipes, deliberately
    # not the CLI code defaults (0.8/0.2): retain more, evict gentler.
    assert fields["eviction_trigger_watermark"].default == 0.85
    assert fields["eviction_ratio"].default == 0.1
    # Both are 0-1 fractions: without declared bounds and a fractional
    # step, the UI stepper walks 0.8 to -0.2 in one click.
    for name in ("eviction_trigger_watermark", "eviction_ratio"):
        assert fields[name].min == 0
        assert fields[name].max == 1
        assert fields[name].step == 0.05
    # Every declared field earns its place: it either fills a placeholder
    # somewhere in the declaration or gates something (a component, a
    # port). And none shadows a reserved platform placeholder.
    declaration = provider.model_dump_json()
    gates = {component.enabled_by for component in provider.components.values()} | {
        entry.enabled_by
        for component in provider.components.values()
        for entry in component.ports
        if not isinstance(entry, str)
    }
    for name in fields:
        assert f"{{{{{name}}}}}" in declaration or name in gates
    assert not set(fields) & {"host", "port", "metrics_port"}
    # Capacity flows through --l1-size-gb on the command line, not env.
    assert not version_config.env
    assert not server.env

    compat = provider.integration_for("vLLM")
    assert compat is not None


def test_metrics_for_resolves_version_override():
    """A version carrying its own metrics block owns it whole; versions
    without one and the custom version read the provider default — and
    a service stored without an explicit version (None) resolves through
    the default version like every other version lookup, so an override
    on the default version reaches the services actually running it."""
    from gpustack.schemas.cache_providers import (
        CacheProvider,
        CacheProviderMetrics,
        CacheProviderMetricValue,
        CacheProviderVersionConfig,
    )

    default = CacheProviderMetrics(
        mappings={"hit_rate": CacheProviderMetricValue(gauge="old_name")}
    )
    renamed = CacheProviderMetrics(
        mappings={"hit_rate": CacheProviderMetricValue(gauge="new_name")}
    )
    provider = CacheProvider(
        name="X",
        default_version="v2",
        versions={
            "v1": CacheProviderVersionConfig(image="img:v1"),
            "v2": CacheProviderVersionConfig(image="img:v2", metrics=renamed),
        },
        default_metrics=default,
    )

    assert provider.metrics_for("v1").mappings["hit_rate"].gauge == "old_name"
    assert provider.metrics_for("v2").mappings["hit_rate"].gauge == "new_name"
    assert provider.metrics_for("custom").mappings["hit_rate"].gauge == "old_name"
    assert provider.metrics_for("v9-unknown").mappings["hit_rate"].gauge == "old_name"
    assert provider.metrics_for(None).mappings["hit_rate"].gauge == "new_name"


def test_lmcache_metrics_declaration():
    provider = get_cache_provider("LMCache")
    assert provider is not None

    metrics = provider.default_metrics
    assert metrics is not None
    assert metrics.path == "/metrics"

    hit_rate = metrics.mappings["hit_rate"]
    assert hit_rate.ratio == {
        "numerator": "lmcache_mp_lookup_hit_tokens_total",
        "denominator": "lmcache_mp_lookup_requested_tokens_total",
    }
    assert (
        metrics.mappings["l1_usage_bytes"].gauge == "lmcache_mp_l1_memory_usage_bytes"
    )
    assert metrics.mappings["l1_usage_ratio"].gauge == "lmcache_mp_l1_usage_ratio"
    assert metrics.mappings["l2_usage_bytes"].gauge == "lmcache_mp_l2_usage_bytes"

    assert set(metrics.throughput) == {
        "l0_l1_store",
        "l0_l1_load",
        "l2_store",
        "l2_load",
    }
    for rule in metrics.throughput.values():
        assert rule.histogram_avg
        assert rule.gauge is None and rule.ratio is None
    # The OTel Prometheus exporter appends the histograms' "GB/s" unit to
    # the exported name; the declaration must carry the exported form.
    assert (
        metrics.throughput["l0_l1_store"].histogram_avg
        == "lmcache_mp_l0_l1_store_throughput_GB_per_second"
    )


def test_lmcache_l2_declaration():
    provider = get_cache_provider("LMCache")
    assert provider is not None
    assert provider.l2_adapter_flag == "--l2-adapter"
    assert set(provider.l2_backends) == {"fs_native", "resp", "s3"}

    fs = provider.l2_backends["fs_native"]
    fs_fields = {field.name: field for field in fs.fields}
    assert set(fs_fields) == {
        "base_path",
        "max_capacity_gb",
        "num_workers",
        "use_odirect",
    }
    assert fs_fields["base_path"].required is True
    # seeded into the form so a plain "add Local Filesystem" works
    # without inventing a path; lands in the platform data dir, which
    # the mirrored deployment mounts from the host
    assert fs_fields["base_path"].default == "/var/lib/gpustack/cache/lmcache/l2"
    assert fs_fields["max_capacity_gb"].type == "number"
    assert fs_fields["num_workers"].type == "number"
    assert fs_fields["use_odirect"].type == "boolean"
    # fs_native fields all ride in the adapter JSON.
    assert all(field.env_name is None for field in fs.fields)

    resp = provider.l2_backends["resp"]
    resp_fields = {field.name: field for field in resp.fields}
    assert set(resp_fields) == {
        "host",
        "port",
        "username",
        "password",
        "max_capacity_gb",
    }
    assert resp_fields["host"].required is True
    assert resp_fields["port"].required is True
    assert resp_fields["port"].type == "number"
    assert resp_fields["max_capacity_gb"].type == "number"
    # Credentials reach the server via env, keeping them off the command line.
    assert resp_fields["username"].env_name == "LMCACHE_RESP_USERNAME"
    assert resp_fields["password"].type == "password"
    assert resp_fields["password"].env_name == "LMCACHE_RESP_PASSWORD"

    s3 = provider.l2_backends["s3"]
    s3_fields = {field.name: field for field in s3.fields}
    assert set(s3_fields) == {
        "s3_endpoint",
        "s3_region",
        "aws_access_key_id",
        "aws_secret_access_key",
        "disable_tls",
        "max_capacity_gb",
    }
    # Virtual-hosted addressing needs both pieces to sign requests.
    assert s3_fields["s3_endpoint"].required is True
    assert s3_fields["s3_region"].required is True
    assert s3_fields["disable_tls"].type == "boolean"
    assert s3_fields["max_capacity_gb"].type == "number"
    # Credentials ride in env (resolved via the boto3 default chain),
    # keeping them off the command line like the resp backend's.
    assert s3_fields["aws_access_key_id"].env_name == "AWS_ACCESS_KEY_ID"
    assert s3_fields["aws_secret_access_key"].type == "password"
    assert s3_fields["aws_secret_access_key"].env_name == "AWS_SECRET_ACCESS_KEY"


def test_mooncake_provider_declaration():
    provider = get_cache_provider("Mooncake")
    assert provider is not None
    # Shipped and supported as part of the GPUStack catalog, like LMCache;
    # "partner" is reserved for vendor-branded providers.
    assert provider.source.value == "built_in"
    # A distributed pool is network-attachable from any worker — spanning
    # (multi-worker) instances attach by design.
    assert provider.attach_locality == "cluster"
    assert provider.icon == "/static/catalog_icons/mooncake.png"
    # Managed only: the platform runs the master (and optional stores)
    # from the vLLM runner images, whose bundled wheel matches the
    # engines' — Mooncake's RPC wire format breaks across builds.
    assert provider.supported_modes == [CacheServiceModeEnum.MANAGED.value]
    assert provider.default_version == "v0.3.10.post2"
    assert provider.custom_version is True
    version = provider.versions["v0.3.10.post2"]
    assert version.runtime_images["cuda"]["13"] == (
        "gpustack/runner:cuda13.0-vllm0.27.1"
    )
    assert version.runtime_images["cuda"]["12"] == (
        "gpustack/runner:cuda12.9-vllm0.27.1"
    )
    # CPU-only workers (a RAM-rich store node) run the plain image.
    assert version.image == "gpustack/runner:cuda12.9-vllm0.27.1"
    # The runner images double as the support matrix, and the wheel they
    # bundle is what the engines pair with: a NPU worker gets the CANN
    # build, a GPU worker the CUDA one.
    assert version.runtime_images["cann"] == {
        "9": "gpustack/runner:cann9.1-910b-vllm0.23.0"
    }
    assert version.supports_runtime("cann") is True
    integration = provider.integration_for("vLLM", "cann")
    assert integration is not None

    # The master coordinates and serves the metrics; capacity is either
    # engine-contributed (embedded, the default) or owned by optional
    # store replicas gated behind pool_mode.
    master = provider.components["master"]
    assert master.attach_endpoint is True
    assert master.serves_metrics is True
    assert master.gpu_access is False
    assert master.health_check.scheme == "http"
    assert master.health_check.target == "metrics"
    store = provider.components["store"]
    assert store.depends_on == "master"
    assert store.replicas_by == "store_replicas"
    assert store.enabled_by == "pool_mode"
    assert store.enabled_when == "standalone-store"
    assert store.gpu_access is False
    # the pool owner is mooncake_client, the process that can hold a disk
    # tier; its master address and advertised host ride flags, and the
    # disk-tier settings ride env the client reads only while offload is on
    assert store.run_command.startswith("mooncake_client")
    assert "--master_server_address {{component.master.address}}" in store.run_command
    assert "--host {{worker_ip}}" in store.run_command
    assert "--enable_offload={{enable_ssd_offload}}" in store.run_command
    assert store.env["MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"] == "{{ssd_offload_path}}"
    assert store.env["MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES"] == (
        "{{ssd_capacity_gb|gib_to_bytes}}"
    )
    # spilled cache belongs on host disk, not in the container layer
    assert store.mounts == ["{{ssd_offload_path}}"]
    # the client's RPC port binds the advertised address alone, so
    # readiness is read from the HTTP server it can be told to run, which
    # answers 503 until it holds a master
    assert "--enable_http_server --http_port {{metrics_port}}" in store.run_command
    assert store.health_check.scheme == "http"
    assert store.health_check.path == "/health"
    assert store.health_check.target == "metrics"
    assert provider.component_enabled("store", None) is False
    assert (
        provider.component_enabled("store", {"pool_mode": "standalone-store"}) is True
    )

    fields = {field.name: field for field in provider.managed_fields}
    assert set(fields) == {
        "enable_ha",
        "etcd_endpoints",
        "master_replicas",
        "pool_mode",
        "engine_segment_size",
        "store_replicas",
        "store_segment_size",
        "enable_ssd_offload",
        "ssd_offload_path",
        "ssd_capacity_gb",
        "protocol",
        "device_name",
        "eviction_high_watermark_ratio",
        "eviction_ratio",
    }
    # HA is off by default and the whole feature hangs off one switch:
    # the etcd endpoint is demanded only while the switch is on, the
    # master pool sizes itself from a field that resolves to a single
    # master while it is off, and clients reach the elected leader
    # through etcd rather than through whichever replica they found.
    # The engine floor is the release that registers the connector: an
    # older one cannot start at all, so it must degrade instead.
    vllm_integration = provider.integration_for("vLLM", "cuda")
    assert vllm_integration.versions == ">=0.21.0"
    # Ascend attaches through its own pool connector: upstream's asserts
    # on registration there, because it expects one packed KV tensor per
    # layer while Ascend keeps K and V apart. Same config file, same
    # master — only the connector and its extra config differ.
    ascend = provider.integration_for("vLLM", "cann")
    assert ascend.injection.kv_transfer_config.kv_connector == "AscendStoreConnector"
    assert ascend.injection.kv_transfer_config.kv_connector_extra_config == {
        "backend": "mooncake",
        "lookup_rpc_port": "0",
    }
    assert (
        ascend.injection.env["MOONCAKE_CONFIG_PATH"]
        == vllm_integration.injection.env["MOONCAKE_CONFIG_PATH"]
    )

    assert fields["enable_ha"].default is False
    assert fields["etcd_endpoints"].required is True
    assert fields["etcd_endpoints"].visible_by == "enable_ha"
    assert fields["etcd_endpoints"].visible_when is True
    # the shape of an endpoint list does not survive prose
    assert fields["etcd_endpoints"].placeholder == "10.0.0.1:2379,10.0.0.2:2379"
    # Masters do not vote among themselves, so this is not a quorum to
    # size: two already survives losing one, and it doubles as the floor
    # that rules out the contradiction of HA with a single master.
    assert fields["master_replicas"].default == 2
    assert fields["master_replicas"].min == 2
    assert fields["master_replicas"].gated_default == 1
    # The form renders fields in declaration order with no group frames,
    # so adjacency is what ties a field to what it governs: the mode sits
    # next to the sizing it switches, and the HA posture goes last.
    order = [field.name for field in provider.managed_fields]
    assert order[:2] == ["pool_mode", "engine_segment_size"]
    assert order[-3:] == ["enable_ha", "etcd_endpoints", "master_replicas"]
    assert master.replicas_by == "master_replicas"
    assert master.address_template == "etcd://{{etcd_endpoints}}"
    ha_off = resolved_field_values(provider.managed_fields, {})
    assert ha_off["master_replicas"] == 1
    assert render_optional_template(master.address_template, ha_off) is None
    ha_on = resolved_field_values(
        provider.managed_fields,
        {"enable_ha": True, "etcd_endpoints": "10.0.0.9:2379"},
    )
    assert ha_on["master_replicas"] == 2
    assert (
        render_optional_template(master.address_template, ha_on)
        == "etcd://10.0.0.9:2379"
    )
    # The leader publishes its own routable address into the election,
    # and the endpoints ride the flag this version actually declares.
    assert "--rpc_address {{worker_ip}}" in master.run_command
    assert "--etcd_endpoints {{etcd_endpoints}}" in master.run_command
    # Masters and clients disagree on the default coordination keyspace
    # (--cluster_id falls back to "mooncake_cluster", a client with no
    # env to "mooncake"), so the service names it on both sides.
    assert "--cluster_id gpustack-cache-service-{{service_id}}" in master.run_command
    assert store.env["MC_STORE_CLUSTER_ID"] == "gpustack-cache-service-{{service_id}}"

    assert fields["pool_mode"].default == "embedded"
    assert fields["pool_mode"].option_values() == ["embedded", "standalone-store"]
    # options carry display labels where the stored value reads poorly
    assert fields["pool_mode"].options[1].label == "Standalone Store"
    # 20 per GPU / 40 per store replica: production-leaning anchors
    # aligned with LMCache's default 20 GiB (note the engine value
    # multiplies by the node's GPU count)
    assert fields["engine_segment_size"].default == 20
    assert fields["store_segment_size"].default == 40
    # Ascend rides CANN's ADXL; one protocol serves the whole service, so
    # a GPU/NPU mix has to stay on TCP.
    assert fields["protocol"].option_values() == ["tcp", "rdma", "ascend"]
    # NPU nodes have one working transport, and the connector refuses any
    # other, so the form takes it from the hardware rather than offering
    # a choice that only fails once an engine attaches.
    assert fields["protocol"].framework_defaults == {"cann": "ascend"}
    assert fields["eviction_high_watermark_ratio"].default == 0.95
    assert fields["eviction_ratio"].default == 0.1
    # the mode-scoped capacity pair renders one at a time, both speaking
    # Mooncake's own vocabulary
    assert fields["engine_segment_size"].label == "Global Segment Size (GiB)"
    assert fields["store_segment_size"].label == "Global Segment Size (GiB)"
    assert fields["engine_segment_size"].visible_when == "embedded"
    # while a standalone store owns the pool, the engine contributes 0
    assert fields["engine_segment_size"].gated_default == 0
    assert fields["store_segment_size"].visible_when == "standalone-store"
    assert fields["store_replicas"].visible_when == "standalone-store"
    assert fields["device_name"].visible_by == "protocol"
    assert fields["device_name"].visible_when == "rdma"
    assert provider.external_fields == []

    # Master-side metrics: the pool's allocated/capacity view on the
    # master's Prometheus endpoint. Lookup-hit accounting lives in the
    # engine-side connector, so no hit_rate.
    metrics = provider.default_metrics
    assert metrics is not None
    assert metrics.path == "/metrics"
    assert "hit_rate" not in metrics.mappings
    assert metrics.mappings["l1_usage_bytes"].gauge == "master_allocated_bytes"
    assert metrics.mappings["l1_usage_ratio"].gauge_ratio == {
        "numerator": "master_allocated_bytes",
        "denominator": "master_total_capacity_bytes",
    }
    assert provider.dashboard_uid == "gpustack-mooncake"


def test_mooncake_injection_renders_store_connector_env():
    provider = get_cache_provider("Mooncake")
    rendered = render_injection(
        provider,
        "vLLM",
        {
            "master_server_address": "10.0.0.9:50051",
            "local_hostname": "10.0.0.7",
        },
    )
    assert rendered is not None
    env, args, files = rendered
    # The connector reads its configuration solely from the JSON file
    # MOONCAKE_CONFIG_PATH points at; the injection materializes it.
    assert env == {
        "MOONCAKE_CONFIG_PATH": "/tmp/gpustack-mooncake.json",
        # Chunk hashes must agree across engine processes; a random
        # per-process hash seed would silently break sharing.
        "PYTHONHASHSEED": "0",
        # TCP transport pools connections instead of opening one per
        # transfer slice, which exhausts ephemeral ports under prefill
        # bursts; the RDMA path ignores the switch.
        "MC_TCP_ENABLE_CONNECTION_POOL": "1",
        # Every client of a service reads the elected leader from the
        # keyspace named after it; upstream's master and client defaults
        # disagree, so it is named explicitly.
        "MC_STORE_CLUSTER_ID": "gpustack-cache-service-{{service_id}}",
    }
    config = json.loads(files["/tmp/gpustack-mooncake.json"])
    # The managed defaults render the mainstream embedded shape: each
    # engine GPU contributes the default segment size.
    assert config["mode"] == "embedded"
    assert config["global_segment_size"] == "20GB"
    assert config["master_server_address"] == "10.0.0.9:50051"
    assert config["metadata_server"] == "P2PHANDSHAKE"
    assert config["protocol"] == "tcp"
    # An unset optional field renders empty in the file; the config
    # schema treats empty device_name as "no RDMA device".
    assert config["device_name"] == ""
    assert config["local_buffer_size"] == "4GB"
    # a declared boolean renders as the literal JSON accepts, not
    # Python's capitalized repr
    assert config["enable_offload"] is False
    assert args[0] == "--kv-transfer-config"
    assert '"kv_connector":"MooncakeStoreConnector"' in args[1]


def test_meshfusion_provider_is_a_branded_lmcache_clone():
    meshfusion = get_cache_provider("XSKY MeshFusion")
    lmcache = get_cache_provider("LMCache")
    assert meshfusion is not None and lmcache is not None

    # XSKY partner branding. The catalog carries the public product name
    # only; AKV-Cache and XDFS are XSKY internal component names. No
    # provider-specific dashboard this version — the engine exposes
    # LMCache's lmcache_mp_* metrics, so both providers fall back to the
    # generic cache-service dashboard.
    assert meshfusion.source.value == "partner"
    assert meshfusion.icon == "/static/catalog_icons/xsky.png"
    assert meshfusion.dashboard_uid is None
    assert lmcache.dashboard_uid is None

    # Functionally an LMCache fork: the runtime contract matches LMCache
    # apart from the branding fields and the XSKY-specific L2 storage.
    brand_fields = {
        "name",
        "display_name",
        "source",
        "icon",
        "description",
        "links",
        "dashboard_uid",
    }
    diverging_fields = {
        # P2P is declared for LMCache alone until XSKY confirms their
        # image ships the coordinator CLI, so only LMCache splits into
        # components — and carries the field that gates them. Both are
        # checked below rather than left unchecked.
        "components",
        "managed_fields",
        "l2_backends",
        "versions",
        "default_version",
        # The staged Ascend build lives in the image templates; the CUDA
        # layout and the run command are asserted equal below.
        "default_runtime_images",
        # The two launch through different slots: MeshFusion's image is
        # expected to start the cache server itself.
        "default_run_command",
        "default_run_args",
        "inference_backend_integrations",
    }
    meshfusion_dump = meshfusion.model_dump()
    lmcache_dump = lmcache.model_dump()
    differing = {
        key
        for key in meshfusion_dump
        if key not in brand_fields | diverging_fields
        and meshfusion_dump[key] != lmcache_dump[key]
    }
    assert differing == set()

    # The versions diverge from LMCache's only by the staged Ascend
    # build (an assumed image for XSKY to correct) and by which version
    # each provider defaults to; at a given version the CUDA builds stay
    # LMCache's.
    mf_version = meshfusion.versions[meshfusion.default_version]
    lm_version = lmcache.versions[meshfusion.default_version]
    assert mf_version.runtime_images["cuda"] == lm_version.runtime_images["cuda"]
    assert "cann" in mf_version.runtime_images
    assert "cann" not in lm_version.runtime_images

    # Every integration is framework-scoped — the catalog is the single
    # accelerator gate. MeshFusion diverges from LMCache only by the
    # extra cann-scoped vLLM entry (an assumed placeholder for XSKY;
    # vllm-ascend trails vLLM, so its attachable range is declared
    # separately). The cuda entries mirror LMCache's.
    vllm_entries = [
        c for c in meshfusion.inference_backend_integrations if c.backend == "vLLM"
    ]
    assert [(c.frameworks, c.versions) for c in vllm_entries] == [
        (["cuda"], ">=0.25.0"),
        (["cann"], ">=0.25.0"),
    ]
    lm_vllm = lmcache.integration_for("vLLM", "cuda")
    assert lm_vllm.frameworks == ["cuda"]

    # The two diverge on P2P alone: LMCache splits into components and
    # declares the switch that gates them, while every other declared
    # field stays in step. Sizing stays shared — LMCache's server
    # component repeats it, but the provider-level profile both read is
    # the same.
    assert set(meshfusion.components) == set()
    assert [field.name for field in meshfusion.managed_fields] == [
        field.name for field in lmcache.managed_fields if field.name != "enable_p2p"
    ]
    assert [
        field.model_dump()
        for field in lmcache.managed_fields
        if field.name != "enable_p2p"
    ] == [field.model_dump() for field in meshfusion.managed_fields]
    for entry in vllm_entries:
        assert entry.injection == lm_vllm.injection
    sglang_entries = [
        c for c in meshfusion.inference_backend_integrations if c.backend == "SGLang"
    ]
    assert sglang_entries == [lmcache.integration_for("SGLang", "cuda")]
    # Framework routing: cuda/cann engine workers each get their scoped
    # entry; an unknown framework (pre-scheduling validation) still
    # answers "attachable"; an undeclared framework gets no contract —
    # for vLLM and SGLang alike (no worker-side half-injection).
    assert meshfusion.integration_for("vLLM", "cuda") is vllm_entries[0]
    assert meshfusion.integration_for("vLLM", "cann") is vllm_entries[1]
    assert meshfusion.integration_for("vLLM") is vllm_entries[0]
    assert meshfusion.integration_for("vLLM", "rocm") is None
    assert meshfusion.integration_for("SGLang", "cann") is None
    assert all(c.frameworks == ["cuda"] for c in lmcache.inference_backend_integrations)

    # MeshFusion adds XSKY's store L2 backend (adapter type "xdfs",
    # branded with the XSKY icon) on top of the LMCache-inherited
    # backends; LMCache has none of it.
    assert "xdfs" not in lmcache.l2_backends
    # The inherited backends must stay byte-identical, not just share keys.
    for key, backend in lmcache.l2_backends.items():
        assert meshfusion.l2_backends[key] == backend
    xdfs = meshfusion.l2_backends["xdfs"]
    assert xdfs.icon == "/static/catalog_icons/xsky.png"
    xdfs_fields = {field.name for field in xdfs.fields}
    assert {"metadata_endpoint", "sdk_config_file", "max_write_inflight_bytes"} <= (
        xdfs_fields
    )
    assert next(f for f in xdfs.fields if f.name == "metadata_endpoint").required
    # No store-side metrics scrape this version: L2 observability rides on
    # the cache server's own lmcache_mp_* metrics.
    assert all(field.metrics_target is False for field in xdfs.fields)

    args, env = render_l2_adapter(
        meshfusion,
        "xdfs",
        {"metadata_endpoint": "10.0.0.20:8000"},
    )
    assert '"metadata_endpoint":"10.0.0.20:8000"' in args[1]
    assert env == {}


def test_provider_brand_links():
    lmcache = get_cache_provider("LMCache")
    assert {link.label for link in lmcache.links} == {"Documentation", "GitHub"}
    assert all(link.url.startswith("https://") for link in lmcache.links)

    mooncake = get_cache_provider("Mooncake")
    assert {link.label for link in mooncake.links} == {"Documentation", "GitHub"}

    meshfusion = get_cache_provider("XSKY MeshFusion")
    assert meshfusion.links, "partner card needs at least one brand link"


def test_version_config_resolves_runtime_image_by_platform_rule():
    cfg = CacheProviderVersionConfig(
        image="repo/x:v1",
        runtime_images={"cuda": {"12.9": "repo/x:v1-cu129", "12": "repo/x:v1-cu12"}},
    )
    # Newest declared version <= the host runtime wins (the rule shared
    # with inference-backend runners); a host older than every declared
    # build gets the oldest one — the closest guess, not the plain
    # (newest-CUDA) image. Other backends fall back to the plain image.
    assert cfg.resolve_image("cuda", "12.9") == "repo/x:v1-cu129"
    assert cfg.resolve_image("cuda", "12.4") == "repo/x:v1-cu12"
    assert cfg.resolve_image("cuda", "11.8") == "repo/x:v1-cu12"
    assert cfg.resolve_image("rocm", "6.1") == "repo/x:v1"


def test_version_config_runtime_support_matrix():
    cfg = CacheProviderVersionConfig(
        image="repo/x:v1",
        runtime_images={"cuda": {"13": "repo/x:v1"}},
    )
    # runtime_images doubles as the support matrix: a foreign
    # accelerator (e.g. Ascend's cann) is rejected instead of falling
    # back to an image built for another family; accelerator-less nodes
    # run the plain image CPU-only.
    assert cfg.supports_runtime("cuda") is True
    assert cfg.supports_runtime("cann") is False
    assert cfg.supports_runtime(None) is True
    unconstrained = CacheProviderVersionConfig(image="repo/x:v1")
    assert unconstrained.supports_runtime("cann") is True


def test_provider_lookup_is_case_insensitive():
    assert get_cache_provider("lmcache") is not None
    assert get_cache_provider("no-such-provider") is None


def test_render_injection_substitutes_host_and_port():
    provider = get_cache_provider("LMCache")
    rendered = render_injection(
        provider,
        "vLLM",
        {
            "host": "10.0.0.5",
            "port": 9000,
            "chunk_size": 256,
            "ram_size": 8,
            "locality": "node_local",
        },
    )
    assert rendered is not None
    env, args, files = rendered
    # The MP connector carries the endpoint in the transfer config and no
    # config file; the only env is the pinned hash seed keeping chunk keys
    # consistent across engine processes on the builtin-hash fallback path.
    assert env == {"PYTHONHASHSEED": "0"}
    assert files == {}
    assert args[0] == "--kv-transfer-config"
    assert '"kv_connector":"LMCacheMPConnector"' in args[1]
    assert '"lmcache.mp.host":"tcp://10.0.0.5"' in args[1]
    assert '"lmcache.mp.port":9000' in args[1]
    # The declaration's locality_params map the resolver's neutral
    # placement fact to LMCache's transfer-mode vocabulary: node-local
    # attachments may negotiate CUDA IPC (auto), remote ones stay on
    # engine-driven copies since IPC handles cannot cross hosts.
    assert '"lmcache.mp.mp_transfer_mode":"auto"' in args[1]
    assert args[2] == "--disable-hybrid-kv-cache-manager"


def test_kv_transfer_config_renders_structured_slot_with_types():
    """The connector slot is declared structured (one owner assembles the
    single-value engine flag) and placeholder types survive into the
    JSON payload — the port must be a number, not a string."""
    provider = get_cache_provider("LMCache")
    integration = provider.integration_for("vLLM", "cuda")
    slot = integration.injection.kv_transfer_config
    assert slot is not None
    assert slot.flag == "--kv-transfer-config"
    assert slot.kv_connector == "LMCacheMPConnector"

    rendered = render_injection(
        provider,
        "vLLM",
        {"host": "10.0.0.5", "port": 9000, "locality": "node_local"},
    )
    assert rendered is not None
    _, args, _ = rendered
    assert args[0] == "--kv-transfer-config"
    payload = json.loads(args[1])
    extra = payload["kv_connector_extra_config"]
    assert extra["lmcache.mp.host"] == "tcp://10.0.0.5"
    assert extra["lmcache.mp.port"] == 9000
    assert isinstance(extra["lmcache.mp.port"], int)
    assert extra["lmcache.mp.mp_transfer_mode"] == "auto"
    # Free-form args follow the slot: the non-hybrid manager requirement
    # and the graceful-shutdown window (CUDA IPC teardown).
    assert args[2:] == [
        "--disable-hybrid-kv-cache-manager",
        "--shutdown-timeout",
        "20",
    ]


def test_secret_and_scrape_fields_never_enter_injection_templates():
    """Injection renders into the snapshot on the model instance row,
    outside the cache-service redaction's reach — so password-typed
    external fields must never be referenced by injection templates,
    and metrics_target fields (scrape addresses, not connector config)
    are excluded from the injection namespace by contract."""
    for provider in load_cache_providers():
        excluded = {
            field.name
            for field in provider.external_fields
            if field.type == "password" or field.metrics_target
        }
        if not excluded:
            continue
        for integration in provider.inference_backend_integrations:
            blob = integration.injection.model_dump_json()
            for name in excluded:
                assert f"{{{{{name}}}}}" not in blob, (
                    f"{provider.name}: injection references excluded " f"field {name}"
                )


def test_render_injection_drops_metrics_target_values():
    from gpustack.schemas.cache_providers import CacheProviderExternalField

    provider = CacheProvider(
        name="scrape-provider",
        external_fields=[
            CacheProviderExternalField(name="metadata_server"),
            CacheProviderExternalField(name="exporter_address", metrics_target=True),
        ],
        inference_backend_integrations=[
            {
                "backend": "vLLM",
                "injection": {"env": {"META": "{{metadata_server}}"}},
            }
        ],
    )
    rendered = render_injection(
        provider,
        "vLLM",
        {"metadata_server": "10.0.0.9:8000", "exporter_address": "10.0.0.9:9100"},
    )
    assert rendered is not None
    env, _, _ = rendered
    assert env == {"META": "10.0.0.9:8000"}


def test_render_injection_maps_node_local_locality_to_auto():
    """Engines attach node-local only (the resolver degrades instead of
    crossing nodes), so the declaration maps the sole placement fact to
    the auto-negotiated zero-copy path."""
    provider = get_cache_provider("LMCache")
    rendered = render_injection(
        provider,
        "vLLM",
        {"host": "10.0.0.5", "port": 9000, "locality": "node_local"},
    )
    assert rendered is not None
    _, args, _ = rendered
    assert '"lmcache.mp.mp_transfer_mode":"auto"' in args[1]


def test_render_injection_explicit_param_beats_locality_default():
    provider = get_cache_provider("LMCache")
    rendered = render_injection(
        provider,
        "vLLM",
        {
            "host": "10.0.0.5",
            "port": 9000,
            "locality": "node_local",
            "mp_transfer_mode": "engine_driven",
        },
    )
    assert rendered is not None
    _, args, _ = rendered
    assert '"lmcache.mp.mp_transfer_mode":"engine_driven"' in args[1]


def test_render_injection_returns_none_for_incompatible_backend():
    provider = get_cache_provider("LMCache")
    rendered = render_injection(
        provider,
        "no-such-backend",
        {"host": "10.0.0.5", "port": 9000},
    )
    assert rendered is None


def test_resolve_image_matches_minor_version_keys():
    """runtime_images accepts full-version keys with the same match rule
    inference-backend runners use: a 12.8 host takes the 12.6 build
    instead of silently falling back to the plain (newest-CUDA) image."""
    version_config = CacheProviderVersionConfig(
        image="cache:latest",
        runtime_images={"cuda": {"12.6": "cache:cu126", "12": "cache:cu12"}},
    )
    assert version_config.resolve_image("cuda", "12.8") == "cache:cu126"
    assert version_config.resolve_image("cuda", "12.3") == "cache:cu12"
    # accelerator-less nodes and undeclared backends keep the plain image
    assert version_config.resolve_image(None, None) == "cache:latest"
    assert version_config.resolve_image("rocm", "6.3") == "cache:latest"


def test_mooncake_injection_backfills_optional_fields_as_empty():
    """A declared field without a default (device_name is empty on TCP)
    must still backstop its placeholder — otherwise the literal
    "{{device_name}}" lands in the rendered config file."""
    provider = get_cache_provider("Mooncake")
    rendered = render_injection(
        provider,
        "vLLM",
        {
            "master_server_address": "10.0.0.9:50051",
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "tcp",
        },
    )
    assert rendered is not None
    _, _, files = rendered
    config = json.loads(files["/tmp/gpustack-mooncake.json"])
    assert config["device_name"] == ""
    assert "{{" not in files["/tmp/gpustack-mooncake.json"]


def test_lmcache_sglang_injection_renders_config_file():
    """SGLang attaches through --enable-lmcache with a YAML config file
    carrying the MP server address; the adapter pulls the chunk size from
    the server, so host/port is the whole contract."""
    provider = get_cache_provider("LMCache")
    compat = provider.integration_for("SGLang")
    assert compat is not None
    # LMCache MP support landed in sglang v0.5.13 (PR #24089).
    assert compat.versions == ">=0.5.13"
    rendered = render_injection(
        provider,
        "SGLang",
        {"host": "10.0.0.5", "port": 9000, "locality": "node_local"},
    )
    assert rendered is not None
    env, args, files = rendered
    assert env == {"PYTHONHASHSEED": "0"}
    assert args == [
        "--enable-lmcache",
        "--lmcache-config-file",
        "/tmp/gpustack-lmcache-sgl.yaml",
    ]
    config = files["/tmp/gpustack-lmcache-sgl.yaml"]
    assert 'mp_host: "10.0.0.5"' in config
    assert "mp_port: 9000" in config
    assert "{{" not in config


def test_integration_for_framework_scoping():
    """A scoped-only declaration attaches on its named frameworks only;
    an unknown framework (validation before scheduling) still answers
    "attachable" so a scoped-only provider is not rejected up front."""
    provider = CacheProvider(
        name="scoped-only",
        inference_backend_integrations=[
            {"backend": "vLLM", "frameworks": ["cann"], "versions": ">=1"},
        ],
    )
    scoped = provider.inference_backend_integrations[0]
    assert provider.integration_for("vLLM", "cann") is scoped
    assert provider.integration_for("vLLM", "cuda") is None
    assert provider.integration_for("vLLM") is scoped
    assert provider.integration_for("SGLang", "cann") is None


def test_bundled_catalog_passes_injection_contract():
    """Every shipped provider must satisfy the placeholder contract the
    loader enforces (a violating provider is excluded at load time)."""
    for provider in load_cache_providers():
        assert validate_injection_templates(provider) == []


def test_injection_contract_flags_violations():
    from gpustack.schemas.cache_providers import CacheProviderExternalField

    provider = CacheProvider(
        name="bad-provider",
        external_fields=[
            CacheProviderExternalField(name="token", type="password"),
            CacheProviderExternalField(name="exporter", metrics_target=True),
        ],
        inference_backend_integrations=[
            {
                "backend": "vLLM",
                "injection": {
                    "env": {"TOKEN": "{{token}}", "EXP": "{{exporter}}"},
                    "args": ["--peer", "{{undeclared_thing}}"],
                    "locality_params": {
                        "node_local": {"mode": "auto"},
                        # "mode" missing here: not common to all buckets
                        "remote": {"other": "x"},
                    },
                    "files": {"/tmp/x": "mode={{mode}}"},
                },
            }
        ],
    )
    errors = validate_injection_templates(provider)
    joined = "\n".join(errors)
    assert "references field 'token'" in joined
    assert "references field 'exporter'" in joined
    assert "placeholder 'undeclared_thing'" in joined
    # "mode" is not present in every locality bucket, so it is
    # unresolvable on the remote path and must be flagged.
    assert "placeholder 'mode'" in joined


def test_mooncake_ssd_offload_is_declared_end_to_end():
    """The disk tier is one switch: the master admits offloading, the
    store process that owns the pool holds the tier, and the engine is
    told the pool has one so it sizes its staging for disk reads."""
    provider = get_cache_provider("Mooncake")
    values = {
        "pool_mode": "standalone-store",
        "enable_ssd_offload": True,
        "ssd_offload_path": "/nvme/mooncake",
        "ssd_capacity_gb": 200,
    }
    params = resolved_field_values(provider.managed_fields, values)

    master = provider.components["master"]
    assert "--enable_offload=true" in render_template(master.run_command, params)

    store = provider.components["store"]
    command = render_template(store.run_command, params)
    assert "--enable_offload=true" in command
    env = {key: render_template(value, params) for key, value in store.env.items()}
    assert env["MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"] == "/nvme/mooncake"
    # the field states GiB; the client takes bytes and would otherwise
    # default of 2 TB bounds no real disk
    assert env["MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES"] == str(200 * 1024**3)

    rendered = render_injection(
        provider,
        "vLLM",
        {
            "master_server_address": "10.0.0.9:50051",
            "local_hostname": "10.0.0.7",
            **values,
        },
    )
    config = json.loads(rendered[2]["/tmp/gpustack-mooncake.json"])
    assert config["enable_offload"] is True


def test_mooncake_ssd_fields_resolve_empty_while_the_tier_is_off():
    """Off, the path and the cap render to nothing: the env entries drop
    and the store's mount is skipped, so a disk the service does not use
    is never bound."""
    provider = get_cache_provider("Mooncake")
    params = resolved_field_values(
        provider.managed_fields, {"pool_mode": "standalone-store"}
    )
    assert params["enable_ssd_offload"] is False
    assert params["ssd_offload_path"] == ""
    assert params["ssd_capacity_gb"] == ""

    store = provider.components["store"]
    assert (
        render_argument(store.env["MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"], params) == ""
    )
    assert (
        render_argument(store.env["MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES"], params)
        == ""
    )
    assert render_argument(store.mounts[0], params) == ""


def test_template_filters_convert_on_the_way_out():
    params = {"cap": 3, "flag": True, "off": False}
    assert render_template("{{cap|gib_to_bytes}}", params) == str(3 * 1024**3)
    # a filtered placeholder is a converted value, never the raw one
    assert render_typed_template("{{cap|gib_to_bytes}}", params) == str(3 * 1024**3)
    assert render_typed_template("{{cap}}", params) == 3
    # one declared boolean serves JSON and gflags alike
    assert render_template("--enable={{flag}} --off={{off}}", params) == (
        "--enable=true --off=false"
    )


def test_a_gated_gate_closes_the_fields_behind_it():
    """The disk-tier path hangs off a switch that hangs off the mode. A
    switch left on in standalone-store must not keep the path alive once
    the service is embedded: gates resolve through the chain, not one
    level."""
    provider = get_cache_provider("Mooncake")
    params = resolved_field_values(
        provider.managed_fields,
        {
            "pool_mode": "embedded",
            "enable_ssd_offload": True,
            "ssd_offload_path": "/nvme/mooncake",
            "ssd_capacity_gb": 200,
        },
    )
    assert params["enable_ssd_offload"] is False
    assert params["ssd_offload_path"] == ""
    assert params["ssd_capacity_gb"] == ""
