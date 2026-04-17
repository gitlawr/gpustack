"""E2E test configuration management."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# Default values
DEFAULT_IMAGE = "gpustack/gpustack:dev"
DEFAULT_PASSWORD = "Admin@123"
DEFAULT_SERVER_URL = "http://localhost:80"


@dataclass
class ServerConfig:
    """GPUStack server configuration."""

    url: str = DEFAULT_SERVER_URL
    admin_password: str = DEFAULT_PASSWORD
    api_key: str = ""
    timeout: int = 30
    verify_ssl: bool = False


@dataclass
class DockerConfig:
    """Docker deployment configuration."""

    image: str = DEFAULT_IMAGE
    container_prefix: str = "gpustack-e2e"
    cache_dir: str = "/tmp/gpustack-e2e/cache"
    runtime: str = "nvidia"
    use_sudo: bool = False
    bootstrap_password: str = DEFAULT_PASSWORD
    startup_wait: int = 30


@dataclass
class GPUConfig:
    """GPU environment configuration."""

    type: str = "nvidia"  # nvidia, amd, ascend, cpu
    count: int = 1
    min_memory_gb: int = 8


@dataclass
class MultimodalModels:
    """Multimodal model configuration."""

    image: str = "Z-Image-Turbo"
    tts: str = "Qwen3-tts-customvoice"
    asr: str = "Qwen3-ASR"


@dataclass
class ModelsConfig:
    """Model configuration."""

    default_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    deploy_timeout: int = 600
    inference_timeout: int = 60
    multimodal: MultimodalModels = field(default_factory=MultimodalModels)


@dataclass
class DigitalOceanConfig:
    """DigitalOcean worker configuration."""

    enabled: bool = False
    api_token: str = ""
    region: str = "nyc1"
    size: str = "g-2vcpu-8gb"


@dataclass
class KubernetesConfig:
    """Kubernetes worker configuration."""

    enabled: bool = False
    kubeconfig: str = "~/.kube/config"
    namespace: str = "gpustack"


@dataclass
class WorkerConfig:
    """Worker configuration."""

    digitalocean: DigitalOceanConfig = field(default_factory=DigitalOceanConfig)
    kubernetes: KubernetesConfig = field(default_factory=KubernetesConfig)


@dataclass
class ProviderConfig:
    """Single provider configuration."""

    enabled: bool = False
    api_key: str = ""
    endpoint: str = ""


@dataclass
class ProvidersConfig:
    """All providers configuration."""

    doubao: ProviderConfig = field(default_factory=ProviderConfig)
    qwen: ProviderConfig = field(default_factory=ProviderConfig)
    openai: ProviderConfig = field(default_factory=ProviderConfig)


@dataclass
class UpgradeConfig:
    """Upgrade test configuration."""

    from_version: str = "2.0.3"
    to_version: str = "2.1.2rc"
    from_image: str = "gpustack/gpustack:v2.0.3"
    to_image: str = "gpustack/gpustack:v2.1.2rc"


@dataclass
class TestBehaviorConfig:
    """Test behavior configuration."""

    cleanup: bool = True
    retry_count: int = 2
    retry_interval: int = 10
    artifacts_dir: str = "./e2e-artifacts"
    save_logs_on_failure: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    grafana_url: str = ""
    prometheus_url: str = ""


@dataclass
class E2EConfig:
    """E2E test complete configuration."""

    server: ServerConfig = field(default_factory=ServerConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    upgrade: UpgradeConfig = field(default_factory=UpgradeConfig)
    test: TestBehaviorConfig = field(default_factory=TestBehaviorConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert dict to dataclass."""
    if data is None:
        return cls()

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key not in field_types:
            continue

        field_type = field_types[key]

        # Handle nested dataclasses
        if hasattr(field_type, "__dataclass_fields__"):
            kwargs[key] = _dict_to_dataclass(field_type, value or {})
        else:
            kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> E2EConfig:
    """
    Load E2E test configuration.

    Priority:
    1. Environment variables (highest)
    2. Config file
    3. Default values (lowest)

    Environment variables:
    - GPUSTACK_SERVER_URL: Server URL
    - GPUSTACK_ADMIN_PASSWORD: Admin password
    - GPUSTACK_API_KEY: API key
    - E2E_DOCKER_IMAGE: Docker image for deployment tests
    - E2E_GPU_TYPE: GPU type (nvidia, amd, ascend, cpu)
    - E2E_SKIP_CLEANUP: Skip cleanup after tests
    - E2E_CONFIG_FILE: Config file path
    """
    # Determine config file path
    if config_path is None:
        config_path = os.environ.get("E2E_CONFIG_FILE")

    if config_path is None:
        # Default config file paths
        default_paths = [
            Path(__file__).parent.parent / "config.yaml",
            Path.cwd() / "e2e" / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        for path in default_paths:
            if path.exists():
                config_path = str(path)
                break

    # Load config file
    config_data = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    # Build config object
    config = _dict_to_dataclass(E2EConfig, config_data)

    # Environment variable overrides (highest priority)
    if os.environ.get("GPUSTACK_SERVER_URL"):
        config.server.url = os.environ["GPUSTACK_SERVER_URL"]
    if os.environ.get("GPUSTACK_ADMIN_PASSWORD"):
        config.server.admin_password = os.environ["GPUSTACK_ADMIN_PASSWORD"]
        config.docker.bootstrap_password = os.environ["GPUSTACK_ADMIN_PASSWORD"]
    if os.environ.get("GPUSTACK_API_KEY"):
        config.server.api_key = os.environ["GPUSTACK_API_KEY"]
    if os.environ.get("E2E_DOCKER_IMAGE"):
        config.docker.image = os.environ["E2E_DOCKER_IMAGE"]
    if os.environ.get("E2E_GPU_TYPE"):
        config.gpu.type = os.environ["E2E_GPU_TYPE"]
    if os.environ.get("E2E_SKIP_CLEANUP", "").lower() in ("true", "1", "yes"):
        config.test.cleanup = False

    return config
