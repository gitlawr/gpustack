"""E2E test pytest configuration and fixtures."""

import logging
import sys
from pathlib import Path
from typing import Generator

import pytest

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from e2e.utils.config import E2EConfig, load_config  # noqa: E402
from e2e.utils.client import GPUStackClient  # noqa: E402
from e2e.utils.docker import DockerManager  # noqa: E402
from e2e.utils.models import ModelHelper  # noqa: E402
from e2e.utils.k8s import K8sManager  # noqa: E402


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--e2e-config",
        action="store",
        default=None,
        help="Path to E2E config file",
    )
    parser.addoption(
        "--gpustack-url",
        action="store",
        default=None,
        help="GPUStack server URL",
    )
    parser.addoption(
        "--admin-password",
        action="store",
        default=None,
        help="GPUStack admin password",
    )
    parser.addoption(
        "--api-key",
        action="store",
        default=None,
        help="GPUStack API key",
    )
    parser.addoption(
        "--gpu-type",
        action="store",
        default=None,
        choices=["nvidia", "amd", "ascend", "cpu"],
        help="GPU type for testing",
    )
    parser.addoption(
        "--skip-cleanup",
        action="store_true",
        default=False,
        help="Skip cleanup after tests",
    )


def pytest_configure(config):
    """pytest configuration hook."""
    # Register custom markers
    config.addinivalue_line("markers", "nvidia: NVIDIA GPU tests")
    config.addinivalue_line("markers", "amd: AMD GPU tests")
    config.addinivalue_line("markers", "ascend: Huawei Ascend NPU tests")
    config.addinivalue_line("markers", "wsl: Windows WSL tests")
    config.addinivalue_line("markers", "cpu_only: CPU only tests")
    config.addinivalue_line("markers", "allinone: All-in-one deployment tests")
    config.addinivalue_line("markers", "server_only: Server-only deployment tests")
    config.addinivalue_line("markers", "distributed: Distributed deployment tests")
    config.addinivalue_line("markers", "installation: Installation tests")
    config.addinivalue_line("markers", "model: Model deployment tests")
    config.addinivalue_line("markers", "worker: Worker management tests")
    config.addinivalue_line("markers", "upgrade: Upgrade tests")
    config.addinivalue_line("markers", "provider: Provider tests")
    config.addinivalue_line("markers", "route: Route tests")
    config.addinivalue_line("markers", "stability: Stability tests")
    config.addinivalue_line("markers", "multimodal: Multimodal model tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests")
    config.addinivalue_line("markers", "monitoring: Monitoring tests")
    config.addinivalue_line("markers", "vllm: vLLM backend tests")
    config.addinivalue_line("markers", "sglang: SGLang backend tests")
    config.addinivalue_line("markers", "mindie: MindIE backend tests")
    config.addinivalue_line("markers", "gguf: GGUF/llama.cpp backend tests")
    config.addinivalue_line("markers", "custom_backend: Custom backend tests")
    config.addinivalue_line("markers", "do_worker: DigitalOcean worker tests")
    config.addinivalue_line("markers", "k8s_worker: Kubernetes worker tests")
    config.addinivalue_line("markers", "smoke: Smoke tests")
    config.addinivalue_line("markers", "regression: Regression tests")
    config.addinivalue_line("markers", "slow: Slow tests")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on GPU type."""
    gpu_type = config.getoption("--gpu-type")
    if not gpu_type:
        return

    # GPU type marker mapping
    gpu_markers = {
        "nvidia": ["nvidia", "wsl"],  # NVIDIA also supports WSL tests
        "amd": ["amd"],
        "ascend": ["ascend"],
        "cpu": ["cpu_only"],
    }

    allowed_markers = gpu_markers.get(gpu_type, [])

    skip_reason = pytest.mark.skip(reason=f"Not applicable for GPU type: {gpu_type}")

    for item in items:
        # Check if test has GPU type markers
        item_markers = [m.name for m in item.iter_markers()]

        # GPU specific markers
        gpu_specific_markers = ["nvidia", "amd", "ascend", "wsl", "cpu_only"]
        has_gpu_marker = any(m in item_markers for m in gpu_specific_markers)

        if has_gpu_marker:
            # Check if any allowed marker is present
            if not any(m in allowed_markers for m in item_markers):
                item.add_marker(skip_reason)


@pytest.fixture(scope="session")
def e2e_config(request) -> E2EConfig:
    """E2E test configuration."""
    config_path = request.config.getoption("--e2e-config")
    config = load_config(config_path)

    # Command line option overrides
    if request.config.getoption("--gpustack-url"):
        config.server.url = request.config.getoption("--gpustack-url")
    if request.config.getoption("--admin-password"):
        config.server.admin_password = request.config.getoption("--admin-password")
    if request.config.getoption("--api-key"):
        config.server.api_key = request.config.getoption("--api-key")
    if request.config.getoption("--gpu-type"):
        config.gpu.type = request.config.getoption("--gpu-type")
    if request.config.getoption("--skip-cleanup"):
        config.test.cleanup = False

    return config


@pytest.fixture(scope="session")
def gpustack_client(e2e_config: E2EConfig) -> Generator[GPUStackClient, None, None]:
    """GPUStack API client."""
    client = GPUStackClient(
        base_url=e2e_config.server.url,
        api_key=e2e_config.server.api_key,
        admin_password=e2e_config.server.admin_password,
        timeout=e2e_config.server.timeout,
        verify_ssl=e2e_config.server.verify_ssl,
    )

    yield client

    client.close()


@pytest.fixture(scope="session")
def docker_manager(e2e_config: E2EConfig) -> DockerManager:
    """Docker manager."""
    return DockerManager(
        image=e2e_config.docker.image,
        container_prefix=e2e_config.docker.container_prefix,
        cache_dir=e2e_config.docker.cache_dir,
        runtime=e2e_config.docker.runtime,
        use_sudo=e2e_config.docker.use_sudo,
    )


@pytest.fixture(scope="session")
def model_helper(gpustack_client: GPUStackClient) -> ModelHelper:
    """Model operation helper."""
    return ModelHelper(gpustack_client)


@pytest.fixture(scope="session")
def k8s_manager(e2e_config: E2EConfig) -> K8sManager:
    """Kubernetes manager."""
    return K8sManager(
        kubeconfig=e2e_config.worker.kubernetes.kubeconfig,
        namespace=e2e_config.worker.kubernetes.namespace,
    )


@pytest.fixture(scope="function")
def cleanup_models(gpustack_client: GPUStackClient, e2e_config: E2EConfig):
    """
    Clean up models after test.

    Usage:
        def test_xxx(cleanup_models):
            cleanup_models.append(model_id)
    """
    model_ids = []

    yield model_ids

    if e2e_config.test.cleanup:
        for model_id in model_ids:
            try:
                gpustack_client.delete_model(model_id)
                logger.info(f"Cleaned up model: {model_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup model {model_id}: {e}")


@pytest.fixture(scope="function")
def cleanup_api_keys(gpustack_client: GPUStackClient, e2e_config: E2EConfig):
    """Clean up API keys after test."""
    key_ids = []

    yield key_ids

    if e2e_config.test.cleanup:
        for key_id in key_ids:
            try:
                gpustack_client.delete_api_key(key_id)
                logger.info(f"Cleaned up API key: {key_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup API key {key_id}: {e}")


@pytest.fixture(scope="function")
def cleanup_providers(gpustack_client: GPUStackClient, e2e_config: E2EConfig):
    """Clean up providers after test."""
    provider_ids = []

    yield provider_ids

    if e2e_config.test.cleanup:
        for provider_id in provider_ids:
            try:
                gpustack_client.delete_model_provider(provider_id)
                logger.info(f"Cleaned up provider: {provider_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup provider {provider_id}: {e}")


@pytest.fixture(scope="function")
def cleanup_routes(gpustack_client: GPUStackClient, e2e_config: E2EConfig):
    """Clean up routes after test."""
    route_ids = []

    yield route_ids

    if e2e_config.test.cleanup:
        for route_id in route_ids:
            try:
                gpustack_client.delete_model_route(route_id)
                logger.info(f"Cleaned up route: {route_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup route {route_id}: {e}")


@pytest.fixture(scope="module")
def artifacts_dir(e2e_config: E2EConfig) -> Path:
    """Test artifacts directory."""
    path = Path(e2e_config.test.artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def save_logs_on_failure(
    request,
    gpustack_client: GPUStackClient,
    artifacts_dir: Path,
    e2e_config: E2EConfig,
):
    """Save logs on test failure."""
    yield

    if e2e_config.test.save_logs_on_failure and request.node.rep_call.failed:
        test_name = request.node.name.replace("/", "_").replace(":", "_")
        log_file = artifacts_dir / f"{test_name}_logs.txt"

        try:
            # Collect model instance logs
            instances = gpustack_client.list_model_instances()
            with open(log_file, "w") as f:
                for instance in instances.get("items", []):
                    f.write(
                        f"\n=== Instance {instance['id']} ({instance.get('name', 'unknown')}) ===\n"
                    )
                    try:
                        logs = gpustack_client.get_model_instance_logs(instance["id"])
                        f.write(logs)
                    except Exception as e:
                        f.write(f"Failed to get logs: {e}\n")

            logger.info(f"Saved failure logs to: {log_file}")
        except Exception as e:
            logger.warning(f"Failed to save logs: {e}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Record test result for saving logs on failure."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)
