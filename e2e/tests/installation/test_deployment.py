"""
Test Case 2: First installation - all-in-one/server-only deployment
Test Case 5: Windows WSL installation
Test Case 6: AMD GPU installation
Test Case 7: Ascend GPU installation

These tests deploy GPUStack via Docker and verify the deployment process.
Run with: pytest e2e/tests/installation/test_deployment.py -v
"""

import logging
import time
import pytest

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig
from e2e.utils.docker import DockerManager
from e2e.utils.wait import wait_for_server_healthy, wait_for_any_worker_ready
from e2e.utils.models import ModelHelper

logger = logging.getLogger(__name__)


def _teardown_container(
    docker_manager: DockerManager,
    container_name: str,
    cleanup: bool,
):
    """Teardown helper: dump logs, then cleanup if enabled."""
    try:
        logs = docker_manager.get_container_logs(container_name, tail=100)
        logger.info(
            "=== Container logs (%s) ===\n%s\n=== End logs ===",
            container_name,
            logs,
        )
    except Exception as e:
        logger.warning("Failed to get container logs for %s: %s", container_name, e)

    if cleanup:
        docker_manager.cleanup_all()
    else:
        logger.info(
            "Skipping cleanup — container '%s' kept for debugging. "
            "Run 'docker rm -f %s' to clean up manually.",
            container_name,
            container_name,
        )


@pytest.mark.installation
@pytest.mark.allinone
@pytest.mark.nvidia
class TestAllinoneDeployment:
    """All-in-one deployment tests (NVIDIA).

    This test class deploys GPUStack in all-in-one mode and verifies:
    - Container starts successfully
    - Server is healthy
    - Worker auto-registers
    - GPU is detected
    """

    @pytest.fixture(scope="class")
    def deployment(self, docker_manager: DockerManager, e2e_config: E2EConfig):
        """Deploy GPUStack in all-in-one mode."""
        docker_manager.cleanup_all()

        password = e2e_config.docker.bootstrap_password
        container_name = docker_manager._get_container_name("allinone")

        container_id = docker_manager.run_allinone(
            bootstrap_password=password,
            debug=True,
            disable_update_check=True,
        )

        time.sleep(e2e_config.docker.startup_wait)

        server_url = docker_manager.get_server_url()

        yield {
            "container_id": container_id,
            "container_name": container_name,
            "server_url": server_url,
            "password": password,
        }

        _teardown_container(docker_manager, container_name, e2e_config.test.cleanup)

    def test_container_running(self, deployment, docker_manager: DockerManager):
        """Verify container is running."""
        assert docker_manager.is_container_running(
            deployment["container_name"]
        ), f"Container {deployment['container_name']} is not running"

    def test_server_healthy(self, deployment):
        """Verify server is healthy and ready."""
        client = GPUStackClient(
            base_url=deployment["server_url"],
            admin_password=deployment["password"],
        )

        with client:
            wait_for_server_healthy(client, timeout=120)

            version = client.get_version()
            assert "version" in version

    def test_worker_auto_registered(self, deployment):
        """Verify worker auto-registers in all-in-one mode."""
        client = GPUStackClient(
            base_url=deployment["server_url"],
            admin_password=deployment["password"],
        )

        with client:
            worker = wait_for_any_worker_ready(client, timeout=120)
            assert worker, "No worker registered in all-in-one mode"

    def test_nvidia_gpu_detected(self, deployment):
        """Verify NVIDIA GPU is detected."""
        client = GPUStackClient(
            base_url=deployment["server_url"],
            admin_password=deployment["password"],
        )

        with client:
            wait_for_any_worker_ready(client, timeout=120)

            result = client.list_workers()
            workers = result.get("items", [])

            assert len(workers) > 0, "No workers found"

            worker = workers[0]
            status = worker.get("status", {})
            gpu_devices = status.get("gpu_devices", [])

            assert len(gpu_devices) > 0, "No GPU detected on worker"

            gpu = gpu_devices[0]
            assert (
                gpu.get("vendor") == "nvidia"
            ), f"Expected nvidia GPU, got {gpu.get('vendor')}"


@pytest.mark.installation
@pytest.mark.server_only
class TestServerOnlyDeployment:
    """Server-only deployment tests.

    This test class deploys GPUStack in server-only mode and verifies:
    - Container starts successfully (without GPU)
    - Server is healthy
    - No worker is auto-registered
    """

    @pytest.fixture(scope="class")
    def deployment(self, docker_manager: DockerManager, e2e_config: E2EConfig):
        """Deploy GPUStack in server-only mode."""
        docker_manager.cleanup_all()

        password = e2e_config.docker.bootstrap_password
        container_name = docker_manager._get_container_name("server")

        container_id = docker_manager.run_server_only(
            bootstrap_password=password,
            debug=True,
            disable_update_check=True,
        )

        time.sleep(e2e_config.docker.startup_wait)

        server_url = docker_manager.get_server_url()

        yield {
            "container_id": container_id,
            "container_name": container_name,
            "server_url": server_url,
            "password": password,
        }

        _teardown_container(docker_manager, container_name, e2e_config.test.cleanup)

    def test_container_running(self, deployment, docker_manager: DockerManager):
        """Verify container is running."""
        assert docker_manager.is_container_running(deployment["container_name"])

    def test_server_healthy(self, deployment):
        """Verify server is healthy."""
        client = GPUStackClient(
            base_url=deployment["server_url"],
            admin_password=deployment["password"],
        )

        with client:
            wait_for_server_healthy(client, timeout=120)

    def test_no_worker_registered(self, deployment):
        """Verify no worker in server-only mode."""
        client = GPUStackClient(
            base_url=deployment["server_url"],
            admin_password=deployment["password"],
        )

        with client:
            time.sleep(10)

            result = client.list_workers()
            workers = result.get("items", [])

            ready_workers = [w for w in workers if w.get("state") == "ready"]
            assert (
                len(ready_workers) == 0
            ), "Expected no ready workers in server-only mode"


@pytest.mark.installation
@pytest.mark.allinone
@pytest.mark.nvidia
class TestModelDeployAfterInstall:
    """Test model deployment after fresh installation."""

    @pytest.fixture(scope="class")
    def deployment(self, docker_manager: DockerManager, e2e_config: E2EConfig):
        """Deploy GPUStack and prepare for model deployment test."""
        docker_manager.cleanup_all()

        password = e2e_config.docker.bootstrap_password
        container_name = docker_manager._get_container_name("allinone")

        docker_manager.run_allinone(
            bootstrap_password=password,
        )

        time.sleep(e2e_config.docker.startup_wait)

        server_url = docker_manager.get_server_url()

        # Wait for server and worker to be ready
        client = GPUStackClient(base_url=server_url, admin_password=password)
        with client:
            wait_for_server_healthy(client, timeout=120)
            wait_for_any_worker_ready(client, timeout=120)

        yield {
            "server_url": server_url,
            "password": password,
        }

        _teardown_container(docker_manager, container_name, e2e_config.test.cleanup)

    def test_deploy_model_from_catalog(self, deployment, e2e_config: E2EConfig):
        """Deploy model from catalog after fresh installation."""
        client = GPUStackClient(
            base_url=deployment["server_url"],
            admin_password=deployment["password"],
        )

        with client:
            helper = ModelHelper(client)

            model = helper.deploy_huggingface_model(
                repo_id=e2e_config.models.default_model,
                name="e2e-install-test-model",
                backend="vLLM",
                replicas=1,
                wait=True,
                timeout=e2e_config.models.deploy_timeout,
            )

            assert model["ready_replicas"] >= 1, "Model not ready"

            response = helper.verify_model_inference(
                model_name=model["name"],
                prompt="Hello",
            )
            assert response["choices"][0]["message"]["content"]

            if e2e_config.test.cleanup:
                client.delete_model(model["id"])


# ============================================================================
# Platform-specific deployment tests
# ============================================================================


@pytest.mark.installation
@pytest.mark.wsl
@pytest.mark.nvidia
class TestWSLDeployment:
    """Windows WSL deployment tests."""

    @pytest.mark.skip(reason="WSL tests require Windows environment - run manually")
    def test_wsl_allinone_deployment(self):
        """Test all-in-one deployment on WSL."""
        pass

    @pytest.mark.skip(reason="WSL tests require Windows environment - run manually")
    def test_wsl_gpu_passthrough(self):
        """Verify GPU passthrough works on WSL."""
        pass


@pytest.mark.installation
@pytest.mark.amd
class TestAMDDeployment:
    """AMD GPU deployment tests.

    Run on AMD GPU server with:
    pytest e2e/tests/installation/test_deployment.py -v -m amd --gpu-type amd
    """

    def test_server_healthy(self, gpustack_client: GPUStackClient):
        """Verify server is healthy on AMD environment."""
        assert gpustack_client.health_check()
        assert gpustack_client.ready_check()

    def test_amd_gpu_detected(self, gpustack_client: GPUStackClient):
        """Verify AMD GPU is detected."""
        result = gpustack_client.list_workers()
        workers = result.get("items", [])

        assert len(workers) > 0, "No workers found"

        found_amd = False
        for worker in workers:
            status = worker.get("status", {})
            for gpu in status.get("gpu_devices", []):
                if gpu.get("vendor") == "amd":
                    found_amd = True
                    break

        assert found_amd, "No AMD GPU detected"


@pytest.mark.installation
@pytest.mark.ascend
class TestAscendDeployment:
    """Huawei Ascend NPU deployment tests.

    Run on Ascend server with:
    pytest e2e/tests/installation/test_deployment.py -v -m ascend --gpu-type ascend
    """

    def test_server_healthy(self, gpustack_client: GPUStackClient):
        """Verify server is healthy on Ascend environment."""
        assert gpustack_client.health_check()
        assert gpustack_client.ready_check()

    def test_ascend_npu_detected(self, gpustack_client: GPUStackClient):
        """Verify Ascend NPU is detected."""
        result = gpustack_client.list_workers()
        workers = result.get("items", [])

        assert len(workers) > 0, "No workers found"

        found_ascend = False
        for worker in workers:
            status = worker.get("status", {})
            for gpu in status.get("gpu_devices", []):
                if gpu.get("vendor") in ["huawei", "ascend"]:
                    found_ascend = True
                    break

        assert found_ascend, "No Ascend NPU detected"
