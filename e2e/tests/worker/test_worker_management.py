"""
Test Case 3: Add Worker via DigitalOcean, deploy/delete model
Test Case 4: Add Worker via K8s, deploy/delete model
"""

import time
import pytest

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig
from e2e.utils.models import ModelHelper
from e2e.utils.k8s import K8sManager


@pytest.mark.worker
@pytest.mark.do_worker
class TestDigitalOceanWorker:
    """DigitalOcean Worker tests"""

    @pytest.fixture
    def do_enabled(self, e2e_config: E2EConfig):
        """Check if DigitalOcean is enabled"""
        if not e2e_config.worker.digitalocean.enabled:
            pytest.skip("DigitalOcean worker not enabled in config")
        if not e2e_config.worker.digitalocean.api_token:
            pytest.skip("DigitalOcean API token not configured")

    def test_create_do_cluster(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        do_enabled,
    ):
        """Create a DigitalOcean Cluster"""
        # First create cloud credential
        credential = gpustack_client.create_cloud_credential(
            name="e2e-do-credential",
            provider="DigitalOcean",
            config={
                "api_token": e2e_config.worker.digitalocean.api_token,
            },
        )

        try:
            # Create Cluster
            cluster = gpustack_client.create_cluster(
                name="e2e-do-cluster",
                provider="DigitalOcean",
                credential_id=credential["id"],
                region=e2e_config.worker.digitalocean.region,
            )

            assert cluster["id"] > 0
            assert cluster["provider"] == "DigitalOcean"

        finally:
            # Cleanup
            if e2e_config.test.cleanup:
                try:
                    gpustack_client.delete_cluster(cluster["id"])
                except Exception:
                    pass
                gpustack_client.delete_cloud_credential(credential["id"])

    def test_add_do_worker_and_deploy_model(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        do_enabled,
        cleanup_models,
    ):
        """Add a DO Worker and deploy a model"""
        # This is a full integration test that requires a real DO environment
        pytest.skip("Full DO integration test - requires manual setup")

    def test_delete_do_worker(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        do_enabled,
    ):
        """Delete a DO Worker"""
        pytest.skip("Full DO integration test - requires manual setup")


@pytest.mark.worker
@pytest.mark.k8s_worker
class TestKubernetesWorker:
    """Kubernetes Worker tests"""

    @pytest.fixture
    def k8s_enabled(self, e2e_config: E2EConfig):
        """Check if K8s is enabled"""
        if not e2e_config.worker.kubernetes.enabled:
            pytest.skip("Kubernetes worker not enabled in config")

    def test_create_k8s_cluster(
        self,
        gpustack_client: GPUStackClient,
        k8s_manager: K8sManager,
        e2e_config: E2EConfig,
        k8s_enabled,
    ):
        """Create a Kubernetes Cluster"""
        # Get the default cluster
        default_cluster = gpustack_client.get_default_cluster()
        assert default_cluster is not None

        # Get registration token
        token_info = gpustack_client.get_registration_token(default_cluster["id"])
        assert "token" in token_info

        # Get K8s deployment manifests
        manifests = gpustack_client.get_k8s_manifests(default_cluster["id"])
        assert manifests, "Empty K8s manifests"
        assert "kind:" in manifests.lower(), "Invalid K8s manifest format"

    def test_deploy_k8s_worker(
        self,
        gpustack_client: GPUStackClient,
        k8s_manager: K8sManager,
        e2e_config: E2EConfig,
        k8s_enabled,
    ):
        """Deploy a K8s Worker"""
        # Get the default cluster
        default_cluster = gpustack_client.get_default_cluster()

        # Get K8s deployment manifests
        manifests = gpustack_client.get_k8s_manifests(default_cluster["id"])

        # Create namespace
        k8s_manager.create_namespace()

        try:
            # Apply manifests
            k8s_manager.apply_manifest(manifests)

            # Wait for deployment to be ready
            k8s_manager.wait_for_deployment_ready("gpustack-worker", timeout=300)

            # Verify Worker registration
            time.sleep(30)
            result = gpustack_client.list_workers()
            workers = result.get("items", [])

            k8s_workers = [w for w in workers if w.get("provider") == "Kubernetes"]
            assert len(k8s_workers) > 0, "No K8s worker registered"

        finally:
            if e2e_config.test.cleanup:
                k8s_manager.cleanup_all()

    def test_k8s_worker_model_deploy(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        k8s_manager: K8sManager,
        e2e_config: E2EConfig,
        k8s_enabled,
        cleanup_models,
    ):
        """Deploy a model on a K8s Worker"""
        # Ensure there is a K8s worker
        result = gpustack_client.list_workers()
        workers = result.get("items", [])
        k8s_workers = [w for w in workers if w.get("state") == "ready"]

        if not k8s_workers:
            pytest.skip("No ready K8s workers available")

        # Deploy model
        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-k8s-model",
            backend="vLLM",
            replicas=1,
            wait=True,
            timeout=e2e_config.models.deploy_timeout,
        )

        cleanup_models.append(model["id"])

        # Verify model inference
        response = model_helper.verify_model_inference(model_name=model["name"])
        assert response["choices"][0]["message"]["content"]

    def test_k8s_worker_model_delete(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        k8s_enabled,
    ):
        """Delete a model on a K8s Worker"""
        # Deploy model
        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-k8s-delete",
            backend="vLLM",
            replicas=1,
            wait=True,
            timeout=e2e_config.models.deploy_timeout,
        )

        # Delete model
        deleted = model_helper.delete_model_and_wait(model["id"])
        assert deleted, "Model deletion failed"

        # Verify model no longer exists
        try:
            gpustack_client.get_model(model["id"])
            pytest.fail("Model should not exist after deletion")
        except Exception as e:
            assert "404" in str(e) or "not found" in str(e).lower()


@pytest.mark.worker
class TestWorkerOperations:
    """General Worker operations tests"""

    def test_list_workers(self, gpustack_client: GPUStackClient):
        """List all Workers"""
        result = gpustack_client.list_workers()

        assert "items" in result
        assert "pagination" in result

    def test_worker_status(self, gpustack_client: GPUStackClient):
        """Verify Worker status information"""
        result = gpustack_client.list_workers()
        workers = result.get("items", [])

        if not workers:
            pytest.skip("No workers available")

        worker = workers[0]

        # Verify required fields
        assert "id" in worker
        assert "name" in worker
        assert "state" in worker
        assert "status" in worker

        # Verify status information
        status = worker["status"]
        if status:
            assert "cpu" in status or "memory" in status

    def test_worker_maintenance_mode(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
    ):
        """Test Worker maintenance mode"""
        result = gpustack_client.list_workers()
        workers = result.get("items", [])

        if not workers:
            pytest.skip("No workers available")

        worker = workers[0]
        worker_id = worker["id"]

        try:
            # Enable maintenance mode
            updated = gpustack_client.set_worker_maintenance(
                worker_id,
                enabled=True,
                message="E2E test maintenance",
            )

            assert updated["maintenance"]["enabled"] is True

            # Disable maintenance mode
            updated = gpustack_client.set_worker_maintenance(
                worker_id,
                enabled=False,
            )

            assert updated["maintenance"]["enabled"] is False

        finally:
            # Ensure recovery
            gpustack_client.set_worker_maintenance(worker_id, enabled=False)
