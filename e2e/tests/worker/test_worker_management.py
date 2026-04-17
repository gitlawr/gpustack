"""
用例 3: 通过 DO 添加 Worker，部署/删除模型
用例 4: 通过 K8s 添加 Worker，部署/删除模型
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
    """DigitalOcean Worker 测试"""

    @pytest.fixture
    def do_enabled(self, e2e_config: E2EConfig):
        """检查 DO 是否启用"""
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
        """创建 DigitalOcean Cluster"""
        # 首先创建云凭证
        credential = gpustack_client.create_cloud_credential(
            name="e2e-do-credential",
            provider="DigitalOcean",
            config={
                "api_token": e2e_config.worker.digitalocean.api_token,
            },
        )

        try:
            # 创建 Cluster
            cluster = gpustack_client.create_cluster(
                name="e2e-do-cluster",
                provider="DigitalOcean",
                credential_id=credential["id"],
                region=e2e_config.worker.digitalocean.region,
            )

            assert cluster["id"] > 0
            assert cluster["provider"] == "DigitalOcean"

        finally:
            # 清理
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
        """添加 DO Worker 并部署模型"""
        # 这是一个完整的集成测试，需要真实的 DO 环境
        pytest.skip("Full DO integration test - requires manual setup")

    def test_delete_do_worker(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        do_enabled,
    ):
        """删除 DO Worker"""
        pytest.skip("Full DO integration test - requires manual setup")


@pytest.mark.worker
@pytest.mark.k8s_worker
class TestKubernetesWorker:
    """Kubernetes Worker 测试"""

    @pytest.fixture
    def k8s_enabled(self, e2e_config: E2EConfig):
        """检查 K8s 是否启用"""
        if not e2e_config.worker.kubernetes.enabled:
            pytest.skip("Kubernetes worker not enabled in config")

    def test_create_k8s_cluster(
        self,
        gpustack_client: GPUStackClient,
        k8s_manager: K8sManager,
        e2e_config: E2EConfig,
        k8s_enabled,
    ):
        """创建 Kubernetes Cluster"""
        # 获取默认 cluster
        default_cluster = gpustack_client.get_default_cluster()
        assert default_cluster is not None

        # 获取注册 token
        token_info = gpustack_client.get_registration_token(default_cluster["id"])
        assert "token" in token_info

        # 获取 K8s 部署清单
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
        """部署 K8s Worker"""
        # 获取默认 cluster
        default_cluster = gpustack_client.get_default_cluster()

        # 获取 K8s 部署清单
        manifests = gpustack_client.get_k8s_manifests(default_cluster["id"])

        # 创建命名空间
        k8s_manager.create_namespace()

        try:
            # 应用清单
            k8s_manager.apply_manifest(manifests)

            # 等待部署就绪
            k8s_manager.wait_for_deployment_ready("gpustack-worker", timeout=300)

            # 验证 Worker 注册
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
        """在 K8s Worker 上部署模型"""
        # 确保有 K8s worker
        result = gpustack_client.list_workers()
        workers = result.get("items", [])
        k8s_workers = [w for w in workers if w.get("state") == "ready"]

        if not k8s_workers:
            pytest.skip("No ready K8s workers available")

        # 部署模型
        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-k8s-model",
            backend="vLLM",
            replicas=1,
            wait=True,
            timeout=e2e_config.models.deploy_timeout,
        )

        cleanup_models.append(model["id"])

        # 验证模型推理
        response = model_helper.verify_model_inference(model_name=model["name"])
        assert response["choices"][0]["message"]["content"]

    def test_k8s_worker_model_delete(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        k8s_enabled,
    ):
        """在 K8s Worker 上删除模型"""
        # 部署模型
        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-k8s-delete",
            backend="vLLM",
            replicas=1,
            wait=True,
            timeout=e2e_config.models.deploy_timeout,
        )

        # 删除模型
        deleted = model_helper.delete_model_and_wait(model["id"])
        assert deleted, "Model deletion failed"

        # 验证模型不存在
        try:
            gpustack_client.get_model(model["id"])
            pytest.fail("Model should not exist after deletion")
        except Exception as e:
            assert "404" in str(e) or "not found" in str(e).lower()


@pytest.mark.worker
class TestWorkerOperations:
    """通用 Worker 操作测试"""

    def test_list_workers(self, gpustack_client: GPUStackClient):
        """列出所有 Workers"""
        result = gpustack_client.list_workers()

        assert "items" in result
        assert "pagination" in result

    def test_worker_status(self, gpustack_client: GPUStackClient):
        """验证 Worker 状态信息"""
        result = gpustack_client.list_workers()
        workers = result.get("items", [])

        if not workers:
            pytest.skip("No workers available")

        worker = workers[0]

        # 验证必需字段
        assert "id" in worker
        assert "name" in worker
        assert "state" in worker
        assert "status" in worker

        # 验证状态信息
        status = worker["status"]
        if status:
            assert "cpu" in status or "memory" in status

    def test_worker_maintenance_mode(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
    ):
        """测试 Worker 维护模式"""
        result = gpustack_client.list_workers()
        workers = result.get("items", [])

        if not workers:
            pytest.skip("No workers available")

        worker = workers[0]
        worker_id = worker["id"]

        try:
            # 启用维护模式
            updated = gpustack_client.set_worker_maintenance(
                worker_id,
                enabled=True,
                message="E2E test maintenance",
            )

            assert updated["maintenance"]["enabled"] is True

            # 禁用维护模式
            updated = gpustack_client.set_worker_maintenance(
                worker_id,
                enabled=False,
            )

            assert updated["maintenance"]["enabled"] is False

        finally:
            # 确保恢复
            gpustack_client.set_worker_maintenance(worker_id, enabled=False)
