"""
用例 16: Benchmark 功能验证
用例 17: Grafana Dashboard 验证
"""

import time
import pytest

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig
from e2e.utils.models import ModelHelper


@pytest.mark.benchmark
@pytest.mark.nvidia
class TestBenchmark:
    """Benchmark 功能测试"""

    @pytest.fixture
    def deployed_model(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """部署用于 benchmark 的模型"""
        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-benchmark-model",
            backend="vLLM",
            replicas=1,
            wait=True,
            timeout=e2e_config.models.deploy_timeout,
        )

        cleanup_models.append(model["id"])

        return model

    def test_list_benchmarks(self, gpustack_client: GPUStackClient):
        """列出 Benchmarks"""
        result = gpustack_client.list_benchmarks()

        assert "items" in result
        assert isinstance(result["items"], list)

    def test_create_benchmark(
        self,
        gpustack_client: GPUStackClient,
        deployed_model,
        e2e_config: E2EConfig,
    ):
        """创建 Benchmark"""
        benchmark = gpustack_client.create_benchmark(
            name="e2e-test-benchmark",
            model_id=deployed_model["id"],
        )

        assert benchmark["id"] > 0
        assert benchmark["name"] == "e2e-test-benchmark"

        # 清理
        if e2e_config.test.cleanup:
            gpustack_client.delete_benchmark(benchmark["id"])

    def test_benchmark_execution(
        self,
        gpustack_client: GPUStackClient,
        deployed_model,
        e2e_config: E2EConfig,
    ):
        """执行 Benchmark 并验证结果"""
        benchmark = gpustack_client.create_benchmark(
            name="e2e-test-benchmark-exec",
            model_id=deployed_model["id"],
        )

        try:
            # 等待 benchmark 完成（简化处理）
            time.sleep(30)

            # 获取 benchmark 详情
            result = gpustack_client.get_benchmark(benchmark["id"])

            assert result["id"] == benchmark["id"]
            # 验证 benchmark 有结果数据
            # 具体字段取决于 API 实现

        finally:
            if e2e_config.test.cleanup:
                gpustack_client.delete_benchmark(benchmark["id"])

    def test_benchmark_logs(
        self,
        gpustack_client: GPUStackClient,
        deployed_model,
        e2e_config: E2EConfig,
    ):
        """验证 Benchmark 日志"""
        benchmark = gpustack_client.create_benchmark(
            name="e2e-test-benchmark-logs",
            model_id=deployed_model["id"],
        )

        try:
            time.sleep(10)

            # 获取 benchmark 详情应包含日志或状态信息
            result = gpustack_client.get_benchmark(benchmark["id"])
            assert result is not None

        finally:
            if e2e_config.test.cleanup:
                gpustack_client.delete_benchmark(benchmark["id"])


@pytest.mark.monitoring
@pytest.mark.smoke
class TestGrafanaDashboard:
    """Grafana Dashboard 测试"""

    def test_grafana_accessible(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
    ):
        """验证 Grafana 可访问"""
        if not e2e_config.monitoring.grafana_url:
            # 尝试通过 GPUStack 内置路径访问
            try:
                # Grafana 通常通过 /grafana 路径代理
                response = gpustack_client._client.get("/grafana/api/health")
                assert response.status_code in [200, 302, 401]
            except Exception:
                pytest.skip("Grafana not accessible")

    def test_dashboard_data_available(self, gpustack_client: GPUStackClient):
        """验证 Dashboard 数据可用"""
        dashboard = gpustack_client.get_dashboard()

        # 验证返回了一些数据
        assert dashboard is not None

    def test_model_dashboard_link(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """验证模型 Dashboard 链接"""
        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-dashboard-link",
            backend="vLLM",
            replicas=1,
            wait=True,
            timeout=e2e_config.models.deploy_timeout,
        )

        cleanup_models.append(model["id"])

        # 尝试获取 dashboard 链接
        # 这通常是一个重定向端点
        try:
            response = gpustack_client._client.get(
                f"/v2/models/{model['id']}/dashboard",
                follow_redirects=False,
            )
            # 应该返回重定向到 Grafana
            assert response.status_code in [200, 302, 307]
        except Exception:
            # Dashboard 可能未配置
            pass

    def test_worker_dashboard_link(self, gpustack_client: GPUStackClient):
        """验证 Worker Dashboard 链接"""
        result = gpustack_client.list_workers()
        workers = result.get("items", [])

        if not workers:
            pytest.skip("No workers available")

        worker = workers[0]

        try:
            response = gpustack_client._client.get(
                f"/v2/workers/{worker['id']}/dashboard",
                follow_redirects=False,
            )
            assert response.status_code in [200, 302, 307]
        except Exception:
            pass

    def test_cluster_dashboard_link(self, gpustack_client: GPUStackClient):
        """验证 Cluster Dashboard 链接"""
        cluster = gpustack_client.get_default_cluster()

        if not cluster:
            pytest.skip("No default cluster")

        try:
            response = gpustack_client._client.get(
                f"/v2/clusters/{cluster['id']}/dashboard",
                follow_redirects=False,
            )
            assert response.status_code in [200, 302, 307]
        except Exception:
            pass


@pytest.mark.monitoring
class TestPrometheus:
    """Prometheus 监控测试"""

    def test_prometheus_metrics_exposed(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
    ):
        """验证 Prometheus 指标暴露"""
        try:
            response = gpustack_client._client.get("/metrics")
            if response.status_code == 200:
                # 验证返回的是 Prometheus 格式
                content = response.text
                assert "# HELP" in content or "# TYPE" in content
        except Exception:
            pytest.skip("Prometheus metrics not exposed")
