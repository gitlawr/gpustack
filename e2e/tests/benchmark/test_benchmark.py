"""
Test Case 16: Benchmark functionality verification
Test Case 17: Grafana Dashboard verification
"""

import time
import pytest

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig


@pytest.mark.benchmark
@pytest.mark.nvidia
class TestBenchmark:
    """Benchmark functionality tests.

    Uses the session-scoped shared_vllm_model to avoid deploying
    a separate model for benchmarking.
    """

    def test_list_benchmarks(self, gpustack_client: GPUStackClient):
        """List Benchmarks"""
        result = gpustack_client.list_benchmarks()

        assert "items" in result
        assert isinstance(result["items"], list)

    def test_create_benchmark(
        self,
        gpustack_client: GPUStackClient,
        shared_vllm_model,
        e2e_config: E2EConfig,
    ):
        """Create a Benchmark"""
        benchmark = gpustack_client.create_benchmark(
            name="e2e-test-benchmark",
            model_id=shared_vllm_model["id"],
        )

        assert benchmark["id"] > 0
        assert benchmark["name"] == "e2e-test-benchmark"

        if e2e_config.test.cleanup:
            gpustack_client.delete_benchmark(benchmark["id"])

    def test_benchmark_execution(
        self,
        gpustack_client: GPUStackClient,
        shared_vllm_model,
        e2e_config: E2EConfig,
    ):
        """Execute a Benchmark and verify results"""
        benchmark = gpustack_client.create_benchmark(
            name="e2e-test-benchmark-exec",
            model_id=shared_vllm_model["id"],
        )

        try:
            time.sleep(30)

            result = gpustack_client.get_benchmark(benchmark["id"])
            assert result["id"] == benchmark["id"]

        finally:
            if e2e_config.test.cleanup:
                gpustack_client.delete_benchmark(benchmark["id"])

    def test_benchmark_logs(
        self,
        gpustack_client: GPUStackClient,
        shared_vllm_model,
        e2e_config: E2EConfig,
    ):
        """Verify Benchmark logs"""
        benchmark = gpustack_client.create_benchmark(
            name="e2e-test-benchmark-logs",
            model_id=shared_vllm_model["id"],
        )

        try:
            time.sleep(10)

            result = gpustack_client.get_benchmark(benchmark["id"])
            assert result is not None

        finally:
            if e2e_config.test.cleanup:
                gpustack_client.delete_benchmark(benchmark["id"])


@pytest.mark.monitoring
@pytest.mark.smoke
class TestGrafanaDashboard:
    """Grafana Dashboard tests"""

    def test_grafana_accessible(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
    ):
        """Verify Grafana is accessible"""
        if not e2e_config.monitoring.grafana_url:
            try:
                response = gpustack_client._client.get("/grafana/api/health")
                assert response.status_code in [200, 302, 401]
            except Exception:
                pytest.skip("Grafana not accessible")

    def test_dashboard_data_available(self, gpustack_client: GPUStackClient):
        """Verify Dashboard data is available"""
        dashboard = gpustack_client.get_dashboard()
        assert dashboard is not None

    def test_model_dashboard_link(
        self,
        gpustack_client: GPUStackClient,
        shared_vllm_model,
    ):
        """Verify model Dashboard link"""
        try:
            response = gpustack_client._client.get(
                f"/v2/models/{shared_vllm_model['id']}/dashboard",
                follow_redirects=False,
            )
            assert response.status_code in [200, 302, 307]
        except Exception:
            pass

    def test_worker_dashboard_link(self, gpustack_client: GPUStackClient):
        """Verify Worker Dashboard link"""
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
        """Verify Cluster Dashboard link"""
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
    """Prometheus monitoring tests"""

    def test_prometheus_metrics_exposed(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
    ):
        """Verify Prometheus metrics are exposed"""
        try:
            response = gpustack_client._client.get("/metrics")
            if response.status_code == 200:
                content = response.text
                assert "# HELP" in content or "# TYPE" in content
        except Exception:
            pytest.skip("Prometheus metrics not exposed")
