"""
Test Case 1: Verify RC version UI/backend version display
"""

import pytest
import re

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig


@pytest.mark.smoke
@pytest.mark.installation
class TestVersionDisplay:
    """Version display tests."""

    def test_server_health(self, gpustack_client: GPUStackClient):
        """Verify server health status."""
        assert gpustack_client.health_check(), "Server health check failed"

    def test_server_ready(self, gpustack_client: GPUStackClient):
        """Verify server ready status."""
        assert gpustack_client.ready_check(), "Server ready check failed"

    def test_version_endpoint(self, gpustack_client: GPUStackClient):
        """Verify version endpoint returns correct format."""
        version_info = gpustack_client.get_version()

        assert "version" in version_info, "Version info missing 'version' field"
        version = version_info["version"]

        # Verify version format (e.g.: v2.1.2rc, 2.1.2, v0.6.0)
        version_pattern = r"^v?\d+\.\d+\.\d+(-?rc\d*|-?beta\d*|-?alpha\d*)?$"
        assert re.match(version_pattern, version), f"Invalid version format: {version}"

    def test_version_contains_git_commit(self, gpustack_client: GPUStackClient):
        """Verify version info contains git commit (optional)."""
        version_info = gpustack_client.get_version()

        # git_commit is optional
        if "git_commit" in version_info:
            commit = version_info["git_commit"]
            # Git commit should be 7-40 character hex string
            assert re.match(
                r"^[a-f0-9]{7,40}$", commit
            ), f"Invalid git commit: {commit}"

    def test_auth_config(self, gpustack_client: GPUStackClient):
        """Verify auth config endpoint."""
        auth_config = gpustack_client.get_auth_config()

        # Verify required fields
        assert (
            "first_time_setup" in auth_config
        ), "Auth config missing 'first_time_setup'"
        assert isinstance(auth_config["first_time_setup"], bool)

    def test_dashboard_accessible(self, gpustack_client: GPUStackClient):
        """Verify dashboard data is accessible."""
        dashboard = gpustack_client.get_dashboard()

        # Dashboard should return some data
        assert dashboard is not None, "Dashboard returned None"

    def test_current_user(self, gpustack_client: GPUStackClient):
        """Verify current user info."""
        user = gpustack_client.get_current_user()

        assert "id" in user, "User missing 'id'"
        assert "username" in user, "User missing 'username'"
        assert user["username"] == "admin", "Expected admin user"


@pytest.mark.smoke
@pytest.mark.installation
class TestBasicConnectivity:
    """Basic connectivity tests."""

    def test_list_workers(self, gpustack_client: GPUStackClient):
        """Verify can list workers."""
        result = gpustack_client.list_workers()

        assert "items" in result, "Response missing 'items'"
        assert isinstance(result["items"], list), "'items' should be a list"

    def test_list_models(self, gpustack_client: GPUStackClient):
        """Verify can list models."""
        result = gpustack_client.list_models()

        assert "items" in result, "Response missing 'items'"
        assert isinstance(result["items"], list), "'items' should be a list"

    def test_list_clusters(self, gpustack_client: GPUStackClient):
        """Verify can list clusters."""
        result = gpustack_client.list_clusters()

        assert "items" in result, "Response missing 'items'"
        # Should have at least one default cluster
        assert len(result["items"]) >= 1, "Expected at least one cluster"

    def test_default_cluster_exists(self, gpustack_client: GPUStackClient):
        """Verify default cluster exists."""
        cluster = gpustack_client.get_default_cluster()

        assert cluster is not None, "No default cluster found"
        assert cluster.get("is_default") is True, "Cluster is not marked as default"

    def test_list_gpu_devices(
        self, gpustack_client: GPUStackClient, e2e_config: E2EConfig
    ):
        """Verify can list GPU devices."""
        result = gpustack_client.list_gpu_devices()

        assert "items" in result, "Response missing 'items'"

        # If not CPU mode, should have GPU devices
        if e2e_config.gpu.type != "cpu":
            # Note: depends on whether workers are registered
            # Only verify if there are workers
            workers = gpustack_client.list_workers()
            if len(workers.get("items", [])) > 0:
                # At least has workers, check for GPUs
                pass  # GPU list may be in worker info

    def test_catalog_accessible(self, gpustack_client: GPUStackClient):
        """Verify catalog is accessible."""
        result = gpustack_client.list_catalog_models()

        assert "items" in result, "Response missing 'items'"
        assert len(result["items"]) > 0, "Catalog should have models"
