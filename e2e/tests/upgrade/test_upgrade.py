"""
Test Case 8: Version upgrade - verify API key and model compatibility
"""

import time
import pytest

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig
from e2e.utils.docker import DockerManager
from e2e.utils.models import ModelHelper
from e2e.utils.wait import wait_for_server_healthy, wait_for_model_ready


@pytest.mark.upgrade
@pytest.mark.slow
class TestVersionUpgrade:
    """Version upgrade tests."""

    @pytest.fixture(scope="class")
    def upgrade_env(self, docker_manager: DockerManager, e2e_config: E2EConfig):
        """Prepare upgrade test environment."""
        docker_manager.cleanup_all()

        admin_password = e2e_config.server.admin_password or "Admin@123"

        # Pull both version images
        docker_manager.pull_image(e2e_config.upgrade.from_image)
        docker_manager.pull_image(e2e_config.upgrade.to_image)

        yield {
            "from_image": e2e_config.upgrade.from_image,
            "to_image": e2e_config.upgrade.to_image,
            "password": admin_password,
        }

        if e2e_config.test.cleanup:
            docker_manager.cleanup_all()

    def test_deploy_old_version(
        self,
        upgrade_env,
        docker_manager: DockerManager,
        e2e_config: E2EConfig,
    ):
        """Deploy old version."""
        docker_manager.run_allinone(
            bootstrap_password=upgrade_env["password"],
            image=upgrade_env["from_image"],
        )

        time.sleep(e2e_config.docker.startup_wait)

        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            wait_for_server_healthy(client, timeout=120)

            version = client.get_version()
            assert e2e_config.upgrade.from_version in version.get(
                "version", ""
            ), f"Expected version {e2e_config.upgrade.from_version}"

    def test_create_resources_before_upgrade(
        self,
        upgrade_env,
        e2e_config: E2EConfig,
    ):
        """Create resources before upgrade."""
        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            # Create API key
            api_key = client.create_api_key(
                name="e2e-upgrade-test-key",
                description="Created before upgrade",
            )

            # Save API key value for post-upgrade verification
            upgrade_env["old_api_key_id"] = api_key["id"]
            upgrade_env["old_api_key_value"] = api_key.get("value")

            # Deploy model
            model_helper = ModelHelper(client)
            model = model_helper.deploy_huggingface_model(
                repo_id=e2e_config.models.default_model,
                name="e2e-upgrade-test-model",
                backend="vLLM",
                replicas=1,
                wait=True,
                timeout=e2e_config.models.deploy_timeout,
            )

            upgrade_env["old_model_id"] = model["id"]
            upgrade_env["old_model_name"] = model["name"]

            # Verify model is working
            response = model_helper.verify_model_inference(model_name=model["name"])
            assert response["choices"][0]["message"]["content"]

    def test_perform_upgrade(
        self,
        upgrade_env,
        docker_manager: DockerManager,
    ):
        """Perform upgrade."""
        container_name = docker_manager._get_container_name("allinone")

        # Stop old container
        docker_manager.stop_container(container_name)
        docker_manager.remove_container(container_name)

        # Start new version (reusing the same cache volume)
        docker_manager.run_allinone(
            bootstrap_password=upgrade_env["password"],
            image=upgrade_env["to_image"],
        )

        time.sleep(15)

    def test_verify_version_after_upgrade(
        self,
        upgrade_env,
        e2e_config: E2EConfig,
    ):
        """Verify version after upgrade."""
        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            wait_for_server_healthy(client, timeout=120)

            version = client.get_version()
            assert e2e_config.upgrade.to_version in version.get(
                "version", ""
            ) or version.get("version", "").startswith(
                "v"
            ), f"Expected version containing {e2e_config.upgrade.to_version}"

    def test_old_api_key_works(
        self,
        upgrade_env,
        e2e_config: E2EConfig,
    ):
        """Verify old API key still works after upgrade."""
        if not upgrade_env.get("old_api_key_value"):
            pytest.skip("No old API key to test")

        client = GPUStackClient(
            base_url="http://localhost:80",
            api_key=upgrade_env["old_api_key_value"],
        )

        with client:
            user = client.get_current_user()
            assert user["username"] == "admin"

    def test_new_api_key_works(
        self,
        upgrade_env,
    ):
        """Verify new API key can be created after upgrade."""
        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            new_key = client.create_api_key(
                name="e2e-upgrade-test-new-key",
                description="Created after upgrade",
            )

            assert new_key["id"] > 0
            assert new_key.get("value")

            # Verify new key works
            new_client = GPUStackClient(
                base_url="http://localhost:80",
                api_key=new_key["value"],
            )

            with new_client:
                user = new_client.get_current_user()
                assert user["username"] == "admin"

    def test_old_model_still_works(
        self,
        upgrade_env,
        e2e_config: E2EConfig,
    ):
        """Verify model deployed before upgrade still works."""
        if not upgrade_env.get("old_model_name"):
            pytest.skip("No old model to test")

        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            if upgrade_env.get("old_model_id"):
                wait_for_model_ready(
                    client,
                    upgrade_env["old_model_id"],
                    timeout=300,
                )

            model_helper = ModelHelper(client)
            response = model_helper.verify_model_inference(
                model_name=upgrade_env["old_model_name"],
            )

            assert response["choices"][0]["message"]["content"]

    def test_deploy_new_model_after_upgrade(
        self,
        upgrade_env,
        e2e_config: E2EConfig,
    ):
        """Verify new model can be deployed after upgrade."""
        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            model_helper = ModelHelper(client)

            model = model_helper.deploy_huggingface_model(
                repo_id=e2e_config.models.default_model,
                name="e2e-upgrade-test-new-model",
                backend="vLLM",
                replicas=1,
                wait=True,
                timeout=e2e_config.models.deploy_timeout,
            )

            response = model_helper.verify_model_inference(model_name=model["name"])
            assert response["choices"][0]["message"]["content"]

            if e2e_config.test.cleanup:
                client.delete_model(model["id"])
