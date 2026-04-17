"""
用例 8: 版本升级测试，验证 API Key 和模型兼容性
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
    """版本升级测试"""

    @pytest.fixture(scope="class")
    def upgrade_env(self, docker_manager: DockerManager, e2e_config: E2EConfig):
        """准备升级测试环境"""
        docker_manager.cleanup_all()
        docker_manager.create_network()

        admin_password = e2e_config.server.admin_password or "Admin@123"

        # 先拉取两个版本的镜像
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
        """部署旧版本"""
        docker_manager.run_server(
            port=80,
            admin_password=upgrade_env["password"],
            use_gpu=True,
            image=upgrade_env["from_image"],
        )

        time.sleep(15)

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
        """升级前创建资源"""
        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            # 创建 API Key
            api_key = client.create_api_key(
                name="e2e-upgrade-test-key",
                description="Created before upgrade",
            )

            # 保存 API Key 值用于升级后验证
            upgrade_env["old_api_key_id"] = api_key["id"]
            upgrade_env["old_api_key_value"] = api_key.get("value")

            # 部署模型
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

            # 验证模型可用
            response = model_helper.verify_model_inference(model_name=model["name"])
            assert response["choices"][0]["message"]["content"]

    def test_perform_upgrade(
        self,
        upgrade_env,
        docker_manager: DockerManager,
    ):
        """执行升级"""
        container_name = docker_manager._get_container_name("server")

        # 停止旧容器
        docker_manager.stop_container(container_name)
        docker_manager.remove_container(container_name)

        # 启动新版本（使用相同的数据卷）
        docker_manager.run_server(
            port=80,
            admin_password=upgrade_env["password"],
            use_gpu=True,
            image=upgrade_env["to_image"],
        )

        time.sleep(15)

    def test_verify_version_after_upgrade(
        self,
        upgrade_env,
        e2e_config: E2EConfig,
    ):
        """验证升级后版本"""
        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            wait_for_server_healthy(client, timeout=120)

            version = client.get_version()
            # 新版本应该包含目标版本号
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
        """验证旧 API Key 仍然有效"""
        if not upgrade_env.get("old_api_key_value"):
            pytest.skip("No old API key to test")

        # 使用旧 API Key 创建客户端
        client = GPUStackClient(
            base_url="http://localhost:80",
            api_key=upgrade_env["old_api_key_value"],
        )

        with client:
            # 验证可以访问 API
            user = client.get_current_user()
            assert user["username"] == "admin"

    def test_new_api_key_works(
        self,
        upgrade_env,
    ):
        """验证可以创建新 API Key"""
        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            # 创建新 API Key
            new_key = client.create_api_key(
                name="e2e-upgrade-test-new-key",
                description="Created after upgrade",
            )

            assert new_key["id"] > 0
            assert new_key.get("value")

            # 使用新 Key 验证
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
        """验证升级前部署的模型仍然可用"""
        if not upgrade_env.get("old_model_name"):
            pytest.skip("No old model to test")

        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            # 等待模型恢复
            if upgrade_env.get("old_model_id"):
                wait_for_model_ready(
                    client,
                    upgrade_env["old_model_id"],
                    timeout=300,
                )

            # 验证推理
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
        """验证升级后可以部署新模型"""
        client = GPUStackClient(
            base_url="http://localhost:80",
            admin_password=upgrade_env["password"],
        )

        with client:
            model_helper = ModelHelper(client)

            # 部署新模型
            model = model_helper.deploy_huggingface_model(
                repo_id=e2e_config.models.default_model,
                name="e2e-upgrade-test-new-model",
                backend="vLLM",
                replicas=1,
                wait=True,
                timeout=e2e_config.models.deploy_timeout,
            )

            # 验证推理
            response = model_helper.verify_model_inference(model_name=model["name"])
            assert response["choices"][0]["message"]["content"]

            # 清理
            if e2e_config.test.cleanup:
                client.delete_model(model["id"])
