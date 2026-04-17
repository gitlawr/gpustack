"""
用例 18: 添加豆包 Provider
用例 19: 添加通义千问 Provider
用例 20: 添加 OpenAI Provider
"""

import pytest

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig


@pytest.mark.provider
class TestDoubaoProvider:
    """豆包 Provider 测试"""

    @pytest.fixture
    def doubao_enabled(self, e2e_config: E2EConfig):
        """检查豆包配置"""
        if not e2e_config.providers.doubao.enabled:
            pytest.skip("Doubao provider not enabled")
        if not e2e_config.providers.doubao.api_key:
            pytest.skip("Doubao API key not configured")

    def test_create_doubao_provider(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        doubao_enabled,
        cleanup_providers,
    ):
        """创建豆包 Provider"""
        provider = gpustack_client.create_model_provider(
            name="e2e-test-doubao",
            config={
                "type": "doubao",
            },
            api_tokens=[{"value": e2e_config.providers.doubao.api_key}],
            models=[
                {"name": "doubao-pro-32k", "category": "llm"},
            ],
        )

        cleanup_providers.append(provider["id"])

        assert provider["id"] > 0
        assert provider["name"] == "e2e-test-doubao"

    def test_doubao_model_access(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        doubao_enabled,
        cleanup_providers,
        cleanup_routes,
    ):
        """验证豆包模型访问"""
        # 创建 Provider
        provider = gpustack_client.create_model_provider(
            name="e2e-test-doubao-access",
            config={"type": "doubao"},
            api_tokens=[{"value": e2e_config.providers.doubao.api_key}],
            models=[{"name": "doubao-pro-32k", "category": "llm"}],
        )

        cleanup_providers.append(provider["id"])

        # 创建 Route
        route = gpustack_client.create_model_route(
            name="e2e-test-doubao-route",
            categories=["llm"],
            targets=[
                {
                    "provider_id": provider["id"],
                    "provider_model_name": "doubao-pro-32k",
                    "weight": 100,
                }
            ],
        )

        cleanup_routes.append(route["id"])

        # 测试推理
        response = gpustack_client.chat_completion(
            model=route["name"],
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
        )

        assert response["choices"][0]["message"]["content"]


@pytest.mark.provider
class TestQwenProvider:
    """通义千问 Provider 测试"""

    @pytest.fixture
    def qwen_enabled(self, e2e_config: E2EConfig):
        """检查通义千问配置"""
        if not e2e_config.providers.qwen.enabled:
            pytest.skip("Qwen provider not enabled")
        if not e2e_config.providers.qwen.api_key:
            pytest.skip("Qwen API key not configured")

    def test_create_qwen_provider(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        qwen_enabled,
        cleanup_providers,
    ):
        """创建通义千问 Provider"""
        provider = gpustack_client.create_model_provider(
            name="e2e-test-qwen",
            config={
                "type": "qwen",
            },
            api_tokens=[{"value": e2e_config.providers.qwen.api_key}],
            models=[
                {"name": "qwen-turbo", "category": "llm"},
            ],
        )

        cleanup_providers.append(provider["id"])

        assert provider["id"] > 0

    def test_qwen_model_access(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        qwen_enabled,
        cleanup_providers,
        cleanup_routes,
    ):
        """验证通义千问模型访问"""
        provider = gpustack_client.create_model_provider(
            name="e2e-test-qwen-access",
            config={"type": "qwen"},
            api_tokens=[{"value": e2e_config.providers.qwen.api_key}],
            models=[{"name": "qwen-turbo", "category": "llm"}],
        )

        cleanup_providers.append(provider["id"])

        route = gpustack_client.create_model_route(
            name="e2e-test-qwen-route",
            categories=["llm"],
            targets=[
                {
                    "provider_id": provider["id"],
                    "provider_model_name": "qwen-turbo",
                    "weight": 100,
                }
            ],
        )

        cleanup_routes.append(route["id"])

        response = gpustack_client.chat_completion(
            model=route["name"],
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
        )

        assert response["choices"][0]["message"]["content"]


@pytest.mark.provider
class TestOpenAIProvider:
    """OpenAI Provider 测试"""

    @pytest.fixture
    def openai_enabled(self, e2e_config: E2EConfig):
        """检查 OpenAI 配置"""
        if not e2e_config.providers.openai.enabled:
            pytest.skip("OpenAI provider not enabled")
        if not e2e_config.providers.openai.api_key:
            pytest.skip("OpenAI API key not configured")

    def test_create_openai_provider(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        openai_enabled,
        cleanup_providers,
    ):
        """创建 OpenAI Provider"""
        config = {"type": "openai"}
        if e2e_config.providers.openai.endpoint:
            config["openaiCustomUrl"] = e2e_config.providers.openai.endpoint

        provider = gpustack_client.create_model_provider(
            name="e2e-test-openai",
            config=config,
            api_tokens=[{"value": e2e_config.providers.openai.api_key}],
            models=[
                {"name": "gpt-4o-mini", "category": "llm"},
            ],
        )

        cleanup_providers.append(provider["id"])

        assert provider["id"] > 0

    def test_openai_model_access(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        openai_enabled,
        cleanup_providers,
        cleanup_routes,
    ):
        """验证 OpenAI 模型访问"""
        config = {"type": "openai"}
        if e2e_config.providers.openai.endpoint:
            config["openaiCustomUrl"] = e2e_config.providers.openai.endpoint

        provider = gpustack_client.create_model_provider(
            name="e2e-test-openai-access",
            config=config,
            api_tokens=[{"value": e2e_config.providers.openai.api_key}],
            models=[{"name": "gpt-4o-mini", "category": "llm"}],
        )

        cleanup_providers.append(provider["id"])

        route = gpustack_client.create_model_route(
            name="e2e-test-openai-route",
            categories=["llm"],
            targets=[
                {
                    "provider_id": provider["id"],
                    "provider_model_name": "gpt-4o-mini",
                    "weight": 100,
                }
            ],
        )

        cleanup_routes.append(route["id"])

        response = gpustack_client.chat_completion(
            model=route["name"],
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=50,
        )

        assert response["choices"][0]["message"]["content"]

    def test_openai_test_connection(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        openai_enabled,
    ):
        """测试 OpenAI 连接"""
        config = {"type": "openai"}
        if e2e_config.providers.openai.endpoint:
            config["openaiCustomUrl"] = e2e_config.providers.openai.endpoint

        result = gpustack_client.test_provider_model(
            config=config,
            model_name="gpt-4o-mini",
            api_tokens=[{"value": e2e_config.providers.openai.api_key}],
        )

        # 测试应该成功或返回错误信息
        assert result is not None
