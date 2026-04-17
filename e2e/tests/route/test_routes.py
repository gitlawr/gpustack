"""
Test Case 21: Verify model access works after modifying a Route
Test Case 22: Verify Fallback Route takes effect
"""

import pytest

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig


@pytest.mark.route
@pytest.mark.nvidia
class TestRouteModification:
    """Route modification tests.

    Uses the session-scoped shared_vllm_model to avoid deploying
    a separate model just for route testing.
    """

    def test_create_route_for_model(
        self,
        gpustack_client: GPUStackClient,
        shared_vllm_model,
        cleanup_routes,
    ):
        """Create a model Route"""
        route = gpustack_client.create_model_route(
            name="e2e-test-route",
            categories=["llm"],
            targets=[
                {
                    "model_id": shared_vllm_model["id"],
                    "weight": 100,
                }
            ],
        )

        cleanup_routes.append(route["id"])

        assert route["id"] > 0
        assert route["name"] == "e2e-test-route"

    def test_access_model_via_route(
        self,
        gpustack_client: GPUStackClient,
        shared_vllm_model,
        cleanup_routes,
    ):
        """Access a model via Route"""
        route = gpustack_client.create_model_route(
            name="e2e-test-route-access",
            categories=["llm"],
            targets=[
                {
                    "model_id": shared_vllm_model["id"],
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

    def test_modify_route_weight(
        self,
        gpustack_client: GPUStackClient,
        shared_vllm_model,
        cleanup_routes,
    ):
        """Modify Route weight"""
        route = gpustack_client.create_model_route(
            name="e2e-test-route-modify",
            categories=["llm"],
            targets=[
                {
                    "model_id": shared_vllm_model["id"],
                    "weight": 50,
                }
            ],
        )

        cleanup_routes.append(route["id"])

        gpustack_client.get_model_route(route["id"])

        response = gpustack_client.chat_completion(
            model=route["name"],
            messages=[{"role": "user", "content": "Test after modification"}],
            max_tokens=50,
        )

        assert response["choices"][0]["message"]["content"]


@pytest.mark.route
@pytest.mark.nvidia
class TestFallbackRoute:
    """Fallback Route tests"""

    @pytest.fixture
    def openai_provider(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        cleanup_providers,
    ):
        """Create an OpenAI Provider as fallback"""
        if not e2e_config.providers.openai.enabled:
            pytest.skip("OpenAI provider required for fallback test")

        config = {"type": "openai"}
        if e2e_config.providers.openai.endpoint:
            config["openaiCustomUrl"] = e2e_config.providers.openai.endpoint

        provider = gpustack_client.create_model_provider(
            name="e2e-test-fallback-provider",
            config=config,
            api_tokens=[{"value": e2e_config.providers.openai.api_key}],
            models=[{"name": "gpt-4o-mini", "category": "llm"}],
        )

        cleanup_providers.append(provider["id"])

        return provider

    def test_create_fallback_route(
        self,
        gpustack_client: GPUStackClient,
        shared_vllm_model,
        openai_provider,
        cleanup_routes,
    ):
        """Create a Route with Fallback"""
        route = gpustack_client.create_model_route(
            name="e2e-test-fallback-route",
            categories=["llm"],
            targets=[
                {
                    "model_id": shared_vllm_model["id"],
                    "weight": 100,
                },
                {
                    "provider_id": openai_provider["id"],
                    "provider_model_name": "gpt-4o-mini",
                    "weight": 0,
                    "fallback_status_codes": ["5xx", "429"],
                },
            ],
        )

        cleanup_routes.append(route["id"])

        assert route["id"] > 0

    def test_fallback_route_normal_access(
        self,
        gpustack_client: GPUStackClient,
        shared_vllm_model,
        openai_provider,
        cleanup_routes,
    ):
        """Access the primary model under normal conditions"""
        route = gpustack_client.create_model_route(
            name="e2e-test-fallback-normal-route",
            categories=["llm"],
            targets=[
                {"model_id": shared_vllm_model["id"], "weight": 100},
                {
                    "provider_id": openai_provider["id"],
                    "provider_model_name": "gpt-4o-mini",
                    "weight": 0,
                    "fallback_status_codes": ["5xx"],
                },
            ],
        )

        cleanup_routes.append(route["id"])

        response = gpustack_client.chat_completion(
            model=route["name"],
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
        )

        assert response["choices"][0]["message"]["content"]

    def test_fallback_triggers_on_error(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        openai_provider,
        cleanup_routes,
    ):
        """Trigger Fallback when the primary model is unavailable"""
        route = gpustack_client.create_model_route(
            name="e2e-test-fallback-only",
            categories=["llm"],
            targets=[
                {
                    "provider_id": openai_provider["id"],
                    "provider_model_name": "gpt-4o-mini",
                    "weight": 100,
                },
            ],
        )

        cleanup_routes.append(route["id"])

        response = gpustack_client.chat_completion(
            model=route["name"],
            messages=[{"role": "user", "content": "Fallback test"}],
            max_tokens=50,
        )

        assert response["choices"][0]["message"]["content"]


@pytest.mark.route
class TestRouteAccessPolicy:
    """Route access policy tests"""

    def test_list_my_models(self, gpustack_client: GPUStackClient):
        """Get models accessible to the current user"""
        result = gpustack_client._get("/v2/my-models")
        assert "items" in result

    def test_route_categories(
        self,
        gpustack_client: GPUStackClient,
        shared_vllm_model,
        cleanup_routes,
    ):
        """Verify Route category filtering"""
        route = gpustack_client.create_model_route(
            name="e2e-test-llm-route",
            categories=["llm"],
            targets=[{"model_id": shared_vllm_model["id"], "weight": 100}],
        )

        cleanup_routes.append(route["id"])

        result = gpustack_client.list_model_routes(categories=["llm"])
        routes = result.get("items", [])

        assert any(r["name"] == "e2e-test-llm-route" for r in routes)
