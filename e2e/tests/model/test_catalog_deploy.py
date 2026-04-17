"""
Test Case 2: Deploy model from catalog (vLLM/SGLang)
"""

import pytest

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig
from e2e.utils.models import ModelHelper


@pytest.mark.smoke
@pytest.mark.model
@pytest.mark.vllm
@pytest.mark.nvidia
class TestVLLMCatalogDeploy:
    """vLLM backend catalog deployment tests.

    Uses the session-scoped shared_vllm_model fixture to avoid
    redundant model deployments across tests.
    """

    def test_deploy_qwen_vllm(self, shared_vllm_model):
        """Verify vLLM model is deployed and ready."""
        assert shared_vllm_model["ready_replicas"] >= 1, "Model not ready"

    def test_vllm_model_inference(self, model_helper: ModelHelper, shared_vllm_model):
        """Verify vLLM model inference."""
        response = model_helper.verify_model_inference(
            model_name=shared_vllm_model["name"],
            prompt="What is 2+2?",
        )
        assert response["choices"][0]["message"]["content"], "Empty response"

    def test_vllm_model_streaming(
        self, gpustack_client: GPUStackClient, shared_vllm_model
    ):
        """Verify vLLM model streaming output."""
        response = gpustack_client.chat_completion(
            model=shared_vllm_model["name"],
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            stream=False,
            max_tokens=50,
        )
        assert response["choices"][0]["message"]["content"]


@pytest.mark.smoke
@pytest.mark.model
@pytest.mark.sglang
@pytest.mark.nvidia
class TestSGLangCatalogDeploy:
    """SGLang backend catalog deployment tests."""

    @pytest.fixture(scope="class")
    def shared_sglang_model(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
    ):
        """Class-scoped shared SGLang model."""
        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-qwen-sglang",
            backend="SGLang",
            replicas=1,
            wait=True,
            timeout=e2e_config.models.deploy_timeout,
        )
        yield model
        if e2e_config.test.cleanup:
            try:
                gpustack_client.delete_model(model["id"])
            except Exception:
                pass

    def test_deploy_qwen_sglang(self, shared_sglang_model):
        """Deploy Qwen model using SGLang from catalog."""
        assert shared_sglang_model["ready_replicas"] >= 1, "Model not ready"

    def test_sglang_model_inference(
        self, model_helper: ModelHelper, shared_sglang_model
    ):
        """Verify SGLang model inference."""
        response = model_helper.verify_model_inference(
            model_name=shared_sglang_model["name"],
            prompt="What is the capital of France?",
        )
        assert response["choices"][0]["message"]["content"]


@pytest.mark.model
@pytest.mark.amd
class TestAMDModelDeploy:
    """AMD GPU model deployment tests."""

    @pytest.fixture(scope="class")
    def shared_amd_model(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
    ):
        """Class-scoped shared AMD model."""
        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-qwen-amd",
            backend="vLLM",
            replicas=1,
            wait=True,
            timeout=e2e_config.models.deploy_timeout,
        )
        yield model
        if e2e_config.test.cleanup:
            try:
                gpustack_client.delete_model(model["id"])
            except Exception:
                pass

    def test_deploy_qwen_amd(self, shared_amd_model):
        """Deploy Qwen model on AMD GPU."""
        assert shared_amd_model["ready_replicas"] >= 1

    def test_amd_model_inference(self, model_helper: ModelHelper, shared_amd_model):
        """Verify AMD GPU model inference."""
        response = model_helper.verify_model_inference(
            model_name=shared_amd_model["name"]
        )
        assert response["choices"][0]["message"]["content"]


@pytest.mark.model
@pytest.mark.ascend
class TestAscendModelDeploy:
    """Ascend NPU model deployment tests."""

    @pytest.fixture(scope="class")
    def shared_ascend_model(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
    ):
        """Class-scoped shared Ascend model."""
        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-qwen-ascend",
            backend="MindIE",
            replicas=1,
            wait=True,
            timeout=e2e_config.models.deploy_timeout,
        )
        yield model
        if e2e_config.test.cleanup:
            try:
                gpustack_client.delete_model(model["id"])
            except Exception:
                pass

    def test_deploy_qwen_ascend(self, shared_ascend_model):
        """Deploy Qwen model on Ascend NPU."""
        assert shared_ascend_model["ready_replicas"] >= 1

    def test_ascend_model_inference(
        self, model_helper: ModelHelper, shared_ascend_model
    ):
        """Verify Ascend NPU model inference."""
        response = model_helper.verify_model_inference(
            model_name=shared_ascend_model["name"]
        )
        assert response["choices"][0]["message"]["content"]


@pytest.mark.model
@pytest.mark.custom_backend
class TestCustomBackendDeploy:
    """Test Case 15: Community inference backend deployment tests."""

    def test_add_custom_backend(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
    ):
        """Add community inference backend."""
        pytest.skip("Custom backend test requires specific backend configuration")

    def test_deploy_with_custom_backend(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """Deploy model using community inference backend."""
        pytest.skip("Custom backend test requires specific backend configuration")
