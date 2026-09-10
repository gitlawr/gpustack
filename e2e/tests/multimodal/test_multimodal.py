"""
Test Case 9: Deploy multimodal models with vLLM (CUDA) - Z-Image-Turbo/Qwen3-tts-customvoice/Qwen3-ASR
Test Case 10: Deploy multimodal models with vLLM (Ascend)
"""

import pytest

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig
from e2e.utils.models import ModelHelper
from e2e.utils.wait import wait_for_model_ready


@pytest.mark.multimodal
@pytest.mark.vllm
@pytest.mark.nvidia
@pytest.mark.slow
class TestMultimodalNVIDIA:
    """NVIDIA GPU multimodal model tests"""

    def test_deploy_image_model(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """Deploy an image generation model (Z-Image-Turbo)"""
        repo_id = e2e_config.models.multimodal.image
        model_name = repo_id.split("/")[-1]

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}",
            source="huggingface",
            huggingface_repo_id=repo_id,  # Requires full repo id
            backend="vLLM",
            categories=["image"],
            replicas=1,
            cluster_id=model_helper._get_default_cluster_id(),
        )

        cleanup_models.append(model["id"])

        # Wait for model to be ready
        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1

    @pytest.fixture(scope="class")
    def shared_tts_model(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
    ):
        """Class-scoped shared TTS model."""
        repo_id = e2e_config.models.multimodal.tts
        model_name = repo_id.split("/")[-1]

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}",
            source="huggingface",
            huggingface_repo_id=repo_id,
            backend="VoxBox",
            categories=["text_to_speech"],
            replicas=1,
            cluster_id=model_helper._get_default_cluster_id(),
        )

        try:
            model = wait_for_model_ready(
                gpustack_client,
                model["id"],
                timeout=e2e_config.models.deploy_timeout,
            )
        except Exception:
            if e2e_config.test.cleanup:
                try:
                    gpustack_client.delete_model(model["id"])
                except Exception:
                    pass
            raise

        yield model

        if e2e_config.test.cleanup:
            try:
                gpustack_client.delete_model(model["id"])
            except Exception:
                pass

    def test_deploy_tts_model(self, shared_tts_model):
        """Deploy a TTS model (Qwen3-tts-customvoice)"""
        assert shared_tts_model["ready_replicas"] >= 1

    def test_tts_inference(
        self,
        gpustack_client: GPUStackClient,
        shared_tts_model,
    ):
        """Test TTS inference"""
        audio_content = gpustack_client.audio_speech(
            model=shared_tts_model["name"],
            input="Hello, this is a test of text to speech.",
            voice="alloy",
        )

        assert len(audio_content) > 0, "Empty audio response"

    def test_deploy_asr_model(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """Deploy an ASR model (Qwen3-ASR)"""
        repo_id = e2e_config.models.multimodal.asr
        model_name = repo_id.split("/")[-1]

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}",
            source="huggingface",
            huggingface_repo_id=repo_id,
            backend="VoxBox",  # ASR uses VoxBox backend
            categories=["speech_to_text"],
            replicas=1,
            cluster_id=model_helper._get_default_cluster_id(),
        )

        cleanup_models.append(model["id"])

        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1

    def test_asr_inference(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """Test ASR inference"""
        # Requires an audio test file
        pytest.skip("ASR inference test requires audio test file")


@pytest.mark.multimodal
@pytest.mark.vllm
@pytest.mark.ascend
@pytest.mark.slow
class TestMultimodalAscend:
    """Ascend NPU multimodal model tests"""

    def test_deploy_image_model_ascend(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """Deploy an image model on Ascend"""
        repo_id = e2e_config.models.multimodal.image
        model_name = repo_id.split("/")[-1]

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}-ascend",
            source="huggingface",
            huggingface_repo_id=repo_id,
            backend="MindIE",
            categories=["image"],
            replicas=1,
            cluster_id=model_helper._get_default_cluster_id(),
        )

        cleanup_models.append(model["id"])

        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1

    def test_deploy_tts_model_ascend(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """Deploy a TTS model on Ascend"""
        repo_id = e2e_config.models.multimodal.tts
        model_name = repo_id.split("/")[-1]

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}-ascend",
            source="huggingface",
            huggingface_repo_id=repo_id,
            backend="VoxBox",
            categories=["text_to_speech"],
            replicas=1,
            cluster_id=model_helper._get_default_cluster_id(),
        )

        cleanup_models.append(model["id"])

        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1

    def test_deploy_asr_model_ascend(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """Deploy an ASR model on Ascend"""
        repo_id = e2e_config.models.multimodal.asr
        model_name = repo_id.split("/")[-1]

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}-ascend",
            source="huggingface",
            huggingface_repo_id=repo_id,
            backend="VoxBox",
            categories=["speech_to_text"],
            replicas=1,
            cluster_id=model_helper._get_default_cluster_id(),
        )

        cleanup_models.append(model["id"])

        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1
