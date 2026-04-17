"""
用例 9: vLLM 部署多模态模型（CUDA）- Z-Image-Turbo/Qwen3-tts-customvoice/Qwen3-ASR
用例 10: vLLM 部署多模态模型（Ascend）
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
    """NVIDIA GPU 多模态模型测试"""

    def test_deploy_image_model(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """部署图像生成模型 (Z-Image-Turbo)"""
        model_name = e2e_config.models.multimodal.image

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}",
            source="huggingface",
            huggingface_repo_id=model_name,  # 需要完整的 repo id
            backend="vLLM",
            categories=["image"],
            replicas=1,
        )

        cleanup_models.append(model["id"])

        # 等待模型就绪
        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1

    def test_deploy_tts_model(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """部署 TTS 模型 (Qwen3-tts-customvoice)"""
        model_name = e2e_config.models.multimodal.tts

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}",
            source="huggingface",
            huggingface_repo_id=model_name,
            backend="VoxBox",  # TTS 使用 VoxBox 后端
            categories=["text_to_speech"],
            replicas=1,
        )

        cleanup_models.append(model["id"])

        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1

    def test_tts_inference(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """测试 TTS 推理"""
        model_name = e2e_config.models.multimodal.tts

        # 部署模型
        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}-inference",
            source="huggingface",
            huggingface_repo_id=model_name,
            backend="VoxBox",
            categories=["text_to_speech"],
            replicas=1,
        )

        cleanup_models.append(model["id"])

        wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        # 测试 TTS
        audio_content = gpustack_client.audio_speech(
            model=model["name"],
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
        """部署 ASR 模型 (Qwen3-ASR)"""
        model_name = e2e_config.models.multimodal.asr

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}",
            source="huggingface",
            huggingface_repo_id=model_name,
            backend="VoxBox",  # ASR 使用 VoxBox 后端
            categories=["speech_to_text"],
            replicas=1,
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
        """测试 ASR 推理"""
        # 需要准备音频测试文件
        pytest.skip("ASR inference test requires audio test file")


@pytest.mark.multimodal
@pytest.mark.vllm
@pytest.mark.ascend
@pytest.mark.slow
class TestMultimodalAscend:
    """昇腾 NPU 多模态模型测试"""

    def test_deploy_image_model_ascend(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """在昇腾上部署图像模型"""
        model_name = e2e_config.models.multimodal.image

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}-ascend",
            source="huggingface",
            huggingface_repo_id=model_name,
            backend="MindIE",
            categories=["image"],
            replicas=1,
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
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """在昇腾上部署 TTS 模型"""
        model_name = e2e_config.models.multimodal.tts

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}-ascend",
            source="huggingface",
            huggingface_repo_id=model_name,
            backend="VoxBox",
            categories=["text_to_speech"],
            replicas=1,
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
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """在昇腾上部署 ASR 模型"""
        model_name = e2e_config.models.multimodal.asr

        model = gpustack_client.create_model(
            name=f"e2e-test-{model_name}-ascend",
            source="huggingface",
            huggingface_repo_id=model_name,
            backend="VoxBox",
            categories=["speech_to_text"],
            replicas=1,
        )

        cleanup_models.append(model["id"])

        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1
