"""
用例 11: vLLM/SGLang 多副本部署
用例 12: vLLM 分布式部署
用例 13: SGLang 分布式部署
用例 14: MindIE 分布式部署
"""

import pytest

from e2e.utils.client import GPUStackClient
from e2e.utils.config import E2EConfig
from e2e.utils.models import ModelHelper
from e2e.utils.wait import wait_for_model_ready


@pytest.mark.distributed
@pytest.mark.vllm
@pytest.mark.nvidia
class TestVLLMMultiReplica:
    """vLLM 多副本部署测试"""

    def test_deploy_multi_replica(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """部署多副本模型"""
        # 检查是否有足够的 GPU
        result = gpustack_client.list_gpu_devices()
        gpu_count = len(result.get("items", []))

        if gpu_count < 2:
            pytest.skip("Need at least 2 GPUs for multi-replica test")

        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-multi-replica",
            backend="vLLM",
            replicas=2,
            wait=True,
            timeout=e2e_config.models.deploy_timeout * 2,
        )

        cleanup_models.append(model["id"])

        assert model["ready_replicas"] >= 2, "Not all replicas ready"

    def test_multi_replica_instance_logs(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """验证多副本实例日志查看"""
        result = gpustack_client.list_gpu_devices()
        gpu_count = len(result.get("items", []))

        if gpu_count < 2:
            pytest.skip("Need at least 2 GPUs for multi-replica test")

        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-multi-replica-logs",
            backend="vLLM",
            replicas=2,
            wait=True,
            timeout=e2e_config.models.deploy_timeout * 2,
        )

        cleanup_models.append(model["id"])

        # 获取实例列表
        instances = model_helper.get_model_instances(model["id"])
        assert len(instances) >= 2, "Expected at least 2 instances"

        # 验证每个实例的日志
        for instance in instances:
            logs = model_helper.get_instance_logs(instance["id"])
            assert logs is not None, f"Failed to get logs for instance {instance['id']}"

    def test_multi_replica_inference(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """验证多副本推理负载均衡"""
        result = gpustack_client.list_gpu_devices()
        gpu_count = len(result.get("items", []))

        if gpu_count < 2:
            pytest.skip("Need at least 2 GPUs for multi-replica test")

        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-multi-replica-inference",
            backend="vLLM",
            replicas=2,
            wait=True,
            timeout=e2e_config.models.deploy_timeout * 2,
        )

        cleanup_models.append(model["id"])

        # 发送多个请求，验证都能成功
        for i in range(5):
            response = model_helper.verify_model_inference(
                model_name=model["name"],
                prompt=f"Count to {i + 1}",
            )
            assert response["choices"][0]["message"]["content"]


@pytest.mark.distributed
@pytest.mark.sglang
@pytest.mark.nvidia
class TestSGLangMultiReplica:
    """SGLang 多副本部署测试"""

    def test_deploy_sglang_multi_replica(
        self,
        gpustack_client: GPUStackClient,
        model_helper: ModelHelper,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """部署 SGLang 多副本"""
        result = gpustack_client.list_gpu_devices()
        gpu_count = len(result.get("items", []))

        if gpu_count < 2:
            pytest.skip("Need at least 2 GPUs for multi-replica test")

        model = model_helper.deploy_huggingface_model(
            repo_id=e2e_config.models.default_model,
            name="e2e-test-sglang-multi-replica",
            backend="SGLang",
            replicas=2,
            wait=True,
            timeout=e2e_config.models.deploy_timeout * 2,
        )

        cleanup_models.append(model["id"])

        assert model["ready_replicas"] >= 2


@pytest.mark.distributed
@pytest.mark.vllm
@pytest.mark.nvidia
@pytest.mark.slow
class TestVLLMDistributed:
    """vLLM 分布式推理测试（Tensor Parallel）"""

    def test_vllm_tensor_parallel(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """测试 vLLM Tensor Parallel 部署"""
        result = gpustack_client.list_gpu_devices()
        gpu_count = len(result.get("items", []))

        if gpu_count < 2:
            pytest.skip("Need at least 2 GPUs for tensor parallel test")

        # 部署使用多 GPU 的大模型
        model = gpustack_client.create_model(
            name="e2e-test-vllm-tp",
            source="huggingface",
            huggingface_repo_id=e2e_config.models.default_model,
            backend="vLLM",
            replicas=1,
            # 使用 tensor parallel
            gpu_selector={"gpus_per_replica": 2},
            distributed_inference_across_workers=True,
        )

        cleanup_models.append(model["id"])

        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1

        # 验证推理
        response = gpustack_client.chat_completion(
            model=model["name"],
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response["choices"][0]["message"]["content"]


@pytest.mark.distributed
@pytest.mark.sglang
@pytest.mark.nvidia
@pytest.mark.slow
class TestSGLangDistributed:
    """SGLang 分布式推理测试"""

    def test_sglang_tensor_parallel(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """测试 SGLang Tensor Parallel 部署"""
        result = gpustack_client.list_gpu_devices()
        gpu_count = len(result.get("items", []))

        if gpu_count < 2:
            pytest.skip("Need at least 2 GPUs for tensor parallel test")

        model = gpustack_client.create_model(
            name="e2e-test-sglang-tp",
            source="huggingface",
            huggingface_repo_id=e2e_config.models.default_model,
            backend="SGLang",
            replicas=1,
            gpu_selector={"gpus_per_replica": 2},
            distributed_inference_across_workers=True,
        )

        cleanup_models.append(model["id"])

        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1

        response = gpustack_client.chat_completion(
            model=model["name"],
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response["choices"][0]["message"]["content"]


@pytest.mark.distributed
@pytest.mark.mindie
@pytest.mark.ascend
@pytest.mark.slow
class TestMindIEDistributed:
    """MindIE 分布式推理测试（昇腾）"""

    def test_mindie_distributed(
        self,
        gpustack_client: GPUStackClient,
        e2e_config: E2EConfig,
        cleanup_models,
    ):
        """测试 MindIE 分布式部署"""
        result = gpustack_client.list_gpu_devices()
        npu_count = len(result.get("items", []))

        if npu_count < 2:
            pytest.skip("Need at least 2 NPUs for distributed test")

        model = gpustack_client.create_model(
            name="e2e-test-mindie-distributed",
            source="huggingface",
            huggingface_repo_id=e2e_config.models.default_model,
            backend="MindIE",
            replicas=1,
            gpu_selector={"gpus_per_replica": 2},
            distributed_inference_across_workers=True,
        )

        cleanup_models.append(model["id"])

        model = wait_for_model_ready(
            gpustack_client,
            model["id"],
            timeout=e2e_config.models.deploy_timeout,
        )

        assert model["ready_replicas"] >= 1

        response = gpustack_client.chat_completion(
            model=model["name"],
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response["choices"][0]["message"]["content"]
