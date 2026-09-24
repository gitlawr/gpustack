from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.policies.candidate_selectors import (
    SGLangResourceFitSelector,
    VLLMResourceFitSelector,
)
from gpustack.schemas.models import ExtendedKVCacheConfig, KVCacheModeEnum
from tests.fixtures.workers.fixtures import linux_nvidia_5_a100_80gx2
from tests.utils.model import new_model

GIB = 1024**3


@pytest.fixture(params=[VLLMResourceFitSelector, SGLangResourceFitSelector])
def selector_class(request):
    return request.param


def _worker(worker_id, ram_gib):
    worker = linux_nvidia_5_a100_80gx2()
    worker.id = worker_id
    worker.name = f"worker-{worker_id}"
    worker.status.memory.total = ram_gib * GIB
    worker.system_reserved.ram = 0
    worker.system_reserved.vram = 0
    for gpu in worker.status.gpu_devices:
        gpu.memory.total = 80 * GIB
    return worker


async def _select(config, selector_class, workers, cache, vram_gib, gpu_count=0):
    parameters = [
        (
            "--gpu-memory-utilization=0.9"
            if selector_class is VLLMResourceFitSelector
            else "--mem-fraction-static=0.9"
        )
    ]
    if gpu_count:
        parameters.append(f"--tensor-parallel-size={gpu_count}")
    model = new_model(
        1,
        "cache-ram-fit",
        huggingface_repo_id="test/model",
        backend_parameters=parameters,
        extended_kv_cache=cache,
        env={"GPUSTACK_MODEL_VRAM_CLAIM": str(vram_gib * GIB)},
    )
    pretrained_config = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=32,
        hidden_size=4096,
        max_position_embeddings=4096,
    )
    with patch(
        "gpustack.policies.candidate_selectors.base_candidate_selector."
        "get_pretrained_config_with_workers",
        new=AsyncMock(return_value=pretrained_config),
    ):
        return await selector_class(config, model, []).select_candidates(workers)


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_count, vram_gib", [(1, 50), (2, 100)])
@pytest.mark.parametrize("explicit_gpu_count", [False, True])
@pytest.mark.parametrize(
    "cache_knobs, ram_per_gpu", [(dict(ram_size=20), 20), (dict(ram_ratio=0.25), 18)]
)
@pytest.mark.parametrize("shortfall", [0, 1])
async def test_single_worker_requires_the_final_cache_ram(
    config,
    selector_class,
    gpu_count,
    vram_gib,
    explicit_gpu_count,
    cache_knobs,
    ram_per_gpu,
    shortfall,
):
    required_ram = gpu_count * ram_per_gpu
    candidates = await _select(
        config,
        selector_class,
        [_worker(1, required_ram - shortfall)],
        ExtendedKVCacheConfig(enabled=True, **cache_knobs),
        vram_gib,
        gpu_count if explicit_gpu_count else 0,
    )

    if shortfall:
        assert candidates == []
    else:
        assert candidates
        for candidate in candidates:
            assert len(candidate.gpu_indexes) == gpu_count
            assert candidate.computed_resource_claim.ram == required_ram * GIB


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_count", [0, 4], ids=["automatic", "explicit"])
@pytest.mark.parametrize(
    "cache_knobs, ram_per_worker", [(dict(ram_size=20), 40), (dict(ram_ratio=0.25), 36)]
)
@pytest.mark.parametrize(
    "shortfalls, expected_worker_ids",
    [([0, 0], {1, 2}), ([0, 1], set()), ([1, 0, 0], {2, 3})],
)
async def test_distributed_cache_ram_is_checked_per_worker(
    config,
    selector_class,
    gpu_count,
    cache_knobs,
    ram_per_worker,
    shortfalls,
    expected_worker_ids,
):
    workers = [
        _worker(index, ram_per_worker - shortfall)
        for index, shortfall in enumerate(shortfalls, start=1)
    ]
    candidates = await _select(
        config,
        selector_class,
        workers,
        ExtendedKVCacheConfig(enabled=True, **cache_knobs),
        vram_gib=200,
        gpu_count=gpu_count,
    )

    if not expected_worker_ids:
        assert candidates == []
        return

    assert len(candidates) == 1
    candidate = candidates[0]
    subordinates = candidate.subordinate_workers
    assert {
        candidate.worker.id,
        *(worker.worker_id for worker in subordinates),
    } == expected_worker_ids
    assert candidate.computed_resource_claim.ram == ram_per_worker * GIB
    assert all(
        worker.computed_resource_claim.ram == ram_per_worker * GIB
        for worker in subordinates
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_count, vram_gib", [(1, 100), (2, 200)])
@pytest.mark.parametrize(
    "cache",
    [
        None,
        ExtendedKVCacheConfig(enabled=False, ram_size=20),
        ExtendedKVCacheConfig(
            enabled=True,
            mode=KVCacheModeEnum.SHARED,
            cache_service_id=1,
            ram_size=20,
        ),
    ],
    ids=["absent", "disabled", "shared"],
)
async def test_no_local_cache_requires_no_cache_ram(
    config, selector_class, worker_count, vram_gib, cache
):
    workers = [_worker(index, 0) for index in range(1, worker_count + 1)]
    candidates = await _select(
        config, selector_class, workers, cache, vram_gib, worker_count * 2
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.computed_resource_claim.ram is None
    subordinates = candidate.subordinate_workers or []
    assert len(subordinates) == worker_count - 1
    assert all(worker.computed_resource_claim.ram is None for worker in subordinates)
