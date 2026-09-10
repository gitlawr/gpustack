"""Handing the scheduler the workload rows alongside the instances.

The rows travel down the call chain as an optional argument so the readings
can be compared while the instances stay authoritative. Two things have to
hold for that: every class whose call site forwards ``self._workloads`` must
actually carry one -- a missing attribute there is a scheduling crash, not a
degraded comparison -- and passing the rows must not change what the
scheduler decides.
"""

import logging
import pytest

from gpustack.config.config import Config
from gpustack.policies.candidate_selectors.ascend_mindie_resource_fit_selector import (
    AscendMindIEResourceFitSelector,
)
from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (
    CustomBackendResourceFitSelector,
)
from gpustack.policies.candidate_selectors.gguf_resource_fit_selector import (
    GGUFResourceFitSelector,
)
from gpustack.policies.candidate_selectors.sglang_resource_fit_selector import (
    MemFractionStaticCalculator,
    SGLangResourceFitSelector,
)
from gpustack.policies.candidate_selectors.vgpu_resource_fit_selector import (
    VGPUResourceFitSelector,
)
from gpustack.policies.candidate_selectors.vllm_resource_fit_selector import (
    VLLMResourceFitSelector,
)
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.server.model_instance_workloads import compile_model_instance
from gpustack.policies.utils import (
    get_worker_allocatable_resource,
    get_worker_model_instances,
)
from gpustack.schemas.models import (
    BackendEnum,
    ComputedResourceClaim,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.workers import (
    GPUCoreInfo,
    GPUDeviceStatus,
    MemoryInfo,
    SystemReserved,
    Worker,
    WorkerStatus,
)

MAIN, SUB = 1, 2


def _model():
    return Model(id=1, name="m", replicas=1, backend=BackendEnum.VLLM)


def _config(tmp_path):
    return Config(token="t", jwt_secret_key="s", data_dir=str(tmp_path))


def _worker(worker_id=MAIN, gpu_type="cuda"):
    return Worker(
        id=worker_id,
        name=f"w{worker_id}",
        hostname=f"w{worker_id}",
        ip=f"10.0.0.{worker_id}",
        system_reserved=SystemReserved(ram=0, vram=0),
        status=WorkerStatus(
            memory=MemoryInfo(total=64_000, used=0, is_unified_memory=False),
            gpu_devices=[
                GPUDeviceStatus(
                    uuid=f"uuid-{worker_id}",
                    name="gpu",
                    vendor="NVIDIA",
                    index=worker_id - 1,
                    type=gpu_type,
                    memory=MemoryInfo(total=48_000, used=0),
                    core=GPUCoreInfo(total=100, utilization_rate=0),
                )
            ],
        ),
    )


def _instance():
    return ModelInstance(
        id=1,
        name="mi",
        worker_id=MAIN,
        gpu_type="cuda",
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(ram=1024, vram={0: 100}),
        distributed_servers=DistributedServers(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
            subordinate_workers=[
                ModelInstanceSubordinateWorker(
                    worker_id=SUB,
                    gpu_type="cuda",
                    gpu_indexes=[1],
                    computed_resource_claim=ComputedResourceClaim(vram={1: 200}),
                )
            ],
        ),
    )


def _rows():
    """Compiled by the server, not hand-built: the readers depend on how a
    row spells a claim and its ports, and a stand-in gets to be wrong."""
    return compile_model_instance(_instance())


# ---------------------------------------------------------------------------
# Every forwarder carries the attribute it forwards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda cfg, rows: VLLMResourceFitSelector(cfg, _model(), [], rows),
            id="vllm",
        ),
        pytest.param(
            lambda cfg, rows: SGLangResourceFitSelector(cfg, _model(), [], rows),
            id="sglang",
        ),
        pytest.param(
            lambda cfg, rows: VGPUResourceFitSelector(cfg, _model(), [], rows),
            id="vgpu",
        ),
        pytest.param(
            lambda cfg, rows: AscendMindIEResourceFitSelector(cfg, _model(), [], rows),
            id="ascend-mindie",
        ),
        pytest.param(
            lambda cfg, rows: CustomBackendResourceFitSelector(cfg, _model(), [], rows),
            id="custom-backend",
        ),
        pytest.param(
            lambda cfg, rows: GGUFResourceFitSelector(_model(), [], None, rows),
            id="gguf",
        ),
        pytest.param(
            lambda cfg, rows: PlacementScorer(_model(), [], workloads=rows),
            id="placement-scorer",
        ),
        pytest.param(
            lambda cfg, rows: MemFractionStaticCalculator(
                _model(), [], None, "cuda", {}, rows
            ),
            id="mem-fraction-static",
        ),
    ],
)
def test_the_rows_reach_every_reader(tmp_path, build):
    """Each of these forwards ``self._workloads`` into the allocatable
    reading, so the attribute has to exist whether or not rows were given."""
    rows = _rows()

    assert build(_config(tmp_path), rows)._workloads is rows
    assert build(_config(tmp_path), None)._workloads is None


# ---------------------------------------------------------------------------
# Passing them changes nothing but the comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("worker_id", [MAIN, SUB])
@pytest.mark.parametrize("gpu_type", [None, "cuda"])
def test_the_allocatable_is_the_same_with_and_without_rows(worker_id, gpu_type):
    instances, worker = [_instance()], _worker(worker_id)

    assert get_worker_allocatable_resource(
        instances, worker, gpu_type, _rows()
    ) == get_worker_allocatable_resource(instances, worker, gpu_type)


@pytest.mark.parametrize("worker_id", [MAIN, SUB])
def test_the_same_instances_are_found_with_and_without_rows(worker_id):
    instances, worker = [_instance()], _worker(worker_id)

    assert get_worker_model_instances(
        instances, worker, _rows()
    ) == get_worker_model_instances(instances, worker)


def test_the_comparison_runs_only_when_rows_are_given(caplog):
    """No rows means the comparison did not run, not that it agreed -- the
    evaluator takes this path, and its silence must not read as evidence."""
    from gpustack.utils.comparison import tally

    tally("Workload allocation").reset()

    with caplog.at_level(logging.INFO):
        get_worker_allocatable_resource([_instance()], _worker(), "cuda")
    assert caplog.text == ""

    with caplog.at_level(logging.INFO):
        get_worker_allocatable_resource([_instance()], _worker(), "cuda", _rows())
    assert "Workload allocation: agreed=1" in caplog.text


def test_a_row_allocating_elsewhere_is_reported(caplog):
    """The allocatable decides what the scheduler thinks is free; reading it
    low overcommits a worker rather than failing visibly."""
    rows = _rows()
    rows[0].computed_resource_claim = {"ram": 1024, "vram": {0: 999}}

    with caplog.at_level(logging.INFO):
        get_worker_allocatable_resource([_instance()], _worker(), "cuda", rows)

    assert "Workload allocation differs on worker 1 of type cuda" in caplog.text
