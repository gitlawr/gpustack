"""Aggregating what a worker has allocated, from workload rows.

The instance-based version walks every distributed instance's embedded
subordinate list, because a JSON column cannot be filtered by worker
portably. A workload row carries its worker in a column. The two have to
agree exactly -- this decides what the scheduler thinks is free.
"""

from types import SimpleNamespace

import pytest

from gpustack.policies.utils import (
    compute_worker_allocated,
    compute_worker_allocated_from_workloads,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstance,
    ModelInstanceSubordinateWorker,
)

MAIN, SUB = 1, 2


def _instance(gpu_type="cuda", sub_gpu_type="cuda", ram=1024):
    return ModelInstance(
        id=1,
        name="mi",
        worker_id=MAIN,
        gpu_type=gpu_type,
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(ram=ram, vram={0: 100}),
        distributed_servers=DistributedServers(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
            subordinate_workers=[
                ModelInstanceSubordinateWorker(
                    worker_id=SUB,
                    gpu_type=sub_gpu_type,
                    gpu_indexes=[1],
                    computed_resource_claim=ComputedResourceClaim(
                        ram=999, vram={1: 200}
                    ),
                )
            ],
        ),
    )


def _rows(gpu_type="cuda", sub_gpu_type="cuda", ram=1024):
    return [
        SimpleNamespace(
            owner_id=1,
            group_index=0,
            worker_id=MAIN,
            gpu_type=gpu_type,
            gpu_indexes=[0],
            computed_resource_claim={"ram": ram, "vram": {0: 100}},
        ),
        SimpleNamespace(
            owner_id=1,
            group_index=1,
            worker_id=SUB,
            gpu_type=sub_gpu_type,
            gpu_indexes=[1],
            computed_resource_claim={"ram": 999, "vram": {1: 200}},
        ),
    ]


@pytest.mark.parametrize("worker_id", [MAIN, SUB])
@pytest.mark.parametrize("gpu_type", [None, "cuda", "rocm"])
def test_the_two_sources_agree(worker_id, gpu_type):
    assert compute_worker_allocated_from_workloads(
        _rows(), worker_id, gpu_type
    ) == compute_worker_allocated([_instance()], worker_id, gpu_type)


def test_a_subordinate_contributes_no_ram():
    """The rpc-server side does not hold the model, so only its vram counts --
    the subordinate's own claim carries a ram figure that is deliberately
    ignored."""
    allocated = compute_worker_allocated_from_workloads(_rows(), SUB)

    assert allocated.ram == 0
    assert allocated.vram == {1: 200}


def test_a_subordinate_is_filtered_by_the_instances_gpu_type_not_its_own():
    """Reproduced deliberately. A subordinate whose type differs from the
    instance's goes uncounted, which reports less allocated than there is; it
    decides placement today, so it is preserved here and changed on its own.
    """
    rows = _rows(gpu_type="cuda", sub_gpu_type="rocm")

    # Asked about the subordinate's own type, it contributes nothing...
    assert compute_worker_allocated_from_workloads(rows, SUB, "rocm").vram == {}
    # ...and it is counted under the instance's instead.
    assert compute_worker_allocated_from_workloads(rows, SUB, "cuda").vram == {1: 200}
    # Which is what the instance-based version does.
    instance = _instance(gpu_type="cuda", sub_gpu_type="rocm")
    assert compute_worker_allocated([instance], SUB, "rocm").vram == {}
    assert compute_worker_allocated([instance], SUB, "cuda").vram == {1: 200}


def test_a_leader_with_no_gpu_type_is_not_filtered_out():
    """The leader branch has an ``is None`` escape the subordinate one does
    not; both are kept as they are."""
    rows = _rows(gpu_type=None)

    assert compute_worker_allocated_from_workloads(rows, MAIN, "cuda").vram == {0: 100}


def test_the_leaders_type_is_read_from_its_own_row():
    """The subordinate filter needs the instance's gpu type, which lives on
    the leader's row -- and that row can be on another worker, so a
    worker-scoped query would not have it."""
    rows = _rows(gpu_type="rocm", sub_gpu_type="cuda")

    assert compute_worker_allocated_from_workloads(rows, SUB, "rocm").vram == {1: 200}
    assert compute_worker_allocated_from_workloads(rows, SUB, "cuda").vram == {}
