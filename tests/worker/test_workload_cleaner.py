from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gpustack.schemas.workloads import Workload, WorkloadOwnerKindEnum
from gpustack.worker.workload_cleaner import WorkloadCleaner
from gpustack_runtime.deployer import WorkloadStatusStateEnum


def _build_cleaner(worker_id: int = 1):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    clientset.benchmarks.list.return_value = SimpleNamespace(items=[])
    clientset.workloads.list.return_value = SimpleNamespace(items=[])
    cleaner = WorkloadCleaner(lambda: worker_id, lambda: clientset)
    return cleaner, clientset


def _cache_workload(name: str):
    return SimpleNamespace(
        name=name,
        labels={"type": "cache-service"},
        # Far past the orphan grace period.
        created_at="2020-01-01T00:00:00Z",
        state=WorkloadStatusStateEnum.RUNNING,
    )


def test_cleanup_keeps_workloads_of_this_workers_instances():
    """The live set of cache-service workload names comes from this
    worker's instance rows; only workloads outside it are orphans."""
    cleaner, clientset = _build_cleaner(worker_id=1)
    instance = Workload(
        id=11,
        name="cache-svc-5-w1",
        owner_kind=WorkloadOwnerKindEnum.CACHE_SERVICE,
        owner_id=5,
        worker_id=1,
        cluster_id=1,
    )
    clientset.workloads.list.return_value = SimpleNamespace(items=[instance])

    workloads = [
        _cache_workload("cache-svc-5-w1"),
        _cache_workload("cache-svc-5-w99"),
    ]
    with (
        patch(
            "gpustack.worker.controlloop.reaper.list_workloads",
            return_value=workloads,
        ),
        patch("gpustack.worker.controlloop.reaper.delete_workload") as delete,
    ):
        cleaner.cleanup_orphan_workloads()

    assert clientset.workloads.list.call_args[1]["params"] == {
        "worker_id": 1,
        "owner_kind": "cache_service",
        "page": -1,
    }
    delete.assert_called_once_with("cache-svc-5-w99")


def test_cleanup_spares_recent_cache_service_workloads():
    """An unmatched cache-service workload inside the grace period is left
    alone: its instance row may simply not have landed yet."""
    cleaner, _ = _build_cleaner(worker_id=1)

    workload = _cache_workload("cache-svc-5-i99")
    with (
        patch(
            "gpustack.worker.controlloop.reaper.list_workloads",
            return_value=[workload],
        ),
        patch(
            "gpustack.worker.controlloop.reaper.network.is_offline",
            return_value=(False, None),
        ),
        patch("gpustack.worker.controlloop.reaper.delete_workload") as delete,
    ):
        cleaner.cleanup_orphan_workloads()

    delete.assert_not_called()
