from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


from gpustack.schemas.cache_services import CacheServiceInstance
from gpustack.worker.workload_cleaner import WorkloadCleaner
from gpustack_runtime.deployer import WorkloadStatusStateEnum


@pytest.fixture(autouse=True)
def namespaces(monkeypatch):
    monkeypatch.setattr(
        "gpustack.worker.workload_cleaner.cache_workload_namespaces", lambda: set()
    )


def _build_cleaner(worker_id: int = 1):
    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[])
    clientset.benchmarks.list.return_value = SimpleNamespace(items=[])
    clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[])
    cleaner = WorkloadCleaner(lambda: worker_id, lambda: clientset)
    return cleaner, clientset


def _cache_workload(name: str):
    return SimpleNamespace(
        name=name,
        namespace=None,
        labels={"type": "cache-service"},
        # Far past the orphan grace period.
        created_at="2020-01-01T00:00:00Z",
        state=WorkloadStatusStateEnum.RUNNING,
    )


def test_cleanup_keeps_workloads_of_this_workers_instances():
    """The live set of cache-service workload names comes from this
    worker's instance rows; only workloads outside it are orphans."""
    cleaner, clientset = _build_cleaner(worker_id=1)
    instance = CacheServiceInstance(
        id=11, name="svc-a1b2c", cache_service_id=5, worker_id=1, cluster_id=1
    )
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[instance]
    )

    workloads = [
        _cache_workload("svc-a1b2c"),
        _cache_workload("cache-svc-5-i99"),
    ]
    with (
        patch(
            "gpustack.worker.workload_cleaner.list_workloads",
            return_value=workloads,
        ),
        patch("gpustack.worker.workload_cleaner.delete_workload") as delete,
    ):
        cleaner.cleanup_orphan_workloads()

    assert clientset.cache_service_instances.list.call_args[1]["params"] == {
        "worker_id": 1,
        "page": -1,
    }
    delete.assert_called_once_with("cache-svc-5-i99")


def test_cleanup_spares_recent_cache_service_workloads():
    """An unmatched cache-service workload inside the grace period is left
    alone: its instance row may simply not have landed yet."""
    cleaner, _ = _build_cleaner(worker_id=1)

    workload = _cache_workload("cache-svc-5-i99")
    with (
        patch(
            "gpustack.worker.workload_cleaner.list_workloads",
            return_value=[workload],
        ),
        patch(
            "gpustack.worker.workload_cleaner.network.is_offline",
            return_value=(False, None),
        ),
        patch("gpustack.worker.workload_cleaner.delete_workload") as delete,
    ):
        cleaner.cleanup_orphan_workloads()

    delete.assert_not_called()


def test_cleanup_discovers_orphans_in_org_namespaces_after_worker_restart(monkeypatch):
    cleaner, clientset = _build_cleaner()
    clientset.cache_service_instances.list.return_value = SimpleNamespace(
        items=[
            CacheServiceInstance(
                id=11,
                name="cache-live",
                cache_service_id=5,
                worker_id=1,
                cluster_id=1,
                workload_namespace="gpustack-acme",
            )
        ]
    )
    live = _cache_workload("cache-live")
    orphan = _cache_workload("cache-orphan")
    live.namespace = orphan.namespace = "gpustack-acme"
    monkeypatch.setattr(
        "gpustack.worker.workload_cleaner.cache_workload_namespaces",
        lambda: {"gpustack-acme"},
    )
    with (
        patch(
            "gpustack.worker.workload_cleaner.list_workloads",
            side_effect=[[], [live, orphan]],
        ) as listing,
        patch("gpustack.worker.workload_cleaner.delete_workload") as delete,
    ):
        cleaner.cleanup_orphan_workloads()
    listing.assert_called_with(
        namespace="gpustack-acme", labels={"type": "cache-service"}
    )
    delete.assert_called_once_with("cache-orphan", namespace="gpustack-acme")
