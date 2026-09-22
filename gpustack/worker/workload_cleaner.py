import logging
from typing import Callable, Set

from gpustack_runtime.deployer import WorkloadStatusStateEnum

from gpustack import envs
from gpustack.client.generated_clientset import ClientSet
from gpustack.schemas.workloads import WorkloadOwnerKindEnum
from gpustack.utils.runtime import is_benchmark_workload, is_cache_service_workload
from gpustack.worker.controlloop import OrphanReaper, WorkloadKind

logger = logging.getLogger(__name__)


class WorkloadCleaner:
    """
    Registers this worker's kinds of workload with the orphan reaper.

    Each entry says how to recognise the kind's containers and which of them
    the server still claims. Adding a kind is one entry; the sweep itself is
    shared.
    """

    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    _clientset_getter: Callable[[], ClientSet]
    _worker_id_getter: Callable[[], int]

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
    ):
        self._worker_id_getter = worker_id_getter
        self._clientset_getter = clientset_getter
        self._reaper = OrphanReaper(
            [
                WorkloadKind(
                    name="benchmark",
                    matches=is_benchmark_workload,
                    live_names=self._current_benchmark_names,
                    grace_period_seconds=(
                        envs.WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD
                    ),
                    # A benchmark is a task: once its container has finished or
                    # failed, the results are already collected and it is only
                    # holding a GPU.
                    reap_states=frozenset(
                        {
                            WorkloadStatusStateEnum.FAILED,
                            WorkloadStatusStateEnum.INACTIVE,
                        }
                    ),
                ),
                WorkloadKind(
                    name="cache service",
                    matches=is_cache_service_workload,
                    live_names=self._current_cache_service_instance_names,
                    grace_period_seconds=(
                        envs.WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD
                    ),
                ),
                # Model instance containers carry no type label, so they are
                # what is left. This has to stay last.
                WorkloadKind(
                    name="model instance",
                    matches=lambda workload: True,
                    live_names=self._current_model_instance_names,
                    grace_period_seconds=(
                        envs.WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD
                    ),
                ),
            ]
        )

    def _current_model_instance_names(self) -> Set[str]:
        names = set()
        model_instances_page = self._clientset.model_instances.list()
        for model_instance in model_instances_page.items or []:
            deployment_metadata = model_instance.get_deployment_metadata(
                self._worker_id,
            )
            if deployment_metadata:
                names.add(deployment_metadata.name)
        return names

    def _current_benchmark_names(self) -> Set[str]:
        names = set()
        benchmarks_page = self._clientset.benchmarks.list()
        for benchmark in benchmarks_page.items or []:
            deployment_metadata = benchmark.get_deployment_metadata()
            if deployment_metadata:
                names.add(deployment_metadata.name)
        return names

    def _current_cache_service_instance_names(self) -> Set[str]:
        instances_page = self._clientset.workloads.list(
            # page=-1 disables pagination: a truncated page would make the
            # cleaner treat live instances as orphans and delete their
            # running cache servers.
            params={
                "worker_id": self._worker_id,
                "owner_kind": WorkloadOwnerKindEnum.CACHE_SERVICE.value,
                "page": -1,
            }
        )
        return {instance.name for instance in instances_page.items or []}

    def cleanup_orphan_workloads(self):
        self._reaper.reap()
