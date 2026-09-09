import logging

from gpustack.client import ClientSet
from gpustack.worker.controlloop import patch_status

logger = logging.getLogger(__name__)


def update_cache_service_instance(clientset: ClientSet, id: int, **kwargs) -> bool:
    """
    Update the workload a cache service instance runs as.

    Shared by the manager (control loop, runs in the worker process) and the
    provisioner (runs in the provisioning subprocess), which write back to the
    same row from two processes.

    Returns:
        Whether the update was applied.
    """
    return patch_status(
        clientset.workloads,
        id,
        "Cache service workload",
        **kwargs,
    )
