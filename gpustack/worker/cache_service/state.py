import logging

from gpustack.client import ClientSet
from gpustack.schemas.cache_services import CacheServiceInstanceUpdate
from gpustack.worker.controlloop import update_resource

logger = logging.getLogger(__name__)


def update_cache_service_instance(clientset: ClientSet, id: int, **kwargs) -> bool:
    """
    Update a cache service instance with the given fields.

    Shared by the manager (control loop, runs in the worker process) and the
    provisioner (runs in the provisioning subprocess), which write back to the
    same row from two processes.

    Returns:
        Whether the update was applied.
    """
    return update_resource(
        clientset.cache_service_instances,
        id,
        CacheServiceInstanceUpdate,
        "Cache service instance",
        **kwargs,
    )
