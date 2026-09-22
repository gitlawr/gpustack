"""
Reading a model instance's workload rows from the worker.

The worker has no database session, so the rows come through the generated
client. Its watch is unfiltered -- the server scopes it to this worker's
cluster -- so the cache already holds the peers' rows, and a filtered read
costs no round trip. When the watch is not authoritative (not yet started,
reconnecting) the client falls back to an unpaginated API call by itself,
which is why this does not pass a page parameter: doing so would skip the
cache on every call and lose that fallback's protection against a truncated
page.
"""

import logging
from typing import Any, Dict, List

from gpustack.schemas.workloads import WorkloadOwnerKindEnum
from gpustack.utils.model_instance_workers import rows_by_owner

logger = logging.getLogger(__name__)


def group_workloads(clientset: Any, instance_id: int) -> List:
    """
    Every workload row of one model instance -- leader and followers.

    Returns an empty list rather than raising: the callers are building an
    argument vector or locating themselves in a group, and an exception there
    would fail a launch over a reading that is still only compared against the
    instance.
    """
    try:
        page = clientset.workloads.list(
            params={
                "owner_kind": WorkloadOwnerKindEnum.MODEL_INSTANCE.value,
                "owner_id": instance_id,
            }
        )
        return list(page.items or [])
    except Exception as e:
        logger.debug(f"Failed to read workloads of model instance {instance_id}: {e}")
        return []


def workloads_by_instance(clientset: Any) -> Dict[int, List]:
    """
    Every model-instance row this worker can see, grouped by instance.

    For the passes that walk every instance at once. Asking per instance
    would filter the whole watch cache once per instance; this filters it
    once and groups in Python.
    """
    try:
        page = clientset.workloads.list(
            params={"owner_kind": WorkloadOwnerKindEnum.MODEL_INSTANCE.value}
        )
        return rows_by_owner(list(page.items or []))
    except Exception as e:
        logger.debug(f"Failed to read model instance workloads: {e}")
        return {}
