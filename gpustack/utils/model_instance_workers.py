import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from gpustack.schemas.models import (
    ComputedResourceClaim,
    ModelInstance,
    ModelInstanceStateEnum,
)
from gpustack.schemas.workloads import Workload, WorkloadOwnerKindEnum

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInstanceWorkerMatch:
    is_main_worker: bool = False
    subordinate_worker_indexes: tuple[int, ...] = ()

    @property
    def matched(self) -> bool:
        return self.is_main_worker or bool(self.subordinate_worker_indexes)


def get_model_instance_worker_match(
    instance: ModelInstance,
    *,
    worker_name: Optional[str] = None,
    worker_id: Optional[int] = None,
) -> ModelInstanceWorkerMatch:
    is_main_worker = False
    if worker_name is not None and instance.worker_name == worker_name:
        is_main_worker = True
    if worker_id is not None and instance.worker_id == worker_id:
        is_main_worker = True

    subordinate_worker_indexes = []
    subordinate_workers = (
        instance.distributed_servers.subordinate_workers
        if instance.distributed_servers
        and instance.distributed_servers.subordinate_workers
        else []
    )
    for index, subordinate_worker in enumerate(subordinate_workers):
        if worker_name is not None and subordinate_worker.worker_name == worker_name:
            subordinate_worker_indexes.append(index)
            continue
        if worker_id is not None and subordinate_worker.worker_id == worker_id:
            subordinate_worker_indexes.append(index)

    return ModelInstanceWorkerMatch(
        is_main_worker=is_main_worker,
        subordinate_worker_indexes=tuple(subordinate_worker_indexes),
    )


async def get_worker_matches_from_workloads(
    session, worker_ids: Iterable[int]
) -> Dict[int, Dict[int, ModelInstanceWorkerMatch]]:
    """
    Which part of which instance each of these workers runs, read off the
    workload rows.

    The same question ``get_model_instance_worker_match`` answers by scanning
    every instance in the cluster and walking its embedded subordinate list.
    A workload row is that answer already -- it has a worker and a position in
    its group -- and ``ix_workloads_worker_id`` indexes exactly this lookup,
    so the scan becomes one query for all the workers at once.

    Returns worker_id -> instance_id -> match. A worker with nothing on it, or
    an instance whose rows have not been compiled yet, is simply absent, which
    is why callers keep the embedded list as the fallback.
    """
    ids = {wid for wid in worker_ids if wid is not None}
    if not ids:
        return {}
    rows = await Workload.all_by_fields(
        session, {"owner_kind": WorkloadOwnerKindEnum.MODEL_INSTANCE}
    )
    positions: Dict[int, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.worker_id in ids and row.owner_id is not None:
            positions[row.worker_id][row.owner_id].append(row.group_index or 0)

    return {
        worker_id: {
            instance_id: ModelInstanceWorkerMatch(
                is_main_worker=0 in group_indexes,
                # group_index i is subordinate i-1; 0 is the leader, which is
                # the instance's own worker rather than an entry in the list.
                subordinate_worker_indexes=tuple(
                    sorted(i - 1 for i in group_indexes if i != 0)
                ),
            )
            for instance_id, group_indexes in by_instance.items()
        }
        for worker_id, by_instance in positions.items()
    }


def report_match_disagreement(
    instance: ModelInstance,
    worker_id: Optional[int],
    embedded: ModelInstanceWorkerMatch,
    from_workloads: Optional[ModelInstanceWorkerMatch],
):
    """Say where the rows and the embedded list place a worker differently.

    The embedded list stays authoritative while this is only being watched, on
    the same footing as the state fold: silence is what says the rows can be
    read instead.
    """
    if from_workloads is None or from_workloads == embedded:
        return
    logger.info(
        f"Workload rows place worker {worker_id} differently on model instance "
        f"{instance.name} (id={instance.id}): embedded {embedded}, "
        f"rows {from_workloads}"
    )


@dataclass(frozen=True)
class InstancePlacement:
    """
    One worker's part in a model instance.

    The instance describes its own worker on itself and every other one in an
    embedded list, so a reader that wants "all of them" has to join the two
    shapes by hand -- and the ones that only walk the list quietly leave the
    instance's own worker out. This is the one shape, which is also what a
    workload row already is.
    """

    group_index: int
    """0 is the instance's own worker. Subordinate i is at i + 1, which is the
    off-by-one between this and the embedded list's own indexing."""

    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_ifname: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[list] = None
    gpu_addresses: Optional[list] = None
    computed_resource_claim: Optional[ComputedResourceClaim] = None
    ports: Optional[list] = None
    pid: Optional[int] = None
    arguments: Optional[list] = None
    state: Optional[ModelInstanceStateEnum] = None
    state_message: Optional[str] = None
    download_progress: Optional[float] = None

    @property
    def is_leader(self) -> bool:
        return self.group_index == 0

    @property
    def subordinate_index(self) -> Optional[int]:
        """Its position in the embedded list, or None for the leader, which
        has no entry there."""
        return None if self.is_leader else self.group_index - 1


def instance_placements(instance: ModelInstance) -> list:
    """
    Every worker that runs part of this instance, leader first.

    Reads the instance and its embedded subordinate list. The workload rows
    carry the same thing and will replace this reading; keeping the callers on
    one shape is what makes that a change here rather than in each of them.
    """
    placements = [
        InstancePlacement(
            group_index=0,
            worker_id=instance.worker_id,
            worker_name=instance.worker_name,
            worker_ip=instance.worker_ip,
            worker_ifname=instance.worker_ifname,
            gpu_type=instance.gpu_type,
            gpu_indexes=instance.gpu_indexes,
            gpu_addresses=instance.gpu_addresses,
            computed_resource_claim=instance.computed_resource_claim,
            ports=instance.ports,
            pid=instance.pid,
            state=instance.state,
            state_message=instance.state_message,
            download_progress=instance.download_progress,
        )
    ]
    return placements + subordinate_placements(instance)


def subordinate_placements(instance: ModelInstance) -> list:
    """
    Everything but the instance's own worker.

    Built without the leader rather than filtered out of the full list: the
    callers that want only these are on the worker, holding an instance whose
    own fields they have no business reading, and constructing the leader to
    discard it would make them depend on all of them.
    """
    subordinates = (
        instance.distributed_servers.subordinate_workers
        if instance.distributed_servers
        and instance.distributed_servers.subordinate_workers
        else []
    )
    return [
        InstancePlacement(
            group_index=index + 1,
            worker_id=sw.worker_id,
            worker_name=sw.worker_name,
            worker_ip=sw.worker_ip,
            worker_ifname=sw.worker_ifname,
            gpu_type=sw.gpu_type,
            gpu_indexes=sw.gpu_indexes,
            gpu_addresses=sw.gpu_addresses,
            computed_resource_claim=sw.computed_resource_claim,
            ports=sw.ports,
            pid=sw.pid,
            arguments=sw.arguments,
            state=sw.state,
            state_message=sw.state_message,
            download_progress=sw.download_progress,
        )
        for index, sw in enumerate(subordinates)
    ]
