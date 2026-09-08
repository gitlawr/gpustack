import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from gpustack.schemas.models import ModelInstance
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
