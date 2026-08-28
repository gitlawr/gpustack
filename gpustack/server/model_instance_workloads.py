"""
Compiling a model instance into the workloads that run it.

A model instance is one logical replica; the containers that serve it are one
or more workloads. A single-worker instance compiles to one, and a distributed
one to a leader plus a follower per subordinate worker, tied together by a
group key -- which is what the embedded ``subordinate_workers`` list and the
indexed path patches around it become.

Validated against real distributed instances in stage 0 of
docs/proposals/workload-resource.md; the findings there are why the names,
the ports and the reservations are shaped the way they are.
"""

import logging
from typing import Dict, List, Optional

from gpustack.schemas.models import (
    DistributedServerCoordinateModeEnum,
    ModelInstance,
    ModelInstanceStateEnum,
)
from gpustack.schemas.workloads import (
    ReservedClaim,
    Workload,
    WorkloadOwnerKindEnum,
    WorkloadRestartPolicyEnum,
    WorkloadRoleEnum,
    WorkloadStateEnum,
    WorkloadUpdate,
)

logger = logging.getLogger(__name__)

SERVICE_PORT = "service"
"""The instance's HTTP port. ``_assign_ports`` makes ``mi.port`` the first
entry of ``mi.ports``, so they are one port; the rest are laid out by backend
and executor and keep their position until each backend names its own."""


PRE_EXECUTION_STATES = frozenset(
    {
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.ANALYZING,
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.INITIALIZING,
        ModelInstanceStateEnum.DOWNLOADING,
    }
)
"""Instance states that happen before a container exists -- scheduling, and
preparing model files. A workload has no counterpart for them; it is simply
pending."""

_TO_WORKLOAD_STATE = {
    ModelInstanceStateEnum.STARTING: WorkloadStateEnum.STARTING,
    ModelInstanceStateEnum.RUNNING: WorkloadStateEnum.RUNNING,
    ModelInstanceStateEnum.UNREACHABLE: WorkloadStateEnum.UNREACHABLE,
    ModelInstanceStateEnum.ERROR: WorkloadStateEnum.ERROR,
}


def to_workload_state(state: Optional[ModelInstanceStateEnum]) -> WorkloadStateEnum:
    return _TO_WORKLOAD_STATE.get(state, WorkloadStateEnum.PENDING)


def named_ports(port: Optional[int], ports: Optional[List[int]]) -> Dict[str, int]:
    """
    ``port`` plus the unnamed ``ports`` list as one named map.

    ``port`` is always ``ports[0]``, so only the first is named here.
    Everything after it is laid out by backend and executor -- vLLM/mp gets
    DP-RPC, master-port and VLLM_PORT, vLLM/ray gets DP-RPC only when dp > 1,
    other backends get none, and the connecting port is always last -- so
    naming them generically would mislabel most of them.
    """
    values = list(ports or [])
    if port is not None and (not values or values[0] != port):
        values.insert(0, port)
    return {
        (SERVICE_PORT if index == 0 else f"port{index}"): value
        for index, value in enumerate(values)
    }


def compile_model_instance(mi: ModelInstance) -> List[Workload]:
    """
    The workloads that run a model instance.

    group_index 0 is the leader, on the instance's own worker; subordinate
    worker i becomes group_index i+1. Names match ``get_deployment_metadata``
    -- the instance name for the leader, ``<name>-f<i>`` for follower i -- so
    the containers keep the names they already have.
    """
    dservers = mi.distributed_servers
    subordinates = (dservers.subordinate_workers if dservers else None) or []
    mode = dservers.mode if dservers else None
    distributed = bool(subordinates)

    # DELEGATED hands the subordinate containers to another framework, so
    # gpustack runs one container and merely holds the resources on the other
    # nodes. Those are the leader's reservations, not workloads of their own:
    # a workload row means a container gpustack runs.
    delegated = mode == DistributedServerCoordinateModeEnum.DELEGATED
    reserved = (
        [
            ReservedClaim(
                worker_id=sw.worker_id,
                gpu_indexes=list(sw.gpu_indexes or []),
                gpu_addresses=list(sw.gpu_addresses or []),
                computed_resource_claim=(
                    sw.computed_resource_claim.model_dump()
                    if sw.computed_resource_claim
                    else None
                ),
            )
            for sw in subordinates
        ]
        if delegated
        else None
    )

    leader = Workload(
        name=mi.name,
        owner_kind=WorkloadOwnerKindEnum.MODEL_INSTANCE,
        owner_id=mi.id,
        owner_principal_id=mi.owner_principal_id,
        cluster_id=mi.cluster_id,
        group_key=mi.name if distributed and not delegated else None,
        group_index=0,
        role=WorkloadRoleEnum.LEADER,
        worker_id=mi.worker_id,
        gpu_type=mi.gpu_type,
        gpu_indexes=list(mi.gpu_indexes or []),
        gpu_addresses=list(mi.gpu_addresses or []),
        computed_resource_claim=(
            mi.computed_resource_claim.model_dump()
            if mi.computed_resource_claim
            else None
        ),
        reserved_claims=reserved,
        restart_policy=WorkloadRestartPolicyEnum.ALWAYS,
        state=to_workload_state(mi.state),
        state_message=mi.state_message,
        ports=named_ports(mi.port, mi.ports) or None,
        pid=mi.pid,
        restart_count=mi.restart_count or 0,
        last_restart_time=mi.last_restart_time,
        progress=mi.download_progress,
    )
    if delegated:
        return [leader]

    workloads = [leader]
    for index, sw in enumerate(subordinates):
        workloads.append(
            Workload(
                name=f"{mi.name}-f{index}",
                owner_kind=WorkloadOwnerKindEnum.MODEL_INSTANCE,
                owner_id=mi.id,
                owner_principal_id=mi.owner_principal_id,
                cluster_id=mi.cluster_id,
                group_key=mi.name,
                group_index=index + 1,
                role=WorkloadRoleEnum.FOLLOWER,
                worker_id=sw.worker_id,
                gpu_type=sw.gpu_type,
                gpu_indexes=list(sw.gpu_indexes or []),
                gpu_addresses=list(sw.gpu_addresses or []),
                computed_resource_claim=(
                    sw.computed_resource_claim.model_dump()
                    if sw.computed_resource_claim
                    else None
                ),
                restart_policy=WorkloadRestartPolicyEnum.ALWAYS,
                state=to_workload_state(sw.state),
                state_message=sw.state_message,
                ports=named_ports(None, sw.ports) or None,
                pid=sw.pid,
                arguments=list(sw.arguments or []),
                progress=sw.download_progress,
            )
        )
    return workloads


SPEC_FIELDS = frozenset(
    {
        "name",
        "owner_kind",
        "owner_id",
        "owner_principal_id",
        "cluster_id",
        "group_key",
        "group_index",
        "role",
        "worker_id",
        "gpu_type",
        "gpu_indexes",
        "gpu_addresses",
        "computed_resource_claim",
        "reserved_claims",
        "restart_policy",
        "active_deadline_seconds",
        "spec_digest",
        "labels",
    }
)
"""What a workload is asked to be, as opposed to what it reports. The worker
owns the rest, so recompiling must not write over it."""


def workload_spec(workload: Workload) -> WorkloadUpdate:
    """The spec half of a compiled workload, for updating an existing row."""
    return WorkloadUpdate(
        **{
            name: getattr(workload, name)
            for name in SPEC_FIELDS
            if hasattr(workload, name)
        }
    )


def aggregate_instance_state(workloads: List[Workload]) -> Optional[dict]:
    """
    Fold a group's execution state back onto its model instance.

    The leader reports the instance's own state; the followers can override it,
    reproducing what the worker decides today in
    ``ServeManager._get_main_worker_distributed_state``: the first follower in
    ERROR wins, then the first UNREACHABLE, and anything short of
    all-followers-RUNNING holds the instance where it is.

    Returns the fields to write, or None when the group says nothing yet --
    which is not the same as "nothing changed"; that comparison is the
    caller's, since only it knows what the instance currently says.
    """
    leader = next((w for w in workloads if w.group_index == 0), None)
    if leader is None:
        return None

    if leader.state == WorkloadStateEnum.PENDING:
        # The mapping is not reversible here: scheduling, initializing and
        # downloading all mirror onto a pending workload, so folding that back
        # would replace the instance's richer state with the poorer one. Those
        # states belong to the instance's own lifecycle; the workload has
        # nothing to say until its container exists.
        return None

    followers = sorted(
        (w for w in workloads if w.group_index != 0), key=lambda w: w.group_index
    )
    override = _distributed_override(followers)
    if override is _HOLD:
        return None

    fields = dict(override) if override else {"state": leader.state.value}
    if not override:
        fields["state_message"] = leader.state_message
    return fields


_HOLD = object()
"""A group that is still coming up: the instance keeps whatever it says."""


def _distributed_override(followers: List[Workload]) -> Optional[dict]:
    if not followers:
        return None

    error = next((w for w in followers if w.state == WorkloadStateEnum.ERROR), None)
    if error:
        return {
            "state": ModelInstanceStateEnum.ERROR,
            "state_message": (
                f"Distributed serving error in subordinate worker "
                f"{error.worker_id}: {error.state_message}."
            ),
        }

    # A follower in ERROR outranks one merely unreachable, so this only runs
    # once none of them errored.
    unreachable = next(
        (w for w in followers if w.state == WorkloadStateEnum.UNREACHABLE), None
    )
    if unreachable:
        return {
            "state": ModelInstanceStateEnum.UNREACHABLE,
            "state_message": (
                f"Distributed serving unreachable in subordinate worker "
                f"{unreachable.worker_id}: {unreachable.state_message}."
            ),
        }

    if not all(w.state == WorkloadStateEnum.RUNNING for w in followers):
        return _HOLD
    return None
