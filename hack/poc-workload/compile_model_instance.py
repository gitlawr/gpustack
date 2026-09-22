"""
ModelInstance <-> Workload, both directions.

Compilation turns one model instance (with its embedded subordinate workers)
into the group of workloads that runs it; aggregation folds the group's
execution state back onto the instance. Round-tripping the two against the
current behaviour is what stage 0 of the proposal is for.

The reference behaviour being reproduced:

- ``ModelInstance.get_deployment_metadata`` (schemas/models.py) for names and
  roles.
- ``ServeManager._dispatch_model_instance_event`` for start ordering.
- ``ServeManager._get_main_worker_distributed_state`` for state aggregation.
"""

from typing import List, Optional

from gpustack.schemas.models import (
    DistributedServerCoordinateModeEnum,
    ModelInstance,
    ModelInstanceStateEnum,
)

from workload import (
    RestartPolicyEnum,
    StartDependency,
    StartGateEnum,
    Workload,
    WorkloadRoleEnum,
    WorkloadStateEnum,
)

OWNER_KIND = "model_instance"

# ModelInstance states that have no workload counterpart: they happen before a
# container exists. The instance keeps them; the workload never sees them.
PRE_EXECUTION_STATES = {
    ModelInstanceStateEnum.PENDING,
    ModelInstanceStateEnum.ANALYZING,
    ModelInstanceStateEnum.SCHEDULED,
    ModelInstanceStateEnum.INITIALIZING,
    ModelInstanceStateEnum.DOWNLOADING,
}

_INSTANCE_TO_WORKLOAD_STATE = {
    ModelInstanceStateEnum.STARTING: WorkloadStateEnum.STARTING,
    ModelInstanceStateEnum.RUNNING: WorkloadStateEnum.RUNNING,
    ModelInstanceStateEnum.UNREACHABLE: WorkloadStateEnum.UNREACHABLE,
    ModelInstanceStateEnum.ERROR: WorkloadStateEnum.ERROR,
}

_WORKLOAD_TO_INSTANCE_STATE = {v: k for k, v in _INSTANCE_TO_WORKLOAD_STATE.items()}


def _to_workload_state(state: ModelInstanceStateEnum) -> WorkloadStateEnum:
    return _INSTANCE_TO_WORKLOAD_STATE.get(state, WorkloadStateEnum.PENDING)


def _to_instance_state(state: WorkloadStateEnum) -> Optional[ModelInstanceStateEnum]:
    return _WORKLOAD_TO_INSTANCE_STATE.get(state)


def compile_model_instance(mi: ModelInstance) -> List[Workload]:
    """
    The workloads that run a model instance.

    group_index 0 is the leader (the instance's own worker); subordinate
    worker i becomes group_index i+1. Names match
    ``get_deployment_metadata``: the instance name for the leader, and
    ``<name>-f<i>`` for follower i, so the container names do not change.
    """
    dservers = mi.distributed_servers
    subordinates = (dservers.subordinate_workers if dservers else None) or []
    mode = dservers.mode if dservers else None
    distributed = bool(subordinates)

    leader = Workload(
        name=mi.name,
        owner_kind=OWNER_KIND,
        owner_id=mi.id,
        owner_principal_id=mi.owner_principal_id,
        cluster_id=mi.cluster_id,
        group_key=mi.name if distributed else None,
        group_index=0,
        role=WorkloadRoleEnum.LEADER,
        worker_id=mi.worker_id,
        worker_name=mi.worker_name,
        worker_ip=mi.worker_ip,
        worker_ifname=mi.worker_ifname,
        gpu_type=mi.gpu_type,
        gpu_indexes=list(mi.gpu_indexes or []),
        gpu_addresses=list(mi.gpu_addresses or []),
        computed_resource_claim=mi.computed_resource_claim,
        restart_policy=RestartPolicyEnum.ALWAYS,
        state=_to_workload_state(mi.state),
        state_message=mi.state_message,
        ports=_named_ports(mi.port, mi.ports),
        pid=mi.pid,
        restart_count=mi.restart_count or 0,
        last_restart_time=mi.last_restart_time,
        progress=mi.download_progress,
    )

    workloads = [leader]
    for index, sw in enumerate(subordinates):
        workloads.append(
            Workload(
                name=f"{mi.name}-f{index}",
                owner_kind=OWNER_KIND,
                owner_id=mi.id,
                owner_principal_id=mi.owner_principal_id,
                cluster_id=mi.cluster_id,
                group_key=mi.name,
                group_index=index + 1,
                role=WorkloadRoleEnum.FOLLOWER,
                # DELEGATED hands the subordinate containers to another
                # framework; the row is only a resource reservation.
                managed=mode != DistributedServerCoordinateModeEnum.DELEGATED,
                worker_id=sw.worker_id,
                worker_name=sw.worker_name,
                worker_ip=sw.worker_ip,
                worker_ifname=sw.worker_ifname,
                gpu_type=sw.gpu_type,
                gpu_indexes=list(sw.gpu_indexes or []),
                gpu_addresses=list(sw.gpu_addresses or []),
                computed_resource_claim=sw.computed_resource_claim,
                restart_policy=RestartPolicyEnum.ALWAYS,
                state=_to_workload_state(sw.state),
                state_message=sw.state_message,
                ports=_named_ports(None, sw.ports),
                pid=sw.pid,
                arguments=list(sw.arguments or []),
                progress=sw.download_progress,
            )
        )

    _apply_start_ordering(workloads, mode)
    return workloads


def _named_ports(port: Optional[int], ports: Optional[List[int]]) -> dict:
    """
    Model instances carry ``port`` plus an unnamed ``ports`` list.

    ``_assign_ports`` sets ``mi.ports = [mi.port]`` and then extends, so
    ``port`` is always ``ports[0]``: they are one port, not two. Only that
    first one can be named generically. Everything after it is laid out by
    backend and executor -- vLLM/mp gets DP-RPC, master-port and VLLM_PORT,
    vLLM/ray gets DP-RPC only when dp > 1, other backends get none, and the
    connecting port is always last -- so a generic compiler cannot name them
    without mislabelling most backends. They keep their position until each
    backend's own compiler names them.
    """
    values = list(ports or [])
    if port is not None and (not values or values[0] != port):
        values.insert(0, port)
    return {
        ("service" if index == 0 else f"port{index}"): value
        for index, value in enumerate(values)
    }


def _apply_start_ordering(
    workloads: List[Workload],
    mode: Optional[DistributedServerCoordinateModeEnum],
):
    """
    Turn the coordinate mode into per-workload start dependencies.

    This is the crux of the proposal: today the ordering is re-derived on every
    event by each worker reading the whole instance (serve_manager's
    ``_dispatch_model_instance_event``). As dependencies it is decided once, by
    the controller, and the worker only ever sees a workload that is allowed to
    start.
    """
    if len(workloads) == 1 or mode is None:
        return

    leader, followers = workloads[0], workloads[1:]

    if mode == DistributedServerCoordinateModeEnum.DELEGATED:
        # Nothing gpustack starts, nothing to order.
        return

    if mode == DistributedServerCoordinateModeEnum.RUN_FIRST:
        leader.start_after = [
            StartDependency(f.group_index, StartGateEnum.READY) for f in followers
        ]
        return

    # INITIALIZE_LATER: a follower waits for the leader to have begun, and for
    # every earlier follower to have settled. The second half is the
    # "phantom read" guard in _dispatch_model_instance_event.
    for position, follower in enumerate(followers):
        follower.start_after = [StartDependency(0, StartGateEnum.STARTED)] + [
            StartDependency(other.group_index, StartGateEnum.READY)
            for other in followers[:position]
        ]


def gate_satisfied(dependency: StartDependency, state: WorkloadStateEnum) -> bool:
    if dependency.gate == StartGateEnum.STARTED:
        return state in (
            WorkloadStateEnum.STARTING,
            WorkloadStateEnum.RUNNING,
            WorkloadStateEnum.UNREACHABLE,
            WorkloadStateEnum.SUCCEEDED,
            WorkloadStateEnum.ERROR,
        )
    return state in (
        WorkloadStateEnum.RUNNING,
        WorkloadStateEnum.SUCCEEDED,
        WorkloadStateEnum.ERROR,
    )


def resolve_blocked(workloads: List[Workload]) -> List[Workload]:
    """
    Controller step: flip BLOCKED workloads to PENDING once their dependencies
    are satisfied. Returns the ones that changed.
    """
    by_index = {w.group_index: w for w in workloads}
    released = []
    for workload in workloads:
        if workload.state != WorkloadStateEnum.BLOCKED:
            continue
        if all(
            gate_satisfied(dep, by_index[dep.group_index].state)
            for dep in workload.start_after
        ):
            workload.state = WorkloadStateEnum.PENDING
            released.append(workload)
    return released


def aggregate_group_state(workloads: List[Workload]) -> Optional[dict]:
    """
    Fold a group's workload states back onto the model instance.

    Reproduces ``ServeManager._get_main_worker_distributed_state``: the first
    follower in ERROR wins, then the first UNREACHABLE, and anything short of
    all-followers-RUNNING holds the instance where it is. Returns None when the
    leader's own state governs.
    """
    followers = [w for w in workloads if w.role == WorkloadRoleEnum.FOLLOWER]
    if not followers:
        return None

    error = next((w for w in followers if w.state == WorkloadStateEnum.ERROR), None)
    if error:
        return {
            "state": ModelInstanceStateEnum.ERROR,
            "state_message": (
                f"Distributed serving error in subordinate worker "
                f"{error.worker_ip}: {error.state_message}."
            ),
        }

    # A follower in ERROR outranks one merely unreachable, so the scan for
    # unreachable only runs once no follower errored.
    unreachable = next(
        (w for w in followers if w.state == WorkloadStateEnum.UNREACHABLE), None
    )
    if unreachable:
        return {
            "state": ModelInstanceStateEnum.UNREACHABLE,
            "state_message": (
                f"Distributed serving unreachable in subordinate worker "
                f"{unreachable.worker_ip}: {unreachable.state_message}."
            ),
        }

    if not all(w.state == WorkloadStateEnum.RUNNING for w in followers):
        return {"hold": True}

    return None
