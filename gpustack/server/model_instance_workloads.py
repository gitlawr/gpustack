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
from enum import Enum
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
        ModelInstanceStateEnum.DOWNLOADING,
        # Despite the name, the server sets this when the model files are
        # ready, before any worker has launched anything. The state that means
        # "the worker spawned the process" is INITIALIZING.
        ModelInstanceStateEnum.STARTING,
    }
)
"""Instance states that happen before a container exists -- scheduling, and
preparing model files. A workload has no counterpart for them; it is simply
pending."""

AWAITING_EXECUTION_STATES = frozenset(
    {
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.ANALYZING,
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.DOWNLOADING,
        ModelInstanceStateEnum.INITIALIZING,
    }
)
"""Instance states in which nothing is waiting on a container's health, so
whatever the workloads report is about a run that is over -- after a restart
they still carry the failure that caused it.

Not the same set as PRE_EXECUTION_STATES, and the difference is the point:
the instance's own STARTING belongs there and not here, because that is
exactly where it waits for its container to be reported running."""

_TO_WORKLOAD_STATE = {
    ModelInstanceStateEnum.INITIALIZING: WorkloadStateEnum.STARTING,
    ModelInstanceStateEnum.RUNNING: WorkloadStateEnum.RUNNING,
    ModelInstanceStateEnum.UNREACHABLE: WorkloadStateEnum.UNREACHABLE,
    ModelInstanceStateEnum.ERROR: WorkloadStateEnum.ERROR,
}
"""Instance state to the workload's. Paired with _TO_INSTANCE_STATE below;
the two names that coincide do not mean the same thing, which is why neither
direction is derived from the other."""

_NOT_REVERSIBLE = frozenset({WorkloadStateEnum.PENDING, WorkloadStateEnum.STARTING})
"""Workload states that more than one instance state maps onto."""

_TO_INSTANCE_STATE = {
    # No entry for STARTING: two instance states mirror onto it and _HOLD
    # below turns it away before it gets here.
    WorkloadStateEnum.RUNNING: ModelInstanceStateEnum.RUNNING,
    WorkloadStateEnum.UNREACHABLE: ModelInstanceStateEnum.UNREACHABLE,
    WorkloadStateEnum.ERROR: ModelInstanceStateEnum.ERROR,
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


def instance_ports(named: Optional[Dict[str, int]]):
    """
    ``named_ports`` read backwards: the (port, ports) an instance carries.

    Only the first is named, so the rest are recovered by position. A gap in
    the sequence ends it rather than being skipped -- the list is positional
    on the instance and closing a hole would shift everything after it.
    """
    if not named:
        return None, None
    ordered = []
    if SERVICE_PORT in named:
        ordered.append(named[SERVICE_PORT])
    index = 1
    while f"port{index}" in named:
        ordered.append(named[f"port{index}"])
        index += 1
    if not ordered:
        return None, None
    return ordered[0], ordered


_RUNTIME_FIELDS = ("pid",)

_MONOTONIC_RUNTIME_FIELDS = ("restart_count", "last_restart_time")
"""Restart bookkeeping only ever moves forward. The row is filled by the
mirror and so lags the instance, and folding it back would reset the count the
backoff escalates on -- an instance would restart at the base delay for
ever."""


def aggregate_instance_runtime(workloads: List[Workload], instance=None) -> dict:
    """
    What the leader's container is, as opposed to how it is doing.

    The worker writes these onto the instance today and stops at stage 3 step
    4; the fold has to carry them or ``mi.port`` -- which is what requests are
    proxied to -- goes stale the moment it does.

    A field the row has nothing for is left out rather than written as None.
    The row is filled by the mirror, so it lags the instance by one write, and
    clearing a live port because the row has not caught up would take the
    instance off the air for a reason that has nothing to do with it.

    The restart bookkeeping is held to moving forward for the same reason,
    given the instance to compare against: a row a write behind reports the
    count from before the restart, and folding that back resets what the
    backoff escalates on, leaving an instance restarting at the base delay for
    ever. Seen as restart_count going from 1 to 0 on a real run.
    """
    leader = next((w for w in workloads if (w.group_index or 0) == 0), None)
    if leader is None:
        return {}

    runtime = {
        name: getattr(leader, name, None)
        for name in _RUNTIME_FIELDS
        if getattr(leader, name, None) is not None
    }
    for name in _MONOTONIC_RUNTIME_FIELDS:
        value = getattr(leader, name, None)
        current = getattr(instance, name, None) if instance is not None else None
        if value is not None and (current is None or value >= current):
            runtime[name] = value
    port, ports = instance_ports(leader.ports)
    if port is not None:
        runtime["port"] = port
    if ports:
        runtime["ports"] = ports
    return runtime


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
        worker_name=mi.worker_name,
        worker_ip=mi.worker_ip,
        worker_ifname=mi.worker_ifname,
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
                worker_name=sw.worker_name,
                worker_ip=sw.worker_ip,
                worker_ifname=sw.worker_ifname,
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
        "worker_name",
        "worker_ip",
        "worker_ifname",
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


async def sync_model_instance_workloads(session, instance: ModelInstance) -> None:
    """
    Bring an instance's workload rows in line with its binding.

    Shared by the scheduler, which calls it in the transaction that writes the
    binding, and by the controller, which is the level-triggered backstop for
    everything the scheduler is not the source of. Without the scheduler call
    the rows appear a controller hop after the instance is bound, and the
    worker can start inside that window -- with nowhere to report to.

    Idempotent: an existing row is updated in place, so the id the worker
    reports against does not move.
    """
    existing = await Workload.all_by_fields(
        session,
        {
            "owner_kind": WorkloadOwnerKindEnum.MODEL_INSTANCE,
            "owner_id": instance.id,
        },
    )
    by_group_index = {workload.group_index: workload for workload in existing}

    for compiled in compile_model_instance(instance):
        current = by_group_index.pop(compiled.group_index, None)
        if current is None:
            await Workload.create(session, compiled)
            continue
        # Spec and binding only. Execution state is the worker's to write --
        # it mirrors it onto these rows as it goes -- and recompiling it from
        # the instance would overwrite what the worker just reported.
        spec = workload_spec(compiled)
        if spec_differs(current, spec):
            await current.update(session, spec)

    # A distributed instance that lost subordinate workers, or a backend that
    # started delegating, leaves rows behind.
    for stale in by_group_index.values():
        await stale.delete(session)


def spec_differs(current, spec: WorkloadUpdate) -> bool:
    """
    Whether the row already says what the spec asks for.

    ``update`` publishes an event whether or not anything changed, and this
    runs on every event about the owning resource. Writing unconditionally
    would turn one instance event into an event per workload it has, each of
    which wakes the fold that reads them back.
    """
    return any(
        getattr(current, name, None) != getattr(spec, name)
        for name in spec.model_fields_set
    )


def workload_spec(workload: Workload) -> WorkloadUpdate:
    """The spec half of a compiled workload, for updating an existing row."""
    return WorkloadUpdate(
        **{
            name: getattr(workload, name)
            for name in SPEC_FIELDS
            if hasattr(workload, name)
        }
    )


def aggregate_instance_state(
    workloads: List[Workload], worker_ips: Optional[Dict[int, str]] = None
) -> Optional[dict]:
    """
    Fold a group's execution state back onto its model instance.

    ``worker_ips`` names the workers a failing follower runs on. The message
    the worker writes today identifies them by IP, and a workload row carries
    only the id, so the caller resolves them; without the map the id is used,
    which changes a user-visible string and is only acceptable in tests.

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

    if leader.state in _NOT_REVERSIBLE:
        # Coming up is where the instance's lifecycle is richer than the
        # workload's, and the mapping back is not a function. Scheduling and
        # file preparation all mirror onto a pending workload; INITIALIZING
        # and the instance's own STARTING both mirror onto a starting one,
        # since the workload has a single state for "launched, not yet
        # healthy". Folding either back would pick one of the two and
        # overwrite the other. Those states belong to the domain resource,
        # which keeps writing them; the workload speaks once it is running,
        # failed or unreachable.
        return None

    followers = sorted(
        (w for w in workloads if w.group_index != 0), key=lambda w: w.group_index
    )
    override = _distributed_override(followers, worker_ips or {})
    if override is _HOLD:
        return None

    if override:
        return dict(override)

    state = _to_instance_state(leader.state)
    if state is None:
        return None
    # "" rather than None: an instance with no message carries the empty
    # string, and a workload that was never given one carries NULL. They mean
    # the same thing, and the fold has to produce the spelling the instance
    # already uses or every healthy instance reports a difference.
    return {"state": state, "state_message": leader.state_message or ""}


class FoldDeclineReason(str, Enum):
    """Why the fold had nothing to say. Kept apart because they are not equally
    benign: two are ordinary points in a start, and one means the group is
    missing the row everything else is derived from."""

    NO_LEADER = "no_leader"
    ROWS_BEHIND = "rows_behind"
    """The instance was written more recently than any of its rows: the worker
    writes it and then mirrors, so in between the rows describe the run before
    this one."""
    INSTANCE_NOT_EXECUTING = "instance_not_executing"
    """The domain resource is before execution -- rescheduled, or preparing
    model files -- so its workloads describe a run that is over."""
    LEADER_PENDING = "leader_pending"
    LEADER_STARTING = "leader_starting"
    FOLLOWERS_NOT_READY = "followers_not_ready"
    NOT_AN_INSTANCE_STATE = "not_an_instance_state"


def fold_decline_reason(workloads: List[Workload]) -> Optional[FoldDeclineReason]:
    """
    Which branch of ``aggregate_instance_state`` returned None.

    Diagnostic only: the fold's own silence is the evidence that it can be made
    authoritative, and silence for these four reasons means four different
    things. ``test_every_decline_branch_reports_its_reason`` pins this against
    the branches it describes, so the two cannot drift apart.
    """
    leader = next((w for w in workloads if w.group_index == 0), None)
    if leader is None:
        return FoldDeclineReason.NO_LEADER
    if leader.state == WorkloadStateEnum.PENDING:
        return FoldDeclineReason.LEADER_PENDING
    if leader.state == WorkloadStateEnum.STARTING:
        return FoldDeclineReason.LEADER_STARTING

    followers = sorted(
        (w for w in workloads if w.group_index != 0), key=lambda w: w.group_index
    )
    if _distributed_override(followers, {}) is _HOLD:
        return FoldDeclineReason.FOLLOWERS_NOT_READY
    if _to_instance_state(leader.state) is None:
        return FoldDeclineReason.NOT_AN_INSTANCE_STATE
    return None


def _to_instance_state(state) -> Optional[ModelInstanceStateEnum]:
    """
    A workload's state as the instance's own.

    Not a cast through the shared value: both enums have a STARTING and they
    do not name the same moment, and the instance has two states -- its own
    STARTING and INITIALIZING -- where the workload has one.

    None for a workload state the instance has no single name for, and for a
    plain string that names no member, since a row loaded from a rolled-back
    database can carry one.
    """
    try:
        return _TO_INSTANCE_STATE.get(WorkloadStateEnum(str(state)))
    except ValueError:
        return None


def _worker_label(workload: Workload, worker_ips: Dict[int, str]) -> str:
    """How the failing worker is named in the instance's state message. The
    worker writes the IP, so the fold has to as well or every follower failure
    reads as a disagreement -- and flipping the fold on would reword a message
    users already see."""
    return str(worker_ips.get(workload.worker_id, workload.worker_id))


_HOLD = object()
"""A group that is still coming up: the instance keeps whatever it says."""


def _distributed_override(
    followers: List[Workload], worker_ips: Dict[int, str]
) -> Optional[dict]:
    if not followers:
        return None

    error = next((w for w in followers if w.state == WorkloadStateEnum.ERROR), None)
    if error:
        return {
            "state": ModelInstanceStateEnum.ERROR,
            "state_message": (
                f"Distributed serving error in subordinate worker "
                f"{_worker_label(error, worker_ips)}: {error.state_message}."
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
                f"{_worker_label(unreachable, worker_ips)}: "
                f"{unreachable.state_message}."
            ),
        }

    if not all(w.state == WorkloadStateEnum.RUNNING for w in followers):
        return _HOLD
    return None


def rows_are_behind(instance, workloads: List[Workload]) -> bool:
    """
    Whether the instance has been written since its rows were.

    The worker writes the instance and then mirrors onto the row, so between
    the two they genuinely disagree, and a fold reading in that window reports
    the run before this one. While the fold only watched, that cost a logged
    difference the confirm absorbed; deciding, it writes -- a real run had it
    put a container that had just died back to RUNNING for sixty milliseconds.

    Comparing against the newest row rather than the leader's: a follower
    changing is news too, and the fold has to be free to act on it.

    This does not silence the fold in the steady state. Its own write makes
    the instance newer, so it says nothing until a row moves again -- which is
    exactly when it has something to say.
    """
    newest = max(
        (w.updated_at for w in workloads if getattr(w, "updated_at", None)),
        default=None,
    )
    if newest is None or getattr(instance, "updated_at", None) is None:
        return False
    return newest < instance.updated_at
