"""
Proposed Workload resource, as a plain dataclass.

Throwaway: this is the POC for docs/proposals/workload-resource.md stage 0. It
exists to be compiled into and aggregated from, so the field set can be judged
against real model instances instead of on paper. Nothing here is wired to the
database, the API, or the worker.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkloadStateEnum(str, Enum):
    """
    The execution subset of a domain resource's lifecycle.

    Deliberately smaller than ModelInstanceStateEnum: ANALYZING and SCHEDULED
    happen before a container exists, and DOWNLOADING is model-file
    preparation. Those stay on the domain resource.
    """

    BLOCKED = "blocked"
    """Created, but a start dependency is not satisfied yet. See
    ``start_after``. The worker ignores workloads in this state."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    UNREACHABLE = "unreachable"
    SUCCEEDED = "succeeded"
    ERROR = "error"


class RestartPolicyEnum(str, Enum):
    ALWAYS = "always"
    ON_FAILURE = "on_failure"
    NEVER = "never"


class StartGateEnum(str, Enum):
    """How far a depended-on workload must get before its dependent may start.

    Two kinds, because the current coordinate modes need both: a follower may
    start once the leader has *begun* (INITIALIZE_LATER waits for the leader to
    reach STARTING), but must wait for a previous follower to have *settled*.
    """

    STARTED = "started"
    """Reached STARTING or beyond, including terminal failure."""

    READY = "ready"
    """Reached RUNNING, or failed trying."""


class WorkloadRoleEnum(str, Enum):
    """A workload's part in its group. Standalone workloads are LEADER of a
    group of one, so consumers never special-case group size."""

    LEADER = "leader"
    FOLLOWER = "follower"


@dataclass(frozen=True)
class StartDependency:
    group_index: int
    gate: "StartGateEnum"


@dataclass
class Workload:
    id: Optional[int] = None
    name: Optional[str] = None

    # -- Ownership -----------------------------------------------------------
    owner_kind: Optional[str] = None
    owner_id: Optional[int] = None
    owner_principal_id: Optional[int] = None
    cluster_id: Optional[int] = None

    # -- Grouping (distributed instances) ------------------------------------
    group_key: Optional[str] = None
    group_index: int = 0
    role: WorkloadRoleEnum = WorkloadRoleEnum.LEADER

    start_after: List["StartDependency"] = field(default_factory=list)
    """Dependencies that must be satisfied before this workload leaves
    BLOCKED. Evaluated by the controller, never by the worker: moving the
    distributed start ordering out of the worker is the whole point."""

    managed: bool = True
    """Whether gpustack runs the container. False marks a binding that only
    reserves resources, the container being another framework's business.
    See the DELEGATED finding in FINDINGS.md."""

    # -- Binding result ------------------------------------------------------
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_ifname: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_indexes: List[int] = field(default_factory=list)
    gpu_addresses: List[str] = field(default_factory=list)
    computed_resource_claim: Optional[Any] = None

    # -- Spec ----------------------------------------------------------------
    restart_policy: RestartPolicyEnum = RestartPolicyEnum.ALWAYS
    active_deadline_seconds: Optional[int] = None
    spec_digest: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)

    # -- Execution status ----------------------------------------------------
    state: WorkloadStateEnum = WorkloadStateEnum.PENDING
    state_message: Optional[str] = None
    ports: Dict[str, int] = field(default_factory=dict)
    pid: Optional[int] = None
    arguments: List[str] = field(default_factory=list)
    restart_count: int = 0
    last_restart_time: Optional[datetime] = None
    healthy: Optional[bool] = None
    last_check_at: Optional[datetime] = None
    progress: Optional[float] = None
    """Provisioning progress, 0-100. Carries model-file download progress for
    model instances. See the download_progress finding in FINDINGS.md."""
