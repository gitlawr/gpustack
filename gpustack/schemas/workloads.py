"""
The Workload resource: what a worker runs, and nothing about why.

A workload carries execution semantics only -- the binding a scheduler or a
controller decided on, the restart policy, and the execution state. The
domain resources (model deployments, benchmarks, cache services) stay the
user-facing API and compile into workloads; users do not create these.

See docs/proposals/workload-resource.md.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy import JSON, Column, Index, String, Text, UniqueConstraint
from sqlmodel import Field, SQLModel

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    PaginatedList,
    UTCDateTime,
    pydantic_column_type,
)


class WorkloadOwnerKindEnum(str, Enum):
    """Which kind of domain resource a workload was compiled from."""

    CACHE_SERVICE = "cache_service"
    MODEL_INSTANCE = "model_instance"
    BENCHMARK = "benchmark"

    def __str__(self):
        # The generated clients filter their watch-backed cache by comparing
        # str(attribute) with str(the queried value), so an enum that renders
        # as "ClassName.MEMBER" silently matches nothing and a filtered read
        # comes back empty.
        return self.value


class WorkloadStateEnum(str, Enum):
    """
    The execution subset of a domain resource's lifecycle.

    Deliberately smaller than, say, ModelInstanceStateEnum: analysing and
    scheduling happen before a container exists, and downloading model files is
    not execution. Those stay on the domain resource, which keeps its own
    lifecycle and maps into this one.
    """

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    UNREACHABLE = "unreachable"
    SUCCEEDED = "succeeded"
    """Ran to completion. Only task workloads reach it."""
    ERROR = "error"

    def __str__(self):
        # The generated clients filter their watch-backed cache by comparing
        # str(attribute) with str(the queried value), so an enum that renders
        # as "ClassName.MEMBER" silently matches nothing and a filtered read
        # comes back empty.
        return self.value


class WorkloadRestartPolicyEnum(str, Enum):
    """What a stop means."""

    ALWAYS = "always"
    """Service workloads: any stop is a fault to recover from."""
    ON_FAILURE = "on_failure"
    NEVER = "never"

    def __str__(self):
        # The generated clients filter their watch-backed cache by comparing
        # str(attribute) with str(the queried value), so an enum that renders
        # as "ClassName.MEMBER" silently matches nothing and a filtered read
        # comes back empty.
        return self.value


class WorkloadRoleEnum(str, Enum):
    """A workload's part in its group. A standalone workload is the leader of
    a group of one, so consumers never special-case group size."""

    LEADER = "leader"
    FOLLOWER = "follower"

    def __str__(self):
        # The generated clients filter their watch-backed cache by comparing
        # str(attribute) with str(the queried value), so an enum that renders
        # as "ClassName.MEMBER" silently matches nothing and a filtered read
        # comes back empty.
        return self.value


class ReservedClaim(BaseModel):
    """
    Resources this workload holds on a worker it does not run a container on.

    Distributed backends that coordinate their own workers (Ray, for one) are
    given the nodes by the scheduler but start the processes themselves. The
    reservation has to be recorded somewhere for accounting, and it belongs to
    the workload that made it rather than to rows for containers gpustack does
    not run: a workload row means a container.
    """

    worker_id: int
    gpu_indexes: Optional[List[int]] = None
    gpu_addresses: Optional[List[str]] = None
    computed_resource_claim: Optional[Dict[str, Any]] = None


class WorkloadBase(SQLModel):
    name: str = Field(index=True)
    """The container name on the worker. Stable across restarts."""

    # -- Ownership -----------------------------------------------------------
    owner_kind: WorkloadOwnerKindEnum = Field(
        sa_column=Column(String(length=64), nullable=False)
    )
    owner_id: int
    """The domain resource this was compiled from. Not a foreign key: the
    target table depends on owner_kind, so the controller owns the lifetime.
    See the proposal's GC section."""

    owner_principal_id: Optional[int] = None
    """Copied from the owner at creation so list queries scope without a join
    to a table that differs per owner_kind."""

    cluster_id: Optional[int] = None

    # -- Grouping ------------------------------------------------------------
    group_key: Optional[str] = Field(default=None, index=True)
    """Ties the workloads of one distributed instance together. None for a
    workload that runs alone."""

    group_index: int = 0
    """Position in the group. 0 is the leader."""

    role: WorkloadRoleEnum = Field(
        default=WorkloadRoleEnum.LEADER,
        sa_column=Column(String(length=32), nullable=False, default="leader"),
    )

    # -- Binding result ------------------------------------------------------
    worker_id: Optional[int] = None
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = Field(sa_column=Column(JSON), default=None)
    gpu_addresses: Optional[List[str]] = Field(sa_column=Column(JSON), default=None)
    computed_resource_claim: Optional[Dict[str, Any]] = Field(
        sa_column=Column(JSON), default=None
    )
    reserved_claims: Optional[List[ReservedClaim]] = Field(
        sa_column=Column(pydantic_column_type(List[ReservedClaim])), default=None
    )

    # -- Spec ----------------------------------------------------------------
    restart_policy: WorkloadRestartPolicyEnum = Field(
        default=WorkloadRestartPolicyEnum.ALWAYS,
        sa_column=Column(String(length=32), nullable=False, default="always"),
    )
    active_deadline_seconds: Optional[int] = None
    """Wall-clock limit for a task workload. None means no limit, which is
    what a service wants."""

    spec_digest: Optional[str] = None
    """Digest of the owner's container-shaping spec as of creation. The
    controller reconciles the workload *set*, not the spec, so this is how
    drift is made visible instead of silent."""

    labels: Optional[Dict[str, str]] = Field(sa_column=Column(JSON), default=None)

    # -- Execution status ----------------------------------------------------
    state: WorkloadStateEnum = Field(
        default=WorkloadStateEnum.PENDING,
        sa_column=Column(String(length=64), nullable=False),
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    ports: Optional[Dict[str, int]] = Field(sa_column=Column(JSON), default=None)
    """Named ports allocated on the worker, e.g. {"service": 40001}. The
    first port is named generically; the rest are named by the compiler that
    knows what they are."""

    pid: Optional[int] = None
    arguments: Optional[List[str]] = Field(sa_column=Column(JSON), default=None)

    restart_count: int = 0
    """Monotonic. Numbers the log files of each start, so it must not be reset;
    the consecutive-crash count the backoff runs on is separate."""

    last_restart_time: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    healthy: Optional[bool] = None
    last_check_at: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    progress: Optional[float] = None
    """Provisioning progress, 0-100, for the preparation a workload does
    before its container exists."""


class Workload(WorkloadBase, BaseModelMixin, table=True):
    __tablename__ = "workloads"
    __table_args__ = (
        UniqueConstraint(
            "owner_kind",
            "owner_id",
            "worker_id",
            "group_index",
            name="uix_workloads_owner_worker_group_index",
        ),
        # The worker's reconcile pass and orphan cleanup, the hottest read.
        Index("ix_workloads_worker_id", "worker_id"),
        # Controller fan-out and state aggregation.
        Index("ix_workloads_owner", "owner_kind", "owner_id"),
        # Endpoint resolution runs this on every model instance scheduled.
        Index("ix_workloads_owner_state", "owner_kind", "owner_id", "state"),
        Index("ix_workloads_cluster_id", "cluster_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)


class WorkloadCreate(WorkloadBase):
    pass


class WorkloadUpdate(WorkloadBase):
    pass


class WorkloadPublic(WorkloadBase):
    id: int
    created_at: datetime
    updated_at: datetime


WorkloadsPublic = PaginatedList[WorkloadPublic]
