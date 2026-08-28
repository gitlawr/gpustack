import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy import JSON, Column, ForeignKey, Integer, String, UniqueConstraint
from sqlmodel import Field, SQLModel, Text

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import (
    ItemList,
    PaginatedList,
    UTCDateTime,
    pydantic_column_type,
)
from gpustack.schemas.principals import _platform_principal_id

CACHE_SERVICE_WORKLOAD_TYPE = "cache-service"


CACHE_SERVICE_PORT = "service"
CACHE_SERVICE_METRICS_PORT = "metrics"
"""Names of the ports a cache server exposes. The provider declaration's
health check target uses the same vocabulary."""


def cache_service_instance_workload_name(cache_service_id: int, worker_id: int) -> str:
    """
    Container name of the cache server a service runs on a worker.

    Keyed by worker rather than by row id so it can be computed before the row
    exists, and because that is the identity anyway: one cache server per
    service per worker, which is what the workload table's unique constraint
    says.
    """
    return f"cache-svc-{cache_service_id}-w{worker_id}"


def cache_service_workload_labels(
    cache_service_id: int, worker_id: int
) -> Dict[str, str]:
    """Labels stamped on the container, for the worker's orphan cleanup."""
    return {
        "type": CACHE_SERVICE_WORKLOAD_TYPE,
        "cache-service-id": str(cache_service_id),
        "cache-service-worker-id": str(worker_id),
    }


class CacheServiceModeEnum(str, Enum):
    MANAGED = "managed"
    EXTERNAL = "external"

    def __str__(self):
        return self.value


class CacheServiceStateEnum(str, Enum):
    r"""
    Enum for Cache Service / Cache Service Instance State

    Transitions (managed instances):

       |- Server -|- - - - - - Worker - - - - - -|
       |          |                              |
    PENDING ---> ---> STARTING ---> RUNNING / ERROR
                                       ^
                                       |(health probe)
                                       v
                                  UNREACHABLE

    A managed service's own state is an aggregate the server controller
    computes over its instances' states. External services skip STARTING:
    the health checker flips them between RUNNING and UNREACHABLE.
    """

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    UNREACHABLE = "unreachable"

    def __str__(self):
        return self.value


class CacheServiceEndpoint(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    url: Optional[str] = None
    """Alternative to host+port for providers addressed by URL."""

    metrics_port: Optional[int] = None
    """External mode: port of the service's Prometheus metrics endpoint,
    combined with host and the provider-declared path."""

    metrics_url: Optional[str] = None
    """External mode: full URL of the metrics endpoint; takes precedence
    over host+metrics_port."""

    params: Dict[str, Any] = {}
    """External mode: values for the provider's declared external_fields,
    keyed by field name (e.g. metadata_server, protocol). Rendered into
    the connector injection alongside the endpoint address."""


class CacheServiceL2Storage(BaseModel):
    """Managed mode only: the L2 storage backend the cache server spills
    KV cache to when its in-memory (L1) capacity is exceeded."""

    backend: str
    """Key into the provider's declared l2_backends."""

    params: Dict[str, Any] = {}
    """Backend field name -> value, per the provider's field declarations."""


class CacheServiceConfig(BaseModel):
    ram_size: Optional[int] = None
    """Cache capacity held in the cache server's memory (unit: GiB)."""

    chunk_size: Optional[int] = None
    """Size for each KV cache chunk (unit: number of tokens)."""

    image: Optional[str] = None
    """Container image for the custom provider version; ignored otherwise."""

    env: Optional[Dict[str, str]] = None
    """Extra environment variables for the managed cache server container."""

    parameters: Optional[List[str]] = None
    """Extra command-line flags appended to the cache server command;
    user-specified flags override template defaults."""

    fields: Optional[Dict[str, Any]] = None
    """Values for the provider's declared managed_fields, keyed by field
    name;
    they fill the fields' {{name}} template placeholders (free-form
    parameters still override any flag the templates produce)."""

    l2_storages: Optional[List[CacheServiceL2Storage]] = None
    """Managed mode only: ordered L2 storage backends forming a cascade.
    The first entry is the preferred read tier; all entries receive writes."""


@dataclass
class CacheServiceDeploymentMetadata:
    name: str
    labels: Dict[str, str]


class CacheServiceBase(SQLModel):
    name: str = Field(index=True)
    provider_name: str
    provider_version: Optional[str] = None
    # Stored as a plain string (the migration creates VARCHAR, not a native
    # DB enum); the enum type still validates values at the pydantic layer.
    mode: CacheServiceModeEnum = Field(
        sa_column=Column(String(length=64), nullable=False)
    )
    cluster_id: int = Field(foreign_key="clusters.id", nullable=False)
    worker_id: Optional[int] = None
    """Managed mode with singleton topology only: the worker the cache
    server runs on, picked at creation. Per-node providers derive their
    placement from the cluster's workers instead."""

    worker_selector: Optional[Dict[str, str]] = Field(
        sa_column=Column(JSON), default=None
    )
    """Managed mode with per_node topology only: labels a cluster worker
    must ALL match for the service to place an instance on it. None or
    empty means every worker of the cluster."""

    config: Optional[CacheServiceConfig] = Field(
        sa_column=Column(pydantic_column_type(CacheServiceConfig)), default=None
    )
    endpoint: Optional[CacheServiceEndpoint] = Field(
        sa_column=Column(pydantic_column_type(CacheServiceEndpoint)), default=None
    )
    """External mode only: connection info of the externally-run cache service."""

    state: CacheServiceStateEnum = Field(
        default=CacheServiceStateEnum.PENDING,
        sa_column=Column(String(length=64), nullable=False),
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    healthy: Optional[bool] = None
    last_check_at: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    restart_on_error: Optional[bool] = True
    """Managed mode: automatically restart (with backoff) when a cache
    server instance exits; False parks the instance in ERROR for manual
    handling. Applies to all of the service's instances."""

    def resolved_endpoint(self) -> CacheServiceEndpoint:
        """External mode: the address inference engines should connect to —
        the registered endpoint as-is. Managed services resolve endpoints
        per instance (instance worker IP + instance port) instead."""
        return self.endpoint or CacheServiceEndpoint()


class CacheService(CacheServiceBase, BaseModelMixin, table=True):
    __tablename__ = "cache_services"
    __table_args__ = (
        UniqueConstraint(
            "owner_principal_id", "name", name="uix_cache_services_name_per_owner"
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_principal_id: int = Field(
        default_factory=_platform_principal_id,
        sa_column=Column(
            Integer, ForeignKey("principals.id", ondelete="CASCADE"), nullable=False
        ),
    )


class CacheServiceCreate(CacheServiceBase):
    pass


class CacheServiceUpdate(CacheServiceBase):
    pass


class CacheServicePublic(CacheServiceBase):
    id: int
    owner_principal_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime


CacheServicesPublic = PaginatedList[CacheServicePublic]


class CacheServiceInstancePublic(BaseModel):
    """
    A managed cache server as the cache service API presents it.

    A view over the Workload that actually runs it: the service's endpoints
    keep this shape so the UI does not have to learn the workload model, and
    the named ports come back apart into the two fields it reads.
    """

    id: int
    name: str
    cache_service_id: int
    worker_id: int
    cluster_id: Optional[int] = None
    port: Optional[int] = None
    metrics_port: Optional[int] = None
    state: CacheServiceStateEnum
    state_message: Optional[str] = None
    healthy: Optional[bool] = None
    last_check_at: Optional[datetime] = None
    restart_count: Optional[int] = 0
    last_restart_time: Optional[datetime] = None
    spec_digest: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_workload(cls, workload) -> "CacheServiceInstancePublic":
        ports = workload.ports or {}
        return cls(
            id=workload.id,
            name=workload.name,
            cache_service_id=workload.owner_id,
            worker_id=workload.worker_id,
            cluster_id=workload.cluster_id,
            port=ports.get(CACHE_SERVICE_PORT),
            metrics_port=ports.get(CACHE_SERVICE_METRICS_PORT),
            state=CacheServiceStateEnum(workload.state.value),
            state_message=workload.state_message,
            healthy=workload.healthy,
            last_check_at=workload.last_check_at,
            restart_count=workload.restart_count,
            last_restart_time=workload.last_restart_time,
            spec_digest=workload.spec_digest,
            created_at=workload.created_at,
            updated_at=workload.updated_at,
        )


CacheServiceInstancesPublic = PaginatedList[CacheServiceInstancePublic]


class CacheServiceModelSummary(BaseModel):
    """A model deployment attached to a cache service, as listed on the
    service detail page. Deliberately lightweight: no instance join."""

    id: int
    name: str
    replicas: int
    ready_replicas: int
    backend: Optional[str] = None


CacheServiceModelsPublic = ItemList[CacheServiceModelSummary]


class TestCacheServiceConnectionRequest(BaseModel):
    provider_name: str
    provider_version: Optional[str] = None
    endpoint: CacheServiceEndpoint


class TestCacheServiceConnectionResponse(BaseModel):
    reachable: bool
    message: Optional[str] = None


def cache_service_spec_digest(service: "CacheService") -> str:
    """Digest of the spec fields that shape a running cache container:
    provider_version and config (image, capacity, parameters, env,
    fields, L2). worker_id/worker_selector are excluded — the controller
    reconciles placement live, so they cannot drift."""
    payload = {
        "provider_version": service.provider_version,
        "config": service.config.model_dump() if service.config else None,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class CacheConfigSnapshot(BaseModel):
    """
    Resolved shared-cache connection info denormalized onto a ModelInstance
    at creation time, so the worker can inject engine config without a
    server round-trip.
    """

    cache_service_id: int
    cache_service_name: Optional[str] = None
    provider_name: Optional[str] = None
    provider_version: Optional[str] = None
    endpoint: Optional[CacheServiceEndpoint] = None
    chunk_size: Optional[int] = None
    env: Dict[str, str] = {}
    args: List[str] = []
    files: Dict[str, str] = {}
    """Connector config files keyed by container path, written by the
    serving script before the engine starts."""

    injected: bool = False
    """False means the instance starts without the shared cache (degraded)."""

    reason: Optional[str] = None
    """Human-readable reason when injected is False."""

    endpoint_live: Optional[bool] = None
    """Whether the endpoint the engine started with is still served
    (None = never evaluated, treated as live). The snapshot records the
    engine's actual startup config and never mutates while it runs; this
    is the one field that tracks the present — the controller re-resolves
    on cache-instance changes and flips it when the recorded endpoint
    stops (or resumes) being attachable, so "attached" indicators do not
    report a cache that is gone."""
