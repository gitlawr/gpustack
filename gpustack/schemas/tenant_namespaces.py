from datetime import datetime
from enum import Enum
from typing import ClassVar, List, Optional

from sqlalchemy import Enum as SQLEnum
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
    Text,
    UniqueConstraint,
)

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList


class TenantNamespaceState(str, Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    READY = "ready"
    ERROR = "error"
    DELETING = "deleting"


class TenantNamespaceBase(SQLModel):
    cluster_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("clusters.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    organization_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    namespace_name: str = Field(nullable=False)
    state: TenantNamespaceState = Field(
        default=TenantNamespaceState.PENDING,
        sa_column=Column(SQLEnum(TenantNamespaceState), nullable=False),
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class TenantNamespace(TenantNamespaceBase, BaseModelMixin, table=True):
    __tablename__ = 'tenant_namespaces'
    __table_args__ = (
        UniqueConstraint(
            'cluster_id', 'organization_id', name='uix_tenant_ns_cluster_org'
        ),
        UniqueConstraint(
            'cluster_id', 'namespace_name', name='uix_tenant_ns_cluster_name'
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)


class TenantNamespaceListParams(ListParams):
    cluster_id: Optional[int] = None
    organization_id: Optional[int] = None
    sortable_fields: ClassVar[List[str]] = [
        "namespace_name",
        "state",
        "created_at",
        "updated_at",
    ]


class TenantNamespacePublic(TenantNamespaceBase):
    id: int
    created_at: datetime
    updated_at: datetime


TenantNamespacesPublic = PaginatedList[TenantNamespacePublic]
