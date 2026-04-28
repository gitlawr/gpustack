from datetime import datetime
from typing import Optional

from sqlalchemy import Enum as SQLEnum
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
)

from gpustack.schemas.common import UTCDateTime
from gpustack.schemas.principals import PrincipalType


class ClusterAccess(SQLModel, table=True):
    __tablename__ = 'cluster_access'

    cluster_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("clusters.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    principal_type: PrincipalType = Field(
        sa_column=Column(SQLEnum(PrincipalType), primary_key=True, nullable=False),
    )
    principal_id: int = Field(
        sa_column=Column(Integer, primary_key=True, nullable=False),
    )
    granted_by: Optional[int] = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(UTCDateTime, nullable=False),
    )


class ClusterAccessPublic(SQLModel):
    cluster_id: int
    principal_type: PrincipalType
    principal_id: int
    granted_by: Optional[int] = None
    created_at: datetime
