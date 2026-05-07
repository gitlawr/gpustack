from datetime import datetime
from typing import ClassVar, List, Optional, TYPE_CHECKING

from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    Relationship,
    SQLModel,
    Text,
    UniqueConstraint,
)

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.common import ListParams, PaginatedList, UTCDateTime

if TYPE_CHECKING:
    from gpustack.schemas.organizations import Organization
    from gpustack.schemas.users import User


class UserGroupUpdate(SQLModel):
    name: str = Field(nullable=False)
    description: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class UserGroupCreate(UserGroupUpdate):
    pass


class UserGroupBase(UserGroupUpdate):
    organization_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )


class UserGroup(UserGroupBase, BaseModelMixin, table=True):
    __tablename__ = 'user_groups'
    __table_args__ = (
        UniqueConstraint('organization_id', 'name', name='uix_user_groups_org_id_name'),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    organization: Optional["Organization"] = Relationship(
        sa_relationship_kwargs={"lazy": "noload"},
    )
    memberships: List["UserGroupMembership"] = Relationship(
        back_populates="group",
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
    )


class UserGroupMembership(SQLModel, table=True):
    __tablename__ = 'user_group_memberships'
    user_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    group_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("user_groups.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(UTCDateTime, nullable=False),
    )

    user: Optional["User"] = Relationship(
        sa_relationship_kwargs={"lazy": "noload"},
    )
    group: Optional[UserGroup] = Relationship(
        back_populates="memberships",
        sa_relationship_kwargs={"lazy": "noload"},
    )


class UserGroupListParams(ListParams):
    organization_id: Optional[int] = None
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "created_at",
        "updated_at",
    ]


class UserGroupPublic(UserGroupBase):
    id: int
    created_at: datetime
    updated_at: datetime


UserGroupsPublic = PaginatedList[UserGroupPublic]


class UserGroupMembershipPublic(SQLModel):
    user_id: int
    group_id: int
    created_at: datetime
    username: Optional[str] = None
    full_name: Optional[str] = None
