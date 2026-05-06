import re
from datetime import datetime
from typing import ClassVar, List, Optional, TYPE_CHECKING

from pydantic import field_validator
from sqlalchemy import Enum as SQLEnum
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
from gpustack.schemas.principals import OrgRole

if TYPE_CHECKING:
    from gpustack.schemas.users import User


# Reserved id of the built-in platform Organization. Created by the
# multi-tenancy foundation migration; system / infrastructure resources
# (worker tokens, default cluster registration tokens, admin-managed
# Models that predate the org switcher) all default to it.
PLATFORM_ORGANIZATION_ID = 1


slug_pattern = r'^[a-z](?:[a-z0-9\-]*[a-z0-9])?$'

# "Personal" is the auto-generated private Org each user gets on
# signup; "Global" is the UI label for admin-curated Platform rows
# (e.g. inference backends with organization_id IS NULL). Letting users
# create regular Orgs with these names would put two same-name entries
# in the Org switcher and confuse role/visibility semantics. Match
# case-insensitively after trimming whitespace.
RESERVED_ORG_NAMES = {"personal", "global"}
RESERVED_ORG_SLUGS = {"personal", "global"}
# Personal Org slug pattern — keep humans from grabbing the slot of a
# user's auto-generated Personal Org.
personal_slug_pattern = re.compile(r'^user-\d+$')


class OrganizationUpdate(SQLModel):
    name: str = Field(nullable=False)
    description: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    billing_account_ref: Optional[str] = Field(default=None, nullable=True)

    @field_validator("name", mode="before")
    def validate_name(cls, v):
        if not isinstance(v, str):
            raise ValueError("name must be a string")
        if v.strip().lower() in RESERVED_ORG_NAMES:
            raise ValueError(
                f"'{v}' is a reserved organization name; please choose another"
            )
        return v


class OrganizationCreate(OrganizationUpdate):
    slug: str = Field(nullable=False)

    @field_validator("slug", mode="before")
    def validate_slug(cls, v):
        if not isinstance(v, str):
            raise ValueError("slug must be a string")
        if not re.match(slug_pattern, v):
            raise ValueError(
                "slug must be lowercase, start with a letter, only contain "
                "letters, numbers, and hyphens, and not end with a hyphen"
            )
        if v.lower() in RESERVED_ORG_SLUGS or personal_slug_pattern.match(v):
            raise ValueError(f"'{v}' is a reserved slug; please choose another")
        return v


class OrganizationBase(OrganizationCreate):
    is_platform: bool = Field(default=False, nullable=False)
    # is_personal Orgs are auto-created one-per-user on signup. They
    # serve as each user's private namespace (à la GitHub personal
    # accounts) and are not surfaced in the admin Organizations list.
    is_personal: bool = Field(default=False, nullable=False)


class Organization(OrganizationBase, BaseModelMixin, table=True):
    __tablename__ = 'organizations'
    __table_args__ = (UniqueConstraint('slug', name='uix_organizations_slug'),)
    id: Optional[int] = Field(default=None, primary_key=True)

    memberships: List["OrganizationMembership"] = Relationship(
        back_populates="organization",
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
    )


class OrganizationMembership(SQLModel, table=True):
    __tablename__ = 'organization_memberships'
    user_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    organization_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("organizations.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    role: OrgRole = Field(
        default=OrgRole.USER,
        sa_column=Column(SQLEnum(OrgRole), nullable=False),
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(UTCDateTime, nullable=False),
    )

    user: Optional["User"] = Relationship(
        sa_relationship_kwargs={"lazy": "noload"},
    )
    organization: Optional[Organization] = Relationship(
        back_populates="memberships",
        sa_relationship_kwargs={"lazy": "noload"},
    )


class OrganizationListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "slug",
        "created_at",
        "updated_at",
    ]


class OrganizationPublic(OrganizationBase):
    id: int
    created_at: datetime
    updated_at: datetime


OrganizationsPublic = PaginatedList[OrganizationPublic]


class OrganizationMembershipPublic(SQLModel):
    user_id: int
    organization_id: int
    role: OrgRole
    created_at: datetime
