import re
from datetime import datetime
from typing import ClassVar, List, Optional, TYPE_CHECKING

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


def _check_reserved_name(name: str) -> None:
    """Raise ValueError if name is reserved for the system."""
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    if name.strip().lower() in RESERVED_ORG_NAMES:
        raise ValueError(
            f"'{name}' is a reserved organization name; please choose another"
        )


def _check_slug_format(slug: str) -> None:
    """Raise ValueError if slug fails the formatting / reserved checks."""
    if not isinstance(slug, str):
        raise ValueError("slug must be a string")
    if not re.match(slug_pattern, slug):
        raise ValueError(
            "slug must be lowercase, start with a letter, only contain "
            "letters, numbers, and hyphens, and not end with a hyphen"
        )
    if slug.lower() in RESERVED_ORG_SLUGS or personal_slug_pattern.match(slug):
        raise ValueError(f"'{slug}' is a reserved slug; please choose another")


def validate_org_input(*, name: Optional[str], slug: Optional[str] = None) -> None:
    """Validate user-supplied Org create/update payloads.

    Called from the route layer rather than from a Pydantic
    field_validator, because the latter would also fire when Pydantic
    serializes existing ORM rows (e.g. the auto-generated "Personal" /
    "user-N" rows are valid as data but reserved as user input).
    """
    if name is not None:
        _check_reserved_name(name)
    if slug is not None:
        _check_slug_format(slug)


class OrganizationUpdate(SQLModel):
    name: str = Field(nullable=False)
    description: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    billing_account_ref: Optional[str] = Field(default=None, nullable=True)


class OrganizationCreate(OrganizationUpdate):
    slug: str = Field(nullable=False)


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
    # Server-resolved labels so the UI list doesn't need a separate
    # `queryUsersList(page=-1)` round trip just to render names.
    username: Optional[str] = None
    full_name: Optional[str] = None
