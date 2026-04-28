"""Unit tests for P2 (Org / Group / Membership / cluster_access) route logic.

The codebase pattern for tests in this repo is mock-based; we follow that
here. Coverage focuses on the authorization branches and the corner cases
that aren't trivially expressible in declarative SQL constraints.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.routes import (
    cluster_access as cluster_access_route,
    organization_members,
    organizations as organizations_route,
    user_groups as user_groups_route,
)
from gpustack.schemas.organizations import Organization, OrganizationMembership
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.user_groups import UserGroup


def _ctx(
    *,
    user_id: int = 1,
    is_admin: bool = False,
    current_org_id: int | None = None,
    org_role: OrgRole | None = None,
):
    ctx = MagicMock()
    ctx.user = MagicMock()
    ctx.user.id = user_id
    ctx.is_platform_admin = is_admin
    ctx.current_org_id = current_org_id
    ctx.org_role = org_role
    return ctx


def _session_returning(*results):
    """Make a mock async session whose successive .exec() return the queued results."""
    session = MagicMock()
    queue = []
    for value in results:
        result = MagicMock()
        if isinstance(value, list):
            scalars = MagicMock()
            scalars.all = MagicMock(return_value=value)
            result.scalars = MagicMock(return_value=scalars)
            result.first = MagicMock(return_value=value[0] if value else None)
        else:
            result.scalar_one_or_none = MagicMock(return_value=value)
            result.first = MagicMock(return_value=value)
        queue.append(result)
    session.exec = AsyncMock(side_effect=queue)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    session.add = MagicMock()
    return session


# ---- _can_manage (organization_members) ------------------------------------


def test_can_manage_platform_admin_always():
    ctx = _ctx(is_admin=True)
    assert organization_members._can_manage(ctx, OrgRole.OWNER) is True


def test_can_manage_owner_can_promote_to_owner():
    ctx = _ctx(org_role=OrgRole.OWNER)
    assert organization_members._can_manage(ctx, OrgRole.OWNER) is True
    assert organization_members._can_manage(ctx, OrgRole.MEMBER) is True


def test_can_manage_admin_cannot_grant_owner():
    ctx = _ctx(org_role=OrgRole.ADMIN)
    assert organization_members._can_manage(ctx, OrgRole.OWNER) is False
    assert organization_members._can_manage(ctx, OrgRole.ADMIN) is True
    assert organization_members._can_manage(ctx, OrgRole.MEMBER) is True


def test_can_manage_member_cannot_manage():
    ctx = _ctx(org_role=OrgRole.MEMBER)
    assert organization_members._can_manage(ctx, OrgRole.MEMBER) is False


# ---- _can_manage_groups ----------------------------------------------------


def test_can_manage_groups_admin_passthrough():
    ctx = _ctx(is_admin=True, current_org_id=None)
    assert user_groups_route._can_manage_groups(ctx, org_id=42) is True


def test_can_manage_groups_member_blocked():
    ctx = _ctx(current_org_id=10, org_role=OrgRole.MEMBER)
    assert user_groups_route._can_manage_groups(ctx, org_id=10) is False


def test_can_manage_groups_admin_role_in_org_passes():
    ctx = _ctx(current_org_id=10, org_role=OrgRole.ADMIN)
    assert user_groups_route._can_manage_groups(ctx, org_id=10) is True


def test_can_manage_groups_wrong_org_blocked():
    ctx = _ctx(current_org_id=99, org_role=OrgRole.OWNER)
    assert user_groups_route._can_manage_groups(ctx, org_id=10) is False


# ---- organizations route ---------------------------------------------------


@pytest.mark.asyncio
async def test_create_organization_rejects_duplicate_slug(monkeypatch):
    session = MagicMock()
    monkeypatch.setattr(
        organizations_route.Organization,
        "one_by_fields",
        AsyncMock(return_value=Organization(name="Existing", slug="acme")),
    )
    with pytest.raises(AlreadyExistsException):
        await organizations_route.create_organization(
            session=session,
            org_in=organizations_route.OrganizationCreate(name="Acme", slug="acme"),
        )


@pytest.mark.asyncio
async def test_delete_platform_org_blocked(monkeypatch):
    platform = Organization(id=1, name="Platform", slug="platform", is_platform=True)
    monkeypatch.setattr(
        organizations_route.Organization,
        "one_by_id",
        AsyncMock(return_value=platform),
    )
    with pytest.raises(ConflictException):
        await organizations_route.delete_organization(session=MagicMock(), id=1)


@pytest.mark.asyncio
async def test_delete_org_blocked_when_resources_exist(monkeypatch):
    org = Organization(id=2, name="Acme", slug="acme", is_platform=False)
    monkeypatch.setattr(
        organizations_route.Organization,
        "one_by_id",
        AsyncMock(return_value=org),
    )
    monkeypatch.setattr(
        organizations_route,
        "_has_resources",
        AsyncMock(return_value=["models", "api_keys"]),
    )
    with pytest.raises(ConflictException) as excinfo:
        await organizations_route.delete_organization(session=MagicMock(), id=2)
    assert "models" in excinfo.value.message


# ---- organization_members route -------------------------------------------


@pytest.mark.asyncio
async def test_add_org_member_blocked_for_org_admin_promoting_to_owner(
    monkeypatch,
):
    org = Organization(id=10, name="Acme", slug="acme")
    monkeypatch.setattr(
        organization_members.Organization,
        "one_by_id",
        AsyncMock(return_value=org),
    )
    ctx = _ctx(current_org_id=10, org_role=OrgRole.ADMIN)
    with pytest.raises(ForbiddenException):
        await organization_members.add_org_member(
            session=MagicMock(),
            ctx=ctx,
            org_id=10,
            body=organization_members.MembershipCreate(user_id=2, role=OrgRole.OWNER),
        )


@pytest.mark.asyncio
async def test_remove_only_owner_blocked(monkeypatch):
    org = Organization(id=10, name="Acme", slug="acme")
    membership = OrganizationMembership(
        user_id=2,
        organization_id=10,
        role=OrgRole.OWNER,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    monkeypatch.setattr(
        organization_members.Organization,
        "one_by_id",
        AsyncMock(return_value=org),
    )
    monkeypatch.setattr(
        organization_members,
        "_find_membership",
        AsyncMock(return_value=membership),
    )
    monkeypatch.setattr(
        organization_members,
        "_has_other_owner",
        AsyncMock(return_value=False),
    )
    ctx = _ctx(is_admin=True)
    with pytest.raises(ConflictException):
        await organization_members.remove_org_member(
            session=MagicMock(), ctx=ctx, org_id=10, user_id=2
        )


@pytest.mark.asyncio
async def test_demote_only_owner_blocked(monkeypatch):
    org = Organization(id=10, name="Acme", slug="acme")
    membership = OrganizationMembership(
        user_id=2,
        organization_id=10,
        role=OrgRole.OWNER,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    monkeypatch.setattr(
        organization_members.Organization,
        "one_by_id",
        AsyncMock(return_value=org),
    )
    monkeypatch.setattr(
        organization_members,
        "_find_membership",
        AsyncMock(return_value=membership),
    )
    monkeypatch.setattr(
        organization_members,
        "_has_other_owner",
        AsyncMock(return_value=False),
    )
    ctx = _ctx(is_admin=True)
    with pytest.raises(ConflictException):
        await organization_members.update_org_member(
            session=MagicMock(),
            ctx=ctx,
            org_id=10,
            user_id=2,
            body=organization_members.MembershipUpdate(role=OrgRole.MEMBER),
        )


# ---- cluster_access route --------------------------------------------------


@pytest.mark.asyncio
async def test_grant_cluster_access_validates_principal(monkeypatch):
    cluster = MagicMock()
    cluster.id = 1
    cluster.deleted_at = None
    monkeypatch.setattr(
        cluster_access_route.Cluster,
        "one_by_id",
        AsyncMock(return_value=cluster),
    )
    monkeypatch.setattr(
        cluster_access_route.Organization,
        "one_by_id",
        AsyncMock(return_value=None),  # missing org
    )
    ctx = _ctx(is_admin=True)
    with pytest.raises(InvalidException):
        await cluster_access_route.grant_cluster_access(
            session=MagicMock(),
            ctx=ctx,
            cluster_id=1,
            body=cluster_access_route.ClusterAccessGrant(
                principal_type=PrincipalType.ORG, principal_id=999
            ),
        )


@pytest.mark.asyncio
async def test_grant_cluster_access_rejects_duplicate(monkeypatch):
    cluster = MagicMock()
    cluster.id = 1
    cluster.deleted_at = None
    org = Organization(id=2, name="Acme", slug="acme")
    monkeypatch.setattr(
        cluster_access_route.Cluster,
        "one_by_id",
        AsyncMock(return_value=cluster),
    )
    monkeypatch.setattr(
        cluster_access_route.Organization,
        "one_by_id",
        AsyncMock(return_value=org),
    )
    session = _session_returning([MagicMock()])  # exec returns existing row
    ctx = _ctx(is_admin=True)
    with pytest.raises(AlreadyExistsException):
        await cluster_access_route.grant_cluster_access(
            session=session,
            ctx=ctx,
            cluster_id=1,
            body=cluster_access_route.ClusterAccessGrant(
                principal_type=PrincipalType.ORG, principal_id=2
            ),
        )


@pytest.mark.asyncio
async def test_revoke_cluster_access_404_when_missing(monkeypatch):
    cluster = MagicMock()
    cluster.id = 1
    cluster.deleted_at = None
    monkeypatch.setattr(
        cluster_access_route.Cluster,
        "one_by_id",
        AsyncMock(return_value=cluster),
    )
    session = _session_returning(None)
    ctx = _ctx(is_admin=True)
    with pytest.raises(NotFoundException):
        await cluster_access_route.revoke_cluster_access(
            session=session,
            ctx=ctx,
            cluster_id=1,
            principal_type=PrincipalType.USER,
            principal_id=42,
        )


# ---- user_groups route -----------------------------------------------------


@pytest.mark.asyncio
async def test_create_group_blocked_for_member(monkeypatch):
    org = Organization(id=10, name="Acme", slug="acme")
    monkeypatch.setattr(
        user_groups_route.Organization,
        "one_by_id",
        AsyncMock(return_value=org),
    )
    ctx = _ctx(current_org_id=10, org_role=OrgRole.MEMBER)
    with pytest.raises(ForbiddenException):
        await user_groups_route.create_group(
            session=MagicMock(),
            ctx=ctx,
            org_id=10,
            body=user_groups_route.UserGroupCreate(name="team-a"),
        )


@pytest.mark.asyncio
async def test_add_group_member_requires_org_membership(monkeypatch):
    org = Organization(id=10, name="Acme", slug="acme")
    group = UserGroup(id=5, organization_id=10, name="team-a")
    monkeypatch.setattr(
        user_groups_route.UserGroup,
        "one_by_id",
        AsyncMock(return_value=group),
    )
    monkeypatch.setattr(
        user_groups_route.Organization,
        "one_by_id",
        AsyncMock(return_value=org),
    )
    session = _session_returning(None)  # no org membership
    ctx = _ctx(is_admin=True)
    with pytest.raises(InvalidException):
        await user_groups_route.add_group_member(
            session=session,
            ctx=ctx,
            org_id=10,
            group_id=5,
            body=user_groups_route.GroupMembershipCreate(user_id=99),
        )
