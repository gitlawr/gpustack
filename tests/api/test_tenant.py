"""Unit tests for TenantContext resolution and role guards."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import ForbiddenException
from gpustack.api.tenant import (
    _resolve_requested_org_id,
    get_tenant_context,
    require_org_role,
    require_platform_admin,
)
from gpustack.schemas.organizations import OrganizationMembership
from gpustack.schemas.principals import OrgRole, PrincipalType


def _request(api_key=None):
    request = MagicMock()
    request.state = MagicMock(spec=[])
    if api_key is not None:
        request.state.api_key = api_key
    return request


def _user(
    id: int = 7,
    is_admin: bool = False,
    is_system: bool = False,
    default_organization_id=None,
):
    user = MagicMock()
    user.id = id
    user.is_admin = is_admin
    user.is_system = is_system
    user.default_organization_id = default_organization_id
    user.is_active = True
    return user


def _api_key(organization_id: int):
    key = MagicMock()
    key.organization_id = organization_id
    return key


def _session_returning(*scalar_lists):
    """Build a mock session whose successive `exec(...)` calls yield the given
    result sets. Each item in `scalar_lists` is either:
      - a list -> wrapped in `.scalars().all()`-style result
      - any other value -> returned by `.scalar_one_or_none()`
    """
    session = MagicMock()
    results = []
    for value in scalar_lists:
        result = MagicMock()
        if isinstance(value, list):
            scalars = MagicMock()
            scalars.all = MagicMock(return_value=value)
            result.scalars = MagicMock(return_value=scalars)
        else:
            result.scalar_one_or_none = MagicMock(return_value=value)
        results.append(result)
    session.exec = AsyncMock(side_effect=results)
    return session


# ---- _resolve_requested_org_id ---------------------------------------------


def test_resolve_org_id_prefers_api_key():
    user = _user(default_organization_id=1)
    request = _request(api_key=_api_key(organization_id=42))
    assert _resolve_requested_org_id(request, user, "999") == 42


def test_resolve_org_id_uses_header_when_no_api_key():
    user = _user(default_organization_id=1)
    request = _request()
    assert _resolve_requested_org_id(request, user, "999") == 999


def test_resolve_org_id_falls_back_to_default():
    user = _user(default_organization_id=1)
    request = _request()
    assert _resolve_requested_org_id(request, user, None) == 1


def test_resolve_org_id_invalid_header_raises():
    user = _user(default_organization_id=1)
    request = _request()
    with pytest.raises(ForbiddenException):
        _resolve_requested_org_id(request, user, "not-an-int")


# ---- get_tenant_context -----------------------------------------------------


@pytest.mark.asyncio
async def test_platform_admin_without_header_has_no_org_filter():
    user = _user(id=1, is_admin=True, default_organization_id=None)
    request = _request()
    session = _session_returning()  # no DB calls expected

    ctx = await get_tenant_context(
        request=request,
        session=session,
        user=user,
        x_organization_id=None,
    )

    assert ctx.is_platform_admin is True
    assert ctx.current_org_id is None
    assert ctx.org_role is None
    assert ctx.accessible_cluster_ids == set()
    assert ctx.accessible_principals == []


@pytest.mark.asyncio
async def test_member_uses_default_org_and_loads_principals():
    user = _user(id=10, is_admin=False, default_organization_id=1)
    request = _request()
    membership = OrganizationMembership(
        user_id=10, organization_id=1, role=OrgRole.MEMBER
    )
    session = _session_returning(
        membership,  # membership lookup
        [5, 6],  # group ids
        [101, 102],  # cluster ids
    )

    ctx = await get_tenant_context(
        request=request,
        session=session,
        user=user,
        x_organization_id=None,
    )

    assert ctx.is_platform_admin is False
    assert ctx.current_org_id == 1
    assert ctx.org_role == OrgRole.MEMBER
    assert ctx.accessible_cluster_ids == {101, 102}
    assert (PrincipalType.ORG, 1) in ctx.accessible_principals
    assert (PrincipalType.USER, 10) in ctx.accessible_principals
    assert (PrincipalType.GROUP, 5) in ctx.accessible_principals
    assert (PrincipalType.GROUP, 6) in ctx.accessible_principals


@pytest.mark.asyncio
async def test_non_member_request_to_other_org_is_rejected():
    user = _user(id=11, is_admin=False, default_organization_id=1)
    request = _request()
    session = _session_returning(None)  # no membership row

    with pytest.raises(ForbiddenException):
        await get_tenant_context(
            request=request,
            session=session,
            user=user,
            x_organization_id="2",
        )


@pytest.mark.asyncio
async def test_platform_admin_can_act_in_org_without_membership():
    user = _user(id=1, is_admin=True, default_organization_id=None)
    request = _request()
    session = _session_returning(
        None,  # no membership; admin should still pass
        [],
        [],
    )

    ctx = await get_tenant_context(
        request=request,
        session=session,
        user=user,
        x_organization_id="7",
    )

    assert ctx.is_platform_admin is True
    assert ctx.current_org_id == 7
    assert ctx.org_role is None  # admin acting in an org they're not a member of


@pytest.mark.asyncio
async def test_api_key_overrides_header():
    user = _user(id=10, is_admin=False, default_organization_id=1)
    request = _request(api_key=_api_key(organization_id=42))
    membership = OrganizationMembership(
        user_id=10, organization_id=42, role=OrgRole.MEMBER
    )
    session = _session_returning(membership, [], [])

    ctx = await get_tenant_context(
        request=request,
        session=session,
        user=user,
        x_organization_id="999",  # ignored when api_key is set
    )

    assert ctx.current_org_id == 42


# ---- require_platform_admin / require_org_role ------------------------------


@pytest.mark.asyncio
async def test_require_platform_admin_blocks_regular_user():
    ctx = MagicMock()
    ctx.is_platform_admin = False
    with pytest.raises(ForbiddenException):
        await require_platform_admin(ctx)


@pytest.mark.asyncio
async def test_require_platform_admin_allows_admin():
    ctx = MagicMock()
    ctx.is_platform_admin = True
    assert await require_platform_admin(ctx) is ctx


@pytest.mark.asyncio
async def test_require_org_role_admin_passthrough():
    dep = require_org_role(OrgRole.OWNER)
    ctx = MagicMock()
    ctx.is_platform_admin = True
    assert await dep(ctx) is ctx


@pytest.mark.asyncio
async def test_require_org_role_blocks_when_no_org_context():
    dep = require_org_role(OrgRole.OWNER)
    ctx = MagicMock()
    ctx.is_platform_admin = False
    ctx.current_org_id = None
    with pytest.raises(ForbiddenException):
        await dep(ctx)


@pytest.mark.asyncio
async def test_require_org_role_blocks_insufficient_role():
    dep = require_org_role(OrgRole.OWNER)

    def _assert_role(*allowed):
        if OrgRole.MEMBER not in allowed:
            raise ForbiddenException(message="nope")

    ctx = MagicMock()
    ctx.is_platform_admin = False
    ctx.current_org_id = 1
    ctx.org_role = OrgRole.MEMBER
    ctx.assert_org_role = _assert_role
    with pytest.raises(ForbiddenException):
        await dep(ctx)


@pytest.mark.asyncio
async def test_require_org_role_passes_for_matching_role():
    dep = require_org_role(OrgRole.OWNER, OrgRole.ADMIN)

    def _assert_role(*allowed):
        if OrgRole.OWNER not in allowed:
            raise AssertionError("did not pass owner role through")

    ctx = MagicMock()
    ctx.is_platform_admin = False
    ctx.current_org_id = 1
    ctx.org_role = OrgRole.OWNER
    ctx.assert_org_role = _assert_role
    assert await dep(ctx) is ctx
