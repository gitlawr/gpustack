"""Unit tests for P4 (ALLOWED_PRINCIPALS extension) route logic."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InvalidException,
    NotFoundException,
)
from gpustack.routes import model_route_principals as principals_route
from gpustack.schemas.model_routes import ModelRoute
from gpustack.schemas.organizations import Organization
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.users import User


def _route(id: int = 1):
    route = MagicMock(spec=ModelRoute)
    route.id = id
    route.deleted_at = None
    return route


def _exec_returning(*results):
    queue = []
    for value in results:
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=value)
        result.first = MagicMock(return_value=value)
        scalars = MagicMock()
        scalars.all = MagicMock(return_value=value if isinstance(value, list) else [])
        result.scalars = MagicMock(return_value=scalars)
        queue.append(result)
    return AsyncMock(side_effect=queue)


def _session(*results):
    s = MagicMock()
    s.exec = _exec_returning(*results)
    s.commit = AsyncMock()
    s.rollback = AsyncMock()
    s.refresh = AsyncMock()
    s.delete = AsyncMock()
    s.add = MagicMock()
    return s


# ---- list -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_principals_returns_attached_links(monkeypatch):
    monkeypatch.setattr(
        principals_route.ModelRoute,
        "one_by_id",
        AsyncMock(return_value=_route()),
    )
    link1 = MagicMock()
    link1.route_id = 1
    link1.principal_type = PrincipalType.ORG
    link1.principal_id = 5
    session = _session([link1])

    result = await principals_route.list_route_principals(session=session, id=1)
    assert result == [link1]


@pytest.mark.asyncio
async def test_list_principals_404_when_route_missing(monkeypatch):
    monkeypatch.setattr(
        principals_route.ModelRoute,
        "one_by_id",
        AsyncMock(return_value=None),
    )
    with pytest.raises(NotFoundException):
        await principals_route.list_route_principals(session=MagicMock(), id=999)


# ---- add --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_principal_validates_org_exists(monkeypatch):
    monkeypatch.setattr(
        principals_route.ModelRoute,
        "one_by_id",
        AsyncMock(return_value=_route()),
    )
    monkeypatch.setattr(
        principals_route.Organization,
        "one_by_id",
        AsyncMock(return_value=None),
    )
    with pytest.raises(InvalidException):
        await principals_route.add_route_principal(
            session=MagicMock(),
            id=1,
            body=principals_route.PrincipalRef(
                principal_type=PrincipalType.ORG, principal_id=999
            ),
        )


@pytest.mark.asyncio
async def test_add_principal_validates_group_exists(monkeypatch):
    monkeypatch.setattr(
        principals_route.ModelRoute,
        "one_by_id",
        AsyncMock(return_value=_route()),
    )
    monkeypatch.setattr(
        principals_route.UserGroup,
        "one_by_id",
        AsyncMock(return_value=None),
    )
    with pytest.raises(InvalidException):
        await principals_route.add_route_principal(
            session=MagicMock(),
            id=1,
            body=principals_route.PrincipalRef(
                principal_type=PrincipalType.GROUP, principal_id=999
            ),
        )


@pytest.mark.asyncio
async def test_add_principal_rejects_system_user(monkeypatch):
    monkeypatch.setattr(
        principals_route.ModelRoute,
        "one_by_id",
        AsyncMock(return_value=_route()),
    )
    sys_user = MagicMock(spec=User)
    sys_user.is_system = True
    sys_user.deleted_at = None
    monkeypatch.setattr(
        principals_route.User, "one_by_id", AsyncMock(return_value=sys_user)
    )
    with pytest.raises(InvalidException):
        await principals_route.add_route_principal(
            session=MagicMock(),
            id=1,
            body=principals_route.PrincipalRef(
                principal_type=PrincipalType.USER, principal_id=2
            ),
        )


@pytest.mark.asyncio
async def test_add_principal_rejects_duplicate(monkeypatch):
    monkeypatch.setattr(
        principals_route.ModelRoute,
        "one_by_id",
        AsyncMock(return_value=_route()),
    )
    org = MagicMock(spec=Organization)
    org.deleted_at = None
    monkeypatch.setattr(
        principals_route.Organization, "one_by_id", AsyncMock(return_value=org)
    )
    existing_link = MagicMock()
    session = _session(existing_link)
    with pytest.raises(AlreadyExistsException):
        await principals_route.add_route_principal(
            session=session,
            id=1,
            body=principals_route.PrincipalRef(
                principal_type=PrincipalType.ORG, principal_id=5
            ),
        )


@pytest.mark.asyncio
async def test_add_principal_creates_link_and_invalidates_cache(monkeypatch):
    monkeypatch.setattr(
        principals_route.ModelRoute,
        "one_by_id",
        AsyncMock(return_value=_route()),
    )
    org = MagicMock(spec=Organization)
    org.deleted_at = None
    monkeypatch.setattr(
        principals_route.Organization, "one_by_id", AsyncMock(return_value=org)
    )
    cache_mock = AsyncMock()
    monkeypatch.setattr(principals_route, "revoke_model_access_cache", cache_mock)

    session = _session(None)  # no existing link
    await principals_route.add_route_principal(
        session=session,
        id=1,
        body=principals_route.PrincipalRef(
            principal_type=PrincipalType.ORG, principal_id=5
        ),
    )
    session.add.assert_called_once()
    session.commit.assert_awaited()
    cache_mock.assert_awaited_once()


# ---- remove -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_principal_404_when_missing(monkeypatch):
    monkeypatch.setattr(
        principals_route.ModelRoute,
        "one_by_id",
        AsyncMock(return_value=_route()),
    )
    session = _session(None)
    with pytest.raises(NotFoundException):
        await principals_route.remove_route_principal(
            session=session,
            id=1,
            principal_type=PrincipalType.USER,
            principal_id=99,
        )


@pytest.mark.asyncio
async def test_remove_principal_invalidates_cache(monkeypatch):
    monkeypatch.setattr(
        principals_route.ModelRoute,
        "one_by_id",
        AsyncMock(return_value=_route()),
    )
    cache_mock = AsyncMock()
    monkeypatch.setattr(principals_route, "revoke_model_access_cache", cache_mock)

    link = MagicMock()
    session = _session(link)
    await principals_route.remove_route_principal(
        session=session,
        id=1,
        principal_type=PrincipalType.ORG,
        principal_id=5,
    )
    session.delete.assert_awaited_once_with(link)
    session.commit.assert_awaited()
    cache_mock.assert_awaited_once()
