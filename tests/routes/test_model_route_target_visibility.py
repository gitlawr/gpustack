"""Target list and watch visibility derives from the owning route."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.tenant import TenantContext
from gpustack.mixins import active_record
from gpustack.routes import model_routes
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    ModelRouteTargetListParams,
)
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.bus import Event, EventBus, EventType


def _ctx(scope):
    return TenantContext(
        user=SimpleNamespace(
            kind=PrincipalType.SYSTEM if "system" in scope else PrincipalType.USER
        ),
        is_platform_admin=scope.startswith("admin"),
        current_principal_id=(
            None
            if scope in ("admin-all", "system", "scoped-system")
            else 7 if scope == "personal" else 101
        ),
        org_role=OrgRole.OWNER,
        scoped_cluster_id=11 if scope == "scoped-system" else None,
    )


def _target(i):
    return ModelRouteTarget(
        id=i,
        name=f"target-{i}",
        route_id=i,
        route_name=f"route-{i}",
        model_id=i,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture
def parents(monkeypatch):
    rows = {
        i: SimpleNamespace(owner_principal_id=owner, deleted_at=None)
        for i, owner in [(1, 101), (2, 202), (3, 303)]
    }
    monkeypatch.setattr(model_routes, "async_session", MagicMock())
    monkeypatch.setattr(
        ModelRoute, "one_by_id", AsyncMock(side_effect=lambda session, i: rows.get(i))
    )
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,expected",
    [
        ("org", [1]),
        ("personal", []),
        ("admin-org", [1]),
        ("admin-all", [1, 2, 3]),
        ("system", [1, 2, 3]),
        ("scoped-system", [1, 2, 3]),
    ],
)
async def test_target_watch_filters_replay_and_bus(
    monkeypatch, parents, scope, expected
):
    rows = [_target(i) for i in (1, 2, 3)]
    monkeypatch.setattr(ModelRouteTarget, "cached_all", AsyncMock(return_value=rows))
    replay_query = AsyncMock(return_value=[row for row in rows if row.id in expected])
    monkeypatch.setattr(ModelRouteTarget, "all_by_fields", replay_query)
    events = [
        Event(type=event_type, data=row)
        for event_type in (EventType.CREATED, EventType.UPDATED, EventType.DELETED)
        for row in rows
    ]
    subscriber = SimpleNamespace(
        receive=AsyncMock(side_effect=[*events, asyncio.CancelledError()])
    )
    bus = MagicMock()
    bus.subscribe.return_value = subscriber
    monkeypatch.setattr(active_record, "event_bus", bus)
    response = await model_routes.get_model_route_targets(
        ctx=_ctx(scope), params=ModelRouteTargetListParams(watch=True)
    )
    frames = []
    with pytest.raises(asyncio.CancelledError):
        async for frame in response.body_iterator:
            frames.append(json.loads(frame))
    assert [frame["data"]["id"] for frame in frames] == expected * 4
    bus.unsubscribe.assert_called_once()
    if scope in ("org", "personal", "admin-org"):
        replay_query.assert_awaited_once()
        ModelRouteTarget.cached_all.assert_not_called()
        assert ModelRoute.one_by_id.await_count == len(events)
    else:
        replay_query.assert_not_called()
        ModelRouteTarget.cached_all.assert_awaited_once()
        ModelRoute.one_by_id.assert_not_called()


@pytest.mark.asyncio
async def test_target_watch_tracks_parent_changes(monkeypatch, parents):
    async def subscribe(**kwargs):
        yield Event(type=EventType.CREATED, data=_target(1))
        parents[1].owner_principal_id = 202
        yield Event(type=EventType.UPDATED, data=_target(1))
        parents[4] = SimpleNamespace(owner_principal_id=101, deleted_at=None)
        yield Event(type=EventType.CREATED, data=_target(4))
        parents[4].deleted_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
        yield Event(type=EventType.UPDATED, data=_target(4))
        yield Event(type=EventType.DELETED, data=_target(99))
        yield Event(type=EventType.DELETED, data={"id": 1})

    monkeypatch.setattr(ModelRouteTarget, "subscribe", subscribe)
    response = await model_routes.get_model_route_targets(
        ctx=_ctx("org"), params=ModelRouteTargetListParams(watch=True)
    )
    frames = [json.loads(frame) async for frame in response.body_iterator]
    assert [frame["data"]["id"] for frame in frames] == [1, 4]


@pytest.mark.asyncio
@pytest.mark.parametrize("row_count", [0, 1, 1000])
async def test_target_replay_uses_one_authorized_query(monkeypatch, parents, row_count):
    rows = [_target(i) for i in range(1, row_count + 1)]
    for row in rows:
        row.route_id = 1
    session = MagicMock()
    session.exec = AsyncMock(return_value=SimpleNamespace(all=lambda: rows))
    model_routes.async_session.return_value.__aenter__.return_value = session
    monkeypatch.setattr(ModelRouteTarget, "cached_all", AsyncMock())
    monkeypatch.setattr(ModelRoute, "cached_all", AsyncMock())
    subscriber = SimpleNamespace(receive=AsyncMock(side_effect=asyncio.CancelledError))
    bus = MagicMock()
    bus.subscribe.return_value = subscriber
    monkeypatch.setattr(active_record, "event_bus", bus)

    response = await model_routes.get_model_route_targets(
        ctx=_ctx("org"),
        params=ModelRouteTargetListParams(watch=True, route_id=1),
        search="target",
    )
    frames = []
    with pytest.raises(asyncio.CancelledError):
        async for frame in response.body_iterator:
            frames.append(json.loads(frame))

    assert [frame["data"]["id"] for frame in frames] == list(range(1, row_count + 1))
    model_routes.async_session.assert_called_once()
    session.exec.assert_awaited_once()
    statement = session.exec.call_args.args[0]
    sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "model_route_targets.route_id IN (SELECT model_routes.id" in sql
    assert "model_routes.owner_principal_id = 101" in sql
    assert "model_routes.deleted_at IS NULL" in sql
    assert "model_route_targets.deleted_at IS NULL" in sql
    assert "model_route_targets.route_id = 1" in sql
    assert "lower(model_route_targets.name) LIKE '%target%'" in sql
    ModelRoute.one_by_id.assert_not_called()
    ModelRoute.cached_all.assert_not_called()
    ModelRouteTarget.cached_all.assert_not_called()
    bus.unsubscribe.assert_called_once()


@pytest.mark.asyncio
async def test_events_during_replay_use_live_parent_authorization(monkeypatch, parents):
    bus = EventBus()
    monkeypatch.setattr(active_record, "event_bus", bus)
    topic = "modelroutetarget"

    async def load_targets(**kwargs):
        assert bus.subscribers.get(topic)
        # The query snapshot authorizes target 1. Its parent then changes,
        # and target 4 is created while the initial query is still in flight.
        snapshot = [_target(1)]
        parents[1].owner_principal_id = 202
        parents[4] = SimpleNamespace(owner_principal_id=101, deleted_at=None)
        await bus.publish(topic, Event(type=EventType.CREATED, data=_target(1)))
        await bus.publish(topic, Event(type=EventType.CREATED, data=_target(4)))
        await bus.publish(topic, Event(type=EventType.HEARTBEAT, data=None))
        return snapshot

    monkeypatch.setattr(
        ModelRouteTarget, "all_by_fields", AsyncMock(side_effect=load_targets)
    )
    response = await model_routes.get_model_route_targets(
        ctx=_ctx("org"), params=ModelRouteTargetListParams(watch=True)
    )

    async def collect():
        frames = []
        try:
            async for frame in response.body_iterator:
                if frame == "\n\n":
                    break
                frames.append(json.loads(frame))
        finally:
            await response.body_iterator.aclose()
        return frames

    frames = await asyncio.wait_for(collect(), timeout=1)
    assert [frame["data"]["id"] for frame in frames] == [1, 4]
    assert all(frame["type"] == EventType.CREATED.value for frame in frames)
    assert [call.args[1] for call in ModelRoute.one_by_id.await_args_list] == [1, 4]
    assert bus.subscribers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,owner", [("org", 101), ("personal", 7), ("admin-org", 101)]
)
async def test_target_list_scopes_parent_even_with_explicit_route_filter(
    monkeypatch, parents, scope, owner
):
    query = AsyncMock(return_value=SimpleNamespace(items=[]))
    monkeypatch.setattr(ModelRouteTarget, "paginated_by_query", query)
    monkeypatch.setattr(model_routes, "_apply_target_plugin_sections", AsyncMock())
    await model_routes.get_model_route_targets(
        ctx=_ctx(scope), params=ModelRouteTargetListParams(route_id=2)
    )
    kwargs = query.call_args.kwargs
    assert kwargs["fields"]["route_id"] == 2
    (condition,) = kwargs["extra_conditions"]
    sql = str(condition.compile(compile_kwargs={"literal_binds": True}))
    assert "model_route_targets.route_id IN (SELECT model_routes.id" in sql
    assert f"model_routes.owner_principal_id = {owner}" in sql
    assert "model_routes.deleted_at IS NULL" in sql
