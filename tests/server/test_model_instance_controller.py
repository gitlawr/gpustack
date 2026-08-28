"""
The model instance controller's workload sync.

Stage 3 step 1: the rows are compiled and written, and nothing reads them yet.
That is what makes this safe to land -- a wrong row is a wrong row rather than
a stopped container -- and what these tests hold it to.
"""

import contextlib
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import ModelInstanceController


_ANY_INSTANCE = object()


class _FakeSessionCtx:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *exc):
        return False


def _compiled(group_index):
    return SimpleNamespace(group_index=group_index, name=f"mi-{group_index}")


def _existing_workload(group_index):
    return SimpleNamespace(
        group_index=group_index,
        update=AsyncMock(),
        delete=AsyncMock(),
    )


@contextlib.contextmanager
def _workload_sync(monkeypatch, existing, compiled, instance=_ANY_INSTANCE):
    monkeypatch.setattr(
        "gpustack.server.controllers.async_session", lambda: _FakeSessionCtx()
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Workload.all_by_fields",
        AsyncMock(return_value=existing),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.ModelInstance.one_by_id",
        AsyncMock(return_value=instance),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.compile_model_instance", lambda mi: compiled
    )
    create = AsyncMock()
    monkeypatch.setattr("gpustack.server.controllers.Workload.create", create)
    yield create


def _event(event_type=EventType.UPDATED, instance_id=3):
    return Event(type=event_type, data={"id": instance_id})


@pytest.mark.asyncio
async def test_sync_creates_the_rows_that_are_missing(monkeypatch):
    with _workload_sync(
        monkeypatch, existing=[], compiled=[_compiled(0), _compiled(1)]
    ) as create:
        await ModelInstanceController(MagicMock())._sync_workloads(_event())

    assert create.await_count == 2


@pytest.mark.asyncio
async def test_sync_updates_rather_than_recreates(monkeypatch):
    """Idempotent: an instance that reports state on every health check must
    not churn its rows, or the id its logs are keyed by keeps moving."""
    existing = [_existing_workload(0)]

    with _workload_sync(
        monkeypatch, existing=existing, compiled=[_compiled(0)]
    ) as create:
        await ModelInstanceController(MagicMock())._sync_workloads(_event())

    create.assert_not_awaited()
    existing[0].update.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_drops_rows_the_instance_no_longer_has(monkeypatch):
    """A distributed instance that lost subordinate workers, or a backend that
    started delegating, leaves followers behind."""
    existing = [_existing_workload(0), _existing_workload(1), _existing_workload(2)]

    with _workload_sync(monkeypatch, existing=existing, compiled=[_compiled(0)]):
        await ModelInstanceController(MagicMock())._sync_workloads(_event())

    existing[0].update.assert_awaited_once()
    existing[1].delete.assert_awaited_once()
    existing[2].delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_deletes_every_row_with_the_instance(monkeypatch):
    existing = [_existing_workload(0), _existing_workload(1)]

    with _workload_sync(monkeypatch, existing=existing, compiled=[]):
        await ModelInstanceController(MagicMock())._sync_workloads(
            _event(EventType.DELETED)
        )

    for workload in existing:
        workload.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_never_fails_the_instances_own_reconcile(monkeypatch, caplog):
    """Nothing reads these rows yet, so a compile that goes wrong must not
    take the reconcile that does matter down with it."""
    monkeypatch.setattr(
        "gpustack.server.controllers.async_session", lambda: _FakeSessionCtx()
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Workload.all_by_fields",
        AsyncMock(side_effect=RuntimeError("db down")),
    )

    with caplog.at_level(logging.ERROR):
        await ModelInstanceController(MagicMock())._sync_workloads(_event())

    assert "Failed to sync workloads of model instance 3" in caplog.text
