"""
The Workload resource and its endpoints.

Workloads are not part of the user-facing API -- the domain resources compile
into them -- but they are read and written over HTTP by workers, so the
scoping has to hold.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpustack.api.exceptions import ForbiddenException, NotFoundException
from gpustack.api.tenant import TenantContext
from gpustack.routes import workloads as workloads_route
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.workloads import (
    ReservedClaim,
    Workload,
    WorkloadOwnerKindEnum,
    WorkloadRestartPolicyEnum,
    WorkloadRoleEnum,
    WorkloadStateEnum,
)

CALLER_PRINCIPAL = 7
OTHER_PRINCIPAL = 8


def _workload(owner_principal_id=CALLER_PRINCIPAL, **overrides):
    fields = dict(
        id=1,
        name="cache-svc-5-i11",
        owner_kind=WorkloadOwnerKindEnum.CACHE_SERVICE,
        owner_id=5,
        owner_principal_id=owner_principal_id,
        cluster_id=1,
        worker_id=1,
        state=WorkloadStateEnum.PENDING,
    )
    fields.update(overrides)
    return Workload(**fields)


def _ctx(principal_id=CALLER_PRINCIPAL, kind=PrincipalType.USER):
    user = MagicMock()
    user.kind = kind
    return TenantContext(
        user=user,
        is_platform_admin=False,
        current_principal_id=principal_id,
        org_role=None,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_defaults_describe_a_standalone_service_workload():
    """A workload that runs alone is the leader of a group of one, so nothing
    downstream has to special-case group size."""
    workload = _workload()

    assert workload.role == WorkloadRoleEnum.LEADER
    assert workload.group_index == 0
    assert workload.group_key is None
    assert workload.restart_policy == WorkloadRestartPolicyEnum.ALWAYS
    assert workload.active_deadline_seconds is None
    assert workload.restart_count == 0


def test_reserved_claims_hold_resources_on_workers_with_no_container():
    """A backend that starts its own distributed workers is given the nodes by
    the scheduler; the reservation belongs to the workload that made it, so a
    row keeps meaning "a container gpustack runs"."""
    workload = _workload(
        reserved_claims=[
            ReservedClaim(worker_id=2, gpu_indexes=[0, 1]),
            ReservedClaim(worker_id=3, gpu_indexes=[0]),
        ]
    )

    assert [claim.worker_id for claim in workload.reserved_claims] == [2, 3]


def test_the_unique_constraint_covers_the_group_position():
    """Leader and follower of one instance can land on the same worker, so
    (owner, worker) alone would reject a legitimate pair."""
    constraint = next(
        c
        for c in Workload.__table__.constraints
        if c.name == "uix_workloads_owner_worker_group_index"
    )

    assert [c.name for c in constraint.columns] == [
        "owner_kind",
        "owner_id",
        "worker_id",
        "group_index",
    ]


def test_the_indexes_cover_the_hot_reads():
    names = {index.name for index in Workload.__table__.indexes}

    assert "ix_workloads_worker_id" in names  # worker reconcile
    assert "ix_workloads_owner" in names  # controller fan-out
    assert "ix_workloads_owner_state" in names  # endpoint resolution


# ---------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_hides_another_principals_workload(monkeypatch):
    monkeypatch.setattr(
        workloads_route.Workload,
        "one_by_id",
        AsyncMock(return_value=_workload(owner_principal_id=OTHER_PRINCIPAL)),
    )

    with pytest.raises(NotFoundException):
        await workloads_route.get_workload(session=MagicMock(), ctx=_ctx(), id=1)


@pytest.mark.asyncio
async def test_get_returns_the_callers_own_workload(monkeypatch):
    """Ownership is on the row, copied from the owner at creation, so this
    needs no join to a table that differs per owner_kind."""
    owned = _workload()
    monkeypatch.setattr(
        workloads_route.Workload, "one_by_id", AsyncMock(return_value=owned)
    )

    assert (
        await workloads_route.get_workload(session=MagicMock(), ctx=_ctx(), id=1)
    ) is owned


@pytest.mark.asyncio
async def test_update_is_refused_to_non_system_principals(monkeypatch):
    """Workers write execution state back; users act on the owning resource."""
    with pytest.raises(ForbiddenException):
        await workloads_route.update_workload(
            session=MagicMock(),
            ctx=_ctx(kind=PrincipalType.USER),
            id=1,
            workload_in=MagicMock(),
        )


@pytest.mark.asyncio
async def test_update_writes_back_for_a_system_principal(monkeypatch):
    workload = _workload()
    monkeypatch.setattr(
        workloads_route.Workload, "one_by_id", AsyncMock(return_value=workload)
    )
    monkeypatch.setattr(workloads_route, "cluster_scoped_system", lambda ctx: False)
    applied = AsyncMock()
    monkeypatch.setattr(workloads_route.Workload, "update", applied)

    result = await workloads_route.update_workload(
        session=MagicMock(),
        ctx=_ctx(kind=PrincipalType.SYSTEM),
        id=1,
        workload_in=MagicMock(),
    )

    applied.assert_awaited_once()
    assert result is workload


@pytest.mark.asyncio
async def test_list_scopes_by_owner_principal(monkeypatch):
    captured = {}

    async def fake_paginated(**kwargs):
        captured.update(kwargs)
        return "page"

    monkeypatch.setattr(workloads_route.Workload, "paginated_by_query", fake_paginated)
    monkeypatch.setattr(workloads_route, "async_session", lambda: _FakeSessionCtx())
    conditions = [object()]
    monkeypatch.setattr(
        workloads_route, "tenant_list_conditions", lambda ctx, model: conditions
    )

    params = SimpleNamespace(watch=False, page=1, perPage=10)
    await workloads_route.get_workloads(
        ctx=_ctx(), params=params, worker_id=3, owner_kind=None
    )

    assert captured["fields"] == {"worker_id": 3}
    assert captured["extra_conditions"] is conditions


class _FakeSessionCtx:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_watch_filters_a_worker_to_its_own_cluster(monkeypatch):
    monkeypatch.setattr(workloads_route, "cluster_scoped_system", lambda ctx: True)
    seen = {}

    def fake_streaming(fields=None, filter_func=None):
        seen["fields"] = fields
        seen["filter_func"] = filter_func
        return iter(())

    monkeypatch.setattr(workloads_route.Workload, "streaming", fake_streaming)
    monkeypatch.setattr(
        workloads_route,
        "scoped_cluster_row_visible",
        lambda ctx, data: data.cluster_id == 1,
    )

    params = SimpleNamespace(watch=True, page=1, perPage=10)
    with patch.object(workloads_route, "StreamingResponse", lambda *a, **k: "stream"):
        await workloads_route.get_workloads(ctx=_ctx(), params=params, worker_id=3)

    assert seen["fields"] == {"worker_id": 3}
    assert seen["filter_func"](_workload(cluster_id=1)) is True
    assert seen["filter_func"](_workload(cluster_id=2)) is False
