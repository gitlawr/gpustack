"""
The Workload resource and its endpoints.

Workloads are not part of the user-facing API -- the domain resources compile
into them -- but they are read and written over HTTP by workers, so the
scoping has to hold.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, insert, select
from sqlalchemy.orm import Session

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
    WorkloadStatusUpdate,
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


# ---------------------------------------------------------------------------
# Enum rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "member",
    [
        WorkloadOwnerKindEnum.MODEL_INSTANCE,
        WorkloadStateEnum.RUNNING,
        WorkloadRestartPolicyEnum.NEVER,
        WorkloadRoleEnum.FOLLOWER,
    ],
)
def test_enums_render_as_their_value(member):
    """The generated clients filter their watch-backed cache by comparing
    str(attribute) with str(the queried value). An enum rendering as
    "ClassName.MEMBER" matches nothing, so a filtered read comes back empty --
    silently, and the caller reads that as "this workload does not exist"."""
    assert str(member) == member.value


def test_a_cache_filter_on_an_enum_field_matches():
    """The failure this guards is not a crash: it is a filtered list that
    quietly returns nothing."""
    workload = _workload()
    queried = WorkloadOwnerKindEnum.CACHE_SERVICE.value

    # Read through a variable attribute name, as the client's cache filter
    # does, rather than the field directly.
    field = "owner_kind"
    assert str(getattr(workload, field)) == str(queried)


def _stored_workload(engine, **overrides):
    """A workload as the database holds it: bare strings in the enum columns,
    written below the ORM so nothing coerces on the way in."""
    values = dict(
        id=1,
        name="cache-svc-5-w1",
        owner_kind="cache_service",
        owner_id=5,
        worker_id=1,
        state="running",
        role="leader",
        restart_policy="always",
    )
    values.update(overrides)
    Workload.__table__.create(engine)
    with Session(engine) as session:
        session.execute(insert(Workload.__table__).values(**values))
        session.commit()
    with Session(engine) as session:
        return session.execute(select(Workload)).scalar_one()


def test_a_string_in_the_database_loads_as_an_enum():
    """The column holds VARCHAR, so the database has no enum type to cast --
    but no reader should have to know that. Converting on load makes an ORM row
    and an API-validated row the same shape, so .value works on both."""
    workload = _stored_workload(create_engine("sqlite://"))

    assert workload.state is WorkloadStateEnum.RUNNING
    assert workload.state.value == "running"
    assert workload.owner_kind is WorkloadOwnerKindEnum.CACHE_SERVICE
    assert workload.role is WorkloadRoleEnum.LEADER
    assert workload.restart_policy is WorkloadRestartPolicyEnum.ALWAYS
    # The enums mix in str, so a caller comparing against a bare string -- what
    # every caller did while the column returned one -- gets the same answer.
    assert workload.state == "running"


def test_an_unrecognized_stored_value_stays_readable():
    """A row written by a newer build, then rolled back to this one. Refusing
    to load a member this build does not have would take out every reader of
    the table over one field."""
    workload = _stored_workload(create_engine("sqlite://"), state="hibernating")

    assert workload.state == "hibernating"


def test_the_public_view_accepts_a_row_loaded_from_the_database():
    from datetime import datetime, timezone

    from gpustack.schemas.cache_services import CacheServiceInstancePublic

    now = datetime.now(timezone.utc)
    workload = Workload(
        id=1,
        name="cache-svc-5-w1",
        owner_kind="cache_service",
        owner_id=5,
        worker_id=1,
        state="running",
        ports={"service": 40001, "metrics": 40002},
        created_at=now,
        updated_at=now,
    )

    view = CacheServiceInstancePublic.from_workload(workload)

    assert view.state == "running"
    assert (view.port, view.metrics_port) == (40001, 40002)


# ---------------------------------------------------------------------------
# Reporting status without sending the spec
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_patch_is_refused_to_non_system_principals(monkeypatch):
    with pytest.raises(ForbiddenException):
        await workloads_route.update_workload_status(
            session=MagicMock(),
            ctx=_ctx(kind=PrincipalType.USER),
            id=1,
            status_in=MagicMock(),
        )


@pytest.mark.asyncio
async def test_a_status_patch_applies_only_what_was_set(monkeypatch):
    """The row's other writer owns the spec. A whole-row write replaces
    whatever it put there since this caller last read, so the model it sends
    has no spec fields to send."""
    workload = _workload()
    monkeypatch.setattr(
        workloads_route.Workload, "one_by_id", AsyncMock(return_value=workload)
    )
    monkeypatch.setattr(workloads_route, "cluster_scoped_system", lambda ctx: False)
    applied = AsyncMock()
    monkeypatch.setattr(workloads_route.Workload, "update", applied)

    status = WorkloadStatusUpdate(state=WorkloadStateEnum.RUNNING)
    await workloads_route.update_workload_status(
        session=MagicMock(),
        ctx=_ctx(kind=PrincipalType.SYSTEM),
        id=1,
        status_in=status,
    )

    sent = applied.await_args[0][1]
    assert sent.model_fields_set == {"state"}


def test_the_status_model_carries_no_spec():
    """The boundary is the model, not an agreement: a caller cannot send a
    binding through this endpoint even by trying."""
    from gpustack.server.model_instance_workloads import SPEC_FIELDS

    assert not (set(WorkloadStatusUpdate.model_fields) & SPEC_FIELDS)
