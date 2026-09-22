"""
Compiling a benchmark into the workload that runs it.

The benchmark is the task-shaped workload: run to completion, never
restarted, bounded by a deadline. The service-shaped kinds exercise neither
half of that, so this is where those two columns get a consumer.
"""

import contextlib
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.schemas.benchmark import Benchmark, BenchmarkStateEnum
from gpustack.schemas.workloads import (
    WorkloadOwnerKindEnum,
    WorkloadRestartPolicyEnum,
    WorkloadStateEnum,
)
from gpustack.server.benchmark_workloads import compile_benchmark
from gpustack.server.bus import Event, EventType
from gpustack.server.controllers import BenchmarkController


def _benchmark(**overrides):
    fields = dict(
        id=5,
        name="bench-1",
        model_instance_name="mi-1",
        cluster_id=1,
        worker_id=2,
        owner_principal_id=42,
        state=BenchmarkStateEnum.PENDING,
    )
    fields.update(overrides)
    return Benchmark(**fields)


def test_a_benchmark_is_never_restarted():
    """It is finished when it exits, not broken: restarting would throw away
    the results it just wrote and run it again."""
    workload = compile_benchmark(_benchmark())

    assert workload.restart_policy == WorkloadRestartPolicyEnum.NEVER
    assert workload.owner_kind == WorkloadOwnerKindEnum.BENCHMARK
    assert workload.owner_id == 5
    assert workload.worker_id == 2


def test_the_time_limit_becomes_the_workloads_deadline():
    """Where a per-run limit belongs. The worker measures it from an in-memory
    start time today, so a worker restart loses it."""
    workload = compile_benchmark(_benchmark(), max_duration_seconds=3600)

    assert workload.active_deadline_seconds == 3600


def test_no_configured_limit_means_no_deadline():
    assert compile_benchmark(_benchmark()).active_deadline_seconds is None
    assert (
        compile_benchmark(_benchmark(), max_duration_seconds=0).active_deadline_seconds
        is None
    )


def test_the_container_keeps_the_name_the_worker_already_uses():
    benchmark = _benchmark()
    workload = compile_benchmark(benchmark)

    assert workload.name == benchmark.get_deployment_metadata().name
    assert workload.labels == benchmark.get_deployment_metadata().labels


def test_completion_is_the_task_shaped_terminal_state():
    """The one state a service-shaped workload never reaches."""
    workload = compile_benchmark(_benchmark(state=BenchmarkStateEnum.COMPLETED))

    assert workload.state == WorkloadStateEnum.SUCCEEDED


@pytest.mark.parametrize(
    "state,expected",
    [
        (BenchmarkStateEnum.RUNNING, WorkloadStateEnum.RUNNING),
        (BenchmarkStateEnum.ERROR, WorkloadStateEnum.ERROR),
        (BenchmarkStateEnum.UNREACHABLE, WorkloadStateEnum.UNREACHABLE),
    ],
)
def test_execution_states_map_across(state, expected):
    assert compile_benchmark(_benchmark(state=state)).state == expected


@pytest.mark.parametrize(
    "state",
    [
        BenchmarkStateEnum.PENDING,
        BenchmarkStateEnum.QUEUED,
        BenchmarkStateEnum.STOPPED,
    ],
)
def test_states_with_no_container_leave_the_workload_pending(state):
    """Queueing is the server deciding which benchmark a worker runs next, not
    execution, and a stopped one has no container left to describe. Same
    boundary a model instance's scheduling and downloading sit on."""
    assert compile_benchmark(_benchmark(state=state)).state == (
        WorkloadStateEnum.PENDING
    )


# ---------------------------------------------------------------------------
# Controller sync
# ---------------------------------------------------------------------------


class _FakeSessionCtx:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *exc):
        return False


@contextlib.contextmanager
def _sync(monkeypatch, existing, benchmark=None, limit=None):
    monkeypatch.setattr(
        "gpustack.server.controllers.async_session", lambda: _FakeSessionCtx()
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Workload.all_by_fields",
        AsyncMock(return_value=existing),
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Benchmark.one_by_id",
        AsyncMock(return_value=benchmark if benchmark is not None else _benchmark()),
    )
    create = AsyncMock()
    monkeypatch.setattr("gpustack.server.controllers.Workload.create", create)
    yield create


def _controller(limit=None):
    return BenchmarkController(SimpleNamespace(benchmark_max_duration_seconds=limit))


def _existing_workload():
    return SimpleNamespace(update=AsyncMock(), delete=AsyncMock())


@pytest.mark.asyncio
async def test_sync_creates_the_workload(monkeypatch):
    with _sync(monkeypatch, existing=[]) as create:
        await _controller(limit=1800)._reconcile(
            Event(type=EventType.CREATED, data={"id": 5})
        )

    compiled = create.await_args.args[1]
    assert compiled.active_deadline_seconds == 1800
    assert compiled.restart_policy == WorkloadRestartPolicyEnum.NEVER


@pytest.mark.asyncio
async def test_sync_updates_the_spec_only(monkeypatch):
    """The worker will own execution state; recompiling it from the benchmark
    on every event would overwrite what the worker reported."""
    existing = [_existing_workload()]

    with _sync(monkeypatch, existing=existing) as create:
        await _controller()._reconcile(Event(type=EventType.UPDATED, data={"id": 5}))

    create.assert_not_awaited()
    update = existing[0].update.await_args.args[1]
    assert update.restart_policy == WorkloadRestartPolicyEnum.NEVER
    assert "state" not in update.model_fields_set


@pytest.mark.asyncio
async def test_sync_deletes_the_workload_with_the_benchmark(monkeypatch):
    existing = [_existing_workload()]

    with _sync(monkeypatch, existing=existing):
        await _controller()._reconcile(Event(type=EventType.DELETED, data={"id": 5}))

    existing[0].delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_never_raises(monkeypatch, caplog):
    """Nothing reads these rows yet, so a compile that goes wrong must not
    take the benchmark's own event stream down with it."""
    monkeypatch.setattr(
        "gpustack.server.controllers.async_session", lambda: _FakeSessionCtx()
    )
    monkeypatch.setattr(
        "gpustack.server.controllers.Workload.all_by_fields",
        AsyncMock(side_effect=RuntimeError("db down")),
    )

    with caplog.at_level(logging.ERROR):
        await _controller()._reconcile(Event(type=EventType.UPDATED, data={"id": 5}))

    assert "Failed to sync workload of benchmark 5" in caplog.text


def test_the_compiled_row_has_no_start_time_yet():
    """started_at is the worker's to write, when the container begins. The row
    exists from the moment the benchmark does, which for a queued one is well
    before that."""
    assert compile_benchmark(_benchmark()).started_at is None
