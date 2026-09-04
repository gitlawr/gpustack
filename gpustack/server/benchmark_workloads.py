"""
Compiling a benchmark into the workload that runs it.

A benchmark is the task-shaped workload: one container, run to completion, no
restart, and a wall-clock limit. That is the half of the restart policy the
service-shaped workloads never exercise, which is why this is the migration
that validates it.

See docs/proposals/workload-resource.md, stage 2.
"""

import logging
from typing import Optional

from gpustack.schemas.benchmark import Benchmark, BenchmarkStateEnum
from gpustack.schemas.workloads import (
    Workload,
    WorkloadOwnerKindEnum,
    WorkloadRestartPolicyEnum,
    WorkloadStateEnum,
)

logger = logging.getLogger(__name__)

BENCHMARK_WORKLOAD_TYPE = "benchmark"

_TO_WORKLOAD_STATE = {
    BenchmarkStateEnum.RUNNING: WorkloadStateEnum.RUNNING,
    BenchmarkStateEnum.COMPLETED: WorkloadStateEnum.SUCCEEDED,
    BenchmarkStateEnum.ERROR: WorkloadStateEnum.ERROR,
    BenchmarkStateEnum.UNREACHABLE: WorkloadStateEnum.UNREACHABLE,
}
"""``pending`` and ``queued`` both precede a container -- queueing is the
server deciding a worker may run this one next, not execution -- and
``stopped`` is a user cancelling, after which there is no container to
describe. All three leave the workload pending, the way a model instance's
scheduling and downloading do."""


def to_workload_state(state: Optional[BenchmarkStateEnum]) -> WorkloadStateEnum:
    return _TO_WORKLOAD_STATE.get(state, WorkloadStateEnum.PENDING)


def compile_benchmark(
    benchmark: Benchmark, max_duration_seconds: Optional[int] = None
) -> Workload:
    """
    The workload that runs a benchmark.

    Args:
        benchmark: The benchmark to compile.
        max_duration_seconds: The deployment's benchmark time limit, if one is
            configured. It becomes the workload's deadline, which is where a
            per-run limit belongs: the worker measures it from an in-memory
            start time today, so a worker restart loses it.
    """
    metadata = benchmark.get_deployment_metadata()
    return Workload(
        # The container name, which the worker already uses.
        name=metadata.name if metadata else benchmark.name,
        owner_kind=WorkloadOwnerKindEnum.BENCHMARK,
        owner_id=benchmark.id,
        owner_principal_id=benchmark.owner_principal_id,
        cluster_id=benchmark.cluster_id,
        worker_id=benchmark.worker_id,
        # A benchmark that exits is finished, not broken: restarting it would
        # throw away the results it just wrote and start the run again.
        restart_policy=WorkloadRestartPolicyEnum.NEVER,
        active_deadline_seconds=max_duration_seconds or None,
        labels=metadata.labels if metadata else None,
        state=to_workload_state(benchmark.state),
        state_message=benchmark.state_message,
        pid=benchmark.pid,
    )
