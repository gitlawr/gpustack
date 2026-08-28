"""
Shared pieces of the worker-side control loops.

Every kind of workload a worker runs — model instances, benchmarks, cache
service instances — is driven by the same skeleton: watch the resource, react
to events, reconcile periodically against the container runtime, and write the
outcome back. These modules hold the parts of that skeleton that carry no
domain knowledge, so a new kind of workload does not arrive with a fourth copy
of them.
"""

from gpustack.worker.controlloop.container_logs import ContainerLogPersistence
from gpustack.worker.controlloop.backoff import (
    RestartActionEnum,
    RestartBudget,
    RestartDecision,
)
from gpustack.worker.controlloop.watcher import watch_forever
from gpustack.worker.controlloop.workload_state import (
    WorkloadPhase,
    classify_workload,
    describe_workload_failure,
    needs_restart,
)
from gpustack.worker.controlloop.writeback import update_resource

__all__ = [
    "ContainerLogPersistence",
    "RestartActionEnum",
    "RestartBudget",
    "RestartDecision",
    "WorkloadPhase",
    "classify_workload",
    "describe_workload_failure",
    "needs_restart",
    "update_resource",
    "watch_forever",
]
