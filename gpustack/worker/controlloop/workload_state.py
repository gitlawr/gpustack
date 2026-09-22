"""
Reading a container workload's status, without domain knowledge.

The container runtime reports one state enum; what it *means* depends on the
kind of workload. A clean exit ends a benchmark successfully and kills a cache
server. That split is the restart policy, so this module reports a neutral
phase and lets the policy decide, instead of each manager hard-coding its own
list of "states that mean trouble".
"""

import logging
from enum import Enum
from typing import Optional

from gpustack_runtime.deployer import WorkloadStatus, WorkloadStatusStateEnum

logger = logging.getLogger(__name__)

WORKLOAD_FAILED_MESSAGE = "Workload exited or unhealthy."
"""Last-resort message for a workload that stopped without saying why."""


class WorkloadPhase(str, Enum):
    """Where a workload is in its life, independent of what it runs."""

    LAUNCHING = "launching"
    """Accepted but not serving yet: pulling the image, running init
    containers, waiting for admission."""

    RUNNING = "running"

    EXITED = "exited"
    """Stopped of its own accord with no error. Success for a task, death for
    a service."""

    FAILED = "failed"
    """Stopped with an error, went unhealthy, or the runtime cannot say what
    it is doing."""

    MISSING = "missing"
    """No workload by that name. Either it never got created or something
    outside gpustack removed it."""


class RestartPolicy(str, Enum):
    """When a stopped workload should be started again."""

    ALWAYS = "always"
    """Service workloads: any stop is a fault to recover from."""

    ON_FAILURE = "on_failure"
    """Task workloads that should retry: a clean exit completes them."""

    NEVER = "never"
    """Task workloads that report their outcome instead of retrying."""


def classify_workload(workload: Optional[WorkloadStatus]) -> WorkloadPhase:
    """
    The phase a workload is in.

    Args:
        workload: The runtime status, None when no such workload exists.
    """
    if workload is None:
        return WorkloadPhase.MISSING

    state = workload.state
    if state in (
        WorkloadStatusStateEnum.PENDING,
        WorkloadStatusStateEnum.INITIALIZING,
    ):
        return WorkloadPhase.LAUNCHING
    if state == WorkloadStatusStateEnum.RUNNING:
        return WorkloadPhase.RUNNING
    if state == WorkloadStatusStateEnum.INACTIVE:
        return WorkloadPhase.EXITED
    # FAILED and UNHEALTHY say so. UNKNOWN is rare but real — a leftover pause
    # container, for one — and a workload the runtime cannot describe is not
    # one anything should keep waiting on, so it counts as failed.
    return WorkloadPhase.FAILED


def needs_restart(phase: WorkloadPhase, restart_policy: RestartPolicy) -> bool:
    """
    Whether a workload in this phase should be started again under this policy.
    """
    if phase in (WorkloadPhase.LAUNCHING, WorkloadPhase.RUNNING):
        return False
    if restart_policy == RestartPolicy.NEVER:
        return False
    if restart_policy == RestartPolicy.ON_FAILURE:
        return phase in (WorkloadPhase.FAILED, WorkloadPhase.MISSING)
    return True


def describe_workload_failure(
    workload: Optional[WorkloadStatus],
    fallback_message: str = WORKLOAD_FAILED_MESSAGE,
) -> str:
    """
    Explain why a workload stopped, for the owning resource's state message.

    The workload's `state_message` is the most specific text the runtime has --
    a Pod's admission rejection, or an image-pull reason plus the registry error
    behind it (gpustack/gpustack#5869) -- but it never carries an exit code, and
    a container that merely crashed leaves it empty on Kubernetes. The
    per-container `exits` carry both, so take the reason from there when there
    is no message, and append the exit code either way
    (gpustack/gpustack#4217).

    Args:
        workload: The runtime WorkloadStatus, None if the workload is gone.
        fallback_message: Used when the runtime says nothing useful. Callers
            pass a message naming what they run, e.g. "Inference server exited
            or unhealthy."

    Returns:
        The failure message to surface on the owning resource.
    """
    if not workload:
        return fallback_message

    message = getattr(workload, "state_message", "") or ""
    # A container blocked from starting reports no exit code, and its reason is
    # already what the state message is built from, so it adds nothing here.
    exits = [
        exit_
        for exit_ in (getattr(workload, "exits", None) or [])
        if exit_.exit_code is not None
    ]
    if not exits:
        return message or fallback_message

    if not message:
        # Deduplicated but order-preserving: sidecars usually die of one cause.
        message = ", ".join(
            dict.fromkeys(exit_.reason for exit_ in exits if exit_.reason)
        )
    codes = ", ".join(
        (
            f"{exit_.name} exit code {exit_.exit_code}"
            if len(exits) > 1
            else f"exit code {exit_.exit_code}"
        )
        for exit_ in exits
    )
    return f"{message or fallback_message} ({codes})"
