"""
Removing containers on this worker that no resource claims any more.

The last line of defence: a deletion whose event was missed, a worker that
crashed mid-teardown, a container left by an older version under a name
nothing uses. Each kind of workload says how to recognise its containers and
which of them are still claimed; the sweep itself is the same for all of them.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Sequence, Set

from gpustack_runtime.deployer import (
    WorkloadStatus,
    WorkloadStatusStateEnum,
    delete_workload,
    list_workloads,
)

from gpustack.utils import network
from gpustack.utils.datetimex import parse_iso8601_to_utc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkloadKind:
    """
    One kind of workload this worker runs.

    Args:
        name: Names the kind in log messages.
        matches: Recognises a container of this kind, usually by a label. The
            last kind registered should match anything, as the fallback.
        live_names: The container names the server still reports for this
            worker. Called at most once per sweep, and only when a container
            of the kind is actually present.
        grace_period_seconds: How long a container is left alone after it was
            created. Without it, one created between the ``live_names`` read
            and the sweep reads as an orphan.
        reap_states: States that make a container reapable whether or not it
            is still claimed -- a task that has finished, for one.
    """

    name: str
    matches: Callable[[WorkloadStatus], bool]
    live_names: Callable[[], Set[str]]
    grace_period_seconds: int
    reap_states: FrozenSet[WorkloadStatusStateEnum] = field(default_factory=frozenset)


class OrphanReaper:
    def __init__(self, kinds: Sequence[WorkloadKind]):
        self._kinds = list(kinds)

    def reap(self):
        """Sweep this worker's containers once."""
        live_names_by_kind = {}

        for workload in list_workloads():
            kind = next((k for k in self._kinds if k.matches(workload)), None)
            if kind is None:
                continue

            if kind.name not in live_names_by_kind:
                try:
                    live_names_by_kind[kind.name] = kind.live_names()
                except Exception as e:
                    # Reaping tears down live containers, so a failed read of
                    # what is still claimed has to skip the kind rather than
                    # read as "nothing is claimed".
                    live_names_by_kind[kind.name] = None
                    logger.warning(
                        f"Skipped reaping {kind.name} workloads: "
                        f"failed to list what is still running: {e}"
                    )
            live_names = live_names_by_kind[kind.name]
            if live_names is None:
                continue

            past_grace, _ = network.is_offline(
                parse_iso8601_to_utc(workload.created_at),
                kind.grace_period_seconds,
            )
            if not past_grace:
                continue
            if workload.name in live_names and workload.state not in kind.reap_states:
                continue

            try:
                delete_workload(workload.name)
            except Exception as e:
                logger.warning(f"Failed to delete orphan workload {workload.name}: {e}")
                continue
            logger.info(
                f"Deleted orphan {kind.name} workload {workload.name}, "
                f"created at {workload.created_at}."
            )
