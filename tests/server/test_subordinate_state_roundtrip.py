"""A subordinate's state, through a workload row and back.

The instance's own lifecycle does not survive this trip: analysing,
scheduling, downloading and its own starting all mirror onto one pending
workload, which is why the fold declines rather than guessing which. A
subordinate is never put in any of those. Its five states map to five
distinct workload states, so the readers that ask whether a follower is
running, errored or unreachable can ask the rows instead.

That is a property of who writes the field, not of the type, so it is
checked here rather than assumed.
"""

import ast
import pathlib

import pytest

from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.schemas.workloads import WorkloadStateEnum
from gpustack.server.model_instance_workloads import (
    SUBORDINATE_STATES,
    to_subordinate_state,
    to_workload_state,
)


@pytest.mark.parametrize("state", sorted(SUBORDINATE_STATES, key=str))
def test_a_subordinate_state_survives_the_round_trip(state):
    assert to_subordinate_state(to_workload_state(state)) is state


def test_no_two_subordinate_states_share_a_workload_state():
    """Round-tripping each one is not enough on its own: it would still hold
    if two of them collided and the inverse happened to name the one being
    checked. What makes the inverse a function is that the images are
    distinct."""
    images = [to_workload_state(s) for s in SUBORDINATE_STATES]

    assert len(set(images)) == len(images)


def test_the_instances_own_lifecycle_does_not():
    """The contrast is the point. Five instance states share one workload
    state, so the same inverse applied to a leader would be a guess."""
    collapsed = [
        s
        for s in ModelInstanceStateEnum
        if to_workload_state(s) is WorkloadStateEnum.PENDING
    ]

    assert len(collapsed) > 1
    assert set(collapsed) - SUBORDINATE_STATES


def test_a_state_no_subordinate_reaches_has_no_reading():
    """Rather than a guess: a caller comparing the two readings skips it
    instead of reporting a difference it cannot act on."""
    assert to_subordinate_state(WorkloadStateEnum.SUCCEEDED) is None
    assert to_subordinate_state(None) is None


def _subordinate_states_written():
    """Every ``ModelInstanceStateEnum`` member assigned to a subordinate's
    ``state`` anywhere in the package, by the name the target is written
    under."""
    found = {}
    for path in pathlib.Path("gpustack").rglob("*.py"):
        if "migrations" in str(path):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Attribute):
                continue
            value = ast.unparse(node.value)
            if not value.startswith("ModelInstanceStateEnum."):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Attribute) or target.attr != "state":
                    continue
                base = ast.unparse(target.value)
                if "subordinate" in base or base == "sw" or base.endswith(".sw"):
                    found.setdefault(value.split(".", 1)[1], set()).add(
                        f"{path}: {ast.unparse(target)}"
                    )
    return found


def test_nothing_puts_a_subordinate_in_a_state_that_would_not_survive():
    """The guard on the premise. ``SUBORDINATE_STATES`` is a claim about what
    the writers do, and a new one reaching for, say, DOWNLOADING would make
    every follower-state reading silently wrong -- it would read back as
    pending, which the callers take to mean "not running yet" and would then
    be right for the wrong reason until the day it is not.
    """
    written = _subordinate_states_written()

    assert written, "found no subordinate state writes at all; the scan is stale"
    unreachable = set(written) - {s.name for s in SUBORDINATE_STATES}
    assert not unreachable, {k: sorted(written[k]) for k in unreachable}
