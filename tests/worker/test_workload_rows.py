"""Reading a model instance's workload rows from the worker.

The readers that use these rows still treat the model instance as
authoritative, so this reading is allowed to come back empty -- but only in
ways that mean "nothing to compare against", never in ways that would make a
launch fail or place a worker wrongly.
"""

import logging
from types import SimpleNamespace

from gpustack.schemas.workloads import WorkloadOwnerKindEnum
from gpustack.worker.controlloop import group_workloads


def _clientset(items=None, raises=None):
    calls = []

    def _list(**kwargs):
        calls.append(kwargs)
        if raises is not None:
            raise raises
        return SimpleNamespace(items=items)

    return SimpleNamespace(workloads=SimpleNamespace(list=_list)), calls


def test_the_group_is_asked_for_by_owner():
    clientset, calls = _clientset(items=[])

    group_workloads(clientset, 7)

    assert calls == [
        {
            "params": {
                "owner_kind": WorkloadOwnerKindEnum.MODEL_INSTANCE.value,
                "owner_id": 7,
            }
        }
    ]


def test_no_page_parameter_is_passed():
    """A page parameter bypasses the watch cache, and with it the client's
    fallback to an unpaginated read when the watch is not authoritative."""
    clientset, calls = _clientset(items=[])

    group_workloads(clientset, 7)

    assert "page" not in calls[0]["params"]


def test_the_rows_come_back_as_a_list():
    rows = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    clientset, _ = _clientset(items=rows)

    assert group_workloads(clientset, 7) == rows


def test_an_empty_page_reads_as_nothing_compiled_yet():
    clientset, _ = _clientset(items=None)

    assert group_workloads(clientset, 7) == []


def test_a_failed_read_does_not_reach_the_caller(caplog):
    """The callers are building an argument vector or locating themselves in a
    group; raising here would fail a launch over a reading that is still only
    compared against the instance."""
    clientset, _ = _clientset(raises=RuntimeError("connection refused"))

    with caplog.at_level(logging.DEBUG):
        assert group_workloads(clientset, 7) == []

    assert "connection refused" in caplog.text
