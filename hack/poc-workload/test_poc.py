"""
Stage 0 POC: does the proposed Workload resource actually hold a distributed
model instance?

Each test states what it is proving about the resource model. Failures here are
findings, not bugs to paper over.
"""

import itertools

import pytest

from gpustack.schemas.models import (
    ComputedResourceClaim,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
)
from gpustack.worker.serve_manager import ServeManager

from compile_model_instance import (
    aggregate_group_state,
    compile_model_instance,
    resolve_blocked,
)
from workload import (
    StartGateEnum,
    WorkloadRoleEnum,
    WorkloadStateEnum,
)


def _instance(mode=None, followers=0, state=ModelInstanceStateEnum.RUNNING, **kw):
    subordinates = [
        ModelInstanceSubordinateWorker(
            worker_id=100 + i,
            worker_name=f"worker-{100 + i}",
            worker_ip=f"10.0.0.{100 + i}",
            worker_ifname="eth0",
            total_gpus=2,
            gpu_type="cuda",
            gpu_indexes=[0, 1],
            gpu_addresses=[f"0000:0{i}:00.0"],
            computed_resource_claim=ComputedResourceClaim(vram={0: 1024}),
            download_progress=100.0,
            pid=2000 + i,
            ports=[40000 + i],
            arguments=["--rank", str(i + 1)],
            state=ModelInstanceStateEnum.RUNNING,
        )
        for i in range(followers)
    ]
    mi = ModelInstance(
        id=1,
        name="distributed-instance",
        model_id=1,
        model_name="test",
        worker_id=1,
        worker_name="worker-1",
        worker_ip="10.0.0.1",
        worker_ifname="eth0",
        cluster_id=7,
        owner_principal_id=42,
        state=state,
        port=8000,
        ports=[8000, 8001],
        pid=999,
        gpu_type="cuda",
        gpu_indexes=[0],
        gpu_addresses=["0000:00:00.0"],
        computed_resource_claim=ComputedResourceClaim(vram={0: 2048}),
    )
    if subordinates or mode is not None:
        mi.distributed_servers = DistributedServers(
            mode=mode or DistributedServerCoordinateModeEnum.DELEGATED,
            subordinate_workers=subordinates,
        )
    for key, value in kw.items():
        setattr(mi, key, value)
    return mi


# ---------------------------------------------------------------------------
# Identity: container names must not change
# ---------------------------------------------------------------------------


def test_standalone_instance_compiles_to_one_leader():
    """A non-distributed instance is a group of one, so nothing downstream has
    to special-case group size."""
    workloads = compile_model_instance(_instance())

    assert len(workloads) == 1
    assert workloads[0].role == WorkloadRoleEnum.LEADER
    assert workloads[0].group_index == 0
    assert workloads[0].name == "distributed-instance"


@pytest.mark.parametrize("followers", [1, 3])
def test_workload_names_match_current_deployment_metadata(followers):
    """The workload name is the container name. If compilation disagreed with
    get_deployment_metadata, migration would orphan every running container."""
    mi = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=followers
    )
    workloads = compile_model_instance(mi)

    for workload in workloads:
        expected = mi.get_deployment_metadata(workload.worker_id)
        assert expected is not None
        assert workload.name == expected.name
        assert (workload.role == WorkloadRoleEnum.LEADER) == expected.distributed_leader


# ---------------------------------------------------------------------------
# Field coverage: nothing may be silently dropped
# ---------------------------------------------------------------------------


# Fields deliberately not carried onto the Workload, with the reason. Anything
# not listed here and not asserted below is an unplanned loss.
SUBORDINATE_FIELDS_NOT_CARRIED = {
    "total_gpus": "derivable from gpu_indexes; redundant on the row",
}


def test_every_subordinate_field_lands_somewhere():
    """ModelInstanceSubordinateWorker is the closest thing to a Workload that
    exists today. Anything it carries that the Workload cannot hold is a hole
    in the resource model."""
    mi = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=1
    )
    sw = mi.distributed_servers.subordinate_workers[0]
    follower = compile_model_instance(mi)[1]

    carried = {
        "worker_id": follower.worker_id,
        "worker_name": follower.worker_name,
        "worker_ip": follower.worker_ip,
        "worker_ifname": follower.worker_ifname,
        "gpu_type": follower.gpu_type,
        "gpu_indexes": follower.gpu_indexes,
        "gpu_addresses": follower.gpu_addresses,
        "computed_resource_claim": follower.computed_resource_claim,
        "pid": follower.pid,
        "arguments": follower.arguments,
        "state_message": follower.state_message,
        "download_progress": follower.progress,
    }

    declared = set(ModelInstanceSubordinateWorker.model_fields)
    accounted = set(carried) | set(SUBORDINATE_FIELDS_NOT_CARRIED) | {"state", "ports"}
    assert declared - accounted == set(), "unaccounted subordinate fields"

    for name, value in carried.items():
        assert value == getattr(sw, name), f"{name} did not survive compilation"

    assert follower.state == WorkloadStateEnum.RUNNING
    assert list(follower.ports.values()) == sw.ports


def test_pre_execution_instance_states_have_no_workload_counterpart():
    """ANALYZING/SCHEDULED/DOWNLOADING happen before a container exists. If the
    Workload had to represent them, it would be the domain resource."""
    for state in (
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.ANALYZING,
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.INITIALIZING,
        ModelInstanceStateEnum.DOWNLOADING,
    ):
        leader = compile_model_instance(_instance(state=state))[0]
        assert leader.state == WorkloadStateEnum.PENDING


# ---------------------------------------------------------------------------
# Start ordering
# ---------------------------------------------------------------------------


def test_initialize_later_orders_followers_behind_leader_and_each_other():
    """Reproduces _dispatch_model_instance_event: a follower waits for the
    leader to have begun, and for every earlier follower to have settled (the
    phantom-read guard)."""
    workloads = compile_model_instance(
        _instance(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=3
        )
    )

    assert workloads[0].start_after == []
    assert [(d.group_index, d.gate) for d in workloads[1].start_after] == [
        (0, StartGateEnum.STARTED)
    ]
    assert [(d.group_index, d.gate) for d in workloads[3].start_after] == [
        (0, StartGateEnum.STARTED),
        (1, StartGateEnum.READY),
        (2, StartGateEnum.READY),
    ]


def test_run_first_orders_the_leader_behind_every_follower():
    workloads = compile_model_instance(
        _instance(mode=DistributedServerCoordinateModeEnum.RUN_FIRST, followers=2)
    )

    assert [(d.group_index, d.gate) for d in workloads[0].start_after] == [
        (1, StartGateEnum.READY),
        (2, StartGateEnum.READY),
    ]
    assert all(w.start_after == [] for w in workloads[1:])


def test_delegated_followers_are_reservations_not_containers():
    """DELEGATED hands the subordinate containers to another framework; today
    the worker returns early for them. The row still has to exist to hold the
    resource claim."""
    workloads = compile_model_instance(
        _instance(mode=DistributedServerCoordinateModeEnum.DELEGATED, followers=2)
    )

    assert workloads[0].managed is True
    assert all(w.managed is False for w in workloads[1:])
    assert all(w.computed_resource_claim is not None for w in workloads[1:])


def test_blocked_workloads_release_in_dependency_order():
    """The controller walks the group forward one gate at a time; the worker
    only ever sees workloads that are already allowed to start."""
    workloads = compile_model_instance(
        _instance(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=2
        )
    )
    for w in workloads[1:]:
        w.state = WorkloadStateEnum.BLOCKED
    workloads[0].state = WorkloadStateEnum.PENDING

    assert resolve_blocked(workloads) == []

    workloads[0].state = WorkloadStateEnum.STARTING
    assert [w.group_index for w in resolve_blocked(workloads)] == [1]

    workloads[1].state = WorkloadStateEnum.RUNNING
    assert [w.group_index for w in resolve_blocked(workloads)] == [2]


# ---------------------------------------------------------------------------
# State aggregation: must reproduce the current behaviour exactly
# ---------------------------------------------------------------------------


AGGREGATION_STATES = [
    ModelInstanceStateEnum.RUNNING,
    ModelInstanceStateEnum.ERROR,
    ModelInstanceStateEnum.UNREACHABLE,
    ModelInstanceStateEnum.STARTING,
]


@pytest.mark.parametrize(
    "states", list(itertools.product(AGGREGATION_STATES, repeat=2))
)
def test_aggregation_matches_current_distributed_state(states):
    """Every combination of two follower states must fold to what
    _get_main_worker_distributed_state produces today."""
    mi = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=2
    )
    for sw, state in zip(mi.distributed_servers.subordinate_workers, states):
        sw.state = state
        sw.state_message = f"because {state}"

    reference = ServeManager._get_main_worker_distributed_state(mi)
    actual = aggregate_group_state(compile_model_instance(mi))

    if reference is None:
        assert actual is None
        return
    if reference.get("state") is None:
        # "hold": the instance keeps whatever state it has.
        assert actual == {"hold": True}
        return
    assert actual["state"] == reference["state"]
    assert actual["state_message"] == reference["state_message"]


def test_aggregation_is_none_without_followers():
    assert aggregate_group_state(compile_model_instance(_instance())) is None


# ---------------------------------------------------------------------------
# Probes: things the first pass got wrong or glossed over
# ---------------------------------------------------------------------------


def test_leader_port_is_not_duplicated_across_names():
    """_assign_ports sets mi.ports = [mi.port] and then extends, so `port` is
    always `ports[0]`. Carrying both as separate named ports would put one
    number under two names and let them drift apart."""
    leader = compile_model_instance(_instance())[0]

    assert len(set(leader.ports.values())) == len(leader.ports)


def test_port_names_beyond_the_first_need_backend_knowledge():
    """_assign_ports lays out ports[1:] by backend and executor -- vLLM/mp gets
    DP-RPC, master-port and VLLM_PORT, vLLM/ray gets DP-RPC only when dp > 1,
    and everything else gets none -- with the connecting port always last.
    A generic compiler cannot name them; the backend's own compiler must."""
    mi = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=1
    )
    mi.port = 8000
    mi.ports = [8000, 8001, 8002, 8003]

    leader = compile_model_instance(mi)[0]

    assert leader.ports["service"] == 8000
    # The rest survive by position and stay unnamed. Naming them here would be
    # a guess that silently mislabels every other backend.
    assert [v for k, v in leader.ports.items() if k != "service"] == [8001, 8002, 8003]


def test_leader_and_follower_may_share_a_worker():
    """get_deployment_metadata looks a workload up by worker_id and returns one
    result, so today a follower co-located with the leader is unreachable: its
    container would never be managed. Compiling by group_index instead of by
    worker lookup removes the ambiguity -- which is why the unique constraint
    needs group_index, not just (owner, worker)."""
    mi = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=1
    )
    mi.distributed_servers.subordinate_workers[0].worker_id = mi.worker_id

    workloads = compile_model_instance(mi)

    assert len(workloads) == 2
    assert workloads[0].worker_id == workloads[1].worker_id
    assert {w.name for w in workloads} == {
        "distributed-instance",
        "distributed-instance-f0",
    }
    # The current lookup collapses both onto the leader.
    assert mi.get_deployment_metadata(mi.worker_id).name == "distributed-instance"


def test_aggregation_does_not_decide_whether_to_write():
    """_get_main_worker_distributed_state folds the state *and* decides whether
    to write it, by comparing with the instance's current state. Aggregation
    should only fold; suppressing no-op writes belongs to the caller, or every
    consumer inherits a flag it has to interpret."""
    mi = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
        followers=1,
        state=ModelInstanceStateEnum.ERROR,
    )
    mi.distributed_servers.subordinate_workers[0].state = ModelInstanceStateEnum.ERROR
    mi.distributed_servers.subordinate_workers[0].state_message = "boom"

    reference = ServeManager._get_main_worker_distributed_state(mi)
    actual = aggregate_group_state(compile_model_instance(mi))

    # Same verdict...
    assert reference["state"] == actual["state"] == ModelInstanceStateEnum.ERROR
    # ...but the reference also says "do not write it", because the instance is
    # already ERROR. That is caller policy, not aggregation.
    assert reference["should_update"] is False
    assert "should_update" not in actual


def test_delegated_subordinate_is_marked_error_by_the_current_sync(monkeypatch):
    """Probe against production code, not the POC.

    The scheduler leaves ``mode`` at DELEGATED for anything outside
    vLLM/SGLang/MindIE, and ``is_gguf_model`` routes GGUF models to a selector
    that does populate ``subordinate_workers``. On such a subordinate worker
    the event path returns early -- DELEGATED containers are another
    framework's business -- so no workload is ever created there. But the
    reconcile pass does not know that: it looks the workload up, finds
    nothing, and writes the subordinate to ERROR.

    Whatever the resource model ends up being, it has to distinguish "a
    container gpustack runs" from "a binding that only reserves resources".
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from gpustack.worker.serve_manager import ServeManager

    subordinate_worker_id = 2
    mi = _instance(mode=DistributedServerCoordinateModeEnum.DELEGATED, followers=1)
    mi.distributed_servers.subordinate_workers[0].worker_id = subordinate_worker_id
    mi.distributed_servers.subordinate_workers[0].state = (
        ModelInstanceStateEnum.RUNNING
    )

    clientset = MagicMock()
    clientset.model_instances.list.return_value = SimpleNamespace(items=[mi])
    manager = ServeManager(
        lambda: subordinate_worker_id, lambda: clientset, SimpleNamespace(log_dir="/tmp")
    )
    manager._inference_backend_manager = MagicMock()

    with (
        patch("gpustack.worker.serve_manager.get_workload", return_value=None),
        patch.object(manager, "_is_provisioning", return_value=False),
        patch.object(manager, "_update_model_instance") as update,
    ):
        manager.sync_model_instances_state()

    assert update.called, "the subordinate was left alone"
    patch_dict = update.call_args[1]
    written = next(iter(patch_dict.values()))
    assert written.state == ModelInstanceStateEnum.ERROR
