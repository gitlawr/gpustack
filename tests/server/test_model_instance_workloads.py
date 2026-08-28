"""
Compiling a model instance into the workloads that run it.

Asserted against the production code the mapping has to agree with:
``get_deployment_metadata`` decides the container names, and
``_assign_ports`` decides what the port list means.
"""

import pytest

from gpustack.schemas.models import (
    ComputedResourceClaim,
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstance,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
)
from gpustack.schemas.workloads import (
    WorkloadOwnerKindEnum,
    WorkloadRestartPolicyEnum,
    WorkloadRoleEnum,
    WorkloadStateEnum,
)
from gpustack.server.model_instance_workloads import (
    SERVICE_PORT,
    compile_model_instance,
    named_ports,
)


def _instance(mode=None, followers=0, state=ModelInstanceStateEnum.RUNNING, **kw):
    subordinates = [
        ModelInstanceSubordinateWorker(
            worker_id=100 + i,
            worker_name=f"worker-{100 + i}",
            worker_ip=f"10.0.0.{100 + i}",
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


def test_a_standalone_instance_compiles_to_one_leader():
    """A workload that runs alone is the leader of a group of one, so nothing
    downstream special-cases group size."""
    workloads = compile_model_instance(_instance())

    assert len(workloads) == 1
    leader = workloads[0]
    assert leader.role == WorkloadRoleEnum.LEADER
    assert leader.group_index == 0
    assert leader.group_key is None
    assert leader.owner_kind == WorkloadOwnerKindEnum.MODEL_INSTANCE
    assert leader.owner_id == 1
    assert leader.owner_principal_id == 42
    assert leader.restart_policy == WorkloadRestartPolicyEnum.ALWAYS


@pytest.mark.parametrize("followers", [1, 3])
def test_names_match_what_the_worker_already_calls_the_containers(followers):
    """The workload name is the container name. If compilation disagreed with
    get_deployment_metadata, adopting these rows would orphan every running
    container."""
    mi = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=followers
    )

    for workload in compile_model_instance(mi):
        expected = mi.get_deployment_metadata(workload.worker_id)
        assert expected is not None
        assert workload.name == expected.name
        assert (workload.role == WorkloadRoleEnum.LEADER) == expected.distributed_leader


def test_a_distributed_instance_ties_its_workloads_together():
    workloads = compile_model_instance(
        _instance(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=2
        )
    )

    assert [w.group_index for w in workloads] == [0, 1, 2]
    assert {w.group_key for w in workloads} == {"distributed-instance"}
    assert [w.role for w in workloads] == [
        WorkloadRoleEnum.LEADER,
        WorkloadRoleEnum.FOLLOWER,
        WorkloadRoleEnum.FOLLOWER,
    ]


def test_every_subordinate_field_survives_compilation():
    """ModelInstanceSubordinateWorker is the closest thing to a workload that
    exists today; anything it carries that a workload cannot hold is a hole in
    the resource model."""
    mi = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=1
    )
    sw = mi.distributed_servers.subordinate_workers[0]
    follower = compile_model_instance(mi)[1]

    assert follower.worker_id == sw.worker_id
    assert follower.gpu_type == sw.gpu_type
    assert follower.gpu_indexes == sw.gpu_indexes
    assert follower.gpu_addresses == sw.gpu_addresses
    assert follower.computed_resource_claim == sw.computed_resource_claim.model_dump()
    assert follower.pid == sw.pid
    assert follower.arguments == sw.arguments
    assert follower.progress == sw.download_progress
    assert follower.state == WorkloadStateEnum.RUNNING
    assert list(follower.ports.values()) == sw.ports


def test_delegated_subordinates_become_reservations_not_workloads():
    """DELEGATED hands the subordinate containers to another framework, so
    gpustack runs one container and only holds the resources on the other
    nodes. A row means a container it runs."""
    workloads = compile_model_instance(
        _instance(mode=DistributedServerCoordinateModeEnum.DELEGATED, followers=2)
    )

    assert len(workloads) == 1
    leader = workloads[0]
    assert leader.group_key is None
    assert [claim.worker_id for claim in leader.reserved_claims] == [100, 101]
    assert leader.reserved_claims[0].gpu_indexes == [0, 1]


def test_coordinated_subordinates_carry_no_reservations():
    workloads = compile_model_instance(
        _instance(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=2
        )
    )

    assert all(w.reserved_claims is None for w in workloads)


@pytest.mark.parametrize(
    "state",
    [
        ModelInstanceStateEnum.PENDING,
        ModelInstanceStateEnum.ANALYZING,
        ModelInstanceStateEnum.SCHEDULED,
        ModelInstanceStateEnum.INITIALIZING,
        ModelInstanceStateEnum.DOWNLOADING,
    ],
)
def test_states_before_a_container_exists_map_to_pending(state):
    """Scheduling and model-file preparation are the instance's lifecycle, not
    the workload's; if a workload had to represent them it would be the domain
    resource."""
    leader = compile_model_instance(_instance(state=state))[0]

    assert leader.state == WorkloadStateEnum.PENDING


def test_the_service_port_is_not_duplicated_under_two_names():
    """_assign_ports sets mi.ports = [mi.port] and then extends, so they are
    one port; carrying both would let them drift apart."""
    ports = named_ports(8000, [8000, 8001])

    assert ports[SERVICE_PORT] == 8000
    assert len(set(ports.values())) == len(ports)


def test_ports_after_the_first_keep_their_position():
    """Their layout depends on backend and executor -- vLLM/mp gets DP-RPC,
    master-port and VLLM_PORT, other backends get none, and the connecting
    port is always last -- so naming them here would mislabel most of them."""
    ports = named_ports(8000, [8000, 8001, 8002, 8003])

    assert [value for key, value in ports.items() if key != SERVICE_PORT] == [
        8001,
        8002,
        8003,
    ]


def test_a_port_list_without_the_service_port_still_names_the_first():
    assert named_ports(None, [40000]) == {SERVICE_PORT: 40000}
    assert named_ports(None, None) == {}
