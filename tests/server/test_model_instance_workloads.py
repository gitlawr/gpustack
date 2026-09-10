"""
Compiling a model instance into the workloads that run it.

Asserted against the production code the mapping has to agree with:
``get_deployment_metadata`` decides the container names, and
``_assign_ports`` decides what the port list means.
"""

import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
    Workload,
    WorkloadOwnerKindEnum,
    WorkloadRestartPolicyEnum,
    WorkloadRoleEnum,
    WorkloadStateEnum,
)
from gpustack.worker.serve_manager import ServeManager
from gpustack.server.model_instance_workloads import (
    SERVICE_PORT,
    PRE_EXECUTION_STATES,
    FoldDeclineReason,
    aggregate_instance_state,
    SPEC_FIELDS,
    aggregate_instance_runtime,
    fold_decline_reason,
    instance_ports,
    EXECUTION_FIELDS,
    mirror_execution_state,
    rows_are_behind,
    to_workload_state,
    compile_model_instance,
    named_ports,
    sync_model_instance_workloads,
    workload_spec,
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


@pytest.mark.parametrize("state", sorted(PRE_EXECUTION_STATES))
def test_states_before_a_container_exists_map_to_pending(state):
    """Scheduling and model-file preparation are the instance's lifecycle, not
    the workload's; if a workload had to represent them it would be the domain
    resource. The instance's own STARTING is one of them -- the server sets it
    when the model files are ready, with nothing launched yet."""
    leader = compile_model_instance(_instance(state=state))[0]

    assert leader.state == WorkloadStateEnum.PENDING


def test_a_spawned_process_is_a_starting_workload():
    """INITIALIZING is what the worker writes when it spawns the process, so
    by then a container exists and the workload is no longer pending. Mapping
    it to pending left the leader's workload reporting nothing between
    scheduling and the first passing health check."""
    leader = compile_model_instance(
        _instance(state=ModelInstanceStateEnum.INITIALIZING)
    )[0]

    assert leader.state == WorkloadStateEnum.STARTING


@pytest.mark.parametrize(
    "instance_state",
    [
        ModelInstanceStateEnum.RUNNING,
        ModelInstanceStateEnum.UNREACHABLE,
        ModelInstanceStateEnum.ERROR,
    ],
)
def test_an_execution_state_survives_the_round_trip(instance_state):
    """The worker mirrors the instance onto the workload and the fold reads it
    back; a state that does not survive both directions would be silently
    rewritten the moment the fold becomes authoritative."""
    workloads = [_workload(0, to_workload_state(instance_state))]

    folded = aggregate_instance_state(workloads)

    assert folded["state"] == instance_state


@pytest.mark.parametrize(
    "coming_up",
    [ModelInstanceStateEnum.INITIALIZING, ModelInstanceStateEnum.STARTING],
)
def test_the_fold_says_nothing_while_an_instance_is_coming_up(coming_up):
    """Both of these mirror onto a workload that is launched but not yet
    healthy -- INITIALIZING onto starting, the instance's own STARTING onto
    pending, since the server sets it before anything is launched. Two states
    onto one means the way back is not a function, so the fold leaves the
    instance's own lifecycle alone rather than picking one of them."""
    workloads = [_workload(0, to_workload_state(coming_up))]

    assert aggregate_instance_state(workloads) is None


def test_no_message_has_one_spelling():
    """An instance with nothing to report carries the empty string and a
    workload that was never given a message carries NULL. Reporting NULL back
    would make every healthy instance look like a disagreement."""
    folded = aggregate_instance_state(
        [_workload(0, WorkloadStateEnum.RUNNING, state_message=None)]
    )

    assert folded["state_message"] == ""


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


# ---------------------------------------------------------------------------
# Folding workload state back onto the instance
# ---------------------------------------------------------------------------


def _workload(group_index, state, state_message=None, worker_id=None):
    return Workload(
        name=f"mi-{group_index}",
        owner_kind=WorkloadOwnerKindEnum.MODEL_INSTANCE,
        owner_id=1,
        worker_id=worker_id if worker_id is not None else 100 + group_index,
        group_index=group_index,
        state=state,
        state_message=state_message,
    )


def test_a_standalone_instance_takes_its_leaders_state():
    folded = aggregate_instance_state(
        [_workload(0, WorkloadStateEnum.RUNNING, state_message="")]
    )

    assert folded == {"state": "running", "state_message": ""}


def test_a_pending_leader_says_nothing():
    """The mapping is not reversible: scheduling, initializing and downloading
    all mirror onto a pending workload, so folding it back would replace the
    instance's richer state with the poorer one."""
    assert aggregate_instance_state([_workload(0, WorkloadStateEnum.PENDING)]) is None


def test_a_group_still_coming_up_holds_the_instance_where_it_is():
    folded = aggregate_instance_state(
        [
            _workload(0, WorkloadStateEnum.RUNNING),
            _workload(1, WorkloadStateEnum.STARTING),
        ]
    )

    assert folded is None


def test_a_failed_follower_takes_the_instance_down_with_it():
    folded = aggregate_instance_state(
        [
            _workload(0, WorkloadStateEnum.RUNNING),
            _workload(1, WorkloadStateEnum.ERROR, "boom", worker_id=2),
        ]
    )

    assert folded["state"] == ModelInstanceStateEnum.ERROR
    assert "subordinate worker 2" in folded["state_message"]
    assert "boom" in folded["state_message"]


def test_an_errored_follower_outranks_an_unreachable_one():
    folded = aggregate_instance_state(
        [
            _workload(0, WorkloadStateEnum.RUNNING),
            _workload(1, WorkloadStateEnum.UNREACHABLE, "gone"),
            _workload(2, WorkloadStateEnum.ERROR, "boom"),
        ]
    )

    assert folded["state"] == ModelInstanceStateEnum.ERROR


def test_an_unreachable_follower_makes_the_instance_unreachable():
    folded = aggregate_instance_state(
        [
            _workload(0, WorkloadStateEnum.RUNNING),
            _workload(1, WorkloadStateEnum.UNREACHABLE, "gone", worker_id=3),
        ]
    )

    assert folded["state"] == ModelInstanceStateEnum.UNREACHABLE
    assert "subordinate worker 3" in folded["state_message"]


def test_all_followers_running_leaves_the_leader_in_charge():
    folded = aggregate_instance_state(
        [
            _workload(0, WorkloadStateEnum.RUNNING, state_message=""),
            _workload(1, WorkloadStateEnum.RUNNING),
            _workload(2, WorkloadStateEnum.RUNNING),
        ]
    )

    assert folded == {"state": "running", "state_message": ""}


def test_a_group_with_no_leader_says_nothing():
    """Followers alone cannot decide the instance's state, and a group missing
    its leader is a compile that has not finished rather than a verdict."""
    assert aggregate_instance_state([_workload(1, WorkloadStateEnum.RUNNING)]) is None


@pytest.mark.parametrize(
    "states",
    list(
        itertools.product(
            [
                ModelInstanceStateEnum.RUNNING,
                ModelInstanceStateEnum.ERROR,
                ModelInstanceStateEnum.UNREACHABLE,
                ModelInstanceStateEnum.STARTING,
            ],
            repeat=2,
        )
    ),
)
def test_the_fold_agrees_with_what_the_worker_decides_today(states):
    """The fold replaces ServeManager._get_main_worker_distributed_state, so
    every combination of two follower states has to come out the same. Only
    the verdict is compared: the message names the worker by id here and by ip
    there, since a workload carries the binding and not the address."""
    mi = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=2
    )
    for sw, state in zip(mi.distributed_servers.subordinate_workers, states):
        sw.state = state
        sw.state_message = f"because {state}"

    reference = ServeManager._get_main_worker_distributed_state(mi)
    folded = aggregate_instance_state(compile_model_instance(mi))

    if reference is None:
        # The leader governs; the fold reports its state rather than nothing.
        assert folded == {"state": "running", "state_message": ""}
    elif reference.get("state") is None:
        assert folded is None  # hold
    else:
        assert folded["state"] == reference["state"]


# ---------------------------------------------------------------------------
# Keeping the rows in step with the binding
# ---------------------------------------------------------------------------


class _Row(SimpleNamespace):
    pass


def _existing(group_index):
    return _Row(group_index=group_index, update=AsyncMock(), delete=AsyncMock())


def _matching_row(compiled):
    """A row that already says exactly what the spec asks for."""
    row = _Row(update=AsyncMock(), delete=AsyncMock())
    for name in workload_spec(compiled).model_fields_set:
        setattr(row, name, getattr(compiled, name))
    row.group_index = compiled.group_index
    return row


@pytest.mark.asyncio
async def test_sync_creates_a_row_per_container(monkeypatch):
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.all_by_fields",
        AsyncMock(return_value=[]),
    )
    create = AsyncMock()
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.create", create
    )

    await sync_model_instance_workloads(
        MagicMock(),
        _instance(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=2
        ),
    )

    assert create.await_count == 3


@pytest.mark.asyncio
async def test_sync_updates_in_place_so_the_row_id_does_not_move(monkeypatch):
    """The worker reports against these rows and keys its logs by their id;
    recreating them on every reconcile would move both."""
    existing = [_existing(0)]
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.all_by_fields",
        AsyncMock(return_value=existing),
    )
    create = AsyncMock()
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.create", create
    )

    await sync_model_instance_workloads(MagicMock(), _instance())

    create.assert_not_awaited()
    existing[0].update.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_drops_rows_for_followers_that_are_gone(monkeypatch):
    existing = [_existing(0), _existing(1)]
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.all_by_fields",
        AsyncMock(return_value=existing),
    )
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.create", AsyncMock()
    )

    await sync_model_instance_workloads(MagicMock(), _instance())

    existing[0].update.assert_awaited_once()
    existing[1].delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_writes_the_spec_not_the_state(monkeypatch):
    existing = [_existing(0)]
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.all_by_fields",
        AsyncMock(return_value=existing),
    )
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.create", AsyncMock()
    )

    await sync_model_instance_workloads(MagicMock(), _instance())

    update = existing[0].update.await_args.args[1]
    assert update.worker_id == 1
    assert "state" not in update.model_fields_set


@pytest.mark.asyncio
async def test_sync_does_not_rewrite_a_row_that_already_matches(monkeypatch):
    """update publishes an event whether or not anything changed, and this
    runs on every event about the instance. Writing unconditionally turns one
    instance event into an event per workload, each of which wakes the fold
    that reads them back."""
    instance = _instance()
    compiled = compile_model_instance(instance)[0]
    current = _matching_row(compiled)
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.all_by_fields",
        AsyncMock(return_value=[current]),
    )
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.create", AsyncMock()
    )

    await sync_model_instance_workloads(MagicMock(), instance)

    current.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_writes_when_the_binding_moved(monkeypatch):
    instance = _instance()
    compiled = compile_model_instance(instance)[0]
    current = _matching_row(compiled)
    current.worker_id = 99  # rescheduled elsewhere

    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.all_by_fields",
        AsyncMock(return_value=[current]),
    )
    monkeypatch.setattr(
        "gpustack.server.model_instance_workloads.Workload.create", AsyncMock()
    )

    await sync_model_instance_workloads(MagicMock(), instance)

    current.update.assert_awaited_once()


def test_the_fold_accepts_a_row_loaded_from_the_database():
    """A row loaded through the ORM carries a plain string, not an enum: the
    column is declared String so PostgreSQL never renders an enum cast."""
    leader = Workload(
        name="mi-0",
        owner_kind=WorkloadOwnerKindEnum.MODEL_INSTANCE,
        owner_id=1,
        group_index=0,
        state="running",
        state_message="",
    )

    assert aggregate_instance_state([leader]) == {
        "state": ModelInstanceStateEnum.RUNNING,
        "state_message": "",
    }


def test_a_state_the_instance_has_no_name_for_folds_to_nothing():
    """succeeded is task-shaped; a model instance workload never reaches it,
    and inventing an instance state for it would be worse than silence."""
    leader = Workload(
        name="mi-0",
        owner_kind=WorkloadOwnerKindEnum.MODEL_INSTANCE,
        owner_id=1,
        group_index=0,
        state="succeeded",
    )

    assert aggregate_instance_state([leader]) is None


@pytest.mark.parametrize(
    "workloads,expected",
    [
        ([], FoldDeclineReason.NO_LEADER),
        (
            [_workload(1, WorkloadStateEnum.RUNNING)],
            FoldDeclineReason.NO_LEADER,
        ),
        (
            [_workload(0, WorkloadStateEnum.PENDING)],
            FoldDeclineReason.LEADER_PENDING,
        ),
        (
            [
                _workload(0, WorkloadStateEnum.RUNNING),
                _workload(1, WorkloadStateEnum.STARTING),
            ],
            FoldDeclineReason.FOLLOWERS_NOT_READY,
        ),
        (
            [_workload(0, WorkloadStateEnum.SUCCEEDED)],
            FoldDeclineReason.NOT_AN_INSTANCE_STATE,
        ),
    ],
)
def test_every_decline_branch_reports_its_reason(workloads, expected):
    """The reason is read off a second walk of the same conditions, so it can
    drift from the branch it names. Each case asserts both at once: the fold
    declines, and the reason says which branch did it."""
    assert aggregate_instance_state(workloads) is None
    assert fold_decline_reason(workloads) is expected


@pytest.mark.parametrize(
    "workloads",
    [
        [_workload(0, WorkloadStateEnum.RUNNING)],
        [
            _workload(0, WorkloadStateEnum.RUNNING),
            _workload(1, WorkloadStateEnum.RUNNING),
        ],
        [
            _workload(0, WorkloadStateEnum.RUNNING),
            _workload(1, WorkloadStateEnum.ERROR),
        ],
    ],
)
def test_a_fold_that_speaks_reports_no_reason(workloads):
    assert aggregate_instance_state(workloads) is not None
    assert fold_decline_reason(workloads) is None


# ---------------------------------------------------------------------------
# The fold against the worker it has to reproduce
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "follower_state,instance_state",
    [
        (WorkloadStateEnum.ERROR, ModelInstanceStateEnum.ERROR),
        (WorkloadStateEnum.UNREACHABLE, ModelInstanceStateEnum.UNREACHABLE),
    ],
)
def test_the_fold_words_a_follower_failure_the_way_the_worker_does(
    follower_state, instance_state
):
    """The fold is meant to replace ``_get_main_worker_distributed_state``, so
    it has to produce the same message, not merely the same state. The worker
    names the failing node by IP; a workload row carries only the id, and using
    it would both read as a disagreement here and reword a message users see
    once the fold is authoritative."""
    from gpustack.worker.serve_manager import ServeManager

    failure = "CUDA out of memory"
    instance = ModelInstance(
        id=1,
        name="mi",
        worker_id=100,
        state=ModelInstanceStateEnum.RUNNING,
        distributed_servers=DistributedServers(
            mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
            subordinate_workers=[
                ModelInstanceSubordinateWorker(
                    worker_id=101,
                    worker_name="worker-101",
                    worker_ip="10.0.0.101",
                    state=ModelInstanceStateEnum(instance_state),
                    state_message=failure,
                )
            ],
        ),
    )
    from_worker = ServeManager._get_main_worker_distributed_state(instance)

    folded = aggregate_instance_state(
        [
            _workload(0, WorkloadStateEnum.RUNNING),
            _workload(1, follower_state, state_message=failure, worker_id=101),
        ],
        worker_ips={101: "10.0.0.101"},
    )

    assert folded["state"] == from_worker["state"]
    assert folded["state_message"] == from_worker["state_message"]
    assert "10.0.0.101" in folded["state_message"]


def test_an_unresolvable_worker_falls_back_to_its_id():
    """A worker row deleted out from under a running group still has to yield
    a message rather than an exception."""
    folded = aggregate_instance_state(
        [
            _workload(0, WorkloadStateEnum.RUNNING),
            _workload(1, WorkloadStateEnum.ERROR, state_message="boom", worker_id=101),
        ],
        worker_ips={},
    )

    assert "subordinate worker 101: boom." in folded["state_message"]


# ---------------------------------------------------------------------------
# The worker as the container has to name it
# ---------------------------------------------------------------------------


def test_a_group_carries_the_addresses_its_containers_need():
    """A backend building an argument vector for a distributed run needs its
    peers by address, and it runs on the worker with no session to join
    ``workers`` with. That is why the embedded list denormalises these, and
    the rows replacing it have to as well."""
    mi = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=1
    )
    mi.worker_name, mi.worker_ip, mi.worker_ifname = "w-main", "10.0.0.1", "eth0"
    sw = mi.distributed_servers.subordinate_workers[0]

    leader, follower = compile_model_instance(mi)

    assert (leader.worker_name, leader.worker_ip, leader.worker_ifname) == (
        "w-main",
        "10.0.0.1",
        "eth0",
    )
    assert (follower.worker_name, follower.worker_ip, follower.worker_ifname) == (
        sw.worker_name,
        sw.worker_ip,
        sw.worker_ifname,
    )


def test_a_worker_that_changes_address_propagates():
    """They are part of the spec, so a reconcile rewrites them. The embedded
    list is only written when the binding is, which leaves it as it was when
    the instance was scheduled."""
    assert {"worker_name", "worker_ip", "worker_ifname"} <= SPEC_FIELDS

    mi = _instance()
    mi.worker_ip = "10.0.0.9"
    spec = workload_spec(compile_model_instance(mi)[0])

    assert spec.worker_ip == "10.0.0.9"


# ---------------------------------------------------------------------------
# What the container is, as opposed to how it is doing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "port,ports",
    [(8000, [8000, 8001, 8002]), (8000, [8000]), (8000, None), (None, None)],
)
def test_ports_survive_the_round_trip(port, ports):
    """Only the first is named; the rest are recovered by position. mi.port is
    what requests are proxied to, so a shift here sends them nowhere."""
    assert instance_ports(named_ports(port, ports)) == (
        port,
        [port] if port is not None and not ports else ports,
    )


def test_a_gap_in_the_sequence_ends_it():
    """The list is positional on the instance, so closing a hole would shift
    everything after it onto the wrong port."""
    assert instance_ports({"service": 8000, "port2": 8002}) == (8000, [8000])


def test_the_runtime_comes_from_the_leader():
    workloads = [
        _workload(0, WorkloadStateEnum.RUNNING),
        _workload(1, WorkloadStateEnum.RUNNING),
    ]
    workloads[0].ports = {"service": 8000, "port1": 8001}
    workloads[0].pid = 4242
    workloads[1].ports = {"service": 9000}
    workloads[1].pid = 5353

    runtime = aggregate_instance_runtime(workloads)

    assert runtime["port"] == 8000
    assert runtime["ports"] == [8000, 8001]
    assert runtime["pid"] == 4242


def test_a_field_the_row_has_nothing_for_is_left_alone():
    """The row is filled by the mirror, so it lags the instance by one write.
    Clearing a live port because the row has not caught up would take the
    instance off the air for a reason unrelated to it."""
    workload = _workload(0, WorkloadStateEnum.RUNNING)
    workload.ports = None
    workload.pid = None

    runtime = aggregate_instance_runtime([workload])

    assert "port" not in runtime
    assert "ports" not in runtime
    assert "pid" not in runtime


# ---------------------------------------------------------------------------
# Reading rows the instance has already moved past
# ---------------------------------------------------------------------------


def _at(seconds):
    from datetime import datetime, timedelta, timezone

    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def test_rows_a_write_behind_are_not_folded():
    """The worker writes the instance and then mirrors, so in between the two
    genuinely disagree. Deciding from that window put a container that had
    just died back to RUNNING for sixty milliseconds on a real run."""
    instance = _instance()
    instance.updated_at = _at(10)
    workload = _workload(0, WorkloadStateEnum.RUNNING)
    workload.updated_at = _at(5)

    assert rows_are_behind(instance, [workload]) is True


def test_a_follower_moving_is_news_too():
    """Compared against the newest row, not the leader's: a follower changing
    is something the fold has to be free to act on."""
    instance = _instance()
    instance.updated_at = _at(10)
    leader = _workload(0, WorkloadStateEnum.RUNNING)
    leader.updated_at = _at(5)
    follower = _workload(1, WorkloadStateEnum.ERROR)
    follower.updated_at = _at(20)

    assert rows_are_behind(instance, [leader, follower]) is False


def test_a_row_written_after_the_instance_is_not_behind():
    instance = _instance()
    instance.updated_at = _at(5)
    workload = _workload(0, WorkloadStateEnum.RUNNING)
    workload.updated_at = _at(10)

    assert rows_are_behind(instance, [workload]) is False


def test_a_restart_count_is_not_folded_backwards():
    """The row lags the instance by a write, so it reports the count from
    before the restart. Folding that back resets what the backoff escalates
    on and leaves an instance restarting at the base delay for ever."""
    instance = _instance()
    instance.restart_count = 1
    workload = _workload(0, WorkloadStateEnum.RUNNING)
    workload.restart_count = 0

    assert "restart_count" not in aggregate_instance_runtime([workload], instance)


def test_a_restart_count_that_moved_forward_is_folded():
    instance = _instance()
    instance.restart_count = 1
    workload = _workload(0, WorkloadStateEnum.RUNNING)
    workload.restart_count = 2

    assert aggregate_instance_runtime([workload], instance)["restart_count"] == 2


# ---------------------------------------------------------------------------
# Deriving the rows from the write they follow
# ---------------------------------------------------------------------------


def _mirror_rows(states):
    """Rows carrying a state and nothing else, so any other field the mirror
    writes shows up as a change."""
    return [
        SimpleNamespace(
            group_index=group_index,
            state=state,
            state_message=None,
            ports=None,
            pid=None,
            restart_count=0,
            last_restart_time=None,
            progress=None,
            update=AsyncMock(),
        )
        for group_index, state in enumerate(states)
    ]


def _rows_matching(instance):
    """Rows already saying exactly what the instance does."""
    return [
        SimpleNamespace(
            **{name: getattr(w, name) for name in EXECUTION_FIELDS},
            group_index=w.group_index,
            update=AsyncMock(),
        )
        for w in compile_model_instance(instance)
    ]


@pytest.mark.asyncio
async def test_the_rows_follow_the_instance_write(monkeypatch):
    """Two processes write an instance -- the worker's sync passes and the
    provisioning subprocess -- and while the worker mirrored, only one of them
    did. Deriving the rows from the write removes the ordering question
    instead of guarding against it."""
    instance = _instance(state=ModelInstanceStateEnum.ERROR)
    instance.state_message = "Error (exit code 137)"
    rows = _mirror_rows([WorkloadStateEnum.RUNNING])
    monkeypatch.setattr(Workload, "all_by_fields", AsyncMock(return_value=rows))

    await mirror_execution_state(MagicMock(), instance)

    applied = rows[0].update.await_args[0][1]
    assert applied["state"] == WorkloadStateEnum.ERROR
    assert applied["state_message"] == "Error (exit code 137)"


@pytest.mark.asyncio
async def test_a_row_already_saying_it_is_left_alone(monkeypatch):
    """update publishes an event whether or not anything changed, and this
    runs on every instance write."""
    instance = _instance(state=ModelInstanceStateEnum.RUNNING)
    rows = _rows_matching(instance)
    monkeypatch.setattr(Workload, "all_by_fields", AsyncMock(return_value=rows))

    await mirror_execution_state(MagicMock(), instance)

    rows[0].update.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_follower_follows_its_own_entry(monkeypatch):
    """The leader's state is the instance's; a follower's is its entry in the
    subordinate list, and putting the leader's on both would report a node
    that had failed as running."""
    instance = _instance(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER, followers=1
    )
    instance.state = ModelInstanceStateEnum.RUNNING
    instance.distributed_servers.subordinate_workers[0].state = (
        ModelInstanceStateEnum.ERROR
    )
    rows = _mirror_rows([WorkloadStateEnum.PENDING, WorkloadStateEnum.PENDING])
    monkeypatch.setattr(Workload, "all_by_fields", AsyncMock(return_value=rows))

    await mirror_execution_state(MagicMock(), instance)

    assert rows[0].update.await_args[0][1]["state"] == WorkloadStateEnum.RUNNING
    assert rows[1].update.await_args[0][1]["state"] == WorkloadStateEnum.ERROR


def test_an_empty_collection_compiles_to_none_not_an_empty_list():
    """The columns default to None, so writing [] gives the same absence two
    spellings and every comparison against the instance reports a difference
    that is not one. Caught on a real instance with no gpu_addresses."""
    mi = _instance()
    mi.gpu_addresses = None

    leader = compile_model_instance(mi)[0]

    assert leader.gpu_addresses is None


def test_a_non_empty_collection_is_copied():
    mi = _instance()
    mi.gpu_addresses = ["0000:01:00.0"]

    leader = compile_model_instance(mi)[0]

    assert leader.gpu_addresses == ["0000:01:00.0"]
    assert leader.gpu_addresses is not mi.gpu_addresses
