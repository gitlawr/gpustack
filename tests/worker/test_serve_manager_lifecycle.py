from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event as ThreadEvent
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpustack.schemas.models import (
    DistributedServerCoordinateModeEnum,
    DistributedServers,
    ModelInstanceStateEnum,
    ModelInstanceSubordinateWorker,
    PortBand,
    SourceEnum,
)
from gpustack.server.bus import Event, EventType
from gpustack.worker import serve_manager
from tests.utils.model import new_model, new_model_instance


def _instance(instance_id=1):
    mi = new_model_instance(
        instance_id,
        f"mp-{instance_id}",
        1,
        worker_id=1,
        state=ModelInstanceStateEnum.RUNNING,
        gpu_indexes=[0],
    )
    mi.source = SourceEnum.HUGGING_FACE
    mi.huggingface_repo_id = "Qwen/Qwen3-0.6B"
    mi.worker_ip = "127.0.0.1"
    mi.distributed_servers = DistributedServers(
        mode=DistributedServerCoordinateModeEnum.INITIALIZE_LATER,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=2, gpu_indexes=[0])
        ],
    )
    mi.port = 40000
    mi.ports = [40000, 40011, 40012, 40013, 40001]
    mi.named_ports = {
        serve_manager._VLLM_MP_CONNECTING_BAND_NAME: PortBand(base=40001, count=10)
    }
    return mi


@pytest.fixture
def manager_factory(monkeypatch, tmp_path):
    process = MagicMock()
    process.is_alive.return_value = False
    process.pid = 1234
    monkeypatch.setattr(
        serve_manager.multiprocessing, "Process", MagicMock(return_value=process)
    )
    monkeypatch.setattr(serve_manager, "delete_workload", MagicMock())
    monkeypatch.setattr(serve_manager, "get_workload", MagicMock(return_value=None))
    monkeypatch.setattr(
        serve_manager.registration, "determine_default_registry", lambda _: None
    )

    def build(worker_id=2):
        clientset = MagicMock()
        clientset.model_instances.list.return_value = SimpleNamespace(items=[])
        clientset.cache_service_instances.list.return_value = SimpleNamespace(items=[])
        cfg = SimpleNamespace(
            log_dir=str(tmp_path),
            service_port_range="40000-40063",
            system_default_container_registry=None,
        )
        manager = serve_manager.ServeManager(lambda: worker_id, lambda: clientset, cfg)
        manager._inference_backend_manager = MagicMock()
        model = new_model(
            1,
            "mp-model",
            huggingface_repo_id="Qwen/Qwen3-0.6B",
            backend_parameters=["--distributed-executor-backend", "mp"],
            env={"GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED": "true"},
        )
        for method in (
            "_align_instance_log_dir",
            "_cleanup_old_logs",
            "_purge_instance_logs",
            "_start_container_log_persistence",
            "_stop_container_log_persistence",
            "_update_model_instance",
        ):
            monkeypatch.setattr(manager, method, MagicMock())
        monkeypatch.setattr(manager, "_get_model", MagicMock(return_value=model))
        return manager

    return build


@pytest.mark.parametrize("worker_id", [1, 2])
@pytest.mark.parametrize("receive_delete", [False, True])
def test_repeated_launch_and_delete_releases_ports(
    manager_factory, worker_id, receive_delete
):
    manager = manager_factory(worker_id)
    for instance_id in range(1, 4):
        mi = _instance(instance_id)
        manager._start_model_instance(mi)
        assert manager._assigned_ports[mi.id] == set(range(40000, 40014))
        assert mi.id in manager._model_instance_by_instance_id
        if receive_delete:
            manager._dispatch_model_instance_event(
                Event(type=EventType.DELETED, data=mi)
            )
        manager.sync_model_instances_state()
        serve_manager.delete_workload.assert_called_with(
            mi.get_deployment_metadata(worker_id).name
        )
        assert not manager._assigned_ports
        assert not manager._model_instance_by_instance_id
        assert not manager._provisioning_processes


def test_failed_teardown_preserves_reservations_until_retry(manager_factory):
    manager = manager_factory()
    mi = _instance()
    manager._start_model_instance(mi)
    serve_manager.delete_workload.side_effect = RuntimeError("runtime unavailable")
    manager.sync_model_instances_state()
    assert mi.id in manager._model_instance_by_instance_id
    assert len(manager._assigned_ports[mi.id]) == 14
    serve_manager.delete_workload.side_effect = None
    manager.sync_model_instances_state()
    assert not manager._assigned_ports
    assert not manager._model_instance_by_instance_id


def test_reap_waits_for_launch_to_finish(manager_factory):
    manager = manager_factory()
    mi = _instance()
    entered, finish = ThreadEvent(), ThreadEvent()

    def start():
        entered.set()
        assert finish.wait(timeout=10)

    serve_manager.multiprocessing.Process.return_value.start.side_effect = start
    with ThreadPoolExecutor(max_workers=1) as executor:
        launch = executor.submit(manager._start_model_instance, mi)
        try:
            assert entered.wait(timeout=10)
            manager.sync_model_instances_state()
            serve_manager.delete_workload.assert_not_called()
            assert len(manager._assigned_ports[mi.id]) == 14
        finally:
            finish.set()
        launch.result(timeout=10)
    manager.sync_model_instances_state()
    serve_manager.delete_workload.assert_called_once()
    assert not manager._assigned_ports


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failure_before_process_creation_cleans_reserved_ports(
    manager_factory, cleanup_fails
):
    manager = manager_factory()
    mi = _instance()
    manager._inference_backend_manager.get_backend_by_name.side_effect = RuntimeError(
        "backend unavailable"
    )
    if cleanup_fails:
        serve_manager.delete_workload.side_effect = RuntimeError("runtime unavailable")
    manager._start_model_instance(mi)
    serve_manager.multiprocessing.Process.assert_not_called()
    serve_manager.delete_workload.assert_called_once()
    assert (mi.id in manager._assigned_ports) == cleanup_fails
    assert (mi.id in manager._model_instance_by_instance_id) == cleanup_fails
    updated = manager._update_model_instance.call_args.kwargs
    assert (
        updated["distributed_servers.subordinate_workers.0"].state
        == ModelInstanceStateEnum.ERROR
    )


@pytest.mark.parametrize(
    "failure_stage", ["backend", "process_start", "log_persistence"]
)
@pytest.mark.parametrize("backoff_count, delay", [(5, 160), (6, 300)])
def test_launch_failure_preserves_restart_backoff(
    manager_factory, monkeypatch, failure_stage, backoff_count, delay
):
    manager = manager_factory(1)
    mi = _instance()
    manager._restart_backoff_counts[mi.id] = backoff_count
    failing_call = {
        "backend": manager._inference_backend_manager.get_backend_by_name,
        "process_start": serve_manager.multiprocessing.Process.return_value.start,
        "log_persistence": manager._start_container_log_persistence,
    }[failure_stage]
    failing_call.side_effect = RuntimeError("launch failed")

    manager._start_model_instance(mi)

    assert manager._restart_backoff_counts[mi.id] == backoff_count
    assert not manager._assigned_ports
    assert not manager._provisioning_processes
    assert (
        manager._update_model_instance.call_args.kwargs["state"]
        == ModelInstanceStateEnum.ERROR
    )

    mi.state = ModelInstanceStateEnum.ERROR
    mi.last_restart_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = MagicMock()
    monkeypatch.setattr(serve_manager, "datetime", clock)
    manager._update_model_instance.reset_mock()
    clock.now.return_value = mi.last_restart_time + timedelta(seconds=delay - 1)
    manager._restart_error_model_instance(mi)
    manager._update_model_instance.assert_not_called()

    clock.now.return_value = mi.last_restart_time + timedelta(seconds=delay)
    manager._restart_error_model_instance(mi)
    assert manager._restart_backoff_counts[mi.id] == backoff_count + 1
    assert (
        manager._update_model_instance.call_args.kwargs["state"]
        == ModelInstanceStateEnum.SCHEDULED
    )


@pytest.mark.parametrize("worker_id", [1, 2])
def test_launch_is_not_blocked_by_instance_lookup_failure(manager_factory, worker_id):
    manager = manager_factory(worker_id)
    mi = _instance()
    if worker_id == 1:
        mi.state = ModelInstanceStateEnum.SCHEDULED
    manager._clientset.model_instances.get.side_effect = ConnectionError("offline")
    manager._dispatch_model_instance_event(Event(type=EventType.UPDATED, data=mi))
    manager._clientset.model_instances.get.assert_not_called()
    serve_manager.multiprocessing.Process.return_value.start.assert_called_once()
    assert len(manager._assigned_ports[mi.id]) == 14
    update = manager._update_model_instance.call_args.kwargs
    if worker_id == 1:
        assert update["state"] == ModelInstanceStateEnum.INITIALIZING
    else:
        assert (
            update["distributed_servers.subordinate_workers.0"].state
            == ModelInstanceStateEnum.RUNNING
        )


@pytest.mark.parametrize(
    "worker_id, delegated, tracked",
    [(2, False, True), (3, False, False), (2, True, False)],
)
def test_only_participating_followers_are_tracked_while_waiting(
    manager_factory, worker_id, delegated, tracked
):
    manager = manager_factory(worker_id)
    mi = _instance()
    mi.state = ModelInstanceStateEnum.DOWNLOADING
    if delegated:
        mi.distributed_servers.mode = DistributedServerCoordinateModeEnum.DELEGATED
    manager._dispatch_model_instance_event(Event(type=EventType.UPDATED, data=mi))
    assert (mi.id in manager._model_instance_by_instance_id) == tracked
    serve_manager.multiprocessing.Process.assert_not_called()


@pytest.mark.parametrize("worker_id", [1, 2])
def test_inference_health_and_proxy_only_use_primary(manager_factory, worker_id):
    manager = manager_factory(worker_id)
    mi = _instance()
    manager._model_instance_by_instance_id[mi.id] = mi
    manager._probe_running_instance = MagicMock(return_value=True)
    manager.sync_model_instances_inference_health()
    assert manager._probe_running_instance.call_count == (1 if worker_id == 1 else 0)
    assert manager.get_instance_port_by_model_instance_id(mi.id) == (
        mi.port if worker_id == 1 else None
    )
