from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kubernetes import client
from gpustack_runtime.deployer import KubernetesWorkloadPlan

from gpustack.worker.cache_service_namespace import prepare_cache_image_pull_secrets


@pytest.mark.parametrize("exists", [True, False])
def test_mirrored_registry_credentials_follow_cache_namespace(monkeypatch, exists):
    monkeypatch.setenv("GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT", "true")
    monkeypatch.setenv("GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME", "worker-1")
    monkeypatch.setattr(
        "gpustack.worker.cache_service_namespace.Path.read_text",
        lambda *a, **kw: "gpustack-system",
    )
    monkeypatch.setattr(
        "gpustack.worker.cache_service_namespace.config.load_config", MagicMock()
    )
    core = MagicMock()
    core.read_namespaced_pod.return_value = SimpleNamespace(
        spec=SimpleNamespace(image_pull_secrets=[SimpleNamespace(name="registry")])
    )
    core.read_namespaced_secret.return_value = client.V1Secret(
        type="kubernetes.io/dockerconfigjson", data={".dockerconfigjson": "test"}
    )
    if exists:
        core.create_namespaced_secret.side_effect = client.ApiException(status=409)
    monkeypatch.setattr(
        "gpustack.worker.cache_service_namespace.client.CoreV1Api", lambda *_: core
    )
    plan = KubernetesWorkloadPlan(
        namespace="gpustack-acme", labels={"type": "cache-service"}
    )
    prepare_cache_image_pull_secrets(plan)
    core.read_namespaced_secret.assert_called_once_with("registry", "gpustack-system")
    namespace, body = core.create_namespaced_secret.call_args.args
    assert namespace == body.metadata.namespace == "gpustack-acme"
    assert body.metadata.name == "registry"
    assert body.data == {".dockerconfigjson": "test"}
    assert core.patch_namespaced_secret.called is exists


def test_same_namespace_needs_no_secret_copy(monkeypatch):
    monkeypatch.setenv("GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT", "true")
    monkeypatch.setenv("GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME", "worker-1")
    monkeypatch.setattr(
        "gpustack.worker.cache_service_namespace.Path.read_text",
        lambda *a, **kw: "gpustack-acme",
    )
    configure = MagicMock()
    monkeypatch.setattr(
        "gpustack.worker.cache_service_namespace.config.load_config", configure
    )
    prepare_cache_image_pull_secrets(
        KubernetesWorkloadPlan(
            namespace="gpustack-acme", labels={"type": "cache-service"}
        )
    )
    configure.assert_not_called()
