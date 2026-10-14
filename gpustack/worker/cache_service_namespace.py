"""Namespace-local resources required by mirrored Kubernetes cache workloads."""

from pathlib import Path

from kubernetes import client, config
from gpustack_runtime import envs as runtime_envs
from gpustack_runtime.deployer import KubernetesWorkloadPlan, WorkloadPlan

from gpustack.utils.runtime import is_cache_service_workload


def prepare_cache_image_pull_secrets(workload: WorkloadPlan) -> None:
    """Copy only the worker's registry secrets into the cache Pod's namespace.

    Mirrored Pods reference the worker's imagePullSecrets by name, and Kubernetes
    resolves those references within the new Pod's namespace.
    """
    if (
        not isinstance(workload, KubernetesWorkloadPlan)
        or not is_cache_service_workload(workload)
        or not workload.namespace
        or not runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT
    ):
        return
    pod_name = runtime_envs.GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME
    if not pod_name:
        return
    try:
        source_namespace = (
            Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        source_namespace = "default"
    if source_namespace == workload.namespace:
        return

    configuration = client.Configuration()
    config.load_config(client_configuration=configuration)
    with client.ApiClient(configuration) as api_client:
        core = client.CoreV1Api(api_client)
        pod = core.read_namespaced_pod(pod_name, source_namespace)
        for ref in pod.spec.image_pull_secrets or []:
            secret = core.read_namespaced_secret(ref.name, source_namespace)
            if secret.type not in (
                "kubernetes.io/dockerconfigjson",
                "kubernetes.io/dockercfg",
            ):
                raise ValueError(
                    "Worker image pull secret is not a registry credential"
                )
            body = client.V1Secret(
                metadata=client.V1ObjectMeta(
                    name=ref.name, namespace=workload.namespace
                ),
                type=secret.type,
                data=secret.data,
            )
            try:
                core.create_namespaced_secret(workload.namespace, body)
            except client.ApiException as e:
                if e.status != 409:
                    raise
                core.patch_namespaced_secret(ref.name, workload.namespace, body)
