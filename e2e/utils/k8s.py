"""Kubernetes utilities for E2E testing."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class K8sError(Exception):
    """Kubernetes operation error."""

    pass


class K8sManager:
    """Kubernetes manager."""

    def __init__(
        self,
        kubeconfig: str = None,
        namespace: str = "gpustack",
        context: str = None,
    ):
        """
        Initialize Kubernetes manager.

        Args:
            kubeconfig: Path to kubeconfig file
            namespace: Kubernetes namespace
            context: kubectl context
        """
        self.kubeconfig = kubeconfig or os.environ.get(
            "KUBECONFIG", os.path.expanduser("~/.kube/config")
        )
        self.namespace = namespace
        self.context = context

        self._client = None
        self._apps_v1 = None
        self._core_v1 = None

    def _init_client(self):
        """Initialize Kubernetes client."""
        if self._client is not None:
            return

        try:
            from kubernetes import client, config

            if self.kubeconfig and Path(self.kubeconfig).exists():
                config.load_kube_config(
                    config_file=self.kubeconfig,
                    context=self.context,
                )
            else:
                # Try in-cluster config
                config.load_incluster_config()

            self._client = client
            self._apps_v1 = client.AppsV1Api()
            self._core_v1 = client.CoreV1Api()

        except ImportError:
            raise K8sError("kubernetes package not installed")
        except Exception as e:
            raise K8sError(f"Failed to initialize Kubernetes client: {e}")

    def create_namespace(self) -> None:
        """Create namespace."""
        self._init_client()

        try:
            self._core_v1.read_namespace(self.namespace)
            logger.debug(f"Namespace {self.namespace} already exists")
        except self._client.rest.ApiException as e:
            if e.status == 404:
                logger.info(f"Creating namespace: {self.namespace}")
                body = self._client.V1Namespace(
                    metadata=self._client.V1ObjectMeta(name=self.namespace)
                )
                self._core_v1.create_namespace(body)
            else:
                raise

    def delete_namespace(self) -> None:
        """Delete namespace."""
        self._init_client()

        try:
            logger.info(f"Deleting namespace: {self.namespace}")
            self._core_v1.delete_namespace(self.namespace)
        except self._client.rest.ApiException as e:
            if e.status != 404:
                raise

    def apply_manifest(self, manifest: str) -> None:
        """
        Apply Kubernetes manifest.

        Args:
            manifest: YAML manifest content
        """
        self._init_client()
        from kubernetes import utils

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(manifest)
            temp_file = f.name

        try:
            logger.info("Applying Kubernetes manifest")
            utils.create_from_yaml(
                self._client.ApiClient(),
                temp_file,
                namespace=self.namespace,
            )
        finally:
            os.unlink(temp_file)

    def get_pods(self, label_selector: str = None) -> list:
        """
        Get pod list.

        Args:
            label_selector: Label selector

        Returns:
            Pod list
        """
        self._init_client()

        pods = self._core_v1.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=label_selector,
        )
        return pods.items

    def get_pod_logs(
        self,
        pod_name: str,
        container: str = None,
        tail_lines: int = 100,
    ) -> str:
        """
        Get pod logs.

        Args:
            pod_name: Pod name
            container: Container name
            tail_lines: Number of log lines

        Returns:
            Log content
        """
        self._init_client()

        return self._core_v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=self.namespace,
            container=container,
            tail_lines=tail_lines,
        )

    def wait_for_deployment_ready(
        self,
        deployment_name: str,
        timeout: int = 300,
    ) -> bool:
        """
        Wait for deployment to be ready.

        Args:
            deployment_name: Deployment name
            timeout: Timeout in seconds

        Returns:
            Whether deployment is ready
        """
        self._init_client()
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                deployment = self._apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.namespace,
                )
                status = deployment.status
                if status.ready_replicas and status.ready_replicas == status.replicas:
                    logger.info(f"Deployment {deployment_name} is ready")
                    return True
            except Exception as e:
                logger.debug(f"Error checking deployment: {e}")

            time.sleep(5)

        raise K8sError(f"Timeout waiting for deployment {deployment_name} to be ready")

    def delete_deployment(self, deployment_name: str) -> None:
        """Delete deployment."""
        self._init_client()

        try:
            self._apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
            )
            logger.info(f"Deleted deployment: {deployment_name}")
        except self._client.rest.ApiException as e:
            if e.status != 404:
                raise

    def get_services(self, label_selector: str = None) -> list:
        """Get service list."""
        self._init_client()

        services = self._core_v1.list_namespaced_service(
            namespace=self.namespace,
            label_selector=label_selector,
        )
        return services.items

    def get_service_endpoint(self, service_name: str) -> Optional[str]:
        """
        Get service endpoint URL.

        Args:
            service_name: Service name

        Returns:
            Endpoint URL
        """
        self._init_client()

        try:
            service = self._core_v1.read_namespaced_service(
                name=service_name,
                namespace=self.namespace,
            )

            if service.spec.type == "LoadBalancer":
                ingress = service.status.load_balancer.ingress
                if ingress:
                    host = ingress[0].ip or ingress[0].hostname
                    port = service.spec.ports[0].port
                    return f"http://{host}:{port}"

            elif service.spec.type == "NodePort":
                node_port = service.spec.ports[0].node_port
                # Get any node IP
                nodes = self._core_v1.list_node()
                if nodes.items:
                    for addr in nodes.items[0].status.addresses:
                        if addr.type == "InternalIP":
                            return f"http://{addr.address}:{node_port}"

            elif service.spec.type == "ClusterIP":
                cluster_ip = service.spec.cluster_ip
                port = service.spec.ports[0].port
                return f"http://{cluster_ip}:{port}"

        except Exception as e:
            logger.warning(f"Failed to get service endpoint: {e}")

        return None

    def cleanup_all(self) -> None:
        """Clean up all test resources."""
        logger.info(f"Cleaning up all resources in namespace: {self.namespace}")
        self.delete_namespace()
