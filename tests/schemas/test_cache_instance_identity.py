import re
import pytest
from gpustack.schemas.cache_services import CacheServiceInstance, cache_instance_name


@pytest.mark.parametrize(
    "name,component",
    [
        ("test-server", ""),
        ("Cache Service", "store"),
        ("X" * 100, "master"),
        ("缓存", ""),
    ],
)
def test_instance_and_pod_share_a_dns_safe_name(name, component):
    identity = cache_instance_name(name, component, "ddo69")
    assert len(identity) <= 63
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", identity)
    instance = CacheServiceInstance(
        id=1,
        name=identity,
        cache_service_id=1,
        worker_id=1,
        cluster_id=1,
        workload_namespace="gpustack-acme",
    )
    metadata = instance.get_deployment_metadata()
    assert metadata.name == instance.name
    assert metadata.namespace == "gpustack-acme"
    assert metadata.labels["cache-service-instance-id"] == "1"
