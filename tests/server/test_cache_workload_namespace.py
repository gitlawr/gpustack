from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.schemas.clusters import ClusterProvider
from gpustack.schemas.principals import PrincipalType
from gpustack.server.cache_services import cache_workload_namespace


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,name,identifier",
    [
        (PrincipalType.ORG, "acme", "acme"),
        (PrincipalType.USER, "alice@example.com", "user-42"),
    ],
)
async def test_kubernetes_namespace_uses_service_owner(
    monkeypatch, kind, name, identifier
):
    monkeypatch.setattr(
        "gpustack.schemas.clusters.Cluster.one_by_id",
        AsyncMock(
            return_value=SimpleNamespace(
                id=1,
                provider=ClusterProvider.Kubernetes,
                registration_token="test-token",
            )
        ),
    )
    monkeypatch.setattr(
        "gpustack.schemas.principals.Principal.one_by_id",
        AsyncMock(return_value=SimpleNamespace(id=42, name=name, kind=kind)),
    )
    monkeypatch.setattr(
        "gpustack.config.config.get_global_config",
        lambda: SimpleNamespace(get_api_port=lambda: 80),
    )
    ops = AsyncMock()
    ops.org_namespace = f"gpustack-{identifier}"
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = ops
    monkeypatch.setattr("gpustack.gpu_instances.cluster_apis.ClusterOps", factory)
    namespace = await cache_workload_namespace(
        MagicMock(), SimpleNamespace(cluster_id=1, owner_principal_id=42)
    )
    assert namespace == f"gpustack-{identifier}"
    assert factory.call_args.kwargs["cluster_owner_principal_identifier"] == identifier
    ops.ensure_org_namespace.assert_awaited_once()


@pytest.mark.asyncio
async def test_docker_retains_runtime_default(monkeypatch):
    monkeypatch.setattr(
        "gpustack.schemas.clusters.Cluster.one_by_id",
        AsyncMock(return_value=SimpleNamespace(provider=ClusterProvider.Docker)),
    )
    assert (
        await cache_workload_namespace(MagicMock(), SimpleNamespace(cluster_id=1))
        is None
    )


@pytest.mark.asyncio
async def test_missing_owner_cannot_fall_back_to_system_namespace(monkeypatch):
    monkeypatch.setattr(
        "gpustack.schemas.clusters.Cluster.one_by_id",
        AsyncMock(return_value=SimpleNamespace(provider=ClusterProvider.Kubernetes)),
    )
    monkeypatch.setattr(
        "gpustack.schemas.principals.Principal.one_by_id", AsyncMock(return_value=None)
    )
    with pytest.raises(ValueError, match="owner not found"):
        await cache_workload_namespace(
            MagicMock(), SimpleNamespace(cluster_id=1, owner_principal_id=42)
        )
