"""Owner-scoped streams share the same tenant and system-account boundary."""

from types import SimpleNamespace

import pytest

from gpustack.api.tenant import TenantContext, tenant_stream_filter
from gpustack.schemas.benchmark import Benchmark
from gpustack.schemas.gpu_devices import GPUDevice
from gpustack.schemas.gpu_instances import GPUInstance
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.workers import Worker


@pytest.mark.parametrize(
    "model", [Worker, GPUDevice, ModelFile, Benchmark, GPUInstance]
)
@pytest.mark.parametrize(
    "kind,admin,principal_id,cluster_id,row,expected",
    [
        ("user", False, 10, None, {"owner_principal_id": 10}, True),
        ("user", False, 10, None, {"owner_principal_id": 20}, False),
        ("user", False, 10, None, {"owner_principal_id": None}, False),
        ("user", False, 10, None, {}, False),
        ("user", False, None, None, {}, False),
        ("user", True, None, None, {"owner_principal_id": 20}, True),
        ("user", True, 10, None, {"owner_principal_id": 20}, False),
        ("system", False, None, None, {"cluster_id": 30}, True),
        ("system", False, None, 30, {"cluster_id": 30}, True),
        ("system", False, None, 30, {"cluster_id": 40}, False),
        ("system", False, None, 30, {"cluster_id": None}, True),
        ("system", False, None, 30, {}, False),
    ],
)
def test_owner_stream_visibility(
    model, kind, admin, principal_id, cluster_id, row, expected
):
    ctx = TenantContext(
        user=SimpleNamespace(kind=PrincipalType(kind)),
        is_platform_admin=admin,
        current_principal_id=principal_id,
        org_role=None,
        scoped_cluster_id=cluster_id,
    )
    visible = tenant_stream_filter(ctx, model)
    assert visible(SimpleNamespace(**row)) is expected
