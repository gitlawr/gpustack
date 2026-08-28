from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    ForbiddenException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.api.tenant import (
    bypass_tenant_filter,
    cluster_scoped_system,
    scoped_cluster_row_visible,
    tenant_list_conditions,
)
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.workloads import (
    Workload,
    WorkloadOwnerKindEnum,
    WorkloadPublic,
    WorkloadStateEnum,
    WorkloadUpdate,
    WorkloadsPublic,
)
from gpustack.server.db import async_session
from gpustack.server.deps import ListParamsDep, SessionDep, TenantContextDep

router = APIRouter()


@router.get("", response_model=WorkloadsPublic)
async def get_workloads(
    ctx: TenantContextDep,
    params: ListParamsDep,
    id: Optional[int] = None,
    owner_kind: Optional[WorkloadOwnerKindEnum] = None,
    owner_id: Optional[int] = None,
    worker_id: Optional[int] = None,
    group_key: Optional[str] = None,
    state: Optional[WorkloadStateEnum] = None,
):
    """
    Workers watch this filtered to their own ``worker_id``; controllers read it
    filtered by owner.
    """
    fields = {}
    if id:
        fields["id"] = id
    if owner_kind:
        fields["owner_kind"] = owner_kind
    if owner_id:
        fields["owner_id"] = owner_id
    if worker_id:
        fields["worker_id"] = worker_id
    if group_key:
        fields["group_key"] = group_key
    if state:
        fields["state"] = state

    if params.watch:
        # Cluster-bound service accounts (worker / cluster bootstrap) stream
        # their own cluster's rows only, via the denormalized cluster_id.
        # Everyone else is filtered by ownership, which a workload carries
        # itself -- copied from its owner at creation, so no join to a table
        # that differs per owner_kind.
        if cluster_scoped_system(ctx):

            def filter_func(data):
                return scoped_cluster_row_visible(ctx, data)

        elif ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):

            def filter_func(data):
                return (
                    getattr(data, "owner_principal_id", None)
                    == ctx.current_principal_id
                )

        else:
            filter_func = None

        return StreamingResponse(
            Workload.streaming(fields=fields, filter_func=filter_func),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await Workload.paginated_by_query(
            session=session,
            fields=fields,
            extra_conditions=tenant_list_conditions(ctx, Workload),
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=WorkloadPublic)
async def get_workload(session: SessionDep, ctx: TenantContextDep, id: int):
    """
    One workload by ID. Workers read one back through this endpoint whenever
    their watch-backed cache is not authoritative (during a stream reconnect,
    for one), so a state write-back never depends on a warm cache.
    """
    workload = await Workload.one_by_id(session, id)
    if workload is None:
        raise NotFoundException(message="Workload not found")

    if not _visible(ctx, workload):
        raise NotFoundException(message="Workload not found")

    return workload


@router.put("/{id}", response_model=WorkloadPublic)
async def update_workload(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    workload_in: WorkloadUpdate,
):
    """
    Worker write-back of execution state: ports, state, health, restart
    bookkeeping. Users act on the owning resource instead; workloads are not
    part of the user-facing API.
    """
    if ctx.user is None or ctx.user.kind != PrincipalType.SYSTEM:
        raise ForbiddenException(message="Only system principals may update workloads")

    workload = await Workload.one_by_id(session, id)
    if workload is None:
        raise NotFoundException(message="Workload not found")

    # Cluster-bound service accounts write their own cluster's rows only,
    # mirroring the read endpoints.
    if cluster_scoped_system(ctx) and not scoped_cluster_row_visible(ctx, workload):
        raise NotFoundException(message="Workload not found")

    try:
        await workload.update(session, workload_in)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update workload: {e}")

    return workload


def _visible(ctx, workload: Workload) -> bool:
    if cluster_scoped_system(ctx):
        return scoped_cluster_row_visible(ctx, workload)
    if ctx.current_principal_id is None or bypass_tenant_filter(ctx):
        return True
    return workload.owner_principal_id == ctx.current_principal_id
