"""The instances of a managed cache service.

A cache server is a workload row; this presents those rows in the shape the
API has always had. Read-only: the worker writes its state back through
/workloads, which is where the row lives.
"""

from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import NotFoundException
from gpustack.api.tenant import (
    bypass_tenant_filter,
    cluster_scoped_system,
    scoped_cluster_row_visible,
    tenant_list_conditions,
)
from gpustack.schemas.cache_service_workloads import instance_public_from_workload
from gpustack.schemas.cache_services import (
    CacheService,
    CacheServiceInstancePublic,
    CacheServiceInstancesPublic,
    CacheServiceStateEnum,
)
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.workloads import (
    Workload,
    WorkloadOwnerKindEnum,
    WorkloadStateEnum,
)
from gpustack.server.db import async_session
from gpustack.server.deps import ListParamsDep, SessionDep, TenantContextDep

router = APIRouter()


@router.get("", response_model=CacheServiceInstancesPublic)
async def get_cache_service_instances(
    ctx: TenantContextDep,
    params: ListParamsDep,
    id: Optional[int] = None,
    cache_service_id: Optional[int] = None,
    worker_id: Optional[int] = None,
    state: Optional[CacheServiceStateEnum] = None,
):
    # owner_kind is not optional: the table also holds the workloads of model
    # instances and benchmarks.
    fields = {"owner_kind": WorkloadOwnerKindEnum.CACHE_SERVICE}
    if id:
        fields["id"] = id

    if cache_service_id:
        fields["owner_id"] = cache_service_id

    if worker_id:
        fields["worker_id"] = worker_id

    if state:
        # The two spellings coincide: a cache server never succeeds, which is
        # the one state a workload has and a cache service does not.
        fields["state"] = WorkloadStateEnum(state.value)

    if params.watch:
        # Cluster-bound service accounts (worker / cluster bootstrap) only
        # stream instances of their own cluster, via the denormalized
        # cluster_id.
        if cluster_scoped_system(ctx):

            def filter_func(data):
                return scoped_cluster_row_visible(ctx, data)

        elif ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):
            principal_id = ctx.current_principal_id

            # A row carries its owner, which the instance table did not: its
            # visibility had to be derived from the parent service.
            def filter_func(data):
                return getattr(data, "owner_principal_id", None) == principal_id

        else:
            filter_func = None

        async def as_instance(event):
            # The stream carries rows; this API has always carried instances.
            if event.data is not None:
                event.data = instance_public_from_workload(event.data)

        return StreamingResponse(
            Workload.streaming(
                fields=fields,
                filter_func=filter_func,
                event_transform=as_instance,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        extra_conditions = tenant_list_conditions(ctx, Workload)
        if ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):
            extra_conditions.append(
                Workload.owner_principal_id == ctx.current_principal_id
            )
        page = await Workload.paginated_by_query(
            session=session,
            fields=fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
        )
        return PaginatedList[CacheServiceInstancePublic](
            items=[instance_public_from_workload(instance) for instance in page.items],
            pagination=Pagination(**page.pagination.model_dump()),
        )


@router.get("/{id}", response_model=CacheServiceInstancePublic)
async def get_cache_service_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    """One instance by ID."""
    instance = await Workload.one_by_id(session, id)
    if instance is None or instance.owner_kind != WorkloadOwnerKindEnum.CACHE_SERVICE:
        raise NotFoundException(message="Cache service instance not found")

    # Visibility mirrors the list endpoint: cluster-bound service accounts
    # see their own cluster's rows, and other tenant-scoped callers see the
    # instances of the services they own.
    if cluster_scoped_system(ctx):
        if not scoped_cluster_row_visible(ctx, instance):
            raise NotFoundException(message="Cache service instance not found")
    elif ctx.current_principal_id is not None and not bypass_tenant_filter(ctx):
        service = await CacheService.one_by_id(session, instance.owner_id)
        if (
            service is None
            or service.deleted_at is not None
            or service.owner_principal_id != ctx.current_principal_id
        ):
            raise NotFoundException(message="Cache service instance not found")

    return instance_public_from_workload(instance)
