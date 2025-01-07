import math
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, Query

from gpustack.policies.utils import get_worker_allocatable_resource
from gpustack.routes.models import NotFoundException, Worker
from gpustack.scheduler.calculator import rough_estimate_model_size
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.models import BackendEnum
from gpustack.server.catalog import (
    ModelSet,
    ModelSetPublic,
    ModelSpec,
    get_model_catalog,
    get_model_set_specs,
)
from gpustack.server.db import get_engine
from gpustack.server.deps import ListParamsDep, SessionDep

router = APIRouter()


@router.get("", response_model=PaginatedList[ModelSetPublic])
async def get_model_sets(
    params: ListParamsDep,
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
    model_catalog: List[ModelSet] = Depends(get_model_catalog),
):
    if search:
        model_catalog = [
            model for model in model_catalog if search.lower() in model.name.lower()
        ]

    if categories:
        model_catalog = [
            model
            for model in model_catalog
            if model.categories is not None
            and any(category in model.categories for category in categories)
        ]

    count = len(model_catalog)
    total_page = math.ceil(count / params.perPage)

    start_index = (params.page - 1) * params.perPage
    end_index = start_index + params.perPage

    paginated_items = model_catalog[start_index:end_index]

    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return PaginatedList[ModelSetPublic](items=paginated_items, pagination=pagination)


@router.get("/{id}/specs", response_model=PaginatedList[ModelSpec])
async def get_model_specs(
    id: int,
    params: ListParamsDep,
    session: SessionDep,
    get_model_set_specs: Dict[int, List[ModelSpec]] = Depends(get_model_set_specs),
):

    specs = get_model_set_specs.get(id, [])
    if not specs:
        raise NotFoundException(message="Model set not found")

    await set_model_specs_compatibility(specs, session)

    count = len(specs)
    total_page = math.ceil(count / params.perPage)
    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return PaginatedList[ModelSpec](items=specs, pagination=pagination)


async def set_model_specs_compatibility(
    model_specs: List[ModelSpec], session: SessionDep
):
    workers = await Worker.all(session)
    can_run_vllm = False
    engine = get_engine()
    total_allocatable_ram = 0
    total_allocatable_vram = 0
    for worker in workers:
        if (
            worker.labels
            and worker.labels.get("os") == "linux"
            and worker.labels.get("arch") == "amd64"
        ):
            can_run_vllm = True

        allocatable = await get_worker_allocatable_resource(engine, worker)
        total_allocatable_ram += allocatable.ram
        total_allocatable_vram += sum(allocatable.vram.values())

    for model_spec in model_specs:
        set_model_spec_compatibility(
            model_spec, total_allocatable_vram, total_allocatable_ram, can_run_vllm
        )


def set_model_spec_compatibility(
    model_spec: ModelSpec,
    total_allocatable_vram: int,
    total_allocatable_ram: int,
    can_run_vllm: bool,
):
    if model_spec.backend == BackendEnum.VLLM and not can_run_vllm:
        model_spec.compatibility = False
        model_spec.compatibility_message = (
            "vLLM backend requires an amd64 Linux worker, but none is available."
        )
        return

    if model_spec.size and model_spec.quantization:
        model_size = rough_estimate_model_size(model_spec.size, model_spec.quantization)
        if model_size > total_allocatable_ram + total_allocatable_vram:
            model_size_gb = model_size / (1024**3)
            total_allocatable_ram_gb = total_allocatable_ram / (1024**3)
            total_allocatable_vram_gb = total_allocatable_vram / (1024**3)
            model_spec.compatibility = False
            model_spec.compatibility_message = f"The model size is too large for the current setup. Estimated size: {model_size_gb:.1f} GB (roughly). Allocatable RAM: {total_allocatable_ram_gb:.1f} GB, VRAM: {total_allocatable_vram_gb:.1f} GB."
            return

    model_spec.compatibility = True
