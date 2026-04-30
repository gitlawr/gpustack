"""Organization management — platform admin only."""

from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.schemas.organizations import (
    Organization,
    OrganizationCreate,
    OrganizationListParams,
    OrganizationPublic,
    OrganizationUpdate,
    OrganizationsPublic,
)
from gpustack.server.deps import SessionDep

router = APIRouter()


@router.get("", response_model=OrganizationsPublic)
async def get_organizations(
    session: SessionDep,
    params: OrganizationListParams = Depends(),
    search: Optional[str] = None,
    include_personal: bool = False,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search, "slug": search}

    # Personal Orgs are auto-managed user namespaces; they shouldn't
    # appear in the admin Organizations CRUD page or in any "pick an
    # Org" picker by default. Callers that genuinely need them
    # (e.g. an internal audit tool) can opt in with include_personal.
    fields = {"deleted_at": None}
    if not include_personal:
        fields["is_personal"] = False

    if params.watch:
        return StreamingResponse(
            Organization.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    return await Organization.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
        order_by=params.order_by,
    )


@router.get("/{id}", response_model=OrganizationPublic)
async def get_organization(session: SessionDep, id: int):
    org = await Organization.one_by_id(session, id)
    if not org or org.deleted_at is not None:
        raise NotFoundException(message="Organization not found")
    return org


@router.post("", response_model=OrganizationPublic)
async def create_organization(session: SessionDep, org_in: OrganizationCreate):
    existing = await Organization.one_by_fields(
        session, {"slug": org_in.slug, "deleted_at": None}
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Organization with slug '{org_in.slug}' already exists"
        )

    try:
        to_create = Organization(
            name=org_in.name,
            slug=org_in.slug,
            description=org_in.description,
            billing_account_ref=org_in.billing_account_ref,
            is_platform=False,
        )
        return await Organization.create(session, to_create)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create organization: {e}"
        )


@router.put("/{id}", response_model=OrganizationPublic)
async def update_organization(session: SessionDep, id: int, org_in: OrganizationUpdate):
    org = await Organization.one_by_id(session, id)
    if not org or org.deleted_at is not None:
        raise NotFoundException(message="Organization not found")

    try:
        await org.update(session, org_in.model_dump(exclude_unset=True))
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update organization: {e}"
        )
    return org


@router.delete("/{id}")
async def delete_organization(session: SessionDep, id: int):
    org = await Organization.one_by_id(session, id)
    if not org or org.deleted_at is not None:
        raise NotFoundException(message="Organization not found")
    if org.is_platform:
        raise ConflictException(
            message="The built-in platform organization cannot be deleted"
        )

    # Block delete when any tenant-owned resource still references this org.
    # We deliberately keep this conservative: even though FK CASCADE would
    # clean up dependents, that would silently destroy users' resources.
    blockers = await _has_resources(session, id)
    if blockers:
        raise ConflictException(
            message=(
                "Organization still owns resources: "
                f"{', '.join(blockers)}. Remove them before deleting."
            )
        )

    try:
        await org.delete(session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete organization: {e}"
        )


async def _has_resources(session, org_id: int) -> list[str]:
    """Return resource types that still belong to this org."""
    from gpustack.schemas.api_keys import ApiKey
    from gpustack.schemas.models import Model, ModelInstance
    from gpustack.schemas.model_routes import ModelRoute
    from gpustack.schemas.tenant_namespaces import TenantNamespace

    blockers: list[str] = []
    for resource_cls, label in (
        (ApiKey, "api_keys"),
        (Model, "models"),
        (ModelInstance, "model_instances"),
        (ModelRoute, "model_routes"),
        (TenantNamespace, "tenant_namespaces"),
    ):
        stmt = (
            select(resource_cls.id)
            .where(
                resource_cls.organization_id == org_id,
                resource_cls.deleted_at.is_(None),
            )
            .limit(1)
        )
        if (await session.exec(stmt)).first() is not None:
            blockers.append(label)
    return blockers
