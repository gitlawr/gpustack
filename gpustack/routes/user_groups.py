"""UserGroup management — Org admin+ or platform admin."""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.organizations import Organization, OrganizationMembership
from gpustack.schemas.principals import OrgRole
from gpustack.schemas.user_groups import (
    UserGroup,
    UserGroupCreate,
    UserGroupListParams,
    UserGroupMembership,
    UserGroupMembershipPublic,
    UserGroupPublic,
    UserGroupUpdate,
    UserGroupsPublic,
)
from gpustack.schemas.users import User
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


class GroupMembershipCreate(BaseModel):
    user_id: int


def _can_manage_groups(ctx, org_id: int) -> bool:
    if ctx.is_platform_admin:
        return True
    if ctx.current_org_id != org_id:
        return False
    return ctx.org_role == OrgRole.ADMIN


async def _load_org(session, org_id: int) -> Organization:
    org = await Organization.one_by_id(session, org_id)
    if not org or org.deleted_at is not None:
        raise NotFoundException(message="Organization not found")
    return org


async def _load_group(session, org_id: int, group_id: int) -> UserGroup:
    group = await UserGroup.one_by_id(session, group_id)
    if not group or group.deleted_at is not None or group.organization_id != org_id:
        raise NotFoundException(message="Group not found")
    return group


# ---- groups ----------------------------------------------------------------


@router.get("/organizations/{org_id}/groups", response_model=UserGroupsPublic)
async def list_groups(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    params: UserGroupListParams = Depends(),
    search: Optional[str] = None,
):
    await _load_org(session, org_id)
    if not ctx.is_platform_admin and ctx.current_org_id != org_id:
        raise ForbiddenException(message="Not a member of this organization")

    fuzzy_fields = {"name": search} if search else {}
    return await UserGroup.paginated_by_query(
        session=session,
        fields={"organization_id": org_id, "deleted_at": None},
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
        order_by=params.order_by,
    )


@router.post("/organizations/{org_id}/groups", response_model=UserGroupPublic)
async def create_group(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    body: UserGroupCreate,
):
    await _load_org(session, org_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    existing = await UserGroup.one_by_fields(
        session,
        {"organization_id": org_id, "name": body.name, "deleted_at": None},
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Group '{body.name}' already exists in this organization"
        )

    try:
        group = UserGroup(
            organization_id=org_id,
            name=body.name,
            description=body.description,
        )
        return await UserGroup.create(session, group)
    except Exception as e:
        raise InvalidException(message=f"Failed to create group: {e}")


@router.put("/organizations/{org_id}/groups/{group_id}", response_model=UserGroupPublic)
async def update_group(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    group_id: int,
    body: UserGroupUpdate,
):
    group = await _load_group(session, org_id, group_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    try:
        await group.update(session, body.model_dump(exclude_unset=True))
    except Exception as e:
        raise InvalidException(message=f"Failed to update group: {e}")
    return group


@router.delete("/organizations/{org_id}/groups/{group_id}")
async def delete_group(
    session: SessionDep, ctx: TenantContextDep, org_id: int, group_id: int
):
    group = await _load_group(session, org_id, group_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    try:
        await group.delete(session)
    except Exception as e:
        raise InvalidException(message=f"Failed to delete group: {e}")


# ---- group members ---------------------------------------------------------


@router.get(
    "/organizations/{org_id}/groups/{group_id}/members",
    response_model=List[UserGroupMembershipPublic],
)
async def list_group_members(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    group_id: int,
):
    await _load_group(session, org_id, group_id)
    if not ctx.is_platform_admin and ctx.current_org_id != org_id:
        raise ForbiddenException(message="Not a member of this organization")

    stmt = select(UserGroupMembership).where(UserGroupMembership.group_id == group_id)
    rows = list((await session.exec(stmt)).all())
    user_ids = {r.user_id for r in rows}
    user_by_id: dict[int, User] = {}
    if user_ids:
        result = await session.exec(select(User).where(User.id.in_(user_ids)))
        user_by_id = {u.id: u for u in result.all()}
    out: List[UserGroupMembershipPublic] = []
    for r in rows:
        u = user_by_id.get(r.user_id)
        out.append(
            UserGroupMembershipPublic(
                user_id=r.user_id,
                group_id=r.group_id,
                created_at=r.created_at,
                username=getattr(u, "username", None),
                full_name=getattr(u, "full_name", None),
            )
        )
    return out


@router.post(
    "/organizations/{org_id}/groups/{group_id}/members",
    response_model=UserGroupMembershipPublic,
)
async def add_group_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    group_id: int,
    body: GroupMembershipCreate,
):
    await _load_group(session, org_id, group_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    # User must be a member of the group's org first.
    membership_stmt = select(OrganizationMembership).where(
        OrganizationMembership.organization_id == org_id,
        OrganizationMembership.user_id == body.user_id,
    )
    org_membership = (await session.exec(membership_stmt)).first()
    if not org_membership:
        raise InvalidException(
            message=f"User {body.user_id} is not a member of organization {org_id}"
        )

    existing_stmt = select(UserGroupMembership).where(
        UserGroupMembership.group_id == group_id,
        UserGroupMembership.user_id == body.user_id,
    )
    if (await session.exec(existing_stmt)).first() is not None:
        raise AlreadyExistsException(
            message=f"User {body.user_id} is already in group {group_id}"
        )

    try:
        link = UserGroupMembership(
            user_id=body.user_id,
            group_id=group_id,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(link)
        await session.commit()
        await session.refresh(link)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to add group member: {e}")
    return link


@router.delete("/organizations/{org_id}/groups/{group_id}/members/{user_id}")
async def remove_group_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    group_id: int,
    user_id: int,
):
    await _load_group(session, org_id, group_id)
    if not _can_manage_groups(ctx, org_id):
        raise ForbiddenException(message="Insufficient permission to manage groups")

    stmt = select(UserGroupMembership).where(
        UserGroupMembership.group_id == group_id,
        UserGroupMembership.user_id == user_id,
    )
    link = (await session.exec(stmt)).first()
    if not link:
        raise NotFoundException(message="Group membership not found")

    try:
        await session.delete(link)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to remove group member: {e}")
