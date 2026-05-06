"""Organization membership management.

These routes are nested under /organizations/{org_id}/members. Both the
platform admin and any Org admin can manage memberships. The last admin
of an Org cannot be demoted or removed — that would leave the Org
without anyone able to manage members or infra.
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.organizations import (
    Organization,
    OrganizationMembership,
    OrganizationMembershipPublic,
)
from gpustack.schemas.principals import OrgRole
from gpustack.schemas.users import User
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


class MembershipCreate(BaseModel):
    user_id: int
    role: OrgRole = OrgRole.USER


class MembershipUpdate(BaseModel):
    role: OrgRole


def _can_manage(ctx) -> bool:
    """Platform admin or any Org admin can manage memberships."""
    if ctx.is_platform_admin:
        return True
    return ctx.org_role == OrgRole.ADMIN


async def _load_org(session, org_id: int) -> Organization:
    org = await Organization.one_by_id(session, org_id)
    if not org or org.deleted_at is not None:
        raise NotFoundException(message="Organization not found")
    return org


async def _list_memberships(session, org_id: int) -> List[OrganizationMembership]:
    stmt = select(OrganizationMembership).where(
        OrganizationMembership.organization_id == org_id
    )
    return list((await session.exec(stmt)).all())


async def _find_membership(
    session, org_id: int, user_id: int
) -> Optional[OrganizationMembership]:
    stmt = select(OrganizationMembership).where(
        OrganizationMembership.organization_id == org_id,
        OrganizationMembership.user_id == user_id,
    )
    return (await session.exec(stmt)).first()


async def _has_other_admin(session, org_id: int, exclude_user_id: int) -> bool:
    stmt = select(OrganizationMembership.user_id).where(
        OrganizationMembership.organization_id == org_id,
        OrganizationMembership.role == OrgRole.ADMIN,
        OrganizationMembership.user_id != exclude_user_id,
    )
    return (await session.exec(stmt)).first() is not None


@router.get(
    "/organizations/{org_id}/members",
    response_model=List[OrganizationMembershipPublic],
)
async def list_org_members(session: SessionDep, ctx: TenantContextDep, org_id: int):
    await _load_org(session, org_id)
    # Either platform admin, or member of this org.
    if not ctx.is_platform_admin and ctx.current_org_id != org_id:
        raise ForbiddenException(message="Not a member of this organization")
    return await _list_memberships(session, org_id)


@router.post(
    "/organizations/{org_id}/members",
    response_model=OrganizationMembershipPublic,
)
async def add_org_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    body: MembershipCreate,
):
    org = await _load_org(session, org_id)

    if not _can_manage(ctx):
        raise ForbiddenException(message="Insufficient permission to add member")

    user = await User.one_by_id(session, body.user_id)
    if not user or user.is_system:
        raise NotFoundException(message="User not found")

    if await _find_membership(session, org_id, body.user_id):
        raise AlreadyExistsException(
            message=f"User {body.user_id} is already a member of organization {org_id}"
        )

    try:
        membership = OrganizationMembership(
            user_id=body.user_id,
            organization_id=org.id,
            role=body.role,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(membership)
        # If the user has no default org yet, this becomes their default.
        if user.default_organization_id is None:
            user.default_organization_id = org.id
            session.add(user)
        await session.commit()
        await session.refresh(membership)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to add member: {e}")

    return membership


@router.put(
    "/organizations/{org_id}/members/{user_id}",
    response_model=OrganizationMembershipPublic,
)
async def update_org_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    user_id: int,
    body: MembershipUpdate,
):
    await _load_org(session, org_id)
    membership = await _find_membership(session, org_id, user_id)
    if not membership:
        raise NotFoundException(message="Membership not found")

    if not _can_manage(ctx):
        raise ForbiddenException(message="Insufficient permission to change role")

    # If demoting the last admin, refuse — leaves the Org with nobody
    # able to manage members or Org-scoped infra.
    if membership.role == OrgRole.ADMIN and body.role != OrgRole.ADMIN:
        if not await _has_other_admin(session, org_id, exclude_user_id=user_id):
            raise ConflictException(
                message="Cannot demote the only admin of this organization"
            )

    try:
        membership.role = body.role
        session.add(membership)
        await session.commit()
        await session.refresh(membership)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to update member: {e}")
    return membership


@router.delete("/organizations/{org_id}/members/{user_id}")
async def remove_org_member(
    session: SessionDep,
    ctx: TenantContextDep,
    org_id: int,
    user_id: int,
):
    await _load_org(session, org_id)
    membership = await _find_membership(session, org_id, user_id)
    if not membership:
        raise NotFoundException(message="Membership not found")

    if not _can_manage(ctx):
        raise ForbiddenException(message="Insufficient permission to remove member")

    if membership.role == OrgRole.ADMIN:
        if not await _has_other_admin(session, org_id, exclude_user_id=user_id):
            raise ConflictException(
                message="Cannot remove the only admin of this organization"
            )

    try:
        # Clear default_organization_id if it points at this org.
        user = await User.one_by_id(session, user_id)
        if user and user.default_organization_id == org_id:
            user.default_organization_id = None
            session.add(user)
        await session.delete(membership)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to remove member: {e}")
