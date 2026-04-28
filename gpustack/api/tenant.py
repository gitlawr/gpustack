"""Tenant context resolution for multi-tenant request handling.

Each authenticated request resolves to a TenantContext that captures:
- the user identity
- whether they are a platform-level super-admin
- which Organization they are operating in for this request (current_org_id)
- which Org-level role they hold there (owner / admin / member)
- which clusters are accessible in that org context
- which principals (org / groups / user) the request can claim ownership for

Resolution order for current_org_id:
1. If authenticated via API key, use api_key.organization_id (header is ignored)
2. Else, X-Organization-Id request header if provided
3. Else, user.default_organization_id
4. For platform admins, all of the above are optional; absent context
   means "act across all orgs" (no filter applied)
"""

from dataclasses import dataclass, field
from typing import Annotated, List, Optional, Set, Tuple

from fastapi import Depends, Header, Request
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import ForbiddenException
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.cluster_access import ClusterAccess
from gpustack.schemas.organizations import OrganizationMembership
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.user_groups import UserGroupMembership
from gpustack.schemas.users import User
from gpustack.server.db import get_session


PlatformAdminError = ForbiddenException
OrgRoleError = ForbiddenException


@dataclass
class TenantContext:
    """Per-request tenant resolution result."""

    user: User
    is_platform_admin: bool
    current_org_id: Optional[int]
    org_role: Optional[OrgRole]
    accessible_cluster_ids: Set[int] = field(default_factory=set)
    accessible_principals: List[Tuple[PrincipalType, int]] = field(default_factory=list)

    @property
    def has_org_context(self) -> bool:
        return self.current_org_id is not None

    def assert_org_role(self, *allowed: OrgRole) -> None:
        """Raise if the current request is not from a member of `current_org_id`
        with one of the `allowed` roles. Platform admins bypass.
        """
        if self.is_platform_admin:
            return
        if self.org_role is None or self.org_role not in allowed:
            raise OrgRoleError(message="Insufficient organization role")


async def _resolve_membership(
    session: AsyncSession, user_id: int, org_id: int
) -> Optional[OrganizationMembership]:
    stmt = select(OrganizationMembership).where(
        OrganizationMembership.user_id == user_id,
        OrganizationMembership.organization_id == org_id,
    )
    return (await session.exec(stmt)).scalar_one_or_none()


async def _user_group_ids(
    session: AsyncSession, user_id: int, org_id: int
) -> List[int]:
    """Group ids in `org_id` that `user_id` is a member of."""
    from gpustack.schemas.user_groups import UserGroup

    stmt = (
        select(UserGroupMembership.group_id)
        .join(UserGroup, UserGroup.id == UserGroupMembership.group_id)
        .where(
            UserGroupMembership.user_id == user_id,
            UserGroup.organization_id == org_id,
        )
    )
    return [row for row in (await session.exec(stmt)).scalars().all()]


async def _accessible_clusters(
    session: AsyncSession,
    user_id: int,
    org_id: int,
    group_ids: List[int],
) -> Set[int]:
    """Cluster ids reachable from any of: org, user, or any joined group."""
    or_clauses = [
        (ClusterAccess.principal_type == PrincipalType.ORG)
        & (ClusterAccess.principal_id == org_id),
        (ClusterAccess.principal_type == PrincipalType.USER)
        & (ClusterAccess.principal_id == user_id),
    ]
    if group_ids:
        or_clauses.append(
            (ClusterAccess.principal_type == PrincipalType.GROUP)
            & (ClusterAccess.principal_id.in_(group_ids))
        )

    from sqlalchemy import or_

    stmt = select(ClusterAccess.cluster_id).where(or_(*or_clauses))
    return set((await session.exec(stmt)).scalars().all())


def _resolve_requested_org_id(
    request: Request,
    user: User,
    header_value: Optional[str],
) -> Optional[int]:
    api_key: Optional[ApiKey] = getattr(request.state, "api_key", None)
    if api_key is not None and api_key.organization_id is not None:
        return api_key.organization_id

    if header_value:
        try:
            return int(header_value)
        except ValueError as exc:
            raise ForbiddenException(message="Invalid X-Organization-Id") from exc

    return user.default_organization_id


async def get_tenant_context(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    user: Annotated[User, Depends(get_current_user)],
    x_organization_id: Annotated[Optional[str], Header()] = None,
) -> TenantContext:
    """Resolve the per-request TenantContext.

    Result is cached on `request.state.tenant_context` so multiple downstream
    dependencies in the same request share one resolution.
    """
    if hasattr(request.state, "tenant_context"):
        return request.state.tenant_context

    is_platform_admin = bool(user.is_admin)
    current_org_id = _resolve_requested_org_id(request, user, x_organization_id)

    org_role: Optional[OrgRole] = None
    accessible_cluster_ids: Set[int] = set()
    accessible_principals: List[Tuple[PrincipalType, int]] = []

    if current_org_id is not None and not user.is_system:
        membership = await _resolve_membership(session, user.id, current_org_id)
        if membership is not None:
            org_role = membership.role
        elif not is_platform_admin:
            # Non-admin users cannot operate in an org they are not a member of.
            raise ForbiddenException(
                message=f"Not a member of organization {current_org_id}"
            )

        group_ids = await _user_group_ids(session, user.id, current_org_id)
        accessible_cluster_ids = await _accessible_clusters(
            session, user.id, current_org_id, group_ids
        )
        accessible_principals = [
            (PrincipalType.ORG, current_org_id),
            (PrincipalType.USER, user.id),
            *[(PrincipalType.GROUP, gid) for gid in group_ids],
        ]

    ctx = TenantContext(
        user=user,
        is_platform_admin=is_platform_admin,
        current_org_id=current_org_id,
        org_role=org_role,
        accessible_cluster_ids=accessible_cluster_ids,
        accessible_principals=accessible_principals,
    )
    request.state.tenant_context = ctx
    return ctx


async def require_platform_admin(
    ctx: Annotated[TenantContext, Depends(get_tenant_context)],
) -> TenantContext:
    """Allow only platform-level super-admin (`users.is_admin = True`)."""
    if not ctx.is_platform_admin:
        raise PlatformAdminError(message="Platform admin permission required")
    return ctx


def require_org_role(*allowed: OrgRole):
    """Build a dependency that requires the requesting user to hold one of the
    given roles in `current_org_id`. Platform admins always pass.
    """

    async def _dep(
        ctx: Annotated[TenantContext, Depends(get_tenant_context)],
    ) -> TenantContext:
        if ctx.is_platform_admin:
            return ctx
        if ctx.current_org_id is None:
            raise OrgRoleError(message="Organization context required")
        ctx.assert_org_role(*allowed)
        return ctx

    return _dep
