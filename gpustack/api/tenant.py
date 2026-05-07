"""Tenant context resolution for multi-tenant request handling.

Each authenticated request resolves to a TenantContext that captures:
- the user identity
- whether they are a platform-level super-admin
- which Organization they are operating in for this request (current_org_id)
- which Org-level role they hold there (admin / user)
- which clusters are accessible in that org context
- which principals (org / groups / user) the request can claim ownership for

Read resolution order for current_org_id:
1. If authenticated via API key, use api_key.organization_id (header is ignored)
2. Else, X-Organization-Id request header if provided
3. Else, for non-admin: user.default_organization_id (always set by the
   foundation migration / Personal-Org provisioning, so non-admin requests
   are never context-less)
4. Else, for platform admin: None — "act across all orgs", read paths skip
   tenant filters via bypass_tenant_filter

Writes that need a concrete owning Org use TenantContext.target_org_id_for_write,
which falls back to user.default_organization_id when current_org_id is None
(covers the OSS host that never sends X-Organization-Id, and admin in "All"
mode hitting a CREATE).
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, List, Optional, Set

from fastapi import Depends, Header, Request
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.auth import get_current_user
from gpustack.api.exceptions import (
    ForbiddenException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.cluster_access import ClusterAccess
from gpustack.schemas.organizations import Organization, OrganizationMembership
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
    # True when ``current_org_id`` points at a Personal Org. Lets endpoints
    # treat the "owner of a one-member org" case differently from a real
    # multi-user Org admin — e.g. ``scope='all'`` on the Usage page should
    # behave like ``self`` here (Personal Org admin = the user themselves).
    current_org_is_personal: bool = False

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

    def target_org_id_for_write(self) -> Optional[int]:
        """Resolve the Org a CREATE / write request should land in.

        Reads happily honor the platform-admin "no current_org_id =
        all-orgs" mode, but writes need an actual Org to stamp on the
        new row. When the request didn't pin a context (no header on
        an admin request, or a built-in client like the OSS host that
        never sends ``X-Organization-Id``), fall back to the user's
        ``default_organization_id`` — which the migration sets to the
        platform (Default) Org for everyone — instead of failing.
        """
        if self.current_org_id is not None:
            return self.current_org_id
        return getattr(self.user, "default_organization_id", None)


async def _resolve_membership(
    session: AsyncSession, user_id: int, org_id: int
) -> Optional[OrganizationMembership]:
    stmt = select(OrganizationMembership).where(
        OrganizationMembership.user_id == user_id,
        OrganizationMembership.organization_id == org_id,
    )
    return (await session.exec(stmt)).first()


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
    return list((await session.exec(stmt)).all())


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
    return set((await session.exec(stmt)).all())


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
            # 400 — the header is structurally bad, not a permission issue.
            raise InvalidException(message="Invalid X-Organization-Id") from exc

    # Platform admins default to "no org context" (cross-org platform view)
    # when nothing is supplied. They opt into act-as mode by sending
    # X-Organization-Id explicitly. Non-admins fall back to their default
    # organization so they can never run requests outside an org context.
    if user.is_admin:
        return None
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
    current_org_is_personal = False

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

        # Cheap one-shot lookup to surface ``is_personal`` for downstream
        # endpoints that need to special-case Personal Org behavior
        # (e.g. usage scope semantics). Uses ``first_by_field`` rather
        # than ``one_by_id`` so the same mocked ``session.exec`` path
        # used by every other lookup in this resolver covers the test
        # suite without a separate ``session.get`` mock.
        org_row = await Organization.first_by_field(session, "id", current_org_id)
        if org_row is not None:
            current_org_is_personal = bool(org_row.is_personal)

    ctx = TenantContext(
        user=user,
        is_platform_admin=is_platform_admin,
        current_org_id=current_org_id,
        org_role=org_role,
        accessible_cluster_ids=accessible_cluster_ids,
        current_org_is_personal=current_org_is_personal,
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


def bypass_tenant_filter(ctx: TenantContext) -> bool:
    """Identify request contexts that should not be tenant-scoped.

    Two categories bypass:
    - Platform admin with no org context (cross-org platform view).
    - System users (worker / cluster service accounts that the server
      itself spawns). They authenticate as ``is_system=True`` and need
      to read every Org's resources to do their job — e.g. a worker
      fetching the Model row for an instance assigned to it.
    """
    if ctx.user is not None and getattr(ctx.user, "is_system", False):
        return True
    if ctx.is_platform_admin and ctx.current_org_id is None:
        return True
    return False


def tenant_list_conditions(
    ctx: TenantContext,
    model: Any,
) -> List[Any]:
    """Build SQLAlchemy WHERE clauses to scope a list query to the caller.

    Visibility model:
    - System users (workers / cluster service accounts) and platform
      admin without org context see everything — returns no conditions.
    - Everyone else with an org context filters by
      ``model.organization_id == ctx.current_org_id``. Membership in
      the org is already enforced by ``get_tenant_context``.
    """
    conditions: List[Any] = []
    if bypass_tenant_filter(ctx):
        return conditions

    if ctx.current_org_id is not None and hasattr(model, "organization_id"):
        conditions.append(model.organization_id == ctx.current_org_id)

    return conditions


def cluster_visibility_conditions(
    ctx: TenantContext,
    model: Any,
) -> List[Any]:
    """Visibility filter specific to Cluster-like infrastructure rows.

    Clusters can be visible to a non-admin caller through TWO independent
    paths, so the regular ``organization_id`` equality filter would be
    too narrow:

    - **Own-Org cluster** (``cluster.organization_id == current_org_id``):
      the org's BYO cluster.
    - **Granted via cluster_access** (``cluster.id`` ∈
      ``ctx.accessible_cluster_ids``): global clusters the admin
      authorised, or another Org's cluster sublet to us via cluster_access.

    Either path makes the cluster visible. System users and platform
    admins (no-org-context) bypass entirely.
    """
    from sqlalchemy import or_

    if bypass_tenant_filter(ctx):
        return []

    or_clauses = []
    if ctx.current_org_id is not None:
        or_clauses.append(model.organization_id == ctx.current_org_id)
    if ctx.accessible_cluster_ids:
        or_clauses.append(model.id.in_(ctx.accessible_cluster_ids))

    if not or_clauses:
        # No avenue to see anything; force an empty result rather than
        # leak the full table when accessible_cluster_ids is empty.
        return [model.id == -1]

    return [or_(*or_clauses)]


def cluster_resource_visibility_conditions(
    ctx: TenantContext,
    model: Any,
) -> List[Any]:
    """Visibility filter for resources that carry BOTH ``organization_id``
    (denormalized from cluster) AND ``cluster_id`` — Worker, ModelFile,
    Benchmark, ModelEvaluation, etc.

    A row is visible if:
    - it's owned by the caller's current Org (``organization_id`` match), OR
    - its cluster is granted via ``cluster_access`` (``cluster_id`` ∈
      ``accessible_cluster_ids``).

    NULL ``organization_id`` rows live on global clusters; they're
    only visible through the second branch (cluster_access) for non-admin.
    """
    from sqlalchemy import or_

    if bypass_tenant_filter(ctx):
        return []

    or_clauses = []
    if ctx.current_org_id is not None and hasattr(model, "organization_id"):
        or_clauses.append(model.organization_id == ctx.current_org_id)
    if ctx.accessible_cluster_ids and hasattr(model, "cluster_id"):
        or_clauses.append(model.cluster_id.in_(ctx.accessible_cluster_ids))

    if not or_clauses:
        # No access path; force empty result rather than leak.
        anchor = getattr(model, "cluster_id", None) or getattr(model, "id", None)
        return [anchor == -1]
    return [or_(*or_clauses)]


def assert_cluster_resource_visible(
    ctx: TenantContext,
    resource: Any,
    *,
    not_found_message: str = "Resource not found",
) -> None:
    """Single-row mirror of ``cluster_resource_visibility_conditions``.

    Resource must carry ``organization_id`` and/or ``cluster_id``.
    """
    if resource is None:
        raise NotFoundException(message=not_found_message)
    if bypass_tenant_filter(ctx):
        return

    org_id = getattr(resource, "organization_id", None)
    cluster_id = getattr(resource, "cluster_id", None)

    if (
        ctx.current_org_id is not None
        and org_id is not None
        and org_id == ctx.current_org_id
    ):
        return
    if cluster_id is not None and cluster_id in ctx.accessible_cluster_ids:
        return
    raise NotFoundException(message=not_found_message)


def assert_cluster_visible(
    ctx: TenantContext,
    cluster: Any,
    *,
    not_found_message: str = "Cluster not found",
) -> None:
    """404 if the caller can't see this cluster (own-Org OR cluster_access)."""
    if cluster is None:
        raise NotFoundException(message=not_found_message)
    if bypass_tenant_filter(ctx):
        return
    cluster_org = getattr(cluster, "organization_id", None)
    if (
        ctx.current_org_id is not None
        and cluster_org is not None
        and cluster_org == ctx.current_org_id
    ):
        return
    if cluster.id in ctx.accessible_cluster_ids:
        return
    raise NotFoundException(message=not_found_message)


def assert_org_owned_writable(
    ctx: TenantContext,
    resource: Any,
    *,
    resource_label: str = "resource",
) -> None:
    """403 if the caller can't mutate an org-owned infrastructure row.

    Used for clusters / cloud_credentials / worker_pools / inference
    backends — anything with a nullable ``organization_id`` and these
    write rules:

    - Platform admin / system user → allowed (bypass via
      ``bypass_tenant_filter`` for "All" mode admin and system users;
      admin in act-as falls through to org-row check, where they're
      treated like an Org admin).
    - **Org-owned** (org_id == current_org_id): an Org admin can write;
      platform admin in act-as bypasses the role check (admin is admin
      everywhere, even when scoped to one Org).
    - **Global** (org_id IS NULL): only "All"-mode admin — Org admins
      and admin-in-act-as cannot mutate Global rows directly. Resource
      handlers redirect such writes to the caller's Org row instead.
    - **Other Org's row**: never writable for non-admin.
    """
    if bypass_tenant_filter(ctx):
        return
    res_org = getattr(resource, "organization_id", None)
    if res_org is None:
        raise PlatformAdminError(
            message=f"Only platform admin can modify global {resource_label}"
        )
    if res_org != ctx.current_org_id:
        raise OrgRoleError(
            message=f"{resource_label.capitalize()} does not belong to current Org"
        )
    # Platform admin acting-as the Org passes the role check unconditionally;
    # for non-admin we require Org admin.
    if not ctx.is_platform_admin and ctx.org_role != OrgRole.ADMIN:
        raise OrgRoleError(
            message=f"Insufficient organization role to modify this {resource_label}"
        )


def assert_cluster_writable(
    ctx: TenantContext,
    cluster: Any,
) -> None:
    assert_org_owned_writable(ctx, cluster, resource_label="cluster")


def validate_org_owned_owner(
    input_org_id: Optional[int],
    ctx: TenantContext,
    *,
    resource_label: str = "resource",
) -> None:
    """Decide whether the caller can create a row owned by ``input_org_id``.

    - Platform admin: any value (including NULL = global)
    - Org admin: must equal current_org_id; can't create global
    """
    if ctx.is_platform_admin:
        return
    if input_org_id is None:
        raise InvalidException(
            message=f"Only platform admin can create global {resource_label}s"
        )
    if ctx.current_org_id is None or input_org_id != ctx.current_org_id:
        raise InvalidException(
            message="organization_id must match the current organization"
        )
    if ctx.org_role != OrgRole.ADMIN:
        raise InvalidException(
            message=f"Insufficient organization role to create a {resource_label}"
        )


def assert_resource_visible(
    ctx: TenantContext,
    resource: Any,
    *,
    not_found_message: str = "Resource not found",
) -> None:
    """Raise 404 if the caller is not allowed to see ``resource``.

    Mirrors the semantics of ``tenant_list_conditions`` for single-item
    GET / PUT / DELETE handlers: same visibility rules, raised as 404
    rather than 403 to avoid leaking the existence of cross-tenant rows.
    """
    if resource is None:
        raise NotFoundException(message=not_found_message)

    if bypass_tenant_filter(ctx):
        return

    org_id = getattr(resource, "organization_id", None)
    if (
        ctx.current_org_id is not None
        and org_id is not None
        and org_id != ctx.current_org_id
    ):
        raise NotFoundException(message=not_found_message)


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
