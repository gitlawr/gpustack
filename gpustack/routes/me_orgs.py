"""Self-service tenant endpoints — what orgs am I in, what clusters can I use."""

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import or_, select

from gpustack.api.exceptions import ForbiddenException, NotFoundException
from gpustack.schemas.cluster_access import ClusterAccess
from gpustack.schemas.clusters import Cluster, ClusterPublic
from gpustack.schemas.organizations import (
    Organization,
    OrganizationMembership,
    OrganizationPublic,
)
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.schemas.user_groups import UserGroup, UserGroupMembership
from gpustack.server.deps import CurrentUserDep, SessionDep, TenantContextDep

router = APIRouter()


class MyOrganization(BaseModel):
    organization: OrganizationPublic
    role: OrgRole

    model_config = {"from_attributes": True}


@router.get("/organizations", response_model=List[MyOrganization])
async def list_my_orgs(session: SessionDep, user: CurrentUserDep):
    stmt = (
        select(OrganizationMembership, Organization)
        .join(
            Organization,
            Organization.id == OrganizationMembership.organization_id,
        )
        .where(
            OrganizationMembership.user_id == user.id,
            Organization.deleted_at.is_(None),
        )
    )
    rows = (await session.exec(stmt)).all()
    return [
        MyOrganization(organization=org, role=membership.role)
        for membership, org in rows
    ]


@router.get("/organizations/{org_id}/clusters", response_model=List[ClusterPublic])
async def list_my_clusters_in_org(
    session: SessionDep, ctx: TenantContextDep, org_id: int
):
    """List clusters accessible to the caller in a specific Org context."""
    org = await Organization.one_by_id(session, org_id)
    if not org or org.deleted_at is not None:
        raise NotFoundException(message="Organization not found")

    if not ctx.is_platform_admin and ctx.current_org_id != org_id:
        raise ForbiddenException(
            message="Cannot inspect clusters of an organization you are not in"
        )

    # Resolve the principals that grant access in this org context.
    user_id = ctx.user.id
    group_stmt = (
        select(UserGroupMembership.group_id)
        .join(UserGroup, UserGroup.id == UserGroupMembership.group_id)
        .where(
            UserGroupMembership.user_id == user_id,
            UserGroup.organization_id == org_id,
        )
    )
    group_ids = list((await session.exec(group_stmt)).all())

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

    cluster_id_stmt = select(ClusterAccess.cluster_id).where(or_(*or_clauses))
    cluster_ids = set((await session.exec(cluster_id_stmt)).all())

    if not cluster_ids:
        return []

    cluster_stmt = select(Cluster).where(
        Cluster.id.in_(cluster_ids), Cluster.deleted_at.is_(None)
    )
    return list((await session.exec(cluster_stmt)).all())
