"""Cluster access authorization — platform admin only.

Lets the platform admin grant or revoke a cluster's accessibility to a
specific principal (org / group / user). Granting access does not auto-create
a tenant namespace; that is reconciled separately in P3.
"""

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.cluster_access import ClusterAccess, ClusterAccessPublic
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.organizations import Organization
from gpustack.schemas.principals import PrincipalType
from gpustack.schemas.user_groups import UserGroup
from gpustack.schemas.users import User
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()


class ClusterAccessGrant(BaseModel):
    principal_type: PrincipalType
    principal_id: int


async def _load_cluster(session, cluster_id: int) -> Cluster:
    cluster = await Cluster.one_by_id(session, cluster_id)
    if not cluster or cluster.deleted_at is not None:
        raise NotFoundException(message="Cluster not found")
    return cluster


async def _validate_principal(
    session, principal_type: PrincipalType, principal_id: int
) -> None:
    """Ensure the referenced org/group/user actually exists and is active."""
    if principal_type == PrincipalType.ORG:
        target = await Organization.one_by_id(session, principal_id)
        if not target or target.deleted_at is not None:
            raise InvalidException(message=f"Organization {principal_id} not found")
    elif principal_type == PrincipalType.GROUP:
        target = await UserGroup.one_by_id(session, principal_id)
        if not target or target.deleted_at is not None:
            raise InvalidException(message=f"User group {principal_id} not found")
    elif principal_type == PrincipalType.USER:
        target = await User.one_by_id(session, principal_id)
        if not target or target.is_system or target.deleted_at is not None:
            raise InvalidException(message=f"User {principal_id} not found")


async def _resolve_principal_names(
    session, rows: List[ClusterAccess]
) -> List[ClusterAccessPublic]:
    """Bulk-resolve human-readable labels for each (principal_type, id).

    Done in three batched lookups (one per principal type) instead of
    per-row joins; cluster_access lists are tiny in practice but the
    pattern keeps it O(1) round-trips no matter how many rows there are.
    """
    org_ids = {r.principal_id for r in rows if r.principal_type == PrincipalType.ORG}
    group_ids = {
        r.principal_id for r in rows if r.principal_type == PrincipalType.GROUP
    }
    user_ids = {r.principal_id for r in rows if r.principal_type == PrincipalType.USER}

    org_names: dict[int, str] = {}
    if org_ids:
        result = await session.exec(
            select(Organization).where(Organization.id.in_(org_ids))
        )
        org_names = {o.id: o.name for o in result.all()}

    group_names: dict[int, str] = {}
    if group_ids:
        result = await session.exec(
            select(UserGroup).where(UserGroup.id.in_(group_ids))
        )
        group_names = {g.id: g.name for g in result.all()}

    user_names: dict[int, str] = {}
    if user_ids:
        result = await session.exec(select(User).where(User.id.in_(user_ids)))
        user_names = {u.id: u.username for u in result.all()}

    out: List[ClusterAccessPublic] = []
    for r in rows:
        if r.principal_type == PrincipalType.ORG:
            name = org_names.get(r.principal_id)
        elif r.principal_type == PrincipalType.GROUP:
            name = group_names.get(r.principal_id)
        else:
            name = user_names.get(r.principal_id)
        out.append(
            ClusterAccessPublic(
                cluster_id=r.cluster_id,
                principal_type=r.principal_type,
                principal_id=r.principal_id,
                principal_name=name,
                granted_by=r.granted_by,
                created_at=r.created_at,
            )
        )
    return out


@router.get("/clusters/{cluster_id}/access", response_model=List[ClusterAccessPublic])
async def list_cluster_access(
    session: SessionDep, ctx: TenantContextDep, cluster_id: int
):
    await _load_cluster(session, cluster_id)
    stmt = select(ClusterAccess).where(ClusterAccess.cluster_id == cluster_id)
    rows = list((await session.exec(stmt)).all())
    return await _resolve_principal_names(session, rows)


@router.post("/clusters/{cluster_id}/access", response_model=ClusterAccessPublic)
async def grant_cluster_access(
    session: SessionDep,
    ctx: TenantContextDep,
    cluster_id: int,
    body: ClusterAccessGrant,
):
    await _load_cluster(session, cluster_id)
    await _validate_principal(session, body.principal_type, body.principal_id)

    existing_stmt = select(ClusterAccess).where(
        ClusterAccess.cluster_id == cluster_id,
        ClusterAccess.principal_type == body.principal_type,
        ClusterAccess.principal_id == body.principal_id,
    )
    if (await session.exec(existing_stmt)).first() is not None:
        raise AlreadyExistsException(message="Access already granted")

    try:
        access = ClusterAccess(
            cluster_id=cluster_id,
            principal_type=body.principal_type,
            principal_id=body.principal_id,
            granted_by=ctx.user.id,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        session.add(access)
        await session.commit()
        await session.refresh(access)
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to grant cluster access: {e}")

    enriched = await _resolve_principal_names(session, [access])
    return enriched[0]


@router.delete("/clusters/{cluster_id}/access/{principal_type}/{principal_id}")
async def revoke_cluster_access(
    session: SessionDep,
    ctx: TenantContextDep,
    cluster_id: int,
    principal_type: PrincipalType,
    principal_id: int,
):
    await _load_cluster(session, cluster_id)
    stmt = select(ClusterAccess).where(
        ClusterAccess.cluster_id == cluster_id,
        ClusterAccess.principal_type == principal_type,
        ClusterAccess.principal_id == principal_id,
    )
    access = (await session.exec(stmt)).first()
    if not access:
        raise NotFoundException(message="Access grant not found")

    try:
        await session.delete(access)
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise InvalidException(message=f"Failed to revoke cluster access: {e}")
