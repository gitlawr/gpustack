from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    ConflictException,
)
from gpustack.security import get_secret_hash
from gpustack.schemas.cluster_access import ClusterAccess
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.links import ModelRoutePrincipalLink
from gpustack.schemas.model_routes import ModelRoute
from gpustack.schemas.organizations import (
    Organization,
    OrganizationMembership,
    PLATFORM_ORGANIZATION_ID,
)
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.db import async_session
from gpustack.server.deps import CurrentUserDep, SessionDep
from gpustack.schemas.users import (
    User,
    UserActivationUpdate,
    UserCreate,
    UserListParams,
    UserUpdate,
    UserPublic,
    UsersPublic,
    UserSelfUpdate,
)
from gpustack.server.services import UserService

router = APIRouter()


@router.get("", response_model=UsersPublic)
async def get_users(
    params: UserListParams = Depends(),
    search: str = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"username": search, "full_name": search}

    if params.watch:
        return StreamingResponse(
            User.streaming(fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await User.paginated_by_query(
            session=session,
            fuzzy_fields=fuzzy_fields,
            page=params.page,
            per_page=params.perPage,
            fields={
                "deleted_at": None,
                "is_system": False,
            },
            order_by=params.order_by,
        )


@router.get("/{id}", response_model=UserPublic)
async def get_user(session: SessionDep, id: int):
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")
    return user


@router.post("", response_model=UserPublic)
async def create_user(session: SessionDep, user_in: UserCreate):
    existing = await User.one_by_field(session, "username", user_in.username)
    if existing:
        raise AlreadyExistsException(message=f"User {user_in.username} already exists")

    try:
        to_create = User(
            username=user_in.username,
            full_name=user_in.full_name,
            is_admin=user_in.is_admin,
            is_active=user_in.is_active,
        )
        if user_in.password:
            to_create.hashed_password = get_secret_hash(user_in.password)
        user = await User.create(session, to_create)

        # Build a Personal Org as the user's default namespace, à la
        # GitHub's per-user account. Admin additionally joins the Default
        # Org as OWNER (so they can manage the platform-shared workspace);
        # regular users do NOT auto-join Default — admin can add them
        # later if shared workspace access is needed.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        personal = Organization(
            name="Personal",
            slug=f"user-{user.id}",
            description="Personal namespace",
            is_personal=True,
            is_platform=False,
        )
        session.add(personal)
        await session.flush()
        session.add(
            OrganizationMembership(
                user_id=user.id,
                organization_id=personal.id,
                role=OrgRole.OWNER,
                created_at=now,
            )
        )
        if user.is_admin:
            session.add(
                OrganizationMembership(
                    user_id=user.id,
                    organization_id=PLATFORM_ORGANIZATION_ID,
                    role=OrgRole.OWNER,
                    created_at=now,
                )
            )

        user.default_organization_id = personal.id
        session.add(user)

        await session.commit()
        await session.refresh(user)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to create user: {e}")

    return user


@router.put("/{id}", response_model=UserPublic)
async def update_user(session: SessionDep, id: int, user_in: UserUpdate):
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")

    if (
        user.is_active
        and user_in.is_active is False
        and await is_only_admin_user(session, user)
    ):
        raise ConflictException(message="Cannot deactivate the only admin user")

    try:
        update_data = user_in.model_dump()
        if user_in.password:
            hashed_password = get_secret_hash(user_in.password)
            update_data["hashed_password"] = hashed_password
        del update_data["password"]
        del update_data["source"]
        await UserService(session).update(user, update_data)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update user: {e}")

    return user


@router.patch("/{id}/activation", response_model=UserPublic)
async def update_user_activation(
    session: SessionDep, id: int, activation_data: UserActivationUpdate
):
    """
    Activate or deactivate a user account.
    Only administrators can perform this action.
    """
    user = await User.one_by_id(session, id)
    if not user:
        raise NotFoundException(message="User not found")

    changed = user.is_active != activation_data.is_active
    if not changed:
        return user

    if (
        user.is_active
        and activation_data.is_active is False
        and await is_only_admin_user(session, user)
    ):
        raise ConflictException(message="Cannot deactivate the only admin user")

    try:
        await UserService(session).update(
            user, {"is_active": activation_data.is_active}
        )
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update user activation: {e}"
        )

    return user


async def _personal_org_has_shared_resources(
    session, personal_org_id: int, user_id: int
) -> Optional[str]:
    """Return a human reason string if the Personal Org owns resources
    that have been shared outside the user's own scope; None otherwise.

    Two flavours of "shared" are checked:

    1. ModelRoutes owned by the Personal Org with `model_route_principals`
       rows pointing to a principal other than (USER, this user). Means
       the user has published one of their routes to other orgs / groups
       / users — deleting the user would yank those publications away.
    2. Clusters owned by the Personal Org with `cluster_access` rows for
       any principal other than (USER, this user) — i.e. they've sublet
       a cluster to other tenants.
    """
    # Find route ids owned by this Personal Org that have any non-self
    # principal_link.
    route_stmt = (
        select(ModelRoutePrincipalLink.route_id)
        .join(ModelRoute, ModelRoute.id == ModelRoutePrincipalLink.route_id)
        .where(
            ModelRoute.organization_id == personal_org_id,
            ~(
                (ModelRoutePrincipalLink.principal_type == PrincipalType.USER)
                & (ModelRoutePrincipalLink.principal_id == user_id)
            ),
        )
        .limit(1)
    )
    if (await session.exec(route_stmt)).first() is not None:
        return "User's Personal Org has model routes published to others"

    # Find cluster_access grants on clusters owned by this Personal Org
    # that point to a principal other than this user.
    access_stmt = (
        select(ClusterAccess.cluster_id)
        .join(Cluster, Cluster.id == ClusterAccess.cluster_id)
        .where(
            Cluster.organization_id == personal_org_id,
            ~(
                (ClusterAccess.principal_type == PrincipalType.USER)
                & (ClusterAccess.principal_id == user_id)
            ),
        )
        .limit(1)
    )
    if (await session.exec(access_stmt)).first() is not None:
        return "User's Personal Org has cluster access granted to others"

    return None


@router.delete("/{id}")
async def delete_user(session: SessionDep, id: int):
    user_service = UserService(session)
    user = await user_service.get_by_id(id)
    if not user:
        raise NotFoundException(message="User not found")

    if await is_only_admin_user(session, user):
        raise ConflictException(message="Cannot delete the only admin user")

    # Block delete if the user's Personal Org has things shared outside.
    # default_organization_id points at the Personal Org for normal users;
    # admins might have it pointing at the Default Org so we look up by
    # slug as a fallback.
    personal_stmt = select(Organization).where(
        Organization.is_personal == True,  # noqa: E712
        Organization.slug == f"user-{user.id}",
    )
    personal = (await session.exec(personal_stmt)).first()
    if personal is not None:
        reason = await _personal_org_has_shared_resources(session, personal.id, user.id)
        if reason:
            raise ConflictException(
                message=(
                    f"{reason}. Reassign or revoke the shared resources before "
                    "deleting the user."
                )
            )

    try:
        await user_service.delete(user)
        # Cascade: also delete the orphan Personal Org. FK on
        # users.default_organization_id is SET NULL, so Org survives the
        # user delete; without explicit cleanup it would linger as a
        # zero-member shell. Note: model / model_route / api_key rows
        # owned by the Org go with it (their FK is CASCADE); clusters
        # owned by it would normally become NULL (platform-shared) but
        # the shared-resource guard above already blocked that case.
        if personal is not None:
            await session.delete(personal)
            await session.commit()
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to delete user: {e}")


async def is_only_admin_user(session: SessionDep, user: User) -> bool:
    if not user.is_admin:
        return False
    admin_count = await User.count_by_fields(
        session, {"is_admin": True, "is_active": True}
    )
    return admin_count == 1


me_router = APIRouter()


@me_router.get("/me", response_model=UserPublic)
async def get_user_me(user: CurrentUserDep):
    return user


@me_router.put("/me", response_model=UserPublic)
async def update_user_me(
    session: SessionDep, user: CurrentUserDep, user_in: UserSelfUpdate
):
    try:
        update_data = user_in.model_dump(exclude_none=True)
        if "password" in update_data:
            hashed_password = get_secret_hash(update_data["password"])
            update_data["hashed_password"] = hashed_password
            del update_data["password"]
        await UserService(session).update(user, update_data)
    except Exception as e:
        raise InternalServerErrorException(message=f"Failed to update user: {e}")

    return user
