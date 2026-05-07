import logging
from typing import List, Optional, Union, Set, Tuple
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import selectinload

from gpustack.api.exceptions import InternalServerErrorException

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
)
from gpustack.schemas.model_routes import (
    ModelRoute,
    MyModel,
    ModelRouteTarget,
    TargetStateEnum,
    AccessPolicyEnum,
    effective_route_name,
)
from gpustack.schemas.organizations import Organization
from gpustack.schemas.users import User
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.workers import Worker
from gpustack.server.usage_buffer import usage_flush_buffer
from gpustack.server.cache import (
    build_cache_key,
    delete_cache_by_key,
    set_cache_by_key,
    locked_cached,
)


logger = logging.getLogger(__name__)


class UserService:

    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, user_id: int) -> Optional[User]:
        result = await User.one_by_id(
            self.session,
            user_id,
            options=[selectinload(User.worker), selectinload(User.cluster)],
        )
        if result is None:
            return None
        if result.worker is not None:
            # detach worker to avoid lazy loading
            self.session.expunge(result.worker)
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_by_username(self, username: str) -> Optional[User]:
        result = await User.one_by_field(self.session, "username", username)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def create(self, user: User):
        return await User.create(self.session, user)

    async def update(self, user: User, source: Union[dict, SQLModel, None] = None):
        result = await user.update(self.session, source)
        await delete_cache_by_key(self.get_by_id, user.id)
        await delete_cache_by_key(self.get_user_accessible_model_names, user.id)
        await delete_cache_by_key(self.get_by_username, user.username)
        return result

    async def delete(self, user: User):
        apikeys = await APIKeyService(self.session).get_by_user_id(user.id)
        result = await user.delete(self.session)
        await delete_cache_by_key(self.get_by_id, user.id)
        await delete_cache_by_key(self.get_user_accessible_model_names, user.id)
        await delete_cache_by_key(self.get_by_username, user.username)
        for apikey in apikeys:
            await delete_cache_by_key(
                APIKeyService.get_by_access_key, apikey.access_key
            )
        return result

    async def model_allowed_for_user(
        self, model_name: str, user_id: int, api_key: Optional[ApiKey]
    ) -> bool:
        limited_model_names: Optional[Set[str]] = (
            set(api_key.allowed_model_names)
            if api_key is not None
            and api_key.allowed_model_names is not None
            and len(api_key.allowed_model_names) > 0
            else None
        )
        accessible_model_names: Set[str] = await self.get_user_accessible_model_names(
            user_id
        )
        allowed = model_name in intersection_nullable_set(
            accessible_model_names, limited_model_names
        )
        if not allowed:
            logger.info(
                "Access denied: model_name=%r user_id=%d " "accessible=%s limited=%s",
                model_name,
                user_id,
                sorted(accessible_model_names),
                sorted(limited_model_names) if limited_model_names else None,
            )
        return allowed

    @locked_cached()
    async def get_user_accessible_model_names(self, user_id: int) -> Set[str]:
        # Get all accessible model names for the user. The set holds two
        # forms per route:
        #   1. Org-effective name (`<slug>/<route>` for non-platform
        #      Orgs, raw for platform) — matches `/v1/models` output and
        #      the gateway's ingress header matcher.
        #   2. Raw `route.name` — matches the post-`modelMapping` value
        #      that Higress's AI proxy hands back via
        #      `x-higress-llm-model` on the auth callback. Without this
        #      the callback would deny chat traffic for non-platform
        #      Orgs even though the gateway already routed it to the
        #      correct ingress.
        # Cross-Org collisions on raw names are fine: each user's set is
        # isolated, and Higress's per-Org ingress already disambiguates
        # which underlying instance receives the request.
        user: User = await self.get_by_id(user_id)
        if user is None:
            return set()
        if user.is_admin or user.is_system:
            routes = await ModelRoute.all_by_field(self.session, "deleted_at", None)
        else:
            routes = await MyModel.all_by_fields(
                self.session, {"user_id": user.id, "deleted_at": None}
            )
        org_ids = {r.organization_id for r in routes if r.organization_id is not None}
        org_by_id = {}
        if org_ids:
            rows = (
                await self.session.exec(
                    select(Organization).where(Organization.id.in_(org_ids))
                )
            ).all()
            org_by_id = {o.id: o for o in rows}
        names: Set[str] = set()
        for r in routes:
            org = org_by_id.get(r.organization_id) if r.organization_id else None
            names.add(
                effective_route_name(
                    r.name,
                    getattr(org, "slug", None),
                    bool(getattr(org, "is_platform", False)),
                )
            )
            names.add(r.name)
        return names


class APIKeyService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_access_key(self, access_key: str) -> Optional[ApiKey]:
        result = await ApiKey.one_by_field(self.session, "access_key", access_key)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def get_by_user_id(self, user_id: int) -> List[ApiKey]:
        results = await ApiKey.all_by_field(self.session, "user_id", user_id)
        if results is None:
            return []
        for result in results:
            self.session.expunge(result)
        return results

    async def update(self, api_key: ApiKey, source: Union[dict, SQLModel, None] = None):
        result = await api_key.update(self.session, source)
        await delete_cache_by_key(self.get_by_access_key, api_key.access_key)
        return result

    async def delete(self, api_key: ApiKey):
        result = await api_key.delete(self.session)
        await delete_cache_by_key(self.get_by_access_key, api_key.access_key)
        return result


class ClusterService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, cluster_id: int) -> Optional[Cluster]:
        result = await Cluster.one_by_id(self.session, cluster_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result


class WorkerService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, worker_id: int) -> Optional[Worker]:
        result = await Worker.one_by_id(self.session, worker_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_by_cluster_id_name(
        self, cluster_id: int, name: str
    ) -> Optional[Worker]:
        result = await Worker.one_by_fields(
            self.session, fields={"cluster_id": cluster_id, "name": name}
        )
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_by_name(self, name: str) -> Optional[Worker]:
        result = await Worker.one_by_field(self.session, "name", name)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def update(
        self, worker: Worker, source: Union[dict, SQLModel, None] = None, **kwargs
    ):
        result = await worker.update(self.session, source, **kwargs)
        # Worker cache is high-frequency, non-security-critical, skip coordinator sync
        await delete_cache_by_key(self.get_by_id, worker.id, sync_coordinator=False)
        await delete_cache_by_key(self.get_by_name, worker.name, sync_coordinator=False)
        return result

    async def batch_update(
        self,
        workers: List[Worker],
        source: Union[dict, SQLModel, None] = None,
        **kwargs,
    ) -> int:
        if not workers:
            return 0

        updated = await Worker.batch_update(self.session, workers)

        for w in workers:
            # Worker cache is high-frequency, non-security-critical, skip coordinator sync
            await delete_cache_by_key(self.get_by_id, w.id, sync_coordinator=False)
            await delete_cache_by_key(self.get_by_name, w.name, sync_coordinator=False)

        return updated

    async def delete(self, worker: Worker, **kwargs):
        worker_id = worker.id
        worker_name = worker.name
        result = await worker.delete(self.session, **kwargs)
        # Worker cache is high-frequency, non-security-critical, skip coordinator sync
        await delete_cache_by_key(self.get_by_id, worker_id, sync_coordinator=False)
        await delete_cache_by_key(self.get_by_name, worker_name, sync_coordinator=False)
        return result


class ModelRouteService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_name(self, name: str) -> Optional[ModelRoute]:
        result = await ModelRoute.one_by_field(self.session, "name", name)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_model_auth_info_by_name(
        self, name: str
    ) -> Optional[Tuple[AccessPolicyEnum, str]]:
        # Higress's auth callback may hand us either the Org-effective
        # name (`<slug>/<route>`) or the raw `route.name` depending on
        # whether `modelMapping` has fired yet. Resolve both forms.
        route: Optional[ModelRoute] = None
        if "/" in name:
            slug, _, rest = name.partition("/")
            if rest:
                org = await Organization.one_by_field(self.session, "slug", slug)
                if org is not None:
                    route = await ModelRoute.one_by_fields(
                        self.session,
                        {"name": rest, "organization_id": org.id},
                    )
        if route is None:
            route = await ModelRoute.one_by_field(self.session, "name", name)
        if route is None:
            return None
        route_targets = await ModelRouteTarget.all_by_fields(
            self.session,
            fields={"route_id": route.id},
        )
        if len(route_targets) == 0:
            return None
        models = await Model.all_by_fields(
            session=self.session,
            fields={},
            extra_conditions=[
                Model.id.in_(
                    [e.model_id for e in route_targets if e.model_id is not None]
                )
            ],
        )
        # set a default static token to avoid empty token response for public maas model route
        registration_token = "static_token_not_found"
        for model in models:
            cluster = await Cluster.one_by_id(self.session, model.cluster_id)
            if cluster.registration_token is not None:
                registration_token = cluster.registration_token
                break

        return route.access_policy, registration_token

    @locked_cached()
    async def get_model_ids_by_model_route_name(self, name: str) -> List[Model]:
        # Clients send the Org-effective name (e.g. "org1/qwen3-0.6b").
        # Targets are stored keyed by raw `route_name`, so split off the
        # slug and constrain by the route's owning Org. Platform Org
        # routes have no prefix — fall back to the legacy lookup.
        org_id: Optional[int] = None
        raw_name = name
        if "/" in name:
            slug, _, rest = name.partition("/")
            if rest:
                org = await Organization.one_by_field(self.session, "slug", slug)
                if org is not None:
                    org_id = org.id
                    raw_name = rest
                # If the slug didn't match an Org, fall through and try
                # the literal name (handles edge cases like a route
                # called "literal/with/slashes" before the prefix
                # convention existed).
        target_fields = {
            "route_name": raw_name,
            "state": TargetStateEnum.ACTIVE,
            "deleted_at": None,
        }
        targets = await ModelRouteTarget.all_by_fields(
            self.session,
            fields=target_fields,
            options=[selectinload(ModelRouteTarget.model)],
        )
        # When an Org slug was parsed, narrow to that Org's route by
        # joining through the parent ModelRoute's `organization_id`.
        # This avoids an extra round-trip when the route name is
        # globally unique (the typical single-Org case).
        if org_id is not None and len(targets) > 0:
            route_ids = {t.route_id for t in targets if t.route_id is not None}
            org_routes = await ModelRoute.all_by_fields(
                self.session,
                fields={"organization_id": org_id, "deleted_at": None},
            )
            allowed_route_ids = {r.id for r in org_routes if r.id in route_ids}
            targets = [t for t in targets if t.route_id in allowed_route_ids]
        models = [target.model for target in targets if target.model is not None]
        for model in models:
            self.session.expunge(model)
        return models

    async def update(
        self,
        model_route: ModelRoute,
        source: Union[dict, SQLModel, None] = None,
        auto_commit: bool = True,
    ):
        result = await model_route.update(self.session, source, auto_commit=auto_commit)
        await delete_cache_by_key(self.get_model_auth_info_by_name, model_route.name)
        await delete_cache_by_key(
            self.get_model_ids_by_model_route_name, model_route.name
        )
        return result

    async def delete(self, model_route: ModelRoute, auto_commit: bool = True):
        result = await model_route.delete(self.session, auto_commit=auto_commit)
        await delete_cache_by_key(self.get_model_auth_info_by_name, model_route.name)
        await delete_cache_by_key(
            self.get_model_ids_by_model_route_name, model_route.name
        )
        return result


class ModelService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, model_id: int) -> Optional[Model]:
        result = await Model.one_by_id(self.session, model_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_by_name(self, name: str) -> Optional[Model]:
        result = await Model.one_by_field(self.session, "name", name)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def update(self, model: Model, source: Union[dict, SQLModel, None] = None):
        result = await model.update(self.session, source)
        await delete_cache_by_key(self.get_by_id, model.id)
        await delete_cache_by_key(self.get_by_name, model.name)
        return result

    async def delete(self, model: Model):
        result = await model.delete(self.session)
        await delete_cache_by_key(self.get_by_id, model.id)
        await delete_cache_by_key(self.get_by_name, model.name)
        return result


class ModelInstanceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, id: int) -> Optional[ModelInstance]:
        result = await ModelInstance.one_by_id(self.session, id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_running_instances(self, model_id: int) -> List[ModelInstance]:
        results = await ModelInstance.all_by_fields(
            self.session,
            fields={"model_id": model_id, "state": ModelInstanceStateEnum.RUNNING},
        )
        if results is None:
            return []

        for result in results:
            self.session.expunge(result)
        return results

    async def create(self, model_instance):
        result = await ModelInstance.create(self.session, model_instance)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        return result

    async def update(
        self, model_instance: ModelInstance, source: Union[dict, SQLModel, None] = None
    ):
        result = await model_instance.update(self.session, source)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        await delete_cache_by_key(self.get_by_id, model_instance.id)
        return result

    async def delete(self, model_instance: ModelInstance):
        result = await model_instance.delete(self.session)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        await delete_cache_by_key(self.get_by_id, model_instance.id)
        return result

    async def batch_delete(self, model_instances: List[ModelInstance]):
        if not model_instances:
            return []

        names = [mi.name for mi in model_instances]
        ids = set()
        try:
            for m in model_instances:
                await m.delete(self.session, auto_commit=False)
                ids.add(m.model_id)
            await self.session.commit()

            for id in ids:
                await delete_cache_by_key(self.get_running_instances, id)

            return names
        except Exception as e:
            await self.session.rollback()
            raise InternalServerErrorException(
                message=f"Failed to delete model instances {names}: {e}"
            )

    async def batch_update(
        self,
        model_instances: List[ModelInstance],
        source: Union[dict, SQLModel, None] = None,
    ):
        names = [mi.name for mi in model_instances]
        ids = set()
        try:
            for m in model_instances:
                await m.update(self.session, source, auto_commit=False)
                ids.add(m.model_id)
            await self.session.commit()

            for id in ids:
                await delete_cache_by_key(self.get_running_instances, id)

            return names
        except Exception as e:
            await self.session.rollback()
            raise InternalServerErrorException(
                message=f"Failed to update model instances {names}: {e}"
            )


class ModelUsageService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_fields(self, fields: dict) -> ModelUsage:
        result = await ModelUsage.one_by_fields(
            self.session,
            fields=fields,
        )
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def create(self, model_usage: ModelUsage):
        return await ModelUsage.create(self.session, model_usage)

    async def update(
        self,
        model_usage: ModelUsage,
        completion_token_count: int,
        prompt_token_count: int,
        prompt_cached_token_count: int,
        fields: dict,
    ):
        model_usage.completion_token_count += completion_token_count
        model_usage.prompt_token_count += prompt_token_count
        model_usage.prompt_cached_token_count += prompt_cached_token_count
        model_usage.request_count += 1

        key = build_cache_key(self.get_by_fields, fields)
        await set_cache_by_key(key, model_usage)
        usage_flush_buffer[key] = model_usage
        return model_usage


class ModelFileService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_resolved_path(self, path: str) -> List[ModelFile]:
        results = await ModelFile.all_by_fields(
            self.session,
        )
        filtered_results = []
        for result in results:
            self.session.expunge(result)
            if path in result.resolved_paths:
                filtered_results.append(result)

        return filtered_results

    async def get_by_source_index(self, source_index: str) -> List[ModelFile]:
        results = await ModelFile.all_by_field(
            self.session, "source_index", source_index
        )
        if results is None:
            return None

        for result in results:
            self.session.expunge(result)
        return results

    async def create(self, model_file: ModelFile):
        return await ModelFile.create(self.session, model_file)


def intersection_nullable_set(set1: Set[str], set2: Optional[Set[str]]) -> Set[str]:
    if set2 is None:
        return set1
    return set1.intersection(set2)


async def delete_accessible_model_cache(
    *user_ids: int,
):
    for user_id in user_ids:
        await delete_cache_by_key(UserService.get_user_accessible_model_names, user_id)


async def revoke_model_access_cache(
    session: AsyncSession,
    model: Optional[ModelRoute] = None,
    extra_user_ids: Optional[set[int]] = None,
):
    user_ids = set()
    if model is None:
        result = await session.exec(select(User.id))
        user_ids = set(result.all())
    else:
        user_ids = {user.id for user in model.users}
    if extra_user_ids:
        user_ids.update(extra_user_ids)
    await delete_accessible_model_cache(*user_ids)
