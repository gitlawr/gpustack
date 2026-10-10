"""Tenant-aware entry point for resource watch streams."""

from inspect import isawaitable
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterable,
    List,
    Optional,
    Union,
)

from gpustack.api.tenant import TenantContext, tenant_stream_filter
from gpustack.server.bus import Event

StreamFilter = Callable[[Any], Union[bool, Awaitable[bool]]]


def _combine_filters(visibility: StreamFilter, business: StreamFilter) -> StreamFilter:
    async def after_visibility(result: Awaitable[bool], data: Any) -> bool:
        if not await result:
            return False
        matched = business(data)
        return await matched if isawaitable(matched) else matched

    def matches(data: Any) -> Union[bool, Awaitable[bool]]:
        visible = visibility(data)
        if isawaitable(visible):
            return after_visibility(visible, data)
        return business(data) if visible else False

    return matches


def tenant_streaming(
    model: Any,
    ctx: TenantContext,
    *,
    visibility_filter: Optional[StreamFilter] = None,
    fields: Optional[dict] = None,
    fuzzy_fields: Optional[dict] = None,
    filter_func: Optional[StreamFilter] = None,
    options: Optional[List] = None,
    event_transform: Optional[Callable[[Event], Awaitable[None]]] = None,
    authorized_replay_loader: Optional[Callable[[], Awaitable[Iterable[Any]]]] = None,
) -> AsyncGenerator[str, None]:
    """Stream resource events within an explicit tenant context.

    Args:
        model: Resource model providing the underlying event stream.
        ctx: Resolved caller context; required even for custom visibility.
        visibility_filter: Custom authorization for shared or parent-owned
            resources. Defaults to owner/cluster scoping for owner-bearing models.
        fields: Exact-match business filters.
        fuzzy_fields: Fuzzy-match business filters.
        filter_func: Additional business predicate, ANDed with authorization.
        options: Relationship loading options for the initial replay.
        event_transform: Public event enrichment after authorization.
        authorized_replay_loader: Snapshot query that enforces tenant visibility
            itself. Runs after subscribing to the bus. Its rows only need the
            business filters; live events still use visibility_filter.

    Returns:
        Serialized events filtered before public projection, for both replay
        and subsequent bus events. Async predicates can consult parent resources.
    """
    if ctx is None:
        raise ValueError("Tenant watch streams require a tenant context")
    if visibility_filter is None:
        if not hasattr(model, "owner_principal_id"):
            raise ValueError(f"{model.__name__} requires an explicit visibility filter")
        visibility_filter = tenant_stream_filter(ctx, model)
    if filter_func is not None:
        visibility_filter = _combine_filters(visibility_filter, filter_func)
    replay_options = {}
    if authorized_replay_loader is not None:
        replay_options = {
            "replay_loader": authorized_replay_loader,
            "replay_filter_func": filter_func or (lambda data: True),
        }
    return model.streaming(
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        filter_func=visibility_filter,
        options=options,
        event_transform=event_transform,
        **replay_options,
    )
