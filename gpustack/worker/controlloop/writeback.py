"""
Writing a resource's state back to the API server from a worker.

NOTE: this is read-modify-write — fetch the row, set the fields, PUT it whole —
because that is what the generated clients offer. Two writers racing on
different fields of the same row therefore lose one of the writes. The
benchmark manager avoids this by PATCHing a dedicated state endpoint; giving
model instances and cache service instances the same endpoint would retire this
shape, and is the reason it is centralised here rather than left in three
managers.
"""

import logging
from typing import Any, Callable, Type

from gpustack.api.exceptions import NotFoundException
from gpustack.utils.attrs import set_attr

logger = logging.getLogger(__name__)


def update_resource(
    client: Any,
    id: int,
    update_cls: Type,
    description: str,
    **fields,
) -> bool:
    """
    Apply the given fields to a resource.

    Args:
        client: The generated client for the resource kind, offering ``get``
            and ``update``.
        id: The ID of the resource to update.
        update_cls: The update model to build the request body from.
        description: The resource kind, for log messages.
        **fields: The fields to set, by name.

    Returns:
        Whether the update was applied. A failed write-back is reported rather
        than raised: callers run on the watch event loop, a sync thread, or a
        subprocess about to exit, where an exception would be dropped, and the
        reconcile pass re-drives what the lost update would have set.
    """
    try:
        current = client.get(id=id)

        update = update_cls(**current.model_dump())
        for key, value in fields.items():
            set_attr(update, key, value)

        client.update(id=id, model_update=update)
        return True
    except NotFoundException:
        logger.warning(f"{description} with ID {id} not found when trying to update.")
        return False
    except Exception as e:
        logger.error(f"Failed to update {description.lower()} {id}: {e}")
        return False


def updater(
    client_getter: Callable[[], Any],
    update_cls: Type,
    description: str,
) -> Callable[..., bool]:
    """
    Bind :func:`update_resource` to one resource kind.

    The client is fetched per call rather than captured, because a worker
    re-registering replaces its clientset.
    """

    def update(id: int, **fields) -> bool:
        return update_resource(
            client_getter(),
            id,
            update_cls,
            description,
            **fields,
        )

    return update
