"""
A managed cache service's containers, as workload rows.

A provider declares components -- a master, its stores, a standalone server --
and the controller places one container per (component, worker). Which
component a container runs, and the address of the one it depends on, are the
provider's vocabulary, not the execution layer's: a workload row says what to
run and where, and a reader that needs to know it is a store asks the domain.

So they travel as labels, under keys this module owns. The row stays a row.
"""

from typing import Any, Dict, Optional

COMPONENT_LABEL = "cache-service.gpustack.ai/component"
"""Which of the provider's declared components this container runs. Present
even when empty, so a row that predates a component-aware provider is told
apart from one whose provider declares none."""

DEPENDS_ON_LABEL = "cache-service.gpustack.ai/depends-on"
"""Which component the address below belongs to. Stored beside it rather than
resolved from the catalog, so a reader holding only the row -- the instance
list, a watch event -- can name it. The two are written in one go and read in
one go, so they cannot drift apart."""

DEPENDS_ON_ADDRESS_LABEL = "cache-service.gpustack.ai/depends-on-address"
"""The address of the component this one depends on, as resolved when the row
was created -- a store's master, say.

Stamped rather than looked up on the worker: the server knows the dependency's
placement, and the running process bakes the address into its config, so a
stamp that stops matching the current address is what marks the row for
replacement. Absent when the component depends on nothing, or when its
dependency is turned off and will never answer."""


def cache_service_workload_name(service_id: int, component: str, worker_id: int) -> str:
    """The container name for one component of one service on one worker.

    Derived from the identity rather than generated, so it can be computed
    before the row exists -- which is what lets a leftover container be
    recognised and reclaimed by a worker that never saw the row.
    """
    suffix = f"-{component}" if component else ""
    return f"cache-service-{service_id}{suffix}-{worker_id}"


def cache_service_workload_labels(
    service_id: int,
    component: str,
    worker_id: int,
    depends_on: Optional[str] = None,
    depends_on_address: Optional[str] = None,
) -> Dict[str, str]:
    """What the domain needs stamped on the row to read it back."""
    labels = {
        "cache-service.gpustack.ai/service-id": str(service_id),
        "cache-service.gpustack.ai/worker-id": str(worker_id),
        COMPONENT_LABEL: component,
    }
    if depends_on and depends_on_address is not None:
        labels[DEPENDS_ON_LABEL] = depends_on
        labels[DEPENDS_ON_ADDRESS_LABEL] = depends_on_address
    return labels


def workload_component(workload: Any) -> str:
    """Which component a row runs, "" for a provider that declares none."""
    return (getattr(workload, "labels", None) or {}).get(COMPONENT_LABEL, "")


def workload_depends_on_address(workload: Any) -> Optional[str]:
    """The dependency address stamped on a row, None when it carries none."""
    return (getattr(workload, "labels", None) or {}).get(DEPENDS_ON_ADDRESS_LABEL)


def component_addresses(workload: Any) -> Dict[str, str]:
    """The stamped addresses in the shape the launch templates read.

    Empty when the component depends on nothing, or when its dependency has
    not come up yet -- the template then resolves empty and the flag carrying
    it drops, rather than leaving a placeholder in the command.
    """
    labels = getattr(workload, "labels", None) or {}
    name = labels.get(DEPENDS_ON_LABEL)
    address = labels.get(DEPENDS_ON_ADDRESS_LABEL)
    if not name or address is None:
        return {}
    return {name: address}


def instance_public_from_workload(workload: Any, address_port_name: str = "port"):
    """A cache service's container as the instances API has always shown it.

    The API predates the rows and does not change with them.

    ``address_port_name`` is what the component's declaration calls the port
    it is addressed by; the caller resolves it, because which of a row's ports
    that is only the provider's catalog knows.
    """
    from gpustack.schemas.cache_services import (
        CacheServiceInstancePublic,
        CacheServiceStateEnum,
    )

    component = workload_component(workload)
    ports = workload.ports or {}
    return CacheServiceInstancePublic(
        id=workload.id,
        name=workload.name,
        cache_service_id=workload.owner_id,
        worker_id=workload.worker_id,
        cluster_id=workload.cluster_id,
        component=component,
        component_addresses=component_addresses(workload) or None,
        ports=ports or None,
        port=ports.get(address_port_name),
        state=CacheServiceStateEnum(workload.state.value),
        state_message=workload.state_message,
        healthy=workload.healthy,
        last_check_at=workload.last_check_at,
        restart_count=workload.restart_count,
        last_restart_time=workload.last_restart_time,
        spec_digest=workload.spec_digest,
        created_at=workload.created_at,
        updated_at=workload.updated_at,
    )


def deployment_metadata_from_workload(workload: Any):
    """What the runtime is asked to create the container under.

    The row's own name is the container's, so there is nothing to derive:
    a name keyed by the row id could not be computed before the row existed,
    and the orphan sweep needs it to be.
    """
    from gpustack.schemas.cache_services import (
        CACHE_SERVICE_WORKLOAD_TYPE,
        CacheServiceDeploymentMetadata,
    )

    return CacheServiceDeploymentMetadata(
        name=workload.name,
        labels={
            "type": CACHE_SERVICE_WORKLOAD_TYPE,
            "cache-service-id": str(workload.owner_id),
            "cache-service-instance-id": str(workload.id),
        },
    )
