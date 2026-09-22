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
    depends_on_address: Optional[str] = None,
) -> Dict[str, str]:
    """What the domain needs stamped on the row to read it back."""
    labels = {
        "cache-service.gpustack.ai/service-id": str(service_id),
        "cache-service.gpustack.ai/worker-id": str(worker_id),
        COMPONENT_LABEL: component,
    }
    if depends_on_address is not None:
        labels[DEPENDS_ON_ADDRESS_LABEL] = depends_on_address
    return labels


def workload_component(workload: Any) -> str:
    """Which component a row runs, "" for a provider that declares none."""
    return (getattr(workload, "labels", None) or {}).get(COMPONENT_LABEL, "")


def workload_depends_on_address(workload: Any) -> Optional[str]:
    """The dependency address stamped on a row, None when it carries none."""
    return (getattr(workload, "labels", None) or {}).get(DEPENDS_ON_ADDRESS_LABEL)


def component_addresses(workload: Any, depends_on: Optional[str]) -> Dict[str, str]:
    """The stamped addresses in the shape the launch templates read.

    ``depends_on`` comes from the provider's declaration for this component;
    the row carries only the address, since the name is derivable and storing
    both would let them disagree.
    """
    address = workload_depends_on_address(workload)
    if not depends_on or address is None:
        return {}
    return {depends_on: address}
