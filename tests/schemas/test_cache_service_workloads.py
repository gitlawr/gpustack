"""A cache service's containers, identified and labelled as workload rows.

A provider declares components -- a master, its stores -- and the controller
places one container per (component, worker). None of that is execution
vocabulary: the row says what to run and where, and the two facts a reader has
to get back to the provider with travel as labels under keys the domain owns.
"""

from types import SimpleNamespace

import pytest

from gpustack.schemas.cache_service_workloads import (
    COMPONENT_LABEL,
    DEPENDS_ON_ADDRESS_LABEL,
    DEPENDS_ON_LABEL,
    cache_service_workload_labels,
    cache_service_workload_name,
    component_addresses,
    workload_component,
    workload_depends_on_address,
)


def _row(**labels):
    return SimpleNamespace(labels=labels or None)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_two_components_on_one_worker_are_two_containers():
    """Which is the whole reason the name carries the component: the unique
    constraint is on it, so a shared name would collapse them into one row."""
    master = cache_service_workload_name(7, "master", 3)
    store = cache_service_workload_name(7, "store", 3)

    assert master != store


def test_the_same_component_on_two_workers_is_two_containers():
    assert cache_service_workload_name(7, "store", 3) != cache_service_workload_name(
        7, "store", 4
    )


def test_two_services_do_not_collide_on_one_worker():
    assert cache_service_workload_name(7, "", 3) != cache_service_workload_name(
        8, "", 3
    )


def test_a_provider_without_components_names_its_container_without_one():
    """An empty component is not a component called "": the name has to stay
    what it was before providers could declare any, or every existing
    container is orphaned by the rename."""
    assert cache_service_workload_name(7, "", 3) == "cache-service-7-3"


def test_the_name_is_derived_rather_than_generated():
    """Computable before the row exists, which is what lets a worker
    recognise a container left behind by a row it never saw."""
    assert cache_service_workload_name(7, "store", 3) == cache_service_workload_name(
        7, "store", 3
    )


# ---------------------------------------------------------------------------
# What the row carries back to the domain
# ---------------------------------------------------------------------------


def test_the_component_is_readable_off_the_row():
    labels = cache_service_workload_labels(7, "store", 3)

    assert workload_component(_row(**labels)) == "store"


def test_a_row_with_no_labels_reads_as_no_component():
    """Rather than raising: the readers run over every row of the owner, and
    one written before the labels existed is a row to reconcile, not a
    crash."""
    assert workload_component(SimpleNamespace(labels=None)) == ""
    assert workload_depends_on_address(SimpleNamespace(labels=None)) is None


def test_a_component_that_depends_on_nothing_stamps_no_address():
    """Absent, not empty: an empty address would read as a dependency that
    resolved to nothing, which is what marks a row for replacement."""
    labels = cache_service_workload_labels(7, "master", 3)

    assert DEPENDS_ON_ADDRESS_LABEL not in labels
    assert workload_depends_on_address(_row(**labels)) is None


def test_the_dependency_address_is_readable_off_the_row():
    labels = cache_service_workload_labels(7, "store", 3, "master", "10.0.0.1:8000")

    assert workload_depends_on_address(_row(**labels)) == "10.0.0.1:8000"


def test_the_component_label_is_present_even_when_empty():
    """Told apart from a row whose provider declares no components at all,
    which is what a missing label means."""
    assert cache_service_workload_labels(7, "", 3)[COMPONENT_LABEL] == ""


# ---------------------------------------------------------------------------
# The shape the launch templates read
# ---------------------------------------------------------------------------


def test_the_stamped_address_is_keyed_by_the_component_it_belongs_to():
    """The templates read {{component.<name>.address}}, so the name has to
    come back -- from the provider's declaration, since storing it beside the
    address would let the two disagree."""
    row = _row(
        **cache_service_workload_labels(7, "store", 3, "master", "10.0.0.1:8000")
    )

    assert component_addresses(row) == {"master": "10.0.0.1:8000"}


@pytest.mark.parametrize(
    "depends_on, address",
    [
        (None, "10.0.0.1:8000"),  # nothing declared to depend on
        ("master", None),  # declared, not yet stamped
        (None, None),
    ],
)
def test_an_unresolved_dependency_contributes_no_address(depends_on, address):
    """An unstamped dependency resolves empty rather than leaving its
    placeholder in the command, so the flag carrying it drops with it."""
    labels = cache_service_workload_labels(7, "store", 3, depends_on, address)

    assert component_addresses(_row(**labels)) == {}


def test_the_dependency_name_travels_with_its_address():
    """A reader holding only the row -- the instance list, a watch event --
    has to be able to name what the address points at, and the two are
    written together so they cannot drift apart."""
    labels = cache_service_workload_labels(7, "store", 3, "master", "10.0.0.1:8000")

    assert labels[DEPENDS_ON_LABEL] == "master"
    assert labels[DEPENDS_ON_ADDRESS_LABEL] == "10.0.0.1:8000"


# ---------------------------------------------------------------------------
# Against the providers actually shipped
# ---------------------------------------------------------------------------


def _shipped_providers():
    import pathlib

    import yaml

    return yaml.safe_load(
        pathlib.Path("gpustack/assets/cache-providers.yaml").read_text()
    )


def test_every_shipped_provider_names_its_containers_apart_on_one_worker():
    """The packaged catalog is what this has to hold for. LMCache places its
    coordinator on one worker and a server on every worker, so on that one
    worker both run -- two containers of one service, which the unique
    constraint lets through only because their names differ."""
    for provider in _shipped_providers():
        components = list(provider.get("components") or {"": {}})
        names = {cache_service_workload_name(7, c, 3) for c in components}

        assert len(names) == len(components), provider.get("name")


def test_the_shipped_dependency_survives_the_round_trip():
    """LMCache's server depends on its coordinator; the address is stamped on
    the server's row and has to come back keyed by the name the template
    reads it under."""
    lmcache = next(p for p in _shipped_providers() if p.get("name") == "LMCache")
    server = lmcache["components"]["server"]
    depends_on = server["depends_on"]

    row = _row(
        **cache_service_workload_labels(7, "server", 3, depends_on, "10.0.0.1:8100")
    )

    assert component_addresses(row) == {"coordinator": "10.0.0.1:8100"}
