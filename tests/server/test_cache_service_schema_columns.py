"""Cache-service and workload enum column typing.

The migrations create these columns as VARCHAR. These tests pin them to
that, so PostgreSQL never renders a native-enum cast
(``$1::cacheservicemodeenum``) for a type the database does not have, and
so values are stored as the enum values rather than the member names.
``EnumString`` converts on the way back out; the storage stays VARCHAR.
"""

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlmodel import select

from gpustack.schemas.cache_services import (
    CacheService,
    CacheServiceModeEnum,
    CacheServiceStateEnum,
)
from gpustack.schemas.workloads import Workload, WorkloadStateEnum

ENUM_COLUMNS = [
    CacheService.__table__.c.mode,
    CacheService.__table__.c.state,
    Workload.__table__.c.state,
    Workload.__table__.c.owner_kind,
    Workload.__table__.c.restart_policy,
    Workload.__table__.c.role,
]


@pytest.mark.parametrize(
    "column", ENUM_COLUMNS, ids=lambda c: f"{c.table.name}.{c.name}"
)
def test_enum_columns_are_stored_as_varchar(column):
    """Asserted on the rendered DDL rather than the Python type: what matters
    is the column the database gets, and the type that produces it converts to
    an enum on load."""
    assert not isinstance(column.type, sa.Enum)
    assert "VARCHAR" in column.type.compile(dialect=postgresql.dialect())


def test_enum_filters_compile_to_string_binds_on_postgresql():
    statement = select(CacheService).where(
        CacheService.mode == CacheServiceModeEnum.EXTERNAL,
        CacheService.state == CacheServiceStateEnum.RUNNING,
    )
    compiled = statement.compile(
        dialect=postgresql.asyncpg.dialect(),
        compile_kwargs={"render_postcompile": True},
    )

    assert "::cacheservicemodeenum" not in str(compiled)
    assert "::cacheservicestateenum" not in str(compiled)
    assert "external" in compiled.params.values()
    assert "running" in compiled.params.values()


def test_instance_enum_filters_compile_to_string_binds_on_postgresql():
    statement = select(Workload).where(
        Workload.state == WorkloadStateEnum.RUNNING,
    )
    compiled = statement.compile(
        dialect=postgresql.asyncpg.dialect(),
        compile_kwargs={"render_postcompile": True},
    )

    assert "::workloadstateenum" not in str(compiled)
    assert "running" in compiled.params.values()
