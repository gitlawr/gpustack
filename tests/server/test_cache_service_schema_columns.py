"""Cache-service column typing checks.

The cache_services migration creates ``mode`` and ``state`` as VARCHAR.
These tests pin the ORM columns to plain strings so PostgreSQL never
renders native-enum casts (``$1::cacheservicemodeenum``) that the
database has no type for, and so values are stored as the enum values,
not the enum member names.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlmodel import select

from gpustack.schemas.cache_services import (
    CacheService,
    CacheServiceModeEnum,
    CacheServiceStateEnum,
)
from gpustack.schemas.workloads import Workload, WorkloadStateEnum


def test_mode_and_state_columns_are_plain_strings():
    for column in (
        CacheService.__table__.c.mode,
        CacheService.__table__.c.state,
        Workload.__table__.c.state,
        Workload.__table__.c.owner_kind,
        Workload.__table__.c.restart_policy,
        Workload.__table__.c.role,
    ):
        assert not isinstance(column.type, sa.Enum)
        assert isinstance(column.type, sa.String)


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
