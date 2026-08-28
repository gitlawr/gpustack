"""add cache instance component

Adds the ``component`` column to ``cache_service_instances``: which of
the provider's declared components (e.g. Mooncake's master / store) the
instance runs. Single-component providers store the empty string — a
real value, not NULL, so it can participate in the widened uniqueness
below (NULLs compare distinct inside unique constraints on every
supported database). Components of one service may share a worker, so
the (service, worker) uniqueness gains the component column.

Revision ID: c9d3e5f7a1b2
Revises: b7e2c4d15a80
Create Date: 2026-08-28 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c9d3e5f7a1b2'
down_revision: Union[str, None] = 'b7e2c4d15a80'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "cache_service_instances"
_OLD_UNIQUE = "uix_cache_service_instances_service_worker"
_NEW_UNIQUE = "uix_cache_service_instances_service_worker_component"


def upgrade() -> None:
    op.add_column(
        _TABLE,
        sa.Column(
            "component",
            sa.String(length=64),
            nullable=False,
            server_default="",
        ),
    )
    op.add_column(
        _TABLE,
        sa.Column("component_addresses", sa.JSON(), nullable=True),
    )
    op.drop_constraint(_OLD_UNIQUE, _TABLE, type_="unique")
    op.create_unique_constraint(
        _NEW_UNIQUE,
        _TABLE,
        ["cache_service_id", "worker_id", "component"],
    )


def downgrade() -> None:
    op.drop_constraint(_NEW_UNIQUE, _TABLE, type_="unique")
    op.create_unique_constraint(
        _OLD_UNIQUE,
        _TABLE,
        ["cache_service_id", "worker_id"],
    )
    op.drop_column(_TABLE, "component_addresses")
    op.drop_column(_TABLE, "component")
