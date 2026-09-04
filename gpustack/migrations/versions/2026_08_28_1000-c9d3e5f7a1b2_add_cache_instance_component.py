"""add cache instance component

Adds the ``component`` column to ``cache_service_instances``: which of
the provider's declared components (e.g. Mooncake's master / store) the
instance runs; single-component providers store the empty string. The
(service, worker) uniqueness goes with it: components of one service
may share a worker, and a replicas component may place several of its
own instances on one worker whose RAM holds them.

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


def downgrade() -> None:
    op.create_unique_constraint(
        _OLD_UNIQUE,
        _TABLE,
        ["cache_service_id", "worker_id"],
    )
    op.drop_column(_TABLE, "component_addresses")
    op.drop_column(_TABLE, "component")
