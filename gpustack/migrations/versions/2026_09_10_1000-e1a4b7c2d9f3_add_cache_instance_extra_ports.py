"""add cache instance extra ports

Adds ``extra_ports`` to ``cache_service_instances``: the ports allocated
for the names a component declares beyond the service and metrics ports
the platform always hands out (port name -> port). Recorded per instance
so a restart keeps the ports it already published to peers.

Revision ID: e1a4b7c2d9f3
Revises: c9d3e5f7a1b2
Create Date: 2026-09-10 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'e1a4b7c2d9f3'
down_revision: Union[str, None] = 'c9d3e5f7a1b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "cache_service_instances"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("extra_ports", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column(_TABLE, "extra_ports")
