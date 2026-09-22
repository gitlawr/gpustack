"""cache services run as workloads

Drops ``cache_service_instances``: a managed cache service's servers are now
rows in ``workloads``, owned by ``owner_kind='cache_service'``.

Not a data migration. The cache service feature is unreleased, so there is
nothing in the table worth carrying over, and the container names change with
it -- they are keyed by worker rather than by row id now, since that is the
identity and it can be computed before the row exists. Any container left from
before is removed by the worker's orphan cleanup.

Revision ID: 9d2c6ab410f5
Revises: c4a91f7be230
Create Date: 2026-08-28 11:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = '9d2c6ab410f5'
down_revision: Union[str, None] = 'c4a91f7be230'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table('cache_service_instances')


def downgrade() -> None:
    op.create_table(
        'cache_service_instances',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('cache_service_id', sa.Integer(), nullable=False),
        sa.Column('worker_id', sa.Integer(), nullable=False),
        sa.Column('cluster_id', sa.Integer(), nullable=False),
        sa.Column('port', sa.Integer(), nullable=True),
        sa.Column('metrics_port', sa.Integer(), nullable=True),
        sa.Column('state', sa.String(length=64), nullable=False),
        sa.Column('state_message', sa.Text(), nullable=True),
        sa.Column('healthy', sa.Boolean(), nullable=True),
        sa.Column('last_check_at', sa.DateTime(), nullable=True),
        sa.Column('restart_count', sa.Integer(), nullable=True),
        sa.Column('last_restart_time', sa.DateTime(), nullable=True),
        sa.Column('spec_digest', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.ForeignKeyConstraint(
            ['cache_service_id'], ['cache_services.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'cache_service_id',
            'worker_id',
            name='uix_cache_service_instances_service_worker',
        ),
    )
    op.create_index(
        op.f('ix_cache_service_instances_cache_service_id'),
        'cache_service_instances',
        ['cache_service_id'],
    )
    op.create_index(
        op.f('ix_cache_service_instances_name'), 'cache_service_instances', ['name']
    )
