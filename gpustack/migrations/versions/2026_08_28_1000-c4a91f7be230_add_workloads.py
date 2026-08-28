"""add workloads

Introduces the generic ``workloads`` table: one row per container a worker
runs, carrying execution semantics only -- the binding a scheduler or a
controller decided on, the restart policy, and the execution state. The
user-facing resources (model deployments, benchmarks, cache services) compile
into it and aggregate execution state back out of it.

``owner_kind``/``owner_id`` name the domain resource a workload was compiled
from. Deliberately not a foreign key: the target table depends on
``owner_kind``, so the controller owns the lifetime instead of the database.

Nothing consumes the table yet; the cache service instances that will be its
first consumer still have their own table at this revision.

See docs/proposals/workload-resource.md.

Revision ID: c4a91f7be230
Revises: b7e2c4d15a80
Create Date: 2026-08-28 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'c4a91f7be230'
down_revision: Union[str, None] = 'b7e2c4d15a80'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'workloads',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        # Ownership.
        sa.Column('owner_kind', sa.String(length=64), nullable=False),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('owner_principal_id', sa.Integer(), nullable=True),
        sa.Column('cluster_id', sa.Integer(), nullable=True),
        # Grouping, for one distributed instance spanning several workloads.
        sa.Column('group_key', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('group_index', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=32), nullable=False),
        # Binding result.
        sa.Column('worker_id', sa.Integer(), nullable=True),
        sa.Column('gpu_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('gpu_indexes', sa.JSON(), nullable=True),
        sa.Column('gpu_addresses', sa.JSON(), nullable=True),
        sa.Column('computed_resource_claim', sa.JSON(), nullable=True),
        sa.Column('reserved_claims', sa.JSON(), nullable=True),
        # Spec.
        sa.Column('restart_policy', sa.String(length=32), nullable=False),
        sa.Column('active_deadline_seconds', sa.Integer(), nullable=True),
        sa.Column('spec_digest', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('labels', sa.JSON(), nullable=True),
        # Execution status.
        sa.Column('state', sa.String(length=64), nullable=False),
        sa.Column('state_message', sa.Text(), nullable=True),
        sa.Column('ports', sa.JSON(), nullable=True),
        sa.Column('pid', sa.Integer(), nullable=True),
        sa.Column('arguments', sa.JSON(), nullable=True),
        sa.Column('restart_count', sa.Integer(), nullable=False),
        sa.Column('last_restart_time', sa.DateTime(), nullable=True),
        sa.Column('healthy', sa.Boolean(), nullable=True),
        sa.Column('last_check_at', sa.DateTime(), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        # One workload per owner per worker per group position. Without it a
        # controller fan-out running twice concurrently creates duplicates,
        # which only shows up under load.
        sa.UniqueConstraint(
            'owner_kind',
            'owner_id',
            'worker_id',
            'group_index',
            name='uix_workloads_owner_worker_group_index',
        ),
    )
    # The worker's reconcile pass and orphan cleanup, the hottest read.
    op.create_index('ix_workloads_worker_id', 'workloads', ['worker_id'])
    # Controller fan-out and state aggregation.
    op.create_index('ix_workloads_owner', 'workloads', ['owner_kind', 'owner_id'])
    # Endpoint resolution runs this for every model instance scheduled, and
    # filters on state, so it is worth the third column.
    op.create_index(
        'ix_workloads_owner_state', 'workloads', ['owner_kind', 'owner_id', 'state']
    )
    op.create_index('ix_workloads_cluster_id', 'workloads', ['cluster_id'])
    op.create_index('ix_workloads_group_key', 'workloads', ['group_key'])
    op.create_index('ix_workloads_name', 'workloads', ['name'])


def downgrade() -> None:
    op.drop_index('ix_workloads_name', table_name='workloads')
    op.drop_index('ix_workloads_group_key', table_name='workloads')
    op.drop_index('ix_workloads_cluster_id', table_name='workloads')
    op.drop_index('ix_workloads_owner_state', table_name='workloads')
    op.drop_index('ix_workloads_owner', table_name='workloads')
    op.drop_index('ix_workloads_worker_id', table_name='workloads')
    op.drop_table('workloads')
