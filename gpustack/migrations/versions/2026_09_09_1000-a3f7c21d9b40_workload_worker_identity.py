"""workload worker identity

Adds ``workloads.worker_name`` / ``worker_ip`` / ``worker_ifname``: the worker
as the container needs to name it, alongside the id the row already carries.

The readers that need these are on the worker -- a backend building an argument
vector for a distributed run needs its peers' addresses, and it has no session
to join ``workers`` with. That is why the embedded
``distributed_servers.subordinate_workers`` denormalises them today, and moving
those readers onto workload rows requires the same.

Backfilled by the controller rather than here: they belong to the spec, so
every reconcile of a model instance rewrites them, and a row compiled before
this migration is brought up to date on its owner's next event. That also makes
them fresher than the embedded copy, which is only written when the binding is.

Revision ID: a3f7c21d9b40
Revises: 5e8b3c92a7d1
Create Date: 2026-09-09 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = 'a3f7c21d9b40'
down_revision: Union[str, None] = '5e8b3c92a7d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for name in ('worker_name', 'worker_ip', 'worker_ifname'):
        op.add_column(
            'workloads',
            sa.Column(name, sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        )


def downgrade() -> None:
    op.drop_column('workloads', 'worker_ifname')
    op.drop_column('workloads', 'worker_ip')
    op.drop_column('workloads', 'worker_name')
