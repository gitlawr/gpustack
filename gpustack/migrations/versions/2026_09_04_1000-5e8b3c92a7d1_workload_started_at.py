"""workload started_at

Adds ``workloads.started_at``: when a workload's container began running, as
opposed to when its row was created.

``active_deadline_seconds`` has to be measured from something, and for a task
that sat queued behind another the row's creation time is not it. Without this
the deadline cannot be enforced from the row at all, which is what the
benchmark migration surfaced.

Revision ID: 5e8b3c92a7d1
Revises: 9d2c6ab410f5
Create Date: 2026-09-04 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5e8b3c92a7d1'
down_revision: Union[str, None] = '9d2c6ab410f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('workloads', sa.Column('started_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('workloads', 'started_at')
