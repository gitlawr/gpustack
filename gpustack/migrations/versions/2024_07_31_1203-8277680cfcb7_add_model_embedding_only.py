"""add model embedding only

Revision ID: 8277680cfcb7
Revises: 1dd9fa5b38ff
Create Date: 2024-07-31 12:03:41.325109

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = '8277680cfcb7'
down_revision: Union[str, None] = '1dd9fa5b38ff'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('models', sa.Column('embedding_only', sa.Boolean(), nullable=True))
    op.execute('UPDATE models SET embedding_only = False')
    with op.batch_alter_table('models') as batch_op:
        batch_op.alter_column('embedding_only', nullable=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('models') as batch_op:
        batch_op.alter_column('embedding_only', nullable=True)
    op.drop_column('models', 'embedding_only')
    # ### end Alembic commands ###
