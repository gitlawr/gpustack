"""add ORG to access_policy enum

Splits out the ORG enum value from the foundation/principals migration
so devs already on revision 3a7e2c91d5b4 pick up the new value via a
fresh upgrade. The view itself doesn't need recreating — it's rebuilt
from `model_user_after_create_view_stmt` whenever its sources change,
and reads new enum values transparently.

Revision ID: c1f2a4b67d0e
Revises: 3a7e2c91d5b4
Create Date: 2026-04-28 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import gpustack.utils.sql_enum as sql_enum
from gpustack.schemas.stmt import (
    model_user_after_create_view_stmt,
    model_user_after_drop_view_stmt,
)


# revision identifiers, used by Alembic.
revision: str = 'c1f2a4b67d0e'
down_revision: Union[str, None] = '3a7e2c91d5b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


access_policy_enum = sa.Enum(
    'PUBLIC',
    'AUTHED',
    'ALLOWED_USERS',
    'ALLOWED_PRINCIPALS',
    name='accesspolicyenum',
)
access_policy_to_add = ['ORG']


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == 'postgresql':
        for value in access_policy_to_add:
            op.execute(
                f"ALTER TYPE accesspolicyenum ADD VALUE IF NOT EXISTS '{value}'"
            )
    else:
        sql_enum.add_enum_values(
            {'model_routes': 'access_policy'},
            access_policy_enum,
            *access_policy_to_add,
        )

    # Recreate the view so the new ORG branch in `stmt.py` takes effect.
    op.execute(model_user_after_drop_view_stmt)
    op.execute(model_user_after_create_view_stmt(bind.dialect.name))


def downgrade() -> None:
    # Postgres can't cleanly drop a single enum value; leaving ORG in
    # place is harmless. The view recreation on downgrade falls back to
    # whatever stmt.py defines at that point.
    pass
