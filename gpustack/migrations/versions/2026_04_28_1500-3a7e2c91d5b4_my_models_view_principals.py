"""recreate non_admin_user_models view to support ALLOWED_PRINCIPALS

The view drives /my-models visibility for non-admin users. Until now it joined
only `usermodelroutelink` (legacy ALLOWED_USERS path). After P0 we have
`model_route_principals` as the new principal-based path; this migration
rewrites the view so routes with `access_policy='ALLOWED_PRINCIPALS'` become
visible to users that match by USER, ORG (organization_memberships) or GROUP
(user_group_memberships).

Revision ID: 3a7e2c91d5b4
Revises: 7c5e3f9a2d18
Create Date: 2026-04-28 15:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import gpustack.utils.sql_enum as sql_enum
from gpustack.schemas.stmt import (
    model_user_after_create_view_stmt,
    model_user_after_drop_view_stmt,
)


access_policy_enum = sa.Enum(
    'PUBLIC', 'AUTHED', 'ALLOWED_USERS', name='accesspolicyenum'
)
access_policy_to_add = ['ALLOWED_PRINCIPALS']


# revision identifiers, used by Alembic.
revision: str = '3a7e2c91d5b4'
down_revision: Union[str, None] = '7c5e3f9a2d18'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Old view definition, restored on downgrade.
def _old_view_stmt(db_type: str) -> str:
    sql_false = '0' if db_type == "sqlite" else 'FALSE'
    pid = (
        "CONCAT(m.id, ':', u.id)"
        if db_type == "mysql"
        else "CAST(m.id AS TEXT) || ':' || CAST(u.id AS TEXT)"
    )
    return f'''
CREATE VIEW non_admin_user_models AS
SELECT
    {pid} AS pid,
    u.id AS user_id,
    m.*
FROM
    users u
INNER JOIN model_routes as m
    ON m.access_policy in ('PUBLIC', 'AUTHED')
    OR EXISTS (
        SELECT 1 FROM usermodelroutelink uml
        WHERE uml.route_id = m.id AND uml.user_id = u.id
    )
WHERE
    u.is_admin = {sql_false} AND u.is_system = {sql_false}
'''


def upgrade() -> None:
    bind = op.get_bind()
    # Add ALLOWED_PRINCIPALS to the access_policy enum on dialects with native
    # enum types (postgres, mysql); sqlite stores it as plain text.
    if bind.dialect.name == 'postgresql':
        # Idempotent: only add if not already present.
        op.execute(
            "ALTER TYPE accesspolicyenum ADD VALUE IF NOT EXISTS 'ALLOWED_PRINCIPALS'"
        )
    else:
        sql_enum.add_enum_values(
            {'model_routes': 'access_policy'},
            access_policy_enum,
            *access_policy_to_add,
        )
    # Drop the existing view and recreate with the new principal-aware
    # definition.
    op.execute(model_user_after_drop_view_stmt)
    op.execute(model_user_after_create_view_stmt(bind.dialect.name))


def downgrade() -> None:
    bind = op.get_bind()
    # Recreate the legacy view definition. We deliberately do NOT remove the
    # 'ALLOWED_PRINCIPALS' enum value — postgres cannot drop a single enum
    # value cleanly when other columns reference the type, and leaving an
    # unused enum value is harmless. If a downstream operator needs to fully
    # revert, they should drop the column default and recreate the type
    # manually.
    op.execute(model_user_after_drop_view_stmt)
    op.execute(_old_view_stmt(bind.dialect.name))
