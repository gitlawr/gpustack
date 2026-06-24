"""add cas auth provider

Revision ID: f4a1c9e2b7d8
Revises: b2c3d4e5f6a7
Create Date: 2026-06-24 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'f4a1c9e2b7d8'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CAS_VALUE = "CAS"

# Both a principal and its group-membership rows carry an auth-source
# enum; CAS-provisioned users and their synced memberships use
# source='CAS'. On PostgreSQL the two columns share one ``authproviderenum``
# type, so a single ADD VALUE covers both; on MySQL the enum is per-column
# and each must be widened.
_CAS_COLUMNS = [("principals", "source"), ("principal_memberships", "source")]


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == 'postgresql':
        conn.execute(
            sa.text(f"ALTER TYPE authproviderenum ADD VALUE '{CAS_VALUE}'")
        )
    elif conn.dialect.name == 'mysql':
        for table, column in _CAS_COLUMNS:
            op.alter_column(
                table,
                column,
                existing_type=sa.Enum(
                    'Local', 'OIDC', 'SAML', name='authproviderenum'
                ),
                type_=sa.Enum(
                    'Local', 'OIDC', 'SAML', CAS_VALUE, name='authproviderenum'
                ),
                existing_nullable=False,
                existing_server_default=sa.text("'Local'"),
            )


def downgrade() -> None:
    # PostgreSQL cannot remove a value from an enum type, so the type is
    # left carrying 'CAS'. On MySQL the column enums are shrunk back,
    # which fails if any CAS-sourced rows still exist — delete them first.
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        for table, column in _CAS_COLUMNS:
            op.alter_column(
                table,
                column,
                existing_type=sa.Enum(
                    'Local', 'OIDC', 'SAML', CAS_VALUE, name='authproviderenum'
                ),
                type_=sa.Enum(
                    'Local', 'OIDC', 'SAML', name='authproviderenum'
                ),
                existing_nullable=False,
                existing_server_default=sa.text("'Local'"),
            )
