"""multi-tenancy foundation: organizations, groups, memberships, access, namespaces, quotas

Revision ID: 7c5e3f9a2d18
Revises: 8bf38a6bb3b5
Create Date: 2026-04-28 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import gpustack
from gpustack.migrations.utils import column_exists, table_exists


# revision identifiers, used by Alembic.
revision: str = '7c5e3f9a2d18'
down_revision: Union[str, None] = '8bf38a6bb3b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PLATFORM_ORG_ID = 1
PLATFORM_ORG_SLUG = 'default'
PLATFORM_ORG_NAME = 'Default'


def _enums():
    org_role = sa.Enum('OWNER', 'MANAGER', 'MEMBER', name='orgrole')
    principal_type = sa.Enum('ORG', 'GROUP', 'USER', name='principaltype')
    tenant_ns_state = sa.Enum(
        'PENDING', 'PROVISIONING', 'READY', 'ERROR', 'DELETING',
        name='tenantnamespacestate',
    )
    return org_role, principal_type, tenant_ns_state


def upgrade() -> None:
    bind = op.get_bind()
    org_role, principal_type, tenant_ns_state = _enums()

    # On postgres, enum types are created lazily by the first column referencing
    # them; subsequent columns reuse the existing type since SQLAlchemy tracks
    # the same enum instance.

    # ---- New tables ------------------------------------------------------

    if not table_exists('organizations'):
        op.create_table(
            'organizations',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('slug', sa.String(length=255), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('billing_account_ref', sa.String(length=255), nullable=True),
            sa.Column('is_platform', sa.Boolean(), nullable=False),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('slug', name='uix_organizations_slug'),
        )

    if not table_exists('organization_memberships'):
        op.create_table(
            'organization_memberships',
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('organization_id', sa.Integer(), nullable=False),
            sa.Column('role', org_role, nullable=False),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.ForeignKeyConstraint(
                ['user_id'], ['users.id'], ondelete='CASCADE',
            ),
            sa.ForeignKeyConstraint(
                ['organization_id'], ['organizations.id'], ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('user_id', 'organization_id'),
        )

    if not table_exists('user_groups'):
        op.create_table(
            'user_groups',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('organization_id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
            sa.ForeignKeyConstraint(
                ['organization_id'], ['organizations.id'], ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint(
                'organization_id', 'name', name='uix_user_groups_org_id_name'
            ),
        )

    if not table_exists('user_group_memberships'):
        op.create_table(
            'user_group_memberships',
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('group_id', sa.Integer(), nullable=False),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.ForeignKeyConstraint(
                ['user_id'], ['users.id'], ondelete='CASCADE',
            ),
            sa.ForeignKeyConstraint(
                ['group_id'], ['user_groups.id'], ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('user_id', 'group_id'),
        )

    if not table_exists('cluster_access'):
        op.create_table(
            'cluster_access',
            sa.Column('cluster_id', sa.Integer(), nullable=False),
            sa.Column('principal_type', principal_type, nullable=False),
            sa.Column('principal_id', sa.Integer(), nullable=False),
            sa.Column('granted_by', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.ForeignKeyConstraint(
                ['cluster_id'], ['clusters.id'], ondelete='CASCADE',
            ),
            sa.ForeignKeyConstraint(
                ['granted_by'], ['users.id'], ondelete='SET NULL',
            ),
            sa.PrimaryKeyConstraint(
                'cluster_id', 'principal_type', 'principal_id'
            ),
        )

    if not table_exists('tenant_namespaces'):
        op.create_table(
            'tenant_namespaces',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('cluster_id', sa.Integer(), nullable=False),
            sa.Column('organization_id', sa.Integer(), nullable=False),
            sa.Column('namespace_name', sa.String(length=255), nullable=False),
            sa.Column('state', tenant_ns_state, nullable=False),
            sa.Column('state_message', sa.Text(), nullable=True),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
            sa.ForeignKeyConstraint(
                ['cluster_id'], ['clusters.id'], ondelete='CASCADE',
            ),
            sa.ForeignKeyConstraint(
                ['organization_id'], ['organizations.id'], ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint(
                'cluster_id', 'organization_id', name='uix_tenant_ns_cluster_org'
            ),
            sa.UniqueConstraint(
                'cluster_id', 'namespace_name', name='uix_tenant_ns_cluster_name'
            ),
        )

    if not table_exists('tenant_quotas'):
        op.create_table(
            'tenant_quotas',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('cluster_id', sa.Integer(), nullable=False),
            sa.Column('organization_id', sa.Integer(), nullable=False),
            sa.Column('gpu', sa.Integer(), nullable=True),
            sa.Column('cpu_milli', sa.Integer(), nullable=True),
            sa.Column('memory_bytes', sa.BigInteger(), nullable=True),
            sa.Column('pod_count', sa.Integer(), nullable=True),
            sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('updated_at', sa.TIMESTAMP(), nullable=False),
            sa.Column('deleted_at', sa.TIMESTAMP(), nullable=True),
            sa.ForeignKeyConstraint(
                ['cluster_id'], ['clusters.id'], ondelete='CASCADE',
            ),
            sa.ForeignKeyConstraint(
                ['organization_id'], ['organizations.id'], ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint(
                'cluster_id', 'organization_id', name='uix_tenant_quota_cluster_org'
            ),
        )

    if not table_exists('model_route_principals'):
        op.create_table(
            'model_route_principals',
            sa.Column('route_id', sa.Integer(), nullable=False),
            sa.Column('principal_type', principal_type, nullable=False),
            sa.Column('principal_id', sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(
                ['route_id'], ['model_routes.id'], ondelete='CASCADE',
            ),
            sa.PrimaryKeyConstraint('route_id', 'principal_type', 'principal_id'),
        )

    # ---- Seed platform Org ----------------------------------------------

    op.execute(
        sa.text(
            """
            INSERT INTO organizations
                (id, name, slug, description, billing_account_ref, is_platform,
                 created_at, updated_at, deleted_at)
            SELECT :id, :name, :slug, :desc, NULL, :is_platform,
                   CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL
            WHERE NOT EXISTS (
                SELECT 1 FROM organizations WHERE id = :id
            )
            """
        ).bindparams(
            id=PLATFORM_ORG_ID,
            name=PLATFORM_ORG_NAME,
            slug=PLATFORM_ORG_SLUG,
            desc='Built-in platform organization',
            is_platform=True,
        )
    )

    # Realign sequence on postgres so the next inserted org doesn't collide.
    if bind.dialect.name == 'postgresql':
        op.execute(
            "SELECT setval(pg_get_serial_sequence('organizations', 'id'), "
            "GREATEST((SELECT MAX(id) FROM organizations), 1))"
        )

    # ---- Backfill organization_memberships -------------------------------

    # admin -> owner, otherwise -> member. Skip system/internal users.
    if bind.dialect.name == 'postgresql':
        op.execute(
            sa.text(
                """
                INSERT INTO organization_memberships
                    (user_id, organization_id, role, created_at)
                SELECT id, :org_id,
                       CASE WHEN is_admin THEN 'OWNER'::orgrole
                            ELSE 'MEMBER'::orgrole END,
                       CURRENT_TIMESTAMP
                FROM users
                WHERE COALESCE(is_system, false) = false
                ON CONFLICT (user_id, organization_id) DO NOTHING
                """
            ).bindparams(org_id=PLATFORM_ORG_ID)
        )
    else:
        op.execute(
            sa.text(
                """
                INSERT OR IGNORE INTO organization_memberships
                    (user_id, organization_id, role, created_at)
                SELECT id, :org_id,
                       CASE WHEN is_admin THEN 'OWNER' ELSE 'MEMBER' END,
                       CURRENT_TIMESTAMP
                FROM users
                WHERE COALESCE(is_system, 0) = 0
                """
            ).bindparams(org_id=PLATFORM_ORG_ID)
        )

    # ---- users.default_organization_id ----------------------------------

    if not column_exists('users', 'default_organization_id'):
        with op.batch_alter_table('users', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('default_organization_id', sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                'fk_users_default_organization_id_organizations',
                'organizations',
                ['default_organization_id'],
                ['id'],
                ondelete='SET NULL',
            )

    op.execute(
        sa.text(
            """
            UPDATE users
            SET default_organization_id = :org_id
            WHERE default_organization_id IS NULL
              AND COALESCE(is_system, FALSE) = FALSE
            """
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )

    # ---- api_keys.organization_id ---------------------------------------

    if not column_exists('api_keys', 'organization_id'):
        with op.batch_alter_table('api_keys', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('organization_id', sa.Integer(), nullable=True)
            )

    op.execute(
        sa.text(
            "UPDATE api_keys SET organization_id = :org_id "
            "WHERE organization_id IS NULL"
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )

    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        batch_op.alter_column(
            'organization_id', existing_type=sa.Integer(), nullable=False
        )
        # Drop and re-create the user_id+name unique constraint to also
        # include organization_id in the uniqueness scope.
        try:
            batch_op.drop_constraint('uix_user_id_name', type_='unique')
        except Exception:
            pass
        batch_op.create_unique_constraint(
            'uix_user_org_name', ['user_id', 'organization_id', 'name']
        )
        batch_op.create_foreign_key(
            'fk_api_keys_organization_id_organizations',
            'organizations',
            ['organization_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ---- models.organization_id, owner_type, owner_id -------------------

    if not column_exists('models', 'organization_id'):
        with op.batch_alter_table('models', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('organization_id', sa.Integer(), nullable=True)
            )
    if not column_exists('models', 'owner_type'):
        with op.batch_alter_table('models', schema=None) as batch_op:
            batch_op.add_column(sa.Column('owner_type', principal_type, nullable=True))
    if not column_exists('models', 'owner_id'):
        with op.batch_alter_table('models', schema=None) as batch_op:
            batch_op.add_column(sa.Column('owner_id', sa.Integer(), nullable=True))

    op.execute(
        sa.text(
            "UPDATE models SET organization_id = :org_id "
            "WHERE organization_id IS NULL"
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )
    if bind.dialect.name == 'postgresql':
        op.execute(
            sa.text(
                "UPDATE models SET owner_type = 'ORG'::principaltype, "
                "owner_id = :org_id WHERE owner_type IS NULL"
            ).bindparams(org_id=PLATFORM_ORG_ID)
        )
    else:
        op.execute(
            sa.text(
                "UPDATE models SET owner_type = 'ORG', owner_id = :org_id "
                "WHERE owner_type IS NULL"
            ).bindparams(org_id=PLATFORM_ORG_ID)
        )

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column(
            'organization_id', existing_type=sa.Integer(), nullable=False
        )
        batch_op.alter_column(
            'owner_type', existing_type=principal_type, nullable=False
        )
        batch_op.alter_column(
            'owner_id', existing_type=sa.Integer(), nullable=False
        )
        batch_op.create_foreign_key(
            'fk_models_organization_id_organizations',
            'organizations',
            ['organization_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ---- model_instances.organization_id --------------------------------

    if not column_exists('model_instances', 'organization_id'):
        with op.batch_alter_table('model_instances', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('organization_id', sa.Integer(), nullable=True)
            )

    op.execute(
        sa.text(
            "UPDATE model_instances SET organization_id = :org_id "
            "WHERE organization_id IS NULL"
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.alter_column(
            'organization_id', existing_type=sa.Integer(), nullable=False
        )
        batch_op.create_foreign_key(
            'fk_model_instances_organization_id_organizations',
            'organizations',
            ['organization_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ---- model_routes.organization_id -----------------------------------

    if not column_exists('model_routes', 'organization_id'):
        with op.batch_alter_table('model_routes', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('organization_id', sa.Integer(), nullable=True)
            )

    op.execute(
        sa.text(
            "UPDATE model_routes SET organization_id = :org_id "
            "WHERE organization_id IS NULL"
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )

    with op.batch_alter_table('model_routes', schema=None) as batch_op:
        batch_op.alter_column(
            'organization_id', existing_type=sa.Integer(), nullable=False
        )
        batch_op.create_foreign_key(
            'fk_model_routes_organization_id_organizations',
            'organizations',
            ['organization_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ---- Backfill cluster_access (platform Org keeps existing access) ---

    if bind.dialect.name == 'postgresql':
        op.execute(
            sa.text(
                """
                INSERT INTO cluster_access
                    (cluster_id, principal_type, principal_id, granted_by, created_at)
                SELECT c.id, 'ORG'::principaltype, :org_id, NULL, CURRENT_TIMESTAMP
                FROM clusters c
                WHERE c.deleted_at IS NULL
                ON CONFLICT (cluster_id, principal_type, principal_id) DO NOTHING
                """
            ).bindparams(org_id=PLATFORM_ORG_ID)
        )
    else:
        op.execute(
            sa.text(
                """
                INSERT OR IGNORE INTO cluster_access
                    (cluster_id, principal_type, principal_id, granted_by, created_at)
                SELECT c.id, 'ORG', :org_id, NULL, CURRENT_TIMESTAMP
                FROM clusters c
                WHERE c.deleted_at IS NULL
                """
            ).bindparams(org_id=PLATFORM_ORG_ID)
        )

    # ---- Backfill model_route_principals from UserModelRouteLink --------

    user_link_table = (
        'usermodelroutelink' if table_exists('usermodelroutelink') else None
    )
    if user_link_table:
        if bind.dialect.name == 'postgresql':
            op.execute(
                sa.text(
                    f"""
                    INSERT INTO model_route_principals
                        (route_id, principal_type, principal_id)
                    SELECT route_id, 'USER'::principaltype, user_id
                    FROM {user_link_table}
                    WHERE route_id IS NOT NULL AND user_id IS NOT NULL
                    ON CONFLICT (route_id, principal_type, principal_id) DO NOTHING
                    """
                )
            )
        else:
            op.execute(
                sa.text(
                    f"""
                    INSERT OR IGNORE INTO model_route_principals
                        (route_id, principal_type, principal_id)
                    SELECT route_id, 'USER', user_id
                    FROM {user_link_table}
                    WHERE route_id IS NOT NULL AND user_id IS NOT NULL
                    """
                )
            )


def downgrade() -> None:
    bind = op.get_bind()

    # ---- model_routes ---------------------------------------------------

    with op.batch_alter_table('model_routes', schema=None) as batch_op:
        try:
            batch_op.drop_constraint(
                'fk_model_routes_organization_id_organizations', type_='foreignkey'
            )
        except Exception:
            pass
        batch_op.drop_column('organization_id')

    # ---- model_instances ------------------------------------------------

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        try:
            batch_op.drop_constraint(
                'fk_model_instances_organization_id_organizations',
                type_='foreignkey',
            )
        except Exception:
            pass
        batch_op.drop_column('organization_id')

    # ---- models ---------------------------------------------------------

    with op.batch_alter_table('models', schema=None) as batch_op:
        try:
            batch_op.drop_constraint(
                'fk_models_organization_id_organizations', type_='foreignkey'
            )
        except Exception:
            pass
        batch_op.drop_column('owner_id')
        batch_op.drop_column('owner_type')
        batch_op.drop_column('organization_id')

    # ---- api_keys -------------------------------------------------------

    with op.batch_alter_table('api_keys', schema=None) as batch_op:
        try:
            batch_op.drop_constraint(
                'fk_api_keys_organization_id_organizations', type_='foreignkey'
            )
        except Exception:
            pass
        try:
            batch_op.drop_constraint('uix_user_org_name', type_='unique')
        except Exception:
            pass
        batch_op.create_unique_constraint(
            'uix_user_id_name', ['user_id', 'name']
        )
        batch_op.drop_column('organization_id')

    # ---- users ----------------------------------------------------------

    with op.batch_alter_table('users', schema=None) as batch_op:
        try:
            batch_op.drop_constraint(
                'fk_users_default_organization_id_organizations', type_='foreignkey'
            )
        except Exception:
            pass
        batch_op.drop_column('default_organization_id')

    # ---- Drop new tables (reverse FK order) -----------------------------

    if table_exists('model_route_principals'):
        op.drop_table('model_route_principals')
    if table_exists('tenant_quotas'):
        op.drop_table('tenant_quotas')
    if table_exists('tenant_namespaces'):
        op.drop_table('tenant_namespaces')
    if table_exists('cluster_access'):
        op.drop_table('cluster_access')
    if table_exists('user_group_memberships'):
        op.drop_table('user_group_memberships')
    if table_exists('user_groups'):
        op.drop_table('user_groups')
    if table_exists('organization_memberships'):
        op.drop_table('organization_memberships')
    if table_exists('organizations'):
        op.drop_table('organizations')

    # ---- Drop enum types on postgres ------------------------------------

    if bind.dialect.name == 'postgresql':
        for enum in reversed(_enums()):
            try:
                enum.drop(bind, checkfirst=True)
            except Exception:
                pass
