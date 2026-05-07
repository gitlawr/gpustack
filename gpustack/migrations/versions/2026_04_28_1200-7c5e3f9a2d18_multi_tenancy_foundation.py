"""multi-tenancy foundation

Sets up the entire tenancy storage layer in one upgrade. The work falls
into a few logical groups, run in order:

1. New tables — organizations / organization_memberships / user_groups /
   user_group_memberships / cluster_access / tenant_quotas /
   model_route_principals.
2. Seed the platform Org (id=1, slug=`default`, name=`Default`).
3. Backfill memberships for every existing user (admin → ADMIN role,
   regular → USER role). System / internal users are skipped.
4. Add `organization_id` to existing tenant-scoped tables: api_keys /
   models / model_instances / model_routes; backfill to platform Org
   and promote to NOT NULL.
5. BYO cluster: `clusters` / `cloud_credentials` / `worker_pools` get
   `organization_id` (NOT NULL after backfill, ON DELETE CASCADE);
   per-Org default cluster expressed via partial unique index.
6. Cluster-derived denormalized columns: workers / model_files /
   benchmarks / model_providers / model_usages each get
   `organization_id` (nullable; SET NULL on Org delete). model_files
   gains `cluster_id` for cluster_access-based filtering.
7. Inference backends Hybrid: `organization_id` (nullable = Platform
   row), composite unique on (backend_name, organization_id).
8. Personal Orgs: `organizations.is_personal` flag + per-user Personal
   Org provisioning for every existing user, becomes their default Org.
   Non-admin users are removed from the platform Org membership (admin
   keeps it for the shared workspace).
9. Backfill `model_route_principals` from the legacy
   `usermodelroutelink` table so ALLOWED_USERS-published routes remain
   visible through the new ALLOWED_PRINCIPALS path.
10. Extend `accesspolicyenum` with ALLOWED_PRINCIPALS and ORG.
11. Recreate `non_admin_user_models` and `gpu_devices_view` so they
    reference the new columns and enum values.

Revision ID: 7c5e3f9a2d18
Revises: 8bf38a6bb3b5
Create Date: 2026-04-28 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

import gpustack  # noqa: F401  (keeps SQLModel registrations side-effect-loaded)
import gpustack.utils.sql_enum as sql_enum
from gpustack.migrations.utils import column_exists, table_exists
from gpustack.schemas.stmt import (
    model_user_after_create_view_stmt,
    model_user_after_drop_view_stmt,
)


# revision identifiers, used by Alembic.
revision: str = '7c5e3f9a2d18'
down_revision: Union[str, None] = '8bf38a6bb3b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PLATFORM_ORG_ID = 1
PLATFORM_ORG_SLUG = 'default'
PLATFORM_ORG_NAME = 'Default'


def _enums():
    org_role = sa.Enum('ADMIN', 'USER', name='orgrole')
    principal_type = sa.Enum('ORG', 'GROUP', 'USER', name='principaltype')
    return org_role, principal_type


# Old non_admin_user_models view definition, restored on downgrade so a
# rolled-back deployment continues to drive /my-models.
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
    org_role, principal_type = _enums()

    # On postgres, enum types are created lazily by the first column
    # referencing them; subsequent columns reuse the existing type since
    # SQLAlchemy tracks the same enum instance.

    # ---- 1. New tables ---------------------------------------------------

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

    if not table_exists('tenant_quotas'):
        op.create_table(
            'tenant_quotas',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('cluster_id', sa.Integer(), nullable=False),
            sa.Column('organization_id', sa.Integer(), nullable=False),
            sa.Column('gpu', sa.Integer(), nullable=True),
            sa.Column('cpu_milli', sa.Integer(), nullable=True),
            sa.Column('memory_bytes', sa.BigInteger(), nullable=True),
            sa.Column('gpu_instance', sa.Integer(), nullable=True),
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

    # ---- 2. Seed platform Org -------------------------------------------

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

    # ---- 3. Backfill organization_memberships ----------------------------

    # Platform admin → ADMIN role, otherwise → USER. Skip system / internal users.
    if bind.dialect.name == 'postgresql':
        op.execute(
            sa.text(
                """
                INSERT INTO organization_memberships
                    (user_id, organization_id, role, created_at)
                SELECT id, :org_id,
                       CASE WHEN is_admin THEN 'ADMIN'::orgrole
                            ELSE 'USER'::orgrole END,
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
                       CASE WHEN is_admin THEN 'ADMIN' ELSE 'USER' END,
                       CURRENT_TIMESTAMP
                FROM users
                WHERE COALESCE(is_system, 0) = 0
                """
            ).bindparams(org_id=PLATFORM_ORG_ID)
        )

    # ---- 4. users.default_organization_id --------------------------------

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

    # Tentative default — overridden later for non-admin users when their
    # Personal Org gets provisioned.
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

    # ---- 5. api_keys.organization_id -------------------------------------

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
        # Unique scope was (user_id, name); add organization_id so the
        # same key name can coexist across Orgs.
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

    # ---- 6. models.organization_id ---------------------------------------

    if not column_exists('models', 'organization_id'):
        with op.batch_alter_table('models', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('organization_id', sa.Integer(), nullable=True)
            )

    op.execute(
        sa.text(
            "UPDATE models SET organization_id = :org_id "
            "WHERE organization_id IS NULL"
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.alter_column(
            'organization_id', existing_type=sa.Integer(), nullable=False
        )
        batch_op.create_foreign_key(
            'fk_models_organization_id_organizations',
            'organizations',
            ['organization_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # ---- 7. model_instances.organization_id ------------------------------

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

    # ---- 8. model_routes.organization_id ---------------------------------

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

    # ---- 9. BYO cluster: clusters / cloud_credentials / worker_pools -----

    # Clusters / cloud credentials / worker_pools are always Org-owned.
    # Cross-Org sharing goes through cluster_access, never NULL ownership.
    # ON DELETE CASCADE: deleting an Org takes its infra rows with it.
    if not column_exists("clusters", "organization_id"):
        with op.batch_alter_table("clusters", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("organization_id", sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_clusters_organization_id_organizations",
                "organizations",
                ["organization_id"],
                ["id"],
                ondelete="CASCADE",
            )

    if not column_exists("cloud_credentials", "organization_id"):
        with op.batch_alter_table("cloud_credentials", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("organization_id", sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_cloud_credentials_organization_id_organizations",
                "organizations",
                ["organization_id"],
                ["id"],
                ondelete="CASCADE",
            )

    if not column_exists("worker_pools", "organization_id"):
        with op.batch_alter_table("worker_pools", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("organization_id", sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_worker_pools_organization_id_organizations",
                "organizations",
                ["organization_id"],
                ["id"],
                ondelete="CASCADE",
            )

    # Backfill any pre-existing rows with NULL org → platform Org so the
    # NOT NULL constraint below holds (admin's existing infra lands in
    # the Default Org as expected).
    op.execute(
        sa.text(
            "UPDATE clusters SET organization_id = :org_id "
            "WHERE organization_id IS NULL"
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )
    op.execute(
        sa.text(
            "UPDATE cloud_credentials SET organization_id = :org_id "
            "WHERE organization_id IS NULL"
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )
    op.execute(
        sa.text(
            "UPDATE worker_pools SET organization_id = :org_id "
            "WHERE organization_id IS NULL"
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )

    with op.batch_alter_table("clusters", schema=None) as batch_op:
        batch_op.alter_column(
            "organization_id", existing_type=sa.Integer(), nullable=False
        )
    with op.batch_alter_table("cloud_credentials", schema=None) as batch_op:
        batch_op.alter_column(
            "organization_id", existing_type=sa.Integer(), nullable=False
        )
    with op.batch_alter_table("worker_pools", schema=None) as batch_op:
        batch_op.alter_column(
            "organization_id", existing_type=sa.Integer(), nullable=False
        )

    # At most one default cluster per Org. Partial unique covers active
    # rows only (excluding soft-deleted), letting an Org "rotate" defaults
    # by soft-deleting the old + flipping the new without conflict.
    if bind.dialect.name == "postgresql":
        op.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uix_clusters_default_per_org "
            "ON clusters (organization_id) "
            "WHERE is_default = true AND deleted_at IS NULL"
        )
    else:
        op.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uix_clusters_default_per_org "
            "ON clusters (organization_id) "
            "WHERE is_default = 1 AND deleted_at IS NULL"
        )

    # ---- 10. Cluster-derived denormalized organization_id ----------------

    # Workers, model_files, benchmarks, model_providers, model_usages all
    # need a tenant pointer for per-row filtering. Nullable (NULL = on a
    # global cluster, admin-managed); ON DELETE SET NULL keeps rows alive
    # when an Org is deleted (Org delete cascades clusters, which cascade
    # their workers anyway).
    for tbl in (
        "workers",
        "model_files",
        "benchmarks",
        "model_providers",
        "model_usages",
    ):
        if not column_exists(tbl, "organization_id"):
            with op.batch_alter_table(tbl, schema=None) as batch_op:
                batch_op.add_column(
                    sa.Column("organization_id", sa.Integer(), nullable=True)
                )
                batch_op.create_foreign_key(
                    f"fk_{tbl}_organization_id_organizations",
                    "organizations",
                    ["organization_id"],
                    ["id"],
                    ondelete="SET NULL",
                )

    # model_files only had worker_id; add cluster_id for direct
    # cluster_access-based filtering.
    if not column_exists("model_files", "cluster_id"):
        with op.batch_alter_table("model_files", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("cluster_id", sa.Integer(), nullable=True)
            )

    # ---- 11. Inference backends Hybrid -----------------------------------

    # NULL organization_id = Platform-managed (admin curates built-ins);
    # non-NULL = an Org's extension/override. backend_name is no longer
    # globally unique — composite unique on (backend_name, organization_id)
    # lets each Org carry their own row alongside the Platform row.
    if not column_exists("inference_backends", "organization_id"):
        with op.batch_alter_table("inference_backends", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column("organization_id", sa.Integer(), nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_inference_backends_organization_id_organizations",
                "organizations",
                ["organization_id"],
                ["id"],
                ondelete="CASCADE",
            )
            # Drop the old single-column unique on backend_name (its name
            # varies by dialect; let batch_alter_table's reflection figure
            # it out — silenced because not every dialect / install has it).
            try:
                batch_op.drop_constraint(
                    "inference_backends_backend_name_key", type_="unique"
                )
            except Exception:
                pass
            try:
                batch_op.drop_index("ix_inference_backends_backend_name")
            except Exception:
                pass
            batch_op.create_unique_constraint(
                "uix_inference_backends_name_org",
                ["backend_name", "organization_id"],
            )
            batch_op.create_index(
                "ix_inference_backends_backend_name", ["backend_name"]
            )

    # ---- 12. Personal Orgs for existing users ----------------------------

    # Add `organizations.is_personal`, then for each existing non-system
    # user create a Personal Org named "Personal" with slug "user-{id}",
    # make the user ADMIN, and point users.default_organization_id at it.
    # Non-admin users are then removed from the Default Org membership —
    # they no longer auto-enroll there; admin must add them explicitly if
    # team-workspace access is wanted.
    if not column_exists("organizations", "is_personal"):
        with op.batch_alter_table("organizations", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "is_personal",
                    sa.Boolean(),
                    nullable=False,
                    server_default=(
                        "0" if bind.dialect.name != "postgresql" else sa.text("false")
                    ),
                )
            )

    # Insert a Personal Org for every user that doesn't already have one.
    # Re-running the migration is idempotent because the lookup is keyed
    # on the canonical `user-{id}` slug pattern.
    if bind.dialect.name == "postgresql":
        op.execute(
            sa.text(
                """
                INSERT INTO organizations
                    (name, slug, description, is_platform, is_personal,
                     created_at, updated_at, deleted_at)
                SELECT 'Personal',
                       'user-' || u.id,
                       'Personal namespace',
                       false,
                       true,
                       CURRENT_TIMESTAMP,
                       CURRENT_TIMESTAMP,
                       NULL
                FROM users u
                WHERE COALESCE(u.is_system, false) = false
                  AND NOT EXISTS (
                    SELECT 1 FROM organizations o
                    WHERE o.slug = 'user-' || u.id
                  )
                """
            )
        )
        op.execute(
            sa.text(
                """
                INSERT INTO organization_memberships
                    (user_id, organization_id, role, created_at)
                SELECT u.id, o.id, 'ADMIN'::orgrole, CURRENT_TIMESTAMP
                FROM users u
                JOIN organizations o
                    ON o.slug = 'user-' || u.id AND o.is_personal = true
                WHERE COALESCE(u.is_system, false) = false
                ON CONFLICT (user_id, organization_id) DO NOTHING
                """
            )
        )
        op.execute(
            sa.text(
                """
                UPDATE users u
                SET default_organization_id = o.id
                FROM organizations o
                WHERE o.slug = 'user-' || u.id
                  AND o.is_personal = true
                  AND COALESCE(u.is_system, false) = false
                """
            )
        )
        # Drop non-admin users from the Default Org. Admin keeps ADMIN
        # there to manage the platform-wide shared workspace.
        op.execute(
            sa.text(
                """
                DELETE FROM organization_memberships
                WHERE organization_id = :org_id
                  AND user_id IN (
                      SELECT id FROM users
                      WHERE COALESCE(is_admin, false) = false
                        AND COALESCE(is_system, false) = false
                  )
                """
            ).bindparams(org_id=PLATFORM_ORG_ID)
        )
    else:
        op.execute(
            sa.text(
                """
                INSERT INTO organizations
                    (name, slug, description, is_platform, is_personal,
                     created_at, updated_at, deleted_at)
                SELECT 'Personal',
                       'user-' || u.id,
                       'Personal namespace',
                       0,
                       1,
                       CURRENT_TIMESTAMP,
                       CURRENT_TIMESTAMP,
                       NULL
                FROM users u
                WHERE COALESCE(u.is_system, 0) = 0
                  AND NOT EXISTS (
                    SELECT 1 FROM organizations o
                    WHERE o.slug = 'user-' || u.id
                  )
                """
            )
        )
        op.execute(
            sa.text(
                """
                INSERT OR IGNORE INTO organization_memberships
                    (user_id, organization_id, role, created_at)
                SELECT u.id, o.id, 'ADMIN', CURRENT_TIMESTAMP
                FROM users u
                JOIN organizations o
                    ON o.slug = 'user-' || u.id AND o.is_personal = 1
                WHERE COALESCE(u.is_system, 0) = 0
                """
            )
        )
        op.execute(
            sa.text(
                """
                UPDATE users
                SET default_organization_id = (
                    SELECT o.id FROM organizations o
                    WHERE o.slug = 'user-' || users.id
                      AND o.is_personal = 1
                )
                WHERE COALESCE(is_system, 0) = 0
                  AND EXISTS (
                    SELECT 1 FROM organizations o
                    WHERE o.slug = 'user-' || users.id
                      AND o.is_personal = 1
                  )
                """
            )
        )
        op.execute(
            sa.text(
                """
                DELETE FROM organization_memberships
                WHERE organization_id = :org_id
                  AND user_id IN (
                      SELECT id FROM users
                      WHERE COALESCE(is_admin, 0) = 0
                        AND COALESCE(is_system, 0) = 0
                  )
                """
            ).bindparams(org_id=PLATFORM_ORG_ID)
        )

    # ---- 13. Backfill model_route_principals -----------------------------

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

    # ---- 14. Extend access_policy enum -----------------------------------

    # ORG = scoped to members of the route's owning Organization (default
    # for non-platform Org routes). ALLOWED_PRINCIPALS = explicit per-user
    # / group / org grants via model_route_principals.
    access_policy_enum = sa.Enum(
        'PUBLIC', 'AUTHED', 'ALLOWED_USERS', name='accesspolicyenum'
    )
    access_policy_to_add = ['ALLOWED_PRINCIPALS', 'ORG']
    if bind.dialect.name == 'postgresql':
        # Postgres won't let a new enum value be referenced in the same
        # transaction that added it ("New enum values must be committed
        # before they can be used"). Drop into autocommit just for the
        # ADD VALUE so the view recreation below — which embeds the new
        # values as string literals — sees a committed enum type.
        with op.get_context().autocommit_block():
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

    # ---- 15. Recreate views ---------------------------------------------

    # non_admin_user_models drives /my-models. Recreate so it joins the
    # new principals table and honors the new ORG / ALLOWED_PRINCIPALS
    # branches.
    op.execute(model_user_after_drop_view_stmt)
    op.execute(model_user_after_create_view_stmt(bind.dialect.name))

    # gpu_devices_view selects from workers; the new
    # workers.organization_id column has to be reflected in the view.
    from gpustack.schemas.stmt import (
        worker_after_drop_view_stmt_sqlite,
        worker_after_create_view_stmt_sqlite,
        worker_after_drop_view_stmt_mysql,
        worker_after_create_view_stmt_mysql,
        worker_after_drop_view_stmt_postgres,
        worker_after_create_view_stmt_postgres,
    )

    if bind.dialect.name == "sqlite":
        op.execute(worker_after_drop_view_stmt_sqlite)
        op.execute(worker_after_create_view_stmt_sqlite)
    elif bind.dialect.name == "mysql":
        op.execute(worker_after_drop_view_stmt_mysql)
        op.execute(worker_after_create_view_stmt_mysql)
    elif bind.dialect.name == "postgresql":
        op.execute(worker_after_drop_view_stmt_postgres)
        op.execute(worker_after_create_view_stmt_postgres)


def downgrade() -> None:
    bind = op.get_bind()

    # ---- Restore legacy non_admin_user_models view ----------------------
    # We deliberately leave new accesspolicyenum values alone — postgres
    # cannot drop a single enum value cleanly when other columns reference
    # the type, and unused enum values are harmless.
    op.execute(model_user_after_drop_view_stmt)
    op.execute(_old_view_stmt(bind.dialect.name))

    # ---- Drop denormalized columns added in step 10 ---------------------

    for tbl in (
        "model_usages",
        "model_providers",
        "benchmarks",
        "model_files",
        "workers",
    ):
        with op.batch_alter_table(tbl, schema=None) as batch_op:
            try:
                batch_op.drop_constraint(
                    f"fk_{tbl}_organization_id_organizations", type_="foreignkey"
                )
            except Exception:
                pass
            try:
                batch_op.drop_column("organization_id")
            except Exception:
                pass

    with op.batch_alter_table("model_files", schema=None) as batch_op:
        try:
            batch_op.drop_column("cluster_id")
        except Exception:
            pass

    # ---- Inference backends Hybrid revert -------------------------------

    with op.batch_alter_table("inference_backends", schema=None) as batch_op:
        try:
            batch_op.drop_index("ix_inference_backends_backend_name")
        except Exception:
            pass
        try:
            batch_op.drop_constraint(
                "uix_inference_backends_name_org", type_="unique"
            )
        except Exception:
            pass
        try:
            batch_op.drop_constraint(
                "fk_inference_backends_organization_id_organizations",
                type_="foreignkey",
            )
        except Exception:
            pass
        try:
            batch_op.drop_column("organization_id")
        except Exception:
            pass

    # ---- BYO cluster column removal -------------------------------------

    op.execute("DROP INDEX IF EXISTS uix_clusters_default_per_org")
    for tbl in ("worker_pools", "cloud_credentials", "clusters"):
        with op.batch_alter_table(tbl, schema=None) as batch_op:
            try:
                batch_op.drop_constraint(
                    f"fk_{tbl}_organization_id_organizations",
                    type_="foreignkey",
                )
            except Exception:
                pass
            try:
                batch_op.drop_column("organization_id")
            except Exception:
                pass

    # ---- organizations.is_personal --------------------------------------

    with op.batch_alter_table("organizations", schema=None) as batch_op:
        try:
            batch_op.drop_column("is_personal")
        except Exception:
            pass

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
