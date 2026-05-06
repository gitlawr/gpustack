"""recreate non_admin_user_models view to support ALLOWED_PRINCIPALS
plus a one-shot top-up of platform Org memberships missed by the foundation
migration on fresh installs (where _init_user creates the admin row *after*
migrations run, so the foundation's backfill SELECT sees an empty users
table).

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
from gpustack.migrations.utils import column_exists
from gpustack.schemas.stmt import (
    model_user_after_create_view_stmt,
    model_user_after_drop_view_stmt,
)


access_policy_enum = sa.Enum(
    'PUBLIC', 'AUTHED', 'ALLOWED_USERS', name='accesspolicyenum'
)
access_policy_to_add = ['ALLOWED_PRINCIPALS']

PLATFORM_ORG_ID = 1


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

    # ---- Top up platform Org memberships --------------------------------
    # Idempotent: any user that's already linked is left alone. Catches the
    # bootstrap admin created by _init_user after the foundation migration
    # had already run against an empty users table on a fresh install.
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
        op.execute(
            sa.text(
                """
                UPDATE users
                SET default_organization_id = :org_id
                WHERE default_organization_id IS NULL
                  AND COALESCE(is_system, false) = false
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
        op.execute(
            sa.text(
                """
                UPDATE users
                SET default_organization_id = :org_id
                WHERE default_organization_id IS NULL
                  AND COALESCE(is_system, 0) = 0
                """
            ).bindparams(org_id=PLATFORM_ORG_ID)
        )

    # ---- BYO cluster / pool: tag with owner Org -------------------------
    # Clusters and worker_pools are always Org-owned (sharing across Orgs
    # is via cluster_access). Cloud credentials remain optional Org-scoped
    # so admin can keep platform-shared providers.
    # ON DELETE CASCADE for clusters/pools — deleting an Org takes its
    # clusters with it; cloud_credentials stays SET NULL since admin's
    # platform-shared creds outlive any single Org.
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
                ondelete="SET NULL",
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
    # NOT NULL constraint below holds, and so admin's existing clusters
    # land in the Default Org as expected.
    op.execute(
        sa.text(
            "UPDATE clusters SET organization_id = :org_id "
            "WHERE organization_id IS NULL"
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )
    op.execute(
        sa.text(
            "UPDATE worker_pools SET organization_id = :org_id "
            "WHERE organization_id IS NULL"
        ).bindparams(org_id=PLATFORM_ORG_ID)
    )

    # Promote the columns to NOT NULL now that no NULLs remain.
    with op.batch_alter_table("clusters", schema=None) as batch_op:
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

    # ---- Cluster-derived resources: denormalize organization_id ----------
    # Workers, GPU view (via workers col), model_files, benchmarks,
    # model_evaluations, model_provider, model_usages all need a
    # tenant pointer for per-row filtering. NULL = belongs to a
    # global cluster (admin-managed). ON DELETE SET NULL keeps
    # rows alive when an Org is deleted (but Org delete cascades clusters
    # which delete their workers anyway).
    # Note: ModelEvaluation is a synchronous request/response (no table),
    # so it doesn't appear here.
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

    # ---- Inference backends: Hybrid model -------------------------------
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
            # varies by dialect; let create_constraint figure that out by
            # going through batch_alter_table's reflection).
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

    # Recreate gpu_devices_view so it picks up the new w.organization_id
    # column; the SELECT is taken from gpustack.schemas.stmt.
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

    # ---- Personal Orgs: every non-system user gets their own namespace
    # Adds an is_personal flag, then for each existing user creates a
    # Personal Org named "Personal" with slug "user-{id}", makes them
    # ADMIN, and points users.default_organization_id at it. Removes
    # non-admin users from the Default Org (id=1) — they no longer get
    # auto-enrolled there; admin must add them explicitly if desired.
    if not column_exists("organizations", "is_personal"):
        with op.batch_alter_table("organizations", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "is_personal",
                    sa.Boolean(),
                    nullable=False,
                    server_default="0" if bind.dialect.name != "postgresql" else sa.text("false"),
                )
            )

    # Insert a Personal Org for every user that doesn't already have one.
    # The "doesn't already have one" check uses the canonical slug pattern
    # so re-running the migration is idempotent.
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
        # Drop non-admin users from the Default Org (id=1). Admin keeps
        # ADMIN role there.
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
