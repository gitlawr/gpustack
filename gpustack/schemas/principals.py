from enum import Enum


class PrincipalType(str, Enum):
    ORG = "org"
    GROUP = "group"
    USER = "user"


class OrgRole(str, Enum):
    # Two-tier Org membership model: ADMIN can manage the Org's infra
    # (resources, members, settings); USER is a plain consumer. The
    # platform-wide superuser lives on `users.is_admin` and is distinct
    # from `OrgRole.ADMIN` — always disambiguate with `is_platform_admin`
    # vs `org_role == OrgRole.ADMIN` in code.
    ADMIN = "admin"
    USER = "user"
