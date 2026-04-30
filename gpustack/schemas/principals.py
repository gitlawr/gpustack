from enum import Enum


class PrincipalType(str, Enum):
    ORG = "org"
    GROUP = "group"
    USER = "user"


class OrgRole(str, Enum):
    OWNER = "owner"
    # Renamed from ADMIN to disambiguate from `users.is_admin` (the
    # platform-level superuser). Functionally same as the old ADMIN —
    # the day-to-day manager of an Org, distinct from owner.
    MANAGER = "manager"
    MEMBER = "member"
