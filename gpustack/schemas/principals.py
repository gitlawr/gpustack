from enum import Enum


class PrincipalType(str, Enum):
    ORG = "org"
    GROUP = "group"
    USER = "user"


class OrgRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
