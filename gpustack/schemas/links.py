from pydantic import ConfigDict
from sqlalchemy import Enum as SQLEnum
from sqlmodel import (
    Column,
    Field,
    ForeignKey,
    Integer,
    SQLModel,
)

from gpustack.schemas.principals import PrincipalType


class ModelInstanceModelFileLink(SQLModel, table=True):
    model_instance_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_instances.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    model_file_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_files.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )

    model_config = ConfigDict(protected_namespaces=())


class ModelInstanceDraftModelFileLink(SQLModel, table=True):
    model_instance_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_instances.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    model_file_id: int = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_files.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
    )

    model_config = ConfigDict(protected_namespaces=())


class UserModelRouteLink(SQLModel, table=True):
    route_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_routes.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    user_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )


class ModelRoutePrincipalLink(SQLModel, table=True):
    __tablename__ = 'model_route_principals'

    route_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer,
            ForeignKey("model_routes.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    principal_type: PrincipalType = Field(
        sa_column=Column(SQLEnum(PrincipalType), primary_key=True, nullable=False),
    )
    principal_id: int = Field(
        sa_column=Column(Integer, primary_key=True, nullable=False),
    )
