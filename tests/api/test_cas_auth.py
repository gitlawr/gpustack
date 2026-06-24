from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from gpustack.api.exceptions import UnauthorizedException
from gpustack.routes.auth import validate_cas_ticket


def _config(**overrides):
    base = dict(
        cas_server_url="https://cas.example.com/cas",
        cas_validate_endpoint="/serviceValidate",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _client_returning(body: str):
    client = MagicMock()
    response = MagicMock()
    response.content = body.encode("utf-8")
    response.raise_for_status = MagicMock()
    client.get = AsyncMock(return_value=response)
    return client


SERVICE = "https://gpu.example.com/auth/cas/callback"
CAS_NS = 'xmlns:cas="http://www.yale.edu/tp/cas"'


@pytest.mark.asyncio
async def test_validate_cas_ticket_success_namespaced_attributes():
    xml = f"""
    <cas:serviceResponse {CAS_NS}>
      <cas:authenticationSuccess>
        <cas:user>alice</cas:user>
        <cas:attributes>
          <cas:displayName>Alice Example</cas:displayName>
          <cas:email>alice@example.com</cas:email>
        </cas:attributes>
      </cas:authenticationSuccess>
    </cas:serviceResponse>
    """
    data = await validate_cas_ticket(
        _client_returning(xml), "ST-abc", SERVICE, _config()
    )
    assert data["user"] == "alice"
    assert data["displayName"] == "Alice Example"
    assert data["email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_validate_cas_ticket_success_bare_tags():
    # Servers that omit the CAS namespace must still parse.
    xml = """
    <serviceResponse>
      <authenticationSuccess>
        <user>bob</user>
        <attributes>
          <displayName>Bob</displayName>
        </attributes>
      </authenticationSuccess>
    </serviceResponse>
    """
    data = await validate_cas_ticket(
        _client_returning(xml), "ST-abc", SERVICE, _config()
    )
    assert data["user"] == "bob"
    assert data["displayName"] == "Bob"


@pytest.mark.asyncio
async def test_validate_cas_ticket_repeated_group_attributes_become_list():
    xml = f"""
    <cas:serviceResponse {CAS_NS}>
      <cas:authenticationSuccess>
        <cas:user>carol</cas:user>
        <cas:attributes>
          <cas:group>devs</cas:group>
          <cas:group>admins</cas:group>
        </cas:attributes>
      </cas:authenticationSuccess>
    </cas:serviceResponse>
    """
    data = await validate_cas_ticket(
        _client_returning(xml), "ST-abc", SERVICE, _config()
    )
    assert data["group"] == ["devs", "admins"]


@pytest.mark.asyncio
async def test_validate_cas_ticket_failure_raises():
    xml = f"""
    <cas:serviceResponse {CAS_NS}>
      <cas:authenticationFailure code="INVALID_TICKET">
        Ticket 'ST-abc' not recognized
      </cas:authenticationFailure>
    </cas:serviceResponse>
    """
    with pytest.raises(UnauthorizedException):
        await validate_cas_ticket(_client_returning(xml), "ST-abc", SERVICE, _config())


@pytest.mark.asyncio
async def test_validate_cas_ticket_missing_username_raises():
    xml = f"""
    <cas:serviceResponse {CAS_NS}>
      <cas:authenticationSuccess>
        <cas:attributes>
          <cas:displayName>No User</cas:displayName>
        </cas:attributes>
      </cas:authenticationSuccess>
    </cas:serviceResponse>
    """
    with pytest.raises(UnauthorizedException):
        await validate_cas_ticket(_client_returning(xml), "ST-abc", SERVICE, _config())
