"""Unit tests for the CAS (Central Authentication Service) flow in
:mod:`gpustack.routes.auth`. Mock-based; no live CAS server required."""

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import UnauthorizedException
from gpustack.routes import auth as auth_route


class _Response:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            # Mirror httpx's raise_for_status shape just enough for the
            # caller's except branch to match.
            import httpx

            request = httpx.Request("GET", "http://example/")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


def _client(xml: str, *, status_code: int = 200) -> MagicMock:
    """A fake httpx.AsyncClient whose ``get`` returns ``xml``."""
    client = MagicMock()
    client.get = AsyncMock(return_value=_Response(xml, status_code=status_code))
    return client


def _config(
    *,
    server_url: str = "https://cas.example.com/cas",
    validate_endpoint: Optional[str] = "/serviceValidate",
) -> MagicMock:
    cfg = MagicMock()
    cfg.cas_server_url = server_url
    cfg.cas_validate_endpoint = validate_endpoint
    return cfg


_SUCCESS_WITH_NS = """<?xml version="1.0" encoding="UTF-8"?>
<cas:serviceResponse xmlns:cas="http://www.yale.edu/tp/cas">
  <cas:authenticationSuccess>
    <cas:user>alice</cas:user>
    <cas:attributes>
      <cas:displayName>Alice Smith</cas:displayName>
      <cas:email>alice@example.com</cas:email>
    </cas:attributes>
  </cas:authenticationSuccess>
</cas:serviceResponse>
"""

_SUCCESS_WITHOUT_NS = """<?xml version="1.0" encoding="UTF-8"?>
<serviceResponse>
  <authenticationSuccess>
    <user>bob</user>
    <attributes>
      <displayName>Bob Jones</displayName>
    </attributes>
  </authenticationSuccess>
</serviceResponse>
"""

_FAILURE = """<?xml version="1.0" encoding="UTF-8"?>
<cas:serviceResponse xmlns:cas="http://www.yale.edu/tp/cas">
  <cas:authenticationFailure code="INVALID_TICKET">
    Ticket not recognised
  </cas:authenticationFailure>
</cas:serviceResponse>
"""

_SUCCESS_WITHOUT_USER = """<?xml version="1.0" encoding="UTF-8"?>
<cas:serviceResponse xmlns:cas="http://www.yale.edu/tp/cas">
  <cas:authenticationSuccess>
    <cas:attributes>
      <cas:displayName>Nameless</cas:displayName>
    </cas:attributes>
  </cas:authenticationSuccess>
</cas:serviceResponse>
"""


@pytest.mark.asyncio
async def test_validate_cas_ticket_parses_namespaced_success():
    data = await auth_route.validate_cas_ticket(
        _client(_SUCCESS_WITH_NS),
        ticket="ST-123",
        service="https://gpustack.example.com/auth/cas/callback",
        config=_config(),
    )
    assert data["username"] == "alice"
    assert data["displayName"] == "Alice Smith"
    assert data["email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_validate_cas_ticket_parses_unnamespaced_success():
    data = await auth_route.validate_cas_ticket(
        _client(_SUCCESS_WITHOUT_NS),
        ticket="ST-456",
        service="https://gpustack.example.com/auth/cas/callback",
        config=_config(),
    )
    assert data["username"] == "bob"
    assert data["displayName"] == "Bob Jones"


@pytest.mark.asyncio
async def test_validate_cas_ticket_surfaces_authentication_failure():
    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.validate_cas_ticket(
            _client(_FAILURE),
            ticket="ST-bad",
            service="https://gpustack.example.com/auth/cas/callback",
            config=_config(),
        )
    msg = str(exc.value.message)
    assert "INVALID_TICKET" in msg
    assert "Ticket not recognised" in msg


@pytest.mark.asyncio
async def test_validate_cas_ticket_rejects_malformed_xml():
    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.validate_cas_ticket(
            _client("<not-xml"),
            ticket="ST-bad",
            service="https://gpustack.example.com/auth/cas/callback",
            config=_config(),
        )
    assert "parse" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_validate_cas_ticket_rejects_response_without_user():
    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.validate_cas_ticket(
            _client(_SUCCESS_WITHOUT_USER),
            ticket="ST-789",
            service="https://gpustack.example.com/auth/cas/callback",
            config=_config(),
        )
    assert "username" in str(exc.value.message).lower()


@pytest.mark.asyncio
async def test_validate_cas_ticket_surfaces_http_error():
    with pytest.raises(UnauthorizedException) as exc:
        await auth_route.validate_cas_ticket(
            _client("ignored", status_code=503),
            ticket="ST-noop",
            service="https://gpustack.example.com/auth/cas/callback",
            config=_config(),
        )
    assert "503" in str(exc.value.message)


@pytest.mark.asyncio
async def test_validate_cas_ticket_strips_trailing_slash_and_uses_endpoint():
    """The validator must rstrip the configured base URL and respect a
    custom ``cas_validate_endpoint``."""
    client = _client(_SUCCESS_WITH_NS)
    await auth_route.validate_cas_ticket(
        client,
        ticket="ST-abc",
        service="https://gpustack.example.com/auth/cas/callback",
        config=_config(
            server_url="https://cas.example.com/cas/",
            validate_endpoint="/p3/serviceValidate",
        ),
    )
    url = client.get.await_args.args[0]
    assert url.startswith("https://cas.example.com/cas/p3/serviceValidate?")
    assert "ticket=ST-abc" in url
    # ``service`` is URL-encoded; ``quote`` keeps slashes by default,
    # but the ``:`` after the scheme must be percent-encoded so the
    # query parser doesn't choke on it.
    assert "service=https%3A" in url
    assert "/auth/cas/callback" in url


def test_cas_added_to_auth_provider_enum():
    from gpustack.schemas.users import AuthProviderEnum

    assert AuthProviderEnum.CAS.value == "CAS"
    assert AuthProviderEnum.CAS == "CAS"
