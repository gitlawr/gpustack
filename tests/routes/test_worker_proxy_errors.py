"""Worker proxy failures preserve retry status codes without exposing URLs."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpustack.api.exceptions import OpenAIAPIException
from gpustack.routes.worker import proxy as worker_proxy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,status", [(RuntimeError, 503), (TimeoutError, 504)]
)
@pytest.mark.parametrize("failure_stage", ["request", "first_chunk"])
async def test_proxy_error_omits_request_and_exception_details(
    monkeypatch, caplog, error_type, status, failure_stage
):
    request = MagicMock()
    request.app.state.worker_ip_getter.return_value = "internal-host"
    request.state.x_target_port = "8080"
    request.url.query = "api_key=private-test-value"
    request.headers = {}
    request.body = AsyncMock(return_value=b"")
    failed_request = AsyncMock(side_effect=error_type("credential=private-test-value"))
    response = MagicMock()
    if failure_stage == "first_chunk":
        failed_request.side_effect = None
        failed_request.return_value = response
        monkeypatch.setattr(
            worker_proxy,
            "_read_first_chunk",
            AsyncMock(side_effect=error_type("credential=private-test-value")),
        )
    request.app.state.http_client_no_proxy.request = failed_request
    monkeypatch.setattr(worker_proxy, "use_proxy_env_for_url", lambda url: False)

    async def await_upstream(_request, operation, _body_consumed, *, discard):
        return await operation

    monkeypatch.setattr(worker_proxy, "_cancel_on_client_disconnect", await_upstream)
    with pytest.raises(OpenAIAPIException) as raised:
        await worker_proxy.proxy("v1/chat/completions", request)
    failed_request.assert_awaited_once()
    assert raised.value.status_code == status
    if failure_stage == "first_chunk":
        response.close.assert_called_once()
    if failure_stage == "first_chunk" and error_type is TimeoutError:
        assert "sent no response data" in raised.value.message
    else:
        assert error_type.__name__ in raised.value.message
    assert "private-test-value" not in raised.value.message
    assert "internal-host" not in raised.value.message
    records = [r for r in caplog.records if r.name == worker_proxy.logger.name]
    assert len(records) == 1
    assert isinstance(records[0].exc_info[1], error_type)
    assert records[0].exc_info[2] is not None
    assert "credential=private-test-value" in caplog.text
