import httpx
import pytest

from gpustack.api.exceptions import (
    ErrorResponse,
    raise_if_response_error,
    BadRequestException,
    InternalServerErrorException,
    NotFoundException,
)


@pytest.mark.parametrize(
    "name, given, expected",
    [
        (
            "valid case",
            {"code": 404, "reason": "NotFound", "message": "Resource not found"},
            True,
        ),
        (
            "valid case with type",
            {
                "code": 404,
                "reason": "NotFound",
                "type": "NotFound",
                "message": "Resource not found",
            },
            True,
        ),
        (
            "invalid type key",
            {"code": 404, "type": "NotFound", "message": "Resource not found"},
            False,
        ),
        (
            "invalid code type",
            {"code": "404", "reason": "NotFound", "message": "Resource not found"},
            False,
        ),
        (
            "invalid reason type",
            {"code": 404, "reason": 123, "message": "Resource not found"},
            False,
        ),
        ("missing message", {"code": 404, "reason": "NotFound"}, False),
        ("missing reason and message", {"code": 404}, False),
        (
            "missing code",
            {"reason": "NotFound", "message": "Resource not found"},
            False,
        ),
    ],
)
def test_error_response_model_validate(name, given, expected):
    try:
        _ = ErrorResponse.model_validate(given)
        assert (
            expected is True
        ), f"Case {name} expected validation to fail but succeeded"
    except Exception as e:
        assert (
            expected is False
        ), f"Case {name} expected validation to succeed but failed: {e}"


@pytest.mark.parametrize(
    "name, given, expected",
    [
        ("valid response", httpx.Response(status_code=200, content="..."), None),
        ("valid response without content", httpx.Response(status_code=204), None),
        (
            "client error response",
            httpx.Response(
                status_code=400,
                json={
                    "code": 400,
                    "reason": "BadRequest",
                    "message": "Invalid request",
                },
            ),
            BadRequestException("Invalid request"),
        ),
        (
            "server error response",
            httpx.Response(
                status_code=500,
                json={
                    "code": 500,
                    "reason": "InternalServerError",
                    "message": "Server error",
                },
            ),
            InternalServerErrorException("Server error"),
        ),
        (
            "not found response",
            httpx.Response(
                status_code=404,
                json={
                    "code": 404,
                    "reason": "NotFound",
                    "message": "Resource not found",
                },
            ),
            NotFoundException("Resource not found"),
        ),
        (
            "client error openai response",
            httpx.Response(
                status_code=400,
                json={
                    "error": {
                        "code": 400,
                        "type": "NotFound",
                        "message": "Invalid request",
                    }
                },
            ),
            BadRequestException("Invalid request"),
        ),
        (
            "server error openai response",
            httpx.Response(
                status_code=500,
                json={
                    "error": {
                        "code": 500,
                        "type": "InternalServerError",
                        "message": "Server error",
                    }
                },
            ),
            InternalServerErrorException("Server error"),
        ),
        (
            "not found openai response",
            httpx.Response(
                status_code=404,
                json={
                    "error": {
                        "code": 404,
                        "type": "NotFound",
                        "message": "Resource not found",
                    }
                },
            ),
            NotFoundException("Resource not found"),
        ),
    ],
)
def test_raise_if_response_error(name, given, expected):
    try:
        raise_if_response_error(given)
        assert expected is None, f"Case {name} expected get exception but none"
    except Exception as e:
        assert str(e) == str(
            expected
        ), f"Case {name} expected exception {expected} but got {e}"


@pytest.mark.asyncio
async def test_upstream_read_error_omits_exception_details(caplog):
    from unittest.mock import AsyncMock
    from gpustack.api.exceptions import HTTPException, async_raise_if_response_error

    response = AsyncMock(status_code=500)
    response.aread.side_effect = httpx.ReadError("credential=private-test-value")
    with pytest.raises(HTTPException) as raised:
        await async_raise_if_response_error(response)
    assert raised.value.status_code == 500
    assert raised.value.message == "Failed to read upstream response: ReadError"
    records = [r for r in caplog.records if r.name == "gpustack.api.exceptions"]
    assert len(records) == 1
    assert records[0].exc_info[1] is response.aread.side_effect
    assert records[0].exc_info[2] is not None
    assert str(response.aread.side_effect) in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [422, 500])
@pytest.mark.parametrize("nested", [False, True])
async def test_chained_http_errors_log_once_without_exposing_diagnostics(
    caplog, status_code, nested
):
    import logging
    from fastapi import FastAPI
    from gpustack.api.exceptions import (
        HTTPException,
        register_handlers,
    )

    app = FastAPI()
    register_handlers(app)
    original = RuntimeError("internal-diagnostic-marker")

    @app.post("/failure")
    async def failure():
        try:
            try:
                raise original
            except RuntimeError as e:
                if nested:
                    raise ValueError("intermediate-diagnostic-marker") from e
                raise
        except Exception as e:
            raise HTTPException(
                status_code=status_code,
                reason="Failure",
                message="Operation failed",
                log_level=logging.WARNING,
            ) from e

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/failure?ticket=query-marker")
    assert response.status_code == status_code
    assert (
        response.json()
        == ErrorResponse(
            code=status_code, reason="Failure", message="Operation failed"
        ).model_dump()
    )
    records = [r for r in caplog.records if r.name == "gpustack.api.exceptions"]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.WARNING
    assert record.exc_info[2] is not None
    cause = record.exc_info[1].__cause__
    assert (cause.__cause__ if nested else cause) is original
    assert "internal-diagnostic-marker" in caplog.text
    assert ("intermediate-diagnostic-marker" in caplog.text) is nested
    assert "path=/failure, method=POST" in record.getMessage()
    assert "query-marker" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("is_openai", [False, True])
@pytest.mark.parametrize("status_code", [400, 500])
@pytest.mark.parametrize("suppress_context", [False, True])
async def test_http_errors_without_explicit_causes_keep_logging_policy(
    caplog, is_openai, status_code, suppress_context
):
    import logging
    from fastapi import FastAPI
    from gpustack.api.exceptions import (
        HTTPException,
        OpenAIAPIException,
        register_handlers,
    )

    app = FastAPI()
    register_handlers(app)
    error_type = OpenAIAPIException if is_openai else HTTPException

    @app.get("/failure")
    async def failure():
        error = error_type(
            status_code=status_code,
            reason="Failure",
            message="Safe message",
            log_level=logging.WARNING,
        )
        if suppress_context:
            try:
                raise RuntimeError("locally-handled-diagnostic-marker")
            except RuntimeError:
                raise error from None
        raise error

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/failure")
    assert response.status_code == status_code
    records = [r for r in caplog.records if r.name == "gpustack.api.exceptions"]
    assert len(records) == (1 if status_code >= 500 and not is_openai else 0)
    if records:
        assert records[0].levelno == logging.WARNING
        assert records[0].exc_info is None
    assert "locally-handled-diagnostic-marker" not in caplog.text
    assert "locally-handled-diagnostic-marker" not in response.text


@pytest.mark.asyncio
async def test_http_error_handler_logs_explicit_exception_outside_except(caplog):
    from fastapi import FastAPI, Request
    from gpustack.api.exceptions import (
        HTTPException,
        register_handlers,
    )

    app = FastAPI()
    register_handlers(app)
    original = RuntimeError("original-diagnostic-marker")
    error = HTTPException(status_code=500, reason="Failure", message="Safe message")
    try:
        try:
            raise original
        except RuntimeError as e:
            raise error from e
    except HTTPException:
        pass

    request = Request(
        {"type": "http", "path": "/failure", "method": "POST", "headers": []}
    )
    response = await app.exception_handlers[HTTPException](request, error)
    assert response.status_code == 500
    record = next(r for r in caplog.records if r.name == "gpustack.api.exceptions")
    assert record.exc_info == (type(error), error, error.__traceback__)
    assert "original-diagnostic-marker" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("chain_cause", [False, True])
async def test_openai_errors_keep_local_traceback_logging(caplog, chain_cause):
    import logging
    from fastapi import FastAPI
    from gpustack.api.exceptions import OpenAIAPIException, register_handlers

    app = FastAPI()
    register_handlers(app)
    route_logger = logging.getLogger("test.openai_route")
    original = RuntimeError("local-diagnostic-marker")

    @app.post("/failure")
    async def failure():
        try:
            raise original
        except RuntimeError as e:
            route_logger.exception("Inference request failed")
            error = OpenAIAPIException(
                status_code=503, reason="ServiceUnavailable", message="Safe message"
            )
            if chain_cause:
                raise error from e
            raise error

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/failure")
    assert response.status_code == 503
    assert response.json() == {
        "error": {"message": "Safe message", "code": 503, "type": "ServiceUnavailable"}
    }
    records = [r for r in caplog.records if r.exc_info]
    assert len(records) == 1
    assert records[0].name == route_logger.name
    assert records[0].exc_info[1] is original
    assert "local-diagnostic-marker" in caplog.text
    assert not any(r.name == "gpustack.api.exceptions" for r in caplog.records)
