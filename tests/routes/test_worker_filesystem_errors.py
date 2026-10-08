"""Filesystem failures expose their category without internal path details."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from gpustack.routes.worker import filesystem


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,status,message",
    [
        (FileNotFoundError, 404, "Model file not found"),
        (NotADirectoryError, 400, "Model path is not a directory"),
        (PermissionError, 403, "Permission denied reading model files"),
        (OSError, 500, "Failed to calculate size: OSError"),
    ],
)
async def test_model_size_errors_keep_status_and_safe_description(
    monkeypatch, caplog, error_type, status, message
):
    monkeypatch.setattr(filesystem, "validate_path_security", lambda path: path)
    monkeypatch.setattr(filesystem.os.path, "exists", lambda path: True)
    monkeypatch.setattr(filesystem.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(filesystem, "is_diffusion_model", lambda path: False)
    monkeypatch.setattr(
        filesystem,
        "calculate_local_model_weight_size",
        MagicMock(side_effect=error_type("/internal/private-test-value")),
    )
    with pytest.raises(HTTPException) as raised:
        await filesystem.get_model_weight_size("/models/test")
    assert raised.value.status_code == status
    assert raised.value.detail == message
    records = [r for r in caplog.records if r.name == filesystem.logger.name]
    assert len(records) == 1
    assert isinstance(records[0].exc_info[1], error_type)
    assert records[0].exc_info[2] is not None
    assert "/internal/private-test-value" in caplog.text
