"""Tests for :mod:`edvise.genai.mapping.identity_agent.grain_inference.databricks_gateway`."""

from __future__ import annotations

import pytest

from edvise.genai.mapping.identity_agent.grain_inference.databricks_gateway import (
    require_databricks_token,
)


def test_require_databricks_token_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi_from_env")
    assert require_databricks_token() == "dapi_from_env"


def test_require_databricks_token_sdk_bearer_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    class _FakeConfig:
        def authenticate(self) -> dict[str, str]:
            return {"Authorization": "Bearer oauth-from-sdk"}

    monkeypatch.setattr(
        "databricks.sdk.core.Config",
        lambda **_k: _FakeConfig(),
    )

    assert require_databricks_token() == "oauth-from-sdk"


def test_require_databricks_token_raises_when_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    class _FakeConfig:
        def authenticate(self) -> dict[str, str]:
            return {}

    monkeypatch.setattr(
        "databricks.sdk.core.Config",
        lambda **_k: _FakeConfig(),
    )

    with pytest.raises(ValueError, match="No Databricks workspace token"):
        require_databricks_token()
