"""Tests for :mod:`edvise.genai.mapping.shared.databricks_gateway`."""

from __future__ import annotations

import pytest

from edvise.genai.mapping.shared.databricks_gateway import require_databricks_token


def test_require_databricks_token_ignores_databricks_token_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PAT-style ``DATABRICKS_TOKEN`` must not be used for the gateway client."""
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi_should_not_be_used")

    class _FakeConfig:
        def authenticate(self) -> dict[str, str]:
            return {"Authorization": "Bearer oauth-from-sdk"}

    monkeypatch.setattr(
        "databricks.sdk.core.Config",
        lambda **_k: _FakeConfig(),
    )

    assert require_databricks_token() == "oauth-from-sdk"


def test_require_databricks_token_sdk_bearer(
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
