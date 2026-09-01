"""Tests for MLflow AI Gateway URL resolution."""

from __future__ import annotations

import pytest

from edvise.genai.mapping.shared import databricks_ai_gateway as dg


def test_resolve_ai_gateway_base_url_explicit_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AI_GATEWAY_BASE_URL", "https://custom.example/ai-gateway/mlflow/v1"
    )
    assert (
        dg.resolve_ai_gateway_base_url()
        == "https://custom.example/ai-gateway/mlflow/v1"
    )


def test_resolve_ai_gateway_base_url_from_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_GATEWAY_BASE_URL", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "dbc-staging.gcp.databricks.com")

    assert (
        dg.resolve_ai_gateway_base_url()
        == "https://dbc-staging.gcp.databricks.com/ai-gateway/mlflow/v1"
    )


def test_resolve_ai_gateway_base_url_raises_without_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_GATEWAY_BASE_URL", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)

    with pytest.raises(ValueError, match="Cannot resolve MLflow AI Gateway"):
        dg.resolve_ai_gateway_base_url()
