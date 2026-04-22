"""Tests for OpenAI / Databricks gateway retry helpers."""

from __future__ import annotations

import httpx
import pytest

from edvise.genai.mapping.identity_agent.grain_inference.databricks_gateway import (
    gateway_run_once_error_text_is_retryable,
    is_retryable_openai_gateway_error,
    wrap_llm_complete_with_retries,
)


def _resp(status: int) -> httpx.Response:
    req = httpx.Request("POST", "https://example.invalid/mlflow/v1/chat/completions")
    return httpx.Response(status, request=req)


def test_is_retryable_403() -> None:
    import openai

    e = openai.PermissionDeniedError("nope", response=_resp(403), body=None)
    assert is_retryable_openai_gateway_error(e) is True


def test_is_retryable_401_false() -> None:
    import openai

    e = openai.AuthenticationError("bad", response=_resp(401), body=None)
    assert is_retryable_openai_gateway_error(e) is False


def test_gateway_run_once_error_text_retryable() -> None:
    assert gateway_run_once_error_text_is_retryable("HTTP 403: denied") is True
    assert gateway_run_once_error_text_is_retryable("HTTP 429: rate") is True
    assert gateway_run_once_error_text_is_retryable("HTTP 401: nope") is False
    assert gateway_run_once_error_text_is_retryable("validation failed") is False


def test_wrap_llm_complete_with_retries_eventually_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    calls: list[int] = []

    def flaky(system: str, user: str) -> str:
        calls.append(1)
        if len(calls) < 3:
            raise openai.InternalServerError(
                "retry me",
                response=_resp(503),
                body=None,
            )
        return "ok"

    monkeypatch.setattr(
        "edvise.genai.mapping.identity_agent.grain_inference.databricks_gateway.time.sleep",
        lambda _s: None,
    )
    wrapped = wrap_llm_complete_with_retries(flaky, max_attempts=5, log=None)
    assert wrapped("s", "u") == "ok"
    assert len(calls) == 3


def test_wrap_llm_complete_with_retries_gives_up_on_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    def bad(_s: str, _u: str) -> str:
        raise openai.AuthenticationError("bad", response=_resp(401), body=None)

    monkeypatch.setattr(
        "edvise.genai.mapping.identity_agent.grain_inference.databricks_gateway.time.sleep",
        lambda _s: None,
    )
    wrapped = wrap_llm_complete_with_retries(bad, max_attempts=5, log=None)
    with pytest.raises(openai.AuthenticationError):
        wrapped("s", "u")
