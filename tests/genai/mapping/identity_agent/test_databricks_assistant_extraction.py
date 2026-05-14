"""Tests for Databricks gateway assistant message extraction (no real HTTP)."""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Any

import pytest

from edvise.genai.mapping.shared import databricks_ai_gateway as dg


def _client_returning(
    content: str | None,
    *,
    refusal: str | None = None,
    finish_reason: str = "stop",
) -> Any:
    msg = SimpleNamespace(content=content, refusal=refusal)
    ch0 = SimpleNamespace(message=msg, finish_reason=finish_reason)

    def model_dump() -> dict[str, int]:
        return {"prompt_tokens": 1, "completion_tokens": 0}

    u = SimpleNamespace()
    u.model_dump = model_dump
    return SimpleNamespace(choices=[ch0], model="test-model", usage=u)


def test_empty_content_raises_informative(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    resp = _client_returning(None, finish_reason="length")
    with pytest.raises(RuntimeError, match="empty assistant message"):
        dg._assistant_text_from_chat_completion_or_raise(  # noqa: SLF001
            resp,
            log=logging.getLogger("test"),
        )
    assert re.search("finish_reason=.+length", caplog.text) is not None


def test_refusal_raises() -> None:
    resp = _client_returning(None, refusal="I cannot", finish_reason="stop")
    with pytest.raises(RuntimeError, match="[Rr]efus"):
        dg._assistant_text_from_chat_completion_or_raise(  # noqa: SLF001
            resp,
            log=logging.getLogger("test"),
        )


def test_list_content_is_concatenated() -> None:
    assert (
        dg._text_from_message_content(  # noqa: SLF001
            [
                {"type": "text", "text": "a"},
            ]
        )
        == "a"
    )


def test_make_llm_complete_raises_on_empty_200() -> None:
    class _Chat:
        def __init__(self) -> None:
            self.completions = self

        def create(  # noqa: PLR0913
            self, *, model: str, messages: list, max_tokens: int, **_k: object
        ) -> Any:
            return _client_returning(None, finish_reason="length")

    class _Cl:
        def __init__(self) -> None:
            self.chat = _Chat()

    c = dg.make_databricks_gateway_llm_complete(_Cl(), model="m", max_tokens=8)
    with pytest.raises(RuntimeError, match="empty"):
        c("a", "b")
