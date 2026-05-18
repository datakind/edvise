"""Unit tests for ``edvise.utils.llm_utils``."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from edvise.utils.llm_utils import (
    LLMRetryExhausted,
    call_with_retry,
    llm_complete_with_parse_retry,
)


def test_llm_complete_with_parse_retry_appends_hint_to_user_on_validation() -> None:
    calls: list[tuple[str, str]] = []

    def llm_complete(system: str, user: str) -> str:
        calls.append((system, user))
        if len(calls) == 1:
            return "[]"
        return '{"a": 1}'

    def parse_fn(s: str) -> dict:
        from pydantic import TypeAdapter

        return TypeAdapter(dict[str, Any]).validate_json(s)

    out = llm_complete_with_parse_retry(
        llm_complete, "s", "base u", parse_fn, max_retries=3
    )
    assert out == {"a": 1}
    assert len(calls) == 2
    assert calls[0] == ("s", "base u")
    _sys, u2 = calls[1]
    assert _sys == "s"
    assert u2.startswith("base u\n\nYour previous response was:")


def test_clean_success_on_first_attempt() -> None:
    calls: list[str | None] = []

    def call_fn(hint: str | None) -> str:
        calls.append(hint)
        return '{"x": 1}'

    def parse_fn(text: str) -> int:
        d = json.loads(text)
        return d["x"]

    assert call_with_retry(call_fn, parse_fn, max_retries=3) == 1
    assert calls == [None]


def test_json_decode_error_on_empty_uses_hint_on_next_call() -> None:
    calls: list[str | None] = []

    def call_fn(hint: str | None) -> str:
        calls.append(hint)
        if len(calls) == 1:
            return ""
        return '{"ok": true}'

    def parse_fn(text: str) -> dict:
        return json.loads(text)

    out = call_with_retry(call_fn, parse_fn, max_retries=3)
    assert out == {"ok": True}
    assert calls[0] is None
    assert calls[1] is not None
    assert "empty" in (calls[1] or "")


def test_json_decode_error_resolves_on_retry_with_no_hint() -> None:
    calls: list[str | None] = []
    good = '{"ok": true}'

    def call_fn(hint: str | None) -> str:
        calls.append(hint)
        if len(calls) == 1:
            return "not valid json{"
        return good

    def parse_fn(text: str) -> dict:
        return json.loads(text)

    out = call_with_retry(call_fn, parse_fn, max_retries=3)
    assert out == {"ok": True}
    assert calls == [None, None]


def test_validation_error_passes_raw_response_and_error_in_next_call() -> None:
    class _Model(BaseModel):
        n: int = Field(ge=0)

    calls: list[str | None] = []

    def call_fn(hint: str | None) -> str:
        calls.append(hint)
        if len(calls) == 1:
            return json.dumps({"n": -1})
        return json.dumps({"n": 0})

    def parse_fn(text: str) -> _Model:
        return _Model.model_validate_json(text)

    result = call_with_retry(call_fn, parse_fn, max_retries=3)
    assert result.n == 0
    assert len(calls) == 2
    assert calls[0] is None
    h = calls[1]
    assert h is not None
    assert "Your previous response was:" in h
    assert '{"n": -1}' in h or "-1" in h
    assert "It failed validation with this error:" in h
    assert "Return a corrected version." in h


def test_exhausted_retries_raises_for_json_with_correct_fields() -> None:
    def call_fn(_hint: str | None) -> str:
        return "bad{"

    def parse_fn(text: str) -> object:
        json.loads(text)
        return object()

    with pytest.raises(LLMRetryExhausted) as ctx:
        call_with_retry(call_fn, parse_fn, max_retries=2, logger=logging.getLogger("t"))

    err = ctx.value
    assert isinstance(err.last_error, json.JSONDecodeError)
    assert err.last_raw_response == "bad{"


def test_exhausted_retries_raises_for_validation_with_correct_fields() -> None:
    class _M(BaseModel):
        a: int

    def call_fn2(_h: str | None) -> str:
        return json.dumps({})

    def parse_fn2(text: str) -> _M:
        return _M.model_validate_json(text)

    with pytest.raises(LLMRetryExhausted) as ctx2:
        call_with_retry(call_fn2, parse_fn2, max_retries=1)

    err2 = ctx2.value
    assert isinstance(err2.last_error, ValidationError)
    assert err2.last_raw_response == json.dumps({})
