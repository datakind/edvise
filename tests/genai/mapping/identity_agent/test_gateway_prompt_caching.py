"""Tests for opt-in Anthropic prompt-caching support in the gateway wrapper."""

from __future__ import annotations

from typing import Any

from edvise.genai.mapping.shared.databricks_ai_gateway import (
    _CACHE_CONTROL_MIN_CHARS,
    build_gateway_message_content,
    llm_complete_combined_message_content,
    make_databricks_gateway_llm_complete,
)

_LONG_SYSTEM = "x" * (_CACHE_CONTROL_MIN_CHARS + 1)
_SHORT_SYSTEM = "short system prompt"


def test_build_gateway_message_content_default_is_plain_string() -> None:
    content = build_gateway_message_content(
        _LONG_SYSTEM, "user text", cache_system_prompt=False, cache_ttl="5m"
    )
    assert content == llm_complete_combined_message_content(_LONG_SYSTEM, "user text")
    assert isinstance(content, str)


def test_build_gateway_message_content_skips_caching_for_short_system() -> None:
    content = build_gateway_message_content(
        _SHORT_SYSTEM, "user text", cache_system_prompt=True, cache_ttl="5m"
    )
    assert isinstance(content, str)
    assert content == llm_complete_combined_message_content(_SHORT_SYSTEM, "user text")


def test_build_gateway_message_content_caches_long_system_with_default_ttl() -> None:
    content = build_gateway_message_content(
        _LONG_SYSTEM, "user text", cache_system_prompt=True, cache_ttl="5m"
    )
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["text"] == _LONG_SYSTEM
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert "user text" in content[1]["text"]
    assert "ttl" not in content[0]["cache_control"]


def test_build_gateway_message_content_caches_long_system_with_1h_ttl() -> None:
    content = build_gateway_message_content(
        _LONG_SYSTEM, "user text", cache_system_prompt=True, cache_ttl="1h"
    )
    assert isinstance(content, list)
    assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.refusal = None


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeUsage:
    def __init__(self, cache_read_input_tokens: int) -> None:
        self._cache_read_input_tokens = cache_read_input_tokens

    def model_dump(self) -> dict[str, Any]:
        return {"cache_read_input_tokens": self._cache_read_input_tokens}


class _FakeResponse:
    def __init__(self, content: str, model: str, usage: Any = None) -> None:
        self.choices = [_FakeChoice(content)]
        self.model = model
        self.usage = usage


class _FakeCompletions:
    def __init__(self, *, usage_cache_read_tokens: int | None = None) -> None:
        self.last_kwargs: dict[str, Any] = {}
        self._usage_cache_read_tokens = usage_cache_read_tokens

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.last_kwargs = kwargs
        usage = (
            _FakeUsage(self._usage_cache_read_tokens)
            if self._usage_cache_read_tokens is not None
            else None
        )
        return _FakeResponse("hello", kwargs["model"], usage=usage)


class _FakeChat:
    def __init__(self, *, usage_cache_read_tokens: int | None = None) -> None:
        self.completions = _FakeCompletions(
            usage_cache_read_tokens=usage_cache_read_tokens
        )


class _FakeClient:
    def __init__(self, *, usage_cache_read_tokens: int | None = None) -> None:
        self.chat = _FakeChat(usage_cache_read_tokens=usage_cache_read_tokens)


def test_make_databricks_gateway_llm_complete_sends_block_content_when_caching_enabled() -> (
    None
):
    client = _FakeClient()
    complete = make_databricks_gateway_llm_complete(
        client,  # type: ignore[arg-type]
        model="claude-sonnet-edvise-genai",
        cache_system_prompt=True,
    )
    result = complete(_LONG_SYSTEM, "user text")
    assert result == "hello"
    sent_messages = client.chat.completions.last_kwargs["messages"]
    assert sent_messages[0]["role"] == "user"
    assert isinstance(sent_messages[0]["content"], list)
    assert sent_messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_make_databricks_gateway_llm_complete_default_unaffected() -> None:
    client = _FakeClient()
    complete = make_databricks_gateway_llm_complete(
        client,  # type: ignore[arg-type]
        model="claude-sonnet-edvise-genai",
    )
    result = complete(_LONG_SYSTEM, "user text")
    assert result == "hello"
    sent_messages = client.chat.completions.last_kwargs["messages"]
    assert isinstance(sent_messages[0]["content"], str)


def test_make_databricks_gateway_llm_complete_logs_cache_usage_at_info(caplog) -> None:
    client = _FakeClient(usage_cache_read_tokens=4242)
    complete = make_databricks_gateway_llm_complete(
        client,  # type: ignore[arg-type]
        model="claude-sonnet-edvise-genai",
        cache_system_prompt=True,
    )
    with caplog.at_level("INFO"):
        result = complete(_LONG_SYSTEM, "user text")
    assert result == "hello"
    assert any(
        "AI Gateway cache usage" in record.getMessage()
        and "cache_read_input_tokens=4242" in record.getMessage()
        for record in caplog.records
    )
