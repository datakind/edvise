"""Tests for opt-in prompt-caching support in the SMA streaming `run_once` path.

Covers the actual production call path for Step 2a / refinement / Step 2b
(`edvise_genai_sma._sma_llm_complete_run_once` -> `eval.run_once`), as opposed to
`make_databricks_gateway_llm_complete` (used by IA and SMA grain resolution only).
"""

from __future__ import annotations

from typing import Any

from edvise.genai.mapping.schema_mapping_agent.manifest.eval import run_once
from edvise.genai.mapping.scripts.edvise_genai_sma import _sma_llm_complete_run_once
from edvise.genai.mapping.shared.databricks_ai_gateway import _CACHE_CONTROL_MIN_CHARS

_LONG_SYSTEM = "x" * (_CACHE_CONTROL_MIN_CHARS + 1)
_SHORT_SYSTEM = "short system prompt"


class _FakeDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChunkChoice:
    def __init__(self, content: str) -> None:
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str, usage: Any = None) -> None:
        self.choices = [_FakeChunkChoice(content)] if content else []
        self.usage = usage


class _FakeUsage:
    def __init__(self, cache_read_input_tokens: int) -> None:
        self._cache_read_input_tokens = cache_read_input_tokens

    def model_dump(self) -> dict[str, Any]:
        return {"cache_read_input_tokens": self._cache_read_input_tokens}


class _FakeCompletions:
    def __init__(self, *, usage_cache_read_tokens: int | None = None) -> None:
        self.last_kwargs: dict[str, Any] = {}
        self._usage_cache_read_tokens = usage_cache_read_tokens

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        chunks = [_FakeChunk('{"ok": true}')]
        if kwargs.get("stream_options", {}).get("include_usage") and (
            self._usage_cache_read_tokens is not None
        ):
            chunks.append(
                _FakeChunk("", usage=_FakeUsage(self._usage_cache_read_tokens))
            )
        return _FakeStreamOfChunks(chunks)


class _FakeStreamOfChunks:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)


class _FakeChat:
    def __init__(self, *, usage_cache_read_tokens: int | None = None) -> None:
        self.completions = _FakeCompletions(
            usage_cache_read_tokens=usage_cache_read_tokens
        )


class _FakeClient:
    def __init__(self, *, usage_cache_read_tokens: int | None = None) -> None:
        self.chat = _FakeChat(usage_cache_read_tokens=usage_cache_read_tokens)


def test_run_once_accepts_plain_string_prompt(monkeypatch) -> None:
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient()
    result = run_once("claude-sonnet-edvise-genai", "plain prompt", client)
    assert result["success"] is True
    assert (
        client.chat.completions.last_kwargs["messages"][0]["content"] == "plain prompt"
    )


def test_run_once_accepts_content_block_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient()
    blocks = [
        {"type": "text", "text": _LONG_SYSTEM, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "user text"},
    ]
    result = run_once("claude-sonnet-edvise-genai", blocks, client)
    assert result["success"] is True
    sent_content = client.chat.completions.last_kwargs["messages"][0]["content"]
    assert sent_content == blocks


def test_sma_llm_complete_run_once_caches_long_system_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient()
    complete = _sma_llm_complete_run_once(
        client, catalog="test_catalog", cache_system_prompt=True
    )
    result = complete(_LONG_SYSTEM, "user text")
    assert result == '{"ok": true}'
    sent_content = client.chat.completions.last_kwargs["messages"][0]["content"]
    assert isinstance(sent_content, list)
    assert sent_content[0]["cache_control"] == {"type": "ephemeral"}
    assert sent_content[0]["text"] == _LONG_SYSTEM


def test_sma_llm_complete_run_once_default_unaffected(monkeypatch) -> None:
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient()
    complete = _sma_llm_complete_run_once(client, catalog="test_catalog")
    result = complete(_LONG_SYSTEM, "user text")
    assert result == '{"ok": true}'
    sent_content = client.chat.completions.last_kwargs["messages"][0]["content"]
    assert isinstance(sent_content, str)


def test_sma_llm_complete_run_once_caching_is_noop_for_empty_system(
    monkeypatch,
) -> None:
    """Step 2a/2b call with system="" - caching flag must not change that path."""
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient()
    complete = _sma_llm_complete_run_once(
        client, catalog="test_catalog", cache_system_prompt=True
    )
    result = complete("", "user only prompt")
    assert result == '{"ok": true}'
    sent_content = client.chat.completions.last_kwargs["messages"][0]["content"]
    assert sent_content == "user only prompt"


def test_sma_llm_complete_run_once_skips_caching_for_short_system(monkeypatch) -> None:
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient()
    complete = _sma_llm_complete_run_once(
        client, catalog="test_catalog", cache_system_prompt=True
    )
    result = complete(_SHORT_SYSTEM, "user text")
    assert result == '{"ok": true}'
    sent_content = client.chat.completions.last_kwargs["messages"][0]["content"]
    assert isinstance(sent_content, str)


def test_run_once_requests_stream_usage_when_log_cache_usage_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient(usage_cache_read_tokens=1234)
    result = run_once(
        "claude-sonnet-edvise-genai", "prompt", client, log_cache_usage=True
    )
    assert result["success"] is True
    assert client.chat.completions.last_kwargs["stream_options"] == {
        "include_usage": True
    }


def test_run_once_logs_cache_usage_at_info_when_enabled(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient(usage_cache_read_tokens=1234)
    with caplog.at_level("INFO"):
        result = run_once(
            "claude-sonnet-edvise-genai", "prompt", client, log_cache_usage=True
        )
    assert result["success"] is True
    assert any(
        "AI Gateway cache usage" in record.getMessage()
        and "cache_read_input_tokens=1234" in record.getMessage()
        for record in caplog.records
    )


def test_run_once_does_not_request_stream_usage_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient()
    result = run_once("claude-sonnet-edvise-genai", "prompt", client)
    assert result["success"] is True
    assert "stream_options" not in client.chat.completions.last_kwargs


def test_sma_llm_complete_run_once_logs_cache_usage_for_cached_call(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient(usage_cache_read_tokens=5678)
    complete = _sma_llm_complete_run_once(
        client, catalog="test_catalog", cache_system_prompt=True
    )
    with caplog.at_level("INFO"):
        result = complete(_LONG_SYSTEM, "user text")
    assert result == '{"ok": true}'
    assert any(
        "AI Gateway cache usage" in record.getMessage()
        and "cache_read_input_tokens=5678" in record.getMessage()
        for record in caplog.records
    )


def test_sma_llm_complete_run_once_does_not_log_for_step2a_style_call(
    monkeypatch, caplog
) -> None:
    """Step 2a/2b call with system="" must not request/log cache usage even when enabled."""
    monkeypatch.setattr(
        "edvise.genai.mapping.schema_mapping_agent.manifest.eval.ChatCompletionChunk",
        _FakeChunk,
    )
    client = _FakeClient(usage_cache_read_tokens=999)
    complete = _sma_llm_complete_run_once(
        client, catalog="test_catalog", cache_system_prompt=True
    )
    with caplog.at_level("INFO"):
        result = complete("", "user only prompt")
    assert result == '{"ok": true}'
    assert "stream_options" not in client.chat.completions.last_kwargs
    assert not any(
        "AI Gateway cache usage" in record.getMessage() for record in caplog.records
    )
