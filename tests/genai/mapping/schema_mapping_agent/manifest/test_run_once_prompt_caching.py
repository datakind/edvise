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
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChunkChoice(content)]


class _FakeStream:
    def __init__(self, text: str) -> None:
        self._text = text

    def __iter__(self):
        yield self._text


class _FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = kwargs
        return _FakeStreamOfChunks('{"ok": true}')


class _FakeStreamOfChunks:
    def __init__(self, text: str) -> None:
        self._text = text

    def __iter__(self):
        return iter([_FakeChunk(self._text)])


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


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
    complete = _sma_llm_complete_run_once(client, cache_system_prompt=True)
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
    complete = _sma_llm_complete_run_once(client)
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
    complete = _sma_llm_complete_run_once(client, cache_system_prompt=True)
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
    complete = _sma_llm_complete_run_once(client, cache_system_prompt=True)
    result = complete(_SHORT_SYSTEM, "user text")
    assert result == '{"ok": true}'
    sent_content = client.chat.completions.last_kwargs["messages"][0]["content"]
    assert isinstance(sent_content, str)
