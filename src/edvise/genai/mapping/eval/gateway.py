"""Streaming gateway call shared by Step 2a and Step 2b."""

from __future__ import annotations

import logging
import time
from typing import Any, cast

from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from edvise.genai.mapping.shared.utilities import strip_json_fences

logger = logging.getLogger(__name__)

# Models that support assistant message prefill (for JSON output formatting)
MODELS_WITH_PREFILL = {
    "claude-sonnet-test-genai-ai-data-cleaning",
    "claude-haiku-test-genai-data-cleaning",
}

# Sonnet 5 streams a reasoning block ahead of the answer. It is not JSON.
_SKIPPED_STREAM_BLOCK_TYPES = frozenset({"reasoning", "thinking", "redacted_thinking"})


def _split_stream_content(
    content: Any,
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Split one streamed ``delta.content`` into answer text and reasoning.

    Sonnet 4.5 sends a string. Sonnet 5 and 5.5 send a list of blocks and
    lead with reasoning. Reasoning, thinking, and redacted thinking are kept
    as their own records so they are not parsed as the answer. A refusal
    fails the call.
    """
    if content is None or content == "":
        return "", [], None
    if isinstance(content, str):
        return content, [], None
    blocks = content if isinstance(content, list) else [content]
    text_parts: list[str] = []
    reasoning: list[dict[str, Any]] = []
    for block in blocks:
        text, record, refusal = _split_stream_block(block)
        if refusal:
            return "".join(text_parts), reasoning, refusal
        if record is not None:
            reasoning.append(record)
        elif text:
            text_parts.append(text)
    return "".join(text_parts), reasoning, None


def _split_stream_block(
    block: Any,
) -> tuple[str, dict[str, Any] | None, str | None]:
    if isinstance(block, str):
        return block, None, None
    block_type, text, refusal = _stream_block_fields(block)
    if block_type == "refusal" or refusal:
        message = refusal or (text if isinstance(text, str) and text else "")
        return "", None, message or "The model refused the request."
    if block_type in _SKIPPED_STREAM_BLOCK_TYPES:
        return "", _json_block(block, block_type), None
    if isinstance(text, str):
        return text, None, None
    if block_type not in {None, "text", "output_text"}:
        logger.debug("Skipping streamed content block type %s", block_type)
    return "", None, None


def _json_block(block: Any, block_type: str | None) -> dict[str, Any]:
    """JSON-safe copy of a reasoning, thinking, or redacted-thinking block."""
    if isinstance(block, dict):
        record = {str(key): _json_value(value) for key, value in block.items()}
    elif hasattr(block, "model_dump"):
        dumped = block.model_dump()
        record = (
            {str(key): _json_value(value) for key, value in dumped.items()}
            if isinstance(dumped, dict)
            else {"type": block_type, "value": _json_value(dumped)}
        )
    else:
        record = {
            "type": block_type,
            "text": _json_value(getattr(block, "text", None)),
            "thinking": _json_value(getattr(block, "thinking", None)),
            "summary": _json_value(getattr(block, "summary", None)),
            "data": _json_value(getattr(block, "data", None)),
            "signature": _json_value(getattr(block, "signature", None)),
        }
        record = {key: value for key, value in record.items() if value is not None}
    if block_type and "type" not in record:
        record["type"] = block_type
    return record


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump())
    return str(value)


def _stream_block_fields(block: Any) -> tuple[str | None, Any, str | None]:
    if isinstance(block, dict):
        block_type = block.get("type")
        text = block.get("text")
        refusal = block.get("refusal")
    else:
        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        refusal = getattr(block, "refusal", None)
    type_name = str(block_type) if block_type is not None else None
    refusal_text = refusal if isinstance(refusal, str) and refusal.strip() else None
    return type_name, text, refusal_text


def run_once(
    model: str,
    prompt: str | list[dict[str, Any]],
    client: OpenAI,
    *,
    log_cache_usage: bool = False,
) -> dict[str, Any]:
    """
    ``prompt`` is normally the full ``role="user"`` message text, but callers that want
    Anthropic/Databricks prompt caching (see
    :func:`~edvise.genai.mapping.shared.databricks_ai_gateway.build_gateway_message_content`)
    may instead pass a pre-built list of content blocks (with ``cache_control`` on the
    static block); it is forwarded to the gateway unchanged either way.

    ``log_cache_usage`` requests a final usage-only chunk from the stream (via
    ``stream_options={"include_usage": True}``) and logs any cache token fields on it (see
    :func:`~edvise.genai.mapping.shared.databricks_ai_gateway.log_gateway_cache_usage_if_present`).
    Only set this for calls that opted into caching via ``prompt``'s ``cache_control``.
    """
    from edvise.genai.mapping.shared.databricks_ai_gateway import (
        log_gateway_cache_usage_if_present,
    )

    start = time.perf_counter()
    reasoning: list[dict[str, Any]] = []
    try:
        full_text = ""
        final_usage: Any = None

        # Build messages - only add assistant prefill for models that support it
        messages_list: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        if model in MODELS_WITH_PREFILL:
            messages_list.append(
                {"role": "assistant", "content": "{"}
            )  # prefill for JSON
        messages = cast(list[ChatCompletionMessageParam], messages_list)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 16000,
            "stream": True,
        }
        if log_cache_usage:
            create_kwargs["stream_options"] = {"include_usage": True}
        resp = client.chat.completions.create(**create_kwargs)
        for chunk in resp:
            # Safely extract content from chunk - handle cases where choices might be empty
            try:
                if not isinstance(chunk, ChatCompletionChunk):
                    continue
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    final_usage = chunk_usage
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                text, blocks, refusal = _split_stream_content(
                    chunk.choices[0].delta.content
                )
                reasoning.extend(blocks)
                if refusal:
                    raise ValueError(refusal)
                if text:
                    full_text += text
            except (IndexError, AttributeError) as e:
                # Skip chunks with malformed structure - this can happen with some models
                logger.debug(f"Skipping malformed chunk: {e}")
                continue

        if log_cache_usage:
            log_gateway_cache_usage_if_present(final_usage, log=logger, model=model)

        if not full_text.strip():
            if reasoning:
                raise ValueError(
                    "Model returned reasoning but no answer text. "
                    "The token budget was spent before the answer."
                )
            raise ValueError("Model returned no answer text.")

        latency_s = time.perf_counter() - start

        # For models with prefill, reattach the brace if needed; for others, just strip fences
        if model in MODELS_WITH_PREFILL:
            # The model's response continues from the prefill "{", so it may already start with "{"
            # Only prepend "{" if the response doesn't already start with it (after stripping whitespace)
            full_text_stripped = full_text.lstrip()
            if full_text_stripped.startswith("{"):
                raw = strip_json_fences(full_text)
            else:
                raw = strip_json_fences("{" + full_text)
        else:
            raw = strip_json_fences(full_text)
        return {
            "model": model,
            "prompt": prompt,
            "success": True,
            "latency_s": round(latency_s, 3),
            "response_chars": len(raw),
            "response": raw,
            "reasoning": reasoning,
            "error": None,
        }
    except Exception as e:
        # Handle all errors - extract as much detail as possible
        latency_s = time.perf_counter() - start
        error_msg = str(e)
        error_type = type(e).__name__

        # Try to extract HTTP status code and response body
        status_code = getattr(e, "status_code", None)
        error_details = error_msg

        # Check for response body in various formats
        response_body = None
        if hasattr(e, "response"):
            if hasattr(e.response, "text"):
                try:
                    response_body = e.response.text
                except Exception:
                    pass
            elif hasattr(e.response, "json"):
                try:
                    response_body = str(e.response.json())
                except Exception:
                    pass

        if not response_body and hasattr(e, "body"):
            try:
                response_body = str(e.body)
            except Exception:
                pass

        # Build detailed error message
        if status_code:
            error_details = f"HTTP {status_code}: {error_msg}"
        else:
            error_details = f"{error_type}: {error_msg}"

        if response_body:
            error_details = f"{error_details}\nResponse body: {response_body}"

        # Also check for additional error attributes
        if hasattr(e, "message") and e.message != error_msg:
            error_details = f"{error_details}\nMessage: {e.message}"

        logger.error(f"Error calling model {model}: {error_details}")

        return {
            "model": model,
            "prompt": prompt,
            "success": False,
            "latency_s": round(latency_s, 3),
            "response_chars": None,
            "response": None,
            "reasoning": reasoning,
            "error": error_details,
        }
