"""Strip markdown code fences from LLM JSON output (IdentityAgent and SchemaMappingAgent)."""

from __future__ import annotations


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :]
    if text.endswith("```"):
        text = text[: text.rindex("```")].rstrip()
    return text


__all__ = ["strip_json_fences"]
