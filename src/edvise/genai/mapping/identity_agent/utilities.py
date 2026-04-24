"""
Shared helpers for IdentityAgent prompts and JSON parsing.

- :func:`strip_json_fences` — strip markdown code fences from model output.
- :func:`concat_model_sources` — SMA-style ``inspect.getsource`` concatenation for system prompts.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :]
    if text.endswith("```"):
        text = text[: text.rindex("```")].rstrip()
    return text


def concat_model_sources(classes: Sequence[type]) -> str:
    """Concatenate Python source for a sequence of types (typically Pydantic models)."""
    sections: list[str] = []
    for cls in classes:
        try:
            sections.append(inspect.getsource(cls))
        except (OSError, TypeError):
            continue
    return "\n\n".join(sections)


__all__ = ["concat_model_sources", "strip_json_fences"]
