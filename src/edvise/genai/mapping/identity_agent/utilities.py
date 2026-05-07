"""
Shared helpers for IdentityAgent prompts and JSON parsing.

- :func:`strip_json_fences` — strip markdown code fences from model output.
- :func:`concat_model_sources` — SMA-style ``inspect.getsource`` concatenation for system prompts.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Sequence
from pathlib import Path


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


def get_top_level_assign_source(
    source_path: str,
    name: str,
    *,
    include_leading_comment: bool = True,
) -> str:
    """
    Return the source of a top-level ``name = ...`` assignment in *source_path*.

    Used to inject type aliases and constants that Pydantic models refer to by name
    (``getsource`` on a class does not include module-level assignments).
    When *include_leading_comment* is true, a single comment line directly above the
    assignment is included if present.
    """
    path = Path(source_path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    mod = ast.parse(text)
    for node in mod.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                start = node.lineno - 1
                if (
                    include_leading_comment
                    and start > 0
                    and lines[start - 1].lstrip().startswith("#")
                ):
                    start -= 1
                return "\n".join(lines[start : node.end_lineno])
    return ""


__all__ = [
    "concat_model_sources",
    "get_top_level_assign_source",
    "strip_json_fences",
]
