"""Serialized HookSpec previews for UC-gated human review before apply/materialize."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec


def assemble_hook_spec_drafts_as_module_text(hook_spec: dict[str, Any]) -> str:
    """
    Concatenate ``functions[].draft`` in order, separated by a blank line — same body layout as
    :func:`~edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize.materialize_hook_spec_to_file`
    (without the generated comment header). Used for reviewer-facing module previews.
    """
    functions = hook_spec.get("functions")
    if not isinstance(functions, list):
        return ""
    parts: list[str] = []
    for fn in functions:
        if not isinstance(fn, dict):
            continue
        draft = (fn.get("draft") or "").strip()
        if draft:
            parts.append(draft)
    return "\n\n".join(parts).strip()


def write_identity_hook_preview_json(
    *,
    output_path: str | Path,
    institution_id: str,
    domain: str,
    specs: list[tuple[str, HookSpec]],
) -> None:
    """
    Write a JSON artifact listing generated hook specs (``item_id`` + ``hook_spec``) for HITL review.

    ``domain`` is ``identity_grain`` or ``identity_term`` (informative for reviewers / UI).
    """
    path = Path(output_path)
    payload: dict[str, Any] = {
        "institution_id": institution_id,
        "domain": domain,
        "specs": [
            {
                "item_id": item_id,
                "hook_spec": spec.model_dump(mode="json"),
            }
            for item_id, spec in specs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["assemble_hook_spec_drafts_as_module_text", "write_identity_hook_preview_json"]
