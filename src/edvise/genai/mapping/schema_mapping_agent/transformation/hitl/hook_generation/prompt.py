"""LLM prompts and user payloads for Step 2b transform ``hook_required`` HookSpec generation."""

from __future__ import annotations

import json
import re
from typing import Any, Literal


def _slug_target(target_field: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", (target_field or "").strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "field"


def sma_transform_hook_item_id(
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    target_field: str,
) -> str:
    """Stable id aligned with :class:`~edvise.genai.mapping.schema_mapping_agent.transformation.hitl.schemas.SMATransformationHookHITLItem` envelopes."""
    return f"{institution_id}_{entity_type}_{_slug_target(target_field)}_hook_required"


def manifest_mapping_for_target(
    manifest_map: dict[str, Any],
    entity_type: Literal["cohort", "course"],
    target_field: str,
) -> dict[str, Any] | None:
    manifests = manifest_map.get("manifests")
    if not isinstance(manifests, dict):
        return None
    section = manifests.get(entity_type)
    if not isinstance(section, dict):
        return None
    mappings = section.get("mappings")
    if not isinstance(mappings, list):
        return None
    for rec in mappings:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("target_field") or "").strip() == target_field:
            return rec
    return None


def build_sma_transform_hook_system_prompt() -> str:
    return """
You are a code-generation assistant for **Schema Mapping Agent** field transforms.

The pipeline executes utilities step-by-step on a pandas Series. When the mapping agent sets
``hook_required`` on a plan, a **custom Python function** must implement the remaining
Series → Series logic for one target field.

Respond with **one JSON object only** — no markdown fences, no preamble. The object must validate as HookSpec (omit ``file`` — the pipeline assigns ``transform_hooks.py``):

{
  "functions": [
    {
      "name": "<must match the required_function_name in the user JSON>",
      "signature": "<optional copy of the def line for display>",
      "description": "<intent for reviewers>",
      "draft": "<complete Python function: full def line + indented body>"
    }
  ]
}

Rules:
- The user message specifies **required_function_name**. Emit **exactly one** function in ``functions`` whose ``name`` and ``draft`` ``def`` line use that name.
- **Signature:** ``def <required_function_name>(s: "pd.Series") -> "pd.Series":`` — use **quoted** annotations if you ``import pandas as pd`` inside the body (same rule as IdentityAgent hooks).
- ``s`` is the **resolved source column** as a pandas Series (same length as the base cohort/course table grain). Return a Series of the same length (aligned index).
- **draft** must be syntactically valid Python (``ast.parse``). Put imports **inside** the function body.
- Implement the semantics described in ``reviewer_notes``, ``validation_notes``, existing ``steps`` (if any), and the manifest mapping snippet — they document what the target Edvise field needs.
- Do **not** include a ``file`` field.
- Output must be parseable JSON (double quotes, no trailing commas).
""".strip()


def build_sma_transform_hook_user_message(
    *,
    item_id: str,
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    target_field: str,
    required_function_name: str,
    plan: dict[str, Any],
    manifest_record: dict[str, Any] | None,
) -> str:
    payload: dict[str, Any] = {
        "item_id": item_id,
        "institution_id": institution_id,
        "entity_type": entity_type,
        "target_field": target_field,
        "required_function_name": required_function_name,
        "field_transformation_plan": dict(plan),
        "manifest_mapping_record": manifest_record,
    }
    return json.dumps(payload, indent=2)


__all__ = [
    "build_sma_transform_hook_system_prompt",
    "build_sma_transform_hook_user_message",
    "manifest_mapping_for_target",
    "sma_transform_hook_item_id",
]
