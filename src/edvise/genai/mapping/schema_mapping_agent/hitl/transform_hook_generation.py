"""
LLM HookSpec generation for Schema Mapping Agent Step 2b ``hook_required`` field plans.

Mirrors IdentityAgent hook generation shape; materializes under the SMA run folder as
``transform_hooks.py`` after UC ``sma_gate_2_hook_preview`` approval.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from edvise.utils.llm_utils import llm_complete_with_parse_retry

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec
from edvise.genai.mapping.identity_agent.hitl.hook_generation.parse import parse_hook_spec
from edvise.genai.mapping.identity_agent.hitl.hook_generation.paths import (
    ensure_hook_spec_file,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain
from edvise.genai.mapping.schema_mapping_agent.hitl.transformation_hook_hitl import (
    iter_hook_required_plans as _iter_hook_required_plans,
)

logger = logging.getLogger(__name__)


def _slug_target(target_field: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", (target_field or "").strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "field"


def sma_transform_hook_item_id(
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    target_field: str,
) -> str:
    """Stable id aligned with :mod:`transformation_hook_hitl` envelopes."""
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


def generate_sma_transform_hook_spec(
    *,
    item_id: str,
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    target_field: str,
    plan: dict[str, Any],
    manifest_record: dict[str, Any] | None,
    llm_complete: Callable[[str, str], str],
) -> HookSpec:
    slug = _slug_target(target_field)
    fn_name = f"transform_{entity_type}_{slug}"
    system = build_sma_transform_hook_system_prompt()
    user = build_sma_transform_hook_user_message(
        item_id=item_id,
        institution_id=institution_id,
        entity_type=entity_type,
        target_field=target_field,
        required_function_name=fn_name,
        plan=plan,
        manifest_record=manifest_record,
    )
    spec = llm_complete_with_parse_retry(
        llm_complete,
        system,
        user,
        parse_hook_spec,
        logger=logger,
    )
    spec = ensure_hook_spec_file(
        spec,
        institution_id=institution_id,
        domain=HITLDomain.TRANSFORM,
    )
    return spec


def generate_sma_transform_hook_preview_rows_for_entity(
    transformation_data: dict[str, Any],
    manifest_map: dict[str, Any],
    *,
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    llm_complete: Callable[[str, str], str],
) -> list[dict[str, Any]]:
    """
    One LLM call per ``hook_required`` plan; returns rows for
    :func:`write_sma_transform_hook_preview_json`.
    """
    rows: list[dict[str, Any]] = []
    for plan in _iter_hook_required_plans(transformation_data, entity_type):
        tf = str(plan.get("target_field") or "").strip()
        item_id = sma_transform_hook_item_id(institution_id, entity_type, tf)
        mrec = manifest_mapping_for_target(manifest_map, entity_type, tf)
        try:
            spec = generate_sma_transform_hook_spec(
                item_id=item_id,
                institution_id=institution_id,
                entity_type=entity_type,
                target_field=tf,
                plan=plan,
                manifest_record=mrec,
                llm_complete=llm_complete,
            )
        except Exception:
            logger.exception(
                "SMA transform hook generation failed for %s %s", entity_type, tf
            )
            raise
        rows.append(
            {
                "item_id": item_id,
                "hook_spec": spec.model_dump(mode="json"),
                "review_context": {
                    "entity_type": entity_type,
                    "target_field": tf,
                    "manifest_mapping_record": mrec,
                    "field_transformation_plan": dict(plan),
                },
            }
        )
    return rows


def load_hook_specs_from_sma_preview_path(path: str | Path) -> list[HookSpec]:
    """Read a written preview JSON and return HookSpec instances (for materialize)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    specs_raw = data.get("specs")
    if not isinstance(specs_raw, list):
        return []
    out: list[HookSpec] = []
    for row in specs_raw:
        if not isinstance(row, dict):
            continue
        hs = row.get("hook_spec")
        if not isinstance(hs, dict):
            continue
        out.append(HookSpec.model_validate(hs))
    return out


__all__ = [
    "build_sma_transform_hook_system_prompt",
    "build_sma_transform_hook_user_message",
    "generate_sma_transform_hook_preview_rows_for_entity",
    "generate_sma_transform_hook_spec",
    "load_hook_specs_from_sma_preview_path",
    "manifest_mapping_for_target",
    "sma_transform_hook_item_id",
]
