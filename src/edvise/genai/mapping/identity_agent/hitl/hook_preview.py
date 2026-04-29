"""Serialized HookSpec previews for UC-gated human review before apply/materialize."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec
from edvise.genai.mapping.identity_agent.hitl.schemas import GrainAmbiguityHITLContext, HITLItem


def _hitl_context_json(ctx: Any) -> Any:
    if ctx is None:
        return None
    if isinstance(ctx, GrainAmbiguityHITLContext):
        return ctx.model_dump(mode="json")
    return ctx


def hook_slug_from_item_id(item_id: str, *, institution_id: str | None = None) -> str:
    """
    Build a stable snake_case slug from ``item_id`` for ``year_extractor_<slug>`` /
    ``season_extractor_<slug>`` preview names.

    Strips an ``institution_id`` prefix when present (legacy items), strips a trailing
    ``_hook``, then keeps only ``[a-zA-Z0-9_]`` with underscores collapsed.
    """
    s = (item_id or "").strip()
    if institution_id:
        prefix = f"{institution_id.strip()}_"
        if s.startswith(prefix):
            s = s[len(prefix) :]
    if s.endswith("_hook"):
        s = s[:-5]
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s:
        s = "hook"
    if s[0].isdigit():
        s = f"_{s}"
    return s


def _year_season_function_old_names(functions: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Match :func:`~edvise.genai.mapping.identity_agent.term_normalization.term_order.load_term_extractors_from_hook_spec` naming rules."""
    names: list[str] = []
    for fn in functions:
        if not isinstance(fn, dict):
            continue
        n = fn.get("name")
        if isinstance(n, str) and n:
            names.append(n)
    year_like = [n for n in names if "year" in n.lower()]
    season_like = [n for n in names if "season" in n.lower()]
    year_names = [n for n in year_like if n not in season_like]
    season_names = [n for n in season_like if n not in year_like]
    if len(year_names) != 1 or len(season_names) != 1:
        return None
    return year_names[0], season_names[0]


def _rewrite_def_line(draft: str, old_name: str, new_name: str) -> str:
    pattern = re.compile(
        rf"^(\s*def\s+){re.escape(old_name)}(\s*\()",
        flags=re.MULTILINE,
    )
    out, n = pattern.subn(lambda m: m.group(1) + new_name + m.group(2), draft, count=1)
    if n == 0:
        return draft
    return out


def apply_term_hook_preview_names_from_item_id(
    hook_spec: dict[str, Any],
    item_id: str,
    *,
    institution_id: str,
) -> dict[str, Any]:
    """
    Return a deep copy of ``hook_spec`` with ``year_extractor_<slug>`` / ``season_extractor_<slug>``
    names (and matching ``draft`` / ``signature`` text) so preview JSON matches unique module symbols.

    No-op if ``functions`` does not contain exactly one year-like and one season-like name.
    """
    out = copy.deepcopy(hook_spec)
    functions = out.get("functions")
    if not isinstance(functions, list):
        return out
    pair = _year_season_function_old_names(functions)
    if pair is None:
        return out
    old_year, old_season = pair
    slug = hook_slug_from_item_id(item_id, institution_id=institution_id)
    new_year = f"year_extractor_{slug}"
    new_season = f"season_extractor_{slug}"
    if old_year == new_year and old_season == new_season:
        return out

    for fn in functions:
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        draft = fn.get("draft")
        sig = fn.get("signature")
        if name == old_year:
            fn["name"] = new_year
            if isinstance(draft, str):
                fn["draft"] = _rewrite_def_line(draft, old_year, new_year)
            if isinstance(sig, str):
                fn["signature"] = sig.replace(old_year, new_year, 1)
        elif name == old_season:
            fn["name"] = new_season
            if isinstance(draft, str):
                fn["draft"] = _rewrite_def_line(draft, old_season, new_season)
            if isinstance(sig, str):
                fn["signature"] = sig.replace(old_season, new_season, 1)
    return out


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
    hitl_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> None:
    """
    Write a JSON artifact listing generated hook specs (``item_id`` + ``hook_spec``) for HITL review.

    ``domain`` is ``identity_grain`` or ``identity_term`` (informative for reviewers / UI).

    When ``hitl_path`` and ``config_path`` are both set (resolver-shaped identity JSON at
    ``config_path``), each spec row also gets ``review_context``: table, HITL question/context,
    reviewer note, hook group metadata, resolver ``target``, and the same ``config_snippet``
    (``grain_contract`` / ``term_config`` slice) passed to hook-generation LLM calls.
    """
    path = Path(output_path)
    items_by_id: dict[str, HITLItem] | None = None
    identity_config: dict[str, Any] | None = None
    if hitl_path is not None or config_path is not None:
        if hitl_path is None or config_path is None:
            raise ValueError(
                "hitl_path and config_path must both be set or both omitted for hook preview enrichment"
            )
        # Local import keeps package load order simple (resolver ↔ hook_generation).
        from edvise.genai.mapping.identity_agent.hitl.hook_generation.prompt import (
            extract_config_snippet_for_hook_item,
        )
        from edvise.genai.mapping.identity_agent.hitl.resolver import get_hook_items

        identity_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        items_by_id = {it.item_id: it for it in get_hook_items(hitl_path)}

    spec_rows: list[dict[str, Any]] = []
    for item_id, spec in specs:
        spec_dump = spec.model_dump(mode="json")
        if domain == "identity_term":
            spec_dump = apply_term_hook_preview_names_from_item_id(
                spec_dump,
                item_id,
                institution_id=institution_id,
            )
        row: dict[str, Any] = {
            "item_id": item_id,
            "hook_spec": spec_dump,
        }
        if items_by_id is not None and identity_config is not None:
            hit = items_by_id.get(item_id)
            if hit is not None:
                snippet = extract_config_snippet_for_hook_item(identity_config, hit)
                row["review_context"] = {
                    "table": hit.table,
                    "hitl_question": hit.hitl_question,
                    "hitl_context": _hitl_context_json(hit.hitl_context),
                    "reviewer_note": hit.reviewer_note,
                    "hook_group_id": hit.hook_group_id,
                    "hook_group_tables": hit.hook_group_tables,
                    "target": hit.target.model_dump(mode="json"),
                    "config_snippet": snippet,
                }
        spec_rows.append(row)

    payload: dict[str, Any] = {
        "institution_id": institution_id,
        "domain": domain,
        "specs": spec_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "apply_term_hook_preview_names_from_item_id",
    "assemble_hook_spec_drafts_as_module_text",
    "hook_slug_from_item_id",
    "write_identity_hook_preview_json",
]
