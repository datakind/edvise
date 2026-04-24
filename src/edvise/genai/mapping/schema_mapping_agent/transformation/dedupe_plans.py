"""
Merge duplicate ``target_field`` entries in Step 2b ``plans`` (single LLM call).

If ``steps`` and ``output_dtype`` match across duplicates, merge notes; otherwise
raise so humans can inspect model confusion.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_NOTE_KEYS = ("reviewer_notes", "validation_notes")


def _canon_key(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _merge_distinct_texts(*parts: str | None) -> str | None:
    seen: list[str] = []
    for p in parts:
        if p is None or p == "":
            continue
        if p not in seen:
            seen.append(p)
    if not seen:
        return None
    if len(seen) == 1:
        return seen[0]
    return "; ".join(seen)


def _merge_plan_group(base: dict[str, Any], others: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(base)
    for key in _NOTE_KEYS:
        merged: str | None = out.get(key)  # type: ignore[assignment]
        for o in others:
            m = o.get(key)
            merged = _merge_distinct_texts(merged, m)
        out[key] = merged
    return out


def _plan_signature(p: dict[str, Any]) -> tuple[str, str]:
    return (
        _canon_key(p.get("output_dtype")),
        _canon_key(p.get("steps") if p.get("steps") is not None else []),
    )


def dedupe_plans_in_section(
    plans: list[dict[str, Any]] | None,
    *,
    entity: str,
    log: logging.Logger | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Deduplicate ``plans`` by ``target_field``.

    Returns the new list and the number of *removed* duplicate plan objects (0 if none).

    Raises
    ------
    ValueError
        If the same ``target_field`` appears with different ``steps`` or ``output_dtype``.
    """
    if not plans:
        return [], 0
    log = log or logger

    by_field: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for p in plans:
        tf = p.get("target_field")
        if not isinstance(tf, str) or not tf:
            raise ValueError(
                f"transformation_maps.{entity!r} plans entry missing string target_field"
            )
        if tf not in by_field:
            order.append(tf)
            by_field[tf] = []
        by_field[tf].append(p)

    removed = 0
    out: list[dict[str, Any]] = []
    for tf in order:
        group = by_field[tf]
        if len(group) == 1:
            out.append(group[0])
            continue
        first, *rest = group
        ref_sig = _plan_signature(first)
        for o in rest:
            if _plan_signature(o) != ref_sig:
                raise ValueError(
                    f"Duplicate target_field {tf!r} in transformation_maps.{entity!r} "
                    f"with conflicting steps or output_dtype (model must not emit "
                    f"inconsistent duplicates)."
                )
        removed += len(rest)
        merged = _merge_plan_group(first, rest)
        log.warning(
            "Merged %d duplicate plan(s) for %s target_field=%r (identical steps/output_dtype); "
            "concatenated distinct notes where needed",
            len(rest),
            entity,
            tf,
        )
        out.append(merged)
    return out, removed


def dedupe_transformation_plans_in_wrapper(
    m: dict[str, Any], *, log: logging.Logger | None = None
) -> int:
    """
    In-place: dedupe ``plans`` under ``transformation_maps.cohort`` and ``.course``.

    Returns the total number of duplicate plan *objects* removed.
    """
    tmaps = m.get("transformation_maps")
    if not isinstance(tmaps, dict):
        return 0
    total = 0
    log = log or logger
    for entity in ("cohort", "course"):
        sec = tmaps.get(entity)
        if not isinstance(sec, dict):
            continue
        pl = sec.get("plans")
        if not isinstance(pl, list):
            continue
        new_plans, n = dedupe_plans_in_section(pl, entity=entity, log=log)
        sec["plans"] = new_plans
        total += n
    return total
