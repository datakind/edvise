"""Detect term configs that require a ``year_semantics`` HITL review."""

from __future__ import annotations

from typing import Any

_CALENDAR_MONTH_RAWS = {f"{m:02d}" for m in range(1, 13)} | {str(m) for m in range(1, 13)}
_SPELLED_SEASON_RAWS = {"fall", "spring", "summer", "winter"}


def _hook_drafts(term_cfg: dict[str, Any]) -> tuple[str, str]:
    hook_spec = term_cfg.get("hook_spec") or {}
    year_draft = ""
    season_draft = ""
    for fn in hook_spec.get("functions") or []:
        name = (fn.get("name") or "").lower()
        draft = (fn.get("draft") or "").lower()
        if "year" in name:
            year_draft = draft
        elif "season" in name:
            season_draft = draft
    return year_draft, season_draft


def _season_map_raws(term_cfg: dict[str, Any]) -> set[str]:
    return {
        str(entry["raw"])
        for entry in (term_cfg.get("season_map") or [])
        if entry.get("raw") is not None
    }


def term_config_needs_year_semantics_review(term_cfg: dict[str, Any] | None) -> bool:
    """
    Return True when the reviewer must confirm ``term_config.year_semantics``.

    Orthogonal to ``term_extraction == hook_required``: datetime hook columns return False;
    compact ``YYYYPP`` / ``YYYY-NN`` period codes and YYYY+suffix encodings return True.
    """
    if not term_cfg:
        return False

    if term_cfg.get("year_semantics") in ("calendar_literal", "academic_year_prefix"):
        return False

    if term_cfg.get("year_col") and term_cfg.get("season_col"):
        return term_cfg.get("term_extraction") == "standard"

    if not term_cfg.get("term_col"):
        return False

    extraction = term_cfg.get("term_extraction")
    raws = _season_map_raws(term_cfg)
    year_draft, season_draft = _hook_drafts(term_cfg)

    if extraction == "hook_required":
        if "to_datetime" in year_draft:
            return False
        if "strftime" in season_draft or "%b" in season_draft or "%B" in season_draft:
            return False
        if raws and raws <= _CALENDAR_MONTH_RAWS:
            return False
        if "[:4]" in year_draft and (
            (raws and not raws <= _CALENDAR_MONTH_RAWS) or "[4:6]" in season_draft
        ):
            return True
        return False

    if extraction == "standard" and raws:
        if {r.lower() for r in raws} <= _SPELLED_SEASON_RAWS:
            return False
        if not any(r.isdigit() for r in raws):
            return True

    return False
