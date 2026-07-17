"""Detect term configs that require a ``year_semantics`` HITL review."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from edvise.genai.mapping.identity_agent.hitl.schemas import HITLItem
    from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
        InstitutionTermContract,
    )

_CALENDAR_MONTH_RAWS = {f"{m:02d}" for m in range(1, 13)} | {
    str(m) for m in range(1, 13)
}
_SPELLED_SEASON_RAWS = {"fall", "spring", "summer", "winter"}
_YEAR_SEMANTICS_VALUES = frozenset({"calendar_literal", "academic_year_prefix"})


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


def _item_covers_table(item: HITLItem, table: str) -> bool:
    if item.table == table:
        return True
    hgt = item.hook_group_tables
    return bool(hgt and table in hgt)


def _resolution_year_semantics(resolution: object | None) -> str | None:
    if resolution is None:
        return None
    if isinstance(resolution, dict):
        value = resolution.get("year_semantics")
    else:
        value = getattr(resolution, "year_semantics", None)
    if value in _YEAR_SEMANTICS_VALUES:
        return str(value)
    return None


def item_offers_year_semantics_choice(item: HITLItem) -> bool:
    """
    True when the item offers both ``calendar_literal`` and ``academic_year_prefix``.

    The LLM must emit a separate terminal HITL item for year meaning; season-map /
    hook-confirmation items do not satisfy this even if they mention calendar year
    in free-text context.
    """
    offered: set[str] = set()
    for opt in item.options:
        ys = _resolution_year_semantics(opt.resolution)
        if ys is not None:
            offered.add(ys)
    return offered >= _YEAR_SEMANTICS_VALUES


def collect_term_year_semantics_hitl_coverage_errors(
    inst: InstitutionTermContract,
    items: list[HITLItem],
) -> list[str]:
    """
    Return errors for datasets that need ``year_semantics`` review but lack HITL coverage.

    A covering item must target the dataset (``table`` or ``hook_group_tables``) and offer
    both ``calendar_literal`` and ``academic_year_prefix`` resolution options.
    """
    from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain

    term_items = [it for it in items if it.domain == HITLDomain.IDENTITY_TERM]
    errors: list[str] = []
    for table, contract in sorted(inst.datasets.items()):
        cfg = contract.term_config
        if cfg is None:
            continue
        cfg_dict = cfg.model_dump(mode="json")
        if not term_config_needs_year_semantics_review(cfg_dict):
            continue
        covered = any(
            _item_covers_table(it, table) and item_offers_year_semantics_choice(it)
            for it in term_items
        )
        if covered:
            continue
        errors.append(
            f"dataset {table!r}: term_config uses a coded year prefix (YYYY+suffix, "
            "YYYY-NN / YYYYPP period codes, or split year + period-code columns) but "
            "hitl_items has no covering reentry='terminal' year_semantics item. Emit a "
            "separate HITLItem for this table (or its hook_group_tables) whose options "
            "set year_semantics to calendar_literal and academic_year_prefix — "
            "independent of season_map / hook confirmation."
        )
    return errors


def assert_term_year_semantics_hitl_coverage(
    inst: InstitutionTermContract,
    items: list[HITLItem],
) -> None:
    """Raise ``ValueError`` when coded-year configs lack a year_semantics HITL item."""
    errors = collect_term_year_semantics_hitl_coverage_errors(inst, items)
    if errors:
        raise ValueError("Term batch JSON: " + " ".join(errors))
