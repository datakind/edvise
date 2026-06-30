"""
Cross-table checks for term-stage GENERATE_HOOK HITL items and ``hook_group_tables``.

Catches incompatible groupings at artifact emit time (before human review) so
``apply_hook_spec`` does not fail late on split ``year_col``/``season_col`` configs.

Also validates semantic mistakes such as ``hook_required`` on a season-only ``term_col``
when year context lives in a separate profile column (use ``year_col`` + ``season_col``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from edvise.genai.mapping.identity_agent.hitl.schemas import (
    HITLDomain,
    HITLItem,
    ReentryDepth,
)
from edvise.genai.mapping.identity_agent.profiling.schemas import (
    RawColumnProfile,
    RawTableProfile,
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    InstitutionTermContract,
    TermContract,
    TermOrderConfig,
)

_SEASON_WORDS = frozenset(
    {"fall", "spring", "summer", "winter", "fa", "sp", "su", "wi"}
)


def term_tables_for_hook_group(
    items: list[HITLItem],
    group_id: str,
) -> list[str]:
    """
    Dataset names that receive the same term hook fan-out as :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.apply_hook_spec`.
    """
    tables: set[str] = set()
    for item in items:
        if item.domain == HITLDomain.IDENTITY_TERM and item.hook_group_id == group_id:
            tables.add(item.table)
            if item.hook_group_tables:
                tables.update(item.hook_group_tables)
    return sorted(tables)


def item_has_generate_hook_path(item: HITLItem) -> bool:
    """True when any option on the item uses ``reentry='generate_hook'``."""
    return any(opt.reentry == ReentryDepth.GENERATE_HOOK for opt in item.options)


def _term_config_hook_target_label(cfg: TermOrderConfig | None) -> str:
    if cfg is None:
        return "term_config is null"
    if cfg.term_col is not None:
        return f"term_col={cfg.term_col!r}"
    if cfg.year_col is not None and cfg.season_col is not None:
        return f"year_col={cfg.year_col!r}, season_col={cfg.season_col!r}"
    return "no term column configured"


def _assert_table_hook_group_compatible(
    *,
    table: str,
    contract: TermContract,
    item_id: str,
) -> None:
    cfg = contract.term_config
    if cfg is None:
        raise ValueError(
            f"Term batch JSON: HITL item {item_id!r} targets dataset {table!r} for hook "
            "generation, but term_config is null."
        )
    has_split = cfg.year_col is not None and cfg.season_col is not None
    has_term_col = cfg.term_col is not None
    if has_split and not has_term_col:
        raise ValueError(
            f"Term batch JSON: dataset {table!r} is listed for GENERATE_HOOK item "
            f"{item_id!r} but uses split year_col/season_col without term_col "
            f"({_term_config_hook_target_label(cfg)}). "
            "Split-column tables cannot share a combined-column hook group — use "
            "term_extraction='standard' with season_map for that table and omit it from "
            "hook_group_tables, or set term_col on the term_config."
        )
    if cfg.term_extraction != "hook_required":
        raise ValueError(
            f"Term batch JSON: dataset {table!r} is listed for GENERATE_HOOK item "
            f"{item_id!r} but term_extraction={cfg.term_extraction!r} "
            f"({_term_config_hook_target_label(cfg)}). "
            "Every table in a hook group must use term_extraction='hook_required' with "
            "term_col set."
        )
    if not has_term_col:
        raise ValueError(
            f"Term batch JSON: dataset {table!r} is listed for GENERATE_HOOK item "
            f"{item_id!r} but has no term_col ({_term_config_hook_target_label(cfg)})."
        )


def assert_term_hook_groups_compatible(
    inst: InstitutionTermContract,
    items: list[HITLItem],
) -> None:
    """
    Validate GENERATE_HOOK term HITL items against per-dataset term_config shapes.

    Raises ``ValueError`` when a hook group (or single-table generate_hook item) would
    fail at :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.apply_hook_spec`.
    """
    seen_groups: set[str] = set()
    for item in items:
        if item.domain != HITLDomain.IDENTITY_TERM:
            continue
        if not item_has_generate_hook_path(item):
            continue

        if item.hook_group_id:
            gid = item.hook_group_id
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
            target_tables = term_tables_for_hook_group(items, gid)
            anchor_item_id = item.item_id
        else:
            target_tables = [item.table]

        for table in target_tables:
            contract = inst.datasets.get(table)
            if contract is None:
                raise ValueError(
                    f"Term batch JSON: HITL item {item.item_id!r} references dataset "
                    f"{table!r}, which is missing from datasets."
                )
            _assert_table_hook_group_compatible(
                table=table,
                contract=contract,
                item_id=anchor_item_id if item.hook_group_id else item.item_id,
            )


def _normalize_draft(draft: str | None) -> str:
    return (draft or "").replace(" ", "")


def _draft_uses_term_datetime(draft: str | None) -> bool:
    return "to_datetime(term)" in _normalize_draft(draft)


def _draft_treats_term_as_string_token(draft: str | None) -> bool:
    norm = _normalize_draft(draft)
    if "to_datetime(term)" in norm:
        return False
    return "term.strip()" in norm or "term).lower()" in norm or norm == "str(term)"


def _hook_extractor_drafts_mismatch_term_shape(cfg: TermOrderConfig) -> str | None:
    """
    Return an error when year_extractor parses ``term`` as a date but season_extractor
    treats ``term`` as a plain season string — impossible on one combined column unless
    ``term_col`` is a datetime (both hooks should then use ``to_datetime(term)``).
    """
    if cfg.term_extraction != "hook_required" or cfg.hook_spec is None:
        return None
    year_fn = next(
        (f for f in cfg.hook_spec.functions if "year" in f.name.lower()),
        None,
    )
    season_fn = next(
        (f for f in cfg.hook_spec.functions if "season" in f.name.lower()),
        None,
    )
    if year_fn is None or season_fn is None:
        return None
    if not _draft_uses_term_datetime(year_fn.draft):
        return None
    if not _draft_treats_term_as_string_token(season_fn.draft):
        return None
    return (
        f"term_config for term_col={cfg.term_col!r} drafts year_extractor with "
        "pd.to_datetime(term) but season_extractor treats term as a plain string "
        f"({season_fn.draft!r}). Hooks receive only term_col values — they cannot read "
        "year from a separate column. When season and year are in different columns, set "
        "term_col=null, year_col=<date-or-year column>, season_col=<season column>, "
        "term_extraction='standard', hook_spec=null."
    )


def _column_samples(col: RawColumnProfile) -> list[Any]:
    if col.unique_values is not None:
        return list(col.unique_values)
    return list(col.sample_values)


def _value_looks_like_season_token(value: object) -> bool:
    s = str(value).strip().lower()
    if not s:
        return False
    if re.search(r"\d{4}", s):
        return False
    if re.search(r"\d{1,2}/\d{1,2}", s):
        return False
    if s in _SEASON_WORDS:
        return True
    return bool(re.fullmatch(r"[a-z]{2,6}", s))


def _column_values_are_season_only(col: RawColumnProfile) -> bool:
    samples = _column_samples(col)
    if not samples:
        return False
    return all(_value_looks_like_season_token(v) for v in samples)


def _column_provides_year_context(col: RawColumnProfile) -> bool:
    dtype = col.dtype.lower()
    if "date" in dtype or "time" in dtype:
        return True
    for value in _column_samples(col):
        text = str(value).strip()
        if re.search(r"\d{4}", text):
            return True
        if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", text):
            return True
    return False


def _profile_columns_for_dataset(
    run_by_dataset: Mapping[str, Mapping[str, object]] | None,
    table: str,
) -> list[RawColumnProfile] | None:
    if run_by_dataset is None or table not in run_by_dataset:
        return None
    row = run_by_dataset[table]
    rtp = row.get("raw_table_profile")
    if isinstance(rtp, RawTableProfile):
        return rtp.columns
    return None


def _split_columns_required_from_profile(
    table: str,
    cfg: TermOrderConfig,
    columns: list[RawColumnProfile],
) -> str | None:
    if cfg.term_col is None or cfg.term_extraction != "hook_required":
        return None
    term_profile = next((c for c in columns if c.name == cfg.term_col), None)
    if term_profile is None or not _column_values_are_season_only(term_profile):
        return None
    year_candidates = [
        c.name
        for c in columns
        if c.name != cfg.term_col and _column_provides_year_context(c)
    ]
    if not year_candidates:
        return None
    year_col = year_candidates[0]
    return (
        f"dataset {table!r}: term_col={cfg.term_col!r} contains season-only tokens "
        f"while {year_col!r} provides calendar year/date context. Use split columns: "
        f"term_col=null, year_col={year_col!r}, season_col={cfg.term_col!r}, "
        "term_extraction='standard', hook_spec=null. Hooks cannot combine values from "
        "two columns — extractors only receive term_col as the term argument."
    )


def collect_term_semantic_validation_errors(
    inst: InstitutionTermContract,
    run_by_dataset: Mapping[str, Mapping[str, object]] | None = None,
) -> list[str]:
    """
    Detect term_config shapes that pass Pydantic but fail at runtime or need split columns.

    When ``run_by_dataset`` is provided (Pass 2 batch context), also checks profiled
    column samples for season-only ``term_col`` plus a separate year/date column.
    """
    errors: list[str] = []
    for table, contract in inst.datasets.items():
        cfg = contract.term_config
        if cfg is None:
            continue
        mismatch = _hook_extractor_drafts_mismatch_term_shape(cfg)
        if mismatch:
            errors.append(f"[{table}] {mismatch}")
        columns = _profile_columns_for_dataset(run_by_dataset, table)
        if columns is not None:
            split_err = _split_columns_required_from_profile(table, cfg, columns)
            if split_err:
                errors.append(split_err)
    return errors


def raise_term_semantic_validation_error_if_any(errors: list[str]) -> None:
    """Raise :class:`ValidationError` so ``llm_complete_with_parse_retry`` can retry."""
    if not errors:
        return
    raise ValidationError.from_exception_data(
        "TermConfigSemanticValidation",
        [
            {
                "type": "value_error",
                "loc": ("term_config", i),
                "input": None,
                "ctx": {"error": ValueError(msg)},
            }
            for i, msg in enumerate(errors)
        ],
    )


def build_parse_institution_term_contracts_with_semantic_checks(
    run_by_dataset: Mapping[str, Mapping[str, object]] | None = None,
) -> Callable[[str | bytes | dict], tuple[InstitutionTermContract, list[HITLItem]]]:
    """
    Parse fn for ``llm_complete_with_parse_retry`` that validates term_config semantics.

    ``run_by_dataset`` should be the same mapping passed to the batch term user payload
    (keys per dataset, values include ``raw_table_profile``).
    """
    from edvise.genai.mapping.identity_agent.term_normalization.prompt import (
        parse_institution_term_contracts_with_hitl,
    )

    def parse(
        raw: str | bytes | dict,
    ) -> tuple[InstitutionTermContract, list[HITLItem]]:
        inst, items = parse_institution_term_contracts_with_hitl(raw)
        raise_term_semantic_validation_error_if_any(
            collect_term_semantic_validation_errors(inst, run_by_dataset)
        )
        return inst, items

    return parse


__all__ = [
    "assert_term_hook_groups_compatible",
    "build_parse_institution_term_contracts_with_semantic_checks",
    "collect_term_semantic_validation_errors",
    "item_has_generate_hook_path",
    "raise_term_semantic_validation_error_if_any",
    "term_tables_for_hook_group",
]
