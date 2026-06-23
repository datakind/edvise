"""
Cross-table checks for term-stage GENERATE_HOOK HITL items and ``hook_group_tables``.

Catches incompatible groupings at artifact emit time (before human review) so
``apply_hook_spec`` does not fail late on split ``year_col``/``season_col`` configs.
"""

from __future__ import annotations

from edvise.genai.mapping.identity_agent.hitl.schemas import (
    HITLDomain,
    HITLItem,
    ReentryDepth,
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    InstitutionTermContract,
    TermContract,
    TermOrderConfig,
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


__all__ = [
    "assert_term_hook_groups_compatible",
    "item_has_generate_hook_path",
    "term_tables_for_hook_group",
]
