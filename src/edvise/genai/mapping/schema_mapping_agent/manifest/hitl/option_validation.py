"""
Deterministic validation for SMA Pass 2 HITL TERMINAL options.

Builds a scratch :class:`FieldMappingManifest` as if ``resolve_sma_items`` had
applied one option (mapping swap + optional ``column_alias``), then runs the same
:class:`~edvise.genai.mapping.schema_mapping_agent.manifest.validation.validate_manifest`
pass as post–Step 2a generate.

Only errors whose ``ManifestValidationError.target_field`` matches the HITL item's
``target_field`` (the row this option replaces) are returned. Other rows come from
Pass 1's ``refined_manifest`` unchanged; failing the option because of those would
block Pass 2 retries with no way to fix them via Pass 2 output (which only carries
``items``, not a revised manifest).

Used inside Pass 2 ``llm_complete_with_parse_retry`` so invalid options trigger a
retry with structured errors in the correction hint.

Also includes :func:`raise_if_pass2_terminal_options_not_distinct`, a separate
deterministic check that flags exact-duplicate TERMINAL options within the same
item (identical sourcing once provenance/confidence metadata is stripped) and
raises through the same retry path. Near-identical-but-not-exact options are
intentionally not flagged yet.
"""

from __future__ import annotations

import logging
import time

from pydantic import ValidationError

_LOG = logging.getLogger(__name__)

from edvise.genai.mapping.shared.schema_contract.schemas import (
    EnrichedSchemaContractForSMA,
)

from ..schemas import ColumnAlias, FieldMappingManifest, FieldMappingRecord
from ..validation import ManifestValidationError, validate_manifest
from .schemas import SMAHITLItem, SMAHITLOption, SMAReentryDepth, add_alias_if_missing


def build_scratch_manifest_for_terminal_option(
    refined_manifest: FieldMappingManifest,
    target_field: str,
    option: SMAHITLOption,
) -> FieldMappingManifest:
    """
    Manifest snapshot if this TERMINAL option were applied (mapping row + alias).

    Raises:
        ValueError: Not a TERMINAL option, missing ``field_mapping``, or
            ``target_field`` absent from ``refined_manifest.mappings``.
    """
    if option.reentry != SMAReentryDepth.TERMINAL or option.field_mapping is None:
        raise ValueError(
            "scratch manifest requires a TERMINAL option with field_mapping"
        )

    scratch = refined_manifest.model_copy(deep=True)
    idx: int | None = None
    for i, m in enumerate(scratch.mappings):
        if m.target_field == target_field:
            idx = i
            break
    if idx is None:
        raise ValueError(
            f"No mapping with target_field={target_field!r} in refined manifest"
        )

    new_mappings = list(scratch.mappings)
    new_mappings[idx] = option.field_mapping
    scratch = scratch.model_copy(update={"mappings": new_mappings})

    if option.column_alias is not None:
        add_alias_if_missing(scratch, option.column_alias)

    return scratch


def validate_terminal_hitl_option(
    refined_manifest: FieldMappingManifest,
    target_field: str,
    option: SMAHITLOption,
    schema_contract: EnrichedSchemaContractForSMA,
) -> list[ManifestValidationError]:
    """
    Run ``validate_manifest`` on the scratch manifest for one TERMINAL option.

    Returns only errors tied to ``target_field`` (the mapping row this option
    replaces). Pre-existing violations on other fields are omitted so Pass 2 can
    validate HITL options independently of Pass 1 gaps elsewhere.
    """
    if option.reentry != SMAReentryDepth.TERMINAL:
        return []
    scratch = build_scratch_manifest_for_terminal_option(
        refined_manifest, target_field, option
    )
    all_errs = validate_manifest(scratch, schema_contract)
    return [e for e in all_errs if e.target_field == target_field]


def collect_pass2_terminal_option_validation_failures(
    refined_manifest: FieldMappingManifest,
    items: list[SMAHITLItem],
    schema_contract: EnrichedSchemaContractForSMA,
) -> list[tuple[str, str, list[ManifestValidationError]]]:
    """
    Validate every TERMINAL option in Pass 2 ``items``.

    Returns:
        List of ``(item_id, option_id, errors)`` tuples for options with any errors.
    """
    failures: list[tuple[str, str, list[ManifestValidationError]]] = []
    n_terminal = 0
    t0 = time.perf_counter()
    for item in items:
        for opt in item.options:
            if opt.reentry != SMAReentryDepth.TERMINAL:
                continue
            n_terminal += 1
            errs = validate_terminal_hitl_option(
                refined_manifest, item.target_field, opt, schema_contract
            )
            if errs:
                failures.append((item.item_id, opt.option_id, errs))
    elapsed = time.perf_counter() - t0
    _LOG.info(
        "SMA Pass 2 TERMINAL option validation: hitl_items=%d terminal_options=%d "
        "failing_options=%d elapsed_s=%.4f",
        len(items),
        n_terminal,
        len(failures),
        elapsed,
    )
    return failures


_FIELD_MAPPING_METADATA_KEYS = frozenset(
    {
        "confidence",
        "rationale",
        "validation_notes",
        "review_status",
        "reviewer_notes",
        "corrected_source_column",
    }
)
"""Keys excluded when fingerprinting a FieldMappingRecord for distinctness.

These describe provenance/confidence, not sourcing behavior — two options that
differ only in these fields resolve to the same executed value.
"""


def _field_mapping_fingerprint(field_mapping: FieldMappingRecord) -> tuple:
    """Structural fingerprint of a FieldMappingRecord, ignoring metadata fields."""
    dumped = field_mapping.model_dump(mode="json")
    for key in _FIELD_MAPPING_METADATA_KEYS:
        dumped.pop(key, None)
    return tuple(sorted(dumped.items()))


def _column_alias_fingerprint(column_alias: ColumnAlias | None) -> tuple | None:
    if column_alias is None:
        return None
    return (
        column_alias.table,
        column_alias.source_column,
        column_alias.canonical_column,
    )


def find_duplicate_terminal_options(
    item: SMAHITLItem,
) -> list[tuple[str, str]]:
    """
    Pairs of TERMINAL ``option_id``\\ s in ``item`` with identical sourcing
    fingerprints (``field_mapping`` minus provenance/confidence metadata, plus
    ``column_alias``).

    Only exact duplicates are flagged — two options that differ in any
    sourcing-relevant field (``source_column``, ``source_table``, ``join``,
    ``row_selection``) are left alone, even if very similar.
    """
    terminal = [opt for opt in item.options if opt.reentry == SMAReentryDepth.TERMINAL]
    dupes: list[tuple[str, str]] = []
    for i in range(len(terminal)):
        for j in range(i + 1, len(terminal)):
            a, b = terminal[i], terminal[j]
            assert a.field_mapping is not None and b.field_mapping is not None
            if _field_mapping_fingerprint(
                a.field_mapping
            ) == _field_mapping_fingerprint(
                b.field_mapping
            ) and _column_alias_fingerprint(
                a.column_alias
            ) == _column_alias_fingerprint(b.column_alias):
                dupes.append((a.option_id, b.option_id))
    return dupes


def collect_pass2_duplicate_terminal_options(
    items: list[SMAHITLItem],
) -> list[tuple[str, str, str]]:
    """
    Find exact-duplicate TERMINAL options across all Pass 2 ``items``.

    Returns:
        List of ``(item_id, option_id_a, option_id_b)`` tuples, one per
        duplicate pair found within an item.
    """
    failures: list[tuple[str, str, str]] = []
    for item in items:
        for opt_a, opt_b in find_duplicate_terminal_options(item):
            failures.append((item.item_id, opt_a, opt_b))
    return failures


def raise_if_pass2_terminal_options_not_distinct(
    items: list[SMAHITLItem],
) -> None:
    """
    Raise :class:`pydantic.ValidationError` if any item has two TERMINAL
    options with identical sourcing (``field_mapping`` sans metadata, plus
    ``column_alias``) — the reviewer would be choosing between two options
    that resolve to the same value (for ``llm_complete_with_parse_retry``).

    Only exact duplicates are flagged; near-identical-but-distinct options
    are intentionally left alone for now.
    """
    failures = collect_pass2_duplicate_terminal_options(items)
    if not failures:
        return

    lines: list[str] = [
        "Two or more TERMINAL HITL options within the same item are exact "
        "duplicates (identical source_column, source_table, join, row_selection, "
        "and column_alias). Each TERMINAL option must represent a distinct "
        "resolution — remove or replace the duplicate so the reviewer isn't "
        "choosing between two options that resolve identically.",
        "",
    ]
    for item_id, opt_a, opt_b in failures:
        lines.append(
            f"- item_id={item_id} options {opt_a!r} and {opt_b!r} are duplicates"
        )

    msg = "\n".join(lines)
    raise ValidationError.from_exception_data(
        "SMAPass2TerminalOptionDistinctness",
        [
            {
                "type": "value_error",
                "loc": ("items",),
                "input": None,
                "ctx": {"error": ValueError(msg)},
            }
        ],
    )


def raise_if_pass2_terminal_options_invalid(
    refined_manifest: FieldMappingManifest,
    items: list[SMAHITLItem],
    schema_contract: EnrichedSchemaContractForSMA,
) -> None:
    """
    Raise :class:`pydantic.ValidationError` if any TERMINAL option fails
    ``validate_manifest`` (for ``llm_complete_with_parse_retry``).

    The error message lists ``item_id``, ``option_id``, and each
    :class:`ManifestValidationError.detail`.
    """
    failures = collect_pass2_terminal_option_validation_failures(
        refined_manifest, items, schema_contract
    )
    if not failures:
        return

    lines: list[str] = [
        "One or more TERMINAL HITL options failed deterministic manifest validation.",
        "Fix join_keys (use canonical names consistent with column_aliases), ",
        "columns, row_selection, and grain keys. Rules match post–Step 2a validate_manifest.",
        "",
    ]
    for item_id, opt_id, errs in failures:
        for e in errs:
            lines.append(
                f"- item_id={item_id} option_id={opt_id} "
                f"target_field={e.target_field} [{e.error_code}]: {e.detail}"
            )

    msg = "\n".join(lines)
    raise ValidationError.from_exception_data(
        "SMAPass2TerminalOptionValidation",
        [
            {
                "type": "value_error",
                "loc": ("items",),
                "input": None,
                "ctx": {"error": ValueError(msg)},
            }
        ],
    )


__all__ = [
    "build_scratch_manifest_for_terminal_option",
    "collect_pass2_duplicate_terminal_options",
    "collect_pass2_terminal_option_validation_failures",
    "find_duplicate_terminal_options",
    "raise_if_pass2_terminal_options_invalid",
    "raise_if_pass2_terminal_options_not_distinct",
    "validate_terminal_hitl_option",
]
