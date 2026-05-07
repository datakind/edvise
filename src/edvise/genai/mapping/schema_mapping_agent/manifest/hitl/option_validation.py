"""
Deterministic validation for SMA Pass 2 HITL TERMINAL options.

Builds a scratch :class:`FieldMappingManifest` as if ``resolve_sma_items`` had
applied one option (mapping swap + optional ``column_alias``), then runs the same
:class:`~edvise.genai.mapping.schema_mapping_agent.manifest.validation.validate_manifest`
pass as post–Step 2a generate.

Used inside Pass 2 ``llm_complete_with_parse_retry`` so invalid options trigger a
retry with structured errors in the correction hint.
"""

from __future__ import annotations

import logging
import time

from pydantic import ValidationError

_LOG = logging.getLogger(__name__)

from edvise.genai.mapping.shared.schema_contract.schemas import (
    EnrichedSchemaContractForSMA,
)

from ..schemas import FieldMappingManifest
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
    """Run ``validate_manifest`` on the scratch manifest for one TERMINAL option."""
    if option.reentry != SMAReentryDepth.TERMINAL:
        return []
    scratch = build_scratch_manifest_for_terminal_option(
        refined_manifest, target_field, option
    )
    return validate_manifest(scratch, schema_contract)


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
                "msg": msg,
                "input": None,
                "ctx": {"error": ValueError(msg)},
            }
        ],
    )


__all__ = [
    "build_scratch_manifest_for_terminal_option",
    "collect_pass2_terminal_option_validation_failures",
    "raise_if_pass2_terminal_options_invalid",
    "validate_terminal_hitl_option",
]
