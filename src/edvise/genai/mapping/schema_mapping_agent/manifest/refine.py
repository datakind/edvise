"""
Post-parse safety nets for SMA refinement LLM output.

Call :func:`apply_refinement_review_status_safety_net` (or
:func:`_enforce_review_status_contract`) immediately after parsing
``refined_manifest`` and ``hitl_items`` from the refinement LLM response,
before writing artifacts.
"""

from __future__ import annotations

from edvise.genai.mapping.schema_mapping_agent.hitl.schemas import (
    HITL_CONFIDENCE_THRESHOLD,
    SMAHITLItem,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    ReviewStatus,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.validation import (
    ManifestValidationError,
)


def _enforce_review_status_contract(
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
    hitl_items: list[SMAHITLItem],
) -> list[str]:
    """
    Post-parse safety net: enforce review_status contract the LLM may have violated.
    Returns list of warning strings for logging — does not raise.
    Mutations applied in place.
    """
    flagged_fields = {e.target_field for e in validation_errors} | {
        item.target_field for item in hitl_items
    }
    warnings: list[str] = []

    for record in manifest.mappings:
        is_low_confidence = record.confidence <= HITL_CONFIDENCE_THRESHOLD
        is_flagged = record.target_field in flagged_fields
        is_auto_approved = record.review_status == ReviewStatus.auto_approved
        is_refined = record.review_status == ReviewStatus.refined_by_llm

        # Case 1: auto_approved on flagged or low confidence field
        if is_auto_approved and (is_low_confidence or is_flagged):
            warnings.append(
                f"[review_status violation] '{record.target_field}' marked "
                f"auto_approved but confidence={record.confidence} "
                f"(threshold={HITL_CONFIDENCE_THRESHOLD}) or has validation "
                f"errors / hitl_items. Forcing to proposed_for_hitl."
            )
            record.review_status = ReviewStatus.proposed_for_hitl

        # Case 2: refined_by_llm on low confidence field — not allowed under Option A
        elif is_refined and is_low_confidence:
            warnings.append(
                f"[review_status violation] '{record.target_field}' marked "
                f"refined_by_llm but confidence={record.confidence} <= "
                f"threshold={HITL_CONFIDENCE_THRESHOLD}. Low confidence fields "
                f"must always be proposed_for_hitl. Forcing."
            )
            record.review_status = ReviewStatus.proposed_for_hitl

    return warnings


def log_refinement_contract_warnings_to_mlflow(warnings: list[str]) -> None:
    """Log refinement contract warnings to MLflow when an active run exists."""
    if not warnings:
        return
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.log_param(
            "sma_refinement_review_status_warning_count",
            len(warnings),
        )
        mlflow.log_text(
            "\n".join(warnings),
            "sma_refinement_review_status_warnings.txt",
        )
    except Exception:
        # No active run, or logging not configured — ignore.
        pass


def apply_refinement_review_status_safety_net(
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
    hitl_items: list[SMAHITLItem],
    *,
    print_warnings: bool = True,
    log_mlflow: bool = True,
) -> list[str]:
    """
    Run :func:`_enforce_review_status_contract` and optionally print / log warnings.
    Returns the same warning list as enforcement.
    """
    warnings = _enforce_review_status_contract(
        manifest, validation_errors, hitl_items
    )
    if print_warnings:
        for w in warnings:
            print(f"⚠  {w}")
    if log_mlflow:
        log_refinement_contract_warnings_to_mlflow(warnings)
    return warnings


__all__ = [
    "_enforce_review_status_contract",
    "apply_refinement_review_status_safety_net",
    "log_refinement_contract_warnings_to_mlflow",
]
