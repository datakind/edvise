"""Every manifest validation code maps to a structural HITL failure mode."""

from __future__ import annotations

from edvise.genai.mapping.schema_mapping_agent.manifest.prompts.refine import (
    _failure_mode_for_validation_errors,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.validation import (
    ManifestValidationError,
    ManifestValidationErrorCode,
)

_EXPECTED_FAILURE_MODE = {
    ManifestValidationErrorCode.SOURCE_TABLE_NOT_FOUND: "column_not_found",
    ManifestValidationErrorCode.SOURCE_COLUMN_NOT_FOUND: "column_not_found",
    ManifestValidationErrorCode.JOIN_BASE_TABLE_NOT_FOUND: "join_structure",
    ManifestValidationErrorCode.JOIN_LOOKUP_TABLE_NOT_FOUND: "join_structure",
    ManifestValidationErrorCode.JOIN_KEY_NOT_IN_BASE_TABLE: "join_structure",
    ManifestValidationErrorCode.JOIN_KEY_NOT_IN_LOOKUP_TABLE: "join_structure",
    ManifestValidationErrorCode.MISSING_COLUMN_ALIAS: "join_structure",
    ManifestValidationErrorCode.JOIN_DECLARED_ON_SAME_TABLE: "join_structure",
    ManifestValidationErrorCode.JOIN_KEY_PARTIAL_IA_TERM_LABEL: "join_structure",
    ManifestValidationErrorCode.CROSS_TABLE_REQUIRES_JOIN: "join_structure",
    ManifestValidationErrorCode.ROW_SELECTION_ORDER_BY_NOT_FOUND: "row_selection",
    ManifestValidationErrorCode.ROW_SELECTION_ORDER_BY_NOT_CHRONOLOGICAL: "row_selection",
    ManifestValidationErrorCode.ROW_SELECTION_CONDITION_COL_NOT_FOUND: "row_selection",
    ManifestValidationErrorCode.ROW_SELECTION_FILTER_COL_NOT_FOUND: "row_selection",
    ManifestValidationErrorCode.MAPPED_FIELD_MISSING_SOURCE: "map_unmap",
    ManifestValidationErrorCode.UNMAPPED_FIELD_HAS_SOURCE: "map_unmap",
    ManifestValidationErrorCode.GRAIN_KEY_MISSING_SOURCE: "map_unmap",
    ManifestValidationErrorCode.CONFERRAL_USES_IA_TERM_COLUMN: "illegal_column",
    ManifestValidationErrorCode.TERM_TARGET_SHOULD_USE_IA_NORMALIZED_COLUMN: (
        "illegal_column"
    ),
}


def test_every_validation_error_code_maps_to_a_structural_mode():
    assert set(_EXPECTED_FAILURE_MODE) == set(ManifestValidationErrorCode)
    for code, mode in _EXPECTED_FAILURE_MODE.items():
        error = ManifestValidationError(
            target_field="learner_id",
            error_code=code,
            detail=code.value,
        )
        assert _failure_mode_for_validation_errors([error]) == mode


def test_no_validation_errors_are_low_confidence():
    assert _failure_mode_for_validation_errors([]) == "low_confidence"


def test_missing_column_outranks_illegal_column():
    errors = [
        _error(ManifestValidationErrorCode.CONFERRAL_USES_IA_TERM_COLUMN),
        _error(ManifestValidationErrorCode.SOURCE_COLUMN_NOT_FOUND),
    ]
    assert _failure_mode_for_validation_errors(errors) == "column_not_found"


def _error(code: ManifestValidationErrorCode) -> ManifestValidationError:
    return ManifestValidationError(
        target_field="learner_id",
        error_code=code,
        detail=code.value,
    )
