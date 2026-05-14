"""Row-selection rules in validate_manifest.

Covers ROW_SELECTION_ORDER_BY_NOT_CHRONOLOGICAL — bans IdentityAgent term columns
that don't encode chronological order (e.g. `_term_grain`) as `row_selection.order_by`.
"""

from __future__ import annotations

import pytest

from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    RowSelectionConfig,
    RowSelectionStrategy,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.validation import (
    ManifestValidationErrorCode,
    validate_manifest,
)
from edvise.genai.mapping.shared.schema_contract import (
    parse_enriched_schema_contract_for_sma,
)


def _cd(name: str) -> dict:
    return {
        "original_name": name,
        "normalized_name": name,
        "null_count": 0,
        "null_percentage": 0.0,
        "unique_count": 1,
        "sample_values": [],
    }


def _student_contract_with_term_columns():
    """student table carrying both _term_order (valid) and the non-chronological IA term columns."""
    return parse_enriched_schema_contract_for_sma(
        {
            "school_id": "x",
            "school_name": "X",
            "datasets": {
                "student": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": ["learner_id", "term"],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/s.csv",
                        "num_rows": 1,
                        "num_columns": 6,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("term"),
                            _cd("declared_major"),
                            _cd("_term_order"),
                            _cd("_term_grain"),
                            _cd("_edvise_term_season"),
                            _cd("_edvise_term_academic_year"),
                        ],
                    },
                }
            },
        }
    )


def _grain_record() -> FieldMappingRecord:
    return FieldMappingRecord(
        target_field="learner_id",
        source_column="learner_id",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=1.0,
        rationale="",
    )


def _first_by_record(order_by: str) -> FieldMappingRecord:
    return FieldMappingRecord(
        target_field="declared_major_at_entry",
        source_column="declared_major",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.first_by, order_by=order_by
        ),
        confidence=0.8,
        rationale="",
    )


def test_first_by_term_order_is_valid():
    contract = _student_contract_with_term_columns()
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_grain_record(), _first_by_record(order_by="_term_order")],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    codes = {e.error_code for e in errors}
    assert (
        ManifestValidationErrorCode.ROW_SELECTION_ORDER_BY_NOT_CHRONOLOGICAL
        not in codes
    )


@pytest.mark.parametrize(
    "bad_order_by",
    ["_term_grain", "_edvise_term_season", "_edvise_term_academic_year"],
)
def test_first_by_non_chronological_term_column_is_flagged(bad_order_by: str):
    contract = _student_contract_with_term_columns()
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_grain_record(), _first_by_record(order_by=bad_order_by)],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    matching = [
        e
        for e in errors
        if e.error_code
        == ManifestValidationErrorCode.ROW_SELECTION_ORDER_BY_NOT_CHRONOLOGICAL
        and e.target_field == "declared_major_at_entry"
    ]
    assert matching, (
        f"expected ROW_SELECTION_ORDER_BY_NOT_CHRONOLOGICAL for order_by={bad_order_by!r}, "
        f"got codes={[e.error_code for e in errors]}"
    )
    assert matching[0].offending_value == bad_order_by
    assert "_term_order" in matching[0].detail


def test_nth_with_term_grain_order_by_is_flagged():
    contract = _student_contract_with_term_columns()
    nth_record = FieldMappingRecord(
        target_field="declared_major_at_entry",
        source_column="declared_major",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.nth, n=1, order_by="_term_grain"
        ),
        confidence=0.7,
        rationale="",
    )
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_grain_record(), nth_record],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    codes = {e.error_code for e in errors}
    assert ManifestValidationErrorCode.ROW_SELECTION_ORDER_BY_NOT_CHRONOLOGICAL in codes
