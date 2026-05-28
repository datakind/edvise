"""TERM_TARGET_SHOULD_USE_IA_NORMALIZED_COLUMN — require IA term columns on base table."""

from __future__ import annotations

from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    JoinConfig,
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


def _student_contract(*, include_ia: bool = True, include_datetime: bool = True):
    cols = [_cd("learner_id")]
    if include_datetime:
        cols.append(_cd("first_enrollment_date"))
    if include_ia:
        cols.extend([_cd("_edvise_term_academic_year"), _cd("_edvise_term_season")])
    return parse_enriched_schema_contract_for_sma(
        {
            "school_id": "x",
            "school_name": "X",
            "datasets": {
                "student": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": ["learner_id", "program_at_graduation"],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/s.csv",
                        "num_rows": 1,
                        "num_columns": len(cols),
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": cols,
                    },
                },
            },
        }
    )


def _grain() -> FieldMappingRecord:
    return FieldMappingRecord(
        target_field="learner_id",
        source_column="learner_id",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=1.0,
        rationale="",
    )


def test_entry_term_from_datetime_rejected_when_ia_columns_on_base():
    """Indiana-like: first_enrollment_date must not replace _edvise_term_season."""
    contract = _student_contract()
    bad = FieldMappingRecord(
        target_field="entry_term",
        source_column="first_enrollment_date",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.first_by,
            order_by="first_enrollment_date",
        ),
        confidence=0.7,
        rationale="",
    )
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_grain(), bad],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    hit = [
        e
        for e in errors
        if e.error_code
        == ManifestValidationErrorCode.TERM_TARGET_SHOULD_USE_IA_NORMALIZED_COLUMN
    ]
    assert hit, f"expected TERM_TARGET_SHOULD_USE_IA_NORMALIZED_COLUMN, got {errors}"
    assert hit[0].target_field == "entry_term"
    assert hit[0].offending_value == "first_enrollment_date"


def test_entry_year_from_edvise_term_on_base_allowed():
    contract = _student_contract()
    good = FieldMappingRecord(
        target_field="entry_year",
        source_column="_edvise_term_academic_year",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=1.0,
        rationale="",
    )
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_grain(), good],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    assert ManifestValidationErrorCode.TERM_TARGET_SHOULD_USE_IA_NORMALIZED_COLUMN not in {
        e.error_code for e in errors
    }


def test_datetime_source_allowed_when_ia_columns_absent_on_base():
    contract = _student_contract(include_ia=False)
    mapping = FieldMappingRecord(
        target_field="entry_term",
        source_column="first_enrollment_date",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.7,
        rationale="",
    )
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_grain(), mapping],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    assert ManifestValidationErrorCode.TERM_TARGET_SHOULD_USE_IA_NORMALIZED_COLUMN not in {
        e.error_code for e in errors
    }


def test_join_lookup_ia_rejected_when_base_already_has_ia_columns():
    contract = parse_enriched_schema_contract_for_sma(
        {
            "school_id": "x",
            "school_name": "X",
            "datasets": {
                "student": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": ["learner_id"],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/s.csv",
                        "num_rows": 1,
                        "num_columns": 3,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("_edvise_term_academic_year"),
                            _cd("_edvise_term_season"),
                        ],
                    },
                },
                "semester": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": ["learner_id", "semester"],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/m.csv",
                        "num_rows": 1,
                        "num_columns": 3,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("_edvise_term_season"),
                            _cd("semester_start_date"),
                        ],
                    },
                },
            },
        }
    )
    joined = FieldMappingRecord(
        target_field="entry_term",
        source_column="_edvise_term_season",
        source_table="semester",
        join=JoinConfig(
            base_table="student",
            lookup_table="semester",
            join_keys=["learner_id"],
        ),
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.first_by,
            order_by="semester_start_date",
        ),
        confidence=0.85,
        rationale="",
    )
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_grain(), joined],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    hit = [
        e
        for e in errors
        if e.error_code
        == ManifestValidationErrorCode.TERM_TARGET_SHOULD_USE_IA_NORMALIZED_COLUMN
    ]
    assert hit, "cross-table IA lookup should fail when base table already has IA columns"


def test_course_academic_term_from_raw_rejected_when_ia_on_base():
    contract = parse_enriched_schema_contract_for_sma(
        {
            "school_id": "x",
            "school_name": "X",
            "datasets": {
                "course": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": ["learner_id", "course_name", "semester"],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/c.csv",
                        "num_rows": 1,
                        "num_columns": 4,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("semester"),
                            _cd("_edvise_term_academic_year"),
                            _cd("_edvise_term_season"),
                        ],
                    },
                },
            },
        }
    )
    grain = FieldMappingRecord(
        target_field="learner_id",
        source_column="learner_id",
        source_table="course",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=1.0,
        rationale="",
    )
    bad = FieldMappingRecord(
        target_field="academic_term",
        source_column="semester",
        source_table="course",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.7,
        rationale="",
    )
    manifest = FieldMappingManifest(
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        mappings=[grain, bad],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    hit = [
        e
        for e in errors
        if e.error_code
        == ManifestValidationErrorCode.TERM_TARGET_SHOULD_USE_IA_NORMALIZED_COLUMN
    ]
    assert hit
    assert hit[0].target_field == "academic_term"
