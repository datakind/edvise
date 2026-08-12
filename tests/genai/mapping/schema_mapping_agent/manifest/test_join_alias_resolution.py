"""Join-key alias resolution: canonical join_keys ↔ physical columns."""

from __future__ import annotations

from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    ColumnAlias,
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


def _course_student_term_contract():
    """course has term_descr; student has term_desc (real-world naming mismatch)."""
    return parse_enriched_schema_contract_for_sma(
        {
            "school_id": "x",
            "school_name": "X",
            "datasets": {
                "course": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": [],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/c.csv",
                        "num_rows": 1,
                        "num_columns": 3,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("term_descr"),
                            _cd("course_grade"),
                        ],
                    },
                },
                "student": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": [],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/s.csv",
                        "num_rows": 1,
                        "num_columns": 3,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("term_desc"),
                            _cd("ugrd_grad_flag"),
                        ],
                    },
                },
            },
        }
    )


def _grain() -> FieldMappingRecord:
    return FieldMappingRecord(
        target_field="learner_id",
        source_column="learner_id",
        source_table="course",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=1.0,
        rationale="",
    )


def _term_degree_join(*, join_keys: list[str]) -> FieldMappingRecord:
    return FieldMappingRecord(
        target_field="term_degree",
        source_column="ugrd_grad_flag",
        source_table="student",
        join=JoinConfig(
            base_table="course",
            lookup_table="student",
            join_keys=join_keys,
        ),
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.9,
        rationale="",
    )


def test_validate_manifest_accepts_canonical_join_key_with_lookup_alias():
    """
    join_keys use course's physical name (canonical); student alias bridges term_desc.
    """
    contract = _course_student_term_contract()
    manifest = FieldMappingManifest(
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        mappings=[
            _grain(),
            _term_degree_join(join_keys=["learner_id", "term_descr"]),
        ],
        column_aliases=[
            ColumnAlias(
                table="student",
                source_column="term_desc",
                canonical_column="term_descr",
                rationale="student term grain column name differs from course",
            )
        ],
    )
    errors = validate_manifest(manifest, contract)
    join_codes = {
        e.error_code
        for e in errors
        if e.target_field == "term_degree"
        and e.error_code
        in {
            ManifestValidationErrorCode.JOIN_KEY_NOT_IN_BASE_TABLE,
            ManifestValidationErrorCode.JOIN_KEY_NOT_IN_LOOKUP_TABLE,
            ManifestValidationErrorCode.MISSING_COLUMN_ALIAS,
        }
    }
    assert join_codes == set()


def test_validate_manifest_rejects_canonical_join_key_without_alias():
    """Incomplete refinement fix: rename join key only, no column_aliases."""
    contract = _course_student_term_contract()
    manifest = FieldMappingManifest(
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        mappings=[
            _grain(),
            _term_degree_join(join_keys=["learner_id", "term_descr"]),
        ],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    assert any(
        e.error_code == ManifestValidationErrorCode.JOIN_KEY_NOT_IN_LOOKUP_TABLE
        and e.target_field == "term_degree"
        and e.offending_value == "term_descr"
        for e in errors
    )
