"""infer_manifest_base_table and CROSS_TABLE_REQUIRES_JOIN manifest validation."""

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
    infer_manifest_base_table,
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


def _terms_student_contract():
    return parse_enriched_schema_contract_for_sma(
        {
            "school_id": "x",
            "school_name": "X",
            "datasets": {
                "terms": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": [],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/t.csv",
                        "num_rows": 1,
                        "num_columns": 2,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("active_degree"),
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
                        "num_columns": 2,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("declared_degree"),
                        ],
                    },
                },
            },
        }
    )


def _grain_record(**kwargs) -> FieldMappingRecord:
    base = dict(
        target_field="learner_id",
        source_column="learner_id",
        source_table="terms",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=1.0,
        rationale="",
    )
    base.update(kwargs)
    return FieldMappingRecord(**base)


def test_infer_manifest_base_table_prefers_join_base_table():
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[
            _grain_record(),
            FieldMappingRecord(
                target_field="intended_program_type",
                source_column="declared_degree",
                source_table="student",
                join=JoinConfig(
                    base_table="terms",
                    lookup_table="student",
                    join_keys=["learner_id"],
                ),
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=0.9,
                rationale="",
            ),
        ],
        column_aliases=[],
    )
    assert infer_manifest_base_table(manifest) == "terms"


def test_infer_manifest_base_table_mode_without_joins():
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[
            _grain_record(),
            FieldMappingRecord(
                target_field="intended_program_type",
                source_column="active_degree",
                source_table="terms",
                join=None,
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=0.9,
                rationale="",
            ),
        ],
        column_aliases=[],
    )
    assert infer_manifest_base_table(manifest) == "terms"


def test_validate_manifest_cross_table_without_join():
    contract = _terms_student_contract()
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[
            _grain_record(),
            FieldMappingRecord(
                target_field="intended_program_type",
                source_column="declared_degree",
                source_table="student",
                join=None,
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=0.7,
                rationale="",
            ),
        ],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    codes = {e.error_code for e in errors}
    assert ManifestValidationErrorCode.CROSS_TABLE_REQUIRES_JOIN in codes
