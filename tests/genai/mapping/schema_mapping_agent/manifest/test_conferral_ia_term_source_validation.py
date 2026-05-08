"""CONFERRAL_USES_IA_TERM_COLUMN — ban `_edvise_term_*` as a conferral source.

Conferral targets must not source any IdentityAgent ``_edvise_term_*`` column,
regardless of source table or whether a join is declared. The manifest only carries
a single ``source_column`` per record and the executor cannot co-resolve a paired
``_edvise_term_season`` from the same selected lookup row alongside
``_edvise_term_academic_year``, so any such mapping silently pairs the academic year
with the wrong season.
"""

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


def _student_with_edvise_term_columns():
    return parse_enriched_schema_contract_for_sma(
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
                        "num_columns": 4,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("degree_term"),
                            _cd("_edvise_term_academic_year"),
                            _cd("_edvise_term_season"),
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
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=1.0,
        rationale="",
    )


def test_conferral_from_edvise_term_on_student_base_is_rejected():
    contract = _student_with_edvise_term_columns()
    bad = FieldMappingRecord(
        target_field="bachelors_degree_conferral_date",
        source_column="_edvise_term_academic_year",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.where_not_null,
            condition_col="degree_term",
        ),
        confidence=0.5,
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
        if e.error_code == ManifestValidationErrorCode.CONFERRAL_USES_IA_TERM_COLUMN
    ]
    assert hit, f"expected CONFERRAL_USES_IA_TERM_COLUMN, got {[e.error_code for e in errors]}"
    assert hit[0].offending_value == "_edvise_term_academic_year"


def test_entry_year_from_edvise_term_on_student_still_allowed():
    contract = _student_with_edvise_term_columns()
    entry = FieldMappingRecord(
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
        mappings=[_grain(), entry],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    assert (
        ManifestValidationErrorCode.CONFERRAL_USES_IA_TERM_COLUMN
        not in {e.error_code for e in errors}
    )


def _two_table_contract():
    return parse_enriched_schema_contract_for_sma(
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
                        "num_columns": 1,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [_cd("learner_id")],
                    },
                },
                "degrees": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": [],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/d.csv",
                        "num_rows": 1,
                        "num_columns": 2,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("_edvise_term_academic_year"),
                            _cd("_edvise_term_season"),
                        ],
                    },
                },
            },
        }
    )


def test_conferral_from_edvise_term_on_lookup_via_join_is_also_rejected():
    """The lookup-row IA path is no longer a legal conferral source.

    Even with a declared join, the executor cannot co-resolve `_edvise_term_season`
    from the same selected lookup row alongside `_edvise_term_academic_year`.
    """
    contract = _two_table_contract()
    joined = FieldMappingRecord(
        target_field="bachelors_degree_conferral_date",
        source_column="_edvise_term_academic_year",
        source_table="degrees",
        join=JoinConfig(
            base_table="student",
            lookup_table="degrees",
            join_keys=["learner_id"],
        ),
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.9,
        rationale="",
    )
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[
            joined,
            _grain(),
        ],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    hit = [
        e
        for e in errors
        if e.error_code == ManifestValidationErrorCode.CONFERRAL_USES_IA_TERM_COLUMN
    ]
    assert hit, (
        f"expected CONFERRAL_USES_IA_TERM_COLUMN even on cross-table mapping, "
        f"got {[e.error_code for e in errors]}"
    )
    assert hit[0].offending_value == "_edvise_term_academic_year"


def test_conferral_from_edvise_term_season_on_lookup_via_join_is_also_rejected():
    """`_edvise_term_season` as the conferral source is also rejected."""
    contract = _two_table_contract()
    joined = FieldMappingRecord(
        target_field="associates_degree_conferral_date",
        source_column="_edvise_term_season",
        source_table="degrees",
        join=JoinConfig(
            base_table="student",
            lookup_table="degrees",
            join_keys=["learner_id"],
        ),
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.9,
        rationale="",
    )
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[joined, _grain()],
        column_aliases=[],
    )
    errors = validate_manifest(manifest, contract)
    hit = [
        e
        for e in errors
        if e.error_code == ManifestValidationErrorCode.CONFERRAL_USES_IA_TERM_COLUMN
    ]
    assert hit, f"expected CONFERRAL_USES_IA_TERM_COLUMN, got {[e.error_code for e in errors]}"
    assert hit[0].offending_value == "_edvise_term_season"
