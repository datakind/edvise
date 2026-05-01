"""Tests for SMA Pass 2 TERMINAL option scratch-manifest validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.option_validation import (
    build_scratch_manifest_for_terminal_option,
    collect_pass2_terminal_option_validation_failures,
    raise_if_pass2_terminal_options_invalid,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas import (
    SMAFailureMode,
    SMAHITLItem,
    SMAHITLOption,
    SMAReentryDepth,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    ColumnAlias,
    FieldMappingManifest,
    FieldMappingRecord,
    JoinConfig,
    RowSelectionConfig,
    RowSelectionStrategy,
)
from edvise.genai.mapping.shared.schema_contract import parse_enriched_schema_contract_for_sma


def _cd(name: str) -> dict:
    return {
        "original_name": name,
        "normalized_name": name,
        "null_count": 0,
        "null_percentage": 0.0,
        "unique_count": 1,
        "sample_values": [],
    }


def _minimal_student_term_contract():
    """student + term tables with Lee-like column names."""
    return parse_enriched_schema_contract_for_sma(
        {
            "school_id": "x",
            "school_name": "X",
            "datasets": {
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
                            _cd("starting_cohort_term"),
                            _cd("active_degree"),
                        ],
                    },
                },
                "term": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": [],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/t.csv",
                        "num_rows": 1,
                        "num_columns": 3,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("term"),
                            _cd("declared_degree"),
                        ],
                    },
                },
            },
        }
    )


def _minimal_cohort_manifest() -> FieldMappingManifest:
    return FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[
            FieldMappingRecord(
                target_field="learner_id",
                source_column="learner_id",
                source_table="student",
                join=None,
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=1.0,
                rationale="",
            ),
            FieldMappingRecord(
                target_field="intended_program_type",
                source_column="active_degree",
                source_table="student",
                join=None,
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=0.7,
                rationale="",
            ),
        ],
        column_aliases=[],
    )


def test_build_scratch_manifest_swaps_mapping_and_merges_alias():
    refined = _minimal_cohort_manifest()
    opt_fm = FieldMappingRecord(
        target_field="intended_program_type",
        source_column="declared_degree",
        source_table="term",
        join=JoinConfig(
            base_table="student",
            lookup_table="term",
            join_keys=["learner_id", "term"],
        ),
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.where_not_null,
            condition_col="declared_degree",
        ),
        confidence=0.85,
        rationale="",
    )
    alias = ColumnAlias(
        table="student",
        source_column="starting_cohort_term",
        canonical_column="term",
        rationale="bridge",
    )
    opt = SMAHITLOption(
        option_id="opt_a",
        label="A",
        description="D",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=opt_fm,
        column_alias=alias,
    )
    scratch = build_scratch_manifest_for_terminal_option(
        refined, "intended_program_type", opt
    )
    row = next(m for m in scratch.mappings if m.target_field == "intended_program_type")
    assert row.source_column == "declared_degree"
    assert scratch.column_aliases == [alias]


def test_bad_join_keys_terminal_option_fails_deterministic_validation():
    """Lee-style join_keys use physical student key name; lookup table has ``term`` not ``starting_cohort_term``."""
    contract = _minimal_student_term_contract()
    refined = _minimal_cohort_manifest()
    bad_fm = FieldMappingRecord(
        target_field="intended_program_type",
        source_column="declared_degree",
        source_table="term",
        join=JoinConfig(
            base_table="student",
            lookup_table="term",
            join_keys=["learner_id", "starting_cohort_term"],
        ),
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.where_not_null,
            condition_col="declared_degree",
        ),
        confidence=0.85,
        rationale="",
    )
    alias = ColumnAlias(
        table="student",
        source_column="starting_cohort_term",
        canonical_column="term",
        rationale="bridge",
    )
    bad_opt = SMAHITLOption(
        option_id="bad",
        label="Bad",
        description="Bad",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=bad_fm,
        column_alias=alias,
    )
    direct = SMAHITLOption(
        option_id="direct_edit",
        label="Edit",
        description="E",
        reentry=SMAReentryDepth.DIRECT_EDIT,
        field_mapping=None,
        column_alias=None,
    )
    item = SMAHITLItem(
        item_id="x_cohort_intended_program_type_low_confidence",
        institution_id="x",
        entity_type="cohort",
        target_field="intended_program_type",
        failure_mode=SMAFailureMode.LOW_CONFIDENCE,
        hitl_question="q",
        hitl_context=None,
        current_field_mapping=refined.mappings[1],
        validation_errors=[],
        options=[bad_opt, direct],
    )
    failures = collect_pass2_terminal_option_validation_failures(
        refined, [item], contract
    )
    assert len(failures) == 1
    _item_id, opt_id, errs = failures[0]
    assert opt_id == "bad"
    assert any(e.error_code.value == "JOIN_KEY_NOT_IN_LOOKUP_TABLE" for e in errs)


def test_raise_if_pass2_terminal_options_invalid_raises():
    contract = _minimal_student_term_contract()
    refined = _minimal_cohort_manifest()
    bad_fm = FieldMappingRecord(
        target_field="intended_program_type",
        source_column="declared_degree",
        source_table="term",
        join=JoinConfig(
            base_table="student",
            lookup_table="term",
            join_keys=["learner_id", "starting_cohort_term"],
        ),
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.where_not_null,
            condition_col="declared_degree",
        ),
        confidence=0.85,
        rationale="",
    )
    bad_opt = SMAHITLOption(
        option_id="bad",
        label="Bad",
        description="Bad",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=bad_fm,
        column_alias=None,
    )
    direct = SMAHITLOption(
        option_id="direct_edit",
        label="Edit",
        description="E",
        reentry=SMAReentryDepth.DIRECT_EDIT,
        field_mapping=None,
        column_alias=None,
    )
    item = SMAHITLItem(
        item_id="x_cohort_intended_program_type_low_confidence",
        institution_id="x",
        entity_type="cohort",
        target_field="intended_program_type",
        failure_mode=SMAFailureMode.LOW_CONFIDENCE,
        hitl_question="q",
        hitl_context=None,
        current_field_mapping=refined.mappings[1],
        validation_errors=[],
        options=[bad_opt, direct],
    )
    with pytest.raises(ValidationError, match="item_id="):
        raise_if_pass2_terminal_options_invalid(refined, [item], contract)
