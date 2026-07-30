"""Tests for SMA Pass 2 TERMINAL option scratch-manifest validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.option_validation import (
    build_scratch_manifest_for_terminal_option,
    collect_pass2_duplicate_terminal_options,
    collect_pass2_terminal_option_validation_failures,
    find_duplicate_terminal_options,
    raise_if_pass2_terminal_options_invalid,
    raise_if_pass2_terminal_options_not_distinct,
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
    leave_unmapped = SMAHITLOption(
        option_id="leave_unmapped",
        label="Leave unmapped",
        description="No source",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=refined.mappings[1].model_copy(
            update={
                "source_column": None,
                "source_table": None,
                "join": None,
                "row_selection": None,
                "confidence": 1.0,
            }
        ),
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
        options=[bad_opt, leave_unmapped, direct],
    )
    failures = collect_pass2_terminal_option_validation_failures(
        refined, [item], contract
    )
    assert len(failures) == 1
    _item_id, opt_id, errs = failures[0]
    assert opt_id == "bad"
    assert any(e.error_code.value == "JOIN_KEY_NOT_IN_LOOKUP_TABLE" for e in errs)


def _student_contract_with_conferral_and_program_columns():
    """student grain + columns for entry-style and conferral-style mappings."""
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
                        "num_columns": 5,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("active_degree"),
                            _cd("degree_term"),
                            _cd("_edvise_term_academic_year"),
                            _cd("_edvise_term_season"),
                        ],
                    },
                },
            },
        }
    )


def test_terminal_option_ignores_validation_errors_on_other_target_fields():
    """
    Pass 1 can leave invalid conferral rows; validating a TERMINAL option for
    another field must not fail solely because full-manifest validate_manifest
    reports those unrelated errors.
    """
    contract = _student_contract_with_conferral_and_program_columns()
    refined = FieldMappingManifest(
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
            FieldMappingRecord(
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
            ),
        ],
        column_aliases=[],
    )
    good_opt = SMAHITLOption(
        option_id="keep_active_degree",
        label="Keep",
        description="D",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=FieldMappingRecord(
            target_field="intended_program_type",
            source_column="active_degree",
            source_table="student",
            join=None,
            row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
            confidence=0.75,
            rationale="",
        ),
        column_alias=None,
    )
    leave_unmapped = SMAHITLOption(
        option_id="leave_unmapped",
        label="Leave unmapped",
        description="No source",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=refined.mappings[1].model_copy(
            update={
                "source_column": None,
                "source_table": None,
                "join": None,
                "row_selection": None,
                "confidence": 1.0,
            }
        ),
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
        options=[good_opt, leave_unmapped, direct],
    )
    failures = collect_pass2_terminal_option_validation_failures(
        refined, [item], contract
    )
    assert failures == []


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
    leave_unmapped = SMAHITLOption(
        option_id="leave_unmapped",
        label="Leave unmapped",
        description="No source",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=refined.mappings[1].model_copy(
            update={
                "source_column": None,
                "source_table": None,
                "join": None,
                "row_selection": None,
                "confidence": 1.0,
            }
        ),
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
        options=[bad_opt, leave_unmapped, direct],
    )
    with pytest.raises(ValidationError, match="item_id="):
        raise_if_pass2_terminal_options_invalid(refined, [item], contract)


def _terminal_opt(
    option_id: str,
    field_mapping: FieldMappingRecord,
    column_alias: ColumnAlias | None = None,
) -> SMAHITLOption:
    return SMAHITLOption(
        option_id=option_id,
        label=option_id,
        description="d",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=field_mapping,
        column_alias=column_alias,
    )


def _direct_edit_opt() -> SMAHITLOption:
    return SMAHITLOption(
        option_id="direct_edit",
        label="Edit",
        description="E",
        reentry=SMAReentryDepth.DIRECT_EDIT,
        field_mapping=None,
        column_alias=None,
    )


def _item_with_options(options: list[SMAHITLOption]) -> SMAHITLItem:
    refined = _minimal_cohort_manifest()
    return SMAHITLItem(
        item_id="x_cohort_intended_program_type_low_confidence",
        institution_id="x",
        entity_type="cohort",
        target_field="intended_program_type",
        failure_mode=SMAFailureMode.LOW_CONFIDENCE,
        hitl_question="q",
        hitl_context=None,
        current_field_mapping=refined.mappings[1],
        validation_errors=[],
        options=[*options, _direct_edit_opt()],
    )


def test_find_duplicate_terminal_options_flags_exact_duplicate():
    """Two TERMINAL options differing only in confidence/rationale are still duplicates."""
    fm_a = FieldMappingRecord(
        target_field="intended_program_type",
        source_column="active_degree",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.8,
        rationale="model reasoning A",
    )
    fm_b = fm_a.model_copy(
        update={"confidence": 0.6, "rationale": "different reasoning text"}
    )
    item = _item_with_options(
        [_terminal_opt("opt_a", fm_a), _terminal_opt("opt_b", fm_b)]
    )
    dupes = find_duplicate_terminal_options(item)
    assert dupes == [("opt_a", "opt_b")]


def test_find_duplicate_terminal_options_ignores_distinct_sourcing():
    fm_a = FieldMappingRecord(
        target_field="intended_program_type",
        source_column="active_degree",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.8,
        rationale="",
    )
    fm_b = FieldMappingRecord(
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
        confidence=0.7,
        rationale="",
    )
    item = _item_with_options(
        [_terminal_opt("opt_a", fm_a), _terminal_opt("opt_b", fm_b)]
    )
    assert find_duplicate_terminal_options(item) == []


def test_find_duplicate_terminal_options_distinguishes_by_column_alias():
    """Identical field_mapping but different column_alias is not (yet) a duplicate."""
    fm = FieldMappingRecord(
        target_field="intended_program_type",
        source_column="declared_degree",
        source_table="term",
        join=JoinConfig(
            base_table="student",
            lookup_table="term",
            join_keys=["learner_id", "term"],
        ),
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.7,
        rationale="",
    )
    alias_a = ColumnAlias(
        table="student", source_column="cohort_term", canonical_column="term"
    )
    alias_b = ColumnAlias(
        table="student", source_column="starting_term", canonical_column="term"
    )
    item = _item_with_options(
        [
            _terminal_opt("opt_a", fm, alias_a),
            _terminal_opt("opt_b", fm.model_copy(), alias_b),
        ]
    )
    assert find_duplicate_terminal_options(item) == []


def test_collect_pass2_duplicate_terminal_options_across_items():
    fm = FieldMappingRecord(
        target_field="intended_program_type",
        source_column="active_degree",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.8,
        rationale="",
    )
    dup_item = _item_with_options(
        [
            _terminal_opt("opt_a", fm.model_copy()),
            _terminal_opt("opt_b", fm.model_copy(update={"confidence": 0.5})),
        ]
    )
    clean_item = _item_with_options([_terminal_opt("only_opt", fm.model_copy())])
    failures = collect_pass2_duplicate_terminal_options([dup_item, clean_item])
    assert failures == [(dup_item.item_id, "opt_a", "opt_b")]


def test_raise_if_pass2_terminal_options_not_distinct_raises():
    fm = FieldMappingRecord(
        target_field="intended_program_type",
        source_column="active_degree",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.8,
        rationale="",
    )
    item = _item_with_options(
        [
            _terminal_opt("opt_a", fm.model_copy()),
            _terminal_opt("opt_b", fm.model_copy(update={"confidence": 0.5})),
        ]
    )
    with pytest.raises(ValidationError, match="duplicates"):
        raise_if_pass2_terminal_options_not_distinct([item])


def test_raise_if_pass2_terminal_options_not_distinct_passes_when_distinct():
    fm_a = FieldMappingRecord(
        target_field="intended_program_type",
        source_column="active_degree",
        source_table="student",
        join=None,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        confidence=0.8,
        rationale="",
    )
    fm_b = fm_a.model_copy(update={"source_column": "other_degree_col"})
    item = _item_with_options(
        [_terminal_opt("opt_a", fm_a), _terminal_opt("opt_b", fm_b)]
    )
    raise_if_pass2_terminal_options_not_distinct([item])  # no raise
