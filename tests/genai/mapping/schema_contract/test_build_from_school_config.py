"""Tests for schema contract primary-key alignment with normalized headers."""

from edvise.genai.mapping.shared.schema_contract.build_from_school_config import (
    _canonical_primary_keys_for_contract,
    _resolve_primary_keys_to_normalized,
)


def test_unique_keys_for_contract_follow_resolved_normalized_columns():
    """
    Grain / config may use a raw or abbreviated header while normalize_columns produces
    snake_case keys; contract unique_keys should match the latter (plus canonical learner).
    """
    column_mapping = {"term_descr": ["TERM_DESCR"], "learner_id": ["student_id_col"]}
    pk_for_resolution = ["learner_id", "TERM_DESCR"]
    normalized_uks = _resolve_primary_keys_to_normalized(
        column_mapping, pk_for_resolution, "logical_ds"
    )
    assert normalized_uks == ["learner_id", "term_descr"]

    unique_keys_for_contract = _canonical_primary_keys_for_contract(
        normalized_uks,
        None,
        canonical_learner_column="learner_id",
    )
    assert unique_keys_for_contract == ["learner_id", "term_descr"]


def test_resolve_primary_keys_learner_id_falls_back_to_student_id():
    """Grain may list learner_id while normalize_columns keys are student_id pre-clean."""
    column_mapping = {
        "student_id": ["SID"],
        "course_name": ["COURSE_NAME"],
    }
    normalized_uks = _resolve_primary_keys_to_normalized(
        column_mapping,
        ["learner_id", "course_name"],
        "course",
    )
    assert normalized_uks == ["student_id", "course_name"]
