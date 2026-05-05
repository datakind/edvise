"""Tests for cross-table merge join-key dtype alignment."""

import pandas as pd

from edvise.genai.mapping.schema_mapping_agent.execution.field_executor import (
    _coerce_join_frames_for_merge,
)


def test_coerce_object_string_vs_int64_merge_matches_values() -> None:
    left = pd.DataFrame({"term": ["F24", "F24"], "course_number": ["101", "102"]})
    right = pd.DataFrame(
        {
            "term": ["F24", "F24"],
            "course_number": pd.array([101, 102], dtype="Int64"),
            "instructional_mode": ["Lecture", "Lab"],
        }
    )
    lm, rm = _coerce_join_frames_for_merge(
        left,
        right,
        ["term", "course_number"],
        ["term", "course_number"],
        log_context="instructional_modality",
    )
    merged = lm.merge(
        rm,
        left_on=["term", "course_number"],
        right_on=["term", "course_number"],
        how="left",
    )
    assert len(merged) == 2
    assert merged["instructional_mode"].tolist() == ["Lecture", "Lab"]


def test_no_coercion_when_key_dtypes_already_align() -> None:
    left = pd.DataFrame({"k": [1, 2]})
    right = pd.DataFrame({"k": [1, 2], "v": ["a", "b"]})
    lm, rm = _coerce_join_frames_for_merge(
        left, right, ["k"], ["k"], log_context="noop_test"
    )
    assert lm is left and rm is right
    assert lm["k"].tolist() == [1, 2]
    assert rm["k"].tolist() == [1, 2]
    out = lm.merge(rm, left_on=["k"], right_on=["k"], how="left")
    assert out["v"].tolist() == ["a", "b"]
