"""Tests for ES grade_map defaults, merge, and unmapped-GPA warnings."""

import logging

import pandas as pd

from edvise.data_audit.default_grade_map import DEFAULT_ES_GRADE_MAP
from edvise.data_audit.raw_course_grade_map import (
    apply_raw_course_grade_map,
    log_unmapped_gpa_grades,
    merge_grade_maps,
    resolve_es_grade_map,
    unmapped_gpa_grade_counts,
)


def test_resolve_es_grade_map_includes_status_defaults() -> None:
    resolved = resolve_es_grade_map(None)
    assert resolved["W1"] == "W"
    assert resolved["OTHER"] == "O"


def test_resolve_es_grade_map_includes_letter_gpa_defaults() -> None:
    resolved = resolve_es_grade_map(None)
    assert resolved["A"] == "4"
    assert resolved["A-"] == "3.7"
    assert resolved["B+"] == "3.3"
    assert resolved["F"] == "0"


def test_institution_grade_map_overrides_default() -> None:
    resolved = resolve_es_grade_map({"NC": "NG", "A": "4", "A-": "3.5"})
    assert resolved["NC"] == "NG"
    assert resolved["A"] == "4"
    assert resolved["A-"] == "3.5"
    assert resolved["W1"] == "W"
    assert resolved["B"] == "3"


def test_merge_grade_maps_empty_override() -> None:
    assert merge_grade_maps(DEFAULT_ES_GRADE_MAP, None) == resolve_es_grade_map(None)


def test_apply_raw_course_grade_map_maps_plus_minus() -> None:
    df = pd.DataFrame({"grade": ["A-", "B+", "OTH"]})
    mapped = apply_raw_course_grade_map(df, resolve_es_grade_map(None))
    assert mapped["grade"].tolist() == ["3.7", "3.3", "O"]


def test_unmapped_gpa_grade_counts_empty_when_fully_mapped() -> None:
    df = pd.DataFrame({"grade": ["4", "3.7", "W", "P"]})
    assert unmapped_gpa_grade_counts(df).empty


def test_unmapped_gpa_grade_counts_letters_after_partial_map() -> None:
    df = pd.DataFrame({"grade": ["A-", "4", "W"]})
    # Simulate school that only mapped plain letters, not plus/minus
    partial = resolve_es_grade_map({"A": "4", "B": "3"})
    mapped = apply_raw_course_grade_map(df, partial)
    counts = unmapped_gpa_grade_counts(mapped)
    assert "A-" not in counts.index  # platform default covers A-
    assert counts.empty or "4" not in counts.index


def test_log_unmapped_gpa_grades_warns(caplog) -> None:
    df = pd.DataFrame({"grade": ["X", "4"]})
    with caplog.at_level(logging.WARNING):
        log_unmapped_gpa_grades(df)
    assert any("non-numeric after grade_map" in r.message for r in caplog.records)


def test_chicago_style_config_gets_plus_minus_from_platform() -> None:
    """Plain A-F in config; plus/minus covered by platform letter defaults."""
    chicago = resolve_es_grade_map({"A": "4", "B": "3", "C": "2", "D": "1", "F": "0"})
    df = pd.DataFrame({"grade": ["A-", "B+", "C", "A"]})
    mapped = apply_raw_course_grade_map(df, chicago)
    assert mapped["grade"].tolist() == ["3.7", "3.3", "2", "4"]
    assert unmapped_gpa_grade_counts(mapped).empty
