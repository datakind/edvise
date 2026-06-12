"""Tests for ES grade_map defaults and merge."""

from edvise.data_audit.default_grade_map import DEFAULT_ES_GRADE_MAP
from edvise.data_audit.raw_course_grade_map import (
    merge_grade_maps,
    resolve_es_grade_map,
)


def test_resolve_es_grade_map_includes_defaults() -> None:
    resolved = resolve_es_grade_map(None)
    assert resolved["W1"] == "W"
    assert resolved["OTHER"] == "O"


def test_institution_grade_map_overrides_default() -> None:
    resolved = resolve_es_grade_map({"NC": "NG", "A": "4"})
    assert resolved["NC"] == "NG"
    assert resolved["A"] == "4"
    assert resolved["W1"] == "W"


def test_merge_grade_maps_empty_override() -> None:
    assert merge_grade_maps(DEFAULT_ES_GRADE_MAP, None) == resolve_es_grade_map(None)
