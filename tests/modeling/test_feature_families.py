import tomllib
from pathlib import Path

from edvise.model_prep.target_associations import feature_family
from edvise.modeling.feature_families import FEATURE_FAMILIES, quota_family

_FEATURES_TABLE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "edvise"
    / "shared"
    / "assets"
    / "features_table.toml"
)


def test_quota_families_match_the_coverage_buckets() -> None:
    assert len(FEATURE_FAMILIES["grades"]) == 38
    assert len(FEATURE_FAMILIES["credits"]) == 23
    assert len(FEATURE_FAMILIES["course_success"]) == 16
    assert len(FEATURE_FAMILIES["grade_variability"]) == 2
    assert len(FEATURE_FAMILIES["course_level"]) == 38
    assert len(FEATURE_FAMILIES["momentum"]) == 20
    assert len(FEATURE_FAMILIES["program_fit"]) == 11
    assert len(FEATURE_FAMILIES["section_context"]) == 4
    assert FEATURE_FAMILIES["enrollment"]


def test_quota_family_is_separate_from_prefix_twins() -> None:
    assert feature_family("frac_courses_course_grade_f") == "course_grade_f"
    assert feature_family("num_courses_course_grade_f") == "course_grade_f"
    assert quota_family("frac_courses_course_grade_f") == "grades"
    assert quota_family("num_courses_course_grade_f") == "grades"
    assert quota_family("frac_courses_passed") == "course_success"
    assert quota_family("not_a_modeled_column") is None


def test_quota_columns_are_defined_in_the_features_table() -> None:
    table = tomllib.loads(_FEATURES_TABLE.read_text())
    missing = [
        column
        for columns in FEATURE_FAMILIES.values()
        for column in columns
        if column not in table
    ]
    assert missing == []
