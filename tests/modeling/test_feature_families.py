from types import SimpleNamespace

from edvise.dataio.read import read_features_table
from edvise.model_prep.cleanup_features import (
    BaseCleanup,
    ESCleanup,
    PDPCleanup,
    extra_snapshot_drop_cols,
)
from edvise.model_prep.target_associations import feature_family
from edvise.modeling.feature_families import FEATURE_FAMILIES, quota_family


def test_quota_families_match_the_coverage_buckets() -> None:
    assert len(FEATURE_FAMILIES["grades"]) == 38
    assert len(FEATURE_FAMILIES["credits"]) == 23
    assert len(FEATURE_FAMILIES["course_success"]) == 16
    assert len(FEATURE_FAMILIES["grade_variability"]) == 2
    assert len(FEATURE_FAMILIES["course_level"]) == 38
    assert len(FEATURE_FAMILIES["momentum"]) == 20
    assert len(FEATURE_FAMILIES["program_fit"]) == 9
    assert len(FEATURE_FAMILIES["section_context"]) == 4
    assert len(FEATURE_FAMILIES["enrollment"]) == 16


def _checkpoint_cfg(**fields: object) -> SimpleNamespace:
    return SimpleNamespace(
        preprocessing=SimpleNamespace(checkpoint=SimpleNamespace(**fields))
    )


def test_quota_columns_are_not_dropped_by_pdp_or_es_cleanup() -> None:
    """Quota members stay off the PDP and ES cleanup drop lists.

    Includes columns dropped at every checkpoint and the first-term / year-1
    snapshots dropped when they duplicate the checkpoint.
    """
    snapshot_drops = (
        set(extra_snapshot_drop_cols(_checkpoint_cfg(type_="first_within_cohort")))
        | set(extra_snapshot_drop_cols(_checkpoint_cfg(type_="nth", n=0)))
        | set(
            extra_snapshot_drop_cols(
                _checkpoint_cfg(type_="nth", n=1, exclude_non_core_terms=True)
            )
        )
    )
    dropped = (
        set(PDPCleanup.cols_to_drop)
        | set(ESCleanup.cols_to_drop)
        | set(BaseCleanup.cols_to_drop_after_masking)
        | snapshot_drops
    )
    included = [
        column
        for columns in FEATURE_FAMILIES.values()
        for column in columns
        if column in dropped
    ]
    assert included == []


def test_quota_family_is_separate_from_prefix_twins() -> None:
    assert feature_family("frac_courses_course_grade_f") == "course_grade_f"
    assert feature_family("num_courses_course_grade_f") == "course_grade_f"
    assert quota_family("frac_courses_course_grade_f") == "grades"
    assert quota_family("num_courses_course_grade_f") == "grades"
    assert quota_family("frac_courses_passed") == "course_success"
    assert quota_family("not_a_modeled_column") is None


def test_quota_columns_are_defined_in_the_features_table() -> None:
    table = read_features_table("shared/assets/features_table.toml")
    missing = [
        column
        for columns in FEATURE_FAMILIES.values()
        for column in columns
        if column not in table
    ]
    assert missing == []
