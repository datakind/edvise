import pandas as pd
import pytest

from edvise.model_prep.cleanup_features import (
    _FIRST_TERM_SNAPSHOTS,
    PDPCleanup,
    extra_snapshot_drop_cols,
)


def _cfg(type_: str | None, n: int | None = None, exclude_non_core: bool = True):
    checkpoint = type("Checkpoint", (), {"type_": type_, "n": n})()
    if type_ == "nth":
        checkpoint.exclude_non_core_terms = exclude_non_core
    return type(
        "Cfg",
        (),
        {"preprocessing": type("Prep", (), {"checkpoint": checkpoint})()},
    )()


@pytest.mark.parametrize(
    ("type_", "n", "exclude_non_core", "expected"),
    [
        ("first_within_cohort", None, True, list(_FIRST_TERM_SNAPSHOTS)),
        ("nth", 0, True, list(_FIRST_TERM_SNAPSHOTS)),
        ("nth", 1, True, ["program_of_study_year_1"]),
        ("nth", 1, False, []),
        ("nth", 3, True, []),
        ("first_at_num_credits_earned", None, True, []),
        (None, None, True, []),
    ],
)
def test_extra_snapshot_drop_cols(
    type_: str | None, n: int | None, exclude_non_core: bool, expected: list[str]
) -> None:
    assert extra_snapshot_drop_cols(_cfg(type_, n, exclude_non_core)) == expected


def test_extra_snapshot_drop_cols_without_config() -> None:
    assert extra_snapshot_drop_cols(None) == []


def _snapshot_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["s1"],
            "year_of_enrollment_at_cohort_inst": [2],
            "cumnum_terms_enrolled": [2],
            "enrollment_intensity_first_term": ["FULL-TIME"],
            "attendance_status_term_1": ["First-Time Full-Time"],
            "program_of_study_term_1": ["24.0101"],
            "program_of_study_year_1": ["24.0101"],
            "student_term_enrollment_intensity": ["FULL-TIME"],
            "term_program_of_study": ["27.0501"],
        }
    )


def test_first_term_checkpoint_drops_snapshots() -> None:
    cleaned = PDPCleanup().clean_up_labeled_dataset_cols_and_vals(
        _snapshot_df(), cfg=_cfg("first_within_cohort")
    )
    assert "student_term_enrollment_intensity" in cleaned.columns
    assert "term_program_of_study" in cleaned.columns
    for col in _FIRST_TERM_SNAPSHOTS:
        assert col not in cleaned.columns


def test_two_core_terms_drops_only_year1_program() -> None:
    cleaned = PDPCleanup().clean_up_labeled_dataset_cols_and_vals(
        _snapshot_df(), cfg=_cfg("nth", n=1)
    )
    assert "program_of_study_year_1" not in cleaned.columns
    assert "program_of_study_term_1" in cleaned.columns
    assert "term_program_of_study" in cleaned.columns


def test_later_checkpoint_keeps_snapshots() -> None:
    cleaned = PDPCleanup().clean_up_labeled_dataset_cols_and_vals(
        _snapshot_df(), cfg=_cfg("nth", n=3)
    )
    for col in _FIRST_TERM_SNAPSHOTS:
        assert col in cleaned.columns
