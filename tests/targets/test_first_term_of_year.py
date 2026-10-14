"""Calendar effects of preprocessing.features.first_term_of_year."""

import pandas as pd
import pytest

from edvise.targets import shared
from edvise.utils.academic_terms import (
    academic_term_order,
    cohort_term_sort_ranks,
)


def test_academic_term_order():
    assert academic_term_order("FALL") == ("FALL", "WINTER", "SPRING", "SUMMER")
    assert academic_term_order("SUMMER") == ("SUMMER", "FALL", "WINTER", "SPRING")
    with pytest.raises(ValueError, match="FALL"):
        academic_term_order("WINTER")


def test_cohort_labels_sort_summer_first_when_summer_starts_the_year():
    order = cohort_term_sort_ranks("SUMMER")
    cohorts = ["fall 2024-25", "summer 2024-25", "spring 2024-25", "winter 2024-25"]

    def key(label: str) -> tuple[int, int]:
        term_key, year_part = label.split()
        return (int(year_part.split("-")[0]), order[term_key])

    assert sorted(cohorts, key=key) == [
        "summer 2024-25",
        "fall 2024-25",
        "winter 2024-25",
        "spring 2024-25",
    ]


def test_four_term_budget_drops_trailing_summer_only_for_fall_start():
    limits = {"FULL-TIME": 16.0, "PART-TIME": 24.0}
    assert shared.apply_on_time_term_budget(
        limits, num_terms_in_year=4, first_term_of_year="FALL"
    ) == {"FULL-TIME": 15.0, "PART-TIME": 23.0}
    assert shared.apply_on_time_term_budget(
        limits, num_terms_in_year=4, first_term_of_year="SUMMER"
    ) == {"FULL-TIME": 16.0, "PART-TIME": 24.0}
    assert shared.apply_on_time_term_budget(
        {"*": 6.0}, num_terms_in_year=3, first_term_of_year="FALL"
    ) == {"*": 6.0}


def test_summer_checkpoint_deadline_depends_on_first_term():
    df = pd.DataFrame(
        {
            "student_id": ["s"],
            "term_rank": [0],
            "academic_term": ["SUMMER"],
            "student_term_enrollment_intensity": ["FULL-TIME"],
            "term_is_core": [False],
        }
    )
    kwargs = dict(
        df=df,
        checkpoint=df.copy(),
        intensity_time_limits={"*": (2.0, "term")},
        max_term_rank=1,
        num_terms_in_year=4,
        student_id_cols="student_id",
    )
    fall_start = shared.get_students_with_max_target_term_in_dataset(
        **kwargs, first_term_of_year="FALL"
    )
    summer_start = shared.get_students_with_max_target_term_in_dataset(
        **kwargs, first_term_of_year="SUMMER"
    )
    assert fall_start.empty
    assert list(summer_start.index) == ["s"]
