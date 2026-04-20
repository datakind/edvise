import pandas as pd
import pytest

from edvise.targets import graduation


@pytest.mark.parametrize(
    [
        "df",
        "intensity_time_limits",
        "num_terms_in_year",
        "max_term_rank",
        "student_id_cols",
        "exp",
    ],
    [
        # base case: all students labelable, one pos and one neg
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["PT", "FT", "FT", "PT", "PT"],
                    "years_to_degree": [4, 4, 4, 5, 5],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [True, False, False, False, False],
                    "term_is_core": [True, True, True, True, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"], "PT": [6, "year"]},
            2,
            15,
            "student_id",
            pd.Series(
                data=[True, False],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # lower max target term so part-time student isn't labelable
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["PT", "FT", "FT", "PT", "PT"],
                    "years_to_degree": [4, 4, 4, 5, 5],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [True, False, False, False, False],
                    "term_is_core": [True, True, True, True, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"], "PT": [6, "year"]},
            2,
            12,
            "student_id",
            pd.Series(
                data=[True, False],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # one time limit for all enrollment intensities
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["PT", "FT", "FT", "PT", "PT"],
                    "years_to_degree": [4, 4, 4, 5, 5],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [True, False, False, False, False],
                    "term_is_core": [True, True, True, True, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"*": [3, "year"]},
            2,
            15,
            "student_id",
            pd.Series(
                data=[True, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # reduce full-time / increase part-time students' years-to-degree
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["PT", "FT", "FT", "PT", "PT"],
                    "years_to_degree": [3, 3, 3, 7, 7],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [True, False, False, False, False],
                    "term_is_core": [True, True, True, True, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"], "PT": [6, "year"]},
            2,
            15,
            "student_id",
            pd.Series(
                data=[False, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # pathological case: years-to-degree varies across student terms
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["PT", "FT", "FT", "PT", "PT"],
                    "years_to_degree": [4, 5, 5, pd.NA, 5],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [True, False, False, False, False],
                    "term_is_core": [True, True, True, True, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"], "PT": [6, "year"]},
            2,
            15,
            "student_id",
            pd.Series(
                data=[True, False],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # four terms per year: fall start to spring of year 4 spans 15 terms; dataset
        # ending at term rank 15 must still label a 4-year on-time FT student (e.g. fall
        # 2020 cohort through spring 2024 graduation term).
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["FT", "FT", "FT", "FT", "FT"],
                    "years_to_degree": [4, 4, 4, 5, 5],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [False, False, False, False, False],
                    "term_is_core": [True, True, True, True, True],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [4, "year"]},
            4,
            15,
            "student_id",
            pd.Series(
                data=[False, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # Two terms per year: no N*4-1 adjustment; 3-year limit => 6 terms; checkpoint
        # rank 1 => last needed term rank 6 (inclusive window via shared helper).
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "02"],
                    "enrollment_intensity": ["FT", "FT", "FT", "FT"],
                    "years_to_degree": [3, 3, 4, 4],
                    "enrollment_year": [1, 1, 1, 1],
                    "term_rank": [1, 2, 1, 2],
                    "term_is_pre_cohort": [False, False, False, False],
                    "term_is_core": [True, True, True, True],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"]},
            2,
            6,
            "student_id",
            pd.Series(
                data=[False, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # Three terms per year: 2-year limit => 6 terms; same boundary at max_term_rank 6.
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "02"],
                    "enrollment_intensity": ["FT", "FT", "FT", "FT"],
                    "years_to_degree": [2, 2, 3, 3],
                    "enrollment_year": [1, 1, 1, 1],
                    "term_rank": [1, 2, 1, 2],
                    "term_is_pre_cohort": [False, False, False, False],
                    "term_is_core": [True, True, True, True],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [2, "year"]},
            3,
            6,
            "student_id",
            pd.Series(
                data=[False, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
    ],
)
def test_compute_target(
    df, intensity_time_limits, num_terms_in_year, max_term_rank, student_id_cols, exp
):
    obs = graduation.compute_target(
        df,
        intensity_time_limits=intensity_time_limits,
        num_terms_in_year=num_terms_in_year,
        max_term_rank=max_term_rank,
        student_id_cols=student_id_cols,
        enrollment_intensity_col="enrollment_intensity",
        years_to_degree_col="years_to_degree",
        enrollment_year_col="enrollment_year",
    )
    assert isinstance(obs, pd.Series)
    print("obs:", obs)
    print("exp:", exp)
    assert pd.testing.assert_series_equal(obs, exp) is None


@pytest.mark.parametrize(
    ["num_terms_in_year", "intensity_time_limits", "max_term_rank"],
    [
        # 3-year limit, 2 terms/year => 6 terms; need max_term_rank >= 6 (rank 5 drops all)
        (2, {"FT": [3, "year"]}, 5),
        # 2-year limit, 3 terms/year => 6 terms; same boundary
        (3, {"FT": [2, "year"]}, 5),
    ],
)
def test_graduation_two_three_term_insufficient_max_term_rank_yields_no_labels(
    num_terms_in_year, intensity_time_limits, max_term_rank
):
    """Eligibility uses full year→term counts for 2- and 3-term schools (no extra -1)."""
    df = pd.DataFrame(
        {
            "student_id": ["01", "01", "02", "02"],
            "enrollment_intensity": ["FT", "FT", "FT", "FT"],
            "years_to_degree": [3, 3, 4, 4],
            "enrollment_year": [1, 1, 1, 1],
            "term_rank": [1, 2, 1, 2],
            "term_is_pre_cohort": [False, False, False, False],
            "term_is_core": [True, True, True, True],
        },
    ).astype({"student_id": "string", "enrollment_intensity": "string"})
    if num_terms_in_year == 3:
        df = df.assign(years_to_degree=[2, 2, 3, 3])

    obs = graduation.compute_target(
        df,
        intensity_time_limits=intensity_time_limits,
        num_terms_in_year=num_terms_in_year,
        max_term_rank=max_term_rank,
        student_id_cols="student_id",
        enrollment_intensity_col="enrollment_intensity",
        years_to_degree_col="years_to_degree",
        enrollment_year_col="enrollment_year",
    )
    assert len(obs) == 0
