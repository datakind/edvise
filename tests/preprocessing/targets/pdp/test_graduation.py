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
        # num_terms_in_year=4 subtracts 2 for eligibility; with max_term_rank=11,
        # num_terms_in_year=2 (subtract 1) yields labelable students, but
        # num_terms_in_year=4 (subtract 2) would need more data so none labelable
        (
            pd.DataFrame(
                {
                    "student_id": [
                        "01",
                        "01",
                        "01",
                        "01",
                        "01",
                        "01",
                        "01",
                        "02",
                        "02",
                        "02",
                    ],
                    "enrollment_intensity": [
                        "PT",
                        "FT",
                        "FT",
                        "FT",
                        "FT",
                        "FT",
                        "FT",
                        "PT",
                        "PT",
                        "PT",
                    ],
                    "years_to_degree": [4, 4, 4, 4, 4, 4, 4, 5, 5, 5],
                    "enrollment_year": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
                    "term_is_pre_cohort": [
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    "term_is_core": [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"], "PT": [6, "year"]},
            4,
            11,
            "student_id",
            pd.Series(
                data=[],
                index=pd.Index([], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # same data but num_terms_in_year=2 (subtract 1) - FT student now labelable
        (
            pd.DataFrame(
                {
                    "student_id": [
                        "01",
                        "01",
                        "01",
                        "01",
                        "01",
                        "01",
                        "01",
                        "02",
                        "02",
                        "02",
                    ],
                    "enrollment_intensity": [
                        "PT",
                        "FT",
                        "FT",
                        "FT",
                        "FT",
                        "FT",
                        "FT",
                        "PT",
                        "PT",
                        "PT",
                    ],
                    "years_to_degree": [4, 4, 4, 4, 4, 4, 4, 5, 5, 5],
                    "enrollment_year": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
                    "term_is_pre_cohort": [
                        True,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    "term_is_core": [
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"], "PT": [6, "year"]},
            2,
            11,
            "student_id",
            pd.Series(
                data=[True],
                index=pd.Index(["01"], dtype="string", name="student_id"),
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


def test_num_terms_in_year_from_config_affects_eligibility():
    """num_terms_in_year from config controls subtract (2 when 4, else 1) for eligibility."""
    df = pd.DataFrame(
        {
            "student_id": ["01", "01", "01", "01", "01", "01", "01"],
            "enrollment_intensity": ["PT", "FT", "FT", "FT", "FT", "FT", "FT"],
            "years_to_degree": [4, 4, 4, 4, 4, 4, 4],
            "enrollment_year": [1, 1, 1, 1, 1, 1, 1],
            "term_rank": [1, 2, 3, 4, 5, 6, 7],
            "term_is_pre_cohort": [True, False, False, False, False, False, False],
            "term_is_core": [True, True, True, True, True, True, True],
        },
    ).astype({"student_id": "string", "enrollment_intensity": "string"})

    # Simulate config kwargs (as pdp_targets does via model_dump())
    config_4 = {
        "intensity_time_limits": {"FT": [3, "year"], "PT": [6, "year"]},
        "years_to_degree_col": "years_to_degree",
        "num_terms_in_year": 4,
        "max_term_rank": 11,
    }
    config_2 = {**config_4, "num_terms_in_year": 2}

    obs_4 = graduation.compute_target(
        df,
        **config_4,
        enrollment_intensity_col="enrollment_intensity",
        enrollment_year_col="enrollment_year",
    )
    obs_2 = graduation.compute_target(
        df,
        **config_2,
        enrollment_intensity_col="enrollment_intensity",
        enrollment_year_col="enrollment_year",
    )

    # num_terms_in_year=4 subtracts 2 → needs more data → empty
    assert len(obs_4) == 0
    # num_terms_in_year=2 subtracts 1 → student labelable
    assert len(obs_2) == 1
    assert obs_2.index[0] == "01"
