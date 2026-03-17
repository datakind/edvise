import functools as ft

import pandas as pd
import pytest

from edvise import checkpoints
from edvise.targets import credits_earned


@pytest.mark.parametrize(
    [
        "df",
        "min_num_credits",
        "checkpoint",
        "intensity_time_limits",
        "num_terms_in_year",
        "max_term_rank",
        "exp",
    ],
    [
        # base case
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "01"],
                    "enrollment_intensity": ["FT", "FT", "FT", "FT"],
                    "num_credits": [12, 24, 36, 48],
                    "term_rank": [1, 2, 3, 5],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            45.0,
            # first term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01"],
                    "enrollment_intensity": ["FT"],
                    "num_credits": [12],
                    "term_rank": [1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [4, "term"]},
            2,
            "infer",
            pd.Series(
                data=[False],
                index=pd.Index(["01"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # multiple students, one full-time the other part-time
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "02"],
                    "enrollment_intensity": ["FT", "FT", "PT", "PT"],
                    "num_credits": [12, 48, 8, 64],
                    "term_rank": [1, 4, 1, 10],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            60.0,
            # first term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01", "02"],
                    "enrollment_intensity": ["FT", "PT"],
                    "num_credits": [12, 8],
                    "term_rank": [1, 1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [4, "term"], "PT": [8, "term"]},
            2,
            "infer",
            pd.Series(
                data=[True, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # checkpoint given as callable
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "02"],
                    "enrollment_intensity": ["FT", "FT", "PT", "PT"],
                    "num_credits": [12, 48, 8, 64],
                    "term_rank": [1, 4, 1, 10],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            60.0,
            # first term as checkpoint via callable
            ft.partial(
                checkpoints.nth_student_terms.first_student_terms,
                student_id_cols="student_id",
                sort_cols="term_rank",
                include_cols=["enrollment_intensity", "num_credits"],
            ),
            {"FT": [4, "term"], "PT": [8, "term"]},
            2,
            "infer",
            pd.Series(
                data=[True, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # not enough terms in dataset to compute target
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "01"],
                    "enrollment_intensity": ["FT", "PT", "PT", "FT"],
                    "num_credits": [12, 24, 36, 48],
                    "term_rank": [1, 2, 6, 7],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            45.0,
            # second term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01"],
                    "enrollment_intensity": ["PT"],
                    "num_credits": [24],
                    "term_rank": [3],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [4, "term"], "PT": [8, "term"]},
            4,
            "infer",
            pd.Series(
                data=[],
                index=pd.Index([], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # one time limit for all enrollment intensities
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "02"],
                    "enrollment_intensity": ["FT", "FT", "PT", "PT"],
                    "num_credits": [12, 48, 8, 64],
                    "term_rank": [1, 4, 1, 8],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            60.0,
            # first term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01", "02"],
                    "enrollment_intensity": ["FT", "PT"],
                    "num_credits": [12, 8],
                    "term_rank": [1, 1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"*": [8, "term"]},
            2,
            10,
            pd.Series(
                data=[True, False],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # num_terms_in_year=4 subtracts 2 for eligibility; with FT [12, "term"],
        # checkpoint at 1, max_term_rank=11: num_terms_in_year=4 (subtract 2) yields
        # labelable student, num_terms_in_year=2 (subtract 1) yields none
        (
            pd.DataFrame(
                {
                    "student_id": ["01"] * 11,
                    "enrollment_intensity": ["FT"] * 11,
                    "num_credits": [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132],
                    "term_rank": list(range(1, 12)),
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            45.0,
            pd.DataFrame(
                {
                    "student_id": ["01"],
                    "enrollment_intensity": ["FT"],
                    "num_credits": [12],
                    "term_rank": [1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [12, "term"]},
            4,
            11,
            pd.Series(
                data=[False],
                index=pd.Index(["01"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # same data but num_terms_in_year=2 (subtract 1) - not enough data, empty
        (
            pd.DataFrame(
                {
                    "student_id": ["01"] * 11,
                    "enrollment_intensity": ["FT"] * 11,
                    "num_credits": [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132],
                    "term_rank": list(range(1, 12)),
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            45.0,
            pd.DataFrame(
                {
                    "student_id": ["01"],
                    "enrollment_intensity": ["FT"],
                    "num_credits": [12],
                    "term_rank": [1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [12, "term"]},
            2,
            11,
            pd.Series(
                data=[],
                index=pd.Index([], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
    ],
)
def test_compute_target(
    df,
    min_num_credits,
    checkpoint,
    intensity_time_limits,
    num_terms_in_year,
    max_term_rank,
    exp,
):
    obs = credits_earned.compute_target(
        df,
        min_num_credits=min_num_credits,
        checkpoint=checkpoint,
        intensity_time_limits=intensity_time_limits,
        num_terms_in_year=num_terms_in_year,
        max_term_rank=max_term_rank,
        student_id_cols="student_id",
        enrollment_intensity_col="enrollment_intensity",
        num_credits_col="num_credits",
        term_rank_col="term_rank",
    )
    assert isinstance(obs, pd.Series)
    assert pd.testing.assert_series_equal(obs, exp) is None


def test_num_terms_in_year_from_config_affects_eligibility():
    """num_terms_in_year from config controls subtract (2 when 4, else 1) for eligibility."""
    df = pd.DataFrame(
        {
            "student_id": ["01"] * 11,
            "enrollment_intensity": ["FT"] * 11,
            "num_credits": [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132],
            "term_rank": list(range(1, 12)),
        },
    ).astype({"student_id": "string", "enrollment_intensity": "string"})
    checkpoint = pd.DataFrame(
        {
            "student_id": ["01"],
            "enrollment_intensity": ["FT"],
            "num_credits": [12],
            "term_rank": [1],
        },
    ).astype({"student_id": "string", "enrollment_intensity": "string"})

    # Simulate config kwargs (as pdp_targets does via model_dump())
    config_4 = {
        "min_num_credits": 45.0,
        "checkpoint": checkpoint,
        "intensity_time_limits": {"FT": [12, "term"]},
        "num_terms_in_year": 4,
        "max_term_rank": 11,
    }
    config_2 = {**config_4, "num_terms_in_year": 2}

    obs_4 = credits_earned.compute_target(df, **config_4)
    obs_2 = credits_earned.compute_target(df, **config_2)

    # num_terms_in_year=4 subtracts 2 → student labelable
    assert len(obs_4) == 1
    assert obs_4.index[0] == "01"
    # num_terms_in_year=2 subtracts 1 → needs more data → empty
    assert len(obs_2) == 0
