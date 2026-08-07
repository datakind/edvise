import pandas as pd
import pytest

from edvise.targets import retention


@pytest.mark.parametrize(
    ["df", "max_academic_year", "exp"],
    [
        # base case
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "04", "03"],
                    "retention": [True, True, False, False, pd.NA],
                    "cohort_id": [
                        "2020-21 FALL",
                        "2020-21 FALL",
                        "2021-22 SPRING",
                        "2022-23 FALL",
                        "2023-24 FALL",
                    ],
                    "term_id": [
                        "2020-21 FALL",
                        "2020-21 SPRING",
                        "2021-22 SPRING",
                        "2022-23 FALL",
                        "2023-24 FALL",
                    ],
                },
            ).astype({"student_id": "string", "retention": "boolean"}),
            "infer",
            pd.Series(
                data=[False, True, True],
                index=pd.Index(["01", "02", "04"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # integer values for retention instead of integers
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "04", "03"],
                    "retention": [1, 1, 0, 0, pd.NA],
                    "cohort_id": [
                        "2020-21 FALL",
                        "2020-21 FALL",
                        "2021-22 SPRING",
                        "2022-23 FALL",
                        "2023-24 FALL",
                    ],
                    "term_id": [
                        "2020-21 FALL",
                        "2020-21 SPRING",
                        "2021-22 SPRING",
                        "2022-23 FALL",
                        "2023-24 FALL",
                    ],
                },
            ).astype({"student_id": "string", "retention": "Int8"}),
            "infer",
            pd.Series(
                data=[False, True, True],
                index=pd.Index(["01", "02", "04"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # manually specified (lower!) max academic year
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "04", "03"],
                    "retention": [True, True, False, False, pd.NA],
                    "cohort_id": [
                        "2020-21 FALL",
                        "2020-21 FALL",
                        "2021-22 SPRING",
                        "2022-23 FALL",
                        "2023-24 FALL",
                    ],
                    "term_id": [
                        "2020-21 FALL",
                        "2020-21 SPRING",
                        "2021-22 SPRING",
                        "2022-23 FALL",
                        "2023-24 FALL",
                    ],
                },
            ).astype({"student_id": "string", "retention": "boolean"}),
            "2022-23",
            pd.Series(
                data=[False, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # pathological case: retention varies across student-terms
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01"],
                    "retention": [True, False, pd.NA],
                    "cohort_id": ["2020-21 FALL", "2020-21 FALL", "2020-21 FALL"],
                    "term_id": ["2020-21 FALL", "2020-21 SPRING", "2021-22 FALL"],
                },
            ).astype({"student_id": "string", "retention": "boolean"}),
            "infer",
            pd.Series(
                data=[False],
                index=pd.Index(["01"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # Summer rolls to end year (like Fall 2025); Spring keeps start year
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "02", "03", "04", "05", "06"],
                    "retention": [True, False, True, False, True, False],
                    "cohort_id": [
                        "2023-24 FALL",
                        "2024-25 WINTER",
                        "2024-25 SPRING",
                        "2024-25 SUMMER",
                        "2024-25 FALL",
                        "2025-26 FALL",
                    ],
                    "term_id": [
                        "2023-24 FALL",
                        "2024-25 WINTER",
                        "2024-25 SPRING",
                        "2024-25 SUMMER",
                        "2024-25 FALL",
                        "2025-26 SUMMER",
                    ],
                },
            ).astype({"student_id": "string", "retention": "boolean"}),
            "2025",
            # kept: FALL/WINTER/SPRING 2024-25 (and earlier); excluded: SUMMER 2024-25, FALL 2025-26
            pd.Series(
                data=[False, True, False, False],
                index=pd.Index(
                    ["01", "02", "03", "05"], dtype="string", name="student_id"
                ),
                name="target",
                dtype="boolean",
            ),
        ),
    ],
)
def test_compute_target(df, max_academic_year, exp):
    obs = retention.compute_target(
        df,
        max_academic_year=max_academic_year,
        student_id_cols="student_id",
        retention_col="retention",
        cohort_id_col="cohort_id",
        term_id_col="term_id",
    )
    assert isinstance(obs, pd.Series)
    assert pd.testing.assert_series_equal(obs, exp) is None
