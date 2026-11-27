from pathlib import Path
import logging
import types

import pandas as pd
import pytest

from edvise import data_audit, dataio
from edvise.data_audit import eda
from edvise.data_audit.eda import EdaSummary, log_grade_distribution


@pytest.mark.parametrize(
    ["df", "ref_col", "exclude_cols", "exp"],
    [
        (
            pd.DataFrame(
                {"col1": [1, 2, 3], "col2": [2, 3, 4], "col3": [3.0, 2.0, 1.0]}
            ),
            None,
            None,
            pd.DataFrame(
                data=[[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
                index=["col1", "col2", "col3"],
                columns=["col1", "col2", "col3"],
                dtype="Float32",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "col1": [1, 2, 3],
                    "col2": [2.0, 2.5, 3.0],
                    "col3": ["A", "A", "B"],
                    "col4": ["X", "Y", "X"],
                }
            ).astype({"col3": "string", "col4": "category"}),
            None,
            None,
            pd.DataFrame(
                data=[
                    [1.0, 1.0, 0.866025, 0.0],
                    [1.0, 1.0, 0.866025, 0.0],
                    [0.866025, 0.866025, 1.0, 0.5],
                    [0.0, 0.0, 0.5, 1.0],
                ],
                index=["col1", "col2", "col3", "col4"],
                columns=["col1", "col2", "col3", "col4"],
                dtype="Float32",
            ),
        ),
        (
            pd.DataFrame(
                {"col1": [1, 2, 3], "col2": [2, 3, 4], "col3": [3.0, 2.0, 1.0]}
            ),
            "col3",
            None,
            pd.DataFrame(
                data=[-1.0, -1.0, 1.0],
                index=["col1", "col2", "col3"],
                columns=["col3"],
                dtype="Float32",
            ),
        ),
        (
            pd.DataFrame(
                {"col1": [1, 2, 3], "col2": [2, 3, 4], "col3": [3.0, 2.0, 1.0]}
            ),
            None,
            "col3",
            pd.DataFrame(
                data=[[1.0, 1.0], [1.0, 1.0]],
                index=["col1", "col2"],
                columns=["col1", "col2"],
                dtype="Float32",
            ),
        ),
    ],
)
def test_compute_pairwise_associations(df, ref_col, exclude_cols, exp):
    obs = data_audit.eda.compute_pairwise_associations(
        df, ref_col=ref_col, exclude_cols=exclude_cols
    )
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp) is None


class TestEdaSummary:
    @pytest.fixture
    def sample_cohort_data(self):
        from edvise.data_audit.schemas import RawPDPCohortDataSchema

        file_path = Path(__file__).parents[1] / "fixtures" / "raw_pdp_cohort_data.csv"
        df = dataio.read.read_raw_pdp_cohort_data(
            file_path=str(file_path), schema=RawPDPCohortDataSchema
        )
        return df

    @pytest.fixture
    def sample_course_data(self):
        from edvise.data_audit.schemas import RawPDPCourseDataSchema

        file_path = Path(__file__).parents[1] / "fixtures" / "raw_pdp_course_data.csv"
        return dataio.read.read_raw_pdp_course_data(
            file_path=str(file_path), schema=RawPDPCourseDataSchema
        )

    def test_cohort_years(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data)
        assert eda.cohort_years(formatted=True) == [
            "2011 - 12",
            "2013 - 14",
            "2015 - 16",
            "2016 - 17",
            "2017 - 18",
            "2018 - 19",
        ]
        assert eda.cohort_years(formatted=False) == [
            "2011-12",
            "2013-14",
            "2015-16",
            "2016-17",
            "2017-18",
            "2018-19",
        ]

    def test_summary_metrics(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data)
        assert eda.total_students == {"name": "Total Students", "value": 9}
        assert eda.transfer_students == {"name": "Transfer Students", "value": 6}
        assert eda.avg_year1_gpa_all_students == {
            "name": "Avg. Year 1 GPA - All Students",
            "value": 3.49,
        }

    def test_avg_year1_gpa_all_students_returns_zero_for_invalid_gpa(self):
        df = pd.DataFrame(
            {
                "student_id": ["student-1", "student-2"],
                "enrollment_type": ["FIRST-TIME", "TRANSFER-IN"],
                "gpa_group_year_1": ["invalid", None],
            }
        )
        eda = EdaSummary(df)
        val = eda.avg_year1_gpa_all_students["value"]
        assert val == 0.0 or pd.isna(val)

    def test_gpa_by_enrollment_type(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).gpa_by_enrollment_type == {
            "cohort_years": [
                "2011 - 12",
                "2013 - 14",
                "2015 - 16",
                "2016 - 17",
                "2017 - 18",
                "2018 - 19",
            ],
            "series": [
                {"name": "First-Time", "data": [None, None, None, 3.65, 3.8, 2.85]},
                {"name": "Transfer-In", "data": [3.12, 3.72, 3.66, 3.9, None, 3.55]},
            ],
            "min_gpa": 2.85,
        }

    def test_gpa_by_enrollment_intensity(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).gpa_by_enrollment_intensity == {
            "cohort_years": [
                "2011 - 12",
                "2013 - 14",
                "2015 - 16",
                "2016 - 17",
                "2017 - 18",
                "2018 - 19",
            ],
            "series": [
                {"name": "Full-Time", "data": [3.12, 3.72, None, 3.78, 3.8, 3.2]},
                {"name": "Part-Time", "data": [None, None, 3.66, None, None, None]},
            ],
            "min_gpa": 3.12,
        }

    def test_students_by_cohort_term(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).students_by_cohort_term == {
            "years": [
                "2011 - 12",
                "2013 - 14",
                "2015 - 16",
                "2016 - 17",
                "2017 - 18",
                "2018 - 19",
            ],
            "by_year": [
                {
                    "year": "2011 - 12",
                    "total": 2,
                    "terms": [
                        {"count": 2, "percentage": 100.0, "name": "Fall"},
                        {"count": 0, "percentage": 0.0, "name": "Spring"},
                    ],
                },
                {
                    "year": "2013 - 14",
                    "total": 1,
                    "terms": [
                        {"count": 1, "percentage": 100.0, "name": "Fall"},
                        {"count": 0, "percentage": 0.0, "name": "Spring"},
                    ],
                },
                {
                    "year": "2015 - 16",
                    "total": 1,
                    "terms": [
                        {"count": 0, "percentage": 0.0, "name": "Fall"},
                        {"count": 1, "percentage": 100.0, "name": "Spring"},
                    ],
                },
                {
                    "year": "2016 - 17",
                    "total": 2,
                    "terms": [
                        {"count": 2, "percentage": 100.0, "name": "Fall"},
                        {"count": 0, "percentage": 0.0, "name": "Spring"},
                    ],
                },
                {
                    "year": "2017 - 18",
                    "total": 1,
                    "terms": [
                        {"count": 0, "percentage": 0.0, "name": "Fall"},
                        {"count": 1, "percentage": 100.0, "name": "Spring"},
                    ],
                },
                {
                    "year": "2018 - 19",
                    "total": 2,
                    "terms": [
                        {"count": 2, "percentage": 100.0, "name": "Fall"},
                        {"count": 0, "percentage": 0.0, "name": "Spring"},
                    ],
                },
            ],
        }

    def test_course_enrollments(self, sample_cohort_data, sample_course_data):
        assert EdaSummary(
            sample_cohort_data, sample_course_data
        ).course_enrollments == {
            "years": [
                "2015 - 16",
                "2016 - 17",
                "2017 - 18",
                "2018 - 19",
                "2020 - 21",
                "2022 - 23",
            ],
            "by_year": [
                {
                    "year": "2015 - 16",
                    "total": 1,
                    "terms": [
                        {"count": 0, "percentage": 0, "name": "Fall"},
                        {"count": 1, "percentage": 100.0, "name": "Spring"},
                        {"count": 0, "percentage": 0, "name": "Summer"},
                    ],
                },
                {
                    "year": "2016 - 17",
                    "total": 1,
                    "terms": [
                        {"count": 0, "percentage": 0, "name": "Fall"},
                        {"count": 1, "percentage": 100.0, "name": "Spring"},
                        {"count": 0, "percentage": 0, "name": "Summer"},
                    ],
                },
                {
                    "year": "2017 - 18",
                    "total": 1,
                    "terms": [
                        {"count": 0, "percentage": 0, "name": "Fall"},
                        {"count": 1, "percentage": 100.0, "name": "Spring"},
                        {"count": 0, "percentage": 0, "name": "Summer"},
                    ],
                },
                {
                    "year": "2018 - 19",
                    "total": 3,
                    "terms": [
                        {"count": 0, "percentage": 0, "name": "Fall"},
                        {"count": 2, "percentage": 66.67, "name": "Spring"},
                        {"count": 1, "percentage": 33.33, "name": "Summer"},
                    ],
                },
                {
                    "year": "2020 - 21",
                    "total": 2,
                    "terms": [
                        {"count": 2, "percentage": 100.0, "name": "Fall"},
                        {"count": 0, "percentage": 0, "name": "Spring"},
                        {"count": 0, "percentage": 0, "name": "Summer"},
                    ],
                },
                {
                    "year": "2022 - 23",
                    "total": 1,
                    "terms": [
                        {"count": 0, "percentage": 0, "name": "Fall"},
                        {"count": 1, "percentage": 100.0, "name": "Spring"},
                        {"count": 0, "percentage": 0, "name": "Summer"},
                    ],
                },
            ],
        }

    def test_degree_types(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).degree_types == {
            "total": 9,
            "degrees": [{"count": 9, "percentage": 100.0, "name": "Bachelor's Degree"}],
        }

    def test_enrollment_type_by_intensity(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).enrollment_type_by_intensity == {
            "total": 9,
            "categories": ["First-Time", "Transfer-In"],
            "series": [
                {
                    "name": "Full-Time",
                    "data": [
                        {
                            "name": "First-Time",
                            "count": 3,
                            "percentage": 33.33,
                        },
                        {
                            "name": "Transfer-In",
                            "count": 5,
                            "percentage": 55.56,
                        },
                    ],
                },
                {
                    "name": "Part-Time",
                    "data": [
                        {"name": "First-Time", "count": 0, "percentage": 0},
                        {
                            "name": "Transfer-In",
                            "count": 1,
                            "percentage": 11.11,
                        },
                    ],
                },
            ],
        }

    def test_pell_recipient_by_first_gen(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).pell_recipient_by_first_gen == None

    def test_pell_recipient_status(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).pell_recipient_status == {
            "series": [{"name": "All Students", "data": {"No": 7, "Yes": 2}}]
        }

    def test_student_age_by_gender(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).student_age_by_gender == {
            "categories": ["F", "M"],
            "series": [
                {"name": "20 And Younger", "data": [2.0, 0.0]},
                {"name": ">20 - 24", "data": [3.0, 0.0]},
                {"name": "Older Than 24", "data": [1.0, 2.0]},
            ],
        }

    def test_race_by_pell_status(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).race_by_pell_status == {
            "categories": [
                "Asian",
                "Black Or African American",
                "Hispanic",
                "Nonresident Alien",
                "Unknown",
                "White",
            ],
            "series": [
                {"name": "No", "data": [1.0, 1.0, 2.0, 1.0, 1.0, 1.0]},
                {"name": "Yes", "data": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]},
            ],
        }

    def test_pell_recipient_status_handles_nulls(self, sample_cohort_data):
        """Missing pell status is imputed to No (same as N)"""
        sample_cohort_data.loc[0:2, "pell_status_first_year"] = pd.NA
        eda = EdaSummary(sample_cohort_data)
        result = eda.pell_recipient_status
        assert "series" in result
        data_keys = result["series"][0]["data"].keys()
        assert all(pd.notna(k) for k in data_keys)
        assert set(data_keys) <= {"Yes", "No"}

    def test_pell_recipient_status_maps_unknown_uk_to_no(self) -> None:
        df = pd.DataFrame(
            {
                "pell_status_first_year": ["Y", "UK", "Uk", "N", pd.NA],
                "first_gen": ["Y", "N", "Y", "N", "Y"],
            }
        )
        result = EdaSummary(df).pell_recipient_status
        assert result == {
            "series": [{"name": "All Students", "data": {"No": 4, "Yes": 1}}]
        }

    def test_pell_recipient_by_first_gen_maps_unknown_uk_to_no(self) -> None:
        df = pd.DataFrame(
            {
                "pell_status_first_year": ["Y", "UK", "Uk", pd.NA],
                "first_gen": ["Y", "N", "Y", "N"],
            }
        )
        result = EdaSummary(df).pell_recipient_by_first_gen
        assert result is not None
        assert set(result["categories"]) == {"No", "Yes"}
        assert "Uk" not in result["categories"]

    def test_race_by_pell_status_maps_unknown_uk_to_no(self) -> None:
        df = pd.DataFrame(
            {
                "pell_status_first_year": ["UK", "Y"],
                "race": ["White", "Hispanic"],
            }
        )
        result = EdaSummary(df).race_by_pell_status
        assert result["categories"] == ["Hispanic", "White"]
        assert [s["name"] for s in result["series"]] == ["No", "Yes"]
        no_series = next(s for s in result["series"] if s["name"] == "No")
        yes_series = next(s for s in result["series"] if s["name"] == "Yes")
        white_idx = result["categories"].index("White")
        hispanic_idx = result["categories"].index("Hispanic")
        assert no_series["data"][white_idx] == 1.0
        assert yes_series["data"][hispanic_idx] == 1.0

    def test_student_age_by_gender_handles_nulls(self, sample_cohort_data):
        """Test that NaN gender values are properly excluded."""
        sample_cohort_data.loc[0:5, "gender"] = pd.NA
        eda = EdaSummary(sample_cohort_data)
        result = eda.student_age_by_gender
        assert "categories" in result
        assert all(pd.notna(cat) for cat in result["categories"])

    def test_validate_false_preserves_data(self, sample_cohort_data):
        """Test that EdaSummary initializes and total_students matches cohort length."""
        eda = EdaSummary(sample_cohort_data)
        assert eda.total_students["value"] == len(sample_cohort_data)

    def test_cached_property_only_computes_once(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data)
        # Access multiple times - should only compute once
        first = eda.gpa_by_enrollment_type
        second = eda.gpa_by_enrollment_type
        # Cached properties return the same object reference
        assert first is second
        assert first == second  # Values should also be equal


class TestEdaSummaryEsColumnAliases:
    """ES cohort columns alias to PDP names so AC EDA charts still compute."""

    @pytest.fixture
    def sample_es_cohort_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "learner_id": ["L1", "L2", "L3", "L4"],
                "entry_year": ["2020-21", "2020-21", "2021-22", "2021-22"],
                "entry_term": ["Fall", "Spring", "Fall", "Fall"],
                "enrollment_type": [
                    "FIRST-TIME",
                    "TRANSFER-IN",
                    "FIRST-TIME",
                    "RE-ADMIT",
                ],
                "intended_program_type": [
                    "Associate of Arts",
                    "Bachelor of Science",
                    "Associate of Arts",
                    "Certificate",
                ],
                "pell_recipient_year1": ["Y", "N", "Y", "N"],
                "first_generation_status": ["Y", "N", "Y", "N"],
                "learner_age": [
                    "20 AND YOUNGER",
                    ">20 - 24",
                    "OLDER THAN 24",
                    "20 AND YOUNGER",
                ],
                "gender": ["F", "M", "F", "M"],
                "race": ["White", "Hispanic", "White", "Asian"],
            }
        )

    def test_normalize_aliases_es_columns(self, sample_es_cohort_data):
        from edvise.data_audit.eda import normalize_eda_cohort_columns

        out = normalize_eda_cohort_columns(sample_es_cohort_data)
        assert "student_id" in out.columns
        assert "cohort" in out.columns
        assert "cohort_term" in out.columns
        assert "credential_type_sought_year_1" in out.columns
        assert "pell_status_first_year" in out.columns
        assert "first_gen" in out.columns
        assert "student_age" in out.columns
        assert "learner_id" not in out.columns

    def test_pdp_column_names_preferred_over_es_aliases(self):
        from edvise.data_audit.eda import normalize_eda_cohort_columns

        df = pd.DataFrame(
            {
                "student_id": ["S1"],
                "learner_id": ["L1"],
                "cohort": ["2020-21"],
                "entry_year": ["2019-20"],
            }
        )
        out = normalize_eda_cohort_columns(df)
        assert list(out["student_id"]) == ["S1"]
        assert list(out["cohort"]) == ["2020-21"]
        assert "learner_id" in out.columns
        assert "entry_year" in out.columns

    def test_first_gen_alias_skipped_for_non_yn_values(self):
        from edvise.data_audit.eda import normalize_eda_cohort_columns

        df = pd.DataFrame(
            {
                "learner_id": ["L1", "L2"],
                "first_generation_status": ["First Gen", "Not First Gen"],
            }
        )
        out = normalize_eda_cohort_columns(df)
        assert "first_gen" not in out.columns
        assert "first_generation_status" in out.columns

    def test_ac_charts_work_with_es_column_names(self, sample_es_cohort_data):
        eda = EdaSummary(sample_es_cohort_data)

        assert eda.total_students == {"name": "Total Students", "value": 4}

        age = eda.student_age_by_gender
        assert age is not None
        assert set(age["categories"]) == {"F", "M"}
        assert {s["name"] for s in age["series"]} == {
            "20 And Younger",
            ">20 - 24",
            "Older Than 24",
        }

        cohort_term = eda.students_by_cohort_term
        assert cohort_term is not None
        assert cohort_term["years"] == ["2020 - 21", "2021 - 22"]
        assert len(cohort_term["by_year"]) == 2

        race = eda.race_by_pell_status
        assert race is not None
        assert set(race["categories"]) == {"Asian", "Hispanic", "White"}
        assert {s["name"] for s in race["series"]} == {"No", "Yes"}

        pell_fg = eda.pell_recipient_by_first_gen
        assert pell_fg is not None
        assert set(pell_fg["categories"]) == {"No", "Yes"}

        degrees = eda.degree_types
        assert degrees is not None
        assert degrees["total"] == 4
        names = {d["name"] for d in degrees["degrees"]}
        assert "Associate Of Arts" in names
        assert "Bachelor Of Science" in names

    def test_optional_es_demo_missing_hides_those_charts_only(
        self, sample_es_cohort_data
    ):
        df = sample_es_cohort_data.drop(
            columns=[
                "pell_recipient_year1",
                "first_generation_status",
                "race",
                "learner_age",
                "gender",
            ]
        )
        eda = EdaSummary(df)
        assert eda.total_students["value"] == 4
        assert eda.students_by_cohort_term is not None
        assert eda.degree_types is not None
        assert eda.pell_recipient_by_first_gen is None
        assert eda.pell_recipient_status is None
        assert eda.race_by_pell_status is None
        assert eda.student_age_by_gender is None


class TestEdaSummaryYear1FeatureDerivation:
    """Derive gpa_group_year_1 / enrollment_intensity_first_term from course rows."""

    @pytest.fixture
    def es_cohort(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "learner_id": ["L1", "L2", "L3"],
                "entry_year": ["2020-21", "2020-21", "2021-22"],
                "entry_term": ["Fall", "Fall", "Spring"],
                "enrollment_type": ["FIRST-TIME", "FIRST-TIME", "TRANSFER-IN"],
            }
        )

    @pytest.fixture
    def es_course(self) -> pd.DataFrame:
        # L1 Fall 2020-21: 3cr A + 3cr B; Spring 3cr A → year GPA (4*3+3*3+4*3)/9
        #     first-term credits = 6 → PART-TIME
        # L2 Fall 2020-21: 12cr A = 4.0 GPA; 12 credits → FULL-TIME
        # L2 later year course ignored for year-1 GPA
        # L3 Spring 2021-22: 4cr numeric 3.5 + 4cr P (excluded from GPA) + 4cr C
        #     → GPA (3.5*4+2*4)/8 = 2.75; first-term credits 12 → FULL-TIME
        return pd.DataFrame(
            {
                "learner_id": [
                    "L1",
                    "L1",
                    "L1",
                    "L2",
                    "L2",
                    "L3",
                    "L3",
                    "L3",
                ],
                "academic_year": [
                    "2020-21",
                    "2020 - 21",
                    "2020-21",
                    "2020-21",
                    "2021-22",
                    "2021-22",
                    "2021-22",
                    "2021-22",
                ],
                "academic_term": [
                    "Fall",
                    "FALL",
                    "Spring",
                    "Fall",
                    "Fall",
                    "Spring",
                    "Spring",
                    "Spring",
                ],
                "grade": ["A", "B", "A", "A", "C", "3.5", "P", "C"],
                "course_credits_attempted": [3, 3, 3, 12, 3, 4, 4, 4],
            }
        )

    def test_enrich_derives_credit_weighted_gpa_and_intensity(
        self, es_cohort, es_course
    ):
        from edvise.data_audit.eda import (
            enrich_eda_cohort_year1_features,
            normalize_eda_cohort_columns,
        )

        out = enrich_eda_cohort_year1_features(
            normalize_eda_cohort_columns(es_cohort), es_course
        )
        by_id = out.set_index("student_id")

        assert by_id.loc["L1", "gpa_group_year_1"] == pytest.approx(33 / 9)
        assert by_id.loc["L2", "gpa_group_year_1"] == pytest.approx(4.0)
        assert by_id.loc["L3", "gpa_group_year_1"] == pytest.approx(2.75)

        assert by_id.loc["L1", "enrollment_intensity_first_term"] == "PART-TIME"
        assert by_id.loc["L2", "enrollment_intensity_first_term"] == "FULL-TIME"
        assert by_id.loc["L3", "enrollment_intensity_first_term"] == "FULL-TIME"

    def test_eda_summary_charts_use_derived_fields(self, es_cohort, es_course):
        eda = EdaSummary(es_cohort, es_course)
        assert eda.avg_year1_gpa_all_students["value"] == pytest.approx(
            (33 / 9 + 4.0 + 2.75) / 3, abs=0.01
        )
        intensity = eda.gpa_by_enrollment_intensity
        assert intensity is not None
        names = {s["name"] for s in intensity["series"]}
        assert names == {"Full-Time", "Part-Time"}

    def test_existing_pdp_columns_not_overwritten(self, es_course):
        from edvise.data_audit.eda import enrich_eda_cohort_year1_features

        cohort = pd.DataFrame(
            {
                "student_id": ["L1"],
                "cohort": ["2020-21"],
                "cohort_term": ["Fall"],
                "gpa_group_year_1": [1.0],
                "enrollment_intensity_first_term": ["FULL-TIME"],
            }
        )
        out = enrich_eda_cohort_year1_features(cohort, es_course)
        assert list(out["gpa_group_year_1"]) == [1.0]
        assert list(out["enrollment_intensity_first_term"]) == ["FULL-TIME"]

    def test_no_course_leaves_features_absent(self, es_cohort):
        eda = EdaSummary(es_cohort, None)
        assert "gpa_group_year_1" not in eda.df_cohort.columns
        assert "enrollment_intensity_first_term" not in eda.df_cohort.columns
        assert eda.avg_year1_gpa_all_students is None
        assert eda.gpa_by_enrollment_intensity is None

    def test_pass_fail_only_courses_skip_gpa_but_can_set_intensity(self):
        cohort = pd.DataFrame(
            {
                "learner_id": ["L1"],
                "entry_year": ["2020-21"],
                "entry_term": ["Fall"],
            }
        )
        course = pd.DataFrame(
            {
                "learner_id": ["L1", "L1"],
                "academic_year": ["2020-21", "2020-21"],
                "academic_term": ["Fall", "Fall"],
                "grade": ["P", "W"],
                "course_credits_attempted": [6, 6],
            }
        )
        eda = EdaSummary(cohort, course)
        assert "gpa_group_year_1" not in eda.df_cohort.columns
        assert eda.df_cohort.loc[0, "enrollment_intensity_first_term"] == "FULL-TIME"


class TestEdaSummaryEnrollmentTypeNormalization:
    """ES enrollment_type labels map to PDP codes for transfer + GPA-by-type charts."""

    def test_normalize_maps_bronx_style_labels(self):
        from edvise.data_audit.eda import normalize_eda_enrollment_type_values

        df = pd.DataFrame(
            {
                "student_id": ["S1", "S2", "S3"],
                "enrollment_type": [
                    "Transfer",
                    "First-time freshman",
                    "TRANSFER-IN",
                ],
            }
        )
        out = normalize_eda_enrollment_type_values(df)
        assert list(out["enrollment_type"]) == [
            "TRANSFER-IN",
            "FIRST-TIME",
            "TRANSFER-IN",
        ]

    def test_transfer_metric_and_gpa_by_type_use_normalized_labels(self):
        cohort = pd.DataFrame(
            {
                "learner_id": ["L1", "L2"],
                "entry_year": ["2020-21", "2020-21"],
                "entry_term": ["Fall", "Fall"],
                "enrollment_type": ["Transfer", "First-time freshman"],
                "gpa_group_year_1": [2.5, 3.0],
                "enrollment_intensity_first_term": ["FULL-TIME", "PART-TIME"],
            }
        )
        eda = EdaSummary(cohort)

        assert eda.transfer_students == {
            "name": "Transfer Students",
            "value": 1,
        }

        gpa_type = eda.gpa_by_enrollment_type
        assert gpa_type is not None
        names = {s["name"] for s in gpa_type["series"]}
        assert names == {"First-Time", "Transfer-In"}
        assert gpa_type["min_gpa"] == pytest.approx(2.5)

        intensity_xtab = eda.enrollment_type_by_intensity
        assert intensity_xtab is not None
        assert set(intensity_xtab["categories"]) == {"First-Time", "Transfer-In"}


def test_log_grade_distribution_flags_lack_of_numeric_grades(caplog):
    """Flag when grades are only status codes (P, F, I, W, A, M, O)."""
    df = pd.DataFrame({"grade": ["P", "F", "P", "W", "I", "M"]})
    log_grade_distribution(df)
    assert "No numeric grades detected" in caplog.text
    assert "P=Pass" in caplog.text
    assert "Unique values" in caplog.text


def test_log_grade_distribution_no_flag_when_numeric_grades_present(caplog):
    """Do not flag when numeric or GPA letter grades exist."""
    df = pd.DataFrame({"grade": ["4.0", "3.5", "P", "F"]})
    log_grade_distribution(df)
    assert "No numeric grades detected" not in caplog.text


def test_log_grade_distribution_no_flag_when_letter_grades_present(caplog):
    """Do not flag when GPA letter grades (B+, C, etc.) exist."""
    df = pd.DataFrame({"grade": ["B+", "C", "A-", "P"]})
    log_grade_distribution(df)
    assert "No numeric grades detected" not in caplog.text


def test_log_grade_distribution_no_flag_when_empty_or_all_null(caplog):
    """Do not flag when there are no grades."""
    df = pd.DataFrame({"grade": [pd.NA, pd.NA]})
    log_grade_distribution(df)
    # No grades at all - should not trigger the numeric check (total_grades == 0)
    assert "No numeric grades detected" not in caplog.text


def test_log_grade_distribution_no_flag_when_numeric_gpa_scale(caplog):
    """Do not flag when numeric grades on GPA scale (1, 2, 3, 4) exist."""
    df = pd.DataFrame({"grade": ["1", "2", "3", "4", "3.5"]})
    log_grade_distribution(df)
    assert "No numeric grades detected" not in caplog.text


def test_log_grade_distribution_flags_only_o_status(caplog):
    """Flag when only O (Other) status grades present."""
    df = pd.DataFrame({"grade": ["O", "O"]})
    log_grade_distribution(df)
    assert "No numeric grades detected" in caplog.text


def test_log_grade_distribution_mix_status_and_numeric_no_flag(caplog):
    """Do not flag when mix of status codes and numeric grades."""
    df = pd.DataFrame({"grade": ["P", "4.0", "F", "3.0", "W"]})
    log_grade_distribution(df)
    assert "No numeric grades detected" not in caplog.text


# -----------------------------------------------------------------------------
# Anomaly percent logging tests
# -----------------------------------------------------------------------------


class TestCheckEarnedVsAttemptedPercent:
    """Tests that check_earned_vs_attempted includes percent of data in summary."""

    def test_summary_includes_pct_columns(self):
        """Summary DataFrame has *_pct columns for each anomaly type."""
        df = pd.DataFrame(
            {
                "student_id": [f"s{i}" for i in range(100)],
                "credits_attempted": [3] * 95
                + [
                    2,
                    2,
                    2,
                    0,
                    0,
                ],  # 3 anomalies: earned>attempted, 2: earned when 0 attempt
                "credits_earned": [3] * 95
                + [4, 4, 4, 1, 1],  # last 2 have earned when attempted=0
            }
        )
        result = data_audit.eda.check_earned_vs_attempted(
            df, earned_col="credits_earned", attempted_col="credits_attempted"
        )
        summary = result["summary"]

        assert "earned_gt_attempted_pct" in summary.columns
        assert "earned_when_no_attempt_pct" in summary.columns
        assert "total_anomalous_rows_pct" in summary.columns

        # 3 earned>attempted (indices 95,96,97) + 2 earned when no attempt (98,99) = 5 total
        assert summary["total_anomalous_rows"].iloc[0] == 5
        assert summary["total_anomalous_rows_pct"].iloc[0] == 5.0  # 5/100

    def test_percent_correct_for_earned_gt_attempted(self):
        """earned_gt_attempted_pct = 100 * count / total_rows."""
        df = pd.DataFrame(
            {
                "x": range(1000),
                "credits_attempted": [3] * 1000,
                "credits_earned": [3] * 985 + [4] * 15,  # 15 anomalies
            }
        )
        result = data_audit.eda.check_earned_vs_attempted(
            df, earned_col="credits_earned", attempted_col="credits_attempted"
        )
        assert result["summary"]["earned_gt_attempted_pct"].iloc[0] == 1.5
        assert result["summary"]["total_anomalous_rows_pct"].iloc[0] == 1.5

    def test_zero_percent_when_no_anomalies(self):
        """All pct columns are 0 when no anomalies."""
        df = pd.DataFrame(
            {
                "x": range(50),
                "credits_attempted": [3] * 50,
                "credits_earned": [3] * 50,
            }
        )
        result = data_audit.eda.check_earned_vs_attempted(
            df, earned_col="credits_earned", attempted_col="credits_attempted"
        )
        assert result["summary"]["total_anomalous_rows"].iloc[0] == 0
        assert result["summary"]["total_anomalous_rows_pct"].iloc[0] == 0
        assert result["summary"]["earned_gt_attempted_pct"].iloc[0] == 0

    def test_zero_division_empty_dataframe(self):
        """Handles empty DataFrame without error; pct should be 0."""
        df = pd.DataFrame(columns=["credits_attempted", "credits_earned"])
        result = data_audit.eda.check_earned_vs_attempted(
            df, earned_col="credits_earned", attempted_col="credits_attempted"
        )
        assert result["summary"]["total_anomalous_rows_pct"].iloc[0] == 0


class TestCheckPfGradeConsistencyPercent:
    """Tests that check_pf_grade_consistency includes percent of data in summary."""

    def test_summary_includes_pct_columns(self):
        """Summary DataFrame has *_pct columns for each anomaly type."""
        df = pd.DataFrame(
            {
                "grade": ["A", "B", "F", "F", "P", "P"] * 100,
                "pass_fail_flag": ["P", "P", "F", "F", "P", "P"] * 100,
                "credits_earned": [3, 3, 0, 1, 3, 0]
                * 100,  # 1 F with credits, 1 P with 0 credits per 6 rows
            }
        )
        anomalies, summary = data_audit.eda.check_pf_grade_consistency(
            df, credits_col="credits_earned"
        )

        assert "earned_with_failing_grade_pct" in summary.columns
        assert "no_credits_with_passing_grade_pct" in summary.columns
        assert "grade_pf_disagree_pct" in summary.columns
        assert "total_anomalous_rows_pct" in summary.columns

    def test_percent_logged_in_warning(self, caplog):
        """LOGGER.warning includes percent when anomalies found."""
        df = pd.DataFrame(
            {
                "grade": ["F"] * 10,
                "pass_fail_flag": ["F"] * 10,
                "credits_earned": [1] * 10,  # 10 anomalies: F with credits
            }
        )
        data_audit.eda.check_pf_grade_consistency(df, credits_col="credits_earned")
        assert "Detected 10 PF/grade consistency anomalies" in caplog.text
        assert "% of data" in caplog.text

    def test_zero_percent_when_no_anomalies(self):
        """All pct columns are 0 when no anomalies."""
        df = pd.DataFrame(
            {
                "grade": ["A", "B", "F"],
                "pass_fail_flag": ["P", "P", "F"],
                "credits_earned": [3, 3, 0],
            }
        )
        _, summary = data_audit.eda.check_pf_grade_consistency(
            df, credits_col="credits_earned"
        )
        assert summary["total_anomalous_rows"].iloc[0] == 0
        assert summary["total_anomalous_rows_pct"].iloc[0] == 0


class TestValidateCreditConsistencyPercent:
    """Tests that validate_credit_consistency includes pct_of_data in summaries."""

    def test_course_anomalies_summary_has_pct_of_data(self):
        """course_anomalies_summary includes pct_of_data when anomalies exist."""
        course_df = pd.DataFrame(
            {
                "student_id": [f"s{i}" for i in range(100)],
                "semester": ["S1"] * 100,
                "course_credits_attempted": [3] * 95 + [2] * 5,
                "course_credits_earned": [3] * 95 + [4] * 5,  # 5 anomalies
            }
        )
        result = data_audit.eda.validate_credit_consistency(course_df=course_df)

        assert result["course_anomalies_summary"] is not None
        assert "pct_of_data" in result["course_anomalies_summary"]
        assert result["course_anomalies_summary"]["pct_of_data"] == 5.0  # 5/100

    def test_course_anomalies_percent_in_log(self, caplog):
        """Course anomaly log includes percent of data."""
        course_df = pd.DataFrame(
            {
                "student_id": ["s1", "s2"],
                "semester": ["S1", "S1"],
                "course_credits_attempted": [2, 3],
                "course_credits_earned": [3, 3],  # 1 anomaly
            }
        )
        data_audit.eda.validate_credit_consistency(course_df=course_df)
        assert "Detected 1 course-level anomalies" in caplog.text
        assert "50.00% of course data" in caplog.text

    def test_cohort_anomalies_percent_in_log(self, caplog):
        """Cohort anomaly log includes percent when anomalies found."""
        course_df = pd.DataFrame(
            {
                "student_id": ["s1"],
                "semester": ["S1"],
                "course_credits_attempted": [3],
                "course_credits_earned": [3],
            }
        )
        cohort_df = pd.DataFrame(
            {
                "student_id": ["c1", "c2", "c3", "c4", "c5"],
                "inst_tot_credits_attempted": [30, 30, 30, 20, 30],
                "inst_tot_credits_earned": [30, 30, 30, 25, 30],  # 1 anomaly
            }
        )
        data_audit.eda.validate_credit_consistency(
            course_df=course_df, cohort_df=cohort_df
        )
        assert "Detected 1 cohort-level anomalies" in caplog.text
        assert "20.00% of cohort data" in caplog.text

    def test_reconciliation_summary_has_pct_of_data(self):
        """reconciliation_summary includes pct_of_data when semester_df provided."""
        course_df = pd.DataFrame(
            {
                "student_id": ["s1", "s1", "s2"],
                "semester": ["S1", "S2", "S1"],
                "course_credits_attempted": [3, 3, 3],
                "course_credits_earned": [3, 3, 3],
            }
        )
        semester_df = pd.DataFrame(
            {
                "student_id": ["s1", "s1", "s2"],
                "semester": ["S1", "S2", "S1"],
                "number_of_semester_credits_attempted": [3, 3, 99],  # 1 mismatch
                "number_of_semester_credits_earned": [3, 3, 3],
            }
        )
        result = data_audit.eda.validate_credit_consistency(
            course_df=course_df, semester_df=semester_df
        )

        assert result["reconciliation_summary"] is not None
        assert "pct_of_data" in result["reconciliation_summary"]
        assert result["reconciliation_summary"]["mismatched_rows"] == 1
        assert result["reconciliation_summary"]["pct_of_data"] == round(100 * 1 / 3, 2)


@pytest.mark.parametrize(
    "course_number,expect_upper_level",
    [
        ("1023", False),
        ("10000", False),
        ("0995", False),
        ("0123", False),
        ("2034", True),
        ("30456", True),
        ("20000", True),
        ("501", True),
        ("9012", True),
    ],
    ids=[
        "1023",
        "10000",
        "0995",
        "0123",
        "2034",
        "30456",
        "20000",
        "501",
        "9012",
    ],
)
def test_compute_gateway_course_ids_first_digit_single_row(
    course_number, expect_upper_level
):
    """First digit of numeric part (<2 vs >=2) for varied lengths: 1023, 2034, 30456, etc."""
    df = pd.DataFrame(
        {
            "math_or_english_gateway": ["M"],
            "course_prefix": ["MATH"],
            "course_number": [course_number],
            "course_cip": ["27.0101"],
        }
    )
    _, _, has_upper, lower_ids, _ = data_audit.eda.compute_gateway_course_ids_and_cips(
        df
    )
    assert has_upper is expect_upper_level
    full_id = f"MATH{course_number}"
    if expect_upper_level:
        assert full_id not in lower_ids
    else:
        assert full_id in lower_ids


def test_compute_gateway_course_ids_mixed_varied_lengths_in_one_frame():
    """Multiple course numbers in one batch: lower = first digit 0–1, upper = 2–9."""
    df = pd.DataFrame(
        {
            "math_or_english_gateway": ["M"] * 7,
            "course_prefix": ["MATH"] * 7,
            "course_number": [
                "1023",
                "10000",
                "0995",
                "2034",
                "30456",
                "20000",
                "501",
            ],
            "course_cip": ["27.0101"] * 7,
        }
    )
    ids, _, has_upper, lower_ids, _ = (
        data_audit.eda.compute_gateway_course_ids_and_cips(df)
    )
    assert has_upper is True
    assert set(lower_ids) == {"MATH1023", "MATH10000", "MATH0995"}
    assert len(ids) == 7
    for n in ("2034", "30456", "20000", "501"):
        assert f"MATH{n}" not in lower_ids


def test_compute_gateway_course_ids_prefix_plus_digits_after_letters():
    """Digits after a letter prefix in course_number still use first numeric digit."""
    df = pd.DataFrame(
        {
            "math_or_english_gateway": ["E"],
            "course_prefix": [""],
            "course_number": ["ENG10000"],
            "course_cip": ["23.0101"],
        }
    )
    _, _, has_upper, lower_ids, _ = data_audit.eda.compute_gateway_course_ids_and_cips(
        df
    )
    assert has_upper is False
    assert "ENG10000" in lower_ids


# -------------------------------------------------------------------
# print_column_pct_breakdowns
# -------------------------------------------------------------------
def test_print_column_pct_breakdowns_logs_mapped_labels(caplog):
    """Test that print_column_pct_breakdowns logs labeled percent breakdowns."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.eda")

    df = pd.DataFrame(
        {
            "enrollment_type": ["FIRST-TIME", "FIRST-TIME", "TRANSFER-IN"],
            "intended_program_type": ["Associate", "Certificate", "Associate"],
        }
    )

    eda.print_column_pct_breakdowns(df, eda.ES_COHORT_PCT_BREAKDOWN_COLS)

    log_message = " ".join(r.getMessage() for r in caplog.records)
    assert "Percent breakdown for enrollment types" in log_message
    assert "Percent breakdown for intended program types" in log_message
    assert "FIRST-TIME" in log_message
    assert "Associate" in log_message


def test_print_column_pct_breakdowns_skips_missing_columns(caplog):
    """Test that missing columns are skipped with a warning."""
    caplog.set_level(logging.WARNING, logger="edvise.data_audit.eda")

    df = pd.DataFrame({"enrollment_type": ["FIRST-TIME", "TRANSFER-IN"]})

    eda.print_column_pct_breakdowns(df, eda.ES_COHORT_PCT_BREAKDOWN_COLS)

    messages = [r.getMessage() for r in caplog.records]
    assert any("intended_program_type" in msg for msg in messages)


# -------------------------------------------------------------------
# log_high_null_columns / check_variable_missingness
# -------------------------------------------------------------------
def test_log_high_null_columns_counts_missing_strings(caplog):
    """Null rate logs treat the string MISSING as missing, including mixed case."""
    caplog.set_level(logging.WARNING, logger="edvise.data_audit.eda")

    df = pd.DataFrame(
        {
            "race": ["A", "MISSING", "missing", " Missing ", "Asian"],
            "gender": ["F", "M", "F", "M", "NA"],
            "age": [1, None, None, None, None],
        }
    )

    eda.log_high_null_columns(df, threshold=0.2)

    messages = [r.getMessage() for r in caplog.records]
    assert any('Column "race" has 60.0%' in msg for msg in messages)
    assert any('Column "age" has 80.0%' in msg for msg in messages)
    assert not any("gender" in msg for msg in messages)


def test_log_high_null_columns_combines_nulls_and_missing_strings(caplog):
    """Pandas nulls and MISSING strings are added into one column missing rate."""
    caplog.set_level(logging.WARNING, logger="edvise.data_audit.eda")

    df = pd.DataFrame(
        {
            "ethnicity": [
                None,
                "MISSING",
                "MISSING",
                "Asian",
                "Asian",
                "Asian",
                "Asian",
                "Asian",
                "Asian",
                "Asian",
            ]
        }
    )

    eda.log_high_null_columns(df, threshold=0.2)

    assert any('Column "ethnicity" has 30.0%' in r.getMessage() for r in caplog.records)


def test_check_variable_missingness_counts_missing_strings(caplog):
    """MISSING strings count toward the missingness warning, not only pandas nulls."""
    caplog.set_level(logging.WARNING, logger="edvise.data_audit.eda")

    df = pd.DataFrame({"race": ["MISSING"] * 3 + [None] * 3 + ["Asian"] * 4})

    eda.check_variable_missingness(df, ["race"], null_threshold_pct=50.0)

    assert any(
        "missingness in 'race'" in r.getMessage() and "60.0%" in r.getMessage()
        for r in caplog.records
    )


def test_check_variable_missingness_does_not_count_other_sentinels(caplog):
    """Other placeholders such as NA stay out of the missingness total."""
    caplog.set_level(logging.WARNING, logger="edvise.data_audit.eda")

    df = pd.DataFrame({"race": ["NA"] * 6 + ["Asian"] * 4})

    eda.check_variable_missingness(df, ["race"], null_threshold_pct=50.0)

    assert not any("missingness in 'race'" in r.getMessage() for r in caplog.records)


# -------------------------------------------------------------------
# warn_null_graduation_target_columns
# -------------------------------------------------------------------
def test_warn_null_graduation_target_columns_noop_for_non_graduation(caplog):
    caplog.set_level(logging.WARNING, logger="edvise.data_audit.eda")
    df = pd.DataFrame({"first_year_to_associates_at_cohort_inst": [None, None]})
    target = types.SimpleNamespace(
        type_="retention",
        years_to_degree_col="first_year_to_associates_at_cohort_inst",
    )

    eda.warn_null_graduation_target_columns(df, target)

    assert not any("Graduation model" in r.getMessage() for r in caplog.records)


def test_warn_null_graduation_target_columns_warns_when_all_null(caplog):
    caplog.set_level(logging.WARNING, logger="edvise.data_audit.eda")
    df = pd.DataFrame({"first_year_to_associates_at_cohort_inst": [None, None]})
    target = types.SimpleNamespace(
        type_="graduation",
        years_to_degree_col="first_year_to_associates_at_cohort_inst",
    )

    eda.warn_null_graduation_target_columns(df, target)

    assert any(
        "first_year_to_associates_at_cohort_inst" in r.getMessage()
        and "100% null" in r.getMessage()
        for r in caplog.records
    )


def test_warn_null_graduation_target_columns_warns_when_missing(caplog):
    caplog.set_level(logging.WARNING, logger="edvise.data_audit.eda")
    df = pd.DataFrame({"other_col": [1, 2]})
    target = types.SimpleNamespace(
        type_="graduation",
        years_to_degree_col="first_year_to_associates_at_cohort_inst",
    )

    eda.warn_null_graduation_target_columns(df, target)

    assert any("missing from cohort data" in r.getMessage() for r in caplog.records)


def test_warn_null_graduation_target_columns_no_warning_when_populated(caplog):
    caplog.set_level(logging.WARNING, logger="edvise.data_audit.eda")
    df = pd.DataFrame({"first_year_to_associates_at_cohort_inst": [1, 2, None]})
    target = types.SimpleNamespace(
        type_="graduation",
        years_to_degree_col="first_year_to_associates_at_cohort_inst",
    )

    eda.warn_null_graduation_target_columns(df, target)

    assert not any("Graduation model" in r.getMessage() for r in caplog.records)


# -------------------------------------------------------------------
# log_top_majors
# -------------------------------------------------------------------
def test_log_top_majors_logs_top_10(caplog):
    """Test that log_top_majors logs the top 10 majors."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.eda")

    df = pd.DataFrame(
        {
            "program_of_study_term_1": ["CS"] * 50
            + ["EE"] * 30
            + ["ME"] * 20
            + ["CE"] * 15
            + ["BIO"] * 10
            + ["CHEM"] * 8
            + ["PHYS"] * 6
            + ["MATH"] * 4
            + ["STAT"] * 3
            + ["DATA"] * 2
            + ["ART"] * 1,
        }
    )

    eda.log_top_majors(df)

    messages = [r.getMessage() for r in caplog.records]
    log_message = " ".join(messages)

    assert "CS" in log_message
    assert "EE" in log_message
    assert "Top majors" in log_message


def test_log_top_majors_handles_less_than_10_majors(caplog):
    """Test that log_top_majors handles dataframes with less than 10 majors."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.eda")

    df = pd.DataFrame(
        {
            "program_of_study_term_1": ["CS", "CS", "EE", "EE", "ME"],
        }
    )

    eda.log_top_majors(df)

    messages = [r.getMessage() for r in caplog.records]
    assert any("Top majors" in msg for msg in messages)


def test_log_top_majors_handles_null_values(caplog):
    """Test that log_top_majors includes null values in counts."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.eda")

    df = pd.DataFrame(
        {
            "program_of_study_term_1": ["CS"] * 10 + [None] * 5 + ["EE"] * 3,
        }
    )

    eda.log_top_majors(df)

    messages = [r.getMessage() for r in caplog.records]
    assert any("Top majors" in msg for msg in messages)


def test_log_top_majors_custom_column(caplog):
    """Test that log_top_majors supports a schema-specific major column."""
    caplog.set_level(logging.INFO, logger="edvise.data_audit.eda")

    df = pd.DataFrame({"declared_major_at_entry": ["Nursing", "Nursing", "Business"]})

    eda.log_top_majors(df, major_col="declared_major_at_entry")

    log_message = " ".join(r.getMessage() for r in caplog.records)
    assert "declared_major_at_entry" in log_message
    assert "Nursing" in log_message


def test_log_top_majors_returns_none():
    """Test that log_top_majors returns None (it's a logging function)."""
    df = pd.DataFrame(
        {
            "program_of_study_term_1": ["CS", "EE"],
        }
    )

    result = eda.log_top_majors(df)
    assert result is None
