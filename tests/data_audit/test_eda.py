from pathlib import Path

import pandas as pd
import pytest

from edvise import data_audit, dataio
from edvise.data_audit.eda import EdaSummary


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
            "categories": ["First-Time", "Transfer-In"],
            "series": [
                {"name": "Full-Time", "data": [3.0, 5.0]},
                {"name": "Part-Time", "data": [0.0, 1.0]},
            ],
        }

    def test_pell_recipient_by_first_gen(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).pell_recipient_by_first_gen == None

    def test_pell_recipient_status(self, sample_cohort_data):
        assert EdaSummary(sample_cohort_data).pell_recipient_status == {
            "series": [{"name": "All Students", "data": {"N": 1, "Nan": 6, "Y": 2}}]
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
            "categories": ["Hispanic", "Nonresident Alien", "White"],
            "series": [
                {"name": "No", "data": [0.0, 1.0, 0.0]},
                {"name": "Yes", "data": [1.0, 0.0, 1.0]},
            ],
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

    def test_pell_recipient_status_handles_nulls(self, sample_cohort_data):
        """Test that NaN pell status values are properly excluded."""
        sample_cohort_data.loc[0:2, "pell_status_first_year"] = pd.NA
        eda = EdaSummary(sample_cohort_data)
        result = eda.pell_recipient_status
        assert "series" in result
        data_keys = result["series"][0]["data"].keys()
        assert all(pd.notna(k) for k in data_keys)

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
