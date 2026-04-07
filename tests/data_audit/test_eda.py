from pathlib import Path

import pandas as pd
import pytest

from edvise import data_audit, dataio
from edvise.data_audit.eda import (
    EdaSummary,
    infer_term_column,
    log_grade_distribution,
    term_column_name_hint_score,
    value_looks_like_term,
)


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

    def test_institution_report_in_result(self):
        """Human-readable report and next steps ship with validate_credit_consistency."""
        course_df = pd.DataFrame(
            {
                "student_id": ["s1"],
                "semester": ["S1"],
                "course_credits_attempted": [3],
                "course_credits_earned": [3],
            }
        )
        result = data_audit.eda.validate_credit_consistency(course_df=course_df)
        rep = result["institution_report"]
        assert isinstance(rep, str)
        assert "CREDIT CONSISTENCY" in rep
        assert "SUGGESTED NEXT STEPS" in rep
        assert "1) Course file" in rep

    def test_strict_columns_skips_course_check_when_names_not_in_frame(self):
        """strict_columns=True does not fall back to alternate credit column names."""
        course_df = pd.DataFrame(
            {
                "student_id": ["s1"],
                "semester": ["S1"],
                "course_credits_attempted": [3],
                "course_credits_earned": [3],
            }
        )
        result = data_audit.eda.validate_credit_consistency(
            course_df=course_df,
            strict_columns=True,
            course_credits_attempted_col="not_a_column",
            course_credits_earned_col="also_missing",
        )
        assert result["course_anomalies_summary"] is None


def test_value_looks_like_term_accepts_common_formats():
    assert value_looks_like_term("Spring 2024") is True
    assert value_looks_like_term("2024 spring") is True
    assert value_looks_like_term("not a term") is False


def test_term_column_name_hint_score():
    hints = ("entry_term", "term")
    assert term_column_name_hint_score("entry_term", hints) == 0.15
    assert term_column_name_hint_score("foo_entry_term_bar", hints) == 0.08
    assert term_column_name_hint_score("other", hints) == 0.0


def test_infer_term_column_picks_term_like_column():
    df = pd.DataFrame(
        {
            "student_id": [1, 2],
            "entry_term": ["Fall 2023", "Spring 2024"],
            "notes": ["x", "y"],
        }
    )
    col = infer_term_column(
        df,
        name_hints=("entry_term",),
    )
    assert col == "entry_term"


def test_infer_student_id_column_prefers_named_id():
    from edvise.data_audit import eda as data_audit_eda

    df = pd.DataFrame(
        {
            "emplid": ["A1", "A2", "A3"],
            "student_id": [1, 2, 3],
            "x": [1, 1, 1],
        }
    )
    col = data_audit_eda.infer_student_id_column(df)
    assert col == "student_id"


def test_normalize_student_id_column_renames():
    from edvise.data_audit import eda as data_audit_eda

    df = pd.DataFrame({"emplid": ["A1", "A2"], "y": [1, 2]})
    out, name = data_audit_eda.normalize_student_id_column(df)
    assert name == "student_id"
    assert "student_id" in out.columns
    assert list(out["student_id"]) == ["A1", "A2"]


def test_infer_inst_tot_credits_columns_distinct():
    from edvise.data_audit import eda as data_audit_eda

    df = pd.DataFrame(
        {
            "student_id": [1, 2],
            "inst_tot_credits_attempted": [120.0, 90.0],
            "inst_tot_credits_earned": [118.0, 88.0],
        }
    )
    a, e = data_audit_eda.infer_inst_tot_credits_columns(df)
    assert a == "inst_tot_credits_attempted"
    assert e == "inst_tot_credits_earned"


def test_infer_inst_tot_credits_columns_semester_style_names():
    from edvise.data_audit import eda as data_audit_eda

    df = pd.DataFrame(
        {
            "student_id": [1, 2],
            "term": ["Fall 2023", "Fall 2023"],
            "Cumulative Credits Attempted": [15.0, 18.0],
            "credits earned semester": [15.0, 17.0],
        }
    )
    a, e = data_audit_eda.infer_inst_tot_credits_columns(df)
    assert a == "Cumulative Credits Attempted"
    assert e == "credits earned semester"


def test_infer_student_audit_columns_includes_age():
    from edvise.data_audit import eda as data_audit_eda

    df = pd.DataFrame(
        {
            "student_id": [1, 2, 3],
            "entry_term": ["Fall 2023", "Fall 2023", "Spring 2024"],
            "entry_type": ["Transfer", "FTIC", "Transfer"],
            "student_age": [19, 20, 22],
            "first_gen": ["Y", "N", "Y"],
        }
    )
    term = data_audit_eda.infer_term_column(df, name_hints=("entry_term",))
    out = data_audit_eda.infer_student_audit_columns(df, term_col=term)
    assert out["age"] == "student_age"
    assert out["student_type"] == "entry_type"


def test_string_looks_like_age_bucket_common_labels():
    from edvise.data_audit import eda as data_audit_eda

    assert data_audit_eda.string_looks_like_age_bucket("<24")
    assert data_audit_eda.string_looks_like_age_bucket("20-24")
    assert data_audit_eda.string_looks_like_age_bucket("older than 24")
    assert data_audit_eda.string_looks_like_age_bucket("24+")
    assert data_audit_eda.string_looks_like_age_bucket("under 18") is True
    assert data_audit_eda.string_looks_like_age_bucket("random notes") is False


def test_infer_age_column_categorical_bands():
    from edvise.data_audit import eda as data_audit_eda

    df = pd.DataFrame(
        {
            "student_id": [1, 2, 3, 4],
            "age_band": ["<24", "20-24", "Older Than 24", "20-24"],
        }
    )
    col = data_audit_eda.infer_age_column(df, exclude_cols={"student_id"})
    assert col == "age_band"


def test_infer_course_credit_and_grade_pf_user_style_columns():
    from edvise.data_audit import eda as data_audit_eda

    course_df = pd.DataFrame(
        {
            "student_id": [1, 1],
            "term": ["Fall 2023", "Fall 2023"],
            "Credit Hours": [3.0, 4.0],
            "No. of Credits Earned": [3.0, 4.0],
            "Class Class Grade": ["A", "B"],
            "Class Completion Status": ["Y", "Y"],
        }
    )
    att, earn = data_audit_eda.infer_course_credit_columns(course_df)
    assert att == "Credit Hours"
    assert earn == "No. of Credits Earned"
    g, p = data_audit_eda.infer_course_grade_pf_columns(
        course_df, exclude_cols={c for c in (att, earn) if c}
    )
    assert g == "Class Class Grade"
    assert p == "Class Completion Status"


def test_percent_of_rows_helpers():
    from edvise.data_audit import eda as eda_mod

    assert eda_mod.percent_of_rows(1, 4) == 25.0
    assert eda_mod.percent_of_rows(0, 0) == 0.0


def test_value_counts_percent_df_named_series():
    from edvise.data_audit import eda as eda_mod

    s = pd.Series(["a", "b", "a"], name="x")
    out = eda_mod.value_counts_percent_df(s)
    assert list(out.columns) == ["x", "pct_of_rows"]
    assert out["pct_of_rows"].sum() == 100.0


def test_iter_pf_grade_anomaly_slices_yields_only_true_rows():
    from edvise.data_audit import eda as eda_mod

    anomalies = pd.DataFrame(
        {
            "earned_with_failing_grade": [True, False],
            "no_credits_with_passing_grade": [False, True],
            "grade_pf_disagree": [False, False],
        }
    )
    parts = list(eda_mod.iter_pf_grade_anomaly_slices(anomalies))
    assert [n for n, _ in parts] == [
        "earned_with_failing_grade",
        "no_credits_with_passing_grade",
    ]
    assert len(parts[0][1]) == 1 and len(parts[1][1]) == 1


def test_iter_pf_grade_anomaly_slices_skips_empty_subframes():
    from edvise.data_audit import eda as eda_mod

    anomalies = pd.DataFrame(
        {
            "earned_with_failing_grade": [True],
            "no_credits_with_passing_grade": [False],
            "grade_pf_disagree": [False],
        }
    )
    parts = list(eda_mod.iter_pf_grade_anomaly_slices(anomalies))
    assert [n for n, _ in parts] == ["earned_with_failing_grade"]
    assert len(parts[0][1]) == 1


def test_infer_pass_fail_flag_tuples_yn():
    from edvise.data_audit import eda as data_audit_eda

    s = pd.Series(["Y", "N", "Y", None])
    p, f = data_audit_eda.infer_pass_fail_flag_tuples(s)
    assert p == ("Y",) and f == ("N",)


def test_infer_check_pf_grade_list_kwargs_merges_observed_grade():
    from edvise.data_audit import eda as data_audit_eda

    df = pd.DataFrame(
        {
            "grade": ["A", "A+"],
            "pf": ["Y", "Y"],
        }
    )
    kw = data_audit_eda.infer_check_pf_grade_list_kwargs(df, "grade", "pf")
    assert kw["pass_flags"] == ("Y",) and kw["fail_flags"] == ("N",)
    assert "A+" in kw["passing_grades"]


def test_infer_semester_credit_aggregate_user_style_columns():
    from edvise.data_audit import eda as data_audit_eda

    sem = pd.DataFrame(
        {
            "student_id": [1],
            "term": ["Fall 2023"],
            "credit_hours": [15.0],
            "no_of_credits_earned": [15.0],
            "no_of_classes": [4],
        }
    )
    a, e, c = data_audit_eda.infer_semester_credit_aggregate_columns(sem)
    assert a == "credit_hours"
    assert e == "no_of_credits_earned"
    assert c == "no_of_classes"


def test_infer_semester_enrollment_intensity_prefers_ftpt():
    from edvise.data_audit import eda as data_audit_eda

    sem = pd.DataFrame(
        {
            "student_id": [1, 2],
            "term": ["T1", "T2"],
            "ftpt": ["FULL-TIME", "PART-TIME"],
            "credit_hours": [12.0, 6.0],
        }
    )
    col = data_audit_eda.infer_semester_enrollment_intensity_column(
        sem,
        exclude_cols={"student_id", "term", "credit_hours"},
    )
    assert col == "ftpt"


def test_infer_semester_enrollment_intensity_skips_instructor_ft_pt():
    from edvise.data_audit import eda as data_audit_eda

    sem = pd.DataFrame(
        {
            "student_id": [1],
            "strm": ["12023"],
            "frac_courses_instructor_ft": [0.5],
            "enrollment_intensity": ["PART-TIME"],
        }
    )
    col = data_audit_eda.infer_semester_enrollment_intensity_column(
        sem, exclude_cols={"student_id", "strm"}
    )
    assert col == "enrollment_intensity"


def test_infer_inst_tot_credits_typo_cumulative_column_is_earned():
    """Misspelled cumulative total without 'attempt' → earned, not attempted."""
    from edvise.data_audit import eda as data_audit_eda

    df = pd.DataFrame(
        {
            "student_id": [1, 2],
            "total_cumlative_credits": [90.0, 120.0],
        }
    )
    att, earn = data_audit_eda.infer_inst_tot_credits_columns(df)
    assert att is None
    assert earn == "total_cumlative_credits"


def test_infer_inst_tot_credits_attempt_vs_typo_cumulative():
    from edvise.data_audit import eda as data_audit_eda

    df = pd.DataFrame(
        {
            "student_id": [1, 2],
            "total_credits_attempted": [100.0, 130.0],
            "total_cumlative_credits": [90.0, 120.0],
        }
    )
    att, earn = data_audit_eda.infer_inst_tot_credits_columns(df)
    assert att == "total_credits_attempted"
    assert earn == "total_cumlative_credits"


def test_credit_column_name_has_attempt_marker():
    from edvise.data_audit import eda as data_audit_eda

    assert data_audit_eda.credit_column_name_has_attempt_marker("total_credits_attempted")
    assert data_audit_eda.credit_column_name_has_attempt_marker("sem_att_credits")
    assert not data_audit_eda.credit_column_name_has_attempt_marker("total_cumlative_credits")
    assert not data_audit_eda.credit_column_name_has_attempt_marker("matter_score")
