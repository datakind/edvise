import numbers
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

    def test_cohort_years_returns_sorted_list(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        expected = [
            str(y).replace("-", " - ")
            for y in sorted(sample_cohort_data["cohort"].dropna().unique().tolist())
        ]
        assert eda.cohort_years(formatted=True) == expected

    def test_summary_metrics_calculate_correctly(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        assert eda.total_students["value"] == sample_cohort_data["student_id"].nunique()
        assert eda.total_students["name"] == "Total Students"
        expected_transfer = int(
            (sample_cohort_data["enrollment_type"] == "TRANSFER-IN").sum()
        )
        assert eda.transfer_students["value"] == expected_transfer
        assert eda.transfer_students["name"] == "Transfer Students"
        assert isinstance(eda.avg_year1_gpa_all_students["value"], float)
        assert (
            eda.avg_year1_gpa_all_students["name"] == "Avg. Year 1 GPA - All Students"
        )

    def test_avg_year1_gpa_all_students_returns_zero_for_invalid_gpa(self):
        df = pd.DataFrame(
            {
                "student_id": ["student-1", "student-2"],
                "enrollment_type": ["FIRST-TIME", "TRANSFER-IN"],
                "gpa_group_year_1": ["invalid", None],
            }
        )
        eda = EdaSummary(df, validate=False)
        val = eda.avg_year1_gpa_all_students["value"]
        assert val == 0.0 or pd.isna(val)

    def test_gpa_by_enrollment_type_structure(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.gpa_by_enrollment_type
        assert "cohort_years" in result
        assert "series" in result
        expected_types = (
            sample_cohort_data["enrollment_type"].dropna().unique().tolist()
        )
        assert len(result["series"]) == len(expected_types)

    def test_gpa_by_enrollment_intensity_structure(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.gpa_by_enrollment_intensity
        assert "cohort_years" in result
        assert "series" in result
        # Check series names match the enrollment intensity values from the data
        series_names = [s["name"] for s in result["series"]]
        # Note: EdaSummary now uses .notna() filter instead of hardcoded "unknown" filter
        expected_intensities = [
            str(s).replace("-", " ")
            for s in sorted(
                sample_cohort_data["enrollment_intensity_first_term"]
                .dropna()
                .unique()
                .tolist()
            )
        ]
        assert len(result["series"]) == len(expected_intensities)
        assert set(series_names) == set(expected_intensities)
        # Check that data arrays match cohort_years length
        for series in result["series"]:
            assert len(series["data"]) == len(result["cohort_years"])

    def test_students_by_cohort_term_structure(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.students_by_cohort_term
        assert "years" in result
        assert "terms" in result
        assert isinstance(result["years"], list)
        assert isinstance(result["terms"], list)
        expected_years = eda.cohort_years(formatted=True)
        assert result["years"] == expected_years
        for t in result["terms"]:
            assert "key" in t and "label" in t and "data" in t
            assert isinstance(t["key"], str)
            assert isinstance(t["label"], str)
            assert isinstance(t["data"], list)
            assert len(t["data"]) == len(result["years"])
            assert all(isinstance(c, int) for c in t["data"])

    def test_course_enrollments_structure(self, sample_cohort_data, sample_course_data):
        eda = EdaSummary(sample_cohort_data, sample_course_data)
        result = eda.course_enrollments
        assert "years" in result
        assert "terms" in result
        assert result["years"] == eda.cohort_years(formatted=True)
        for t in result["terms"]:
            assert "key" in t and "label" in t and "data" in t
            assert len(t["data"]) == len(result["years"])
            assert all(isinstance(c, int) for c in t["data"])

    def test_course_enrollments_empty_when_no_course_data(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, df_course=None)
        assert eda.course_enrollments is None

    def test_degree_types_structure(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.degree_types
        assert isinstance(result, list)
        assert len(result) > 0
        # Check structure of each item
        for item in result:
            assert "count" in item
            assert "percentage" in item
            assert "name" in item
            assert isinstance(item["count"], int)
            assert isinstance(item["percentage"], float)
            assert isinstance(item["name"], str)

    def test_degree_types_calculates_correctly(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.degree_types
        expected_counts = sample_cohort_data[
            "credential_type_sought_year_1"
        ].value_counts()
        assert len(result) == len(expected_counts)
        # Check that percentages sum to approximately 100 (within rounding)
        total_percentage = sum(item["percentage"] for item in result)
        assert abs(total_percentage - 100.0) < 0.1
        # Check that counts match expected values
        degree_counts = {item["name"]: item["count"] for item in result}
        for degree_type, count in expected_counts.items():
            assert degree_counts[degree_type] == int(count)

    def test_enrollment_type_by_intensity_structure(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.enrollment_type_by_intensity
        assert "categories" in result
        assert "series" in result
        assert isinstance(result["categories"], list)
        assert isinstance(result["series"], list)
        # Check series structure
        for series in result["series"]:
            assert "name" in series
            assert "data" in series
            assert isinstance(series["data"], list)
            # Data length should match categories length
            assert len(series["data"]) == len(result["categories"])

    def test_enrollment_type_by_intensity_calculates_correctly(
        self, sample_cohort_data
    ):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.enrollment_type_by_intensity
        raw_categories = sorted(
            sample_cohort_data["enrollment_type"].dropna().unique().tolist()
        )
        normalized_names = {
            "FIRST-TIME": "First Time",
            "RE-ADMIT": "Re-admit",
            "TRANSFER-IN": "Transfer",
        }
        expected_categories = [
            normalized_names.get(str(c).lower(), str(c).replace("-", " ").strip())
            for c in raw_categories
        ]
        assert result["categories"] == expected_categories
        # Check series names match the enrollment intensity values from the data
        series_names = [s["name"] for s in result["series"]]
        expected_intensities = sorted(
            sample_cohort_data["enrollment_intensity_first_term"]
            .dropna()
            .unique()
            .tolist()
        )
        assert set(series_names) == set(expected_intensities)
        # Check data values are numeric counts (int, float, or None from _format_series_data)
        for series in result["series"]:
            assert all(
                c is None or isinstance(c, (int, float, numbers.Integral))
                for c in series["data"]
            )
        # Check that data matches expected counts
        for intensity in expected_intensities:
            expected_counts = (
                sample_cohort_data[
                    sample_cohort_data["enrollment_intensity_first_term"] == intensity
                ]
                .groupby("enrollment_type", observed=True)
                .size()
                .reindex(raw_categories, fill_value=0)
                .tolist()
            )
            series = next(s for s in result["series"] if s["name"] == intensity)
            assert series["data"] == expected_counts

    def test_pell_recipient_by_first_gen_structure(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.pell_recipient_by_first_gen
        if result == {}:
            return
        assert "categories" in result
        assert "series" in result
        assert isinstance(result["categories"], list)
        assert isinstance(result["series"], list)
        assert len(result["series"]) > 0
        for series in result["series"]:
            assert "name" in series
            assert "data" in series
            assert isinstance(series["data"], list)
            assert len(series["data"]) == len(result["categories"])

    def test_pell_recipient_by_first_gen_calculates_correctly(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.pell_recipient_by_first_gen
        if result == {}:
            return
        df = sample_cohort_data.copy()
        # Note: EdaSummary now uses .dropna() instead of filtering "UK"
        pell_categories = sorted(
            df.dropna(subset=["pell_status_first_year"])["pell_status_first_year"]
            .unique()
            .tolist()
        )
        assert result["categories"] == pell_categories
        for series in result["series"]:
            expected_counts = (
                df.dropna(subset=["pell_status_first_year"])
                .assign(first_gen=lambda d: d["first_gen"].fillna("N"))
                .loc[lambda d: d["first_gen"] == series["name"]]
                .groupby("pell_status_first_year")
                .size()
                .reindex(pell_categories, fill_value=0)
                .tolist()
            )
            assert series["data"] == expected_counts

    def test_pell_recipient_by_first_gen_fills_missing(self, sample_cohort_data):
        sample_cohort_data["first_gen"] = None
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.pell_recipient_by_first_gen
        assert result == {}

    def test_pell_recipient_status_structure(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.pell_recipient_status
        assert "series" in result
        assert isinstance(result["series"], list)
        assert len(result["series"]) == 1
        assert result["series"][0]["name"] == "All Students"
        assert isinstance(result["series"][0]["data"], dict)

    def test_pell_recipient_status_calculates_correctly(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.pell_recipient_status
        expected_counts = (
            sample_cohort_data.dropna(subset=["pell_status_first_year"])
            .groupby("pell_status_first_year", observed=True)
            .size()
            .to_dict()
        )
        assert result["series"][0]["data"] == expected_counts

    def test_student_age_by_gender_structure(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.student_age_by_gender
        assert "categories" in result
        assert "series" in result
        assert isinstance(result["categories"], list)
        assert isinstance(result["series"], list)
        assert len(result["series"]) > 0
        # Check series structure
        for series in result["series"]:
            assert "name" in series
            assert "data" in series
            assert isinstance(series["data"], list)
            # Data length should match categories length
            assert len(series["data"]) == len(result["categories"])

    def test_student_age_by_gender_calculates_correctly(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.student_age_by_gender
        # Check categories are sorted
        assert result["categories"] == sorted(result["categories"])
        # Check data values are numeric counts (int, float, or None from _format_series_data)
        for series in result["series"]:
            assert all(
                c is None or isinstance(c, (int, float, numbers.Integral))
                for c in series["data"]
            )
        # Note: EdaSummary now uses .dropna() instead of filtering "UK"
        # If schema validation hasn't been applied, "UK" values will remain
        gender_categories = sorted(
            sample_cohort_data["gender"].dropna().unique().tolist()
        )
        age_df = (
            sample_cohort_data[["gender", "student_age"]]
            .dropna()
            .value_counts()
            .unstack(fill_value=0)
        )
        assert result["categories"] == sorted(gender_categories)
        for series in result["series"]:
            expected_counts = (
                age_df.loc[:, series["name"]]
                .reindex(sorted(gender_categories), fill_value=0)
                .tolist()
            )
            assert series["data"] == expected_counts

    def test_race_by_pell_status_structure(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.race_by_pell_status
        assert "categories" in result
        assert "series" in result
        assert isinstance(result["categories"], list)
        assert isinstance(result["series"], list)
        assert len(result["series"]) > 0
        # Check series structure
        for series in result["series"]:
            assert "name" in series
            assert "data" in series
            assert isinstance(series["data"], list)
            # Data length should match categories length
            assert len(series["data"]) == len(result["categories"])
        # Check Pell status names are normalized
        series_names = [s["name"] for s in result["series"]]
        assert all(name in ["Yes", "No"] for name in series_names)

    def test_race_by_pell_status_calculates_correctly(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.race_by_pell_status
        pell_map = {"Y": "Yes", "N": "No", "y": "Yes", "n": "No"}
        race_df = (
            sample_cohort_data[["race", "pell_status_first_year"]]
            .dropna()
            .assign(
                pell_status_first_year=lambda d: d["pell_status_first_year"]
                .astype(str)
                .replace(pell_map)
            )
        )
        race_df = race_df[race_df["pell_status_first_year"].isin(["Yes", "No"])]
        counts_df = (
            race_df.groupby(["pell_status_first_year", "race"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        expected_order = counts_df.columns.tolist()
        assert result["categories"] == expected_order
        # Check data values are numeric counts (int, float, or None from _format_series_data)
        for series in result["series"]:
            assert all(
                c is None or isinstance(c, (int, float, numbers.Integral))
                for c in series["data"]
            )
        for series in result["series"]:
            expected_counts = (
                counts_df.loc[series["name"]]
                .reindex(expected_order, fill_value=0)
                .tolist()
            )
            assert series["data"] == expected_counts

    def test_gpa_by_enrollment_intensity_handles_nulls(self, sample_cohort_data):
        """Test that NaN enrollment_intensity values are properly excluded."""
        sample_cohort_data.loc[0:5, "enrollment_intensity_first_term"] = pd.NA
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.gpa_by_enrollment_intensity
        assert "series" in result
        series_names = [s["name"] for s in result["series"]]
        assert all(pd.notna(name) for name in series_names)

    def test_pell_recipient_status_handles_nulls(self, sample_cohort_data):
        """Test that NaN pell status values are properly excluded."""
        sample_cohort_data.loc[0:2, "pell_status_first_year"] = pd.NA
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.pell_recipient_status
        assert "series" in result
        data_keys = result["series"][0]["data"].keys()
        assert all(pd.notna(k) for k in data_keys)

    def test_student_age_by_gender_handles_nulls(self, sample_cohort_data):
        """Test that NaN gender values are properly excluded."""
        sample_cohort_data.loc[0:5, "gender"] = pd.NA
        eda = EdaSummary(sample_cohort_data, validate=True)
        result = eda.student_age_by_gender
        assert "categories" in result
        assert all(pd.notna(cat) for cat in result["categories"])

    def test_validate_false_preserves_data(self, sample_cohort_data):
        """Test that validate=False skips validation and preserves data as-is."""
        eda_no_validate = EdaSummary(sample_cohort_data, validate=False)
        assert eda_no_validate.total_students["value"] > 0

    def test_cached_property_only_computes_once(self, sample_cohort_data):
        eda = EdaSummary(sample_cohort_data, validate=True)
        # Access multiple times - should only compute once
        first = eda.gpa_by_enrollment_type
        second = eda.gpa_by_enrollment_type
        # Cached properties return the same object reference
        assert first is second
        assert first == second  # Values should also be equal
