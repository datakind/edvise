import pandas as pd
import pytest
from unittest.mock import patch
from collections.abc import Iterable

from edvise.utils import data_cleaning


@pytest.mark.parametrize(
    ["eles", "exp"],
    [
        ([2, 1, 2, 2, 1, 3], [2, 1, 3]),
        (("a", "c", "b", "b", "c", "a"), ["a", "c", "b"]),
    ],
)
def test_unique_elements_in_order(eles, exp):
    obs = data_cleaning.unique_elements_in_order(eles)
    assert isinstance(obs, Iterable)
    assert list(obs) == exp


@pytest.mark.parametrize(
    ["val", "exp"],
    [
        ("Student GUID", "student_guid"),
        ("Credential Type Sought Year 1", "credential_type_sought_year_1"),
        ("Years to Bachelors at cohort inst.", "years_to_bachelors_at_cohort_inst"),
        ("Enrolled at Other Institution(s)", "enrolled_at_other_institution_s"),
    ],
)
def test_convert_to_snake_case(val, exp):
    obs = data_cleaning.convert_to_snake_case(val)
    assert obs == exp


class TestInferStudentIdCol:
    """Tests for _infer_student_id_col function."""

    @pytest.mark.parametrize(
        "columns, expected",
        [
            (["student_guid", "name", "age"], "student_guid"),
            (["study_id", "name", "age"], "study_id"),
            (["student_id", "name", "age"], "student_id"),
            (["name", "age"], "student_id"),  # default fallback
            (["study_id", "student_guid", "student_id"], "student_guid"),  # priority
        ],
    )
    def test_infer_student_id_col(self, columns, expected):
        df = pd.DataFrame({col: [] for col in columns})
        result = data_cleaning._infer_student_id_col(df)
        assert result == expected


class TestInferCreditsCol:
    """Tests for _infer_credits_col function."""

    @pytest.mark.parametrize(
        "columns, expected",
        [
            (["course_credits", "name"], "course_credits"),
            (["number_of_credits_attempted", "name"], "number_of_credits_attempted"),
            (["name", "age"], None),
            (
                ["course_credits", "number_of_credits_attempted"],
                "course_credits",
            ),  # priority
        ],
    )
    def test_infer_credits_col(self, columns, expected):
        df = pd.DataFrame({col: [] for col in columns})
        result = data_cleaning._infer_credits_col(df)
        assert result == expected


class TestIsLabLectureCombo:
    """Tests for _is_lab_lecture_combo function."""

    @pytest.mark.parametrize(
        "values, expected",
        [
            (["Lab", "Lecture"], True),
            (["lab", "lecture"], True),
            (["LAB", "LECTURE"], True),
            (["Lab", "Lecture", "Lab"], True),
            (["Lab"], False),
            (["Lecture"], False),
            (["Lab", "Lab"], False),
            (["Lecture", "Lecture"], False),
            (["Lab", "Other"], False),
            (["Lecture", "Other"], False),
            (["Other", "Another"], False),
            (["lab", "LECTURE", "Lab"], True),  # mixed case
        ],
    )
    def test_is_lab_lecture_combo(self, values, expected):
        series = pd.Series(values)
        result = data_cleaning._is_lab_lecture_combo(series)
        assert result == expected

    def test_is_lab_lecture_combo_with_nulls(self):
        series = pd.Series(["Lab", None, "Lecture", None])
        result = data_cleaning._is_lab_lecture_combo(series)
        assert result is True

    def test_is_lab_lecture_combo_all_nulls(self):
        series = pd.Series([None, None])
        result = data_cleaning._is_lab_lecture_combo(series)
        assert result is False


class TestFindPdpRowsToRenumber:
    """Tests for _find_pdp_rows_to_renumber function."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "student_id": ["A", "A", "A", "A", "B", "B"],
                "academic_year": ["2024"] * 6,
                "academic_term": ["FALL"] * 6,
                "course_prefix": ["MATH", "MATH", "PHYS", "PHYS", "ENGL", "ENGL"],
                "course_number": ["101", "101", "201", "201", "102", "102"],
                "section_id": ["001", "001", "002", "002", "003", "003"],
                "course_name": [
                    "Calculus I",
                    "Calculus II",
                    "Physics",
                    "Physics",
                    "English",
                    "English",
                ],
            }
        )

    def test_find_rows_with_different_names(self, sample_df):
        unique_cols = [
            "student_id",
            "academic_year",
            "academic_term",
            "course_prefix",
            "course_number",
            "section_id",
        ]
        dup_mask = sample_df.duplicated(unique_cols, keep=False)
        result = data_cleaning._find_pdp_rows_to_renumber(
            sample_df, dup_mask, unique_cols
        )
        # Should return indices 0, 1 (different course names for MATH 101)
        assert set(result) == {0, 1}

    def test_find_rows_with_same_names(self):
        df = pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "academic_year": ["2024", "2024"],
                "academic_term": ["FALL", "FALL"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "section_id": ["001", "001"],
                "course_name": ["Calculus I", "Calculus I"],
            }
        )
        unique_cols = [
            "student_id",
            "academic_year",
            "academic_term",
            "course_prefix",
            "course_number",
            "section_id",
        ]
        dup_mask = df.duplicated(unique_cols, keep=False)
        result = data_cleaning._find_pdp_rows_to_renumber(df, dup_mask, unique_cols)
        # Should return empty list (same names)
        assert result == []


class TestLogPdpDuplicateDrop:
    """Tests for _log_pdp_duplicate_drop function."""

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_logs_warning_for_small_percentage(self, mock_logger):
        df = pd.DataFrame({"col": range(10000)})
        dup_mask = pd.Series([True, True] + [False] * 9998)
        data_cleaning._log_pdp_duplicate_drop(df, dup_mask)
        assert mock_logger.warning.called

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_logs_warning_for_large_percentage(self, mock_logger):
        df = pd.DataFrame({"col": range(100)})
        dup_mask = pd.Series([True] * 20 + [False] * 80)
        data_cleaning._log_pdp_duplicate_drop(df, dup_mask)
        assert mock_logger.warning.called


class TestClassifyDuplicateGroups:
    """Tests for _classify_duplicate_groups function."""

    @pytest.fixture
    def sample_duplicate_rows(self):
        return pd.DataFrame(
            {
                "student_id": ["A", "A", "B", "B", "C", "C"],
                "term": ["F2024"] * 6,
                "course_subject": ["MATH", "MATH", "PHYS", "PHYS", "ENGL", "ENGL"],
                "course_number": ["101", "101", "201", "201", "102", "102"],
                "course_type": [
                    "Lab",
                    "Lecture",
                    "Lecture",
                    "Lecture",
                    "Lab",
                    "Lab",
                ],
                "course_name": [
                    "Math Lab",
                    "Math Lecture",
                    "Physics",
                    "Physics",
                    "English",
                    "English",
                ],
                "course_credits": [1.0, 3.0, 3.0, 3.0, 2.0, 1.0],
            }
        )

    def test_classify_with_varying_types(self, sample_duplicate_rows):
        unique_cols = ["student_id", "term", "course_subject", "course_number"]
        result = data_cleaning._classify_duplicate_groups(
            sample_duplicate_rows,
            unique_cols,
            has_course_type=True,
            has_course_name=True,
            credits_col="course_credits",
        )
        renumber_idx, drop_idx, renumber_groups, drop_groups, lab_lecture_rows = result

        # MATH 101 should be renumbered (different types and names)
        # PHYS 201 should be dropped (same type and name)
        # ENGL 102 should be dropped (same type and name)
        assert renumber_groups == 1
        assert drop_groups == 2
        assert len(renumber_idx) == 2  # MATH rows
        assert len(drop_idx) == 2  # one from PHYS, one from ENGL

    def test_classify_without_course_type(self):
        df = pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "term": ["F2024", "F2024"],
                "course_subject": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "course_name": ["Calculus I", "Calculus II"],
                "course_credits": [3.0, 3.0],
            }
        )
        unique_cols = ["student_id", "term", "course_subject", "course_number"]
        result = data_cleaning._classify_duplicate_groups(
            df,
            unique_cols,
            has_course_type=False,
            has_course_name=True,
            credits_col="course_credits",
        )
        renumber_idx, drop_idx, renumber_groups, drop_groups, lab_lecture_rows = result

        # Should renumber due to different names
        assert renumber_groups == 1
        assert drop_groups == 0


class TestDropTrueDuplicateRows:
    """Tests for _drop_true_duplicate_rows function."""

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_drops_specified_rows(self, mock_logger):
        df = pd.DataFrame({"col": range(10)})
        drop_idx = [0, 2, 4]
        result = data_cleaning._drop_true_duplicate_rows(df, drop_idx)
        assert len(result) == 7
        assert 0 not in result.index
        assert 2 not in result.index
        assert 4 not in result.index

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_no_drops_when_empty_list(self, mock_logger):
        df = pd.DataFrame({"col": range(10)})
        result = data_cleaning._drop_true_duplicate_rows(df, [])
        assert len(result) == 10
        assert not mock_logger.warning.called


class TestRenumberDuplicates:
    """Tests for _renumber_duplicates function."""

    @patch("edvise.utils.data_cleaning.dedupe_by_renumbering_courses")
    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_renumbers_courses(self, _mock_logger, mock_dedupe):  # noqa: ARG002
        df = pd.DataFrame(
            {
                "course_subject": ["MATH", "MATH", "PHYS"],
                "course_number": ["101", "101", "201"],
                "course_type": ["Lab", "Lecture", "Lecture"],
            }
        )
        # Mock dedupe function to return modified course numbers
        mock_dedupe_result = df.copy()
        mock_dedupe_result.loc[0, "course_number"] = "101-1"
        mock_dedupe_result.loc[1, "course_number"] = "101-2"
        mock_dedupe.return_value = mock_dedupe_result

        data_cleaning._renumber_duplicates(
            df,
            renumber_work_idx=[0, 1],
            has_course_type=True,
            has_course_name=False,
            credits_col=None,
        )

        # Check that dedupe function was called
        assert mock_dedupe.called

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_returns_unchanged_when_no_idx(self, _mock_logger):  # noqa: ARG002
        df = pd.DataFrame({"course_subject": ["MATH"], "course_number": ["101"]})
        result = data_cleaning._renumber_duplicates(
            df,
            renumber_work_idx=[],
            has_course_type=False,
            has_course_name=False,
            credits_col=None,
        )
        assert result.equals(df)


class TestHandlePdpDuplicates:
    """Integration tests for _handle_pdp_duplicates function."""

    @pytest.fixture
    def pdp_df_with_different_names(self):
        return pd.DataFrame(
            {
                "student_guid": ["A", "A"],
                "academic_year": ["2024", "2024"],
                "academic_term": ["FALL", "FALL"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "section_id": ["001", "001"],
                "course_name": ["Calculus I", "Calculus II"],
                "number_of_credits_attempted": [3.0, 3.0],
            }
        )

    @pytest.fixture
    def pdp_df_with_same_names(self):
        return pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "academic_year": ["2024", "2024"],
                "academic_term": ["FALL", "FALL"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "section_id": ["001", "001"],
                "course_name": ["Calculus I", "Calculus I"],
                "number_of_credits_attempted": [3.0, 4.0],
            }
        )

    @patch("edvise.utils.data_cleaning.LOGGER")
    @patch("edvise.utils.data_cleaning.dedupe_by_renumbering_courses")
    def test_renumbers_when_names_differ(
        self, mock_dedupe, mock_logger, pdp_df_with_different_names
    ):
        mock_dedupe.return_value = pdp_df_with_different_names.copy()
        result = data_cleaning._handle_pdp_duplicates(pdp_df_with_different_names)
        assert mock_dedupe.called

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_drops_when_names_same(self, mock_logger, pdp_df_with_same_names):
        result = data_cleaning._handle_pdp_duplicates(pdp_df_with_same_names)
        assert len(result) == 1  # One duplicate dropped


class TestHandleSchemaDuplicates:
    """Integration tests for _handle_schema_duplicates function."""

    @pytest.fixture
    def schema_df_with_type_variation(self):
        return pd.DataFrame(
            {
                "student_id": ["A", "A", "B"],
                "term": ["F2024", "F2024", "F2024"],
                "course_subject": ["MATH", "MATH", "PHYS"],
                "course_number": ["101", "101", "201"],
                "course_type": ["Lab", "Lecture", "Lecture"],
                "course_name": ["Math Lab", "Math Lecture", "Physics"],
                "course_credits": [1.0, 3.0, 3.0],
            }
        )

    @pytest.fixture
    def schema_df_true_duplicates(self):
        return pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "term": ["F2024", "F2024"],
                "course_subject": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "course_type": ["Lecture", "Lecture"],
                "course_name": ["Calculus I", "Calculus I"],
                "course_credits": [3.0, 4.0],
            }
        )

    @patch("edvise.utils.data_cleaning.LOGGER")
    @patch("edvise.utils.data_cleaning.dedupe_by_renumbering_courses")
    def test_renumbers_with_type_variation(
        self, mock_dedupe, mock_logger, schema_df_with_type_variation
    ):
        mock_result = schema_df_with_type_variation.copy()
        mock_result.loc[0, "course_number"] = "101-1"
        mock_result.loc[1, "course_number"] = "101-2"
        mock_dedupe.return_value = mock_result

        result = data_cleaning._handle_schema_duplicates(schema_df_with_type_variation)
        assert "course_id" in result.columns
        assert mock_dedupe.called

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_drops_true_duplicates(self, mock_logger, schema_df_true_duplicates):
        result = data_cleaning._handle_schema_duplicates(schema_df_true_duplicates)
        assert len(result) == 1  # One duplicate dropped
        # Should keep the one with higher credits (4.0)
        assert result.iloc[0]["course_credits"] == 4.0


class TestHandlingDuplicates:
    """Integration tests for the main handling_duplicates function."""

    @pytest.fixture
    def pdp_sample_df(self):
        return pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "academic_year": ["2024", "2024"],
                "academic_term": ["FALL", "FALL"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "section_id": ["001", "001"],
                "course_name": ["Calculus I", "Calculus I"],
                "number_of_credits_attempted": [3.0, 3.0],
            }
        )

    @pytest.fixture
    def schema_sample_df(self):
        return pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "term": ["F2024", "F2024"],
                "course_subject": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "course_type": ["Lecture", "Lecture"],
                "course_credits": [3.0, 3.0],
            }
        )

    def test_raises_error_for_invalid_school_type(self, pdp_sample_df):
        with pytest.raises(ValueError, match="school_type must be either"):
            data_cleaning.handling_duplicates(pdp_sample_df, "invalid")

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_calls_pdp_handler_for_pdp_mode(self, mock_logger, pdp_sample_df):
        result = data_cleaning.handling_duplicates(pdp_sample_df, "pdp")
        assert isinstance(result, pd.DataFrame)

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_calls_schema_handler_for_schema_mode(self, mock_logger, schema_sample_df):
        result = data_cleaning.handling_duplicates(schema_sample_df, "schema")
        assert isinstance(result, pd.DataFrame)
        assert "course_id" in result.columns

    def test_handles_whitespace_in_school_type(self, pdp_sample_df):
        with patch("edvise.utils.data_cleaning.LOGGER"):
            result1 = data_cleaning.handling_duplicates(pdp_sample_df, "  pdp  ")
            result2 = data_cleaning.handling_duplicates(pdp_sample_df, "PDP")
            assert isinstance(result1, pd.DataFrame)
            assert isinstance(result2, pd.DataFrame)
