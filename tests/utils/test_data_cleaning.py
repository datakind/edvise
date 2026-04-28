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


# TestInferCreditsCol removed - credits column is now auto-detected in _handle_schema_duplicates


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


_PDP_DUP_UNIQUE_COLS = [
    "student_id",
    "academic_year",
    "academic_term",
    "course_prefix",
    "course_number",
    "section_id",
]


class TestClassifyDuplicateGroupsPdpStyle:
    """PDP-style keys and columns through _classify_duplicate_groups."""

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

    def test_renumbers_when_names_differ(self, sample_df):
        unique_cols = _PDP_DUP_UNIQUE_COLS
        dup_rows = sample_df[sample_df.duplicated(unique_cols, keep=False)]
        renumber_idx, drop_idx, rg, dg, _ = data_cleaning._classify_duplicate_groups(
            dup_rows,
            unique_cols,
            course_type_col=None,
            course_name_col="course_name",
            credits_col=None,
        )
        assert rg == 1
        assert dg == 2
        assert set(renumber_idx) == {0, 1}
        assert len(drop_idx) == 2

    def test_drops_when_same_name_no_credits_or_grade(self):
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
        unique_cols = _PDP_DUP_UNIQUE_COLS
        dup_rows = df[df.duplicated(unique_cols, keep=False)]
        renumber_idx, drop_idx, rg, dg, _ = data_cleaning._classify_duplicate_groups(
            dup_rows,
            unique_cols,
            course_type_col=None,
            course_name_col="course_name",
            credits_col=None,
        )
        assert rg == 0
        assert dg == 1
        assert renumber_idx == []
        assert len(drop_idx) == 1

    def test_renumbers_when_same_name_different_credits(self):
        df = pd.DataFrame(
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
        unique_cols = _PDP_DUP_UNIQUE_COLS
        dup_rows = df[df.duplicated(unique_cols, keep=False)]
        renumber_idx, drop_idx, rg, dg, _ = data_cleaning._classify_duplicate_groups(
            dup_rows,
            unique_cols,
            course_type_col=None,
            course_name_col="course_name",
            credits_col="number_of_credits_attempted",
        )
        assert rg == 1
        assert dg == 0
        assert set(renumber_idx) == {0, 1}
        assert drop_idx == []


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
                "academic_term": ["F2024"] * 6,
                "course_prefix": ["MATH", "MATH", "PHYS", "PHYS", "ENGL", "ENGL"],
                "course_number": ["101", "101", "201", "201", "102", "102"],
                "course_classification": [
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
                "course_credits_attempted": [1.0, 3.0, 3.0, 3.0, 2.0, 1.0],
            }
        )

    def test_classify_with_varying_types(self, sample_duplicate_rows):
        unique_cols = ["student_id", "academic_term", "course_prefix", "course_number"]
        result = data_cleaning._classify_duplicate_groups(
            sample_duplicate_rows,
            unique_cols,
            course_type_col="course_classification",
            course_name_col="course_name",
            credits_col="course_credits_attempted",
        )
        renumber_idx, drop_idx, renumber_groups, drop_groups, lab_lecture_rows = result

        # MATH 101: type+name vary -> renumber
        # ENGL 102: same type+name but different credits -> renumber
        # PHYS 201: true duplicate key -> drop one
        assert renumber_groups == 2
        assert drop_groups == 1
        assert len(renumber_idx) == 4
        assert len(drop_idx) == 1
        assert lab_lecture_rows == 2

    def test_classify_same_name_different_grades_renumbers(self):
        df = pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "academic_term": ["F2024", "F2024"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "course_classification": ["Lecture", "Lecture"],
                "course_name": ["Calculus I", "Calculus I"],
                "course_credits_attempted": [3.0, 3.0],
                "grade": ["C", "A"],
            }
        )
        unique_cols = ["student_id", "academic_term", "course_prefix", "course_number"]
        result = data_cleaning._classify_duplicate_groups(
            df,
            unique_cols,
            course_type_col="course_classification",
            course_name_col="course_name",
            credits_col="course_credits_attempted",
            grade_col="grade",
        )
        renumber_idx, drop_idx, renumber_groups, drop_groups, _ = result
        assert renumber_groups == 1
        assert drop_groups == 0
        assert len(renumber_idx) == 2
        assert drop_idx == []

    def test_classify_drops_when_only_non_material_columns_differ(self):
        """Classification, name, credits, and grade match; another column differs."""
        df = pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "academic_term": ["F2024", "F2024"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "course_classification": ["Lecture", "Lecture"],
                "course_name": ["Calculus I", "Calculus I"],
                "course_credits_attempted": [3.0, 3.0],
                "grade": ["B", "B"],
                "delivery_method": ["F", "O"],
            }
        )
        unique_cols = ["student_id", "academic_term", "course_prefix", "course_number"]
        _, drop_idx, renumber_groups, drop_groups, _ = (
            data_cleaning._classify_duplicate_groups(
                df,
                unique_cols,
                course_type_col="course_classification",
                course_name_col="course_name",
                credits_col="course_credits_attempted",
                grade_col="grade",
            )
        )
        assert renumber_groups == 0
        assert drop_groups == 1
        assert len(drop_idx) == 1

    def test_classify_without_course_type(self):
        df = pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "academic_term": ["F2024", "F2024"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "course_name": ["Calculus I", "Calculus II"],
                "course_credits_attempted": [3.0, 3.0],
            }
        )
        unique_cols = ["student_id", "academic_term", "course_prefix", "course_number"]
        result = data_cleaning._classify_duplicate_groups(
            df,
            unique_cols,
            course_type_col=None,
            course_name_col="course_name",
            credits_col="course_credits_attempted",
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
                "course_prefix": ["MATH", "MATH", "PHYS"],
                "course_number": ["101", "101", "201"],
                "course_classification": ["Lab", "Lecture", "Lecture"],
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
            unique_cols=None,
            credits_col=None,
            course_type_col="course_classification",
            course_name_col=None,
        )

        # Check that dedupe function was called
        assert mock_dedupe.called

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_returns_unchanged_when_no_idx(self, _mock_logger):  # noqa: ARG002
        df = pd.DataFrame({"course_prefix": ["MATH"], "course_number": ["101"]})
        result = data_cleaning._renumber_duplicates(
            df,
            renumber_work_idx=[],
            unique_cols=None,
            credits_col=None,
            course_type_col=None,
            course_name_col=None,
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
    @patch("edvise.utils.data_cleaning.dedupe_by_renumbering_courses")
    def test_renumbers_when_names_same_but_credits_differ(
        self, mock_dedupe, mock_logger, pdp_df_with_same_names
    ):
        mock_dedupe.return_value = pdp_df_with_same_names.copy()
        result = data_cleaning._handle_pdp_duplicates(pdp_df_with_same_names)
        assert mock_dedupe.called
        assert len(result) == 2

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_drops_when_names_same_and_credits_match(
        self, mock_logger, pdp_df_with_same_names
    ):
        df = pdp_df_with_same_names.copy()
        df["number_of_credits_attempted"] = [3.0, 3.0]
        result = data_cleaning._handle_pdp_duplicates(df)
        assert len(result) == 1


class TestHandleSchemaDuplicates:
    """Integration tests for _handle_schema_duplicates function."""

    @pytest.fixture
    def schema_df_with_type_variation(self):
        return pd.DataFrame(
            {
                "student_id": ["A", "A", "B"],
                "academic_term": ["F2024", "F2024", "F2024"],
                "course_prefix": ["MATH", "MATH", "PHYS"],
                "course_number": ["101", "101", "201"],
                "course_classification": ["Lab", "Lecture", "Lecture"],
                "course_name": ["Math Lab", "Math Lecture", "Physics"],
                "course_credits_attempted": [1.0, 3.0, 3.0],
            }
        )

    @pytest.fixture
    def schema_df_true_duplicates(self):
        return pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "academic_term": ["F2024", "F2024"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "course_classification": ["Lecture", "Lecture"],
                "course_name": ["Calculus I", "Calculus I"],
                "course_credits_attempted": [3.0, 4.0],
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
    @patch("edvise.utils.data_cleaning.dedupe_by_renumbering_courses")
    def test_renumbers_when_same_title_but_credits_differ(
        self, mock_dedupe, mock_logger, schema_df_true_duplicates
    ):
        mock_dedupe.return_value = schema_df_true_duplicates.copy()
        result = data_cleaning._handle_schema_duplicates(schema_df_true_duplicates)
        assert mock_dedupe.called
        assert len(result) == 2

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_drops_identical_true_duplicates(
        self, mock_logger, schema_df_true_duplicates
    ):
        df = schema_df_true_duplicates.copy()
        df.loc[1, "course_credits_attempted"] = 3.0
        result = data_cleaning._handle_schema_duplicates(df)
        assert len(result) == 1
        assert result.iloc[0]["course_credits_attempted"] == 3.0

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_drops_when_only_extraneous_column_differs(
        self, mock_logger, schema_df_true_duplicates
    ):
        df = schema_df_true_duplicates.copy()
        df.loc[1, "course_credits_attempted"] = 3.0
        df["delivery_method"] = ["F", "O"]
        result = data_cleaning._handle_schema_duplicates(df)
        assert len(result) == 1


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
                "academic_term": ["F2024", "F2024"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "course_classification": ["Lecture", "Lecture"],
                "course_name": ["Calculus I", "Calculus I"],
                "course_credits_attempted": [3.0, 3.0],
            }
        )

    def test_raises_error_for_invalid_schema_type(self, pdp_sample_df):
        with pytest.raises(ValueError, match="schema_type must be either"):
            data_cleaning.handling_duplicates(pdp_sample_df, "invalid")

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_calls_pdp_handler_for_pdp_mode(self, mock_logger, pdp_sample_df):
        result = data_cleaning.handling_duplicates(pdp_sample_df, "pdp")
        assert isinstance(result, pd.DataFrame)

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_calls_schema_handler_for_schema_mode(self, mock_logger, schema_sample_df):
        result = data_cleaning.handling_duplicates(schema_sample_df, "es")
        assert isinstance(result, pd.DataFrame)
        assert "course_id" in result.columns

    def test_handles_whitespace_in_schema_type(self, pdp_sample_df):
        with patch("edvise.utils.data_cleaning.LOGGER"):
            result1 = data_cleaning.handling_duplicates(pdp_sample_df, "  pdp  ")
            result2 = data_cleaning.handling_duplicates(pdp_sample_df, "PDP")
            assert isinstance(result1, pd.DataFrame)
            assert isinstance(result2, pd.DataFrame)
