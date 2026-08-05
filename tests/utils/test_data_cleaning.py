import pandas as pd
import pytest
from unittest.mock import patch
from collections.abc import Iterable

from edvise.utils import data_cleaning

_PDP_DUP_UNIQUE_COLS = [
    "student_id",
    "academic_year",
    "academic_term",
    "course_prefix",
    "course_number",
    "section_id",
]


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
    @pytest.mark.parametrize(
        "columns, expected",
        [
            (["student_guid", "name", "age"], "student_guid"),
            (["study_id", "name", "age"], "study_id"),
            (["student_id", "name", "age"], "student_id"),
            (["name", "age"], "student_id"),
            (["study_id", "student_guid", "student_id"], "student_guid"),
        ],
    )
    def test_infer_student_id_col(self, columns, expected):
        df = pd.DataFrame({col: [] for col in columns})
        assert data_cleaning._infer_student_id_col(df) == expected


class TestOmitSectionFromDupKey:
    def test_keeps_section_when_null_fraction_at_or_below_threshold(self):
        df = pd.DataFrame({"section_id": [pd.NA, pd.NA, pd.NA, "001"]})
        cols = ["student_id", "section_id"]
        assert data_cleaning._omit_section_from_dup_key_if_unusable(df, cols) == cols

    def test_omits_section_when_null_fraction_exceeds_threshold(self):
        df = pd.DataFrame({"section_id": [pd.NA, pd.NA, pd.NA, pd.NA, "001"]})
        cols = ["student_id", "section_id"]
        assert data_cleaning._omit_section_from_dup_key_if_unusable(df, cols) == [
            "student_id"
        ]


class TestClassifyDuplicateGroups:
    def test_suffixes_when_names_differ(self):
        df = pd.DataFrame(
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
        dup_rows = df[df.duplicated(_PDP_DUP_UNIQUE_COLS, keep=False)]
        suffix_idx, drop_idx, rg, dg = data_cleaning._classify_duplicate_groups(
            dup_rows,
            _PDP_DUP_UNIQUE_COLS,
            course_type_col=None,
            course_name_col="course_name",
            credits_col=None,
        )
        assert rg == 1
        assert dg == 2
        assert set(suffix_idx) == {0, 1}
        assert len(drop_idx) == 2

    def test_suffixes_when_same_name_different_credits(self):
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
        suffix_idx, drop_idx, rg, dg = data_cleaning._classify_duplicate_groups(
            df[df.duplicated(_PDP_DUP_UNIQUE_COLS, keep=False)],
            _PDP_DUP_UNIQUE_COLS,
            course_type_col=None,
            course_name_col="course_name",
            credits_col="number_of_credits_attempted",
        )
        assert rg == 1
        assert dg == 0
        assert set(suffix_idx) == {0, 1}
        assert drop_idx == []

    def test_suffixes_when_grades_differ(self):
        df = pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "academic_year": ["2024", "2024"],
                "academic_term": ["FALL", "FALL"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "section_id": ["001", "001"],
                "course_name": ["Calculus I", "Calculus I"],
                "number_of_credits_attempted": [3.0, 3.0],
                "grade": ["C", "A"],
            }
        )
        suffix_idx, drop_idx, rg, dg = data_cleaning._classify_duplicate_groups(
            df,
            _PDP_DUP_UNIQUE_COLS,
            course_type_col=None,
            course_name_col="course_name",
            credits_col="number_of_credits_attempted",
            grade_col="grade",
        )
        assert rg == 1
        assert dg == 0
        assert len(suffix_idx) == 2
        assert drop_idx == []

    def test_drops_when_only_non_material_columns_differ(self):
        df = pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "academic_year": ["2024", "2024"],
                "academic_term": ["FALL", "FALL"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "section_id": ["001", "001"],
                "course_name": ["Calculus I", "Calculus I"],
                "number_of_credits_attempted": [3.0, 3.0],
                "grade": ["B", "B"],
                "delivery_method": ["F", "O"],
            }
        )
        _, drop_idx, rg, dg = data_cleaning._classify_duplicate_groups(
            df,
            _PDP_DUP_UNIQUE_COLS,
            course_type_col=None,
            course_name_col="course_name",
            credits_col="number_of_credits_attempted",
            grade_col="grade",
        )
        assert rg == 0
        assert dg == 1
        assert len(drop_idx) == 1


class TestDropTrueDuplicateRows:
    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_drops_specified_rows(self, mock_logger):
        df = pd.DataFrame({"col": range(10)})
        result = data_cleaning._drop_true_duplicate_rows(df, [0, 2, 4])
        assert len(result) == 7
        assert 0 not in result.index

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_no_drops_when_empty_list(self, mock_logger):
        df = pd.DataFrame({"col": range(10)})
        result = data_cleaning._drop_true_duplicate_rows(df, [])
        assert len(result) == 10
        assert not mock_logger.warning.called


class TestSuffixDuplicates:
    @patch("edvise.utils.data_cleaning.dedupe_by_suffixing_courses")
    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_suffixes_courses(self, _mock_logger, mock_dedupe):
        df = pd.DataFrame(
            {
                "course_prefix": ["MATH", "MATH", "PHYS"],
                "course_number": ["101", "101", "201"],
                "course_type": ["Lab", "Lecture", "Lecture"],
            }
        )
        mock_result = df.copy()
        mock_result.loc[0, "course_number"] = "101-1"
        mock_result.loc[1, "course_number"] = "101-2"
        mock_dedupe.return_value = mock_result

        data_cleaning._suffix_duplicates(
            df,
            suffix_work_idx=[0, 1],
            unique_cols=["course_prefix", "course_number"],
            credits_col=None,
            course_type_col="course_type",
            course_name_col=None,
        )
        assert mock_dedupe.called

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_returns_unchanged_when_no_idx(self, _mock_logger):
        df = pd.DataFrame({"course_prefix": ["MATH"], "course_number": ["101"]})
        result = data_cleaning._suffix_duplicates(
            df,
            suffix_work_idx=[],
            unique_cols=["course_prefix", "course_number"],
            credits_col=None,
            course_type_col=None,
            course_name_col=None,
        )
        assert result.equals(df)


class TestHandlePdpDuplicates:
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
    @patch("edvise.utils.data_cleaning.dedupe_by_suffixing_courses")
    def test_suffixes_when_names_differ(
        self, mock_dedupe, mock_logger, pdp_df_with_different_names
    ):
        mock_dedupe.return_value = pdp_df_with_different_names.copy()
        data_cleaning._handle_pdp_duplicates(pdp_df_with_different_names)
        assert mock_dedupe.called

    @patch("edvise.utils.data_cleaning.LOGGER")
    @patch("edvise.utils.data_cleaning.dedupe_by_suffixing_courses")
    def test_suffixes_when_names_same_but_credits_differ(
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
        assert len(data_cleaning._handle_pdp_duplicates(df)) == 1

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_different_sections_not_key_duplicates(self, mock_logger):
        df = pd.DataFrame(
            {
                "student_id": ["A", "A"],
                "academic_year": ["2024", "2024"],
                "academic_term": ["FALL", "FALL"],
                "course_prefix": ["MATH", "MATH"],
                "course_number": ["101", "101"],
                "section_id": ["001", "002"],
                "course_name": ["Calculus I", "Calculus I"],
                "number_of_credits_attempted": [3.0, 3.0],
            }
        )
        assert len(data_cleaning._handle_pdp_duplicates(df)) == 2

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_omits_mostly_null_section_from_pdp_key(self, mock_logger):
        df = pd.DataFrame(
            {
                "student_id": ["A"] * 5,
                "academic_year": ["2024"] * 5,
                "academic_term": ["FALL"] * 5,
                "course_prefix": ["MATH"] * 5,
                "course_number": ["101"] * 5,
                "section_id": [pd.NA, pd.NA, pd.NA, pd.NA, "001"],
                "course_name": ["Calculus I"] * 5,
                "number_of_credits_attempted": [3.0] * 5,
            }
        )
        assert len(data_cleaning._handle_pdp_duplicates(df)) == 1


class TestHandlingDuplicates:
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

    @patch("edvise.utils.data_cleaning.LOGGER")
    def test_runs_pdp_handler(self, mock_logger, pdp_sample_df):
        result = data_cleaning.handling_duplicates(pdp_sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
