import pandas as pd

from edvise.dataio.batch_frame_matching import bind_filenames_to_datasets
from edvise.student_selection.eligible_inference_terms import (
    resolve_standardized_eligible_inference_terms,
)
from edvise.student_selection.genai_eligible_inference_terms import (
    apply_genai_mapping_for_eligible_terms,
    resolve_genai_eligible_inference_terms,
)
from edvise.student_selection.legacy_eligible_inference_terms import (
    resolve_legacy_eligible_inference_terms,
)


def test_bind_filenames_to_datasets_assigns_unique_matches() -> None:
    bound = bind_filenames_to_datasets(
        [
            "Edvise Learner Report_20260916.csv",
            "Edvise Course Report_20260916.csv",
        ],
        {
            "student": ["CCC Student File.csv"],
            "course": ["CCC Course File.csv"],
        },
    )
    assert bound["student"] == "Edvise Learner Report_20260916.csv"
    assert bound["course"] == "Edvise Course Report_20260916.csv"


def test_resolve_standardized_eligible_inference_terms_excludes_training_cohorts() -> (
    None
):
    students = pd.DataFrame(
        {
            "student_id": [1, 2, 3],
            "enrollment_type": [" first-time ", "FIRST-TIME", "TRANSFER"],
            "cohort_term": ["FALL", "SPRING", "FALL"],
            "cohort": ["2022-23", "2023-24", "2023-24"],
        }
    )
    courses = pd.DataFrame(
        {
            "student_id": ["1", "2", "2", "3"],
            "academic_term": ["FALL", "FALL", "SPRING", "SPRING"],
            "academic_year": ["2024-25", "2024-25", "2024-25", "2024-25"],
        }
    )
    result = resolve_standardized_eligible_inference_terms(
        students,
        courses,
        {
            "student_criteria": {"enrollment_type": "FIRST-TIME"},
            "training_cohorts": ["fall 2022-23"],
        },
        "inference batch",
    )
    assert result.status == "valid"
    assert [term.term_label for term in result.terms] == [
        "spring 2024-25",
        "fall 2024-25",
    ]


def _genai_term_output(institution_id: str = "genai_school") -> dict:
    return {
        "institution_id": institution_id,
        "datasets": {
            "course": {
                "institution_id": institution_id,
                "table": "course",
                "term_config": {
                    "year_col": "yr",
                    "season_col": "sem",
                    "season_map": [
                        {"raw": "SP", "canonical": "SPRING"},
                        {"raw": "FA", "canonical": "FALL"},
                    ],
                    "term_extraction": "standard",
                },
                "confidence": 0.95,
                "hitl_flag": False,
                "reasoning": "split year and season columns",
            },
            "student": {
                "institution_id": institution_id,
                "table": "student",
                "term_config": None,
                "confidence": 1.0,
                "hitl_flag": False,
                "reasoning": "no term column on student file",
            },
        },
    }


def test_apply_genai_mapping_normalizes_identity_term_columns() -> None:
    students, courses = apply_genai_mapping_for_eligible_terms(
        {
            "school_student_file.csv": pd.DataFrame(
                {"student_id": [1, 2], "enrollment_type": ["FIRST-TIME", "TRANSFER"]}
            ),
            "school_course_file.csv": pd.DataFrame(
                {
                    "student_id": [1, 1, 2],
                    "yr": [2024, 2024, 2024],
                    "sem": ["FA", "SP", "FA"],
                }
            ),
        },
        term_output=_genai_term_output(),
        dataset_files={
            "student": ["school_student_file.csv"],
            "course": ["school_course_file.csv"],
        },
        institution_id="genai_school",
    )
    assert "academic_term" in courses.columns
    assert "academic_year" in courses.columns
    result = resolve_genai_eligible_inference_terms(
        {
            "school_student_file.csv": pd.DataFrame(
                {"student_id": [1, 2], "enrollment_type": ["FIRST-TIME", "TRANSFER"]}
            ),
            "school_course_file.csv": pd.DataFrame(
                {
                    "student_id": [1, 1, 2],
                    "yr": [2024, 2024, 2024],
                    "sem": ["FA", "SP", "FA"],
                }
            ),
        },
        {"student_criteria": {"enrollment_type": "FIRST-TIME"}},
        term_output=_genai_term_output(),
        dataset_files={
            "student": ["school_student_file.csv"],
            "course": ["school_course_file.csv"],
        },
        institution_id="genai_school",
        batch_name="batch",
    )
    assert result.status == "valid"
    labels = {term.term_label for term in result.terms}
    assert "fall 2024-25" in labels
    assert any(label.startswith("spring ") for label in labels)


def test_resolve_genai_eligible_inference_terms_invalid_without_file_match() -> None:
    result = resolve_genai_eligible_inference_terms(
        {"unrelated.csv": pd.DataFrame({"student_id": [1]})},
        {},
        term_output=_genai_term_output(),
        dataset_files={
            "student": ["student.csv"],
            "course": ["course.csv"],
        },
        batch_name="batch",
    )
    assert result.status == "invalid"
    assert "match" in (result.reason or "").lower()


def test_resolve_legacy_eligible_inference_terms_from_native_term_column() -> None:
    result = resolve_legacy_eligible_inference_terms(
        {},
        {},
        frames_by_filename={
            "cohort_extract.csv": pd.DataFrame({"student_id": [10, 20]}),
            "course_extract.csv": pd.DataFrame(
                {
                    "student_id": [10, 10, 20],
                    "term_order": [202501, 202503, 202501],
                }
            ),
        },
        batch_name="legacy-batch",
    )
    assert result.status == "valid"
    assert result.batch_name == "legacy-batch"
    labels = {term.term_label: term.valid_student_count for term in result.terms}
    assert labels["202501"] == 2
    assert labels["202503"] == 1


def test_resolve_legacy_eligible_inference_terms_uses_pdp_columns_when_present() -> (
    None
):
    result = resolve_legacy_eligible_inference_terms(
        {
            "STUDENT": pd.DataFrame({"student_id": [1]}),
            "COURSE": pd.DataFrame(
                {
                    "student_id": [1],
                    "academic_term": ["FALL"],
                    "academic_year": ["2024-25"],
                }
            ),
        },
        {},
        batch_name="batch",
    )
    assert result.status == "valid"
    assert result.terms[0].term_label == "fall 2024-25"
