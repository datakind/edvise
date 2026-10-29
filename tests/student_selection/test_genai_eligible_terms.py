"""In-memory projection of a GenAI batch onto eligible-term columns."""

from pathlib import Path

import pandas as pd
import pytest

from edvise.student_selection.eligible_terms.genai import (
    BatchInputFrame,
    GenaiTermProjectionError,
    project_genai_eligible_terms,
)

_INST = "school"
_SEASONS = [
    {"raw": "Spring", "canonical": "SPRING"},
    {"raw": "Fall", "canonical": "FALL"},
]
_CODES = [
    {"raw": "SP", "canonical": "SPRING"},
    {"raw": "SU1", "canonical": "SUMMER"},
    {"raw": "SU2", "canonical": "SUMMER"},
    {"raw": "FA", "canonical": "FALL"},
]
_HOOK_SPEC = {
    "file": "identity_hooks/school/term_hooks.py",
    "functions": [
        {"name": "year_extractor_term", "description": "year"},
        {"name": "season_extractor_term", "description": "season"},
    ],
}


def _dataset(table: str, term_config: dict | None) -> dict:
    return {
        "institution_id": _INST,
        "table": table,
        "term_config": term_config,
        "confidence": 0.9,
        "hitl_flag": False,
        "reasoning": "test",
    }


def _term_config(**overrides: object) -> dict:
    config: dict[str, object] = {
        "term_extraction": "standard",
        "year_semantics": "calendar_literal",
        "season_map": _SEASONS,
    }
    config.update(overrides)
    return config


def _grain(course_pk: list[str] | None = None) -> dict:
    return {
        "institution_id": _INST,
        "datasets": {
            "enroll": {
                "learner_id_alias": "SID",
                "row_selection_required": True,
                "post_clean_primary_key": course_pk
                or ["SID", "TERM_CODE", "course_id"],
            },
            "person": {
                "learner_id_alias": "SID",
                "row_selection_required": False,
                "post_clean_primary_key": ["SID"],
            },
        },
    }


def _files() -> list[BatchInputFrame]:
    return [
        BatchInputFrame(
            "person.csv",
            pd.DataFrame(
                {
                    "SID": [1, 2, 3],
                    "Entry Term": ["Fall 2022", "Fall 2023", "Fall 2023"],
                    "student_type": ["FIRST-TIME", "FIRST-TIME", "TRANSFER"],
                }
            ),
        ),
        BatchInputFrame(
            "enroll.csv",
            pd.DataFrame(
                {
                    "SID": [1, 2, 2, 3],
                    "TERM_CODE": ["2024FA", "2024FA", "2025SP", "2024FA"],
                    "course_id": ["C1", "C2", "C3", "C4"],
                }
            ),
        ),
    ]


def _contract() -> dict:
    return {
        "institution_id": _INST,
        "datasets": {
            "enroll": _dataset(
                "enroll",
                _term_config(term_col="TERM_CODE", season_map=_CODES),
            ),
            "person": _dataset(
                "person",
                _term_config(term_col="Entry Term"),
            ),
        },
    }


def test_project_named_datasets_onto_term_columns() -> None:
    students, courses = project_genai_eligible_terms(
        _files(),
        _contract(),
        grain_contract=_grain(),
        student_id_col="learner_id",
    )

    assert list(students["learner_id"]) == [1, 2, 3]
    assert list(students["entry_term"]) == ["FALL", "FALL", "FALL"]
    assert list(students["entry_year"]) == ["2022-23", "2023-24", "2023-24"]
    assert list(students["student_type"]) == ["FIRST-TIME", "FIRST-TIME", "TRANSFER"]
    assert list(courses["learner_id"]) == [1, 2, 2, 3]
    assert list(courses["academic_term"]) == ["FALL", "FALL", "SPRING", "FALL"]
    assert list(courses["academic_year"]) == ["2024-25"] * 4
    assert "_edvise_term_season" not in courses.columns
    assert "_year" not in students.columns


def test_quarter_sessions_share_one_season_label() -> None:
    _students, courses = project_genai_eligible_terms(
        [
            BatchInputFrame(
                "person.csv",
                pd.DataFrame({"SID": [1, 2]}),
            ),
            BatchInputFrame(
                "enroll.csv",
                pd.DataFrame(
                    {
                        "SID": [1, 1, 2],
                        "TERM_CODE": ["2024SU1", "2024SU2", "2024SU1"],
                    }
                ),
            ),
        ],
        {
            "institution_id": _INST,
            "datasets": {
                "enroll": _dataset(
                    "enroll",
                    _term_config(term_col="TERM_CODE", season_map=_CODES),
                ),
                "person": _dataset("person", None),
            },
        },
        grain_contract=_grain(course_pk=["SID", "TERM_CODE"]),
    )

    assert set(
        zip(courses["academic_term"], courses["academic_year"], strict=True)
    ) == {("SUMMER", "2023-24")}
    assert "entry_term" not in _students.columns


def test_academic_year_prefix_rolls_spring_forward() -> None:
    _students, courses = project_genai_eligible_terms(
        [
            BatchInputFrame(
                "person.csv",
                pd.DataFrame({"sid": ["1"]}),
            ),
            BatchInputFrame(
                "enroll.csv",
                pd.DataFrame({"sid": ["1"], "term_code": ["2017SR"]}),
            ),
        ],
        {
            "institution_id": _INST,
            "datasets": {
                "enroll": _dataset(
                    "enroll",
                    _term_config(
                        term_col="term_code",
                        year_semantics="academic_year_prefix",
                        season_map=[
                            {"raw": "SR", "canonical": "SPRING"},
                            {"raw": "FA", "canonical": "FALL"},
                        ],
                    ),
                ),
                "person": _dataset("person", None),
            },
        },
        grain_contract=_grain(),
    )

    assert courses.loc[0, "academic_term"] == "SPRING"
    assert courses.loc[0, "academic_year"] == "2017-18"


def test_split_year_and_season_columns() -> None:
    _students, courses = project_genai_eligible_terms(
        [
            BatchInputFrame(
                "person.csv",
                pd.DataFrame({"sid": ["1"]}),
            ),
            BatchInputFrame(
                "enroll.csv",
                pd.DataFrame(
                    {"sid": ["1"], "Term Year": [2024], "Term Season": ["Fall"]}
                ),
            ),
        ],
        {
            "institution_id": _INST,
            "datasets": {
                "enroll": _dataset(
                    "enroll",
                    _term_config(year_col="Term Year", season_col="Term Season"),
                ),
                "person": _dataset("person", None),
            },
        },
        grain_contract=_grain(),
    )

    assert courses.loc[0, "academic_term"] == "FALL"
    assert courses.loc[0, "academic_year"] == "2024-25"


def test_hook_required_without_module_fails() -> None:
    with pytest.raises(GenaiTermProjectionError, match="hook_required") as raised:
        project_genai_eligible_terms(
            _files(),
            {
                "institution_id": _INST,
                "datasets": {
                    "enroll": _dataset(
                        "enroll",
                        _term_config(
                            term_col="TERM_CODE",
                            season_map=_CODES,
                            term_extraction="hook_required",
                            hook_spec=_HOOK_SPEC,
                        ),
                    ),
                    "person": _dataset("person", None),
                },
            },
            grain_contract=_grain(),
        )

    assert "promoted term hook is not available" in raised.value.reason


def test_student_hook_required_without_module_fails() -> None:
    with pytest.raises(GenaiTermProjectionError, match="Student dataset") as raised:
        project_genai_eligible_terms(
            _files(),
            {
                "institution_id": _INST,
                "datasets": {
                    "enroll": _dataset(
                        "enroll",
                        _term_config(term_col="TERM_CODE", season_map=_CODES),
                    ),
                    "person": _dataset(
                        "person",
                        _term_config(
                            term_col="Entry Term",
                            term_extraction="hook_required",
                            hook_spec={
                                "file": "identity_hooks/school/term_hooks.py",
                                "functions": [
                                    {
                                        "name": "year_extractor_entry",
                                        "description": "year",
                                    },
                                    {
                                        "name": "season_extractor_entry",
                                        "description": "season",
                                    },
                                ],
                            },
                        ),
                    ),
                },
            },
            grain_contract=_grain(),
        )

    assert "hook_required" in raised.value.reason


def test_promoted_hook_labels_nonstandard_terms(tmp_path: Path) -> None:
    hook_dir = tmp_path / "identity_hooks" / "school"
    hook_dir.mkdir(parents=True)
    (hook_dir / "term_hooks.py").write_text(
        "def year_extractor_term(term):\n"
        "    return 2000 + int(str(term).strip()[-2:])\n"
        "\n"
        "def season_extractor_term(term):\n"
        "    return str(term).strip()[:2].upper()\n",
        encoding="utf-8",
    )
    _students, courses = project_genai_eligible_terms(
        [
            BatchInputFrame(
                "person.csv",
                pd.DataFrame({"SID": [1]}),
            ),
            BatchInputFrame(
                "enroll.csv",
                pd.DataFrame({"SID": [1], "TERM_CODE": ["FA19"]}),
            ),
        ],
        {
            "institution_id": _INST,
            "datasets": {
                "enroll": _dataset(
                    "enroll",
                    _term_config(
                        term_col="TERM_CODE",
                        season_map=[
                            {"raw": "SP", "canonical": "SPRING"},
                            {"raw": "FA", "canonical": "FALL"},
                        ],
                        term_extraction="hook_required",
                        hook_spec=_HOOK_SPEC,
                    ),
                ),
                "person": _dataset("person", None),
            },
        },
        grain_contract=_grain(),
        hook_modules_root=tmp_path,
    )

    assert courses.loc[0, "academic_term"] == "FALL"
    assert courses.loc[0, "academic_year"] == "2019-20"


def test_missing_hook_file_fails(tmp_path: Path) -> None:
    with pytest.raises(GenaiTermProjectionError, match="Could not project terms"):
        project_genai_eligible_terms(
            _files(),
            {
                "institution_id": _INST,
                "datasets": {
                    "enroll": _dataset(
                        "enroll",
                        _term_config(
                            term_col="TERM_CODE",
                            season_map=_CODES,
                            term_extraction="hook_required",
                            hook_spec=_HOOK_SPEC,
                        ),
                    ),
                    "person": _dataset("person", None),
                },
            },
            grain_contract=_grain(),
            hook_modules_root=tmp_path,
        )


def test_configured_filenames_bind_the_files() -> None:
    _students, courses = project_genai_eligible_terms(
        [
            BatchInputFrame(
                "roster.csv",
                pd.DataFrame({"sid": ["1"], "term_code": ["2024FA"]}),
            ),
            BatchInputFrame(
                "people.csv",
                pd.DataFrame({"sid": ["1"], "entry_term": ["Fall 2022"]}),
            ),
        ],
        _contract(),
        grain_contract=_grain(),
        dataset_files={
            "enroll": ["genai_mapping/onboard/roster.csv"],
            "person": ["people.csv"],
        },
    )

    assert list(courses["academic_term"]) == ["FALL"]
    assert list(courses["academic_year"]) == ["2024-25"]
    assert list(_students["entry_term"]) == ["FALL"]
    assert list(_students["entry_year"]) == ["2022-23"]


def test_configured_course_file_missing_columns_is_not_replaced() -> None:
    with pytest.raises(GenaiTermProjectionError, match="missing term columns"):
        project_genai_eligible_terms(
            [
                BatchInputFrame(
                    "roster.csv",
                    pd.DataFrame({"sid": ["1"]}),
                ),
                BatchInputFrame(
                    "other.csv",
                    pd.DataFrame({"sid": ["1"], "term_code": ["2024FA"]}),
                ),
            ],
            _contract(),
            grain_contract=_grain(),
            dataset_files={
                "enroll": ["roster.csv"],
                "person": ["other.csv"],
            },
        )


def test_ambiguous_dataset_names_are_not_guessed() -> None:
    with pytest.raises(GenaiTermProjectionError, match="Multiple batch files"):
        project_genai_eligible_terms(
            [
                BatchInputFrame(
                    "enroll.csv",
                    pd.DataFrame({"sid": ["1"], "term_code": ["2024FA"]}),
                ),
                BatchInputFrame(
                    "enroll_spring.csv",
                    pd.DataFrame({"sid": ["2"], "term_code": ["2024SP"]}),
                ),
                BatchInputFrame(
                    "person.csv",
                    pd.DataFrame({"sid": ["1"], "entry_term": ["Fall 2022"]}),
                ),
            ],
            _contract(),
            grain_contract=_grain(course_pk=["sid"]),
        )
