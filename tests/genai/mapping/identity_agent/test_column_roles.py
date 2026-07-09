import json

import pandas as pd
import pytest
from pydantic import ValidationError

from edvise.utils.llm_utils import llm_complete_with_parse_retry
from edvise.genai.mapping.identity_agent.column_roles.file_kinds import FileKind
from edvise.genai.mapping.identity_agent.column_roles.fallback import (
    apply_column_role_fallbacks,
)
from edvise.genai.mapping.identity_agent.column_roles.prompt import (
    build_column_roles_user_message,
    parse_column_roles_response,
)
from edvise.genai.mapping.identity_agent.column_roles.runner import (
    run_column_roles_for_dataset,
)
from edvise.genai.mapping.identity_agent.column_roles.schemas import (
    ColumnRole,
    ColumnRoleAssignment,
    ColumnRolesResult,
)
from edvise.genai.mapping.identity_agent.profiling.candidate_keys import (
    profile_candidate_keys,
)
from edvise.genai.mapping.identity_agent.profiling.schemas import (
    RawColumnProfile,
    RawTableProfile,
)
from edvise.genai.mapping.identity_agent.profiling.semantic_keys import (
    build_semantic_key_column_sets,
)


def _roles(**kwargs: object) -> ColumnRolesResult:
    defaults: dict[str, object] = {
        "institution_id": "test_u",
        "dataset": "student",
        "file_kind": FileKind.STUDENT,
        "file_kind_confidence": 0.95,
        "file_kind_rationale": "test fixture",
    }
    defaults.update(kwargs)
    return ColumnRolesResult.model_validate(defaults)


def test_parse_column_roles_response():
    payload = {
        "file_kind": "student",
        "file_kind_confidence": 0.92,
        "file_kind_rationale": "demographics table",
        "assignments": [
            {
                "column": "pidm",
                "role": "learner_id",
                "confidence": 0.95,
                "rationale": "person id",
            },
            {
                "column": "program_at_graduation",
                "role": "program",
                "confidence": 0.9,
                "rationale": "program field",
            },
        ],
        "low_confidence_columns": [],
    }
    result = parse_column_roles_response(
        json.dumps(payload),
        institution_id="test_u",
        dataset="student",
        expected_columns=["pidm", "program_at_graduation"],
    )
    assert result.file_kind == FileKind.STUDENT
    assert result.file_kind_confidence == 0.92
    assert result.learner_id_column() == "pidm"
    assert result.columns_with_role(ColumnRole.PROGRAM) == ["program_at_graduation"]


def test_parse_column_roles_response_requires_file_kind():
    payload = {
        "assignments": [
            {
                "column": "pidm",
                "role": "learner_id",
                "confidence": 0.95,
                "rationale": "person id",
            },
        ],
        "low_confidence_columns": [],
    }
    with pytest.raises(ValidationError, match="file_kind"):
        parse_column_roles_response(
            json.dumps(payload),
            institution_id="test_u",
            dataset="student",
            expected_columns=["pidm"],
        )


def _column_roles_payload(columns: list[str]) -> dict:
    return {
        "file_kind": "student",
        "file_kind_confidence": 0.9,
        "file_kind_rationale": "demographics table",
        "assignments": [
            {
                "column": c,
                "role": "learner_id" if c == "pidm" else "measure",
                "confidence": 0.9,
                "rationale": "test",
            }
            for c in columns
        ],
        "low_confidence_columns": [],
    }


def test_parse_column_roles_response_missing_column_raises_validation_error():
    """A missing column must raise ``ValidationError`` so the shared retry helper retries."""
    payload = _column_roles_payload(["pidm"])
    with pytest.raises(ValidationError, match="dev_engl"):
        parse_column_roles_response(
            json.dumps(payload),
            institution_id="test_u",
            dataset="student",
            expected_columns=["pidm", "dev_engl"],
        )


def test_parse_column_roles_response_extra_column_raises_validation_error():
    payload = _column_roles_payload(["pidm", "surprise"])
    with pytest.raises(ValidationError, match="surprise"):
        parse_column_roles_response(
            json.dumps(payload),
            institution_id="test_u",
            dataset="student",
            expected_columns=["pidm"],
        )


def test_missing_column_is_retried_and_recovers():
    """Omitting a column on the first response triggers a retry that then succeeds."""
    calls: list[str] = []
    incomplete = json.dumps(_column_roles_payload(["pidm"]))
    complete = json.dumps(_column_roles_payload(["pidm", "dev_engl"]))

    def llm_complete(system: str, user: str) -> str:
        calls.append(user)
        return incomplete if len(calls) == 1 else complete

    def parse(raw: str):
        return parse_column_roles_response(
            raw,
            institution_id="test_u",
            dataset="student",
            expected_columns=["pidm", "dev_engl"],
        )

    result = llm_complete_with_parse_retry(
        llm_complete, "sys", "user", parse, max_retries=3
    )
    assert len(calls) == 2
    assert {a.column for a in result.assignments} == {"pidm", "dev_engl"}
    # The correction hint must be appended to the user message on the retry.
    assert "dev_engl" in calls[1]


def test_build_column_roles_user_message_includes_expected_assignment_count():
    profile = RawTableProfile(
        institution_id="test_u",
        dataset="student",
        row_count=100,
        column_count=2,
        columns=[
            RawColumnProfile(
                name="pidm",
                dtype="int64",
                null_rate=0.0,
                null_rate_including_tokens=0.0,
                unique_count=100,
                sample_values=["1", "2"],
                is_term_candidate=False,
            ),
            RawColumnProfile(
                name="dev_engl",
                dtype="object",
                null_rate=0.1,
                null_rate_including_tokens=0.1,
                unique_count=3,
                sample_values=["Y", "N"],
                is_term_candidate=False,
            ),
        ],
    )
    message = build_column_roles_user_message("test_u", "student", profile)
    payload = json.loads(message.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
    assert payload["expected_assignment_count"] == 2
    assert len(payload["columns"]) == 2


def test_parse_column_roles_response_allows_missing_when_completeness_disabled():
    payload = _column_roles_payload(["pidm"])
    result = parse_column_roles_response(
        json.dumps(payload),
        institution_id="test_u",
        dataset="student",
        expected_columns=["pidm", "dev_engl"],
        validate_completeness=False,
    )
    assert {a.column for a in result.assignments} == {"pidm"}


def test_fallback_assigns_measure_for_omitted_dev_engl():
    result = _roles(
        assignments=[
            ColumnRoleAssignment(
                column="pidm",
                role=ColumnRole.LEARNER_ID,
                confidence=0.95,
                rationale="person id",
            ),
        ],
    )
    patched = apply_column_role_fallbacks(result, columns=["pidm", "dev_engl"])
    assert patched.role_for("dev_engl") == ColumnRole.MEASURE
    assert "dev_engl" in patched.fallback_applied
    assert "dev_engl" in patched.low_confidence_columns


def test_fallback_assigns_other_for_omitted_unknown_column():
    result = _roles(
        assignments=[
            ColumnRoleAssignment(
                column="pidm",
                role=ColumnRole.LEARNER_ID,
                confidence=0.95,
                rationale="person id",
            ),
        ],
    )
    patched = apply_column_role_fallbacks(result, columns=["pidm", "mystery_flag"])
    assert patched.role_for("mystery_flag") == ColumnRole.OTHER
    assert "mystery_flag" in patched.fallback_applied


def test_missing_column_fallback_after_retry_exhausted():
    incomplete = json.dumps(_column_roles_payload(["pidm"]))
    df = pd.DataFrame({"pidm": [1, 2], "dev_engl": ["Y", "N"]})

    def llm_complete(system: str, user: str) -> str:
        return incomplete

    result = run_column_roles_for_dataset(
        institution_id="test_u",
        dataset="student",
        df=df,
        llm_complete=llm_complete,
    )
    assert result.role_for("pidm") == ColumnRole.LEARNER_ID
    assert result.role_for("dev_engl") == ColumnRole.MEASURE
    assert "dev_engl" in result.fallback_applied


def test_fallback_assigns_learner_id_from_name_pattern():
    result = _roles(
        assignments=[
            ColumnRoleAssignment(
                column="student_id",
                role=ColumnRole.MEASURE,
                confidence=0.5,
                rationale="wrong",
            ),
        ],
        low_confidence_columns=["student_id"],
    )
    patched = apply_column_role_fallbacks(result, columns=["student_id"])
    assert patched.learner_id_column() == "student_id"
    assert patched.assignments[0].role == ColumnRole.LEARNER_ID
    assert patched.file_kind == FileKind.STUDENT
    assert "student_id" in patched.fallback_applied


def test_build_semantic_key_column_sets_student():
    roles = _roles(
        assignments=[
            ColumnRoleAssignment(
                column="student_id",
                role=ColumnRole.LEARNER_ID,
                confidence=0.99,
            ),
            ColumnRoleAssignment(
                column="program_at_graduation",
                role=ColumnRole.PROGRAM,
                confidence=0.9,
            ),
            ColumnRoleAssignment(
                column="major_at_graduation",
                role=ColumnRole.MAJOR,
                confidence=0.9,
            ),
        ],
    )
    keys = build_semantic_key_column_sets("student", roles)
    assert ["student_id"] in keys
    assert ["student_id", "program_at_graduation"] in keys
    assert ["student_id", "major_at_graduation"] in keys
    assert ["student_id", "program_at_graduation", "major_at_graduation"] in keys


def test_build_semantic_key_column_sets_course_composite_course_id():
    """Multiple course_id columns form a composite semantic grain (IIT-shaped)."""
    roles = _roles(
        dataset="course",
        file_kind=FileKind.COURSE,
        assignments=[
            ColumnRoleAssignment(
                column="student_id",
                role=ColumnRole.LEARNER_ID,
                confidence=0.99,
            ),
            ColumnRoleAssignment(
                column="semester",
                role=ColumnRole.TERM,
                confidence=0.99,
            ),
            ColumnRoleAssignment(
                column="course_prefix",
                role=ColumnRole.COURSE_ID,
                confidence=0.85,
            ),
            ColumnRoleAssignment(
                column="course_number",
                role=ColumnRole.COURSE_ID,
                confidence=0.8,
            ),
            ColumnRoleAssignment(
                column="course_section",
                role=ColumnRole.COURSE_ID,
                confidence=0.85,
            ),
            ColumnRoleAssignment(
                column="course_name",
                role=ColumnRole.METADATA,
                confidence=0.95,
            ),
        ],
    )
    keys = build_semantic_key_column_sets("course", roles)
    assert keys[0] == [
        "student_id",
        "course_prefix",
        "course_number",
        "course_section",
        "semester",
    ]
    assert [
        "student_id",
        "course_prefix",
        "course_number",
        "course_section",
    ] in keys
    assert ["student_id", "semester"] in keys


def test_build_semantic_key_column_sets_uses_file_kind_not_dataset_key():
    """Misnamed logical dataset key still gets course seeds when file_kind is course."""
    roles = _roles(
        dataset="registration",
        file_kind=FileKind.COURSE,
        assignments=[
            ColumnRoleAssignment(
                column="student_id",
                role=ColumnRole.LEARNER_ID,
                confidence=0.99,
            ),
            ColumnRoleAssignment(
                column="semester",
                role=ColumnRole.TERM,
                confidence=0.99,
            ),
            ColumnRoleAssignment(
                column="course_number",
                role=ColumnRole.COURSE_ID,
                confidence=0.8,
            ),
            ColumnRoleAssignment(
                column="course_section",
                role=ColumnRole.COURSE_ID,
                confidence=0.85,
            ),
        ],
    )
    keys = build_semantic_key_column_sets("registration", roles)
    assert keys[0] == [
        "student_id",
        "course_number",
        "course_section",
        "semester",
    ]


def test_build_semantic_key_column_sets_other_role_driven():
    roles = _roles(
        dataset="program_codes",
        file_kind=FileKind.OTHER,
        assignments=[
            ColumnRoleAssignment(
                column="term_code",
                role=ColumnRole.TERM,
                confidence=0.95,
            ),
        ],
    )
    keys = build_semantic_key_column_sets("program_codes", roles)
    assert keys == [["term_code"]]


def test_build_semantic_key_column_sets_course_single_course_id():
    roles = _roles(
        dataset="course",
        file_kind=FileKind.COURSE,
        assignments=[
            ColumnRoleAssignment(
                column="student_id",
                role=ColumnRole.LEARNER_ID,
                confidence=0.99,
            ),
            ColumnRoleAssignment(
                column="term",
                role=ColumnRole.TERM,
                confidence=0.99,
            ),
            ColumnRoleAssignment(
                column="crn",
                role=ColumnRole.COURSE_ID,
                confidence=0.95,
            ),
        ],
    )
    keys = build_semantic_key_column_sets("course", roles)
    assert keys[0] == ["student_id", "crn", "term"]
    assert ["student_id", "crn"] in keys


def test_profile_candidate_keys_includes_composite_course_semantic_key_first():
    df = pd.DataFrame(
        {
            "student_id": ["a", "a", "b", "b"],
            "course_number": [100, 100, 200, 200],
            "course_section": ["AA", "AB", "AA", "AA"],
            "semester": ["2024-10", "2024-10", "2024-10", "2025-20"],
            "grade": ["A", "B", "A", "A"],
        }
    )
    roles = _roles(
        dataset="course",
        file_kind=FileKind.COURSE,
        assignments=[
            ColumnRoleAssignment(
                column="student_id",
                role=ColumnRole.LEARNER_ID,
                confidence=0.99,
            ),
            ColumnRoleAssignment(
                column="semester",
                role=ColumnRole.TERM,
                confidence=0.99,
            ),
            ColumnRoleAssignment(
                column="course_number",
                role=ColumnRole.COURSE_ID,
                confidence=0.8,
            ),
            ColumnRoleAssignment(
                column="course_section",
                role=ColumnRole.COURSE_ID,
                confidence=0.85,
            ),
            ColumnRoleAssignment(
                column="grade",
                role=ColumnRole.MEASURE,
                confidence=0.95,
            ),
        ],
    )
    result = profile_candidate_keys(
        df,
        institution_id="test_u",
        dataset="course",
        column_roles=roles,
    )
    first_cols = result.key_profile.candidate_key_profiles[0].candidate_key.columns
    assert first_cols == [
        "student_id",
        "course_number",
        "course_section",
        "semester",
    ]


def test_profile_candidate_keys_includes_semantic_keys_first():
    df = pd.DataFrame(
        {
            "student_id": [1, 2, 3, 1],
            "program_at_graduation": ["A", "B", "C", "A"],
            "total_credits": [10.0, 20.0, 30.0, 10.0],
        }
    )
    roles = _roles(
        assignments=[
            ColumnRoleAssignment(
                column="student_id",
                role=ColumnRole.LEARNER_ID,
                confidence=0.99,
            ),
            ColumnRoleAssignment(
                column="program_at_graduation",
                role=ColumnRole.PROGRAM,
                confidence=0.9,
            ),
            ColumnRoleAssignment(
                column="total_credits",
                role=ColumnRole.MEASURE,
                confidence=0.95,
            ),
        ],
    )
    result = profile_candidate_keys(
        df,
        institution_id="test_u",
        dataset="student",
        column_roles=roles,
    )
    first_cols = result.key_profile.candidate_key_profiles[0].candidate_key.columns
    assert first_cols == ["student_id"]
    col_sets = [
        p.candidate_key.columns for p in result.key_profile.candidate_key_profiles
    ]
    assert ["student_id", "program_at_graduation"] in col_sets
