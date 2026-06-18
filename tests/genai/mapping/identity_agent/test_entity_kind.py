"""Tests for entity kind resolution and backwards-compatible semantic templates."""

from edvise.genai.mapping.identity_agent.column_roles.schemas import (
    ColumnRole,
    ColumnRoleAssignment,
    ColumnRolesResult,
)
from edvise.genai.mapping.identity_agent.profiling.entity_kind import (
    resolve_entity_kind,
)
from edvise.genai.mapping.identity_agent.profiling.semantic_keys import (
    build_semantic_key_column_sets,
)


def _course_roles() -> ColumnRolesResult:
    return ColumnRolesResult(
        institution_id="test_u",
        dataset="registration",
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


def test_resolve_entity_kind_backwards_compat_canonical_names():
    assert resolve_entity_kind("student") == "student"
    assert resolve_entity_kind("course") == "course"
    assert resolve_entity_kind("semester") == "semester"


def test_resolve_entity_kind_builtin_aliases():
    assert resolve_entity_kind("registration") == "course"
    assert resolve_entity_kind("degrees") == "degree"


def test_resolve_entity_kind_explicit_override():
    assert resolve_entity_kind("my_enrollments", configured_kind="course") == "course"
    assert resolve_entity_kind("student", configured_kind="semester") == "semester"


def test_build_semantic_keys_registration_alias_uses_course_template():
    roles = _course_roles()
    keys = build_semantic_key_column_sets("registration", roles)
    assert keys[0] == [
        "student_id",
        "course_number",
        "course_section",
        "semester",
    ]


def test_build_semantic_keys_entity_kind_override_on_custom_name():
    roles = _course_roles()
    keys = build_semantic_key_column_sets(
        "my_file",
        roles,
        entity_kind="course",
    )
    assert keys[0] == [
        "student_id",
        "course_number",
        "course_section",
        "semester",
    ]


def test_build_semantic_keys_unknown_uses_role_driven_fallback():
    roles = ColumnRolesResult(
        institution_id="test_u",
        dataset="term_calendar",
        assignments=[
            ColumnRoleAssignment(
                column="term_code",
                role=ColumnRole.TERM,
                confidence=0.95,
            ),
        ],
    )
    keys = build_semantic_key_column_sets("term_calendar", roles)
    assert keys == [["term_code"]]
