"""Table-class labels for ColumnRolesAgent (file kind only — not grain)."""

from __future__ import annotations

from enum import Enum


class FileKind(str, Enum):
    """
    Semantic class of a raw CSV table.

    Used for semantic key seeding and downstream domain priors. Does not encode grain
    or dedup policy — those are inferred later from profiling and the grain LLM.
    """

    STUDENT = "student"
    COURSE = "course"
    SEMESTER = "semester"
    DEGREE = "degree"
    OTHER = "other"


# Expected contents per kind (prompt / docs only).
FILE_KIND_EXPECTED_CONTENTS: dict[FileKind, str] = {
    FileKind.STUDENT: (
        "Student or person snapshot: learner identifier plus demographics, program, "
        "major, cohort, or similar attributes. Typically no course enrollment detail "
        "(no grades or credits per class)."
    ),
    FileKind.COURSE: (
        "Course enrollment or registration detail: learner identifier, one or more "
        "course/class/section identifier columns, academic term, and often grades or "
        "credits."
    ),
    FileKind.SEMESTER: (
        "Student-term summary: learner identifier, academic term, and term-level "
        "measures (term GPA, credits earned in term, enrollment status)."
    ),
    FileKind.DEGREE: (
        "Degree, award, or completion records: learner identifier, program or major "
        "context, credential or degree type, and completion or conferral term."
    ),
    FileKind.OTHER: (
        "Reference or lookup tables (program codes, section catalogs without "
        "enrollments, crosswalks), or tables that do not clearly match the kinds "
        "above. Downstream profiling relies more on statistical key search."
    ),
}


def file_kind_prompt_section() -> str:
    """Bullet list of file kinds for the ColumnRolesAgent system prompt."""
    lines = ["## File kinds (exact strings — pick one)", ""]
    for kind in FileKind:
        lines.append(f"- `{kind.value}` — {FILE_KIND_EXPECTED_CONTENTS[kind]}")
    return "\n".join(lines)


__all__ = [
    "FILE_KIND_EXPECTED_CONTENTS",
    "FileKind",
    "file_kind_prompt_section",
]
