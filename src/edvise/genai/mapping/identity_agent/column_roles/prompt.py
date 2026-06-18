"""Prompts and parsing for ColumnRolesAgent."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from edvise.genai.mapping.identity_agent.profiling.schemas import RawTableProfile
from edvise.genai.mapping.shared.hitl import PIPELINE_HITL_CONFIDENCE_THRESHOLD
from edvise.genai.mapping.shared.utilities import strip_json_fences

from .schemas import ColumnRole, ColumnRoleAssignment, ColumnRolesResult

logger = logging.getLogger(__name__)

COLUMN_ROLES_SYSTEM_PROMPT = """
You classify raw CSV columns by semantic role for higher-ed student data pipelines.

## Output
Return a single JSON object (no markdown fences) with:
- `assignments`: array of `{ "column", "role", "confidence", "rationale" }` — one entry per input column
- `low_confidence_columns`: subset of column names where confidence < 0.7

## Roles (exact strings)
- `learner_id` — person/student identifier (student_id, pidm, banner_id, emplid, sis_id, etc.)
- `term` — academic term, semester, session, strm, enroll term
- `course_id` — course/class/section identifier (catalog number, class_nbr, crn, course_code)
- `program` — degree program, program_at_graduation, intended_program, college
- `major` — major, concentration, field of study
- `cohort` — cohort year, entry term, admit term, class year
- `measure` — numeric or coded outcomes: GPA, credits, grades, counts, rates, amounts
- `index` — synthetic row index (Unnamed: 0, row_number)
- `metadata` — descriptive text not part of grain (names, titles, descriptions)
- `other` — none of the above

## Rules
- Assign exactly one role per column from the input list.
- Use column names **and** sample values; institutions use different naming (pidm vs student_id).
- Float columns with grade/credit/GPA names are usually `measure`, not `learner_id`.
- High-cardinality opaque IDs matching person patterns → `learner_id` even if not named student_id.
- Do not invent columns; use exact names from the input.
- Be conservative: if unsure between `measure` and `metadata`, prefer `measure` for numeric dtypes.
""".strip()

COLUMN_ROLES_CONFIDENCE_THRESHOLD = PIPELINE_HITL_CONFIDENCE_THRESHOLD


def _column_summaries_for_prompt(
    raw_table_profile: RawTableProfile,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in raw_table_profile.columns:
        rows.append(
            {
                "column": col.name,
                "dtype": col.dtype,
                "null_rate": col.null_rate,
                "unique_count": col.unique_count,
                "sample_values": col.sample_values,
                "is_term_candidate_heuristic": col.is_term_candidate,
            }
        )
    return rows


def build_column_roles_user_message(
    institution_id: str,
    dataset: str,
    raw_table_profile: RawTableProfile,
) -> str:
    payload = {
        "institution_id": institution_id,
        "dataset": dataset,
        "dataset_kind_hint": dataset,
        "row_count": raw_table_profile.row_count,
        "columns": _column_summaries_for_prompt(raw_table_profile),
    }
    return (
        "Classify each column by semantic role.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )


def parse_column_roles_response(
    raw: str,
    *,
    institution_id: str,
    dataset: str,
    expected_columns: list[str],
) -> ColumnRolesResult:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    data = cast(dict[str, Any], json.loads(strip_json_fences(text)))

    assignments_raw = data.get("assignments")
    if not isinstance(assignments_raw, list):
        raise ValueError("assignments must be a list")

    assignments = [ColumnRoleAssignment.model_validate(x) for x in assignments_raw]
    assigned_names = {a.column for a in assignments}
    expected = list(expected_columns)
    missing = [c for c in expected if c not in assigned_names]
    if missing:
        raise ValueError(f"Missing role assignments for columns: {missing}")

    extra = assigned_names - set(expected)
    if extra:
        raise ValueError(f"Unexpected columns in assignments: {sorted(extra)}")

    low_raw = data.get("low_confidence_columns", [])
    if low_raw is None:
        low_raw = []
    if not isinstance(low_raw, list):
        raise ValueError("low_confidence_columns must be a list or null")
    low_confidence = [str(c) for c in low_raw]

    # Reconcile low_confidence with per-assignment confidence.
    for a in assignments:
        if (
            a.confidence < COLUMN_ROLES_CONFIDENCE_THRESHOLD
            and a.column not in low_confidence
        ):
            low_confidence.append(a.column)

    return ColumnRolesResult(
        institution_id=institution_id,
        dataset=dataset,
        assignments=assignments,
        low_confidence_columns=sorted(set(low_confidence)),
    )
