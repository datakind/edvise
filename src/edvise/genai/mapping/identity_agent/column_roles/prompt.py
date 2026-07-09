"""Prompts and parsing for ColumnRolesAgent."""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from edvise.genai.mapping.identity_agent.profiling.schemas import RawTableProfile
from edvise.genai.mapping.shared.hitl import PIPELINE_HITL_CONFIDENCE_THRESHOLD
from edvise.genai.mapping.shared.utilities import strip_json_fences

from .file_kinds import file_kind_prompt_section
from .schemas import ColumnRolesLLMResponse, ColumnRolesResult

logger = logging.getLogger(__name__)

COLUMN_ROLES_SYSTEM_PROMPT = f"""
You classify raw higher-ed CSV tables: first the **file kind** (table class), then each column's semantic role.

{file_kind_prompt_section()}

## Output
Return a single JSON object (no markdown fences) with:
- `file_kind`: one of the file kind strings above
- `file_kind_confidence`: 0.0–1.0
- `file_kind_rationale`: one short sentence
- `assignments`: array of `{{ "column", "role", "confidence", "rationale" }}` — one entry per input column
- `low_confidence_columns`: subset of column names where confidence < 0.7

## Column roles (exact strings)
- `learner_id` — person/student identifier (student_id, pidm, banner_id, emplid, sis_id, etc.)
- `term` — academic term, semester, session, strm, enroll term
- `course_id` — course/class/section identifier component (catalog number, section, prefix,
  class_nbr, crn, course_code). On course/enrollment tables tag **each** key component as
  `course_id`. Do not tag titles (`course_name`, `course_title`) as `course_id`; use `metadata`.
- `program` — degree program, program_at_graduation, intended_program, college
- `major` — major, concentration, field of study
- `cohort` — cohort year, entry term, admit term, class year
- `measure` — numeric or coded outcomes: GPA, credits, grades, counts, rates, amounts;
  developmental/remedial placement flags (`dev_engl`, `dev_math`, `dev_read`, gateway flags)
- `index` — synthetic row index (Unnamed: 0, row_number)
- `metadata` — descriptive text not part of identifiers (names, titles, descriptions)
- `other` — none of the above

## Rules
1. Choose `file_kind` from column names, dtypes, and sample values — **not** from the logical
   dataset label in the user message (institutions misname files).
2. Assign exactly one role per column from the input list. The `assignments` array length must
   equal `expected_assignment_count` in the user message — never omit a column; use `other` if unsure.
3. Use column names **and** sample values; institutions use different naming (pidm vs student_id).
4. Float columns with grade/credit/GPA names are usually `measure`, not `learner_id`.
5. High-cardinality opaque IDs matching person patterns → `learner_id` even if not named student_id.
6. Do not invent columns; use exact names from the input.
7. Multiple columns may share the same role (especially composite `course_id`).
8. When unsure between `measure` and `metadata`, prefer `measure` for numeric dtypes.
9. Section catalogs, program code lists, and join lookups → `file_kind`: `other`.
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
    columns = _column_summaries_for_prompt(raw_table_profile)
    payload = {
        "institution_id": institution_id,
        "dataset": dataset,
        "row_count": raw_table_profile.row_count,
        "expected_assignment_count": len(columns),
        "columns": columns,
    }
    return (
        "Classify the file kind, then classify each column by semantic role.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )


def parse_column_roles_response(
    raw: str,
    *,
    institution_id: str,
    dataset: str,
    expected_columns: list[str],
    validate_completeness: bool = True,
) -> ColumnRolesResult:
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    data = cast(dict[str, Any], json.loads(strip_json_fences(text)))

    # Structural + completeness validation lives in the Pydantic model so that an
    # incomplete ``assignments`` list (e.g. the LLM omitting a column) raises a
    # ``ValidationError`` and is retried with a correction hint by
    # ``call_with_retry`` instead of hard-failing on the first response.
    context: dict[str, list[str]] | None = None
    if validate_completeness:
        context = {"expected_columns": list(expected_columns)}
    response = ColumnRolesLLMResponse.model_validate(data, context=context)

    low_confidence = list(response.low_confidence_columns)
    for a in response.assignments:
        if (
            a.confidence < COLUMN_ROLES_CONFIDENCE_THRESHOLD
            and a.column not in low_confidence
        ):
            low_confidence.append(a.column)

    return ColumnRolesResult(
        institution_id=institution_id,
        dataset=dataset,
        file_kind=response.file_kind,
        file_kind_confidence=response.file_kind_confidence,
        file_kind_rationale=response.file_kind_rationale,
        assignments=response.assignments,
        low_confidence_columns=sorted(set(low_confidence)),
    )
