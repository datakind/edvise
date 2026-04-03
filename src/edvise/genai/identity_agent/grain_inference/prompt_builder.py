"""
Step 2 — Prompt assembly for IdentityAgent (grain contract inference).

Mirrors the style of ``schema_mapping_agent.manifest.prompt_builder``: composable
sections, explicit builders for system vs user content, and JSON fence stripping
for model output.
"""

from __future__ import annotations

import json
import logging
from typing import Union

import pandas as pd

from edvise.genai.identity_agent.grain_contract.schemas import IdentityGrainContract
from edvise.genai.identity_agent.profiling.key_profiler import KeyProfile

logger = logging.getLogger(__name__)

RawContractInput = Union[str, bytes, dict]


# ── System prompt sections ────────────────────────────────────────────────────


def _identity_role_and_inputs() -> str:
    return """
You are IdentityAgent, responsible for inferring the grain contract for a single institution
dataset. You will receive:
  - The dataset name and institution ID
  - The full column list with dtypes (one column per line: `name: dtype`)
  - A key profile: JSON from `KeyProfile` — ranked `candidate_key_profiles` with uniqueness
    scores and, for each candidate key, within-group variance for non-key columns on non-unique rows

Your job is to produce a grain contract that downstream agents (SchemaMappingAgent Step 2a)
will use to understand:
  1. The post-clean primary key — what uniquely identifies one row after cleaning
     (this maps to schema contract `unique_keys` for this table).
  2. Whether row_selection is required in 2a (i.e. the table is intentionally multi-row per student)
  3. Safe join keys for joining this table to other tables
  4. Whether deduplication is needed, and if so what policy applies
  5. Any flags requiring human review before the contract is confirmed
"""


def _identity_domain_priors() -> str:
    return """
## DOMAIN PRIORS — apply these before reasoning from data

### Student / demographic tables
- Expected grain is ONE ROW PER STUDENT after cleaning.
- If the natural key from profiling includes a non-temporal dimension beyond student_id
  (e.g. program, major, cohort_year, degree), FLAG this immediately.
  The cohort base policy requires a human to decide whether to collapse to student-only
  and how to handle the dropped dimension (keep first enrollment, keep most recent, etc.).
- Do NOT silently collapse a student-program key to student-only without flagging.

### Course / enrollment-detail tables
- Expected grain is (student_id, course_identifier, term).
- If a candidate key achieves high uniqueness on (student_id, course_identifier) WITHOUT term,
  this likely means the institution uses globally unique class numbers across terms
  (i.e. term is implicit in the class number). FLAG this explicitly.
  The semantic grain contract should ALWAYS include term as a dimension even if dedup
  does not require it — 2a needs term for longitudinal field resolution and join safety.
- row_selection IS required on course tables. Do not recommend collapsing to student grain.

### Semester / term-summary tables
- Expected grain is (student_id, term).
- True duplicates on this key (within-group variance = 0 across all columns) should be
  dropped — these are system artifacts, not meaningful records.
- row_selection IS required on semester tables.

### Consistency
- For course and semester tables: `cleaning_collapses_to_student_grain` must be false and
  `row_selection_required` must be true.
- For student/demographic tables intended at student grain: typically
  `cleaning_collapses_to_student_grain` is true and `row_selection_required` is false
  after cleaning.
"""


def _identity_reasoning_steps() -> str:
    return """
## REASONING STEPS

1. Identify the best candidate key
   - Prefer the shortest key with the highest uniqueness score that includes student_id.
   - Prefer keys with meaningful semantic columns (term, class_number) over keys that
     achieve uniqueness by adding measure columns (grade, gpa, section_size).
   - A key is "meaningful" if its columns are identifiers or time dimensions, not outcomes.

2. Interpret within-group variance on non-unique rows
   - High variance on a TEMPORAL column (term, semester, date) → grain is under-specified,
     that column belongs in the key.
   - High variance on a NON-TEMPORAL, NON-MEASURE column (program, major, cohort) →
     grain ambiguity requiring human policy decision. FLAG for HITL.
   - Zero variance across all columns → true duplicates, safe to drop.
   - Mixed variance across measure columns only (gpa, credits, grade) → competing values,
     business rule needed for which row to keep. FLAG for HITL.

3. Apply domain priors (see above) — these override data inference when they conflict.

4. Determine dedup policy
   - true_duplicate: drop all but one row (any_row is safe)
   - temporal_collapse: keep earliest / latest / specific row by a sort column — specify which
   - no_dedup: table is intentionally multi-row, cleaning should not collapse it

   For `dedup_policy.strategy`, use exactly one of these string literals (not a pipe list):
   `"true_duplicate"`, `"temporal_collapse"`, or `"no_dedup"`.

5. Set row_selection_required
   - True if the post-clean table remains multi-row per student (course, semester tables)
   - False if cleaning collapses to one row per student (student/demo tables)
   - When True, 2a is permitted and expected to use first_by / where_not_null strategies
   - When False, 2a should use any_row only — flag if it attempts otherwise

6. Determine join keys for 2a
   - Always include the full semantic grain as join keys, even if a subset achieves uniqueness.
   - Example: if (student_id, class_number) is unique but term is semantically required,
     emit join_keys_for_2a = [student_id, class_number, term].

7. Assign confidence and flag for HITL if needed
   - HIGH: all signals agree, domain prior confirms, zero ambiguity
   - MEDIUM: data inference is clear but domain prior doesn't fully apply, or minor variance
   - LOW: conflicting signals, ambiguous grain, policy decision required → always FLAG
   - `hitl_flag` MUST be true whenever `confidence` is LOW. For MEDIUM, set `hitl_flag` true
     when a policy choice is still required.
"""


def _identity_output_format() -> str:
    return """
## OUTPUT FORMAT

Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.

{
  "institution_id": "<institution_id>",
  "table": "<dataset_name>",
  "post_clean_primary_key": ["<col1>", "<col2>"],
  "dedup_policy": {
    "strategy": "<one of: true_duplicate | temporal_collapse | no_dedup>",
    "sort_by": "<column_name or null>",
    "keep": "<first | last or null>",
    "notes": "<brief explanation>"
  },
  "cleaning_collapses_to_student_grain": true,
  "row_selection_required": false,
  "join_keys_for_2a": ["<col1>", "<col2>"],
  "term_order_column": "<column name or null — if set, executor runs add_term_order after dedup>",
  "confidence": "HIGH | MEDIUM | LOW",
  "hitl_flag": true,
  "hitl_question": "<specific question for human reviewer, or null if no flag>",
  "reasoning": "<2-3 sentence summary of the inference chain>"
}
"""


def build_identity_agent_system_prompt() -> str:
    """Full system prompt for IdentityAgent (grain contract)."""
    return (
        _identity_role_and_inputs().strip()
        + "\n\n---\n"
        + _identity_domain_priors()
        + "\n---\n"
        + _identity_reasoning_steps()
        + "\n---\n"
        + _identity_output_format()
    )


IDENTITY_AGENT_SYSTEM_PROMPT = build_identity_agent_system_prompt()


def _user_message_template() -> str:
    return """
Institution ID: {institution_id}
Dataset: {dataset_name}

Column list (name: dtype, one per line):
{column_list}

Key profile JSON (`KeyProfile` — ranked candidate keys, uniqueness, within-group variance):
{key_profile_json}
"""


IDENTITY_AGENT_USER_TEMPLATE = _user_message_template()


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from JSON text (same idea as SMA ``strip_json_fences``)."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :]
    if text.endswith("```"):
        text = text[: text.rindex("```")].rstrip()
    return text


def format_column_list(df: pd.DataFrame) -> str:
    """Format columns as `name: dtype` lines for the IdentityAgent user prompt."""
    lines = [f"  {col}: {df[col].dtype}" for col in df.columns]
    return "\n".join(lines)


def build_identity_agent_user_message(
    institution_id: str,
    dataset_name: str,
    key_profile: KeyProfile,
    *,
    column_list: str | None = None,
    df: pd.DataFrame | None = None,
) -> str:
    """
    Build the user message body for IdentityAgent.

    Pass exactly one of ``column_list`` (pre-formatted) or ``df`` (columns inferred).
    """
    if df is not None and column_list is not None:
        raise ValueError("Pass only one of column_list or df")
    if df is None and column_list is None:
        raise ValueError("Provide exactly one of column_list or df")
    resolved_columns = format_column_list(df) if df is not None else column_list
    key_profile_json = json.dumps(
        key_profile.model_dump(mode="json"),
        indent=2,
    )
    return IDENTITY_AGENT_USER_TEMPLATE.format(
        institution_id=institution_id,
        dataset_name=dataset_name,
        column_list=resolved_columns,
        key_profile_json=key_profile_json,
    )


def parse_identity_grain_contract(raw: RawContractInput) -> IdentityGrainContract:
    """
    Parse and validate IdentityAgent JSON output into ``IdentityGrainContract``.

    Accepts raw model text (optionally fenced), UTF-8 bytes, or an already-parsed dict.
    """
    if isinstance(raw, dict):
        return IdentityGrainContract.model_validate(raw)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    text = strip_json_fences(text)
    try:
        return IdentityGrainContract.model_validate_json(text)
    except Exception:
        logger.debug("Identity grain contract parse failed; raw (truncated): %s", text[:500])
        raise
