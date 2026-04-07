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

from edvise.genai.identity_agent.profiling import RankedCandidateProfiles

from .schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    IdentityGrainContract,
    InstitutionGrainContracts,
)

logger = logging.getLogger(__name__)

RawContractInput = Union[str, bytes, dict]


# ── System prompt sections ────────────────────────────────────────────────────


def _identity_role_and_inputs() -> str:
    return """
You are IdentityAgent, responsible for inferring the grain contract for a single institution
dataset. You will receive:
  - The dataset name and institution ID
  - The full column list with dtypes (one column per line: `name: dtype`)
  - A key profile: JSON from `RankedCandidateProfiles` — ranked `candidate_key_profiles` with uniqueness
    scores and, for each candidate key, within-group variance for non-key columns on non-unique rows

Your job is to produce a grain contract that downstream agents (SchemaMappingAgent Step 2a)
will use to understand:
  1. The grain primary key — which columns uniquely identify one logical row (see
     **STUDENT IDENTIFIER COLUMN** for how to name the student id column; maps to schema
     contract `unique_keys` after canonical cleaning).
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
- For course and semester tables: `row_selection_required` must be true.
- For student/demographic tables intended at one row per student after cleaning: typically
  `row_selection_required` is false (2a may use `any_row` on that table where appropriate).
"""


def _identity_reasoning_steps() -> str:
    return """
## REASONING STEPS

### Mandatory: non-unique rows vs. uniqueness scores
- Do **not** treat a candidate key as clean **only** because `uniqueness_score` is 1.0 or very
  close to 1.0.
- If `non_unique_rows` > 0 for that candidate key, you **must** inspect `within_group_variance`
  and apply the flag / grain logic in step 2 before concluding the key is acceptable — however
  few those rows are.
- A key with **2** non-unique rows is **not** the same as a fully unique key with zero collisions.

1. Identify the best candidate key
   - Prefer the shortest key with the highest uniqueness score that includes the student
     identifier column (use the same column name as in the column list — see
     **STUDENT IDENTIFIER COLUMN** when it is not literally `student_id`).
   - Prefer keys with meaningful semantic columns (term, class_number) over keys that
     achieve uniqueness by adding measure or count columns (grade, gpa, section_size,
     credits_earned, years_earned, flags).
   - A key is "meaningful" if its columns are identifiers or time dimensions, not outcomes
     or measures.
   - IMPORTANT: If candidate keys ranked #1 or #2 achieve uniqueness_score=1.0 but include
     measure or count columns (e.g. gpa, credits, years_earned, graduation flags), IGNORE
     those candidates. Locate the shortest key composed only of identifier and temporal
     columns — even if it ranks lower and has non_unique_rows > 0 — and treat that as the
     primary grain candidate.

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
   Use exactly one of these string literals for `dedup_policy.strategy`:

   - `"true_duplicate"`: all non-key columns are identical across duplicate rows — drop all
     but one (any_row is safe after dedup). Use when within-group variance = 0.
   - `"temporal_collapse"`: keep earliest / latest row by a sort column — specify `sort_by`
     and `keep` ("first" or "last"). Use when grain is clear but one row per key is needed.
   - `"no_dedup"`: table is intentionally multi-row per student — cleaning should not
     collapse it. Use for course and semester tables.
   - `"policy_required"`: a collapse is needed but the rule cannot be determined from data
     alone — a human must specify the policy before cleaning runs. Use when `hitl_flag` is
     true and the grain ambiguity is in the collapse decision itself (e.g. student-program
     tables where the cohort base policy is unresolved). The executor will refuse to run
     cleaning until HITL resolves this and updates the contract.

   For `dedup_policy.keep`, use only `"first"`, `"last"`, or JSON `null`.
   **Never** put `any_row` here — `any_row` is a row_selection strategy in SchemaMappingAgent
   Step 2a, not a dedup `keep` value.

5. Set row_selection_required
   - True if the post-clean table remains multi-row per student (course, semester tables)
   - False if the table is one row per student after cleaning (typical student/demo tables)
   - When True, 2a is permitted and expected to use first_by / where_not_null strategies
   - When False, 2a should use any_row only — flag if it attempts otherwise

6. Determine join keys for 2a
   - Always include the full semantic grain as join keys, even if a subset achieves uniqueness.
   - Use the **same student-identifier column name** as in `post_clean_primary_key` / column list
     (when `student_id_alias` is non-null, that name — not the literal string `student_id`).
   - Example: if (student_id, class_number) is unique but term is semantically required,
     emit join_keys_for_2a = [<student column>, class_number, term] using the actual column
     name from the column list for the student slot.

7. Assign numeric `confidence` and `hitl_flag` per **CONFIDENCE SCORING** (next section).
"""


def _identity_confidence_scoring() -> str:
    t = IDENTITY_CONFIDENCE_HITL_THRESHOLD
    return f"""
## CONFIDENCE SCORING

Use a **number from 0.0 to 1.0** (same scale as Schema Mapping Agent field mappings). In JSON,
`confidence` must be a numeric value, not a string.

- Prefer round scores when possible (e.g. 0.6, 0.7, 0.8, 0.9, 1.0).
- **0.85–1.0**: all signals agree, domain prior confirms, zero ambiguity
- **{t}–0.85**: data inference is clear but domain prior doesn't fully apply, or minor variance
- **0.0–{t}**: conflicting signals, ambiguous grain, or policy decision required → always set
  `hitl_flag` true

- `hitl_flag` MUST be true whenever `confidence` < {t}. In the mid band ({t}–0.85), set
  `hitl_flag` true when a policy choice is still required.
"""


def _identity_student_id_and_keys() -> str:
    return """
## STUDENT IDENTIFIER COLUMN (`student_id_alias` vs primary keys and join keys)

Distinguish these two ideas:

1. **`student_id_alias`** — The institution's student-identifier column **as it appears in the
   column list** you receive (header-normalized, typically snake_case). Examples:
   `student_id_randomized_datakind`, or a normalized form of a raw header like `STUDENT_ID`.
   Set to JSON `null` when the column list already shows `student_id`, or when this dataset's
   grain does not include a student identifier.

2. **`post_clean_primary_key`**, **`join_keys_for_2a`**, and **`dedup_policy.sort_by`** — Where the
   grain includes the student identifier, use **that same column name** (the alias string when
   `student_id_alias` is non-null). Do **not** substitute the literal string `student_id` in those
   arrays unless the column list already uses `student_id`. This keeps the contract, dedup key,
   and join keys **consistent with the dataframe column names before the canonical rename**.

3. **Downstream cleaning** maps `student_id_alias` to canonical `student_id` **once**, as part of
   the cleaning pass **after** grain dedup and term-order hooks. Your JSON should describe the
   pre-rename names so execution stays consistent.

**Inference:** From the column list and key profile, decide which column is the student
identifier; set `student_id_alias` accordingly; emit keys in `post_clean_primary_key` /
`join_keys_for_2a` / `sort_by` using that name wherever the student id participates in the grain.
"""


def _identity_term_config() -> str:
    return """
## TERM CONFIG

### When to emit term_config
Set `term_config: null` when no term column exists or term ordering is not needed for this table.

When a term column is present and `row_selection_required` is true, always populate `term_config`.

### Step 1 — Identify the term column
Inspect column names and unique values for a column encoding academic term or semester.
Common names: `term`, `term_desc`, `term_descr`, `semester`, `strm`, `acad_year`, `term_code`, `compl_term`.
If multiple term-like columns exist, select the most semantically authoritative one (prefer
coded identifiers over display labels; prefer non-null columns).

### Step 2 — Inspect unique values and dtype
Use the unique values and dtype provided in the column profile to reason about format.

Known formats:
- `YYYYTT` — 4-digit year + season code suffix: `"2018FA"`, `"2019SP"`, `"2018S1"` → `term_extraction: "standard"`
- `Season_YYYY` — spelled season + year: `"Fall 2019"`, `"Spring 2021"` → `term_extraction: "standard"`
- `datetime` dtype — date column, no season codes visible → `term_extraction: "custom"` (date-based extraction needed)
- Opaque numeric codes — e.g. `"1192"`, `"1199"` with no visible year or season string → `term_extraction: "custom"`
- Float/int dtype with numeric codes — e.g. `1192.0` → treat as opaque numeric, `term_extraction: "custom"`

### Step 3 — Build season_map
From unique values, identify all distinct season tokens. List them in **chronological order within a calendar year** (not academic year order). Assign a canonical label to each.

Canonical labels must be one of: `FALL`, `SPRING`, `SUMMER`, `WINTER`.

Multiple raw tokens may share the same canonical label (e.g. `S1` and `S2` both map to `SUMMER`), but must appear as separate entries in `season_map` to preserve their distinct chronological positions.

Example:
```json
"season_map": [
    {"raw": "SP", "canonical": "SPRING"},
    {"raw": "S1", "canonical": "SUMMER"},
    {"raw": "S2", "canonical": "SUMMER"},
    {"raw": "FA", "canonical": "FALL"},
    {"raw": "WI", "canonical": "WINTER"}
]
```

For `Season_YYYY` formats, raw tokens are the spelled season words as they appear in the data:
```json
"season_map": [
    {"raw": "Spring", "canonical": "SPRING"},
    {"raw": "Summer", "canonical": "SUMMER"},
    {"raw": "Fall", "canonical": "FALL"}
]
```

For opaque numeric or date formats where season cannot be observed from values, set `season_map: []` and `term_extraction: "custom"`.

### Step 4 — Assess extractability and set term_extraction
Set `term_extraction: "standard"` when:
- A 4-digit year is extractable via regex from the raw value, AND
- Season tokens are matchable via prefix or suffix against `season_map` keys

Set `term_extraction: "custom"` when:
- The dtype is `datetime` or `date`
- The raw values are opaque numeric codes with no visible year string or season token
- The dtype is float or int (numeric codes that require coercion before parsing)

### Step 5 — Populate hook_spec when term_extraction == "custom"
When `term_extraction: "custom"`, always populate `hook_spec`. The LLM should draft the extractor functions based on observed patterns in unique values.

For **opaque numeric codes** (e.g. CUNY `1192`):
- Reason about positional structure from unique value samples
- Draft `year_extractor` and `season_extractor` as single-expression Python
- `season_extractor` output must match a `raw` key in `season_map`

For **date columns**:
- `year_extractor`: `lambda t: pd.to_datetime(t).year`
- `season_extractor`: infer from month bands (1-4 → Spring, 5-7 → Summer, 8-11 → Fall, 12 → Winter)
- `season_map` should reflect the canonical mapping for those month-inferred seasons

All draft functions are marked as drafts requiring human review before use.

### Step 6 — term_config null check
If `row_selection_required` is false and the table is student-grain (one row per student), `term_config` is typically null unless a term column is explicitly part of the grain key.
"""


def _identity_output_format() -> str:
    return """
## OUTPUT FORMAT

Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.

Follow **STUDENT IDENTIFIER COLUMN** for `student_id_alias` and for how to name the student id
in `post_clean_primary_key`, `join_keys_for_2a`, and `dedup_policy.sort_by`.

Follow **TERM CONFIG** for `term_config` (use JSON `null` when no term column or ordering is not needed).

{
  "institution_id": "<institution_id>",
  "table": "<dataset_name>",
  "student_id_alias": "<column name from column list, or null>",
  "post_clean_primary_key": ["<col1>", "<col2>"],
  "dedup_policy": {
    "strategy": "<true_duplicate | temporal_collapse | no_dedup | policy_required>",
    "sort_by": "<column_name or null>",
    "keep": "<\"first\" | \"last\" or null — never any_row>",
    "notes": "<brief explanation>"
  },
  "row_selection_required": false,
  "join_keys_for_2a": ["<col1>", "<col2>"],
  "confidence": 0.92,
  "hitl_flag": true,
  "hitl_question": "<specific question for human reviewer, or null if no flag>",
  "reasoning": "<2-3 sentence summary of the inference chain>",
  "notes": "<optional short notes for reviewers, or empty string>",
  "term_config": {
    "term_col": "<column name>",
    "season_map": [
        {"raw": "<raw token>", "canonical": "<FALL|SPRING|SUMMER|WINTER>"}
    ],
    "term_extraction": "<standard|custom>",
    "hook_spec": null
  }
}

Or for custom:
```json
"term_config": {
    "term_col": "<column name>",
    "season_map": [],
    "term_extraction": "custom",
    "hook_spec": {
        "file": "pipelines/<institution_id>/helpers/term_hooks.py",
        "functions": [
            {
                "name": "year_extractor",
                "signature": "def year_extractor(term: str) -> int",
                "description": "<what it does>",
                "example_input": "<value>",
                "example_output": "<int>",
                "draft": "<expression>"
            },
            {
                "name": "season_extractor",
                "signature": "def season_extractor(term: str) -> str",
                "description": "<what it does>",
                "example_input": "<value>",
                "example_output": "<raw token>",
                "draft": "<expression>"
            }
        ]
    }
}
```
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
        + _identity_confidence_scoring()
        + "\n---\n"
        + _identity_student_id_and_keys().strip()
        + "\n---\n"
        + _identity_term_config().strip()
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

Key profile JSON (`RankedCandidateProfiles` — ranked candidate keys, uniqueness, within-group variance):
{key_profile_json}
"""


IDENTITY_AGENT_USER_TEMPLATE = _user_message_template()


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from JSON text."""
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
    key_profile: RankedCandidateProfiles,
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
        logger.debug(
            "Identity grain contract parse failed; raw (truncated): %s", text[:500]
        )
        raise


def parse_institution_grain_contracts(
    raw: RawContractInput,
) -> InstitutionGrainContracts:
    """
    Parse a single JSON file containing ``institution_id`` and a ``datasets`` map of contracts.

    Accepts raw text (optionally fenced), UTF-8 bytes, or an already-parsed dict.
    """
    if isinstance(raw, dict):
        return InstitutionGrainContracts.model_validate(raw)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    text = strip_json_fences(text)
    try:
        return InstitutionGrainContracts.model_validate_json(text)
    except Exception:
        logger.debug(
            "Institution grain contracts parse failed; raw (truncated): %s", text[:500]
        )
        raise
