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

from edvise.genai.mapping.identity_agent.hitl.schemas import (
    HITLItem,
    get_grain_hitl_item_schema_context,
)
from edvise.genai.mapping.identity_agent.utilities import strip_json_fences
from edvise.genai.mapping.identity_agent.profiling import RankedCandidateProfiles

from .schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    GrainContract,
    InstitutionGrainContract,
    get_grain_contract_schema_context,
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

### Term dimension columns (course, semester, any grain that includes term)
Mirror term **batch** normalization when choosing which column is the term dimension
in `post_clean_primary_key` and `join_keys_for_2a`. Apply these preferences **in order**:

1. Prefer columns whose raw values directly encode both season and year in a human-readable
   or parseable string format — e.g. `"Fall 2020"`, `"Spring 2021"`, `"2019FA"` — over opaque
   numeric identifiers such as `"1730"` or `"1192"`, **even when the numeric column appears
   in a shorter or higher-uniqueness candidate key** from the key profile. Opaque numeric
   term columns should only be chosen when no readable alternative exists or the readable
   column cannot safely represent the grain (e.g. excessive nulls, not 1:1 with rows).
2. Among readable columns, prefer columns with fewer nulls / more complete coverage.
3. Among equally readable, equally complete columns, any may be used; prefer the one that
   best matches human-interpretable longitudinal joins.

Note: "coded identifier" means a short, parseable code like `"2020FA"` or `"SP2019"` — not an
opaque integer like `"1700"`. Preferring coded identifiers (in the term stage) means compact parseable
strings over verbose display labels — **not** preferring numeric keys over readable strings.
For grain, do **not** pick an opaque numeric term code when a suitable descriptive or
parseable string column is available for the same enrollment rows.

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
   - When the grain includes a term dimension, choose the term column per **Term dimension
     columns** in DOMAIN PRIORS (readable / parseable term values over opaque numeric codes).

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
   - `"temporal_collapse"`: collapse to one row per key by sorting on a column and keeping
     the first row. Always use `keep="first"` and control direction via `sort_ascending`:
     - Keep **earliest** value: `sort_by="<col>"`, `sort_ascending=true`,  `keep="first"`
     - Keep **latest** value:   `sort_by="<col>"`, `sort_ascending=false`, `keep="first"`
     Never use `keep="last"` — it is ambiguous to reviewers and not a valid resolution target.
     Use when grain is clear but one row per key is needed.
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
   - Use the **same term column** as in `post_clean_primary_key` (chosen per **Term dimension
     columns** in DOMAIN PRIORS).
   - Example: if (student_id, class_number) is unique but term is semantically required,
     emit join_keys_for_2a = [<student column>, class_number, term] using the actual column
     name from the column list for the student slot.

7. Assign numeric `confidence` and `hitl_flag` per **CONFIDENCE SCORING** (next section).

8. Emit `hitl_items` when `hitl_flag` is true
   - Emit one `HITLItem` per distinct ambiguity. A single table may have multiple items
     if independent questions arise (e.g. grain ambiguity + dedup policy).
   - `options`: 2–5 options. Last option must always be `option_id: "custom"` with
     `resolution: null` and `reentry: "generate_hook"`. Use more options when the
     resolution space is genuinely wider — e.g. grain ambiguity cases where
     "keep earliest", "keep latest", and "keep as multi-row" are all meaningful
     and distinct choices. Avoid padding with options that are not meaningfully
     different.
   - Non-custom options must have a non-null `resolution` with concrete `dedup_strategy`,
     `dedup_sort_by`, `dedup_sort_ascending`, and `dedup_keep` values where applicable.
   - Use `reentry: "terminal"` for parameterized resolutions (true_duplicate, temporal_collapse,
     no_dedup). Use `reentry: "generate_hook"` when a custom hook is required.
   - For `policy_required` dedup patterns, options must cover the full resolution
     spectrum — never jump from "keep all rows" directly to "collapse further" without
     offering the middle ground. The three options should be:
       1. Keep as intentionally multi-row (`no_dedup`) — if the table may be a
          legitimate multi-row-per-student table (e.g. semester summary)
       2. Collapse to the semantic grain with a tiebreak on the ambiguous column
          (`temporal_collapse` on the variance column, keeping student-term grain)
       3. `custom` escape hatch
     Do not offer "collapse to student only" as an option unless the grain contract
     already specifies a student-only key. Collapsing further than the semantic grain
     is a destructive decision that requires explicit reviewer intent.
   - For `temporal_collapse` resolutions in HITLItem options, always set:
       - `dedup_sort_ascending`: true = keep earliest value, false = keep latest value
       - `dedup_keep`: always "first" — never "last"
     Both fields are required when `dedup_strategy` is `temporal_collapse`.
     The Pydantic validator will reject the output if either is missing or if
     `dedup_keep` is not "first".

     Example — keep earliest COHORT_YEAR:
     ```json
     {
       "dedup_strategy": "temporal_collapse",
       "dedup_sort_by": "COHORT_YEAR",
       "dedup_sort_ascending": true,
       "dedup_keep": "first"
     }
     ```

     Example — keep latest COHORT_YEAR:
     ```json
     {
       "dedup_strategy": "temporal_collapse",
       "dedup_sort_by": "COHORT_YEAR",
       "dedup_sort_ascending": false,
       "dedup_keep": "first"
     }
     ```
     Two non-custom options for the same sort column should differ only in
     `dedup_sort_ascending` (true vs false), not in `dedup_keep`.
   - Set `hook_group_id` when multiple tables share the same dedup pattern.
   - `hitl_context` must include the specific raw values, uniqueness scores, or variance
     patterns that triggered the flag — give the reviewer the evidence they need without
     requiring them to look at the data.
   - When `hitl_flag` is false, emit `hitl_items: []`.
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


def _identity_output_format() -> str:
    return """
## OUTPUT FORMAT

Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.

Follow **STUDENT IDENTIFIER COLUMN** for `student_id_alias` and for how to name the student id
in `post_clean_primary_key`, `join_keys_for_2a`, and `dedup_policy.sort_by`.

```json
{
  "institution_id": "<institution_id>",
  "table": "<dataset_name>",
  "student_id_alias": "<column name from column list, or null>",
  "post_clean_primary_key": ["<col1>", "<col2>"],
  "dedup_policy": {
    "strategy": "<true_duplicate | temporal_collapse | no_dedup | policy_required>",
    "sort_by": "<column_name or null>",
    "sort_ascending": "<true | false | null — required when strategy is temporal_collapse>",
    "keep": "<\"first\" | \"last\" or null — never any_row; prefer \"first\" with sort_ascending for temporal_collapse>",
    "hook_spec": null,
    "notes": "<brief explanation>"
  },
  "row_selection_required": false,
  "join_keys_for_2a": ["<col1>", "<col2>"],
  "confidence": 0.92,
  "hitl_flag": true,
  "reasoning": "<2-3 sentence summary of the inference chain>",
  "notes": "<optional short notes for reviewers, or empty string>"
}
```

When `hitl_flag` is true, emit HITLItems in the response's top-level `hitl_items` list.
The pipeline writes these to `identity_grain_hitl.json` separately from the grain contracts.

HITLItem shape for grain:

```json
{
  "item_id": "<institution_id>_<table>_<short_descriptor>",
  "institution_id": "<institution_id>",
  "table": "<dataset_name>",
  "domain": "identity_grain",
  "hook_group_id": null,
  "hitl_question": "<specific, actionable question naming the column, values, and decision needed>",
  "hitl_context": "<raw values, uniqueness scores, or variance patterns that triggered this flag>",
  "options": [
    {
      "option_id": "<snake_case_id>",
      "label": "<short label ~4 words>",
      "description": "<one sentence consequence>",
      "resolution": {
        "candidate_key_override": null,
        "dedup_strategy": "<true_duplicate | temporal_collapse | no_dedup>",
        "dedup_sort_by": "<column or null>",
        "dedup_sort_ascending": "<true for earliest | false for latest | null>",
        "dedup_keep": "first",
        "hook_spec": null
      },
      "reentry": "terminal"
    },
    {
      "option_id": "hook_required",
      "label": "Generate custom hook",
      "description": "<one sentence explaining why a hook is needed>",
      "resolution": {
        "candidate_key_override": null,
        "dedup_strategy": null,
        "dedup_sort_by": null,
        "dedup_sort_ascending": null,
        "dedup_keep": null,
        "hook_spec": null
      },
      "reentry": "generate_hook"
    },
    {
      "option_id": "custom",
      "label": "Specify custom handling",
      "description": "Reviewer provides explicit instructions.",
      "resolution": null,
      "reentry": "generate_hook"
    }
  ],
  "target": {
    "institution_id": "<institution_id>",
    "table": "<dataset_name>",
    "config": "grain_contract",
    "field": "dedup_policy"
  },
  "status": "pending",
  "resolution": null
}
```

VALIDITY RULES

- `hitl_flag: true` requires at least one corresponding item in the top-level `hitl_items`.
- `hitl_flag: false` means no items for this table appear in `hitl_items`.
- `confidence < 0.5` requires `hitl_flag: true`.
- Every HITLItem must have 2–5 options. Last option must be `option_id: "custom"`
  with `resolution: null`. Use more options only when the resolution space is
  genuinely wider — avoid padding.
- Non-custom options must have a non-null `resolution`.
- `item_id` must be unique — use `<institution_id>_<table>_<descriptor>`.
"""


def _identity_pydantic_schema_reference() -> str:
    """Inject Pydantic model source (Schema Mapping Agent pattern) for validated JSON shapes."""
    return f"""
## AUTHORITATIVE PYDANTIC MODELS

The following Python class definitions are the contract your JSON must satisfy after parsing
(same pattern as Schema Mapping Agent: model source injected into the prompt). Field names,
types, and nesting must match; do not add extra keys at validated levels.

<grain_contract_schema_reference>
{get_grain_contract_schema_context()}
</grain_contract_schema_reference>

<hitl_item_schema_reference>
{get_grain_hitl_item_schema_context()}
</hitl_item_schema_reference>
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
        + _identity_output_format()
        + "\n---\n"
        + _identity_pydantic_schema_reference().strip()
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


def _grain_payload_as_dict(raw: RawContractInput) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return json.loads(strip_json_fences(text))


def parse_grain_contract_with_hitl(
    raw: RawContractInput,
) -> tuple[GrainContract, list[HITLItem]]:
    """
    Parse grain-stage JSON into :class:`GrainContract` plus structured ``hitl_items``.

    Top-level ``hitl_items`` (if present) is validated as :class:`HITLItem` and removed
    before grain validation. Legacy ``term_config`` is stripped from the payload.
    """
    try:
        d = _grain_payload_as_dict(raw)
    except Exception:
        logger.debug("Grain contract parse failed to load JSON")
        raise
    hitl_raw = d.pop("hitl_items", None)
    if hitl_raw is None:
        hitl_raw = []
    if not isinstance(hitl_raw, list):
        raise ValueError("hitl_items must be a list or null")
    if "term_config" in d:
        d.pop("term_config", None)
        logger.debug(
            "Stripped legacy term_config from grain JSON (use term stage for terms)"
        )
    items = [HITLItem.model_validate(x) for x in hitl_raw]
    try:
        return GrainContract.model_validate(d), items
    except Exception:
        logger.debug(
            "Grain contract validation failed; raw (truncated): %s",
            str(raw)[:500] if not isinstance(raw, dict) else str(raw)[:500],
        )
        raise


def parse_grain_contract(raw: RawContractInput) -> GrainContract:
    """
    Parse and validate IdentityAgent grain-stage JSON into :class:`GrainContract`.

    Strips ``hitl_items`` when present (use :func:`parse_grain_contract_with_hitl` to keep it).
    If a legacy top-level ``term_config`` key is present, it is removed.

    Accepts raw model text (optionally fenced), UTF-8 bytes, or an already-parsed dict.
    """
    gc, _ = parse_grain_contract_with_hitl(raw)
    return gc


def parse_institution_grain_contracts(
    raw: RawContractInput,
) -> InstitutionGrainContract:
    """
    Parse a single JSON file containing ``institution_id`` and a ``datasets`` map of contracts.

    Accepts raw text (optionally fenced), UTF-8 bytes, or an already-parsed dict.
    """
    if isinstance(raw, dict):
        return InstitutionGrainContract.model_validate(raw)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    text = strip_json_fences(text)
    try:
        return InstitutionGrainContract.model_validate_json(text)
    except Exception:
        logger.debug(
            "Institution grain contracts parse failed; raw (truncated): %s", text[:500]
        )
        raise
