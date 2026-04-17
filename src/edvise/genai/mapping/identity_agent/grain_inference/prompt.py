"""
Step 2 — Prompt assembly for IdentityAgent (grain contract inference).

Mirrors the style of ``schema_mapping_agent.manifest.prompts``: composable
sections, explicit builders for system vs user content, and JSON fence stripping
for model output.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Union, cast

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

### Multi-column variance (honors, sub_plan, program, etc.)

When two or more non-temporal, non-measure columns have variance within the same candidate key:

- Do NOT use `no_dedup` as a catch-all. You must still choose a collapse strategy or declare
  the grain is wider.
- Prioritize **institutional semantic meaning**:
  - `honors` (Cum Laude, Magna, Summa, etc.) is an **award attribute**, not a grain dimension.
    Collapse on honors (keep highest distinction) rather than including it in the grain.
  - `sub_plan` (PADHRMGT, PADMGTOP, etc.) is a **program track or concentration**. If students
    can be enrolled in multiple tracks per program-term, it belongs in the grain. If it's an
    attribute of a single enrollment, collapse it.
  - `program` / `acad_prog_primary` (UGRD, GRAD, MPA) — if duplicates differ on this, decide:
    is the grain student-term (collapse to one career) or student-term-career (keep multi-row)?
- When uncertain about a column's semantic role, **collapse via tiebreak on the highest-priority
  column, flag HITL, and let the reviewer confirm**.

### Degree / award / completion tables
- Expected grain is multi-dimensional: student (or learner) identifier, program context
  (e.g. major or program) when present, term or completion cohort when present, **and**
  a column that distinguishes **which credential or degree** when the institution can award
  more than one per student/program/term (e.g. `awarded_degree`, `degree_type`, certificate
  vs associate). Do **not** satisfy uniqueness by adding measure or elapsed-time columns
  to the key when a degree-type or credential column exists or belongs in the grain.
- If multiple rows per (student, major, term) reflect different awards (AA vs AS, certificate
  vs degree), the grain is either intentionally multi-row at that key or must include a
  degree/credential dimension — FLAG for HITL when the collapse policy is unclear.
- row_selection is often required when the table remains multi-row per student after cleaning.

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

   a) **Temporal columns (term, date, semester):**
      High variance → grain is under-specified, that column belongs in the key.
      Example: (student, program) with 40% variance on term → grain should be
      (student, program, term). Update post_clean_primary_key.

   b) **Single non-temporal, non-measure column with variance:**
      Grain ambiguity. Decide: is this column part of the grain or a tiebreak?
      Example: (student, program, term) with 60% variance on honors →
      - Is honors part of the grain? (unlikely; honors describes the award, not the logical row)
      - Then collapse via honors tiebreak (keep highest distinction)
      - Or declare grain is intentionally wider at (student, program, term, honors) and flag HITL

   c) **MULTIPLE non-temporal, semi-dimensional columns with variance (e.g., honors AND sub_plan):**
      This is grain-width ambiguity. You must decide — do NOT use `no_dedup`:

      **Option i) All variance is noise; grain is as identified; collapse via tiebreak on ONE column**
         - Choose the column with clearest business semantics (e.g., honors over sub_plan
           if honors is institutional priority).
         - Accept that you're losing the other column's distinctions.
         - Example: (sid, plan, term, degree) is the grain; collapse on honors descending
           (keep Summa > Magna > Cum > none); sub_plan distinctions are dropped.
         - Use `temporal_collapse` with `sort_by="honors"`, `sort_ascending=false`.

      **Option ii) One or both columns belong in the grain (table is multi-row by design)**
         - Declare the grain is wider: e.g., (sid, plan, term, degree, sub_plan) where honors
           is collapsed, or (sid, plan, term, degree, honors) where sub_plan is collapsed.
         - You're still collapsing one variance column; you're just widening the grain for the other.
         - Example: grain = (sid, plan, term, degree, honors); collapse on sub_plan ascending.

      **Option iii) Ambiguous; need human review**
         - Flag HITL with explicit options showing what each choice drops/widens.
         - Never default to `no_dedup` when multiple columns have variance.

   d) **Zero variance across all columns:**
      True duplicates, safe to drop.

   e) **Mixed variance across measure columns only (gpa, credits, grade):**
      Competing values, business rule needed for which row to keep. Flag for HITL.

3. Apply domain priors (see above) — these override data inference when they conflict.

4. Determine dedup policy — PRIORITY: prefer collapse to clean grain over multi-row ambiguity

   **Validity check:** If the candidate key has `non_unique_rows` > 0, you CANNOT use `no_dedup`.
   Use `temporal_collapse`, `true_duplicate`, or `policy_required` instead.

   Use exactly one of these string literals for `dedup_policy.strategy`:

   - `"true_duplicate"`: within-group variance = 0 across all columns — drop all but one
   - `"temporal_collapse"`: collapse to one row per key by sorting on a column and keeping first.
     Always use `keep="first"` and control direction via `sort_ascending`:
     - Keep **earliest** value: `sort_by="<col>"`, `sort_ascending=true`, `keep="first"`
     - Keep **latest** value: `sort_by="<col>"`, `sort_ascending=false`, `keep="first"`
     Never use `keep="last"`.
   - `"no_dedup"`: table is intentionally multi-row AND has ZERO duplicates on the semantic grain.
     This is only valid when `non_unique_rows` = 0 for the candidate key.
   - `"policy_required"`: grain is ambiguous or collapse rule is unclear; human must decide
     before cleaning runs. Use when `hitl_flag` is true and the ambiguity is in the collapse
     decision (e.g. student-program cohort policy). The executor will refuse to run cleaning
     until HITL resolves this and updates the contract.

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
     (when `learner_id_alias` is non-null, that name — not the literal string `student_id`).
   - Use the **same term column** as in `post_clean_primary_key` (chosen per **Term dimension
     columns** in DOMAIN PRIORS).
   - Example: if (student_id, class_number) is unique but term is semantically required,
     emit join_keys_for_2a = [<student column>, class_number, term] using the actual column
     name from the column list for the student slot.

7. Assign numeric `confidence` and `hitl_flag` per **CONFIDENCE SCORING** (next section).

8. Emit `hitl_items` when `hitl_flag` is true

   - Emit one `HITLItem` per distinct ambiguity. A single table may have multiple items
     if independent questions arise (e.g. grain ambiguity + dedup policy).

   **For multi-column variance (honors + sub_plan, etc.):**
     Do not emit a single `no_dedup` option. Instead, emit 3–4 concrete options:

     Option 1: "Collapse on Column A (keep highest/first); drop Column B distinctions"
       - `dedup_strategy: "temporal_collapse"`
       - `dedup_sort_by: "column_A"`
       - `dedup_sort_ascending: true/false` (based on semantics)
       - `dedup_keep: "first"`
       - `candidate_key_override: null` (grain stays as identified)

     Option 2: "Collapse on Column B (keep X); drop Column A distinctions"
       - `dedup_strategy: "temporal_collapse"`
       - `dedup_sort_by: "column_B"`
       - `dedup_sort_ascending` / `dedup_keep: "first"` per semantics
       - Alternative tiebreak for the ambiguous columns

     Option 3 (if grain-widening is plausible): "Grain includes Column A; collapse Column B"
       - `candidate_key_override: ["student", "term", ..., "column_A"]` (wider grain)
       - `dedup_strategy: "temporal_collapse"`
       - `dedup_sort_by: "column_B"`
       - `dedup_keep: "first"`
       - Shows that you're keeping Column A distinctions by widening the grain

     Option 4: "Specify custom handling"
       - `resolution: null`
       - `reentry: "generate_hook"`

     Each option's `description` must explicitly state what is **collapsed/dropped** and what
     is **preserved/widened** (e.g., "Keep highest honors; drop sub_plan distinctions").

   - `options`: 2–5 options. Last option must always be `option_id: "custom"` with
     `resolution: null` and `reentry: "generate_hook"`. Use more options when the
     resolution space is genuinely wider — e.g. grain ambiguity cases where
     "keep earliest", "keep latest", and (when `non_unique_rows` = 0) "keep as multi-row
     without dedup" are all meaningful and distinct choices. Avoid padding with options
     that are not meaningfully different.
   - Non-custom options must have a non-null `resolution` with concrete `dedup_strategy`,
     `dedup_sort_by`, `dedup_sort_ascending`, and `dedup_keep` values where applicable.
   - Use `reentry: "terminal"` for parameterized resolutions (true_duplicate, temporal_collapse,
     no_dedup). Use `reentry: "generate_hook"` when a custom hook is required.
   - For `policy_required` dedup patterns, options must cover the full resolution
     spectrum — never jump from "keep all rows" directly to "collapse further" without
     offering the middle ground. Typical options:
       1. Keep as intentionally multi-row (`no_dedup`) — **only** when `non_unique_rows` = 0
          for the candidate key (no duplicate rows at the semantic grain). If
          `non_unique_rows` > 0, **do not** offer `no_dedup`; offer collapse strategies instead.
       2. Collapse to the semantic grain with a tiebreak on the ambiguous column
          (`temporal_collapse` on the variance column, keeping the proposed grain)
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
   - `hitl_context` must give the reviewer the evidence that triggered the flag: either a short
     freeform string, **or** a structured object with `candidate_keys` (ranked columns,
     uniqueness scores, notes) and optional `variance_profile` (column → variance summary).
     Prefer the structured form when key-profiling output is available.
   - **Structured `candidate_keys` ordering (reasoning step, not the profiler):** the key
     profile lists candidates by **profiling heuristics** (uniqueness-first). When you emit
     `hitl_context.candidate_keys`, **replace that ordering entirely**: sort **every** entry
     into **descending order of likely semantic grain** (most plausible identifier + temporal
     grain first, then next-best, and so on). Assign `rank` **1..n** to match that order
     so **all** ranks reflect semantic likelihood, not the profiler's uniqueness rank.
     `rank` **1** is your best semantic candidate (identifiers + time dimensions; measure
     columns in the key only when unavoidable and called out in `notes`). **Its `columns`
     must match** the grain you already emitted in **`post_clean_primary_key`** for this
     table in the **same JSON response** (same columns as the contract grain; use the same
     ordering you used there). **`rank` 2..n** list **alternative** grain keys the reviewer
     might adopt via `candidate_key_override` on an option — not a second guess of rank 1.
     Preserve each candidate's `uniqueness_score` from the profile for the same `columns`
     list. You may mention the profiler's original ordering in `notes` if it helps reviewers
     (e.g. "Profiler rank was 2").
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
- **above {t} and below 0.85**: data inference is clear but domain prior doesn't fully apply, or minor variance
- **at or below {t}**: conflicting signals, ambiguous grain, or policy decision required → always set
  `hitl_flag` true

- `hitl_flag` MUST be true whenever `confidence` ≤ {t}. In the mid band (above {t} through below 0.85), set
  `hitl_flag` true when a policy choice is still required.
"""


def _identity_student_id_and_keys() -> str:
    return """
## LEARNER IDENTIFIER COLUMN (`learner_id_alias` vs primary keys and join keys)

Distinguish these two ideas:

1. **`learner_id_alias`** — The institution's learner/student-identifier column **as it appears in the
   column list** you receive (header-normalized, typically snake_case). Examples:
   `student_id_randomized_datakind`, or a normalized form of a raw header like `STUDENT_ID`.
   Set to JSON `null` when the column list already shows `student_id`, or when this dataset's
   grain does not include a person identifier.

2. **`post_clean_primary_key`**, **`join_keys_for_2a`**, and **`dedup_policy.sort_by`** — Where the
   grain includes the learner identifier, use **that same column name** (the alias string when
   `learner_id_alias` is non-null). Do **not** substitute the literal string `student_id` in those
   arrays unless the column list already uses `student_id`. This keeps the contract, dedup key,
   and join keys **consistent with the dataframe column names before the canonical rename** to `student_id`.

3. **Downstream cleaning** maps `learner_id_alias` to canonical `student_id` **once**, as part of
   the cleaning pass **after** grain dedup and term-order hooks. Your JSON should describe the
   pre-rename names so execution stays consistent (schema contracts and SMA use learner-oriented naming).

**Inference:** From the column list and key profile, decide which column is the learner
identifier; set `learner_id_alias` accordingly; emit keys in `post_clean_primary_key` /
`join_keys_for_2a` / `sort_by` using that name wherever the learner id participates in the grain.
"""


def _identity_output_format() -> str:
    t = IDENTITY_CONFIDENCE_HITL_THRESHOLD
    return """
## OUTPUT FORMAT

Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.

Follow **LEARNER IDENTIFIER COLUMN** for `learner_id_alias` and for how to name the learner id
in `post_clean_primary_key`, `join_keys_for_2a`, and `dedup_policy.sort_by`.

```json
{
  "institution_id": "<institution_id>",
  "table": "<dataset_name>",
  "learner_id_alias": "<column name from column list, or null>",
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
  "hitl_context": "<short freeform evidence string>",
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

When key-profiling output is available, use structured `hitl_context` instead of a string (same
options/target shape as above; only `hitl_context` changes):

`candidate_keys`: **fully re-ranked** by **semantic grain plausibility** (see **REASONING STEPS**
and **HITL** bullets above)—`rank` **1** **matches `post_clean_primary_key`** from this response;
**2..n** are **override** candidates. This is **not** the profiler's uniqueness-first order.

```json
{
  "item_id": "<institution_id>_<table>_<short_descriptor>",
  "institution_id": "<institution_id>",
  "table": "<dataset_name>",
  "domain": "identity_grain",
  "hook_group_id": null,
  "hook_group_tables": null,
  "hitl_question": "<specific question>",
  "hitl_context": {
    "candidate_keys": [
      {
        "rank": 1,
        "columns": ["STUDENT_ID", "TERM_DESC"],
        "uniqueness_score": 0.85,
        "notes": "<optional caveat — e.g. profiler ranked this #2; semantic grain>"
      }
    ],
    "variance_profile": {
      "COHORT_YEAR": "25%–58.8% within groups",
      "TERM_DESC": "41.2%–62% within groups"
    }
  },
  "options": [ "..." ],
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
- `no_dedup` is only valid when `non_unique_rows` = 0 for the candidate key in the key profile.
  If any candidate key has `non_unique_rows` > 0, use `temporal_collapse`, `true_duplicate`,
  or `policy_required` — never `no_dedup`.
""" + f"- `confidence` ≤ {t} requires `hitl_flag: true`.\n" + """
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


IDENTITY_SYSTEM_SECTION_KEYS: tuple[str, ...] = (
    "role_and_inputs",
    "domain_priors",
    "reasoning_steps",
    "confidence_scoring",
    "student_id_and_keys",
    "output_format",
    "pydantic_schema_reference",
)


def get_identity_agent_system_sections() -> dict[str, str]:
    """Named sections of the grain IdentityAgent system prompt (token audit / inspection)."""
    return {
        "role_and_inputs": _identity_role_and_inputs().strip(),
        "domain_priors": _identity_domain_priors(),
        "reasoning_steps": _identity_reasoning_steps(),
        "confidence_scoring": _identity_confidence_scoring(),
        "student_id_and_keys": _identity_student_id_and_keys().strip(),
        "output_format": _identity_output_format(),
        "pydantic_schema_reference": _identity_pydantic_schema_reference().strip(),
    }


def join_identity_agent_system_sections(sections: dict[str, str]) -> str:
    """Join system sections with the same delimiters as the original monolithic builder."""
    keys = IDENTITY_SYSTEM_SECTION_KEYS
    parts = [sections[k] for k in keys]
    return parts[0] + "\n\n---\n" + "\n---\n".join(parts[1:])


def build_identity_agent_system_prompt() -> str:
    """Full system prompt for IdentityAgent (grain contract)."""
    return join_identity_agent_system_sections(get_identity_agent_system_sections())


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


def get_identity_agent_user_sections(
    institution_id: str,
    dataset_name: str,
    *,
    column_list: str,
    key_profile_json: str,
) -> dict[str, str]:
    """Named sections of the grain user message (variable-size vs static instructions)."""
    return {
        "institution_and_dataset": f"Institution ID: {institution_id}\nDataset: {dataset_name}",
        "column_list_block": (
            f"\n\nColumn list (name: dtype, one per line):\n{column_list}\n"
        ),
        "key_profile_json": (
            "\nKey profile JSON (`RankedCandidateProfiles` — ranked candidate keys, uniqueness, "
            "within-group variance):\n" + key_profile_json
        ),
    }


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
    assert resolved_columns is not None
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


def audit_identity_agent_prompt(
    institution_id: str,
    dataset_name: str,
    key_profile: RankedCandidateProfiles,
    *,
    column_list: str | None = None,
    df: pd.DataFrame | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """
    Local estimated token counts for grain inference (system + user sections).

    Uses ``len(text) // chars_per_token`` (see :mod:`edvise.genai.mapping.shared.token_audit.prompt_token_audit`).
    """
    from edvise.genai.mapping.shared.token_audit.prompt_token_audit import audit_prompt_sections

    if df is not None and column_list is not None:
        raise ValueError("Pass only one of column_list or df")
    if df is None and column_list is None:
        raise ValueError("Provide exactly one of column_list or df")
    resolved_columns = format_column_list(df) if df is not None else column_list
    assert resolved_columns is not None
    key_profile_json = json.dumps(
        key_profile.model_dump(mode="json"),
        indent=2,
    )
    sys_sections = get_identity_agent_system_sections()
    user_sections = get_identity_agent_user_sections(
        institution_id,
        dataset_name,
        column_list=resolved_columns,
        key_profile_json=key_profile_json,
    )
    combined: dict[str, str] = {f"system.{k}": v for k, v in sys_sections.items()}
    combined.update({f"user.{k}": v for k, v in user_sections.items()})
    return audit_prompt_sections(
        combined,
        builder="identity_agent.grain_inference",
        institution_id=institution_id,
        dataset_name=dataset_name,
        log=log,
    )


def _grain_payload_as_dict(raw: RawContractInput) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return cast(dict[str, Any], json.loads(strip_json_fences(text)))


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
