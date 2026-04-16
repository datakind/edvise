"""
Term — Prompt assembly for IdentityAgent term normalization (``term_config`` / ``TermOrderConfig``).

Composable sections, explicit builders for system vs user content, and JSON fence stripping
for model output — mirrors :mod:`edvise.genai.mapping.identity_agent.grain_inference.prompt`.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from typing import Any, Union, cast

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    GrainContract,
)
from edvise.genai.mapping.identity_agent.hitl.artifacts import (
    unique_hitl_items_by_item_id,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import (
    HITLItem,
    get_term_hitl_item_schema_context,
)
from edvise.genai.mapping.identity_agent.profiling.schemas import RawTableProfile
from edvise.genai.mapping.identity_agent.utilities import strip_json_fences

from .schemas import (
    InstitutionTermContract,
    TermContract,
    get_term_contract_schema_context,
)

logger = logging.getLogger(__name__)

RawTermPassInput = Union[str, bytes, dict]


# ── System prompt sections ────────────────────────────────────────────────────


def _tn_role_and_inputs() -> str:
    return """
You are IdentityAgent (term), responsible for inferring the term normalization config for a single institution dataset.

You will receive:

- The institution ID and dataset name
- `row_selection_required` from grain inference
- `term_candidates`: columns flagged as likely term columns, with dtype, unique values, and sample values
- `raw_table_profile`: all columns in the table, for context when term candidates are ambiguous or missing

Your job is to produce a `TermOrderConfig` that tells the cleaning layer how to derive standardized columns from the raw term identifier:

- `_year` — 4-digit calendar year (integer)
- `_season` — raw season token as matched from the source (e.g. `FA`, `Spring`)
- `_edvise_term_season` — canonical season label: `FALL`, `SPRING`, `SUMMER`, or `WINTER`
- `_edvise_term_academic_year` — e.g. `"2017-18"`
- `_term_order` — chronological sort key: `year * 100 + season_rank` (rank = 1-indexed position in `season_map`)

`TermOrderConfig` (as a JSON-compatible dict) is consumed by:

- `add_edvise_term_order(df, term_config, year_extractor, season_extractor)` — produces `_year`, `_season`, `_term_order`, `_edvise_term_season`, `_edvise_term_academic_year` (it runs term labels automatically)
- `add_edvise_term_labels(df, term_config)` — use only when `_year` and `_season` already exist; adds `_edvise_term_season`, `_edvise_term_academic_year`
"""


def _tn_when_null() -> str:
    return """
## WHEN TO RETURN NULL

Return `"term_config": null` when:

- No term column exists in the table after inspecting both `term_candidates` and `raw_table_profile`

**Student-grain tables (`row_selection_required` is false):** Do **not** return null solely because
multi-row term ordering is unnecessary. If the table still has a usable **cohort / entry /
matriculation** column, return a non-null `term_config` so the cleaning layer can materialize
`_edvise_term_academic_year`, `_edvise_term_season`, and `_term_order` on **one row per learner**.

- **Term codes / strings:** Prefer scalar term-encoded columns (e.g. `starting_cohort_term`,
  `entry_term_code`, `first_term_at_institution`) whose values use the **same institutional encoding**
  as other datasets (YYYYTT, season+year strings, etc.). Set `term_col` to the **authoritative cohort or
  entry** column from Step 1.
- **Datetime columns:** If the only term-related candidates are datetimes, set `term_col` only on a
  column that represents **start of cohort, entry, or matriculation** (e.g. program start, first
  enrollment start). Do **not** use end dates, graduation dates, or unrelated timestamps as the
  student-grain term column. Use `term_extraction` / `hook_spec` appropriate for datetime (see Step 2).

If **no** column plausibly represents cohort, entry, or matriculation term or start (only demographics,
IDs, unrelated dates, etc.), return `"term_config": null` — do not invent a term config from non-entry
columns.

When `term_candidates` is empty but `row_selection_required` is **true**, inspect `raw_table_profile` for any column that could encode term information before returning null. If still nothing found, return null with `hitl_flag: true`.
"""


def _tn_reasoning_steps() -> str:
    return """
## REASONING STEPS

### Step 1 — Select the authoritative term column

From `term_candidates`, select the most authoritative term column. Prefer:

- Coded identifiers over display labels (e.g. `term_code` over `term_desc`)
- Non-null columns over sparse ones
- Columns that are part of `post_clean_primary_key` from grain inference

**Student-grain (one row per learner):** Prefer the column that defines **cohort / entry /
matriculation start** for Edvise (e.g. `starting_cohort_term`, `cohort_term`, `entry_term`) when it
uses the institution's standard term encoding, even if other term-like columns exist (e.g. expected
graduation term). **Datetime fields:** only choose a datetime if it is clearly **start**-aligned
(entry / cohort / matriculation start), not program end or degree dates. If only one suitable
term-like scalar exists, use it as `term_col`. The goal is one `term_config` whose `season_map`
matches other datasets so `_edvise_term_*` labels align across tables.

If multiple valid term columns exist, select the one most suitable for parsing and note the others.

If no term candidate is suitable, inspect `raw_table_profile.columns` for any overlooked term-like column before giving up.

### Step 2 — Inspect dtype and unique values

Use dtype and `unique_values` (or `sample_values` if `unique_values` is null) to classify the term format.

**Known standard formats** (`term_extraction`: `"standard"`):

- **YYYYTT** — 4-digit year + season code suffix: `"2018FA"`, `"2019SP"`, `"2018S1"` — year extractable via 4-digit regex, season matchable via suffix
- **Season_YYYY** — spelled season + year: `"Fall 2019"`, `"Spring 2021"` — year extractable via 4-digit regex, season matchable via prefix

**Hook-required formats** (`term_extraction`: `"hook_required"`):

- `datetime` or `date` dtype — date-based extraction needed
- Opaque numeric codes — e.g. `"1192"`, `"1199"` with no visible year string or season token
- `float` or `int` dtype with numeric codes — e.g. `1192.0` — treat as opaque numeric

### Step 3 — Build `season_map`

From unique values, identify all distinct season tokens. List them in **chronological order within a calendar year** (not academic year order).

Rules:

- Canonical label must be one of: `FALL`, `SPRING`, `SUMMER`, `WINTER`
- Multiple raw tokens may share the same canonical label (e.g. `S1` and `S2` → `SUMMER`) but must appear as **separate entries** to preserve distinct chronological positions
- Position in the list determines `season_rank` (1-indexed) used for `_term_order`
- For **Season_YYYY** formats, raw tokens are the spelled words as they appear: `"Fall"`, `"Spring"`
- For opaque numeric or date formats where season cannot be observed, set `season_map: []`

Example for YYYYTT:

```json
"season_map": [
    {"raw": "SP", "canonical": "SPRING"},
    {"raw": "S1", "canonical": "SUMMER"},
    {"raw": "S2", "canonical": "SUMMER"},
    {"raw": "FA", "canonical": "FALL"},
    {"raw": "WI", "canonical": "WINTER"}
]
```

### Step 4 — Set `term_extraction` and populate `hook_spec` if needed

Set `term_extraction`: `"standard"` when:

- A 4-digit year is extractable via regex from the raw value, **AND**
- Season tokens are matchable via prefix or suffix against `season_map` keys

Set `term_extraction`: `"hook_required"` when:

- dtype is `datetime` or `date`
- Raw values are opaque numeric codes with no visible year string or season token
- dtype is `float` or `int`

When `term_extraction`: `"hook_required"`, always populate `hook_spec`. Draft extractor functions based on observed patterns in unique values.

For **opaque numeric codes** (e.g. `"1192"` with no visible year in the string):

- Reason about positional structure from unique value samples
- Draft `year_extractor` and `season_extractor` as single Python expressions
- `season_extractor` output must match a `raw` key in `season_map`
- Mark all drafts as requiring human review

For **date columns**:

- `year_extractor`: `pd.to_datetime(term).year`
- `season_extractor`: infer from month bands — 1-4 → Spring, 5-7 → Summer, 8-11 → Fall, 12 → Winter
- `season_map` should reflect the canonical mapping for those month-inferred seasons

### Step 5 — Set `hitl_flag`

Set `hitl_flag`: `true` when:

- `term_extraction`: `"hook_required"` — hook functions require human review before use
- `term_candidates` was empty and term column was inferred from `raw_table_profile`
- Unique values contain unrecognized tokens that could not be mapped to a canonical season
- Confidence in the term column selection is low (multiple ambiguous candidates)

### ACADEMIC YEAR CONVENTION (do not emit — for your reasoning only)

`_edvise_term_academic_year` is derived deterministically by `add_edvise_term_labels`:

- `FALL` or `WINTER` of year N → `"N-(N+1 2-digit)"` e.g. `"2017-18"`
- `SPRING` or `SUMMER` of year N → `"(N-1)-N"` 2-digit e.g. `"2017-18"` when N=2018

You do not need to emit academic year logic — just ensure `season_map` canonical labels are correct.
"""


def _tn_reasoning_steps_batch() -> str:
    return """
## REASONING STEPS

### Step 1 — Select the authoritative term column

From `term_candidates`, select the most authoritative term column. Apply these preferences in order:

1. Prefer columns whose raw values directly encode both season and year in a human-readable or parseable string format — e.g. `"Fall 2020"`, `"Spring 2021"`, `"2019FA"` — over opaque numeric identifiers such as `"1730"` or `"1192"`, even when the numeric column appears in `grain_post_clean_primary_key`. Opaque numeric columns should only be selected when no readable alternative exists.
2. Among readable columns, prefer zero-null columns over sparse ones.
3. Among equally readable, equally complete columns, prefer the one that appears in `grain_post_clean_primary_key`.

Note: "coded identifier" means a short, parseable code like `"2020FA"` or `"SP2019"` — not an opaque integer like `"1700"`. The preference for coded identifiers is a preference for compact parseable strings over verbose display labels, not a preference for numeric keys over readable strings.

If multiple valid term columns exist, select the best one by the rules above and note the others in `reasoning`.
If no term candidate is suitable, inspect `raw_table_profile.columns` for any overlooked term-like column before giving up.

### Step 2 — Inspect dtype and unique values

Use dtype and `unique_values` (or `sample_values` if `unique_values` is null) to classify the term format.

**Known standard formats** (`term_extraction`: `"standard"`):

- **YYYYTT** — 4-digit year + season code suffix: `"2018FA"`, `"2019SP"`, `"2018S1"` — year extractable via 4-digit regex, season matchable via suffix
- **Season_YYYY** — spelled season + year: `"Fall 2019"`, `"Spring 2021"` — year extractable via 4-digit regex, season matchable via prefix

**Hook-required formats** (`term_extraction`: `"hook_required"`):

- `datetime` or `date` dtype — date-based extraction needed
- Opaque numeric codes — e.g. `"1192"`, `"1199"` with no visible year string or season token
- `float` or `int` dtype with numeric codes — e.g. `1192.0` — treat as opaque numeric

### Step 3 — Build `season_map`

From unique values, identify all distinct season tokens. List them in **chronological order within a calendar year** (not academic year order).

Rules:

- Canonical label must be one of: `FALL`, `SPRING`, `SUMMER`, `WINTER`
- Multiple raw tokens may share the same canonical label (e.g. `S1` and `S2` → `SUMMER`) but must appear as **separate entries** to preserve distinct chronological positions
- Position in the list determines `season_rank` (1-indexed) used for `_term_order`
- For **Season_YYYY** formats, raw tokens are the spelled words as they appear: `"Fall"`, `"Spring"`
- For opaque numeric or date formats where season cannot be observed, set `season_map: []`

For opaque numeric or date formats (`term_extraction`: `"hook_required"`): set `season_map: []`.
Do not speculate about raw season tokens or canonical mappings that are not directly observable
as strings in the unique values. The reviewer will supply the confirmed mapping via
`season_map_append` in the HITL resolution. Never add a canonical season that is not evidenced
by the data.

Example for YYYYTT:

```json
"season_map": [
    {"raw": "SP", "canonical": "SPRING"},
    {"raw": "S1", "canonical": "SUMMER"},
    {"raw": "S2", "canonical": "SUMMER"},
    {"raw": "FA", "canonical": "FALL"},
    {"raw": "WI", "canonical": "WINTER"}
]
```

### Step 4 — Set `term_extraction` and populate `hook_spec` if needed

Set `term_extraction`: `"standard"` when:

- A 4-digit year is extractable via regex from the raw value, **AND**
- Season tokens are matchable via prefix or suffix against `season_map` keys

Set `term_extraction`: `"hook_required"` when:

- dtype is `datetime` or `date`
- Raw values are opaque numeric codes with no visible year string or season token
- dtype is `float` or `int`

When `term_extraction`: `"hook_required"`, always populate `hook_spec`. Draft extractor functions based on observed patterns in unique values.

For **opaque numeric codes** (e.g. `"1192"` with no visible year in the string):

- Reason about positional structure from unique value samples
- Draft `year_extractor` and `season_extractor` as single Python expressions
- `season_extractor` output must match a `raw` key in `season_map`
- Mark all drafts as requiring human review

For **date columns**:

- `year_extractor`: `pd.to_datetime(term).year`
- `season_extractor`: infer from month bands — 1-4 → Spring, 5-7 → Summer, 8-11 → Fall, 12 → Winter
- `season_map` should reflect the canonical mapping for those month-inferred seasons

When drafting `season_extractor` for hook-required tables: the function must return the **raw
token as a string** — the same value that will appear as a raw key in `season_map_replace`. Do
not return canonical labels (`FALL`, `SPRING`, …) from the extractor. The cleaning layer looks up
the raw token in `season_map` to get the canonical label — if the extractor returns the canonical
label directly, the lookup fails.

### Step 5 — Set `hitl_flag` and emit `hitl_items`

Set `hitl_flag`: `true` and emit at least one `HITLItem` when any of the following apply:

- `term_extraction`: `"hook_required"` — always, unconditionally, regardless of confidence.
  Hook drafts require human confirmation before execution. Confidence level does not gate this.
- `term_candidates` was empty and term column was inferred from `raw_table_profile`
- Unique values contain unrecognized tokens that could not be mapped to a canonical season
- Confidence in the term column selection is low (multiple ambiguous candidates)

When `hitl_flag` is `true`, emit one `HITLItem` per distinct ambiguity in `hitl_items`.
When `hitl_flag` is `false`, emit `hitl_items: []`.

Each HITLItem must have:

- `hitl_question`: specific and actionable — name the column, the specific values or
  patterns that are ambiguous, and what the reviewer needs to decide.
- `hitl_context`: the raw values or samples that triggered the flag. Give the reviewer
  the evidence they need without requiring them to look at the data.
- `options`: 2–5 options. Last option must always be `option_id: "custom"` with
  `reentry: "generate_hook"`. For `resolution`: use `null` when the reviewer must supply
  everything out-of-band, **or** (preferred when raw→canonical mapping is already clear from
  samples but extractors need rewriting) a **partial** `TermResolution` **without** `hook_spec` —
  e.g. `{"season_map_replace": [...]}` only — so `resolve_items` can persist `season_map` while
  `reviewer_note` + hook generation supply the code. Never put `hook_spec` on `custom`.
  Use more options when the resolution space is genuinely wider — e.g. grain ambiguity cases where
  "keep earliest", "keep latest", and "keep as multi-row" are all meaningful
  and distinct choices. Avoid padding with options that are not meaningfully
  different.
- Non-custom options must always have a non-null `resolution`. For hook-confirmation items,
  the resolution must carry the confirmed `hook_spec` inline (e.g. `{"hook_spec": {...}}`).
  Never set `resolution: null` on a non-custom option, even when `reentry` is `"generate_hook"`.
- `reentry: "terminal"` for parameterized resolutions (exclude_tokens, season_map_append,
  term_col_override). `reentry: "generate_hook"` when a hook is required.
- `hook_group_id`: set to a shared snake_case string when multiple tables share the same
  term encoding e.g. `"shared_term_encoding_a"`. Null for unique encodings.
- `hook_group_tables`: when `hook_group_id` is set, list **every** logical dataset name that
  shares this hook (same strings as keys under `datasets` in `identity_term_output.json`).
  Required whenever one HITL item represents a group so `apply_hook_spec` can update every table.
- When emitting `exclude_tokens` in a TermResolution, always use the shortest
  stable prefix that uniquely identifies the token pattern. Do not enumerate
  year-specific variants.

  Correct:   `"exclude_tokens": ["Custom label"]`
  Incorrect: `"exclude_tokens": ["Custom label 2020-2021", "Custom label 2021-2022", ...]`

  The resolver matches by prefix — all values starting with the token will be
  excluded automatically, including future academic years not yet in the data.

Good `hitl_question` examples:

- "`TERM_DESCR` contains unrecognized tokens `'Custom label 2020-2021'`, `'Custom label 2021-2022'`
  that do not map to a canonical season. Should these rows be excluded from term ordering,
  or mapped to a proxy canonical season?"
- "`STRM` is an opaque int64 column (e.g. 1700, 1730). Year offset logic was inferred from
  samples. Please confirm the extraction rule before hook generation proceeds."

### ACADEMIC YEAR CONVENTION (do not emit — for your reasoning only)

`_edvise_term_academic_year` is derived deterministically by `add_edvise_term_labels`:

- `FALL` or `WINTER` of year N → `"N-(N+1 2-digit)"` e.g. `"2017-18"`
- `SPRING` or `SUMMER` of year N → `"(N-1)-N"` 2-digit e.g. `"2017-18"` when N=2018

You do not need to emit academic year logic — just ensure `season_map` canonical labels are correct.
"""


def _tn_confidence_scoring() -> str:
    t = IDENTITY_CONFIDENCE_HITL_THRESHOLD
    return f"""
## CONFIDENCE SCORING

Use a **number from 0.0 to 1.0** (same scale as Schema Mapping Agent field mappings). In JSON,
`confidence` must be a numeric value, not a string.

- Prefer round scores when possible (e.g. 0.6, 0.7, 0.8, 0.9, 1.0).
- **0.85–1.0**: clear term column, format is standard or unambiguous hook-required extractors
- **above {t} and below 0.85**: workable inference with minor ambiguity
- **at or below {t}**: conflicting signals or policy required → always set `hitl_flag` true

- `hitl_flag` MUST be true whenever `confidence` ≤ {t}. In the mid band (above {t} through below 0.85), set `hitl_flag` true when human review is still required (e.g. hook-required extractors).
"""


def _tn_output_format() -> str:
    t = IDENTITY_CONFIDENCE_HITL_THRESHOLD
    return """
## OUTPUT FORMAT

Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.

Return `"term_config": null` with `hitl_flag: false` and `hitl_items: []` when term config is not needed.

**Standard extraction (no HITL):**

```json
{
    "institution_id": "<institution_id>",
    "table": "<dataset_name>",
    "term_config": {
        "term_col": "<column name>",
        "season_map": [
            {"raw": "<raw token>", "canonical": "<FALL|SPRING|SUMMER|WINTER>"}
        ],
        "term_extraction": "standard",
        "hook_spec": null
    },
    "confidence": 0.9,
    "hitl_flag": false,
    "reasoning": "<2-3 sentence summary of term column selection and format inference>"
}
```

**Custom extraction or unrecognized tokens (HITL required):**

```json
{
    "institution_id": "<institution_id>",
    "table": "<dataset_name>",
    "term_config": {
        "term_col": "<column name>",
        "season_map": [
            {"raw": "<raw token>", "canonical": "<FALL|SPRING|SUMMER|WINTER>"}
        ],
        "term_extraction": "custom",
        "hook_spec": null
    },
    "confidence": 0.6,
    "hitl_flag": true,
    "reasoning": "<2-3 sentence summary>"
}
```

HITL items for flagged tables are emitted in the top-level `hitl_items` list only —
see OUTPUT FORMAT (batch) for the full response shape and HITLItem structure.

VALIDITY RULES

- `hitl_flag: true` requires at least one corresponding item in the top-level `hitl_items`.
- `hitl_flag: false` means no items for this table appear in `hitl_items`.
""" + f"- `confidence` ≤ {t} requires `hitl_flag: true`.\n" + """
- `confidence` must be a numeric float, never a string.
"""


def _tn_pydantic_schema_reference(*, include_institution_envelope: bool) -> str:
    """Inject Pydantic model source (Schema Mapping Agent pattern) for term + HITL JSON shapes."""
    return f"""
## AUTHORITATIVE PYDANTIC MODELS

The following Python class definitions are the contract your JSON must satisfy after parsing
(same pattern as Schema Mapping Agent: model source injected into the prompt). Field names,
types, and nesting must match; do not add extra keys at validated levels.

<term_contract_schema_reference>
{get_term_contract_schema_context(include_institution_envelope=include_institution_envelope)}
</term_contract_schema_reference>

<hitl_item_schema_reference>
{get_term_hitl_item_schema_context()}
</hitl_item_schema_reference>
"""


TERM_NORMALIZATION_SYSTEM_SECTION_KEYS: tuple[str, ...] = (
    "role_and_inputs",
    "when_null",
    "reasoning_steps",
    "confidence_scoring",
    "output_format",
    "pydantic_schema_reference",
)


def get_term_normalization_system_sections() -> dict[str, str]:
    """Named sections of the single-table term normalization system prompt."""
    return {
        "role_and_inputs": _tn_role_and_inputs().strip(),
        "when_null": _tn_when_null(),
        "reasoning_steps": _tn_reasoning_steps(),
        "confidence_scoring": _tn_confidence_scoring(),
        "output_format": _tn_output_format(),
        "pydantic_schema_reference": _tn_pydantic_schema_reference(
            include_institution_envelope=False
        ).strip(),
    }


def join_term_normalization_system_sections(sections: dict[str, str]) -> str:
    parts = [sections[k] for k in TERM_NORMALIZATION_SYSTEM_SECTION_KEYS]
    return parts[0] + "\n\n---\n" + "\n---\n".join(parts[1:])


def build_term_normalization_system_prompt() -> str:
    """Full system prompt for IdentityAgent term stage (term normalization / ``TermOrderConfig``)."""
    return join_term_normalization_system_sections(
        get_term_normalization_system_sections()
    )


TERM_NORMALIZATION_SYSTEM_PROMPT = build_term_normalization_system_prompt()


def _tn_batch_role_and_inputs() -> str:
    return """
You are IdentityAgent (term, **batch**), responsible for inferring term normalization configs
for **every** institution dataset in one response.

You will receive a single JSON object with:

- `institution_id`
- `datasets`: an object whose keys are dataset names. Each value includes:
  - `row_selection_required` (from grain inference)
  - `grain_post_clean_primary_key` (grain key columns, for context)
  - `term_candidates` and `columns` (profiled table metadata)

Apply the same per-table reasoning rules as single-dataset term inference (term column selection,
`season_map`, `term_extraction`, `hook_spec` when hook_required) **independently for each dataset**.

**Cross-table:** When several tables share the same term encoding, set the **same**
``hook_group_id`` on the **HITLItem** and set ``hook_group_tables`` to the full list of dataset
names (map keys) in that group. Do **not** put group membership on ``term_config`` — the HITL
file is the source of truth for which tables share a hook. The resolver applies one generated
``hook_spec`` to every dataset listed. They share one canonical module path (typically
``identity_hooks/<institution_id>/term_hooks.py``). Use **distinct** ``hook_spec.functions`` names
per table only when extraction logic truly differs; when logic is identical, use one shared pair of
names (e.g. ``year_extractor_shared`` / ``season_extractor_shared``) in drafts. Do not merge
distinct encodings into one group.

**Coverage:** Emit exactly one `TermContract`-shaped object per key under `datasets` in the
input. Do not omit datasets.
"""


def _tn_batch_output_format() -> str:
    t = IDENTITY_CONFIDENCE_HITL_THRESHOLD
    return (
        """
## OUTPUT FORMAT (batch)

Respond ONLY with one JSON object. No preamble, no markdown, no explanation outside the JSON.

Top level:

- `institution_id` — same as in the user payload
- `datasets` — object mapping **each** dataset name from the user payload to a full per-table
  contract (same fields as single-table term output).
- `hitl_items` — **only** place HITLItem objects are emitted: the flat canonical list across
  all tables. Empty list when no flags were raised. This is written to a separate file by the
  pipeline. Per-dataset `hitl_items` must always be `[]` — never duplicate HITLItems under each
  table.

Shape:

```json
{
  "institution_id": "<institution_id>",
  "datasets": {
    "<dataset_a>": {
      "institution_id": "<institution_id>",
      "table": "<dataset_a>",
      "term_config": null,
      "confidence": 0.9,
      "hitl_flag": false,
      "reasoning": "<2-3 sentences for this table>",
      "hitl_items": []
    },
    "<dataset_b>": {
      "institution_id": "<institution_id>",
      "table": "<dataset_b>",
      "term_config": {
        "term_col": "<column>",
        "season_map": [{"raw": "<token>", "canonical": "FALL"}],
        "term_extraction": "standard",
        "hook_spec": null
      },
      "confidence": 0.9,
      "hitl_flag": false,
      "reasoning": "<...>",
      "hitl_items": []
    },
    "<dataset_c>": {
      "institution_id": "<institution_id>",
      "table": "<dataset_c>",
      "term_config": {
        "term_col": "<column>",
        "season_map": [{"raw": "<token>", "canonical": "FALL"}],
        "term_extraction": "custom",
        "hook_spec": null
      },
      "confidence": 0.6,
      "hitl_flag": true,
      "reasoning": "<...>",
      "hitl_items": []
    }
  },
  "hitl_items": [
    {
      "item_id": "<institution_id>_<dataset_c>_<short_descriptor>",
      "institution_id": "<institution_id>",
      "table": "<dataset_c>",
      "domain": "identity_term",
      "hook_group_id": null,
      "hook_group_tables": null,
      "hitl_question": "<specific, actionable question>",
      "hitl_context": "<evidence for the reviewer>",
      "options": [
        {
          "option_id": "confirm_extraction",
          "label": "<~4 words>",
          "description": "<one sentence — confirm drafted extractors before hook generation>",
          "resolution": {
            "hook_spec": {
              "file": "identity_hooks/<institution_id>/term_hooks.py",
              "functions": [
                {
                  "name": "year_extractor",
                  "signature": "def year_extractor(term: str) -> int",
                  "description": "<...>",
                  "draft": "<Python expression>"
                },
                {
                  "name": "season_extractor",
                  "signature": "def season_extractor(term: str) -> str",
                  "description": "<...>",
                  "draft": "<Python expression>"
                }
              ]
            }
          },
          "reentry": "generate_hook"
        },
        {
          "option_id": "custom",
          "label": "<...>",
          "description": "Reviewer provides explicit instructions; optional partial season_map_replace if mapping is known.",
          "resolution": null,
          "reentry": "generate_hook"
        }
      ],
      "target": {
        "institution_id": "<institution_id>",
        "table": "<dataset_c>",
        "config": "term_config",
        "field": "hook_spec"
      },
      "choice": null
    }
  ]
}
```

CROSS-TABLE: When multiple tables share the same term encoding, set `hook_group_id` to the
same snake_case string on the HITL item and set `hook_group_tables` to the full list of dataset
names in that group (e.g. `["student", "course", "semester"]`). The pipeline generates one hook
and `apply_hook_spec` fans it out to every listed table.

VALIDITY RULES

- Per-dataset `hitl_items` must always be `[]` — never populate nested lists; emit every
  HITLItem only in the top-level `hitl_items` list.
- `hitl_flag: true` for a dataset requires at least one HITLItem in the top-level `hitl_items`
  whose `"table"` matches that dataset's key.
- `hitl_flag: false` for a dataset means no HITLItem in the top-level list has `"table"`
  matching that dataset's key.
- `confidence` ≤ __PIPELINE_HITL_T__ requires `hitl_flag: true`.
- `term_config: null` requires `reasoning` to explain why.
- `season_map` must only contain tokens directly observable in the unique values as strings.
  For opaque numeric or date term columns, set `season_map: []` — do not infer or speculate
  raw tokens.
- **Hook-required HITL resolution contract — `season_extractor` and `season_map_replace` are two
  halves of the same pipeline and must always appear together in a non-custom option resolution:**
  - `season_extractor` must return a **raw token** (e.g. `"9"`, `"2"`, `"6"`) — not a canonical
    label like `"FALL"`. The raw token is the bridge between the extractor and the season map.
  - `season_map_replace` must map **every** raw token that `season_extractor` can return to a
    canonical label (`FALL`, `SPRING`, `SUMMER`, `WINTER`). The set of raw keys in
    `season_map_replace` must **exactly** match the set of possible return values from
    `season_extractor`.
  - Never return canonical labels directly from `season_extractor` — that bypasses `season_map`
    and breaks `add_edvise_term_order`.
  - Both must appear together in every non-custom hook-required resolution. A resolution with
    `hook_spec` but no `season_map_replace` is incomplete.

  Example (correct — raw tokens in the extractor match `season_map_replace` keys):

```json
"resolution": {
  "hook_spec": {
    "file": "identity_hooks/<institution_id>/term_hooks.py",
    "functions": [
      {
        "name": "year_extractor_<table>",
        "signature": "def year_extractor_<table>(term: str) -> int",
        "description": "Extract year from opaque numeric term code.",
        "draft": "int(str(term)[1:3]) + 2000"
      },
      {
        "name": "season_extractor_<table>",
        "signature": "def season_extractor_<table>(term: str) -> str",
        "description": "Extract raw season token from term code. Returns one of: '9', '2', '6'.",
        "draft": "str(term)[3:]"
      }
    ]
  },
  "season_map_replace": [
    {"raw": "2", "canonical": "SPRING"},
    {"raw": "6", "canonical": "SUMMER"},
    {"raw": "9", "canonical": "FALL"}
  ]
}
```

  The raw values in `season_map_replace` (`"2"`, `"6"`, `"9"`) exactly match what
  `season_extractor` returns. The resolver writes `season_map_replace` to `term_config.season_map`
  and writes `hook_spec` to `term_config.hook_spec` in one atomic operation.

  Optional — `custom` + `generate_hook` with mapping known but extractors disputed (no `hook_spec` on `custom`):

```json
"option_id": "custom",
"resolution": {
  "season_map_replace": [
    {"raw": "2", "canonical": "SPRING"},
    {"raw": "6", "canonical": "SUMMER"},
    {"raw": "9", "canonical": "FALL"}
  ]
},
"reentry": "generate_hook"
```
- When `term_extraction` is `"hook_required"`, `term_config.hook_spec` must always be populated
  — it is the draft. Draft the extractor functions inline from observed value patterns; do not
  defer hook drafting to HITL resolution. The HITL item exists to get human confirmation of that
  draft, not to store or replace it. When multiple tables share an encoding, populate identical
  `hook_spec` drafts on each table's `term_config`, set the same `hook_group_id` on the HITL item,
  and list every dataset in `hook_group_tables` — the resolver fans out the confirmed spec to
  those tables automatically.
- `term_extraction`: `"hook_required"` always requires `hitl_flag`: `true` and at least one
  `HITLItem` in the top-level `hitl_items` list. This is unconditional — confidence level does
  not gate HITL emission for hook-required tables.
- `confidence` must be a numeric float, never a string.
- Every HITLItem must have 2–5 options. Last option must be `option_id: "custom"`.
  Its `resolution` is `null` **or** a partial `TermResolution` without `hook_spec` (e.g.
  `season_map_replace` only) when `reentry` is `"generate_hook"` — see Step 5. Use more options
  only when the resolution space is genuinely wider — avoid padding.
- Non-custom options (any `option_id` other than `"custom"`) must always carry a non-null
  `resolution` object with concrete field mutations. For hook-confirmation items (`reentry`:
  `"generate_hook"`), the resolution must include the confirmed `hook_spec` inline — do not
  leave `resolution: null` on a non-custom option.
- `item_id` must be unique across the entire response.
- Each nested object under `datasets` must set `"table"` to the same string as its key in `datasets`.
"""
        .replace("__PIPELINE_HITL_T__", f"{t}")
    )


TERM_NORMALIZATION_BATCH_SYSTEM_SECTION_KEYS: tuple[str, ...] = (
    "batch_role_and_inputs",
    "when_null",
    "reasoning_steps_batch",
    "confidence_scoring",
    "batch_output_format",
    "pydantic_schema_reference",
)


def get_term_normalization_batch_system_sections() -> dict[str, str]:
    """Named sections of the batch term normalization system prompt."""
    return {
        "batch_role_and_inputs": _tn_batch_role_and_inputs().strip(),
        "when_null": _tn_when_null(),
        "reasoning_steps_batch": _tn_reasoning_steps_batch(),
        "confidence_scoring": _tn_confidence_scoring(),
        "batch_output_format": _tn_batch_output_format(),
        "pydantic_schema_reference": _tn_pydantic_schema_reference(
            include_institution_envelope=True
        ).strip(),
    }


def join_term_normalization_batch_system_sections(sections: dict[str, str]) -> str:
    parts = [sections[k] for k in TERM_NORMALIZATION_BATCH_SYSTEM_SECTION_KEYS]
    return parts[0] + "\n\n---\n" + "\n---\n".join(parts[1:])


def build_term_normalization_batch_system_prompt() -> str:
    """System prompt for the term stage when all datasets are inferred in one LLM call."""
    return join_term_normalization_batch_system_sections(
        get_term_normalization_batch_system_sections()
    )


TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT = build_term_normalization_batch_system_prompt()


def _user_message_template() -> str:
    return """
Institution: {institution_id}
Dataset: {dataset}
Row selection required: {row_selection_required}

Term candidate columns:
{term_candidates_json}

Full column list (for context):
{raw_table_profile_columns_json}
"""


TERM_NORMALIZATION_USER_TEMPLATE = _user_message_template()


def get_term_normalization_user_sections(
    institution_id: str,
    dataset: str,
    row_selection_required: bool,
    *,
    term_candidates_json: str,
    raw_table_profile_columns_json: str,
) -> dict[str, str]:
    """Named sections of the term user message (profile JSON dominates size)."""
    rs = json.dumps(row_selection_required)
    return {
        "header": (
            f"Institution: {institution_id}\nDataset: {dataset}\n"
            f"Row selection required: {rs}"
        ),
        "term_candidates": "\n\nTerm candidate columns:\n" + term_candidates_json,
        "full_columns": "\n\nFull column list (for context):\n"
        + raw_table_profile_columns_json,
    }


def build_term_normalization_user_message(
    institution_id: str,
    dataset: str,
    row_selection_required: bool,
    *,
    term_candidates_json: str,
    raw_table_profile_columns_json: str,
) -> str:
    """
    Build the user message body for IdentityAgent term stage.

    Parameters
    ----------
    term_candidates_json, raw_table_profile_columns_json
        Pre-serialized JSON strings (e.g. from ``json.dumps(..., indent=2)``). The term stage does not
        prescribe the exact shape beyond what the system prompt describes; callers typically pass
        serialized :class:`~edvise.genai.mapping.identity_agent.profiling.schemas.RawColumnProfile` lists
        or similar.
    """
    return TERM_NORMALIZATION_USER_TEMPLATE.format(
        institution_id=institution_id,
        dataset=dataset,
        row_selection_required=json.dumps(row_selection_required),
        term_candidates_json=term_candidates_json,
        raw_table_profile_columns_json=raw_table_profile_columns_json,
    )


def audit_term_normalization_prompt(
    institution_id: str,
    dataset: str,
    row_selection_required: bool,
    *,
    term_candidates_json: str,
    raw_table_profile_columns_json: str,
    log: bool = True,
    batch_system: bool = False,
) -> dict[str, Any]:
    """
    Local estimated token counts for term normalization (single-table user message).

    Set ``batch_system=True`` to measure the **batch** system prompt instead of the single-table one
    (user message is unchanged).
    """
    from edvise.genai.mapping.shared.token_audit.prompt_token_audit import audit_prompt_sections

    if batch_system:
        sys_sections = get_term_normalization_batch_system_sections()
    else:
        sys_sections = get_term_normalization_system_sections()
    user_sections = get_term_normalization_user_sections(
        institution_id,
        dataset,
        row_selection_required,
        term_candidates_json=term_candidates_json,
        raw_table_profile_columns_json=raw_table_profile_columns_json,
    )
    combined: dict[str, str] = {f"system.{k}": v for k, v in sys_sections.items()}
    combined.update({f"user.{k}": v for k, v in user_sections.items()})
    return audit_prompt_sections(
        combined,
        builder="identity_agent.term_normalization"
        + (".batch_system" if batch_system else ".single_system"),
        institution_id=institution_id,
        dataset_name=dataset,
        log=log,
    )


def audit_term_normalization_batch_user_prompt(
    institution_id: str,
    grain_contracts_by_dataset: Mapping[str, GrainContract],
    run_by_dataset: Mapping[str, Mapping[str, object]],
    *,
    log: bool = True,
) -> dict[str, Any]:
    """
    Local estimated token counts for batch term inference (system sections + one user JSON payload).

    Sections: ``system.*`` and ``user.batch_payload``.
    """
    from edvise.genai.mapping.shared.token_audit.prompt_token_audit import audit_prompt_sections

    sys_sections = get_term_normalization_batch_system_sections()
    user_text = build_term_normalization_batch_user_message_from_grain_and_profiles(
        institution_id, grain_contracts_by_dataset, run_by_dataset
    )
    combined = {f"system.{k}": v for k, v in sys_sections.items()}
    combined["user.batch_payload"] = user_text
    return audit_prompt_sections(
        combined,
        builder="identity_agent.term_normalization.batch_full",
        institution_id=institution_id,
        log=log,
    )


def build_term_normalization_user_message_from_profiles(
    institution_id: str,
    dataset: str,
    row_selection_required: bool,
    raw_table_profile: RawTableProfile,
) -> str:
    """
    Convenience: serialize term candidates and full column list from a raw table profile.
    """
    term_candidates_json = json.dumps(
        [c.model_dump(mode="json") for c in raw_table_profile.term_candidates],
        indent=2,
    )
    raw_table_profile_columns_json = json.dumps(
        [c.model_dump(mode="json") for c in raw_table_profile.columns],
        indent=2,
    )
    return build_term_normalization_user_message(
        institution_id,
        dataset,
        row_selection_required,
        term_candidates_json=term_candidates_json,
        raw_table_profile_columns_json=raw_table_profile_columns_json,
    )


def build_term_normalization_batch_user_payload(
    institution_id: str,
    grain_contracts_by_dataset: Mapping[str, GrainContract],
    run_by_dataset: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """
    Build the JSON-serializable user payload for batch term inference.

    ``run_by_dataset`` must contain ``raw_table_profile`` per dataset name (as produced by the
    profiling notebook cell).
    """
    datasets: dict[str, object] = {}
    for name, gc in grain_contracts_by_dataset.items():
        if name not in run_by_dataset:
            raise KeyError(
                f"Dataset {name!r} missing from run_by_dataset "
                f"(have {list(run_by_dataset.keys())!r})"
            )
        row = run_by_dataset[name]
        rtp = row["raw_table_profile"]
        if not isinstance(rtp, RawTableProfile):
            raise TypeError(
                f"run_by_dataset[{name!r}]['raw_table_profile'] must be RawTableProfile"
            )
        datasets[name] = {
            "row_selection_required": gc.row_selection_required,
            "grain_post_clean_primary_key": list(gc.post_clean_primary_key),
            "term_candidates": [c.model_dump(mode="json") for c in rtp.term_candidates],
            "columns": [c.model_dump(mode="json") for c in rtp.columns],
        }
    return {"institution_id": institution_id, "datasets": datasets}


def build_term_normalization_batch_user_message_from_grain_and_profiles(
    institution_id: str,
    grain_contracts_by_dataset: Mapping[str, GrainContract],
    run_by_dataset: Mapping[str, Mapping[str, object]],
) -> str:
    """Serialize :func:`build_term_normalization_batch_user_payload` for the LLM user message."""
    payload = build_term_normalization_batch_user_payload(
        institution_id, grain_contracts_by_dataset, run_by_dataset
    )
    return json.dumps(payload, indent=2)


def _term_payload_as_dict(raw: RawTermPassInput) -> dict:
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    text = strip_json_fences(text)
    return cast(dict[str, Any], json.loads(text))


def _strip_term_batch_hitl_payload(d: dict) -> tuple[dict, list[HITLItem]]:
    """Remove ``hitl_items`` from top level and per-dataset entries; validate items."""
    d = copy.deepcopy(d)
    collected: list[HITLItem] = []
    top = d.pop("hitl_items", None)
    if top:
        collected.extend(HITLItem.model_validate(x) for x in top)
    datasets = d.get("datasets")
    if isinstance(datasets, dict):
        for _k, v in datasets.items():
            if not isinstance(v, dict):
                continue
            nested = v.pop("hitl_items", None)
            if nested:
                collected.extend(HITLItem.model_validate(x) for x in nested)
    return d, unique_hitl_items_by_item_id(collected)


def parse_institution_term_contracts_with_hitl(
    raw: RawTermPassInput,
) -> tuple[InstitutionTermContract, list[HITLItem]]:
    """
    Parse batch term-stage JSON into :class:`InstitutionTermContract` plus ``hitl_items``.

    Strips ``hitl_items`` from the payload before contract validation.
    """
    try:
        d = _term_payload_as_dict(raw)
        d2, items = _strip_term_batch_hitl_payload(d)
        return InstitutionTermContract.model_validate(d2), items
    except Exception:
        text = raw if isinstance(raw, str) else str(raw)[:500]
        logger.debug(
            "Institution term contract parse failed; raw (truncated): %s", text
        )
        raise


def parse_institution_term_contracts(raw: RawTermPassInput) -> InstitutionTermContract:
    """
    Parse batch term-stage JSON into :class:`InstitutionTermContract`.

    Strips ``hitl_items`` when present (use :func:`parse_institution_term_contracts_with_hitl`
    to retain them). Accepts raw model text (optionally fenced), UTF-8 bytes, or a dict.
    """
    inst, _ = parse_institution_term_contracts_with_hitl(raw)
    return inst


def parse_term_normalization_pass_output(
    raw: RawTermPassInput,
) -> TermContract:
    """
    Parse and validate IdentityAgent term-stage JSON into :class:`TermContract`.

    Accepts raw model text (optionally fenced), UTF-8 bytes, or an already-parsed dict.
    """
    if isinstance(raw, dict):
        return TermContract.model_validate(raw)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    text = strip_json_fences(text)
    try:
        return TermContract.model_validate_json(text)
    except Exception:
        logger.debug(
            "Term normalization output parse failed; raw (truncated): %s", text[:500]
        )
        raise
