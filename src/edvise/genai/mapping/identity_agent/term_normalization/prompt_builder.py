"""
Term — Prompt assembly for IdentityAgent term normalization (``term_config`` / ``TermOrderConfig``).

Composable sections, explicit builders for system vs user content, and JSON fence stripping
for model output — mirrors :mod:`edvise.genai.mapping.identity_agent.grain_inference.prompt_builder`.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from typing import Union

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    GrainContract,
)
from edvise.genai.mapping.identity_agent.hitl.artifacts import unique_hitl_items_by_item_id
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLItem, get_hitl_item_schema_context
from edvise.genai.mapping.identity_agent.profiling.schemas import RawTableProfile
from edvise.genai.mapping.identity_agent.utilities import strip_json_fences

from .schemas import InstitutionTermContract, TermContract, get_term_contract_schema_context

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

- `row_selection_required` is **false** — term ordering is not needed for student-grain tables
- No term column exists in the table after inspecting both `term_candidates` and `raw_table_profile`

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

For **opaque numeric codes** (e.g. CUNY `"1192"`):

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

For **opaque numeric codes** (e.g. CUNY `"1192"`):

- Reason about positional structure from unique value samples
- Draft `year_extractor` and `season_extractor` as single Python expressions
- `season_extractor` output must match a `raw` key in `season_map`
- Mark all drafts as requiring human review

For **date columns**:

- `year_extractor`: `pd.to_datetime(term).year`
- `season_extractor`: infer from month bands — 1-4 → Spring, 5-7 → Summer, 8-11 → Fall, 12 → Winter
- `season_map` should reflect the canonical mapping for those month-inferred seasons

### Step 5 — Set `hitl_flag` and emit `hitl_items`

Set `hitl_flag`: `true` when any of the following apply:

- `term_extraction`: `"custom"` — hook functions require human review before use
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
- `options`: exactly 2–3 options. Last option must always be `option_id: "custom"` with
  `resolution: null` and `reentry: "generate_hook"`.
- Non-custom options must have a non-null `resolution` with concrete field mutations.
- `reentry: "terminal"` for parameterized resolutions (exclude_tokens, season_map_append,
  term_col_override). `reentry: "generate_hook"` when a hook is required.
- `hook_group_id`: set to a shared snake_case string when multiple tables share the same
  term encoding e.g. `"jjc_term_format_a"`. Null for unique encodings.

Good `hitl_question` examples:

- "`TERM_DESCR` contains unrecognized tokens `'Med Year 2020-2021'`, `'Med Year 2021-2022'`
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
- **{t}–0.85**: workable inference with minor ambiguity
- **0.0–{t}**: conflicting signals or policy required → always set `hitl_flag` true

- `hitl_flag` MUST be true whenever `confidence` < {t}. In the mid band, set `hitl_flag` true when human review is still required (e.g. hook-required extractors).
"""


def _tn_output_format() -> str:
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
- `confidence < 0.5` requires `hitl_flag: true`.
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
{get_hitl_item_schema_context()}
</hitl_item_schema_reference>
"""


def build_term_normalization_system_prompt() -> str:
    """Full system prompt for IdentityAgent term stage (term normalization / ``TermOrderConfig``)."""
    return (
        _tn_role_and_inputs().strip()
        + "\n\n---\n"
        + _tn_when_null()
        + "\n---\n"
        + _tn_reasoning_steps()
        + "\n---\n"
        + _tn_confidence_scoring()
        + "\n---\n"
        + _tn_output_format()
        + "\n---\n"
        + _tn_pydantic_schema_reference(include_institution_envelope=False).strip()
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

**Cross-table:** When several tables share the same term encoding, you may reuse one
`hook_spec.file` path in `term_config`, but use **distinct function names** inside
`hook_spec.functions` per table when the extractors differ. Do not merge distinct encodings.

**Coverage:** Emit exactly one `TermContract`-shaped object per key under `datasets` in the
input. Do not omit datasets.
"""


def _tn_batch_output_format() -> str:
    return """
## OUTPUT FORMAT (batch)

Respond ONLY with one JSON object. No preamble, no markdown, no explanation outside the JSON.

Top level:

- `institution_id` — same as in the user payload
- `datasets` — object mapping **each** dataset name from the user payload to a full per-table
  contract (same fields as single-table term output).
- `hitl_items` — flat list of all HITLItem objects across all tables. Empty list when no
  flags were raised. This is written to a separate file by the pipeline.

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
      "hitl_items": ["<see HITLItem shape in single-table output format>"]
    }
  },
  "hitl_items": ["<flat list — all HITLItems across all tables, same objects as nested above>"]
}
```

CROSS-TABLE: When multiple tables share the same term encoding, set `hook_group_id` to the
same snake_case string on all related HITLItems e.g. `"jjc_term_format_a"`. The pipeline
will generate one hook and fan it out to all tables in the group.

VALIDITY RULES

- `hitl_flag: true` requires at least one item in the table's `hitl_items`.
- `hitl_flag: false` requires `hitl_items: []` for that table.
- `confidence < 0.5` requires `hitl_flag: true`.
- `term_config: null` requires `reasoning` to explain why.
- `confidence` must be a numeric float, never a string.
- Every HITLItem must have exactly 2–3 options. Last option must be `option_id: "custom"`.
- `item_id` must be unique across the entire response.
- Top-level `hitl_items` must contain exactly the same objects as the per-table `hitl_items`
  combined — no duplicates, no omissions.
- Each nested object must set `"table"` to the same string as its key in `datasets`.
"""


def build_term_normalization_batch_system_prompt() -> str:
    """System prompt for the term stage when all datasets are inferred in one LLM call."""
    return (
        _tn_batch_role_and_inputs().strip()
        + "\n\n---\n"
        + _tn_when_null()
        + "\n---\n"
        + _tn_reasoning_steps_batch()
        + "\n---\n"
        + _tn_confidence_scoring()
        + "\n---\n"
        + _tn_batch_output_format()
        + "\n---\n"
        + _tn_pydantic_schema_reference(include_institution_envelope=True).strip()
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
    return json.loads(text)


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
        logger.debug("Institution term contract parse failed; raw (truncated): %s", text)
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
