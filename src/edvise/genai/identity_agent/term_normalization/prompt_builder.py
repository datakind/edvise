"""
Pass 2 — Prompt assembly for IdentityAgent term normalization (``term_config`` / ``TermOrderConfig``).

Composable sections, explicit builders for system vs user content, and JSON fence stripping
for model output — mirrors :mod:`edvise.genai.identity_agent.grain_inference.prompt_builder`.
"""

from __future__ import annotations

import json
import logging
from typing import Union

from edvise.genai.identity_agent.grain_inference.schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
)
from edvise.genai.identity_agent.profiling.schemas import RawTableProfile

from .schemas import TermContract, TermOrderConfig

logger = logging.getLogger(__name__)

RawTermPassInput = Union[str, bytes, dict]


# ── System prompt sections ────────────────────────────────────────────────────


def _tn_role_and_inputs() -> str:
    return """
You are IdentityAgent (Pass 2), responsible for inferring the term normalization config for a single institution dataset.

You will receive:

- The institution ID and dataset name
- `row_selection_required` from Pass 1 grain inference
- `term_candidates`: columns flagged as likely term columns, with dtype, unique values, and sample values
- `raw_table_profile`: all columns in the table, for context when term candidates are ambiguous or missing

Your job is to produce a `TermOrderConfig` that tells the cleaning layer how to derive three standardized columns from the raw term identifier:

- `_edvise_term_season` — canonical season label: `FALL`, `SPRING`, `SUMMER`, or `WINTER`
- `_edvise_term_year` — 4-digit integer year
- `_edvise_term_academic_year` — e.g. `"2017-18"`
- `_term_order` — chronological sort key: `year * 100 + season_rank` (rank = 1-indexed position in `season_map`)

These are produced by two functions that consume `TermOrderConfig` (as a JSON-compatible dict):

- `add_edvise_term_order(df, term_config, year_extractor, season_extractor)` — produces `_year`, `_season`, `_term_order`
- `add_edvise_term_labels(df, term_config)` — produces `_edvise_term_season`, `_edvise_term_year`, `_edvise_term_academic_year`
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
- Columns that are part of `post_clean_primary_key` from Pass 1

If multiple valid term columns exist, select the one most suitable for parsing and note the others.

If no term candidate is suitable, inspect `raw_table_profile.columns` for any overlooked term-like column before giving up.

### Step 2 — Inspect dtype and unique values

Use dtype and `unique_values` (or `sample_values` if `unique_values` is null) to classify the term format.

**Known standard formats** (`term_extraction`: `"standard"`):

- **YYYYTT** — 4-digit year + season code suffix: `"2018FA"`, `"2019SP"`, `"2018S1"` — year extractable via 4-digit regex, season matchable via suffix
- **Season_YYYY** — spelled season + year: `"Fall 2019"`, `"Spring 2021"` — year extractable via 4-digit regex, season matchable via prefix

**Custom formats** (`term_extraction`: `"custom"`):

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

Set `term_extraction`: `"custom"` when:

- dtype is `datetime` or `date`
- Raw values are opaque numeric codes with no visible year string or season token
- dtype is `float` or `int`

When `term_extraction`: `"custom"`, always populate `hook_spec`. Draft extractor functions based on observed patterns in unique values.

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

- `term_extraction`: `"custom"` — hook functions require human review before use
- `term_candidates` was empty and term column was inferred from `raw_table_profile`
- Unique values contain unrecognized tokens that could not be mapped to a canonical season
- Confidence in the term column selection is low (multiple ambiguous candidates)

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
- **0.85–1.0**: clear term column, format is standard or unambiguous custom hooks
- **{t}–0.85**: workable inference with minor ambiguity
- **0.0–{t}**: conflicting signals or policy required → always set `hitl_flag` true

- `hitl_flag` MUST be true whenever `confidence` < {t}. In the mid band, set `hitl_flag` true when human review is still required (e.g. custom hooks).
"""


def _tn_output_format() -> str:
    return """
## OUTPUT FORMAT

Respond ONLY with a JSON object. No preamble, no markdown, no explanation outside the JSON.

Return `"term_config": null` with `hitl_flag: false` when term config is not needed.

**Standard extraction:**

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
    "hitl_question": null,
    "reasoning": "<2-3 sentence summary of term column selection and format inference>"
}
```

**Custom extraction:**

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
        "hook_spec": {
            "file": "pipelines/<institution_id>/helpers/term_hooks.py",
            "functions": [
                {
                    "name": "year_extractor",
                    "signature": "def year_extractor(term: str) -> int",
                    "description": "<what it does>",
                    "example_input": "<raw value from unique_values>",
                    "example_output": "<expected int year>",
                    "draft": "<single Python expression>"
                },
                {
                    "name": "season_extractor",
                    "signature": "def season_extractor(term: str) -> str",
                    "description": "<what it does>",
                    "example_input": "<raw value from unique_values>",
                    "example_output": "<raw token matching a 'raw' key in season_map>",
                    "draft": "<single Python expression>"
                }
            ]
        }
    },
    "confidence": 0.6,
    "hitl_flag": true,
    "hitl_question": "<specific question for human reviewer describing what needs to be validated in the draft hook functions>",
    "reasoning": "<2-3 sentence summary>"
}
```
"""


def build_term_normalization_system_prompt() -> str:
    """Full system prompt for IdentityAgent Pass 2 (term normalization / ``TermOrderConfig``)."""
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
    )


TERM_NORMALIZATION_SYSTEM_PROMPT = build_term_normalization_system_prompt()


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


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :]
    if text.endswith("```"):
        text = text[: text.rindex("```")].rstrip()
    return text


def build_term_normalization_user_message(
    institution_id: str,
    dataset: str,
    row_selection_required: bool,
    *,
    term_candidates_json: str,
    raw_table_profile_columns_json: str,
) -> str:
    """
    Build the user message body for IdentityAgent Pass 2.

    Parameters
    ----------
    term_candidates_json, raw_table_profile_columns_json
        Pre-serialized JSON strings (e.g. from ``json.dumps(..., indent=2)``). Pass 2 does not
        prescribe the exact shape beyond what the system prompt describes; callers typically pass
        serialized :class:`~edvise.genai.identity_agent.profiling.schemas.RawColumnProfile` lists
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


def parse_term_normalization_pass_output(
    raw: RawTermPassInput,
) -> TermContract:
    """
    Parse and validate IdentityAgent Pass 2 JSON into :class:`TermContract`.

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
            "Term normalization pass parse failed; raw (truncated): %s", text[:500]
        )
        raise
