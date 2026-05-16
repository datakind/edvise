"""
Step 2b Prompt Assembly — SchemaMappingAgent Transformation Map
Builds the prompt for generating a transformation map from a mapping manifest + reference examples.
"""

import inspect
import json
from typing import Any, cast

from edvise.genai.mapping.shared.hitl.confidence import (
    PIPELINE_HITL_CONFIDENCE_THRESHOLD,
)
from edvise.genai.mapping.shared.token_audit.prompt_token_audit import (
    audit_prompt_sections,
)

from .schemas import (
    RAW_EDVISE_FIELDS_FORBIDDING_MAP_VALUES,
    get_transformation_map_schema_context,
)
from .utilities import COMPACT_TERM_CODE_SUFFIX_TO_SEASON

from ..manifest.prompts import (
    extract_schema_descriptor,
    summarize_schema_contract,
)


def _step2b_cohort_entry_term_transformation_rules() -> str:
    """Step 2b: transformation steps for cohort entry_year / entry_term (align with manifest hierarchy)."""
    return """
COHORT entry_year AND entry_term — transformation steps (mirror manifest hierarchy)

**(1) Preferred — `_edvise_term_*` on the cohort base table are the manifest sources**
- Use only `strip_whitespace` and `cast_string` on `_edvise_term_academic_year` and `_edvise_term_season`.
- Do **not** use `extract_academic_year_from_term_code` or `extract_term_season_from_term_code`.

**(2) Fallback — manifest used a `term` / `course` join with `first_by` `_term_order`**
- Same strip/cast on the resolved `_edvise_term_*` Series from the lookup row.
- **reviewer_notes** or **validation_notes:** remind that semantics are **first term in history**, not necessarily cohort.
- **HITL:** This manifest-contracted path is **not** a discretionary proxy — do **not** set
  `review_required` or `flagged_steps` with `reason: proxy_source` solely because the Series is
  student- or term-table grain. Use **confidence 0.9** when the chain is only `strip_whitespace` /
  `cast_string` on IdentityAgent `_edvise_term_*` columns (same as a direct cohort-base mapping).

**(3) Manifest could not use normalized columns — unmappable or raw term fallback**
- If the manifest points at raw term codes and **validation_notes** flagged an IA gap, use extract
  utilities matching that column's format and **reviewer_notes:** IdentityAgent must materialize
  `_edvise_term_*` on student.
- If unmappable in the manifest, use empty `steps` with appropriate reviewer_notes.
"""


def _step2b_course_academic_term_transformation_rules() -> str:
    """Step 2b: transformation steps for course academic_year / academic_term."""
    return """
COURSE academic_year AND academic_term — transformation steps (mirror manifest hierarchy)

**(1) Preferred — manifest sources are `_edvise_term_*` on the course base table**
- Use only `strip_whitespace` and `cast_string` on `_edvise_term_academic_year` and `_edvise_term_season`.
- Do **not** use extract utilities on those columns.

**(2) Manifest maps raw `term` on the course row (IA gap; temporary)**
- Use `extract_academic_year_from_term_code` / `extract_term_season_from_term_code` only when the manifest
  explicitly maps from a raw term code column on **course**; **reviewer_notes** flag upstream IA.

**(3) Unmappable in manifest**
- Empty `steps`; do not invent transforms.
"""


def _step2b_cohort_completion_in_raw_edvise_rules() -> str:
    """Step 2b: how Edvise models cohort completion (matches RawEdviseStudentDataSchema)."""
    return """
COHORT completion — exact RawEdvise student columns (`RawEdviseStudentDataSchema`, raw_edvise_student.py)

**Term / cohort start (required; not completion):** `entry_year` (string, YEAR pattern), `entry_term`
(category — `TERM_CATEGORIES`). Degree/certificate **completion timing** is only the datetime columns listed next,
not another term-style field on this schema.

**Degree and certificate completion — datetime (optional, nullable, `datetime64[ns]` when present):**
1. `bachelors_degree_conferral_date`
2. `associates_degree_conferral_date`
3. `certificate1_date`
4. `certificate2_date`
5. `certificate3_date`

Those five are the **only** RawEdvise columns that carry completion **timing** as timestamps.

**Completion-related metadata — string (optional, nullable):**
1. `conferred_credential_type`
2. `major_at_completion`

*(Other datetimes on the schema are not completion — e.g. `matriculation_date` is matriculation / entry timing.)*

**Transformation plans for conferral / certificate dates**
- Follow **COHORT degree- and certificate-related DATETIME** rules below. The manifest will have
  resolved each target to a single `source_column` that is either a true calendar datetime or a
  raw term-code column on the wide student row or on a joined award/degree lookup row.
- IdentityAgent `_edvise_term_*` columns are never a legal manifest source for these targets —
  if the manifest points at one, that is a manifest bug; emit `hook_required: true` with empty
  `steps` and `reviewer_notes` flagging it.
- Do **not** treat `entry_term` / student `_edvise_term_*` as conferral completion proxies.
"""


def _step2b_cross_table_degree_datetime_rules() -> str:
    """Step 2b: degree conferral / certificate dates from a single source column."""
    _executor_suffix_json = json.dumps(
        COMPACT_TERM_CODE_SUFFIX_TO_SEASON, sort_keys=True, indent=2
    )
    return f"""
COHORT degree- and certificate-related DATETIME fields

Targets: `bachelors_degree_conferral_date`, `associates_degree_conferral_date`,
`certificate1_date`, `certificate2_date`, `certificate3_date`.

The Step 2a manifest always resolves these to a single `source_column` that is **either** a
true calendar datetime **or** a raw term-code column whose token embeds the season.
IdentityAgent `_edvise_term_*` columns are never a legal manifest source for conferral
targets (the manifest carries one `source_column` per record and the executor cannot
co-resolve a paired `_edvise_term_season` from the same selected lookup row). If you ever
see `_edvise_term_academic_year` or `_edvise_term_season` as the manifest `source_column`
for one of these targets, that is a manifest error: emit empty `steps` with
`hook_required: true` and `reviewer_notes` explicitly flagging the manifest fix.

**(1) Manifest `source_column` is a true calendar datetime / date**
- The manifest sourced a `datetime64[ns]` (or date-typed) column directly. Use no
  transformation steps, or at most a single `coerce_datetime` cast if the contract dtype
  is string but sample_values are unambiguous calendar timestamps.
- High confidence; no `review_required` unless coercion is uncertain.

**(2) Manifest `source_column` is a raw term-code column — format-driven with column grounding**

COLUMN GROUNDING (critical): derive format exclusively from the manifest's `source_column`
own `sample_values` in the schema contract. Never inherit format assumptions from any other
term column visible in the schema contract, even if both columns appear on the same table.
Different columns on the same table routinely use different term encodings.

TERM CONFIG CONTEXT (when provided): If `institution_term_config` is present in the
prompt, inspect the `season_map` for the dataset whose term column most plausibly
matches the conferral source column's encoding. Use it to pre-fill the `map_values`
mapping rather than inferring from sample_values alone — this raises confidence when
the season map covers all observed fragments. Still flag `review_required: true` and
`reason: inferred_season_mapping` when the match between term config and conferral
column encoding is uncertain. Do NOT apply a season_map from a different encoding
scheme (e.g. entry term Season_YYYY season_map applied to a YYYYMM conferral column).
When the plan chains to ``compact_term_code_to_conferral_date``, every **full** token
after ``map_values`` must be ``YYYY`` + a suffix from the **EXECUTOR SUFFIX TABLE** below
(IdentityAgent ``season_map`` may use institution-specific spellings — you must **translate**
those into allowed compact suffixes, e.g. never emit ``MW`` or other codes outside the table).

**EXECUTOR SUFFIX TABLE** for ``compact_term_code_to_conferral_date`` (must match runtime;
case-insensitive; token = 4-digit calendar year + suffix with **no** separator):
```json
{_executor_suffix_json}
```
Only these suffix spellings (and their ASCII case variants) parse to a conferral proxy.
Any other suffix → null at execution. If the institution needs a season outside this set,
use ``hook_required: true`` with empty ``steps`` rather than inventing suffixes.

This branch applies whether the manifest's `source_table` is an award/degree lookup
(joined from student) or the wide student row directly. Step 2a handles join + filter +
order; the resolved Series arrives at Step 2b as a single value per student.

CONFERRAL-STYLE DATETIME — EXACT ``function_name`` VALUES (copy these strings into JSON)
- ``compact_term_code_to_conferral_date`` — **one** resolved source column with **contiguous** compact tokens such as
  ``2025SP``, ``2024FA``, ``2015S1`` (4-digit year immediately followed by the suffix; **no** space or punctuation
  between year and season). Values like ``2019 Spring`` or ``2020-Fall`` are **not** compact: ``strip_whitespace``
  only trims ends and will **not** produce a valid token — insert ``map_values`` first (each observed raw string → a
  contiguous code, e.g. ``2019 Spring`` → ``2019SP``) then ``compact_term_code_to_conferral_date``. When samples are
  already contiguous, typical chain: ``strip_whitespace`` → ``compact_term_code_to_conferral_date`` only (no
  ``extract_year`` before it).
- ``academic_year_and_canonical_season_to_conferral_date`` — **two** inputs: the pipelined ``column`` must carry
  a 4-digit calendar year (string may embed it, e.g. ``2022-23``); ``extra_columns`` must include
  ``{{"season_series": "<name of a real base-table column>"}}`` whose values are canonical FALL / SPRING /
  SUMMER / WINTER. Use only when both columns exist on the cohort **base** table.

HOW TO CHOOSE CONFERRAL UTILITIES (executor contract — read before picking steps)
- Infer cohort **execution base table** the same way Step 2a does: first manifest mapping with a join →
  that join's `base_table` (almost always `student` for cohort); otherwise the mode of non-null `source_table`.
- Compare this target's manifest **`source_table`** to that base:
  - **Lookup / not the base** (e.g. `degree`, `term` while base is `student`): Step 2b only ever receives **one**
    resolved Series — the manifest's `source_column` after join + row_selection. There is **no** second lookup
    column on the base frame, and **prior-step outputs are not base-table columns**. Therefore you **must not**
    use `academic_year_and_canonical_season_to_conferral_date` or `term_components_to_datetime` here (they need `extra_columns` from
    **physical base-table columns**). For one token that encodes year + season as a **contiguous** compact code
    (``2025SP``, ``2024FA``, …): `strip_whitespace` → `compact_term_code_to_conferral_date`. If samples show a
    separator (e.g. ``2019 Spring``), add `map_values` before `compact_term_code_to_conferral_date` as in the
    YYYY+suffix rules below. For true YYYYMM calendar encodings on that same
    single column: `strip_trailing_decimal` → `coerce_datetime(fmt="%Y%m")` when justified by sample_values.
    If the format needs a year column + a separate season column but only one lookup column exists →
    `hook_required: true`, empty `steps`, explain in `reviewer_notes` (executor gap).
  - **Same as base** (manifest `source_table` equals cohort base, typically `student`): you **may** use
    `academic_year_and_canonical_season_to_conferral_date` or `term_components_to_datetime` **only if** `extra_columns` names real
    columns that **exist on that base table** in the cleaned data (e.g. `_edvise_term_academic_year` bound as
    the primary `column` / pipelined series and `_edvise_term_season` in `extra_columns` — but conferral
    targets must not source `_edvise_term_*` as the manifest `source_column` per rules above; this case is
    rare for conferral and usually means a different target or a wide row with paired columns).

- **YYYY + compact season suffix** (e.g. ``2025SP``, ``2024FA``, ``2015S1`` — year and suffix **touching**):
  - If sample_values show a **separator** between the year and season words (space, hyphen, slash), treat as **not**
    compact: `strip_whitespace` → `map_values` (distinct raw strings → contiguous tokens such as ``2019SP``) →
    `compact_term_code_to_conferral_date` on the **same** pipelined Series. (This is not using `map_values` output as
    `extra_columns` — it is normalizing the token before the compact parser.)
  - When values are already contiguous compact codes: `strip_whitespace` → `compact_term_code_to_conferral_date` — **one
    Series only**; do not chain `academic_year_and_canonical_season_to_conferral_date` with `extra_columns` here (lookup
    columns like `term` are not on the cohort base table, and you cannot point `extra_columns` at prior-step output).
  - Flag `review_required: true` when suffix coverage is inferred from sample_values.

- **YYYYMM-style compact numeric** (e.g. sample_values show "202301.0", "202305.0"):
  - `strip_trailing_decimal` → leaves "202301", "202305"
  - Inspect digits 5–6 to classify: if they correspond to calendar months (01–12 with plausible
    distribution across all 12), treat as true YYYYMM → `coerce_datetime(fmt="%Y%m")`.
  - If digits 5–6 appear to be season codes (e.g. only "01", "05", "08" appear — sparse, not all
    12 months): **do not** chain to `academic_year_and_canonical_season_to_conferral_date` when the manifest `source_table` is a
    **lookup** — `map_values` output is not a base-table column and cannot be passed via `extra_columns`.
    Prefer `hook_required: true` with empty `steps` and `reviewer_notes` describing the encoding until a
    single-Series utility or manifest change exists. If the manifest `source_table` **is** the cohort base and
    two **physical** base columns supply year fragment + season fragment, you may design a base-table-only
    chain; otherwise do not simulate paired columns through Step 2b alone.
  - Any inferred season fragment `map_values` on conferral: flag `reason=inferred_season_mapping`,
    `review_required: true`, confidence ≤ PIPELINE_HITL_CONFIDENCE_THRESHOLD when applicable.

- **Season_YYYY string** (e.g. "Fall 2023", "Spring 2022"):
  - Only when the manifest `source_column` lives on the **cohort base table** so a second column for
    `academic_year_and_canonical_season_to_conferral_date` can legally be resolved: e.g. split into year + season columns that
    both exist on `student`, or use `compact_term_code_to_conferral_date` / `coerce_datetime` if a single
    token column is easier.
  - If you truly have paired year and season **as base-table columns**, you may use
    `academic_year_and_canonical_season_to_conferral_date` with `extra_columns` pointing only at those base columns — never at
    a joined lookup-only column.
  - `map_values` is flagged: `reason=inferred_season_mapping` unless `unique_values` provides complete
    explicit coverage.

- **Opaque format** — cannot classify from sample_values alone:
  - `hook_required: true`, empty `steps`, `reviewer_notes` explaining what was observed.

When the manifest's `source_table` is the wide student row (no degree lookup), additionally:
- Always lower confidence, `review_required: true`.
- `reviewer_notes` must state: proxy conferral date from student row term code, no degree lookup
  available, lossiness is intentional.

**(3) Unmappable / non-datetime, non-term-code source in manifest**
- Empty `steps` and explain in `reviewer_notes`; do not invent a datetime pipeline.
"""


def _step2b_confidence_and_hitl_rules() -> str:
    """Confidence scoring, review_required, flagged_steps, and plan-level HITL options for Step 2b."""
    t = PIPELINE_HITL_CONFIDENCE_THRESHOLD
    return f"""
CONFIDENCE AND HITL (threshold = {t}, constant name PIPELINE_HITL_CONFIDENCE_THRESHOLD)

CONFIDENCE SCORING
- **1.0** — direct utility chain, no format ambiguity, source column dtype matches target exactly.
- **0.9** — standard chain with a well-evidenced format assumption (e.g. `cast_string` on a string column).
  Includes **manifest-contracted** cohort `entry_year` / `entry_term` paths where the manifest
  intentionally resolves `_edvise_term_*` from student or term joins (see COHORT entry_year AND
  entry_term rules): strip/cast only — document semantic caveats in notes, not HITL.
- **0.7–0.8** — inferred mapping from sample_values, ambiguous format, or an **ad-hoc** semantic
  proxy not justified by the manifest rules (see below).
- **≤ {t}** — `review_required` must be true; always set when:
  - `map_values` mapping was inferred from sample_values rather than explicit schema evidence;
  - format was ambiguous and a utility chain was chosen by best-guess;
  - manifest `source_column` for a **conferral / completion datetime** target is an IdentityAgent
    `_edvise_term_*` column (the manifest validator should have rejected this; if it slipped through,
    emit `hook_required: true` with empty `steps` and flag the manifest fix in `reviewer_notes` —
    see COHORT degree- and certificate-related DATETIME rules);
  - term code column required season fragment extraction and mapping.

MAPPING PROXY VS MANIFEST-CONTRACTED SOURCE
- **Do not** treat IdentityAgent term columns on student/lookup rows as `proxy_source` when the
  mapping manifest (and Step 2b cohort entry rules) **explicitly** chose that source for
  `entry_year` / `entry_term`. Lower grain or “first term in history” semantics belong in
  `reviewer_notes` / `validation_notes` only.
- Reserve `proxy_source` and HITL for cases where the pipeline is **stretching** semantics beyond
  what the manifest documents (or beyond the conferral-proxy rules that already require review).

REVIEW REQUIRED
- Set `review_required: true` on any plan where confidence ≤ PIPELINE_HITL_CONFIDENCE_THRESHOLD.
- Also set `review_required: true` when a **specific step** was inferred even if overall confidence is
  above the threshold — e.g. a `map_values` season mapping inferred from sample_values inside an
  otherwise high-confidence chain.
- Omit `review_required` (null) for high-confidence plans with no inferred steps.

FLAGGED STEPS (`flagged_steps` on the plan)
- For each plan where `review_required` is true, identify which specific steps drove uncertainty and list
  them in `flagged_steps`.
- Each entry must have:
  - `step_index`: 0-based index in the plan's `steps` array;
  - `function_name`: matches that step's `function_name`;
  - `reason`: one of `inferred_season_mapping` | `inferred_value_mapping` | `ambiguous_format` |
    `low_confidence_utility_chain` | `proxy_source` (use `proxy_source` only for **uncontracted**
    semantic stretch — not manifest-documented cohort entry term sourcing);
  - `context`: evidence — sample_values inspected, inferred mapping dict, format assumptions made.
- Multiple steps may be flagged in one plan.
- Flagged steps are **evidence only** — reviewer resolution is always at the **plan** level.

HITL OPTIONS (`hitl_options` on the plan)
- Every plan with `review_required: true` must include **exactly three** options **in order**:
  1. **approve** — reviewer accepts the proposed steps as-is.
     `resolution`: `{{"approved": true}}`
  2. **correct** — reviewer supplies corrected steps.
     `resolution`: null (out-of-band correction) **or** pre-filled with the proposed steps for in-place editing.
  3. **unmappable** — reviewer decides the field cannot be mapped.
     `resolution`: `{{"steps": [], "output_dtype": null}}`
- **Label** and **description** must be specific — name the target field and what is being confirmed
  (e.g. "Confirm season fragment mapping for deg_comp_term", not a generic "Approve mapping").
- Omit `hitl_options` (null) when `review_required` is omitted.

See schema definitions for `FlaggedStep`, `TransformationHITLOption`, `TransformationHITLItem`, and
`TransformationReview` in the transformation map schema context.
"""


def get_transformation_utilities_context() -> str:
    """
    Extract transformation utilities documentation from transformation_utilities.py.
    Returns the full source code of the module for reference.
    """
    from . import utilities

    return inspect.getsource(utilities)


def _step2b_term_config_context(institution_term_config: dict) -> str:
    return (
        "<institution_term_config>\n"
        "IdentityAgent term normalization output for this institution. "
        "Use ONLY to inform season fragment mapping decisions on raw term-code "
        "conferral date columns (case 2 in COHORT degree- and certificate-related "
        "DATETIME rules). Do NOT use to override the manifest's source_column "
        "choice. Do NOT apply entry term season_map to a conferral column without "
        "first verifying via sample_values that both columns share the same "
        "encoding — different columns on the same table routinely use different "
        "term encodings.\n"
        f"{json.dumps(institution_term_config, indent=2)}\n"
        "</institution_term_config>"
    )


# ── Prompt assembly ────────────────────────────────────────────────────────────

STEP2B_PROMPT_SECTION_KEYS: tuple[str, ...] = (
    "preamble",
    "reference_transformation_maps",
    "mapping_manifest",
    "schema_contract",
    "term_config",
    "target_schemas",
    "transformation_map_schema",
    "transformation_utilities",
    "rules",
)


def _step2b_preamble(institution_id: str, output_path: str) -> str:
    return f"""Please act as the SchemaMappingAgent and generate a transformation map for institution_id={institution_id} at:
{output_path}

The transformation map is a pure value transformation specification.
The field executor has already resolved every source Series from the manifest —
including cross-table joins.
Transformation steps receive a plain Series and must only transform values,
never resolve sources or perform joins.
"""


def _step2b_reference_maps(
    reference_institution_ids: list[str],
    reference_transformation_maps: list[dict],
) -> str:
    return "\n\n".join(
        f'<reference_transformation_map institution="{ref_id}" role="structural_reference_only">\n'
        f"{json.dumps(tmap, indent=2)}\n"
        f"</reference_transformation_map>"
        for ref_id, tmap in zip(
            reference_institution_ids, reference_transformation_maps
        )
    )


def _step2b_mapping_manifest(
    institution_id: str, institution_mapping_manifest: dict
) -> str:
    return (
        f'<mapping_manifest institution="{institution_id}">\n'
        f"{json.dumps(institution_mapping_manifest, indent=2)}\n"
        "</mapping_manifest>"
    )


def _step2b_schema_contract(
    institution_id: str, institution_schema_contract: dict
) -> str:
    contract_summary = summarize_schema_contract(institution_schema_contract)
    return (
        f'<schema_contract institution="{institution_id}">\n'
        f"{json.dumps(contract_summary, indent=2)}\n"
        "</schema_contract>"
    )


def _step2b_target_schemas(cohort_schema_class: Any, course_schema_class: Any) -> str:
    cohort_descriptor = extract_schema_descriptor(cohort_schema_class)
    course_descriptor = extract_schema_descriptor(course_schema_class)
    target_cohort = (
        '<target_schema name="RawEdviseStudentDataSchema" entity="cohort">\n'
        f"{json.dumps(cohort_descriptor, indent=2)}\n"
        "</target_schema>"
    )
    target_course = (
        '<target_schema name="RawEdviseCourseDataSchema" entity="course">\n'
        f"{json.dumps(course_descriptor, indent=2)}\n"
        "</target_schema>"
    )
    return f"{target_cohort}\n\n{target_course}"


def _step2b_tmap_schema() -> str:
    return (
        "<transformation_map_schema>\n"
        f"{get_transformation_map_schema_context()}\n"
        "</transformation_map_schema>"
    )


def _step2b_utilities() -> str:
    return (
        "<transformation_utilities>\n"
        f"{get_transformation_utilities_context()}\n"
        "</transformation_utilities>"
    )


def _step2b_rules() -> str:
    _raw_edvise_no_map_values = ", ".join(
        sorted(RAW_EDVISE_FIELDS_FORBIDDING_MAP_VALUES)
    )
    return f"""<rules>
STRUCTURE
- Match the reference transformation map shape: transformation_maps with cohort + course sections,
  each containing entity_type, target_schema, plans array.
  Do not output top-level release or institution fields — the pipeline adds them when saving.
- **Uniqueness:** In each `plans` array (cohort and course separately), every `target_field` value must
  appear **at most once**. Never emit two plan objects for the same `target_field`; merge steps and
  any reviewer notes into a single plan. The executor rejects duplicate `target_field` entries.
- Each plan must include: target_field, output_dtype, and steps array
- Do not include review_status on plans — the pipeline assigns it after validation
  and optional refinement
- Generate separate transformation maps for cohort and course entities
- When `review_required` is true: set `confidence`, non-empty `flagged_steps`, and exactly three
  `hitl_options` (approve / correct / unmappable) per CONFIDENCE AND HITL RULES below; omit both
  `flagged_steps` and `hitl_options` when `review_required` is omitted

TRANSFORMATION STEPS
- Use only utilities from the transformation_utilities library
- Each step must specify function_name (matching a utility function), column (source column name from manifest),
  and optional rationale
- Steps are applied in order to the resolved source Series from the manifest

HOOK REQUIRED
- When no existing utility or combination of utilities can produce the correct output,
  set "hook_required": true on the plan and provide a full explanation in reviewer_notes:
  what transformation is needed, what was attempted, and why no existing utility covers it.
  (Same vocabulary as IdentityAgent's hook_required — custom hook work, distinct from built-in utilities.)
- hook_required is a first-class plan field — not a step type, not a step in the steps array.
- steps may be empty or contain a best-effort partial chain alongside hook_required: true.
- Do not use hook_required as a shortcut — always attempt a utility chain first.

MANIFEST ALIGNMENT
- Generate plans only for target_fields that have mappings in the mapping manifest
- The manifest defines source_column, source_table, join, and row_selection —
  the transformation map only handles value transformations
- Use the source_column names from the manifest when specifying column in transformation steps
- For fields with a join in the manifest, the field executor resolves the cross-table Series before transformation steps run.
  Do NOT add join logic, merge logic, or cross-table lookups inside transformation steps —
  the steps receive a plain Series and must treat it as such.
  If the manifest is missing a join for a cross-table field, that is a manifest error — do not compensate in the transformation map

SOURCE COLUMN FORMAT AWARENESS
- Before choosing transformation steps, check the schema_contract for **the manifest's source_column**
  (not a different column that also appears in the contract) — sample_values and unique_values apply to that column
- If sample_values reveal a specific format
  (e.g. "202301.0" indicates YYYYMM with trailing decimal, "2018-19" indicates already-normalized YYYY-YY),
  choose the utility that matches that format exactly **for fields where the manifest sources that column**
- Do not assume a column's format from its name alone — verify against sample_values in the schema contract
- For cross-table degree/certificate datetime fields, follow **COHORT degree- and certificate-related DATETIME**
  rules below: the manifest's `source_column` is the only signal — choose utilities from that column's own
  `sample_values`, never from a different term column visible in the contract

OUTPUT DTYPES
- Set output_dtype to the RawEdvise / pandas name: "string", "Int64", "Float64" (extension dtypes — not numpy int64/float64), "category" (Pandera categoricals: entry_term, academic_term, pell_recipient_year1, term_pell_recipient), "boolean", "datetime64[ns]".
- Steps produce actual dtypes; output_dtype is the declared target for review and eval only.


STEP ORDERING
- Apply string cleaning (strip_whitespace, lowercase, uppercase) before value mapping
- Never place `extract_year` before `compact_term_code_to_conferral_date` in the same plan — the compact
  parser needs the full token (e.g. 2025SP) on the pipelined Series. Schema validation rejects that order.
  For chains that use `map_values` on full tokens then `extract_year` for
  `academic_year_and_canonical_season_to_conferral_date`, run `map_values` **before**
  `extract_year`.
- If map_values key matching depends on a normalized form (e.g. uppercase grade tokens),
  apply the normalizing step (normalize_grade, uppercase, etc.) BEFORE map_values —
  not after. The map keys must match the values that actually arrive at that step.
- Apply type casting steps (cast_string, cast_nullable_int, etc.) after value transformations
  unless an earlier step requires a specific type as input
- Apply domain-specific normalization (normalize_grade, etc.) as needed; canonical term season and academic year come from IdentityAgent `_edvise_term_*` columns (or other manifest-listed IA term columns), not SMA string parsers

{_step2b_confidence_and_hitl_rules()}
{_step2b_cohort_entry_term_transformation_rules()}
{_step2b_course_academic_term_transformation_rules()}
{_step2b_cohort_completion_in_raw_edvise_rules()}
{_step2b_cross_table_degree_datetime_rules()}
EXTRA COLUMNS
- Some utilities require extra_columns
  (e.g., birthyear_to_age_bucket needs reference_year_series, conditional_credits needs grade_series,
  term_components_to_datetime / academic_year_and_canonical_season_to_conferral_date need a second column on the **base** table)
- Specify extra_columns as a dict mapping parameter names to source column names: {{"param_name": "column_name"}}
- These columns are resolved from the base DataFrame before the step runs — they must already exist
  on the base DataFrame in source space. Do **not** name a column from a joined lookup table here;
  the cross-table resolver only fetches the manifest's `source_column`, not arbitrary co-resolved
  columns from the same selected lookup row. The output of an earlier Step 2b step is **not** a new
  base-table column — you cannot point `extra_columns` at ``"term"`` (or any lookup-only field) to
  stand in for a prior `map_values` result; use `compact_term_code_to_conferral_date` for single-token
  codes like ``2025SP`` from degree/award lookups instead.

NULL HANDLING
- Missing values are already pd.NA upstream. Do not fill nulls with invented labels
- Target fields marked non-nullable in the target schema: never use fill_nulls, map_values defaults, or other utilities
  to manufacture a value when the source is missing — that masks bad or incomplete source data.
- Demographics / bias fields (gender, race, ethnicity, first_gen,
  pell_status_first_year, military_status, disability_status, incarcerated_status, employment_status): never use
  placeholder strings like 'Unknown / Not Specified' here — bucket unknowns downstream if needed

CONSTANT FIELDS
- For fields that are derivable as institutional constants (e.g., all students are Bachelor's seekers), use fill_constant
- The column parameter in fill_constant is used only for length — the value parameter is the constant string

MAP VALUES USAGE
- **Do not use map_values** on these RawEdvise targets — the pipeline rejects such plans;
  preserve institution source wording (use strip_whitespace, cast_string, or empty steps only): {_raw_edvise_no_map_values}.
- Intermediate map_values on **datetime conferral / certificate targets** is allowed when required by
  **COHORT degree- and certificate-related DATETIME** (token shaping before parsers such as
  compact_term_code_to_conferral_date). That pattern is not value-remapping of free-text cohort or term-snapshot labels.
- Otherwise, map_values is appropriate only where the target schema enforces a constrained
  allowed-value set: category fields (academic_term, entry_term, pell_recipient_year1,
  term_pell_recipient), learner_age (isin LEARNER_AGE_BUCKETS), and grade
  (ALLOWED_LETTER_GRADES or numeric 0.0–4.0). These are the main fields where source
  values can violate a schema constraint that map_values can fix.
- Every other field in RawEdviseStudentDataSchema and RawEdviseCourseDataSchema is either
  free text (StringDtype, nullable=True, no value constraint) or a dtype-only field
  (Float64, datetime64[ns]) where casting handles conformance. Do NOT add map_values
  to any of these fields — there is no allowed-value set to map toward.
- map_values default="passthrough" keeps original values for unmapped entries
- map_values default=null fills unmapped entries with NA
- Explicit null mappings (e.g., {{"(Blank)": null}}) are preserved even with default="passthrough"

TRANSFORMATION MAP COMPACTNESS
- Include all required top-level keys and all required plan records
- For each plan, omit optional fields when their value would be null (e.g., reviewer_notes)
- For each step, omit optional fields when their value would be null (e.g., rationale, extra_columns, default)
- Do not omit steps array — use empty array [] for unmappable fields

TARGET SCHEMA AUTHORITY
- Use the target schema definitions as the authoritative source for expected output types and allowed values
- Ensure transformation steps produce values that match the target schema field constraints
  (regex patterns, allowed values, nullability)

JSON OUTPUT (strict, machine-parseable)
- The entire response must be one JSON object that passes a strict JSON parser (RFC 8259);
  no markdown fences, preamble, or text after the closing brace
- Strings use double quotes only; apostrophes in prose (e.g. Bachelor's) need no backslash.
  Never write a backslash before a single quote — that is invalid JSON
- Use only standard JSON string escapes (e.g. backslash-doublequote for a literal quote,
  doubled backslash for a backslash, backslash-n for newline, backslash-u plus four hex digits for Unicode)
- No trailing commas, comments, or single-quoted keys
</rules>

Generate the complete transformation map JSON now.
Output only that object — valid JSON throughout."""


def get_step2b_prompt_sections(
    institution_id: str,
    output_path: str,
    institution_mapping_manifest: dict,
    institution_schema_contract: dict,
    cohort_schema_class: Any,
    course_schema_class: Any,
    reference_transformation_maps: list[dict],
    reference_institution_ids: list[str] | None = None,
    institution_term_config: dict | None = None,
) -> dict[str, str]:
    """Named sections for Step 2b prompt (mirrors IA get_*_sections pattern)."""
    if reference_institution_ids is None:
        reference_institution_ids = [
            m.get("institution_id", f"reference_{i}")
            for i, m in enumerate(reference_transformation_maps)
        ]

    return {
        "preamble": _step2b_preamble(institution_id, output_path),
        "reference_transformation_maps": _step2b_reference_maps(
            reference_institution_ids,
            reference_transformation_maps,
        ),
        "mapping_manifest": _step2b_mapping_manifest(
            institution_id, institution_mapping_manifest
        ),
        "schema_contract": _step2b_schema_contract(
            institution_id, institution_schema_contract
        ),
        "term_config": (
            _step2b_term_config_context(institution_term_config)
            if institution_term_config is not None
            else ""
        ),
        "target_schemas": _step2b_target_schemas(
            cohort_schema_class, course_schema_class
        ),
        "transformation_map_schema": _step2b_tmap_schema(),
        "transformation_utilities": _step2b_utilities(),
        "rules": _step2b_rules(),
    }


def join_step2b_prompt_sections(sections: dict[str, str]) -> str:
    """Join sections in STEP2B_PROMPT_SECTION_KEYS order."""
    parts = [sections[k] for k in STEP2B_PROMPT_SECTION_KEYS if sections[k]]
    return parts[0] + "\n\n---\n" + "\n\n---\n".join(parts[1:])


def build_step2b_prompt(
    institution_id: str,
    output_path: str,
    institution_mapping_manifest: dict,
    institution_schema_contract: dict,
    cohort_schema_class: Any,
    course_schema_class: Any,
    reference_transformation_maps: list[dict],
    reference_institution_ids: list[str] | None = None,
    institution_term_config: dict | None = None,
) -> str:
    """
    Build the Step 2b transformation map prompt.

    Args:
        institution_id:                 e.g. "synthetic_coastal_cc"
        output_path:                    Destination path for the transformation map
        institution_mapping_manifest:   Parsed mapping manifest JSON for the target institution
        institution_schema_contract:    Parsed schema contract JSON for the target institution
        cohort_schema_class:            RawEdviseStudentDataSchema (Pandera class)
        course_schema_class:            RawEdviseCourseDataSchema (Pandera class)
        reference_transformation_maps:  List of parsed transformation map JSONs for reference
                                        institutions (e.g. [ref_a_map, ref_b_map])
        reference_institution_ids:      Labels for reference XML blocks, parallel to
                                        reference_transformation_maps. Defaults to each map's
                                        ``institution_id`` if not provided.
        institution_term_config:        Optional parsed IdentityAgent term normalization output
                                        (InstitutionTermContract as dict). When provided, injected as context for
                                        season fragment mapping on raw term code conferral date columns only.
                                        Pass None (default) when term config is unavailable or not needed.
    """
    sections = get_step2b_prompt_sections(
        institution_id,
        output_path,
        institution_mapping_manifest,
        institution_schema_contract,
        cohort_schema_class,
        course_schema_class,
        reference_transformation_maps,
        reference_institution_ids,
        institution_term_config,
    )
    return join_step2b_prompt_sections(sections)


def audit_step2b_prompt(
    institution_id: str,
    output_path: str,
    institution_mapping_manifest: dict,
    institution_schema_contract: dict,
    cohort_schema_class: Any,
    course_schema_class: Any,
    reference_transformation_maps: list[dict],
    reference_institution_ids: list[str] | None = None,
    institution_term_config: dict | None = None,
    *,
    log: bool = True,
) -> dict[str, Any]:
    """Local estimated token counts for Step 2b (single user blob)."""
    sections = get_step2b_prompt_sections(
        institution_id,
        output_path,
        institution_mapping_manifest,
        institution_schema_contract,
        cohort_schema_class,
        course_schema_class,
        reference_transformation_maps,
        reference_institution_ids,
        institution_term_config,
    )
    prefixed = {f"user.{k}": v for k, v in sections.items()}
    return audit_prompt_sections(
        prefixed,
        builder="schema_mapping_agent.step2b",
        institution_id=institution_id,
        log=log,
    )


# ── Convenience loader ─────────────────────────────────────────────────────────


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return cast(dict[str, Any], json.load(f))
