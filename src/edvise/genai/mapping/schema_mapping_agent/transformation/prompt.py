"""
Step 2b Prompt Assembly — SchemaMappingAgent Transformation Map
Builds the prompt for generating a transformation map from a mapping manifest + reference examples.
"""

import inspect
import json
from typing import Any, cast

from edvise.genai.mapping.shared.token_audit.prompt_token_audit import (
    audit_prompt_sections,
)

from .schemas import get_transformation_map_schema_context

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
- **reviewer_notes:** remind that semantics are **first term in history**, not necessarily cohort.

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
- Follow **COHORT degree- and certificate-related DATETIME** rules below — joins to degree/award rows,
  `_edvise_term_*` on that row when contracted, student-side **datetime** when it is true conferral timing,
  raw award codes for Step 2b parsing when IA columns are missing.
- Do **not** treat `entry_term` / student `_edvise_term_*` as conferral completion proxies.
"""


def _step2b_cross_table_degree_datetime_rules() -> str:
    """Step 2b: degree conferral / certificate dates from joined lookup tables."""
    return """
COHORT degree- and certificate-related DATETIME fields (cross-table joins to degree or similar)

These targets include e.g. `bachelors_degree_conferral_date`, `associates_degree_conferral_date`,
`certificate1_date`, `certificate2_date`, `certificate3_date` when the mapping manifest uses a
lookup table (e.g. `degree`) with filters and `row_selection`, or a **true calendar datetime / date**
column on wide `student`. The manifest must **not** source conferral-style targets from `student._edvise_term_*`
(entry semantics — handle timing via degree-row IA columns, award-row raw codes, or Step 2b parsing).

**(1) Manifest `source_column` is IdentityAgent term metadata — REQUIRED shape**
- If the manifest's `source_column` for the field is `_edvise_term_academic_year` (or another IA
  academic-year column on the **lookup** row), the resolved Series is already that column after the join.
  Use:
  - `strip_whitespace` then `term_components_to_datetime` with
    `extra_columns`: `{"season_series": "_edvise_term_season"}`
  - Do **not** substitute `coerce_datetime` on raw institution `term` / YYYYMM / term-code columns for
    these targets — even when `schema_contract` sample_values for `term` look easy to parse. Parsing
    `term` when the manifest sourced `_edvise_term_academic_year` ignores the manifest and breaks
    alignment with IdentityAgent normalization.
- Do **not** use `term_components_to_datetime` with **`student._edvise_term_*`** for these conferral targets;
  if the manifest incorrectly points there, prefer empty `steps` with **reviewer_notes** calling out the manifest fix.

**(2) Manifest `source_column` is a raw code or date column — format-driven**
- Only if the manifest explicitly maps `source_column` to `term`, a literal date column, or another
  raw code field (not `_edvise_term_academic_year`), choose utilities from `sample_values` /
  `unique_values` (e.g. `strip_trailing_decimal` + `coerce_datetime` with `fmt` for YYYYMM).

**(3) Unmappable / non-datetime source in manifest**
- Empty `steps` and explain in `reviewer_notes`; do not invent a datetime pipeline.
"""


def get_transformation_utilities_context() -> str:
    """
    Extract transformation utilities documentation from transformation_utilities.py.
    Returns the full source code of the module for reference.
    """
    from . import utilities

    return inspect.getsource(utilities)


# ── Prompt assembly ────────────────────────────────────────────────────────────

STEP2B_PROMPT_SECTION_KEYS: tuple[str, ...] = (
    "preamble",
    "reference_transformation_maps",
    "mapping_manifest",
    "schema_contract",
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


def _step2b_mapping_manifest(institution_id: str, institution_mapping_manifest: dict) -> str:
    return (
        f'<mapping_manifest institution="{institution_id}">\n'
        f"{json.dumps(institution_mapping_manifest, indent=2)}\n"
        "</mapping_manifest>"
    )


def _step2b_schema_contract(institution_id: str, institution_schema_contract: dict) -> str:
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
  rules below: do not pick utilities from raw `term` sample_values when the manifest sourced
  `_edvise_term_academic_year` (or another manifest-listed IA academic-year column on that row)

OUTPUT DTYPES
- Set output_dtype to the RawEdvise / pandas name: "string", "Int64", "Float64" (extension dtypes — not numpy int64/float64), "category" (Pandera categoricals: entry_term, academic_term, pell_recipient_year1, term_pell_recipient), "boolean", "datetime64[ns]".
- Steps produce actual dtypes; output_dtype is the declared target for review and eval only.


STEP ORDERING
- Apply string cleaning (strip_whitespace, lowercase, uppercase) before value mapping
- If map_values key matching depends on a normalized form (e.g. uppercase grade tokens),
  apply the normalizing step (normalize_grade, uppercase, etc.) BEFORE map_values —
  not after. The map keys must match the values that actually arrive at that step.
- Apply type casting steps (cast_string, cast_nullable_int, etc.) after value transformations
  unless an earlier step requires a specific type as input
- Apply domain-specific normalization (normalize_grade, etc.) as needed; canonical term season and academic year come from IdentityAgent `_edvise_term_*` columns (or other manifest-listed IA term columns), not SMA string parsers

{_step2b_cohort_entry_term_transformation_rules()}
{_step2b_course_academic_term_transformation_rules()}
{_step2b_cohort_completion_in_raw_edvise_rules()}
{_step2b_cross_table_degree_datetime_rules()}
EXTRA COLUMNS
- Some utilities require extra_columns
  (e.g., birthyear_to_age_bucket needs reference_year_series, conditional_credits needs grade_series,
  term_components_to_datetime needs season_series bound to the manifest's paired IA season column —
  typically ``_edvise_term_season`` alongside ``_edvise_term_academic_year``)
- Specify extra_columns as a dict mapping parameter names to source column names: {{"param_name": "column_name"}}
- These columns are resolved from the base DataFrame before the step runs

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
- map_values is appropriate only for fields whose target schema enforces a constrained
  allowed-value set: category fields (academic_term, entry_term, pell_recipient_year1,
  term_pell_recipient), learner_age (isin LEARNER_AGE_BUCKETS), and grade
  (ALLOWED_LETTER_GRADES or numeric 0.0–4.0). These are the only fields where source
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
        "target_schemas": _step2b_target_schemas(
            cohort_schema_class, course_schema_class
        ),
        "transformation_map_schema": _step2b_tmap_schema(),
        "transformation_utilities": _step2b_utilities(),
        "rules": _step2b_rules(),
    }


def join_step2b_prompt_sections(sections: dict[str, str]) -> str:
    """Join sections in STEP2B_PROMPT_SECTION_KEYS order."""
    parts = [sections[k] for k in STEP2B_PROMPT_SECTION_KEYS]
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


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :]
    if text.endswith("```"):
        text = text[: text.rindex("```")].rstrip()
    return text
