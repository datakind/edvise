"""
Step 2b Prompt Assembly — SchemaMappingAgent Transformation Map
Builds the prompt for generating a transformation map from a mapping manifest + reference examples.
"""

import inspect
import json

from .schemas import get_transformation_map_schema_context

from ..manifest.prompt_builder import (
    extract_schema_descriptor,
    summarize_schema_contract,
)


def get_transformation_utilities_context() -> str:
    """
    Extract transformation utilities documentation from transformation_utilities.py.
    Returns the full source code of the module for reference.
    """
    from . import utilities

    return inspect.getsource(utilities)


# ── Prompt assembly ────────────────────────────────────────────────────────────


def build_step2b_prompt(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_mapping_manifest: dict,
    institution_schema_contract: dict,
    cohort_schema_class,
    course_schema_class,
    reference_transformation_maps: list[dict],
    reference_institution_names: list[str] | None = None,
) -> str:
    """
    Build the Step 2b transformation map prompt.

    Args:
        institution_id:                 e.g. "lee_col"
        institution_name:               e.g. "Lee College"
        output_path:                    Destination path for the transformation map
        institution_mapping_manifest:   Parsed mapping manifest JSON for the target institution
        institution_schema_contract:    Parsed schema contract JSON for the target institution
        cohort_schema_class:            RawEdviseStudentDataSchema (Pandera class)
        course_schema_class:            RawEdviseCourseDataSchema (Pandera class)
        reference_transformation_maps:  List of parsed transformation map JSONs for reference
                                        institutions (e.g. [ucf_map, lc_map])
        reference_institution_names:    Display names for reference institutions, parallel
                                        to reference_transformation_maps. Defaults to
                                        institution_id values if not provided.
    """
    contract_summary = summarize_schema_contract(institution_schema_contract)
    cohort_descriptor = extract_schema_descriptor(cohort_schema_class)
    course_descriptor = extract_schema_descriptor(course_schema_class)

    # Build reference transformation map blocks
    if reference_institution_names is None:
        reference_institution_names = [
            m.get("institution_id", f"reference_{i}")
            for i, m in enumerate(reference_transformation_maps)
        ]

    reference_blocks = "\n\n".join(
        f'<reference_transformation_map institution="{name}" role="structural_reference_only">\n'
        f"{json.dumps(tmap, indent=2)}\n"
        f"</reference_transformation_map>"
        for name, tmap in zip(
            reference_institution_names, reference_transformation_maps
        )
    )

    prompt = f"""Please act as the SchemaMappingAgent and generate a transformation map for {institution_name} at:
{output_path}

The transformation map is a pure value transformation specification.
The field executor has already resolved every source Series from the manifest —
including cross-table joins.
Transformation steps receive a plain Series and must only transform values,
never resolve sources or perform joins.

{reference_blocks}

<mapping_manifest institution="{institution_name}">
{json.dumps(institution_mapping_manifest, indent=2)}
</mapping_manifest>

<schema_contract institution="{institution_name}">
{json.dumps(contract_summary, indent=2)}
</schema_contract>

<target_schema name="RawEdviseStudentDataSchema" entity="cohort">
{json.dumps(cohort_descriptor, indent=2)}
</target_schema>

<target_schema name="RawEdviseCourseDataSchema" entity="course">
{json.dumps(course_descriptor, indent=2)}
</target_schema>

<transformation_map_schema>
{get_transformation_map_schema_context()}
</transformation_map_schema>

<transformation_utilities>
{get_transformation_utilities_context()}
</transformation_utilities>

<rules>
STRUCTURE
- Match the reference transformation map JSON structure exactly (schema_version, institution_id,
  transformation_maps with cohort + course sections, each containing entity_type, target_schema, plans array)
- Each plan must include: target_field, output_dtype, steps array, review_status
- Set review_status: "pending" on all records — "approved" is only set after human review
- Generate separate transformation maps for cohort and course entities

TRANSFORMATION STEPS
- Use only utilities from the transformation_utilities library
- Each step must specify function_name (matching a utility function), column (source column name from manifest),
  and optional rationale
- Steps are applied in order to the resolved source Series from the manifest
- Use NEW_UTILITY_NEEDED when no existing utility can produce the correct output, even in combination with other steps.
  Do not approximate with a nearby utility. Provide: description (what the function should do),
  rationale (why no existing utility covers it),
  notes (any relevant data patterns, example input/output values, or implementation hints).

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
- Before choosing transformation steps, check the schema_contract for the source column's sample_values and unique_values
- If sample_values reveal a specific format
  (e.g. "202301.0" indicates YYYYMM with trailing decimal, "2018-19" indicates already-normalized YYYY-YY),
  choose the utility that matches that format exactly
- Do not assume a column's format from its name alone — verify against sample_values in the schema contract

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
- Apply domain-specific normalization (normalize_term_code, normalize_grade, etc.) as needed

EXTRA COLUMNS
- Some utilities require extra_columns
  (e.g., birthyear_to_age_bucket needs reference_year_series, conditional_credits needs grade_series)
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

    return prompt


# ── Convenience loader ─────────────────────────────────────────────────────────


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :]
    if text.endswith("```"):
        text = text[: text.rindex("```")].rstrip()
    return text
