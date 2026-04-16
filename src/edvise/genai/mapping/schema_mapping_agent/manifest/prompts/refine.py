"""
SMA refinement + HITL prompts.

Builds the system and user prompts for the SMA refinement+HITL LLM call.

This call runs after:
  1. Original 2a LLM produces sma_manifest_output.json
  2. Deterministic validation produces list[ManifestValidationError]
  3. Low confidence fields are identified (confidence <= HITL_CONFIDENCE_THRESHOLD)

The LLM receives the manifest, validation errors, low confidence fields, and
the enriched schema contract. It produces a single JSON output:
  {
    "refined_manifest": FieldMappingManifest,
    "hitl_items": list[SMAHITLItem]
  }

The caller:
  - Writes refined_manifest back to sma_manifest_output.json (in place)
  - Writes hitl_items to sma_hitl.json as InstitutionSMAHITLItems

Public API:
    build_refinement_system_prompt() -> str
    build_refinement_user_prompt(
        institution_id,
        entity_type,
        manifest,
        validation_errors,
        schema_contract,
    ) -> str
"""

from __future__ import annotations

from edvise.genai.mapping.shared.schema_contract.schemas import (
    EnrichedSchemaContractForSMA,
)

from ...hitl.schemas import HITL_CONFIDENCE_THRESHOLD
from ..schemas import FieldMappingManifest, get_compact_manifest_schema_reference
from ..validation import ManifestValidationError


# ---------------------------------------------------------------------------
# Shared schema context (injected once into system prompt)
# ---------------------------------------------------------------------------

_MANIFEST_SCHEMA = get_compact_manifest_schema_reference()

_HITL_OUTPUT_SCHEMA = """
SMAHITLOption: {
  option_id: str,                          # snake_case, unique within item e.g. "use_term_descr"
  label: str,                              # ~4 words e.g. "Use term_descr column"
  description: str,                        # one sentence consequence
  reentry: "terminal" | "direct_edit",
  field_mapping: FieldMappingRecord | null, # null only for direct_edit option
  column_alias: ColumnAlias | null          # only for JOIN_STRUCTURE fixes needing alias bridge
}

SMAHITLItem: {
  item_id: str,              # "{institution_id}_{entity_type}_{target_field}_{failure_mode}"
  institution_id: str,
  entity_type: "cohort" | "course",
  target_field: str,
  failure_mode: "low_confidence" | "column_not_found" | "join_structure" | "row_selection" | "map_unmap",
  hitl_question: str,        # specific actionable question naming field, issue, decision needed
  hitl_context: str | null,  # evidence: sample values, available columns, rationale, error details
  current_field_mapping: FieldMappingRecord,  # your best refined attempt
  validation_errors: list[str],              # ManifestValidationError.detail strings, empty if low_confidence only
  options: list[SMAHITLOption],              # 2-5 options, last always option_id="direct_edit"
  choice: null,              # always null — reviewer sets this
  reviewer_note: null,       # always null — reviewer sets this
  direct_edit_field_mapping: null  # always null — reviewer sets this
}

ColumnAlias: {table: str!, source_column: str!, canonical_column: str!, rationale?: str}

ReviewStatus (set by you on each FieldMappingRecord):
  "auto_approved"    — passed validation + confidence threshold, no changes made
  "refined_by_llm"   — you corrected a validation error or low confidence field
  "proposed_for_hitl" — you could not fix, sending to human reviewer
"""

_AUTO_FIX_RULES = """
AUTO-CORRECT these without generating a HITLItem:
  - Column name is a clear typo or truncation with an obvious match in available columns
    (e.g. "term_desc" when "term_descr" is in the schema contract)
  - join declared on same table as source_table (remove the join)
  - Unmapped field has stale source_table or join set (clear them)
  - Missing column_alias where the canonical name is unambiguous from context

ESCALATE to a HITLItem when:
  - The correct source column is genuinely ambiguous among multiple candidates
  - A referenced table does not exist and no obvious substitute is present
  - Row selection strategy is unclear from field semantics alone
  - Join structure is wrong in a way requiring schema domain knowledge
  - Model confidence is at or below {threshold} and the mapping is not obviously correct
  - Multiple validation errors on the same field whose combined fix is ambiguous
""".format(threshold=HITL_CONFIDENCE_THRESHOLD)

_OPTION_GENERATION_RULES = """
OPTION GENERATION RULES — for each HITLItem:

  General:
  - Generate 2-4 TERMINAL options + always append a final direct_edit option (total 2-5).
  - Each TERMINAL option is a complete FieldMappingRecord — not a delta.
  - Options must be meaningfully distinct. Do not generate near-duplicate options.
  - last option is ALWAYS: {option_id: "direct_edit", label: "Edit directly",
    description: "Manually correct this field in the manifest editor.",
    reentry: "direct_edit", field_mapping: null, column_alias: null}

  By failure_mode:

  low_confidence:
    - Options are the top 2-3 source column candidates you were choosing between.
    - Include the original mapping as one option if it's still plausible.
    - Include the original rationale in hitl_context.

  column_not_found:
    - Options are close-match column candidates from training.column_details.
    - Order by similarity to the offending value (closest match first).
    - Each option has the corrected source_column set, all other fields preserved.

  join_structure:
    - Options are valid join key combinations from the schema contract.
    - If join keys differ by name across tables, include a column_alias on the option.
    - If the fix is to remove an unnecessary join (same-table field), make that an option.
    - Include the base_table and lookup_table in the option's join config.

  row_selection:
    - Options are valid RowSelectionStrategy alternatives for this field type.
    - Pre-fill required args: order_by for first_by/nth, condition_col for where_not_null.
    - Order by most likely correct strategy given field semantics.
    - Each option has the corrected row_selection set, all other fields preserved.

  map_unmap:
    - Always exactly two TERMINAL options + direct_edit (total 3):
      Option 1: confirm mapped — preserve source_column, source_table, join, row_selection.
      Option 2: mark unmapped — source_column=null, source_table=null, join=null,
                row_selection=null, confidence=1.0, rationale="Field is not mappable
                for this institution."
"""

_FIELD_COLLAPSE_RULE = """
MULTIPLE VALIDATION ERRORS ON THE SAME FIELD:
  Collapse into one HITLItem. Set failure_mode to the most actionable category
  (join_structure > column_not_found > row_selection > map_unmap > low_confidence).
  List all ManifestValidationError.detail strings in validation_errors.
  Generate options that fix all errors simultaneously — each option is a complete
  FieldMappingRecord that resolves every flagged issue on that field.
"""

_OUTPUT_FORMAT = """
OUTPUT FORMAT — respond with a single JSON object, no preamble, no markdown:
{
  "refined_manifest": { ...FieldMappingManifest with review_status set on every record... },
  "hitl_items": [ ...list[SMAHITLItem] for fields you could not auto-correct... ]
}

CRITICAL:
  - Every FieldMappingRecord in refined_manifest must have review_status set.
  - hitl_items may be empty [] if you were able to auto-correct all flagged fields.
  - Do not invent columns or tables not present in the schema contract.
  - Do not change auto_approved fields — copy them through unchanged.
  - All field_mapping entries in options must pass FieldMappingRecord validation
    (source_table required when source_column set, row_selection required for
    mappable non-constant fields, join required when source_table != base_table).
"""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def build_refinement_system_prompt() -> str:
    return f"""You are the Schema Mapping Agent refinement step for the Edvise institution onboarding pipeline.

Your job is to review a field mapping manifest produced by the original 2a LLM, correct errors where possible, and generate structured HITL review items for everything you cannot confidently fix.

## Manifest schema

{_MANIFEST_SCHEMA}

## Output schema

{_HITL_OUTPUT_SCHEMA}

## When to auto-correct vs. escalate

{_AUTO_FIX_RULES}

## Option generation rules

{_OPTION_GENERATION_RULES}

## Collapsing multiple errors per field

{_FIELD_COLLAPSE_RULE}

## Output format

{_OUTPUT_FORMAT}
"""


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------


def _format_validation_errors(
    errors: list[ManifestValidationError],
) -> str:
    """
    Group ManifestValidationErrors by target_field and format for prompt injection.
    Returns 'None' if no errors.
    """
    if not errors:
        return "None"

    by_field: dict[str, list[ManifestValidationError]] = {}
    for e in errors:
        by_field.setdefault(e.target_field, []).append(e)

    lines = []
    for field, field_errors in by_field.items():
        lines.append(f"  {field}:")
        for e in field_errors:
            code = (
                e.error_code.value
                if hasattr(e.error_code, "value")
                else str(e.error_code)
            )
            lines.append(f"    - [{code}] {e.detail}")
    return "\n".join(lines)


def _format_low_confidence_fields(
    manifest: FieldMappingManifest,
) -> str:
    """
    List fields below HITL_CONFIDENCE_THRESHOLD with their confidence and rationale.
    Returns 'None' if no low confidence fields.
    """
    low = [
        r for r in manifest.mappings
        if r.confidence <= HITL_CONFIDENCE_THRESHOLD
    ]
    if not low:
        return "None"

    lines = []
    for r in low:
        rationale = r.rationale or "No rationale provided."
        lines.append(
            f"  {r.target_field}: confidence={r.confidence:.2f}\n"
            f"    rationale: {rationale}"
        )
    return "\n".join(lines)


def _format_schema_contract_summary(
    schema_contract: EnrichedSchemaContractForSMA,
) -> str:
    """
    Compact schema contract summary for prompt injection.
    Per table: column list + sample values for low-cardinality columns.
    """
    lines = []
    for table_name, dataset in schema_contract.datasets.items():
        col_names = [cd.normalized_name for cd in dataset.training.column_details]
        lines.append(f"  {table_name}:")
        lines.append(f"    columns: {col_names}")

        # Surface sample values for low-cardinality columns (≤20 unique values)
        # — helps the LLM reason about map_values and row selection filters
        for cd in dataset.training.column_details:
            if cd.unique_values and len(cd.unique_values) <= 20:
                lines.append(
                    f"    {cd.normalized_name} samples: {cd.unique_values[:10]}"
                )
    return "\n".join(lines)


def build_refinement_user_prompt(
    institution_id: str,
    entity_type: str,
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
    schema_contract: EnrichedSchemaContractForSMA,
) -> str:
    """
    Build the user prompt for the SMA refinement+HITL LLM call.

    Parameters
    ----------
    institution_id:
        Institution identifier e.g. "ucf".
    entity_type:
        "cohort" or "course".
    manifest:
        FieldMappingManifest from the original 2a LLM call.
        Will be iterated on in place — caller writes refined_manifest
        back to sma_manifest_output.json.
    validation_errors:
        list[ManifestValidationError] from validate_manifest(). May be empty.
    schema_contract:
        EnrichedSchemaContractForSMA from IdentityAgent output.
    """
    manifest_json = manifest.model_dump_json(indent=2)
    validation_section = _format_validation_errors(validation_errors)
    low_confidence_section = _format_low_confidence_fields(manifest)
    schema_summary = _format_schema_contract_summary(schema_contract)

    flagged_fields = {e.target_field for e in validation_errors} | {
        r.target_field
        for r in manifest.mappings
        if r.confidence <= HITL_CONFIDENCE_THRESHOLD
    }
    auto_approved_fields = [
        r.target_field for r in manifest.mappings
        if r.target_field not in flagged_fields
    ]

    return f"""## Institution
institution_id: {institution_id}
entity_type: {entity_type}

## Schema contract — available tables and columns
{schema_summary}

## Current manifest (original 2a output)
{manifest_json}

## Deterministic validation errors
{validation_section}

## Low confidence fields (confidence ≤ {HITL_CONFIDENCE_THRESHOLD})
{low_confidence_section}

## Auto-approved fields (no action needed — copy through unchanged)
{auto_approved_fields}

## Your task
1. For each flagged field (validation errors or low confidence), attempt to auto-correct.
   Set review_status="refined_by_llm" on fields you fix.

2. For fields you cannot confidently fix, generate a SMAHITLItem with:
   - Your best attempt as current_field_mapping (review_status="proposed_for_hitl")
   - 2-4 complete FieldMappingRecord options covering the most plausible corrections
   - A final direct_edit option
   - hitl_context that includes validation error details and/or original rationale

3. Copy all auto-approved fields through unchanged with review_status="auto_approved".

4. Return the single JSON object described in your instructions.
"""


__all__ = [
    "build_refinement_system_prompt",
    "build_refinement_user_prompt",
]
