"""
SMA refinement + HITL: prompts, orchestration, and post-parse safety nets (single module).

**Pass 1** — refinement + HITL flagging (no options): one LLM call per entity
(cohort and course are separate calls).

**Pass 2** — option generation: one LLM call per Pass 1 flag (parallelizable).

Also includes :func:`run_sma_refinement` (two-pass LLM calls) and
:func:`apply_refinement_review_status_safety_net` (``review_status`` enforcement).

Pass 1 runs after:
  1. Original 2a LLM produces sma_manifest_output.json
  2. Deterministic validation produces list[ManifestValidationError]
  3. Low confidence fields are identified (confidence <= HITL_CONFIDENCE_THRESHOLD)

Prompt builders:
    build_refinement_pass1_system_prompt() -> str
    build_refinement_pass1_user_prompt(...) -> str
    build_refinement_combined_pass1_system_prompt() -> str
    build_refinement_combined_pass1_user_prompt(...) -> str
    build_refinement_pass2_system_prompt() -> str
    build_refinement_pass2_user_prompt(hitl_flag, schema_contract, max_options=3) -> str

Backward-compatible aliases (Pass 1):
    build_refinement_system_prompt -> build_refinement_pass1_system_prompt
    build_refinement_user_prompt -> build_refinement_pass1_user_prompt
    build_refinement_combined_system_prompt -> build_refinement_combined_pass1_system_prompt
    build_refinement_combined_user_prompt -> build_refinement_combined_pass1_user_prompt

Runtime:
    run_sma_refinement, apply_refinement_review_status_safety_net, ...

The caller:
  - Writes refined_manifest back to sma_manifest_output.json (in place)
  - Writes Pass 2 output as InstitutionSMAHITLItems to sma_hitl.json

``hitl`` types are imported lazily inside orchestration helpers so importing this module
does not require a fully initialized ``hitl`` package (see ``hitl.artifacts``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from edvise.genai.mapping.shared.hitl.confidence import PIPELINE_HITL_CONFIDENCE_THRESHOLD
from edvise.genai.mapping.shared.schema_contract.schemas import (
    EnrichedSchemaContractForSMA,
)

from ..schemas import FieldMappingManifest, ReviewStatus, get_compact_manifest_schema_reference
from ..validation import ManifestValidationError

from .generate import strip_json_fences

# Same value as ``hitl.schemas.HITL_CONFIDENCE_THRESHOLD`` — defined here so prompt + runtime
# code share one constant without importing ``hitl`` at module load (circular with ``artifacts``).
HITL_CONFIDENCE_THRESHOLD = PIPELINE_HITL_CONFIDENCE_THRESHOLD


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
  current_field_mapping: FieldMappingRecord,  # ORIGINAL from input manifest, unchanged — options hold fixes
  validation_errors: list[str],              # ManifestValidationError.detail strings, empty if low_confidence only
  options: list[SMAHITLOption],              # 2-4 options, last always option_id="direct_edit"
  choice: null,              # always null — reviewer sets this
  reviewer_note: null,       # always null — reviewer sets this
  direct_edit_field_mapping: null  # always null — reviewer sets this
}

ColumnAlias: {table: str!, source_column: str!, canonical_column: str!, rationale?: str}

ReviewStatus (set by Pass 1 on each FieldMappingRecord in refined_manifest):
  "auto_approved"    — passed validation + confidence threshold, no changes made
  "refined_by_llm"   — Pass 1 corrected a validation error or low confidence field
  "proposed_for_hitl" — Pass 1 could not fix, sending to human reviewer (options from Pass 2)
"""

_PASS1_OUTPUT_SCHEMA = """
Pass 1 output — respond with a single JSON object, no preamble, no markdown:
{
  "refined_manifest": { ...FieldMappingManifest with review_status set on every record... },
  "hitl_flags": [
    {
      "item_id": "{institution_id}_{entity_type}_{target_field}_{failure_mode}",
      "institution_id": str,
      "entity_type": "cohort" | "course",
      "target_field": str,
      "failure_mode": "low_confidence" | "column_not_found" | "join_structure" | "row_selection" | "map_unmap",
      "hitl_question": str,   # specific actionable question
      "hitl_context": str | null,  # evidence: validation errors, rationale, sample values
      "current_field_mapping": FieldMappingRecord,  # original generating agent output, unchanged
      "validation_errors": list[str]  # ValidationError.detail strings, empty if low_confidence only
    }
  ]
}

CRITICAL:
  - hitl_flags may be empty [] if all fields were auto-corrected or auto-approved.
  - current_field_mapping is ALWAYS the original generating agent's record, copied unchanged.
  - Do not generate options in Pass 1 — options are generated in Pass 2.
  - Every FieldMappingRecord in refined_manifest must have review_status set.
  - Do not change confidence on any field.
"""

_PASS2_OUTPUT_SCHEMA = """
Pass 2 output — respond with a single JSON object, no preamble, no markdown:
{
  "item_id": str,
  "institution_id": str,
  "entity_type": str,
  "target_field": str,
  "failure_mode": str,
  "hitl_question": str,
  "hitl_context": str | null,
  "current_field_mapping": FieldMappingRecord,  # copied from input, unchanged
  "validation_errors": list[str],
  "options": [
    // 1-3 TERMINAL options (complete FieldMappingRecord per option)
    // + always one final direct_edit option
    // Maximum 4 options total (3 TERMINAL + direct_edit)
    // Option 1 is always your recommended fix
    // Include original mapping as an option if still plausible
  ],
  "choice": null,
  "reviewer_note": null,
  "direct_edit_field_mapping": null
}

CRITICAL:
  - Maximum 3 TERMINAL options + 1 direct_edit = 4 options total.
  - Minimum 1 TERMINAL option + 1 direct_edit = 2 options total.
  - Last option ALWAYS option_id="direct_edit", reentry="direct_edit", field_mapping=null.
  - current_field_mapping copied from input unchanged — never modify it.
  - Each TERMINAL option is a complete FieldMappingRecord.
  - column_alias is only set on JOIN_STRUCTURE options that need alias bridging.
"""

_AUTO_FIX_RULES = """
AUTO-CORRECT (set review_status="refined_by_llm") ONLY when ALL of these are true:
  - confidence > {threshold}
  - The fix is unambiguous and deterministic:
      • Column name is a clear typo/truncation with an obvious match in available columns
      • join declared on same table as source_table (remove the join)
      • Unmapped field has stale source_table or join set (clear them)
      • Missing column_alias where the canonical name is unambiguous from context

ALWAYS flag as proposed_for_hitl + emit a hitl_flags entry when ANY of these are true:
  - confidence <= {threshold}  ← no exceptions, even if the mapping looks correct
  - Validation error exists AND the fix requires judgment (not a clear typo/structural fix)
  - Multiple validation errors whose combined fix is ambiguous

NOTE: refined_by_llm is ONLY valid when confidence > {threshold} AND you made a
clear deterministic fix. Never use refined_by_llm for low confidence fields.

REVIEW STATUS RULES — strictly enforced:
  - NEVER set review_status="auto_approved" on any field where confidence <= {threshold}.
  - NEVER set review_status="auto_approved" on any field that has a validation error.
  - NEVER set review_status="refined_by_llm" on any field where confidence <= {threshold}.
  - Do not change confidence — it reflects the generating agent's uncertainty
    and is frozen at generation time. review_status communicates what happened.
  - Fields in the auto_approved_fields list must be copied through with
    review_status="auto_approved" and NO changes to any other field.
  - Fields you corrected (clear deterministic fix, confidence > {threshold}) must
    have review_status="refined_by_llm".
  - Fields you could not fix OR any field with confidence <= {threshold} must have
    review_status="proposed_for_hitl" in refined_manifest AND appear in hitl_flags
    (Pass 2 will attach options).
  - There is no fourth status — every field gets exactly one of these three statuses.
""".format(threshold=HITL_CONFIDENCE_THRESHOLD)

_OPTION_GENERATION_RULES = """
OPTION GENERATION RULES — Pass 2 only; for each flag from Pass 1:

  CRITICAL — current_field_mapping:
    current_field_mapping must always be the original generating agent's FieldMappingRecord,
    copied through unchanged from the Pass 1 hitl_flag. Do not modify it.
    Your recommended fix is option 1 in the options list.
    The reviewer sees the original mapping and chooses from your suggestions.

  General:
  - Generate 1-3 TERMINAL options + always append a final direct_edit option (2-4 options total).
  - Each TERMINAL option is a complete FieldMappingRecord — not a delta.
  - Options must be meaningfully distinct. Do not generate near-duplicate options.
  - Option 1 is always your recommended correction — label it "Recommended fix" or
    a specific label like "Mark unmapped" if your recommendation is clear.
  - If the original mapping is still plausible despite the flag, include it as an
    option labeled "Keep original mapping" — never silently discard it.
  - last option is ALWAYS: {option_id: "direct_edit", label: "Edit directly",
    description: "Manually correct this field in the manifest editor.",
    reentry: "direct_edit", field_mapping: null, column_alias: null}

  By failure_mode:

  low_confidence:
    - Option 1 is your recommended source column candidate.
    - Include the original mapping as an option labeled "Keep original mapping"
      if it is still plausible.
    - Include other strong candidates if they exist (stay within TERMINAL cap).
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
    - Use at most two TERMINAL options + direct_edit when the decision is binary;
      otherwise use up to three TERMINAL options + direct_edit within the global cap.
    - Option 1 is your recommendation (either "Mark unmapped" or "Keep mapped").
    - If recommending unmapped: include mark unmapped vs keep original as distinct TERMINAL options.
    - If recommending mapped: Option 1 = your corrected FieldMappingRecord,
      include mark unmapped as needed within the cap.
"""

_FIELD_COLLAPSE_RULE = """
MULTIPLE VALIDATION ERRORS ON THE SAME FIELD (Pass 2):
  Pass 1 should collapse multiple errors into one hitl_flags entry per target_field when possible.
  Set failure_mode to the most actionable category
  (join_structure > column_not_found > row_selection > map_unmap > low_confidence).
  List all ManifestValidationError.detail strings in validation_errors.
  Generate options that fix all errors simultaneously — each option is a complete
  FieldMappingRecord that resolves every flagged issue on that field.
"""

_PASS1_OUTPUT_FORMAT = """
OUTPUT FORMAT — respond with a single JSON object, no preamble, no markdown:
{
  "refined_manifest": { ...FieldMappingManifest with review_status set on every record... },
  "hitl_flags": [ ...Pass 1 flags for fields you could not auto-correct — no options... ]
}

CRITICAL:
  - Every FieldMappingRecord in refined_manifest must have review_status set.
  - hitl_flags may be empty [] if you were able to auto-correct all flagged fields.
  - Do not invent columns or tables not present in the schema contract.
  - Do not change auto_approved fields — copy them through unchanged.
  - Do not emit options in Pass 1.
"""

_PASS1_OUTPUT_FORMAT_COMBINED = """
OUTPUT FORMAT — respond with a single JSON object, no preamble, no markdown:
{
  "refined_manifests": {
    "<entity_type>": { ...FieldMappingManifest with review_status set on every record... },
    ... one entry per entity_type listed in the user message under "entities" ...
  },
  "hitl_flags_by_entity": {
    "<entity_type>": [ ...Pass 1 flags for that entity only — no options... ],
    ...
  }
}

CRITICAL:
  - refined_manifests and hitl_flags_by_entity must contain exactly the same keys,
    matching the entity list in the user message (e.g. cohort and course).
  - Each flag's entity_type must match the bucket it is placed in.
  - Every FieldMappingRecord in each refined manifest must have review_status set.
  - Each hitl_flags_by_entity list may be empty [] if you auto-corrected all
    flagged fields for that entity.
  - Do not invent columns or tables not present in the schema contract.
  - Do not change auto_approved fields — copy them through unchanged.
  - Do not emit options in Pass 1.
"""


def _build_pass1_system_prompt(*, output_format: str) -> str:
    return f"""You are Pass 1 of the Schema Mapping Agent refinement step for the Edvise institution onboarding pipeline.

Architecture: Pass 1 refines the manifest and emits hitl_flags (no options). Pass 2 (separate calls)
generates reviewer options for each flag. You only run Pass 1.

Your job is to review a field mapping manifest produced by the original 2a LLM, correct errors where
possible, and emit structured hitl_flags for everything you cannot confidently fix — without
generating options (that is Pass 2).

## Manifest schema

{_MANIFEST_SCHEMA}

## Pass 1 output schema

{_PASS1_OUTPUT_SCHEMA}

## When to auto-correct vs. escalate

{_AUTO_FIX_RULES}

## Output format

{output_format}
"""


def build_refinement_pass1_system_prompt() -> str:
    return _build_pass1_system_prompt(output_format=_PASS1_OUTPUT_FORMAT)


def build_refinement_combined_pass1_system_prompt() -> str:
    return _build_pass1_system_prompt(output_format=_PASS1_OUTPUT_FORMAT_COMBINED)


def build_refinement_pass2_system_prompt() -> str:
    return f"""You are Pass 2 of the Schema Mapping Agent refinement for the Edvise institution onboarding pipeline.

Pass 1 already refined the manifest and produced one hitl_flag for this field. Your only job is to
generate reviewer-facing options (complete FieldMappingRecord per TERMINAL option) plus a final
direct_edit escape hatch.

## Manifest schema (compact reference)

{_MANIFEST_SCHEMA}

## Output schemas

{_HITL_OUTPUT_SCHEMA}

{_PASS2_OUTPUT_SCHEMA}

## Option generation rules

{_OPTION_GENERATION_RULES}

## Collapsing multiple errors per field

{_FIELD_COLLAPSE_RULE}

CRITICAL — current_field_mapping:
  current_field_mapping in your output must be copied unchanged from the hitl_flag input.
  Never modify it. Corrections belong in options (option 1 = recommended).
"""


# ---------------------------------------------------------------------------
# Backward-compatible aliases (Pass 1)
# ---------------------------------------------------------------------------

def build_refinement_system_prompt() -> str:
    return build_refinement_pass1_system_prompt()


def build_refinement_combined_system_prompt() -> str:
    return build_refinement_combined_pass1_system_prompt()


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------

_ENTITY_KEY_ORDER = ("cohort", "course")


def _sort_entity_keys(keys: list[str]) -> list[str]:
    """Stable order: cohort, course, then any other keys alphabetically."""
    known = [k for k in _ENTITY_KEY_ORDER if k in keys]
    rest = sorted(k for k in keys if k not in _ENTITY_KEY_ORDER)
    return known + rest


def _refinement_entity_section(
    entity_type: str,
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
) -> str:
    """
    Per-entity manifest, validation context, and auto-approved list for refinement prompts.
    """
    manifest_json = manifest.model_dump_json(indent=2)
    validation_section = _format_validation_errors(validation_errors)
    low_confidence_section = _format_low_confidence_fields(manifest)

    flagged_fields = {e.target_field for e in validation_errors} | {
        r.target_field
        for r in manifest.mappings
        if r.confidence <= HITL_CONFIDENCE_THRESHOLD
    }
    auto_approved_fields = [
        r.target_field for r in manifest.mappings
        if r.target_field not in flagged_fields
    ]

    return f"""### Entity: {entity_type}

## Current manifest (original 2a output)
{manifest_json}

## Deterministic validation errors
{validation_section}

## Low confidence fields (confidence ≤ {HITL_CONFIDENCE_THRESHOLD})
{low_confidence_section}

## Auto-approved fields (no action needed — copy through unchanged)
{auto_approved_fields}
"""


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


def build_refinement_pass1_user_prompt(
    institution_id: str,
    entity_type: str,
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
    schema_contract: EnrichedSchemaContractForSMA,
) -> str:
    """
    Build the user prompt for SMA refinement Pass 1 (manifest + hitl_flags, no options).

    Parameters
    ----------
    institution_id:
        Institution identifier e.g. "ucf".
    entity_type:
        "cohort" or "course".
    manifest:
        FieldMappingManifest from the original 2a LLM call.
    validation_errors:
        list[ManifestValidationError] from validate_manifest(). May be empty.
    schema_contract:
        EnrichedSchemaContractForSMA from IdentityAgent output.
    """
    schema_summary = _format_schema_contract_summary(schema_contract)
    entity_block = _refinement_entity_section(entity_type, manifest, validation_errors)

    return f"""## Institution
institution_id: {institution_id}
entity_type: {entity_type}

## Schema contract — available tables and columns
{schema_summary}

{entity_block}

## Your task (Pass 1 only)
1. For each flagged field (validation errors or low confidence), attempt to auto-correct.
   Set review_status="refined_by_llm" on fields you fix (subject to confidence rules).

2. For fields you cannot confidently fix, emit one hitl_flags entry per field with:
   - current_field_mapping = the ORIGINAL field mapping from the input manifest,
     copied unchanged
   - hitl_question and hitl_context that explain what failed and what you need from a reviewer
   - validation_errors populated from ManifestValidationError.detail strings (or empty for low_confidence-only)
   - Do NOT include options — Pass 2 will generate those.

3. Copy all auto-approved fields through unchanged with review_status="auto_approved".

4. Return the single Pass 1 JSON object (refined_manifest + hitl_flags) described in your instructions.
"""


def build_refinement_user_prompt(
    institution_id: str,
    entity_type: str,
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
    schema_contract: EnrichedSchemaContractForSMA,
) -> str:
    """Backward-compatible alias for :func:`build_refinement_pass1_user_prompt`."""
    return build_refinement_pass1_user_prompt(
        institution_id,
        entity_type,
        manifest,
        validation_errors,
        schema_contract,
    )


def build_refinement_combined_pass1_user_prompt(
    institution_id: str,
    manifests_by_entity: Mapping[str, FieldMappingManifest],
    validation_errors_by_entity: Mapping[str, list[ManifestValidationError]],
    schema_contract: EnrichedSchemaContractForSMA,
) -> str:
    """
    User prompt for Pass 1 covering multiple entities (e.g. cohort + course in one call).

    For separate cohort/course gateway calls, use :func:`build_refinement_pass1_user_prompt`
    once per entity instead.
    """
    entity_keys = _sort_entity_keys(list(manifests_by_entity.keys()))
    schema_summary = _format_schema_contract_summary(schema_contract)
    blocks = [
        _refinement_entity_section(
            ek,
            manifests_by_entity[ek],
            list(validation_errors_by_entity.get(ek, [])),
        )
        for ek in entity_keys
    ]

    return f"""## Institution
institution_id: {institution_id}
entities: {entity_keys}

You must return JSON with ``refined_manifests`` and ``hitl_flags_by_entity`` containing
exactly these keys: {entity_keys}.

## Schema contract — available tables and columns (shared)
{schema_summary}

{chr(10).join(blocks)}

## Your task (Pass 1 only)
For EACH entity section above, apply the refinement rules independently.

1. For each flagged field (validation errors or low confidence), attempt to auto-correct.
   Set review_status="refined_by_llm" on fields you fix (subject to confidence rules).

2. For fields you cannot confidently fix, emit hitl_flags entries for that entity with:
   - current_field_mapping = the ORIGINAL field mapping from the input manifest,
     copied unchanged
   - hitl_question and hitl_context that explain what failed
   - validation_errors from ManifestValidationError.detail strings (or empty for low_confidence-only)
   - Do NOT include options — Pass 2 will generate those.

3. Copy all auto-approved fields through unchanged with review_status="auto_approved".

4. Return the single combined Pass 1 JSON object described in your instructions.
"""


def build_refinement_combined_user_prompt(
    institution_id: str,
    manifests_by_entity: Mapping[str, FieldMappingManifest],
    validation_errors_by_entity: Mapping[str, list[ManifestValidationError]],
    schema_contract: EnrichedSchemaContractForSMA,
) -> str:
    """Backward-compatible alias for :func:`build_refinement_combined_pass1_user_prompt`."""
    return build_refinement_combined_pass1_user_prompt(
        institution_id,
        manifests_by_entity,
        validation_errors_by_entity,
        schema_contract,
    )


def build_refinement_pass2_user_prompt(
    hitl_flag: dict,
    schema_contract: EnrichedSchemaContractForSMA,
    max_options: int = 3,
) -> str:
    """
    Build the user prompt for SMA refinement Pass 2 (options for one Pass 1 flag).

    Parameters
    ----------
    hitl_flag:
        One element from Pass 1 ``hitl_flags`` (dict-shaped).
    schema_contract:
        EnrichedSchemaContractForSMA from IdentityAgent output.
    max_options:
        Cap TERMINAL options at this value (clamped to 1–3). There is always
        one additional direct_edit option (2–4 options total).
    """
    terminal_cap = max(1, min(3, max_options))
    schema_summary = _format_schema_contract_summary(schema_contract)
    flag_json = json.dumps(hitl_flag, indent=2, default=str)

    return f"""## Pass 1 flag (generate options only)
{flag_json}

## Schema contract — available tables and columns
{schema_summary}

## Option budget
- At most {terminal_cap} TERMINAL option(s), plus exactly one final direct_edit option.
- Total options: between 2 and {terminal_cap + 1} inclusive.

## Your task
Return the Pass 2 JSON object for this single flag: a complete SMAHITLItem shape with options
(last option always direct_edit). Option 1 must be your recommended fix.
"""


# ---------------------------------------------------------------------------
# Orchestration + post-parse safety nets (same module as prompts)
# ---------------------------------------------------------------------------


def _hitl_target_fields(
    hitl: list[Mapping[str, Any]] | list[Any],
) -> set[str]:
    """Target fields covered by Pass 1 flags and/or Pass 2 items."""
    from edvise.genai.mapping.schema_mapping_agent.hitl.schemas import SMAHITLItem

    out: set[str] = set()
    for item in hitl:
        if isinstance(item, SMAHITLItem):
            out.add(item.target_field)
            continue
        tf = item.get("target_field")
        if isinstance(tf, str):
            out.add(tf)
    return out


def _enforce_review_status_contract(
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
    hitl_flags: list[Mapping[str, Any]] | list[Any],
) -> list[str]:
    """
    Post-parse safety net: enforce review_status contract the LLM may have violated.
    Returns list of warning strings for logging — does not raise.
    Mutations applied in place.
    """
    flagged_fields = {e.target_field for e in validation_errors} | _hitl_target_fields(
        hitl_flags
    )
    warnings: list[str] = []

    for record in manifest.mappings:
        is_low_confidence = record.confidence <= HITL_CONFIDENCE_THRESHOLD
        is_flagged = record.target_field in flagged_fields
        is_auto_approved = record.review_status == ReviewStatus.auto_approved
        is_refined = record.review_status == ReviewStatus.refined_by_llm

        if is_auto_approved and (is_low_confidence or is_flagged):
            warnings.append(
                f"[review_status violation] '{record.target_field}' marked "
                f"auto_approved but confidence={record.confidence} "
                f"(threshold={HITL_CONFIDENCE_THRESHOLD}) or has validation "
                f"errors / HITL flags. Forcing to proposed_for_hitl."
            )
            record.review_status = ReviewStatus.proposed_for_hitl

        elif is_refined and is_low_confidence:
            warnings.append(
                f"[review_status violation] '{record.target_field}' marked "
                f"refined_by_llm but confidence={record.confidence} <= "
                f"threshold={HITL_CONFIDENCE_THRESHOLD}. Low confidence fields "
                f"must always be proposed_for_hitl. Forcing."
            )
            record.review_status = ReviewStatus.proposed_for_hitl

    return warnings


def log_refinement_contract_warnings_to_mlflow(warnings: list[str]) -> None:
    """Log refinement contract warnings to MLflow when an active run exists."""
    if not warnings:
        return
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.log_param(
            "sma_refinement_review_status_warning_count",
            len(warnings),
        )
        mlflow.log_text(
            "\n".join(warnings),
            "sma_refinement_review_status_warnings.txt",
        )
    except Exception:
        pass


def apply_refinement_review_status_safety_net(
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
    hitl_flags: list[Mapping[str, Any]] | list[Any],
    *,
    print_warnings: bool = True,
    log_mlflow: bool = True,
) -> list[str]:
    """
    Run :func:`_enforce_review_status_contract` and optionally print / log warnings.

    ``hitl_flags`` may be Pass 1 dicts or validated :class:`~edvise.genai.mapping.schema_mapping_agent.hitl.schemas.SMAHITLItem` instances.
    """
    warnings = _enforce_review_status_contract(
        manifest, validation_errors, hitl_flags
    )
    if print_warnings:
        for w in warnings:
            print(f"⚠  {w}")
    if log_mlflow:
        log_refinement_contract_warnings_to_mlflow(warnings)
    return warnings


def _default_llm_complete() -> Callable[[str, str], str]:
    from edvise.genai.mapping.identity_agent.grain_inference.databricks_gateway import (
        create_openai_client_for_databricks_gateway,
        make_databricks_gateway_llm_complete,
        resolve_gateway_model_id,
    )

    client = create_openai_client_for_databricks_gateway()
    return make_databricks_gateway_llm_complete(
        client, model=resolve_gateway_model_id()
    )


def _run_pass1(
    institution_id: str,
    entity_type: str,
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
    schema_contract: EnrichedSchemaContractForSMA,
    llm_complete: Callable[[str, str], str],
) -> dict[str, Any]:
    system = build_refinement_pass1_system_prompt()
    user = build_refinement_pass1_user_prompt(
        institution_id,
        entity_type,
        manifest,
        validation_errors,
        schema_contract,
    )
    raw = llm_complete(system, user)
    data = json.loads(strip_json_fences(raw))
    if not isinstance(data, dict):
        raise ValueError("Pass 1 LLM output must be a JSON object")
    return data


def _run_pass2(
    hitl_flag: dict[str, Any],
    schema_contract: EnrichedSchemaContractForSMA,
    llm_complete: Callable[[str, str], str],
    *,
    max_options: int = 3,
) -> dict[str, Any]:
    system = build_refinement_pass2_system_prompt()
    user = build_refinement_pass2_user_prompt(
        hitl_flag, schema_contract, max_options=max_options
    )
    raw = llm_complete(system, user)
    data = json.loads(strip_json_fences(raw))
    if not isinstance(data, dict):
        raise ValueError("Pass 2 LLM output must be a JSON object")
    return data


def run_sma_refinement(
    institution_id: str,
    entity_type: str,
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
    schema_contract: EnrichedSchemaContractForSMA,
    resolved_by: str | None = None,
    *,
    llm_complete: Callable[[str, str], str] | None = None,
    pass2_max_options: int = 3,
) -> tuple[FieldMappingManifest, "InstitutionSMAHITLItems"]:
    """
    Two-pass SMA refinement.

    Pass 1: refine manifest + identify HITL flags (no options).
    Pass 2: generate options for each flag (one call per flag; parallelizable by caller).

    Parameters
    ----------
    resolved_by:
        Optional audit label (reserved for future logging).
    llm_complete:
        Callable ``(system_prompt, user_prompt) -> raw_text`` compatible with the
        Databricks gateway pattern. If omitted, uses the default gateway client
        (requires ``DATABRICKS_TOKEN`` and gateway env configuration).

    Returns
    -------
    Refined manifest and complete :class:`~edvise.genai.mapping.schema_mapping_agent.hitl.schemas.InstitutionSMAHITLItems`.
    """
    from edvise.genai.mapping.schema_mapping_agent.hitl.schemas import (
        InstitutionSMAHITLItems,
        SMAHITLItem,
    )

    _ = resolved_by
    complete = llm_complete if llm_complete is not None else _default_llm_complete()

    pass1_result = _run_pass1(
        institution_id,
        entity_type,
        manifest,
        validation_errors,
        schema_contract,
        complete,
    )
    refined_raw = pass1_result.get("refined_manifest")
    if refined_raw is None:
        raise ValueError("Pass 1 output missing refined_manifest")
    refined_manifest = FieldMappingManifest.model_validate(refined_raw)

    hitl_raw = pass1_result.get("hitl_flags")
    if hitl_raw is None:
        hitl_flags: list[dict[str, Any]] = []
    elif not isinstance(hitl_raw, list):
        raise ValueError("Pass 1 hitl_flags must be a list")
    else:
        hitl_flags = [h for h in hitl_raw if isinstance(h, dict)]

    warnings = _enforce_review_status_contract(
        refined_manifest,
        validation_errors,
        hitl_flags,
    )
    for w in warnings:
        print(f"⚠  {w}")

    if not hitl_flags:
        return refined_manifest, InstitutionSMAHITLItems(
            institution_id=institution_id,
            entity_type=entity_type,
            items=[],
        )

    hitl_items: list[Any] = []
    for flag in hitl_flags:
        item = _run_pass2(
            flag,
            schema_contract,
            complete,
            max_options=pass2_max_options,
        )
        hitl_items.append(SMAHITLItem.model_validate(item))

    return refined_manifest, InstitutionSMAHITLItems(
        institution_id=institution_id,
        entity_type=entity_type,
        items=hitl_items,
    )


__all__ = [
    "_enforce_review_status_contract",
    "apply_refinement_review_status_safety_net",
    "build_refinement_combined_pass1_system_prompt",
    "build_refinement_combined_pass1_user_prompt",
    "build_refinement_combined_system_prompt",
    "build_refinement_combined_user_prompt",
    "build_refinement_pass1_system_prompt",
    "build_refinement_pass1_user_prompt",
    "build_refinement_pass2_system_prompt",
    "build_refinement_pass2_user_prompt",
    "build_refinement_system_prompt",
    "build_refinement_user_prompt",
    "log_refinement_contract_warnings_to_mlflow",
    "run_sma_refinement",
]
