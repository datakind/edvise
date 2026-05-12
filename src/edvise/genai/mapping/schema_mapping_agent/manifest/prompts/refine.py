"""
SMA refinement + HITL: prompts, orchestration, and post-parse safety nets (single module).

**Pass 1** — refinement + HITL flagging (slim JSON: ``field_statuses``,
``refined_corrections``, ``hitl_flags``; no full manifest): one LLM call per entity
(cohort and course are separate calls).

**Pass 2** — option generation: one LLM call per entity with all Pass 1 flags for
that entity in a single ``items`` array (cohort + course = 2 Pass 2 calls per institution;
4 gateway calls total per institution). Each TERMINAL option is checked with the same
deterministic ``validate_manifest`` pass as post–Step 2a generate (scratch manifest);
failures trigger :func:`~edvise.utils.llm_utils.llm_complete_with_parse_retry`.

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
    build_refinement_pass2_user_prompt(
        institution_id, entity_type, hitl_flags, schema_contract, refined_manifest=...
    ) -> str

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
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter, ValidationError

if TYPE_CHECKING:
    from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas import (
        InstitutionSMAHITLItems,
    )

from edvise.genai.mapping.shared.hitl.confidence import (
    PIPELINE_HITL_CONFIDENCE_THRESHOLD,
)
from edvise.genai.mapping.shared.schema_contract.schemas import (
    EnrichedSchemaContractForSMA,
)

from ..schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    ReviewStatus,
    get_compact_manifest_schema_reference,
)
from ..validation import ManifestValidationError, infer_manifest_base_table

from edvise.utils.llm_utils import llm_complete_with_parse_retry

from .generate import strip_json_fences

# Same value as ``manifest.hitl.schemas.HITL_CONFIDENCE_THRESHOLD`` — defined here so prompt + runtime
# code share one constant without importing ``hitl`` at module load (circular with ``artifacts``).
HITL_CONFIDENCE_THRESHOLD = PIPELINE_HITL_CONFIDENCE_THRESHOLD

_LOG = logging.getLogger(__name__)

# Shared one-liner: executor + validation agree on base-table inference (see validate_manifest).
_BASE_TABLE_JOIN_HINT = (
    "Omitted join ⇒ source_column is read only from the inferred base table (any mapping's "
    "join.base_table, else the mode of source_table). For CROSS_TABLE_REQUIRES_JOIN or "
    "wrong-table sourcing, add a full join + lookup source_table and keys — not source_table alone."
)

# Pass 2: TERMINAL options are scratch-validated per FieldMappingRecord — models often
# forget join on low_confidence even when Step 2a had zero manifest-level errors.
_PASS2_TERMINAL_EXECUTION_BASE_RULE = """
  EXECUTION BASE TABLE — applies to EVERY TERMINAL option (all failure_mode values):
    The executor infers one execution base table per entity manifest: the first mapping row
    that declares a non-null join uses that row's join.base_table; otherwise the mode
    (most frequent) of non-null source_table across mappings. Pass 2 may also print this
    value explicitly in the user message — treat it as authoritative when present.

    For each TERMINAL option's field_mapping (including option 1 and "Keep original mapping"):
    - If source_table is null (e.g. leave_unmapped): join must be null — OK.
    - If source_table equals the inferred execution base: join is usually null (same-table read).
    - If source_table is non-null and differs from the inferred base: join MUST be non-null with
      join.base_table = inferred base, join.lookup_table = field_mapping.source_table, and
      join_keys that exist on both tables after column_aliases resolution; use column_alias on
      the SMAHITLOption when names differ across tables.
    Output that violates the above is rejected with CROSS_TABLE_REQUIRES_JOIN before the run
    succeeds — do not emit cross-table TERMINAL rows without a complete join.

    If you cannot justify join_keys defensibly, do not offer that cross-table TERMINAL; use
    same-base-table alternatives, leave_unmapped, or keep the original mapping instead.
"""


def _parse_sma_refinement_llm_dict(raw: str) -> dict[str, Any]:
    """Parse gateway JSON (after fence strip) to a top-level object; use Pydantic errors for retry."""
    return TypeAdapter(dict[str, Any]).validate_json(strip_json_fences(raw))


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
  options: list[SMAHITLOption],              # 3-5 options, last always option_id="direct_edit"
  choice: null,              # always null — reviewer sets this
  reviewer_note: null,       # always null — reviewer sets this
  direct_edit_field_mapping: FieldMappingRecord  # REQUIRED: deep copy of current_field_mapping (JSON) for reviewer to edit
}

ColumnAlias: {table: str!, source_column: str!, canonical_column: str!, rationale?: str}

ReviewStatus (derived from Pass 1 ``field_statuses`` and merged onto each FieldMappingRecord):
  "auto_approved"    — passed validation + confidence threshold, no changes made
  "refined_by_llm"   — Pass 1 corrected with confidence above threshold (deterministic fix)
  "refined_and_proposed_for_hitl" — Pass 1 corrected but confidence at/below threshold (still HITL)
  "proposed_for_hitl" — Pass 1 could not fix or low confidence with no correction (HITL)
"""

_PASS1_OUTPUT_SCHEMA = """
Pass 1 output — respond with a single JSON object, no preamble, no markdown:
{
  "field_statuses": {
    "<target_field>": "auto_approved" | "refined_by_llm" | "refined_and_proposed_for_hitl" | "proposed_for_hitl"
    // one entry per field in the manifest — every field must be present
  },
  "refined_corrections": {
    "<target_field>": {
      // only keys that changed from the input record
      // valid keys: source_column, source_table, join, row_selection,
      //             rationale, validation_notes
      // never include: confidence, review_status, reviewer_notes,
      //                corrected_source_column, target_field
    }
    // only for fields with field_statuses[target_field]="refined_by_llm" or "refined_and_proposed_for_hitl"
    // omit key entirely if no fields were corrected
  },
  "hitl_flags": [
    {
      "item_id": "{institution_id}_{entity_type}_{target_field}_{failure_mode}",
      "institution_id": str,
      "entity_type": "cohort" | "course",
      "target_field": str,
      "failure_mode": "low_confidence" | "column_not_found" | "join_structure" | "row_selection" | "map_unmap",
      "hitl_question": str,
      "hitl_context": str | null,
      "current_field_mapping": FieldMappingRecord,  // original input record, copied unchanged
      "validation_errors": list[str]
    }
    // one entry per field with field_statuses[target_field]="proposed_for_hitl" or "refined_and_proposed_for_hitl"
    // may be empty []
  ]
}

CRITICAL:
  - field_statuses must contain every target_field — no omissions.
  - refined_corrections contains ONLY changed keys — never re-output unchanged fields.
  - current_field_mapping in hitl_flags is ALWAYS the original input record, never modified.
  - Do not generate options in Pass 1.
  - Do not change confidence on any field.
  - Every proposed_for_hitl or refined_and_proposed_for_hitl field must appear in hitl_flags.
  - Every refined_by_llm or refined_and_proposed_for_hitl field with a correction must appear in refined_corrections.
"""

_PASS1_OUTPUT_SCHEMA_COMBINED = """
Pass 1 combined output (multiple entities in one response) — single JSON object, no preamble, no markdown:
{
  "field_statuses_by_entity": {
    "<entity_type>": {
      "<target_field>": "auto_approved" | "refined_by_llm" | "refined_and_proposed_for_hitl" | "proposed_for_hitl"
    }
    // one entry per entity listed in the user message — every target_field per entity
  },
  "refined_corrections_by_entity": {
    "<entity_type>": {
      "<target_field>": { /* only changed keys, same rules as refined_corrections */ }
    }
    // omit entity key if no corrections for that entity
  },
  "hitl_flags_by_entity": {
    "<entity_type>": [ /* same shape as hitl_flags in single-entity Pass 1 */ ]
  }
}

CRITICAL:
  - field_statuses_by_entity, refined_corrections_by_entity, and hitl_flags_by_entity
    must contain exactly the same entity_type keys as listed in the user message.
  - Per-entity rules match single-entity Pass 1 (complete field_statuses, corrections, flags).
  - Do not emit full manifests — slim keys only.
"""

_PASS2_OUTPUT_SCHEMA = """
Pass 2 output — respond with a single JSON object, no preamble, no markdown:
{
  "items": [
    {
      "item_id": str,            // copied from input flag
      "institution_id": str,
      "entity_type": str,
      "target_field": str,
      "failure_mode": str,
      "hitl_question": str,
      "hitl_context": str | null,
      "current_field_mapping": FieldMappingRecord,  // copied from input flag, unchanged
      "validation_errors": list[str],
      "options": [
        // list of SMAHITLOption — 2-4 TERMINAL (reentry=terminal, field_mapping=FieldMappingRecord)
        // + always one final direct_edit option (see HITL output schema above)
        // 3-5 options total
        // option 1 is always your recommended fix
        // include original mapping as an option if still plausible
        // include one TERMINAL "leave_unmapped" option when applicable (see OPTION RULES)
      ],
      "choice": null,
      "reviewer_note": null,
      "direct_edit_field_mapping": { }  // same FieldMappingRecord JSON as current_field_mapping — reviewer starting point
    }
    // one item per flag in the input
  ]
}
"""

_AUTO_FIX_RULES = """
FIELD STATUS — use exactly one status per target_field:

  - Use "refined_and_proposed_for_hitl" when confidence <= {threshold} AND you made
    a correction. The correction goes in refined_corrections, the flag goes in
    hitl_flags. Confidence at or below threshold always triggers HITL — the
    correction is surfaced as option 1 for the reviewer to confirm or override.
  - Use "proposed_for_hitl" when confidence <= {threshold} AND you made no correction.
  - Use "refined_by_llm" ONLY when confidence > {threshold} AND the fix is
    unambiguous and deterministic.
  - Use "auto_approved" ONLY when confidence > {threshold} AND no validation errors.

HIGH CONFIDENCE — validation errors:
  - Deterministic fix (typo, structural): set field_statuses[target_field]="refined_by_llm",
    emit deltas in refined_corrections, no hitl_flags entry for that field.
  - Fix requires judgment or is ambiguous: set field_statuses[target_field]="proposed_for_hitl"
    and emit hitl_flags.

LOW CONFIDENCE:
  - Never use "refined_by_llm" or "auto_approved".

REFINED_CORRECTIONS / HITL_FLAGS:
  - refined_corrections: only for fields with status refined_by_llm or refined_and_proposed_for_hitl
    (only changed keys).
  - hitl_flags: required for every field with status proposed_for_hitl or refined_and_proposed_for_hitl.
  - Do not change confidence — it reflects the generating agent's uncertainty.

FAILURE_MODE vs Pass 2 (improves option quality):
  - Step 2a may report zero deterministic validation errors while a mapping is still weak or uses
    another table without a join. When the reviewer decision will require fixing join keys,
    lookup table linkage, or declaring base↔lookup join_keys, set failure_mode to join_structure
    (priority over low_confidence per the collapse rule when both apply). Pass 2 then applies
    join_structure option rules explicitly. Use low_confidence only when the issue is not
    primarily join/cross-table linkage.

OTHER:
  - Fields in the auto_approved_fields list must have field_statuses[target_field]="auto_approved"
    and NO refined_corrections entry and NO hitl_flags entry.
  - {base_table_join_hint}
""".format(
    threshold=HITL_CONFIDENCE_THRESHOLD,
    base_table_join_hint=_BASE_TABLE_JOIN_HINT,
)

_OPTION_GENERATION_RULES = f"""
OPTION RULES — Pass 2 only; you receive all Pass 1 flags for one entity in one batch:

  CRITICAL — current_field_mapping:
    current_field_mapping in each output item must be copied unchanged from the corresponding
    Pass 1 hitl_flag. Do not modify it. Your recommended fix is option 1 in the options list.

  - Maximum 4 TERMINAL options + 1 direct_edit = 5 total.
  - Minimum 2 TERMINAL options + 1 direct_edit = 3 total.
  - Last option ALWAYS: option_id="direct_edit", reentry="direct_edit",
    field_mapping=null, column_alias=null (see HITL output schema for labels).
  - Each TERMINAL option is a complete FieldMappingRecord inside SMAHITLOption.field_mapping.
{_PASS2_TERMINAL_EXECUTION_BASE_RULE}
  - column_alias on any TERMINAL option when join keys need a name bridge (not only join_structure).
  - Options must be meaningfully distinct — no near-duplicates.
  - Option 1 is your recommended fix, labeled clearly.
  - Include original mapping as an option labeled "Keep original mapping"
    if it is still plausible.

  LEAVE UNMAPPED (required TERMINAL whenever the field may legitimately have no source):
    - Include exactly one TERMINAL option with option_id="leave_unmapped" (unless every
      TERMINAL you emit is already either fully unmapped or fully mapped and this would duplicate).
    - Its field_mapping must be a complete FieldMappingRecord with the same target_field as
      current_field_mapping, and source_column=null, source_table=null, join=null, row_selection=null.
      Use confidence 1.0 when affirming intentional unmappable; otherwise keep the flag's confidence
      if that is more honest.
    - This option must appear before the final direct_edit option.

  direct_edit_field_mapping (required on every item):
    - Set to the same FieldMappingRecord JSON object as current_field_mapping (deep copy).
    - The reviewer edits this blob in the UI; do not leave it null.

BY FAILURE MODE:

  low_confidence:
    - Option 1: your recommended mapping. If it sources from a non-base table, it MUST include
      the full join block (see EXECUTION BASE TABLE above) — low_confidence is not an excuse
      to omit join.
    - Include leave_unmapped as described above.
    - Then: "Keep original mapping" if still plausible, and/or an alternative candidate.
    - Alternatives that read term/degree/etc. columns must each be join-complete or be rejected;
      prefer same-base-table TERMINAL options when join_keys would be guesswork.

  column_not_found:
    - Options are close-match column candidates ordered by similarity.
    - If a candidate column lives on a table other than the inferred execution base, the
      TERMINAL field_mapping must include the full join (same rule as EXECUTION BASE TABLE).
    - Include leave_unmapped when none of the candidates are acceptable.

  join_structure:
    - {_BASE_TABLE_JOIN_HINT}
    - Options are valid join key combinations from the schema contract.
    - Include column_alias on options that bridge a name mismatch.
    - Include "Remove join (same-table field)" as an option if applicable.
    - Include leave_unmapped when the field should not be sourced.

  row_selection:
    - Options are valid RowSelectionStrategy alternatives with args pre-filled.
    - Order by most likely correct given field semantics.
    - For wide-row conferral / certificate-date proxies, options may add or adjust `row_selection.filter` on a same-row
      award-type discriminator (join null) — filtered-out rows become null for that field only.
    - Include leave_unmapped when row selection cannot be fixed defensibly.

  map_unmap:
    - At least 2 TERMINAL options before direct_edit: one mapped and one unmapped (swap order
      so option 1 is your recommendation). You may add a third TERMINAL if there is another
      distinct mapped alternative. Always include leave_unmapped if it is not already one of those two.
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

_COHORT_TARGET_SEMANTICS_FOR_REFINEMENT = """
## Cohort target semantics (RawEdviseStudentDataSchema)

When the manifest section is **cohort** (student / RawEdviseStudentDataSchema):

- **`intended_program_type`** and **`declared_major_at_entry`** are **entry-time snapshots**
  (conceptually aligned with `entry_year` / `entry_term`). Using **current** primary program/major,
  **`last_by` / latest term** row selection, or **completion / exit** fields for these targets is a
  **semantic error** — not merely weak confidence.
- Prefer corrections that map to **admit / entry / first-term / cohort** sources, or longitudinal
  data filtered to **at or before entry** (`first_by` on term order, not latest). Document any
  remaining proxy in **validation_notes**.
- If Step 2a mapped either field from an obvious **current-only** column or latest-row semantics,
  do **not** leave it as **auto_approved** when an entry-aligned source exists in the contract;
  apply **refined_by_llm** / **refined_and_proposed_for_hitl** or **proposed_for_hitl** as appropriate.
- Never use **`major_at_completion`** mapping logic for **`declared_major_at_entry`** unless the
  source column is provably entry-time (unusual).
"""

_COHORT_SEMANTICS_PASS2 = """
## Cohort entry vs outcome (Pass 2 options)

For flags on **`intended_program_type`** or **`declared_major_at_entry`**, TERMINAL options should
favor **entry / admit / first-term** sources. If the flagged mapping used **current** or **latest**
major/program semantics, option 1 should normally be the entry-aligned alternative; describe any
remaining proxy honestly. Options for **`major_at_completion`** must not be reused as substitutes
for **`declared_major_at_entry`** unless the column meaning supports entry time.
"""

_PASS1_OUTPUT_FORMAT = """
OUTPUT FORMAT — respond with a single JSON object, no preamble, no markdown:
{
  "field_statuses": { ...every target_field... },
  "refined_corrections": { ...optional — refined_by_llm / refined_and_proposed_for_hitl deltas... },
  "hitl_flags": [ ...optional — proposed_for_hitl / refined_and_proposed_for_hitl... ]
}

CRITICAL:
  - field_statuses must list every target_field from the manifest — no omissions.
  - Do not invent columns or tables not present in the schema contract.
  - Do not change confidence on any field.
  - Do not emit options in Pass 1.
  - Do not output a full manifest — field_statuses + refined_corrections + hitl_flags only.
"""

_PASS1_OUTPUT_FORMAT_COMBINED = """
OUTPUT FORMAT — respond with a single JSON object, no preamble, no markdown:
{
  "field_statuses_by_entity": {
    "<entity_type>": { ...every target_field for that entity... },
    ...
  },
  "refined_corrections_by_entity": {
    "<entity_type>": { ...optional per-entity refined_corrections... },
    ...
  },
  "hitl_flags_by_entity": {
    "<entity_type>": [ ...Pass 1 flags for that entity only — no options... ],
    ...
  }
}

CRITICAL:
  - All three top-level objects must contain exactly the same entity_type keys
    as listed in the user message (e.g. cohort and course).
  - Each field_statuses_by_entity entry must list every target_field for that entity.
  - Do not invent columns or tables not present in the schema contract.
  - Do not change confidence on any field.
  - Do not emit options in Pass 1.
  - Do not output full manifests — slim keys only.
"""


def _build_pass1_system_prompt(*, output_format: str, pass1_schema: str) -> str:
    return f"""You are Pass 1 of the Schema Mapping Agent refinement step for the Edvise institution onboarding pipeline.

Architecture: Pass 1 emits slim JSON (field_statuses + refined_corrections + hitl_flags; no full manifest).
Pass 2 generates reviewer options for all flags from Pass 1 for each entity. You only run Pass 1.

Your job is to review a field mapping manifest produced by the original 2a LLM, correct errors where
possible, and emit structured hitl_flags for everything you cannot confidently fix — without
generating options (that is Pass 2).

## Manifest schema

{_MANIFEST_SCHEMA}

## Pass 1 output schema

{pass1_schema}

## When to auto-correct vs. escalate

{_AUTO_FIX_RULES}

{_COHORT_TARGET_SEMANTICS_FOR_REFINEMENT}

## Output format

{output_format}
"""


def build_refinement_pass1_system_prompt() -> str:
    return _build_pass1_system_prompt(
        output_format=_PASS1_OUTPUT_FORMAT,
        pass1_schema=_PASS1_OUTPUT_SCHEMA,
    )


def build_refinement_combined_pass1_system_prompt() -> str:
    return _build_pass1_system_prompt(
        output_format=_PASS1_OUTPUT_FORMAT_COMBINED,
        pass1_schema=_PASS1_OUTPUT_SCHEMA_COMBINED,
    )


def build_refinement_pass2_system_prompt() -> str:
    return f"""You are Pass 2 of the Schema Mapping Agent refinement for the Edvise institution onboarding pipeline.

Pass 1 produced hitl_flags for one entity (cohort or course). Your job is to return a single JSON object
with an "items" array: one complete SMAHITLItem per flag, each with reviewer-facing options
(SMAHITLOption TERMINAL entries with complete FieldMappingRecord payloads) plus a final direct_edit
escape hatch on every item.

## Manifest schema (compact reference)

{_MANIFEST_SCHEMA}

## Output schemas

{_HITL_OUTPUT_SCHEMA}

{_PASS2_OUTPUT_SCHEMA}

## Option generation rules

{_OPTION_GENERATION_RULES}

## Collapsing multiple errors per field

{_FIELD_COLLAPSE_RULE}

{_COHORT_SEMANTICS_PASS2}

CRITICAL — current_field_mapping:
  current_field_mapping in each item must be copied unchanged from the corresponding Pass 1 hitl_flag.
  Never modify it. Corrections belong in options (option 1 = recommended).

CRITICAL — TERMINAL scratch validation:
  Every TERMINAL option is checked with the same validate_manifest rules as Step 2a on a scratch
  manifest. Cross-table TERMINAL rows without join are rejected (retries exhaust). Follow
  EXECUTION BASE TABLE in the option rules and any explicit base table line in the user message.
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
        r.target_field
        for r in manifest.mappings
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
    low = [r for r in manifest.mappings if r.confidence <= HITL_CONFIDENCE_THRESHOLD]
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
        Institution identifier e.g. "synthetic_coastal_cc".
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

## Your task
1. For each flagged field (validation errors or confidence <= {HITL_CONFIDENCE_THRESHOLD}), attempt
   to auto-correct when the fix is unambiguous and deterministic.
   - If confidence > {HITL_CONFIDENCE_THRESHOLD}: set field_statuses[target_field]="refined_by_llm"
     and include only changed keys in refined_corrections (no hitl_flag for that field unless also needed for another reason).
   - If confidence <= {HITL_CONFIDENCE_THRESHOLD} and you made a correction: set
     field_statuses[target_field]="refined_and_proposed_for_hitl", put deltas in refined_corrections,
     and add a hitl_flag (correction is option 1 in Pass 2).

2. For fields you cannot confidently fix (including low confidence with no correction), set
   field_statuses[target_field]="proposed_for_hitl" and add a hitl_flag with current_field_mapping
   copied unchanged from the input.

3. For all remaining fields, set field_statuses[target_field]="auto_approved"
   (confidence > {HITL_CONFIDENCE_THRESHOLD} and no validation errors).
   Do not include them in refined_corrections or hitl_flags.

4. Return the single JSON object described in your instructions.
   Do not output a full manifest — field_statuses + refined_corrections + hitl_flags only.
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

You must return JSON with ``field_statuses_by_entity``, ``refined_corrections_by_entity``, and
``hitl_flags_by_entity`` containing exactly these keys: {entity_keys}.

## Schema contract — available tables and columns (shared)
{schema_summary}

{chr(10).join(blocks)}

## Your task
For EACH entity section above, apply the refinement rules independently (same as single-entity Pass 1).

1. For each flagged field (validation errors or confidence <= {HITL_CONFIDENCE_THRESHOLD}), attempt
   to auto-correct when the fix is unambiguous and deterministic.
   - If confidence > {HITL_CONFIDENCE_THRESHOLD}: set field_statuses_by_entity[entity][target_field]="refined_by_llm"
     and include only changed keys in refined_corrections_by_entity.
   - If confidence <= {HITL_CONFIDENCE_THRESHOLD} and you made a correction: set
     field_statuses_by_entity[entity][target_field]="refined_and_proposed_for_hitl", put deltas in
     refined_corrections_by_entity, and add a hitl_flags_by_entity[entity] entry.

2. For fields you cannot confidently fix (including low confidence with no correction), set
   field_statuses_by_entity[entity][target_field]="proposed_for_hitl" and add hitl_flags_by_entity[entity]
   with current_field_mapping copied unchanged from the input.

3. For all remaining fields per entity, set field_statuses_by_entity[entity][target_field]="auto_approved"
   (confidence > {HITL_CONFIDENCE_THRESHOLD} and no validation errors).
   Do not include them in refined_corrections or hitl_flags.

4. Return the single combined JSON object described in your instructions.
   Do not output full manifests — field_statuses_by_entity + refined_corrections_by_entity +
   hitl_flags_by_entity only.
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
    institution_id: str,
    entity_type: str,
    hitl_flags: list[dict],
    schema_contract: EnrichedSchemaContractForSMA,
    *,
    refined_manifest: FieldMappingManifest | None = None,
) -> str:
    """
    Build the user prompt for SMA refinement Pass 2 (options for all Pass 1 flags for one entity).

    Parameters
    ----------
    institution_id:
        Institution identifier e.g. ``"synthetic_coastal_cc"``.
    entity_type:
        ``"cohort"`` or ``"course"``.
    hitl_flags:
        Pass 1 ``hitl_flags`` for this entity (list of dict-shaped flags).
    schema_contract:
        EnrichedSchemaContractForSMA from IdentityAgent output.
    refined_manifest:
        Manifest after Pass 1 merge. When provided, the prompt includes the inferred
        execution base table (same rule as ``validate_manifest`` / executor) so Pass 2
        can align cross-table TERMINAL options with explicit ``join`` blocks.
    """
    flags_json = json.dumps(hitl_flags, indent=2, default=str)
    schema_summary = _format_schema_contract_summary(schema_contract)

    base_section = ""
    if refined_manifest is not None:
        try:
            inferred_base = infer_manifest_base_table(refined_manifest)
        except ValueError:
            inferred_base = None
        if inferred_base is not None:
            base_section = f"""
## Inferred execution base table (deterministic — TERMINAL options must align)

For this entity's manifest after Pass 1, the pipeline infers execution base table: **{inferred_base}**.

Any TERMINAL `field_mapping` with non-null `source_table` not equal to `{inferred_base}` must include
non-null `join` with `join.base_table` = `{inferred_base}`, `join.lookup_table` = that row's
`source_table`, and valid `join_keys` for both tables (canonical names per manifest `column_aliases`;
use `column_alias` on the option when bridging a rename). Same-table reads from `{inferred_base}` keep `join` null.
"""

    return f"""## Institution
institution_id: {institution_id}
entity_type: {entity_type}
{base_section}
## Schema contract — available tables and columns
{schema_summary}

## HITL flags requiring options
{flags_json}

## Your task
For each flag in the list above, generate a complete SMAHITLItem with options.
Follow the option generation rules in your instructions.
Return a single JSON object with an "items" array containing one entry per flag.
"""


def _apply_pass1_result(
    institution_id: str,
    input_manifest: FieldMappingManifest,
    pass1_result: dict[str, Any],
) -> tuple[FieldMappingManifest, list[dict[str, Any]]]:
    """
    Reconstruct full manifest from Pass 1 slim output.
    Merges refined_corrections onto input records.
    Sets review_status on every record from field_statuses.
    """
    field_statuses = pass1_result.get("field_statuses")
    if not isinstance(field_statuses, dict):
        raise ValueError("Pass 1 output missing or invalid field_statuses")
    refined_corrections = pass1_result.get("refined_corrections") or {}
    if not isinstance(refined_corrections, dict):
        refined_corrections = {}
    hitl_raw = pass1_result.get("hitl_flags")
    if hitl_raw is None:
        hitl_flags: list[dict[str, Any]] = []
    elif not isinstance(hitl_raw, list):
        raise ValueError("Pass 1 hitl_flags must be a list")
    else:
        hitl_flags = [h for h in hitl_raw if isinstance(h, dict)]

    entity_type = input_manifest.entity_type
    et_str = entity_type.value if hasattr(entity_type, "value") else str(entity_type)

    updated_mappings: list[FieldMappingRecord] = []
    for record in input_manifest.mappings:
        tf = record.target_field
        status = field_statuses.get(tf)
        if status is None:
            print(
                f"⚠  Pass 1 missing status for '{tf}' — defaulting to proposed_for_hitl"
            )
            status = "proposed_for_hitl"
            hitl_flags.append(
                {
                    "item_id": f"{institution_id}_{et_str}_{tf}_missing_status",
                    "institution_id": institution_id,
                    "entity_type": et_str,
                    "target_field": tf,
                    "failure_mode": "low_confidence",
                    "hitl_question": (
                        f"Pass 1 did not return a status for {tf} — review manually."
                    ),
                    "hitl_context": None,
                    "current_field_mapping": record.model_dump(mode="json"),
                    "validation_errors": [],
                }
            )

        corrections = refined_corrections.get(tf) or {}
        if not isinstance(corrections, dict):
            corrections = {}
        if status not in ("refined_by_llm", "refined_and_proposed_for_hitl"):
            corrections = {}

        record_dict = record.model_dump(mode="json")
        record_dict.update(corrections)
        record_dict["review_status"] = status
        updated_mappings.append(FieldMappingRecord.model_validate(record_dict))

    refined_manifest = FieldMappingManifest(
        entity_type=input_manifest.entity_type,
        target_schema=input_manifest.target_schema,
        mappings=updated_mappings,
        column_aliases=input_manifest.column_aliases,
    )
    return refined_manifest, hitl_flags


# ---------------------------------------------------------------------------
# Orchestration + post-parse safety nets (same module as prompts)
# ---------------------------------------------------------------------------


def _hitl_target_fields(
    hitl: list[Mapping[str, Any]] | list[Any],
) -> set[str]:
    """Target fields covered by Pass 1 flags and/or Pass 2 items."""
    from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas import (
        SMAHITLItem,
    )

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
    refined_corrections: Mapping[str, Any],
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
            # LLM used refined_by_llm on a field at or below threshold.
            # Check if a correction exists in refined_corrections — if so,
            # downgrade to refined_and_proposed_for_hitl to preserve the correction
            # while still flagging for HITL. If no correction, force proposed_for_hitl.
            has_correction = record.target_field in refined_corrections
            new_status = (
                ReviewStatus.refined_and_proposed_for_hitl
                if has_correction
                else ReviewStatus.proposed_for_hitl
            )
            warnings.append(
                f"[review_status violation] '{record.target_field}' marked "
                f"refined_by_llm but confidence={record.confidence} <= "
                f"threshold={HITL_CONFIDENCE_THRESHOLD}. "
                f"Forcing to {new_status.value}."
            )
            record.review_status = new_status

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
    refined_corrections: Mapping[str, Any] | None = None,
    *,
    print_warnings: bool = True,
    log_mlflow: bool = True,
) -> list[str]:
    """
    Run :func:`_enforce_review_status_contract` and optionally print / log warnings.

    ``hitl_flags`` may be Pass 1 dicts or validated :class:`~edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas.SMAHITLItem` instances.

    ``refined_corrections`` is the Pass 1 slim-output map of target_field → correction deltas.
    When omitted (e.g. when re-checking artifacts without Pass 1 context), pass an empty dict
    or None — low-confidence ``refined_by_llm`` will downgrade to ``proposed_for_hitl``.
    """
    rc: dict[str, Any] = (
        dict(refined_corrections) if refined_corrections is not None else {}
    )
    warnings = _enforce_review_status_contract(
        manifest, validation_errors, hitl_flags, rc
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


def _run_pass1_llm_call(
    institution_id: str,
    entity_type: Literal["cohort", "course"],
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
    return llm_complete_with_parse_retry(
        llm_complete,
        system,
        user,
        _parse_sma_refinement_llm_dict,
        logger=_LOG,
    )


def _run_pass2_llm_call(
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    hitl_flags: list[dict[str, Any]],
    schema_contract: EnrichedSchemaContractForSMA,
    refined_manifest: FieldMappingManifest,
    llm_complete: Callable[[str, str], str],
) -> dict[str, Any]:
    from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.option_validation import (
        raise_if_pass2_terminal_options_invalid,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas import (
        SMAHITLItem,
        prefill_sma_hitl_direct_edit_if_missing,
    )

    system = build_refinement_pass2_system_prompt()
    user = build_refinement_pass2_user_prompt(
        institution_id,
        entity_type,
        hitl_flags,
        schema_contract,
        refined_manifest=refined_manifest,
    )

    def _parse_pass2_with_terminal_validation(raw: str) -> dict[str, Any]:
        data = _parse_sma_refinement_llm_dict(raw)
        items_raw = data.get("items")
        if items_raw is None:
            ve = ValueError(
                "Pass 2 output must include top-level key 'items' (array of SMAHITLItem)."
            )
            raise ValidationError.from_exception_data(
                "Pass2JSON",
                [
                    {
                        "type": "missing",
                        "loc": ("items",),
                        "input": data,
                        "ctx": {"error": ve},
                    }
                ],
            )
        if not isinstance(items_raw, list):
            ve = ValueError("Pass 2 'items' must be a JSON array.")
            raise ValidationError.from_exception_data(
                "Pass2JSON",
                [
                    {
                        "type": "value_error",
                        "loc": ("items",),
                        "input": items_raw,
                        "ctx": {"error": ve},
                    }
                ],
            )
        hitl_items = [SMAHITLItem.model_validate(item) for item in items_raw]
        hitl_items = [prefill_sma_hitl_direct_edit_if_missing(i) for i in hitl_items]
        raise_if_pass2_terminal_options_invalid(
            refined_manifest, hitl_items, schema_contract
        )
        data["items"] = [i.model_dump(mode="json") for i in hitl_items]
        return data

    return llm_complete_with_parse_retry(
        llm_complete,
        system,
        user,
        _parse_pass2_with_terminal_validation,
        logger=_LOG,
    )


def run_sma_refinement(
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    manifest: FieldMappingManifest,
    validation_errors: list[ManifestValidationError],
    schema_contract: EnrichedSchemaContractForSMA,
    resolved_by: str | None = None,
    *,
    llm_complete: Callable[[str, str], str] | None = None,
) -> tuple[FieldMappingManifest, InstitutionSMAHITLItems]:
    """
    Two-pass SMA refinement for one entity (cohort or course).
    Call once for cohort, once for course — 2 calls per entity = 4 total per institution.

    Pass 1: slim output — field_statuses + refined_corrections + hitl_flags
    Pass 2: option generation for all hitl_flags in one call

    Parameters
    ----------
    resolved_by:
        Optional audit label (reserved for future logging).
    llm_complete:
        Callable ``(system_prompt, user_prompt) -> raw_text`` compatible with the
        Databricks gateway pattern. If omitted, uses the default gateway client
        (Databricks SDK default auth and gateway env configuration).

    Returns
    -------
    Refined manifest and complete :class:`~edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas.InstitutionSMAHITLItems`.
    """
    from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas import (
        InstitutionSMAHITLItems,
        SMAHITLItem,
        prefill_sma_hitl_direct_edit_if_missing,
    )

    _ = resolved_by
    complete = llm_complete if llm_complete is not None else _default_llm_complete()

    pass1_raw = _run_pass1_llm_call(
        institution_id,
        entity_type,
        manifest,
        validation_errors,
        schema_contract,
        complete,
    )
    refined_manifest, hitl_flags = _apply_pass1_result(
        institution_id, manifest, pass1_raw
    )

    _rc = pass1_raw.get("refined_corrections") or {}
    refined_corrections_pass1: dict[str, Any] = _rc if isinstance(_rc, dict) else {}
    warnings = _enforce_review_status_contract(
        refined_manifest,
        validation_errors,
        hitl_flags,
        refined_corrections_pass1,
    )
    for w in warnings:
        print(f"⚠  {w}")

    if not hitl_flags:
        print(
            f"✓ No HITL flags for {entity_type} — all fields auto-approved or refined."
        )
        return refined_manifest, InstitutionSMAHITLItems(
            institution_id=institution_id,
            entity_type=entity_type,
            items=[],
        )

    pass2_raw = _run_pass2_llm_call(
        institution_id,
        entity_type,
        hitl_flags,
        schema_contract,
        refined_manifest,
        complete,
    )
    items_raw = pass2_raw.get("items")
    if items_raw is None:
        raise ValueError("Pass 2 output missing items")
    if not isinstance(items_raw, list):
        raise ValueError("Pass 2 items must be a list")
    hitl_items = [
        prefill_sma_hitl_direct_edit_if_missing(SMAHITLItem.model_validate(item))
        for item in items_raw
    ]

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
