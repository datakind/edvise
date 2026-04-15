"""
Step 2a Prompt Assembly — SchemaMappingAgent Field Mapping
Builds the prompt for generating a mapping manifest from a schema contract + reference examples.
"""

import json
from typing import Any, Literal, cast

from edvise.genai.prompt_token_audit import audit_prompt_sections

from edvise.genai.mapping.schema_contract import (
    EnrichedSchemaContractForSMA,
    parse_enriched_schema_contract_for_sma,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    get_compact_manifest_schema_reference,
    get_manifest_schema_context,
)


def _manifest_schema_for_prompt(*, compact: bool = True) -> str:
    return (
        get_compact_manifest_schema_reference()
        if compact
        else get_manifest_schema_context()
    )


def extract_schema_descriptor(schema_class: Any) -> dict[str, Any]:
    """Extract a compact field descriptor from a Pandera DataFrameModel for prompt injection."""
    fields = {}

    for field_name, field_info in schema_class.__annotations__.items():
        pa_field = schema_class.__fields__.get(field_name)
        descriptor = {"type": str(field_info)}

        if pa_field is not None:
            if hasattr(pa_field, "description"):
                descriptor["description"] = pa_field.description
            if hasattr(pa_field, "nullable"):
                descriptor["nullable"] = pa_field.nullable

        fields[field_name] = descriptor

    return {"schema_name": schema_class.__name__, "fields": fields}


# ── Schema contract summarization ─────────────────────────────────────────────


def summarize_schema_contract(
    contract: dict[str, Any] | EnrichedSchemaContractForSMA,
) -> dict:
    """
    Slim down the schema contract for prompt injection.

    Input is validated/accepted as :class:`~edvise.genai.mapping.schema_contract.EnrichedSchemaContractForSMA`
    (IdentityAgent enriched JSON).

    Keeps: column names, dtypes (from each dataset's frozen ``dtypes`` map), null %,
    unique values (if present), sample values from ``training.column_details``.
    Drops: column_order_hash, normalization maps, file paths, boolean_map, null_tokens.
    """
    parsed = (
        contract
        if isinstance(contract, EnrichedSchemaContractForSMA)
        else parse_enriched_schema_contract_for_sma(contract)
    )
    summary: dict[str, Any] = {
        "school_id": parsed.school_id,
        "school_name": parsed.school_name,
        "datasets": {},
    }
    if parsed.learner_id_alias:
        summary["learner_id_alias"] = parsed.learner_id_alias
    if parsed.canonical_learner_column:
        summary["canonical_learner_column"] = parsed.canonical_learner_column

    for table_name, table_info in parsed.datasets.items():
        dtypes_map = table_info.dtypes or {}
        columns: list[dict[str, Any]] = []
        for col in table_info.training.column_details:
            norm = col.normalized_name
            # Single source of truth: frozen dtypes on the dataset (enforce_schema).
            # Legacy contracts may still carry inferred_dtype on column_details only.
            dtype = dtypes_map.get(norm) or col.inferred_dtype or "unknown"
            col_summary: dict[str, Any] = {
                "name": norm,
                "dtype": dtype,
                "null_pct": col.null_percentage,
                "sample_values": col.sample_values,
            }
            if col.unique_values is not None:
                col_summary["unique_values"] = col.unique_values
            columns.append(col_summary)

        summary["datasets"][table_name] = {
            "unique_keys": list(table_info.unique_keys),
            "num_rows": table_info.training.num_rows,
            "columns": columns,
        }

    return summary


def slim_reference_manifest(manifest: dict) -> dict:
    """
    Strip narrative and correction fields from a reference manifest for few-shot injection.
    Keeps: all structural fields, source_column, source_table, row_selection, join,
           column_aliases, confidence, rationale.
    Drops: reviewer_notes, corrected_source_column, validation_notes, review_status.
    """
    OMIT_FIELDS = {
        "reviewer_notes",
        "corrected_source_column",
        "validation_notes",
        "review_status",
    }

    slimmed = {k: v for k, v in manifest.items() if k != "manifests"}
    slimmed["manifests"] = {}

    for entity, entity_manifest in manifest.get("manifests", {}).items():
        mappings = [
            {k: v for k, v in record.items() if k not in OMIT_FIELDS}
            for record in entity_manifest.get("mappings", [])
        ]
        # Canonical FieldMappingManifest key order (matches Pydantic / gold): mappings before column_aliases
        slimmed_entity = {
            "entity_type": entity_manifest["entity_type"],
            "target_schema": entity_manifest["target_schema"],
            "mappings": mappings,
            "column_aliases": entity_manifest.get("column_aliases") or [],
        }
        slimmed["manifests"][entity] = slimmed_entity

    return slimmed


def _column_aliases_scope_bullet(scope: Literal["single_pass", "entity_pass"]) -> str:
    if scope == "single_pass":
        return """- column_aliases is scoped per manifest section — declare independently for cohort and
  course based only on joins present in that section. Never copy between sections."""
    return """- column_aliases is scoped per manifest section — declare them only for the entity you
  are producing in this response, based only on joins present in that section."""


def _step2a_cohort_entry_term_rules() -> str:
    """Step 2a: mapping rules for cohort entry_year / entry_term (manifest)."""
    return """
COHORT entry_year AND entry_term — decision hierarchy (apply in order)

**(1) Preferred — IdentityAgent normalized columns on the student (cohort base) table**
- When the schema contract **dtypes** for the cohort base table (typically `student`) include
  `_edvise_term_academic_year` and `_edvise_term_season`, map `entry_year` and `entry_term` to those
  columns with `source_table` = that base table and `row_selection.strategy`: `"any_row"`.
- Do **not** treat raw institutional term-code strings on student (e.g. `starting_cohort_term`) as an
  equivalent preferred source when `_edvise_term_*` are present.

**(2) Fallback — join to `term` or `course` (student-term grain)**
- Use only when **(1)** is impossible: `_edvise_term_*` are missing on the student row **and** a viable
  join exists from student → `term` or `course` on `student_id` only (see JOINS AND ALIASES).
- Resolve `entry_year` / `entry_term` from the lookup table's `_edvise_term_academic_year` /
  `_edvise_term_season` using `row_selection.strategy`: `"first_by"`, `row_selection.order_by`:
  `"_term_order"` (IdentityAgent chronological key).
- **Semantic caveat:** this path means **first term in enrollment history** (earliest term row), **not**
  necessarily institutional cohort / starting cohort. State that explicitly in **rationale** and
  **validation_notes**.
- Use **low confidence** (typically ≤ 0.65) and flag **HITL review** — this is a proxy, not cohort.

**(3) Unmapped — upstream IdentityAgent / enrichment gap**
- If neither **(1)** nor **(2)** is defensible, set `entry_year` / `entry_term` to unmappable
  (`source_column` / `source_table` / `row_selection` null as required by UNMAPPABLE FIELDS).
- **validation_notes** must state that **IdentityAgent must resolve** materializing `_edvise_term_*` on
  student or providing a joinable normalized path — **surface to the human reviewer**; this is not fixable
  by SMA alone.
"""


def _step2a_course_academic_term_rules() -> str:
    """Step 2a: mapping rules for course academic_year / academic_term (manifest)."""
    return """
COURSE academic_year AND academic_term — decision hierarchy (apply in order)

**(1) Preferred — IdentityAgent normalized columns on the course base table**
- When **dtypes** for the course entity base table (e.g. `course`) include `_edvise_term_academic_year`
  and `_edvise_term_season`, map `academic_year` and `academic_term` to those columns with
  `source_table` = that base table and `row_selection.strategy`: `"any_row"`.
- Do **not** map from raw `term` code strings on the course row when `_edvise_term_*` are present.

**(2) Missing `_edvise_*` on course — do not join to the `term` table**
- Each course row already carries the term identifier (e.g. `term`). Joining course → `term` only to copy
  `_edvise_term_*` does not fix the real issue: IdentityAgent should materialize those columns **on the
  course frame** for that same key. Prefer treating this as an **upstream IA / enrichment gap**:
  **unmappable** with **validation_notes** that IA must resolve (term order on course), **surface to reviewer**.
- Only if you must ship a non-null mapping and a raw term code column exists **on the same course row**,
  map from that column as a **temporary** fallback with low confidence, **HITL**, and **validation_notes**
  flagging IA — transformation will use extract utilities; this is not equivalent to **(1)**.

**(3) No usable term column on course**
- Leave `academic_year` / `academic_term` unmappable; document in rationale / validation_notes.
"""


def _step2a_rules_after_structure(
    institution_name: str,
    *,
    column_aliases_scope: Literal["single_pass", "entity_pass"],
    term_rules: Literal[
        "cohort_and_course", "cohort_only", "course_only"
    ] = "cohort_and_course",
) -> str:
    """Shared rules (SOURCE COLUMNS → TARGET SCHEMA AUTHORITY) for all Step 2a prompt variants.

    ``term_rules`` controls which term-field hierarchies are injected (cohort entry vs course academic):
    ``cohort_and_course`` for single-pass prompts; ``cohort_only`` / ``course_only`` for entity passes
    so each pass only sees instructions for its entity.
    """
    alias_bullet = _column_aliases_scope_bullet(column_aliases_scope)
    if term_rules == "cohort_and_course":
        term_blocks = (
            _step2a_cohort_entry_term_rules() + _step2a_course_academic_term_rules()
        )
    elif term_rules == "cohort_only":
        term_blocks = _step2a_cohort_entry_term_rules()
    else:
        term_blocks = _step2a_course_academic_term_rules()

    return f"""SOURCE COLUMNS
- Use only source columns and tables present in the {institution_name} schema contract
- Do not use any {institution_name} codebase or prior cleaning scripts as a reference
- Flag unmappable fields with source_column: null, source_table: null, row_selection: null


JOINS AND ALIASES
Step 1 — Decide if a join is needed
- If source_column lives in a different table than the entity base table, you MUST
  declare a join object on that mapping record. A cross-table mapping with no join
  is invalid and will cause a runtime error.
- If source_column is in the entity base table, omit the join object entirely.

Step 2 — Determine join_keys from grain reasoning
- join_keys are not a property of the lookup table alone — they depend on the
  relationship between the base table grain and the lookup table grain.
- Determine the grain of each table from its unique_keys in the schema contract:
    - unique_keys: ["student_id"] → student grain (one row per student)
    - unique_keys: ["student_id", "term"] → student-term grain (one row per student per term)
    - unique_keys: ["term", "course_reference_number"] or ["term", "course_number", "section_number"] → course-section grain
    - unique_keys: ["student_id", "term", "course_number"] → student-term-course grain
- Set join_keys to the minimal shared columns that connect the base table grain to
  the lookup table grain:
    - student grain base → student grain lookup: ["student_id"]
    - student grain base → student-term grain lookup: ["student_id"] only — the term
      key is not shared with a student-grain base table, so it cannot be a join key
    - student-course grain base → student grain lookup: ["student_id"]
    - student-course grain base → student-term grain lookup: ["student_id", "term"]
    - student-course grain base → section grain lookup: ["term", "course_reference_number"]
- CRITICAL: If the join produces multiple rows per base row, that is intentional and
  correct. Do not add columns to join_keys to reduce that multiplicity — that is
  row_selection's job (Step 4).
- CRITICAL: Never add filter columns (e.g. awarded_degree) or ordering columns
  (e.g. term_order) to join_keys. Those belong in row_selection.filter and
  row_selection.order_by respectively.
- Only include a column in join_keys if it exists on both the base table and the
  lookup table with the same canonical name (OR aliased — see Step 3).

Step 3 — Declare a column_alias only for mismatched join key names
- column_aliases exist for exactly one purpose: resolving a join key column that has
  different names in the base table vs the lookup table. Nothing else.
- Before declaring any alias, ask: is this column listed in join_keys?
    - If NO  → do NOT declare an alias. It does not matter that the source column
               you are fetching has a different name, a semantically meaningful label,
               or shares a name with a target field. Source column name differences
               are irrelevant to column_aliases.
    - If YES → check whether it has the same name in both tables.
               Same name → no alias needed.
               Different name → declare one alias entry for the lookup table:
                   table: <lookup_table>
                   source_column: <name of the key in the lookup table>
                   canonical_column: <name of the key in the base table>
               Then use the canonical_column value in join_keys.

Concrete examples:

  CORRECT — alias needed (join key has different names across tables):
    base_table has key column named "term_code"
    lookup_table has the same conceptual key named "term_id"
    join_keys: ["student_id", "term_code"]
    → Declare alias: table: <lookup_table>, source_column: "term_id",
                     canonical_column: "term_code"
    Reason: "term_code" is in join_keys and its name differs in the lookup table.

  CORRECT — alias needed (lookup table uses a different key column name):
    base_table has key column named "course_code"
    lookup_table has the same conceptual key named "cip_code"
    join_keys: ["course_code"]
    → Declare alias: table: <lookup_table>, source_column: "cip_code",
                     canonical_column: "course_code"
    Reason: "course_code" is in join_keys and has a different name in the lookup table.

  WRONG — alias not needed (join key names match in both tables):
    base_table key columns: "student_id", "term"
    lookup_table key columns: "student_id", "term"
    join_keys: ["student_id", "term"]
    → Do NOT declare any alias.
    Reason: all join key columns have identical names in both tables.

  WRONG — alias on a non-key source column:
    join_keys: ["student_id", "term"]
    source_column being fetched from lookup_table: "major"
    target_field this maps to: "term_declared_major"
    → Do NOT alias "major". It is not a join key.
    Reason: "major" is the column being retrieved, not a column used to perform
    the join. The fact that it resembles or maps to a target field name is irrelevant.

- One alias per mismatched key per lookup table. Never more.
- A join with a mismatched key name and no alias is always an error.
- If a section has no cross-table joins, set column_aliases to [].

{alias_bullet}
- In the output JSON, column_aliases is nested inside each entity manifest object
  (alongside entity_type, target_schema, mappings) — not at the top level.
- Key order inside each entity manifest object must be exactly:
  entity_type, then target_schema, then mappings, then column_aliases (always last).
  Always include column_aliases (use [] when no join-key renames are needed).
- The ColumnAlias and FieldMappingManifest definitions in manifest_schema_reference
  are authoritative for field names and types.

Step 4 — row_selection resolves multiplicity after the join
- After joining, row_selection picks which row(s) to use. This is where ordering,
  filtering, and nth selection belong — not in join_keys.
- See the ROW SELECTION section below for allowed strategies and when to use each.
{term_blocks}
ROW SELECTION
- row_selection.strategy: "any_row", "first_by", "where_not_null", "constant", or "nth"
- row_selection.filter.operator (optional): ONLY "contains", "equals", "startswith", or "isin" —
  for pre-filtering rows before selection
- CRITICAL: Do not mix strategies with filter operators — they are separate concepts with different allowed values
- Use "any_row" strategy only when all rows for a student/course are guaranteed to have the same value for this field
  (e.g. a demographic that never changes across terms)
- Use "first_by" strategy only when row ordering has semantic meaning — e.g. earliest term, most recent record.
  The sort column must be present in the schema contract and must have a meaningful ordinal interpretation.
  Do not use first_by with an arbitrary column just to resolve grain ambiguity
- Use "nth" strategy only when you need the Nth matching row after sorting (e.g. 2nd certificate, 3rd award).
  You MUST set row_selection.n to a positive integer (1 = first row after sort, 2 = second, …) and
  row_selection.order_by to the contract column used for sorting — never omit n or use null for nth
- Use "where_not_null" strategy when the field is sparsely populated across rows and you want the most complete record —
  not as a substitute for first_by when ordering matters
- Use "constant" strategy only for institutionally-derived constant values with no source column
- Filters (row_selection.filter) must only be applied when the schema contract provides explicit evidence
  that a subpopulation should be excluded (e.g. a flag column with known values distinguishing populations).
  Do not use filters to resolve ambiguity or simplify a complex mapping — flag for HITL review instead

UNMAPPABLE FIELDS
- Unmappable fields are not automatically confidence: 1.0
- If a related but insufficient column exists (e.g. a flag instead of a count, a proxy instead of a direct match),
  lower confidence to reflect the ambiguity and flag for HITL review
- Document what was found and why it is insufficient in the rationale

MISSING SOURCE VALUES
- The schema contract already reflects missing data as null (null_pct, and no sentinel strings in
  sample_values / unique_values). Do not assume raw CSV tokens like (Blank) appear as distinct values.
- For student group fields (gender, race, ethnicity, first_gen, pell_status_first_year, military_status,
  disability_status, incarcerated_status, employment_status): when the chosen source is missing,
  map to 'Unknown / Not Specified' via map_values where required by the target schema — note this in the rationale
- For other fields: represent missing source as JSON null in the rationale where appropriate;
  the field executor maps JSON null to pd.NA using pandas nullable types

CONFIDENCE SCORING
- 1.0 — direct column match, no transformation ambiguity, single table, no join required
- 0.9 — requires value mapping with complete unique_values coverage, no join required
- 0.7–0.8 — any of: proxy field, incomplete value coverage, multi-column derivation,
  or join required with a clearly identifiable join key
- 0.6–0.7 — join required but join key is uncertain, or first_by used without a clearly meaningful sort column,
  or filter applied to resolve ambiguity
- < 0.6 — ambiguous mapping where the correct source, strategy, or join cannot be determined from the schema contract alone;
  always flag for HITL review
- CRITICAL: Do not inflate confidence because a mapping is structurally plausible.
  Confidence reflects how certain you are that this mapping produces the correct semantic output
  for every student/course record, not just that it produces valid output

RATIONALE
- Focus on mapping evidence: which source column(s) plausibly carry the target semantics, grain and table choice,
  when joins or row_selection are required and why, and what drives confidence or ambiguity
- Do not use rationale to call out unconstrained or free-text typing — avoid phrases whose main point is that the
  target field accepts any string, has no controlled vocabulary, is passthrough, or is "strip/cast only because
  the schema does not enumerate values". The Pandera field descriptor already states dtype; repeating it is noise
- Do not center the rationale on schema typing; center it on why this source column is the right one for this institution's contract
- Unmappable entries: keep rationale aligned with UNMAPPABLE FIELDS (what was considered, why nothing suffices)

VALIDATION NOTES
- Add validation_notes to any field where the mapping or transformation could produce values that fail Pandera validation:
  regex pattern mismatches, out-of-range values, nulls on non-nullable fields, categorical values not in allowed set
- Use the target schema definitions as the authoritative source for allowed values, nullability, and regex patterns
- Leave validation_notes: null if no validation risk identified
- Do not use validation_notes only to remark that a string field is unconstrained or free-text — that is not a validation risk by itself

MANIFEST COMPACTNESS
- Include all required top-level keys and all required mapping records
- For each mapping object, omit optional metadata fields when their value would be null
- You may omit join only when the source column is in the entity base table —
  never omit a join for a cross-table field
- Specifically, you may omit reviewer_notes, corrected_source_column, and validation_notes when they are null
- Do not omit source_column, source_table, or row_selection when the intent is to explicitly mark a field as unmappable;
  in that case set them to null

TARGET SCHEMA AUTHORITY
- Use the Pandera schemas as the authoritative definition of target fields
- Allowed values, nullability, and regex patterns from the schema should directly inform confidence scoring,
  validation_notes, and when lower confidence is warranted — but follow RATIONALE: do not restate unconstrained-string
  or free-text facts about the target field in prose

DATETIME AND DATE TARGET FIELDS
- Pandera datetime targets fall into two groups; apply the correct rule by target_field name.

  (1) STRICT — source column dtype in the schema contract must already be datetime (e.g. datetime64[ns]). Do not map
      term codes, term labels, or non-datetime columns to these targets; leave unmappable (null source_column,
      source_table, row_selection) if no datetime column exists.
      - Cohort (student) entity: matriculation_date
      - Course entity: course_begin_date, course_end_date

  (2) OUTCOME CONFERRAL-STYLE — try to map these from the best available timing signal when it is semantically
      appropriate (including term fields, coded terms, or other non-datetime contract dtypes). The
      transformation step may coerce to datetime64[ns]; use lower confidence and validation_notes when the source
      is a term proxy rather than a true calendar conferral timestamp.
      - Cohort (student) entity: bachelors_degree_conferral_date, associates_degree_conferral_date,
        certificate1_date, certificate2_date, certificate3_date

- For STRICT fields, do not treat numeric encodings (e.g. YYYYMM) as sufficient unless the contract lists that column
  as datetime — unmappable if only integers or strings without a datetime dtype.
- For OUTCOME CONFERRAL-STYLE fields, numeric or fixed-format encodings (e.g. YYYYMM) may be mapped when interpretation
  is reliable; flag lower confidence and validation_notes when parsing or term-to-date conversion is required."""


def _step2a_json_output_rules() -> str:
    return """JSON OUTPUT (strict, machine-parseable)
- The entire response must be one JSON object that passes a strict JSON parser (RFC 8259);
  no markdown fences, preamble, or text after the closing brace
- Strings use double quotes only; apostrophes in prose (e.g. Bachelor's) need no backslash.
  Never write a backslash before a single quote — that is invalid JSON
- Use only standard JSON string escapes (e.g. backslash-doublequote for a literal quote,
  doubled backslash for a backslash, backslash-n for newline, backslash-u plus four hex digits for Unicode)
- No trailing commas, comments, or single-quoted keys"""


def _step2a_prompt_close(*, generate_line: str) -> str:
    return f"""</rules>

{generate_line}
Output only that object — valid JSON throughout."""


def _step2a_reference_blocks(
    reference_manifests: list[dict],
    reference_institution_names: list[str] | None,
) -> tuple[list[str], str]:
    if reference_institution_names is None:
        reference_institution_names = [
            m.get("institution_id", f"reference_{i}")
            for i, m in enumerate(reference_manifests)
        ]
    blocks = "\n\n".join(
        f'<reference_manifest institution="{name}" role="structural_reference_only">\n'
        f"{json.dumps(slim_reference_manifest(manifest), indent=2)}\n"
        f"</reference_manifest>"
        for name, manifest in zip(reference_institution_names, reference_manifests)
    )
    return reference_institution_names, blocks


def merge_step2a_entity_manifests(
    cohort_pass: dict,
    course_pass: dict,
    *,
    institution_id: str | None = None,
    schema_version: str = "0.1.0",
) -> dict:
    """
    Merge cohort-only and course-only Step 2a JSON objects into one full manifest envelope.

    Each pass may return either a fragment with top-level ``manifests`` containing a
    single entity key, or a partial ``FieldMappingManifest``-shaped object (keys
    ``entity_type``, ``target_schema``, ``mappings``, ``column_aliases`` only).

    When passes omit envelope-level fields, supply ``institution_id`` (and optionally
    ``schema_version``); otherwise they are taken from the cohort pass when present.
    """

    def _extract(pass_dict: dict, role: Literal["cohort", "course"]) -> dict[str, Any]:
        manifests = pass_dict.get("manifests")
        if isinstance(manifests, dict) and role in manifests:
            return cast(dict[str, Any], manifests[role])
        if pass_dict.get("entity_type") != role:
            raise ValueError(
                f"expected entity_type {role!r} for this pass, got {pass_dict.get('entity_type')!r}"
            )
        return cast(dict[str, Any], pass_dict)

    cohort_entity = _extract(cohort_pass, "cohort")
    course_entity = _extract(course_pass, "course")

    inst_id = institution_id
    if inst_id is None:
        inst_id = cohort_pass.get("institution_id") or course_pass.get("institution_id")
    if not inst_id:
        raise ValueError(
            "institution_id is required when merging partial entity manifests "
            "(pass institution_id=... to merge_step2a_entity_manifests)"
        )

    sv = schema_version
    if "schema_version" in cohort_pass:
        sv = cohort_pass.get("schema_version", sv)
    elif "schema_version" in course_pass:
        sv = course_pass.get("schema_version", sv)

    return {
        "schema_version": sv,
        "institution_id": inst_id,
        "manifests": {
            "cohort": cohort_entity,
            "course": course_entity,
        },
    }


# ── Prompt assembly ────────────────────────────────────────────────────────────


def collect_step2a_prompt_sections(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_schema_contract: dict,
    reference_manifests: list[dict],
    cohort_schema_class: Any,
    course_schema_class: Any,
    reference_institution_names: list[str] | None = None,
    *,
    compact_manifest_schema: bool = True,
) -> dict[str, str]:
    """
    Named sections of the Step 2a single-pass prompt (reference manifests, schema contract,
    Pandera descriptors, manifest schema, rules).

    ``compact_manifest_schema`` (default True) injects :func:`get_compact_manifest_schema_reference`;
    set False for full Pydantic source via :func:`get_manifest_schema_context`.
    """
    _ = institution_id  # envelope only; same args as builders for call-site symmetry
    contract_summary = summarize_schema_contract(institution_schema_contract)
    cohort_descriptor = extract_schema_descriptor(cohort_schema_class)
    course_descriptor = extract_schema_descriptor(course_schema_class)
    _, reference_blocks = _step2a_reference_blocks(
        reference_manifests, reference_institution_names
    )
    contract_json = json.dumps(contract_summary, indent=2)
    cohort_json = json.dumps(cohort_descriptor, indent=2)
    course_json = json.dumps(course_descriptor, indent=2)
    manifest_schema = _manifest_schema_for_prompt(compact=compact_manifest_schema)
    preamble = f"""Please act as the SchemaMappingAgent and generate a mapping manifest for {institution_name} at:
{output_path}

The mapping manifest is a machine-consumed specification.
A deterministic field executor reads each record and resolves the source Series —
it cannot infer missing join declarations or table relationships.
Every structural decision (join, row_selection, column_aliases) must be fully and explicitly declared.
"""
    rules = f"""<rules>
STRUCTURE
- Match the reference manifest JSON structure exactly (schema_version, institution_id,
  manifests with cohort + course sections; each section: entity_type, target_schema,
  mappings array, then column_aliases last)
- Each mapping entry must include: target_field, source_column, source_table,
  row_selection, confidence, rationale, and a review_status.
- Set review_status: "pending" on all records — "approved" is only set after human review
- Do not copy row_selection strategies, filters, or join configurations from the reference manifests —
  these are institution-specific and must be derived from the {institution_name} schema contract.
  The reference manifests are structural reference only: use them to understand the expected shape
  of the output and concise rationale (structural mapping reasoning per RATIONALE rules), not as a source of mapping decisions

{_step2a_rules_after_structure(institution_name, column_aliases_scope="single_pass")}
{_step2a_json_output_rules()}
{_step2a_prompt_close(generate_line="Generate the complete mapping manifest JSON now.")}"""
    return {
        "preamble": preamble,
        "reference_manifests": reference_blocks,
        "schema_contract": (
            f'<schema_contract institution="{institution_name}">\n'
            f"{contract_json}\n"
            f"</schema_contract>"
        ),
        "target_schema_cohort": (
            '<target_schema name="RawEdviseStudentDataSchema" entity="cohort">\n'
            f"{cohort_json}\n"
            "</target_schema>"
        ),
        "target_schema_course": (
            '<target_schema name="RawEdviseCourseDataSchema" entity="course">\n'
            f"{course_json}\n"
            "</target_schema>"
        ),
        "manifest_schema_reference": (
            f"<manifest_schema_reference>\n{manifest_schema}\n</manifest_schema_reference>"
        ),
        "rules": rules,
    }


def assemble_step2a_prompt_from_sections(sections: dict[str, str]) -> str:
    """Join :func:`collect_step2a_prompt_sections` parts with the same spacing as the legacy f-string."""
    order = (
        "preamble",
        "reference_manifests",
        "schema_contract",
        "target_schema_cohort",
        "target_schema_course",
        "manifest_schema_reference",
        "rules",
    )
    return "\n\n".join(sections[k] for k in order)


def collect_step2a_prompt_batched_sections(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_schema_contract: dict,
    reference_manifests: list[dict],
    cohort_schema_class: Any,
    course_schema_class: Any,
    reference_institution_names: list[str] | None = None,
    *,
    compact_manifest_schema: bool = True,
) -> dict[str, str]:
    """
    Sections for Step 2a **batched** prompt: one LLM call with both cohort and course manifests.

    Shared blocks (schema contract, rules, reference manifests, manifest schema reference) appear
    once — same section layout as :func:`collect_step2a_prompt_sections`, with a batched preamble,
    PART 1 / PART 2 target-schema labels, and a MappingManifestEnvelope-oriented closing line.
    """
    _ = institution_id
    contract_summary = summarize_schema_contract(institution_schema_contract)
    cohort_descriptor = extract_schema_descriptor(cohort_schema_class)
    course_descriptor = extract_schema_descriptor(course_schema_class)
    _, reference_blocks = _step2a_reference_blocks(
        reference_manifests, reference_institution_names
    )
    contract_json = json.dumps(contract_summary, indent=2)
    cohort_json = json.dumps(cohort_descriptor, indent=2)
    course_json = json.dumps(course_descriptor, indent=2)
    manifest_schema = _manifest_schema_for_prompt(compact=compact_manifest_schema)
    preamble = f"""Please act as the SchemaMappingAgent and generate a single batched mapping manifest for {institution_name} at:
{output_path}

This replaces separate cohort-only and course-only Step 2a passes: produce the full envelope in one LLM call.

Generate a MappingManifestEnvelope with manifests containing BOTH "cohort" and "course" keys.

The mapping manifest is a machine-consumed specification.
A deterministic field executor reads each record and resolves the source Series —
it cannot infer missing join declarations or table relationships.
Every structural decision (join, row_selection, column_aliases) must be fully and explicitly declared.
"""
    rules = f"""<rules>
STRUCTURE
- Match the reference manifest JSON structure exactly (schema_version, institution_id,
  manifests with cohort + course sections; each section: entity_type, target_schema,
  mappings array, then column_aliases last)
- Each mapping entry must include: target_field, source_column, source_table,
  row_selection, confidence, rationale, and a review_status.
- Set review_status: "pending" on all records — "approved" is only set after human review
- Do not copy row_selection strategies, filters, or join configurations from the reference manifests —
  these are institution-specific and must be derived from the {institution_name} schema contract.
  The reference manifests are structural reference only: use them to understand the expected shape
  of the output and concise rationale (structural mapping reasoning per RATIONALE rules), not as a source of mapping decisions

{_step2a_rules_after_structure(institution_name, column_aliases_scope="single_pass")}
{_step2a_json_output_rules()}
{_step2a_prompt_close(generate_line="Respond with valid JSON matching MappingManifestEnvelope.")}"""
    return {
        "preamble": preamble,
        "reference_manifests": reference_blocks,
        "schema_contract": (
            f'<schema_contract institution="{institution_name}">\n'
            f"{contract_json}\n"
            f"</schema_contract>"
        ),
        "target_schema_cohort": (
            "PART 1: Cohort (RawEdviseStudentDataSchema)\n\n"
            '<target_schema name="RawEdviseStudentDataSchema" entity="cohort">\n'
            f"{cohort_json}\n"
            "</target_schema>"
        ),
        "target_schema_course": (
            "PART 2: Course (RawEdviseCourseDataSchema)\n\n"
            '<target_schema name="RawEdviseCourseDataSchema" entity="course">\n'
            f"{course_json}\n"
            "</target_schema>"
        ),
        "manifest_schema_reference": (
            f"<manifest_schema_reference>\n{manifest_schema}\n</manifest_schema_reference>"
        ),
        "rules": rules,
    }


def assemble_step2a_batched_prompt_from_sections(sections: dict[str, str]) -> str:
    """Join :func:`collect_step2a_prompt_batched_sections` in the same order as the single-pass prompt."""
    return assemble_step2a_prompt_from_sections(sections)


def collect_step2a_prompt_cohort_pass_sections(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_schema_contract: dict,
    reference_manifests: list[dict],
    cohort_schema_class: Any,
    reference_institution_names: list[str] | None = None,
    *,
    compact_manifest_schema: bool = True,
) -> dict[str, str]:
    """Sections for Step 2a cohort-only pass."""
    contract_summary = summarize_schema_contract(institution_schema_contract)
    cohort_descriptor = extract_schema_descriptor(cohort_schema_class)
    _, reference_blocks = _step2a_reference_blocks(
        reference_manifests, reference_institution_names
    )
    contract_json = json.dumps(contract_summary, indent=2)
    cohort_json = json.dumps(cohort_descriptor, indent=2)
    manifest_schema = _manifest_schema_for_prompt(compact=compact_manifest_schema)
    preamble = f"""Please act as the SchemaMappingAgent. This is PASS 1 of 2 for {institution_name}.
Generate only the **cohort** (student) entity mapping manifest. A second pass will produce course.
Destination path for the combined manifest (for context):
{output_path}

The mapping manifest is a machine-consumed specification.
A deterministic field executor reads each record and resolves the source Series —
it cannot infer missing join declarations or table relationships.
Every structural decision (join, row_selection, column_aliases) must be fully and explicitly declared.
"""
    rules = f"""<rules>
STRUCTURE (cohort pass only)
- Output one JSON object with: schema_version, institution_id, and manifests
- manifests MUST contain exactly one key: \"cohort\" — do not include \"course\" or any course mappings
- manifests.cohort must include entity_type, target_schema, mappings for every
  target field in the cohort Pandera schema above, and column_aliases (last; [] if none)
- Set institution_id to \"{institution_id}\" (use this exact value)
- Each mapping entry must include: target_field, source_column, source_table,
  row_selection, confidence, rationale, and a review_status.
- Set review_status: \"pending\" on all records — \"approved\" is only set after human review
- Do not copy row_selection strategies, filters, or join configurations from the reference manifests —
  these are institution-specific and must be derived from the {institution_name} schema contract.
  The reference manifests are structural reference only

{_step2a_rules_after_structure(institution_name, column_aliases_scope="entity_pass", term_rules="cohort_only")}
{_step2a_json_output_rules()}
- Output only the entity manifest object — a JSON object with keys entity_type, target_schema, mappings, and column_aliases. Do not include institution_id, schema_version, or any envelope-level fields. These are added by the calling code.
- The manifests object MUST contain only the \"cohort\" key; omit \"course\" entirely
{_step2a_prompt_close(generate_line="Generate the cohort-only mapping manifest JSON now.")}"""
    return {
        "preamble": preamble,
        "reference_manifests": reference_blocks,
        "schema_contract": (
            f'<schema_contract institution="{institution_name}">\n'
            f"{contract_json}\n"
            "</schema_contract>"
        ),
        "target_schema_cohort": (
            '<target_schema name="RawEdviseStudentDataSchema" entity="cohort">\n'
            f"{cohort_json}\n"
            "</target_schema>"
        ),
        "manifest_schema_reference": (
            f"<manifest_schema_reference>\n{manifest_schema}\n</manifest_schema_reference>"
        ),
        "rules": rules,
    }


def assemble_step2a_cohort_pass_prompt_from_sections(sections: dict[str, str]) -> str:
    order = (
        "preamble",
        "reference_manifests",
        "schema_contract",
        "target_schema_cohort",
        "manifest_schema_reference",
        "rules",
    )
    return "\n\n".join(sections[k] for k in order)


def collect_step2a_prompt_course_pass_sections(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_schema_contract: dict,
    reference_manifests: list[dict],
    course_schema_class: Any,
    reference_institution_names: list[str] | None = None,
    *,
    compact_manifest_schema: bool = True,
) -> dict[str, str]:
    """Sections for Step 2a course-only pass."""
    contract_summary = summarize_schema_contract(institution_schema_contract)
    course_descriptor = extract_schema_descriptor(course_schema_class)
    _, reference_blocks = _step2a_reference_blocks(
        reference_manifests, reference_institution_names
    )
    contract_json = json.dumps(contract_summary, indent=2)
    course_json = json.dumps(course_descriptor, indent=2)
    manifest_schema = _manifest_schema_for_prompt(compact=compact_manifest_schema)
    preamble = f"""Please act as the SchemaMappingAgent. This is PASS 2 of 2 for {institution_name}.
Generate only the **course** entity mapping manifest (pass 1, run separately, covers cohort only).
Destination path for the combined manifest (for context):
{output_path}

The mapping manifest is a machine-consumed specification.
A deterministic field executor reads each record and resolves the source Series —
it cannot infer missing join declarations or table relationships.
Every structural decision (join, row_selection, column_aliases) must be fully and explicitly declared.
"""
    rules = f"""<rules>
STRUCTURE (course pass only)
- Output one JSON object with: schema_version, institution_id, and manifests
- manifests MUST contain exactly one key: \"course\" — do not include \"cohort\" or duplicate cohort mappings
- manifests.course must include entity_type, target_schema, mappings for every
  target field in the course Pandera schema above, and column_aliases (last; [] if none)
- Set institution_id to \"{institution_id}\" (use this exact value)
- Set schema_version to match the reference mapping manifests (typically \"0.1.0\")
- Each mapping entry must include: target_field, source_column, source_table,
  row_selection, confidence, rationale, and a review_status.
- Set review_status: \"pending\" on all records — \"approved\" is only set after human review
- Do not copy row_selection strategies, filters, or join configurations from the reference manifests —
  these are institution-specific and must be derived from the {institution_name} schema contract.
  The reference manifests are structural reference only

{_step2a_rules_after_structure(institution_name, column_aliases_scope="entity_pass", term_rules="course_only")}
{_step2a_json_output_rules()}
- Output only the entity manifest object — a JSON object with keys entity_type, target_schema, mappings, and column_aliases. Do not include institution_id, schema_version, or any envelope-level fields. These are added by the calling code.
- The manifests object MUST contain only the \"course\" key; omit \"cohort\" entirely
{_step2a_prompt_close(generate_line="Generate the course-only mapping manifest JSON now.")}"""
    return {
        "preamble": preamble,
        "reference_manifests": reference_blocks,
        "schema_contract": (
            f'<schema_contract institution="{institution_name}">\n'
            f"{contract_json}\n"
            "</schema_contract>"
        ),
        "target_schema_course": (
            '<target_schema name="RawEdviseCourseDataSchema" entity="course">\n'
            f"{course_json}\n"
            "</target_schema>"
        ),
        "manifest_schema_reference": (
            f"<manifest_schema_reference>\n{manifest_schema}\n</manifest_schema_reference>"
        ),
        "rules": rules,
    }


def assemble_step2a_course_pass_prompt_from_sections(sections: dict[str, str]) -> str:
    order = (
        "preamble",
        "reference_manifests",
        "schema_contract",
        "target_schema_course",
        "manifest_schema_reference",
        "rules",
    )
    return "\n\n".join(sections[k] for k in order)


def audit_step2a_prompt(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_schema_contract: dict,
    reference_manifests: list[dict],
    cohort_schema_class: Any,
    course_schema_class: Any,
    reference_institution_names: list[str] | None = None,
    *,
    variant: Literal["single", "cohort_pass", "course_pass"] = "single",
    log: bool = True,
    compact_manifest_schema: bool = True,
) -> dict[str, Any]:
    """
    Local estimated token counts for Step 2a prompts (single user blob; no system message).

    ``variant`` selects which prompt shape to measure (combined vs entity passes).
    """
    if variant == "single":
        sections = collect_step2a_prompt_sections(
            institution_id,
            institution_name,
            output_path,
            institution_schema_contract,
            reference_manifests,
            cohort_schema_class,
            course_schema_class,
            reference_institution_names,
            compact_manifest_schema=compact_manifest_schema,
        )
        builder = "schema_mapping_agent.step2a.single"
    elif variant == "cohort_pass":
        sections = collect_step2a_prompt_cohort_pass_sections(
            institution_id,
            institution_name,
            output_path,
            institution_schema_contract,
            reference_manifests,
            cohort_schema_class,
            reference_institution_names,
            compact_manifest_schema=compact_manifest_schema,
        )
        builder = "schema_mapping_agent.step2a.cohort_pass"
    else:
        sections = collect_step2a_prompt_course_pass_sections(
            institution_id,
            institution_name,
            output_path,
            institution_schema_contract,
            reference_manifests,
            course_schema_class,
            reference_institution_names,
            compact_manifest_schema=compact_manifest_schema,
        )
        builder = "schema_mapping_agent.step2a.course_pass"
    return audit_prompt_sections(
        sections,
        builder=builder,
        institution_id=institution_id,
        institution_name=institution_name,
        log=log,
    )


def audit_step2a_batched_prompt(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_schema_contract: dict,
    reference_manifests: list[dict],
    cohort_schema_class: Any,
    course_schema_class: Any,
    reference_institution_names: list[str] | None = None,
    *,
    log: bool = True,
    compact_manifest_schema: bool = True,
) -> dict[str, Any]:
    """
    Estimated token counts for the batched Step 2a prompt (one user blob; no system message).

    Section keys match :func:`collect_step2a_prompt_batched_sections`; totals should be far below
    ``audit_step2a_prompt(..., variant=\"cohort_pass\")`` + ``variant=\"course_pass\"`` because
    schema contract, rules, references, and manifest schema are not duplicated.
    """
    sections = collect_step2a_prompt_batched_sections(
        institution_id,
        institution_name,
        output_path,
        institution_schema_contract,
        reference_manifests,
        cohort_schema_class,
        course_schema_class,
        reference_institution_names,
        compact_manifest_schema=compact_manifest_schema,
    )
    return audit_prompt_sections(
        sections,
        builder="schema_mapping_agent.step2a.batched",
        institution_id=institution_id,
        institution_name=institution_name,
        log=log,
    )


def build_step2a_prompt(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_schema_contract: dict,
    reference_manifests: list[dict],
    cohort_schema_class: Any,
    course_schema_class: Any,
    reference_institution_names: list[str] | None = None,
    *,
    compact_manifest_schema: bool = True,
) -> str:
    """
    Build the Step 2a field mapping prompt.

    Args:
        institution_id:               e.g. "lee_col"
        institution_name:             e.g. "Lee College"
        output_path:                  Destination path for the manifest (for prompt header)
        institution_schema_contract:  Parsed schema contract JSON for the target institution
        reference_manifests:          List of parsed mapping manifest JSONs for reference
                                      institutions (e.g. [ucf_manifest, lc_manifest])
        cohort_schema_class:          RawEdviseStudentDataSchema (Pandera class)
        course_schema_class:          RawEdviseCourseDataSchema (Pandera class)
        reference_institution_names:  Display names for reference institutions, parallel
                                      to reference_manifests. Defaults to manifest
                                      institution_id values if not provided.
    """
    sections = collect_step2a_prompt_sections(
        institution_id,
        institution_name,
        output_path,
        institution_schema_contract,
        reference_manifests,
        cohort_schema_class,
        course_schema_class,
        reference_institution_names,
        compact_manifest_schema=compact_manifest_schema,
    )
    return assemble_step2a_prompt_from_sections(sections)


def build_step2a_prompt_cohort_pass(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_schema_contract: dict,
    reference_manifests: list[dict],
    cohort_schema_class: Any,
    reference_institution_names: list[str] | None = None,
    *,
    compact_manifest_schema: bool = True,
) -> str:
    """
    Step 2a pass 1: cohort (student) entity only.

    The model must return JSON with top-level schema_version, institution_id, and
    manifests containing only the \"cohort\" key (no \"course\" section).
    """
    sections = collect_step2a_prompt_cohort_pass_sections(
        institution_id,
        institution_name,
        output_path,
        institution_schema_contract,
        reference_manifests,
        cohort_schema_class,
        reference_institution_names,
        compact_manifest_schema=compact_manifest_schema,
    )
    return assemble_step2a_cohort_pass_prompt_from_sections(sections)


def build_step2a_prompt_course_pass(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_schema_contract: dict,
    reference_manifests: list[dict],
    course_schema_class: Any,
    reference_institution_names: list[str] | None = None,
    *,
    compact_manifest_schema: bool = True,
) -> str:
    """
    Step 2a pass 2: course entity only.

    The model must return JSON with manifests containing only the \"course\" key.
    Orchestration merges this JSON with the cohort pass offline; the course prompt does
    not receive cohort output.
    """
    sections = collect_step2a_prompt_course_pass_sections(
        institution_id,
        institution_name,
        output_path,
        institution_schema_contract,
        reference_manifests,
        course_schema_class,
        reference_institution_names,
        compact_manifest_schema=compact_manifest_schema,
    )
    return assemble_step2a_course_pass_prompt_from_sections(sections)


def build_step2a_batched_prompt(
    institution_id: str,
    institution_name: str,
    output_path: str,
    institution_schema_contract: dict,
    reference_manifests: list[dict],
    cohort_schema_class: Any,
    course_schema_class: Any,
    reference_institution_names: list[str] | None = None,
    *,
    compact_manifest_schema: bool = True,
) -> str:
    """
    Step 2a batched: one LLM call that returns a full :class:`~.schemas.MappingManifestEnvelope`
    with both ``cohort`` and ``course`` keys (replaces separate ``cohort_pass`` + ``course_pass`` calls).

    Shared prompt blocks are included once; see :func:`collect_step2a_prompt_batched_sections`.
    """
    sections = collect_step2a_prompt_batched_sections(
        institution_id,
        institution_name,
        output_path,
        institution_schema_contract,
        reference_manifests,
        cohort_schema_class,
        course_schema_class,
        reference_institution_names,
        compact_manifest_schema=compact_manifest_schema,
    )
    return assemble_step2a_batched_prompt_from_sections(sections)


# ── Convenience helpers ────────────────────────────────────────────────────────


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
