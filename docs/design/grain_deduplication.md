# Deduplication semantics: “drop distinctions” (Option B)

Grain-stage deduplication (`true_duplicate`, `temporal_collapse`, and parameterized sort + `drop_duplicates` in `drop_duplicate_keys`) **collapses rows** to one per `post_clean_primary_key`. It does **not** remove columns from the dataframe.

When documentation or HITL options say **“drop [column] distinctions”**, it means:

- **Rows** are collapsed so there is one row per grain key.
- The **[column] remains** in the cleaned dataset (it is not deleted from the schema).
- Only **one value** of that column appears per grain key (row-level diversity on that column may be lost).
- **SchemaMappingAgent 2a** still sees the full column set and applies `row_selection` when the table remains multi-row at a coarser join key.

This is **not** the same as **removing** the column entirely. Column removal is a separate schema-narrowing concern, not grain dedup.

## Example: degrees-style table

**Raw** (three rows sharing the same narrower grain key, differing on honors and sub_plan):

| sid | plan | term    | honors    | sub_plan  |
|-----|------|---------|-----------|-----------|
| 1   | MBA  | Fall2023| Summa     | PADHRMGT  |
| 1   | MBA  | Fall2023| Cum Laude | PADMGTOP  |
| 1   | MBA  | Fall2023| none      | PADISGORG |

**After** “collapse on honors (keep highest), drop sub_plan distinctions” at grain `(sid, plan, term, degree)`:

- One row per grain; **sub_plan** is still a column, with the value from the kept row (e.g. PADHRMGT from the Summa row, depending on sort rules).

## Alternative: widen grain

To **keep** multiple sub_plan values per `(sid, plan, term, degree)`, widen the grain to include `sub_plan`. The table may then stay multi-row at the narrower join key; 2a may set `row_selection_required` accordingly.

## Implementation reference

- `edvise.genai.mapping.identity_agent.grain_inference.deduplication.drop_duplicate_keys`
- `edvise.genai.mapping.identity_agent.execution.contract_utilities.apply_grain_dedup`
