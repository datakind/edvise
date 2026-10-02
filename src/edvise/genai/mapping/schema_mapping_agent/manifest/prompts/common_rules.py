"""Prompt fragments shared by Step 2a generate and refinement (HITL) prompts.

Keep these rules in one place so generate, Pass 1, and Pass 2 stay aligned.
"""

# Injected once into generate rules and once into each refinement system prompt.
# Do not duplicate the body in generate.py / refine.py — point at this heading instead.
STEP2A_SHARED_POLICY_RULES = """
SHARED POLICY — datetime parsing and all-null sources
(applies to original Step 2a generate and to refinement Pass 1 / Pass 2)

DATETIME PARSING IS NOT A HITL REASON
- Needing Step 2b to parse or coerce a mapped source into datetime (ISO strings, YYYYMM,
  raw term codes, `coerce_datetime`, conferral term-code utilities, etc.) is **not** by itself
  a reason to lower confidence, flag HITL, or emit a HITL item.
- HITL is for source **identity** and **semantics**: wrong column, grain / join / row_selection
  ambiguity, credential discriminator, proxy vs true entry or outcome. If the source column is
  the right one, map it at the confidence that semantic match deserves; put parse details in
  **validation_notes** only.
- Do **not** create a HITL item whose only question is confirming datetime parsing (e.g.
  "is this term code parseable as a date?").
- Refinement exception to the usual ≤-threshold ⇒ HITL rule: if the mapping is otherwise sound
  and the only uncertainty in rationale / validation_notes is datetime parse/coerce, do **not**
  emit hitl_flags. Use auto_approved or refined_by_llm. Leave confidence unchanged if you
  cannot honestly raise it; status still must not route that field to HITL.

ALL-NULL SOURCE COLUMNS ARE AUTOMATICALLY UNMAPPED
- Schema contract `null_pct` is a 0–100 percentage. If a candidate source column is **100**
  (every value missing), treat it as absent — do **not** map it.
- If every otherwise-plausible source for a target is 100% null, leave the target unmappable
  (`source_column` / `source_table` / `row_selection` null) with confidence **1.0**. Do **not**
  flag HITL solely to confirm skipping an all-null column.
- Prefer a usable non-null column when one exists; never keep an all-null mapping "in case"
  values appear later.
- Generate (2a): never emit a mapping whose `source_column` has `null_pct` 100.
- Refinement: if the current mapping's `source_column` is 100% null in the contract, correct
  to unmappable (confidence 1.0) as refined_by_llm / auto_approved — do **not** emit hitl_flags
  solely for that correction. This is an allowed exception to "do not change confidence."
"""
