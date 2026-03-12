"""
Deterministic join resolver for Agent 2 pipeline.

Runs after 2a (FieldMappingManifest approval) and before 2c (TransformationMap).
Infers join graph from:
  1. Approved manifest — which source_tables are referenced per entity type
  2. Schema contract   — unique_keys + column inventory per dataset

Produces a JoinGraph per entity type (cohort / course) which is:
  - Persisted to schema_contract for human review alongside 2c
  - Executed deterministically by execute_join_graph() to produce flat input DataFrames

Design:
  - Base table = most granular table (unique keys are superset of other tables' unique keys)
  - Join candidates = all other referenced tables
  - Join keys = intersection of base table canonical columns and candidate canonical unique keys
  - Foreign key fallback = scan all available canonical columns for match to candidate unique keys
  - Fan-out detection = flag when candidate grain is finer than base table grain
  - Join order = topological sort by key dependencies (base table first, then dependents)
  - Alias handling = manifest column_aliases rename source columns to canonical names for
    key matching; original names are preserved for actual DataFrame merge execution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from edvise.data_audit.genai.agent2.mapping_schemas import EntityType, FieldMappingManifest

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class JoinStep:
    """A single join operation in the join graph."""
    left: str                           # Left table name (running DataFrame after prior steps)
    right: str                          # Right table name
    left_on: list[str]                  # Join keys on left (original column names)
    right_on: list[str]                 # Join keys on right (original column names, may differ)
    how: str = "left"                   # Join type — always left for safety
    fan_out_risk: bool = False          # True if right table grain is finer than left
    fan_out_note: Optional[str] = None  # Human-readable explanation of fan-out risk


@dataclass
class JoinGraph:
    """
    Complete join graph for one entity type.
    Produced by JoinResolver.resolve(), reviewed by human, executed by execute_join_graph().
    """
    entity_type: EntityType
    base_table: str
    steps: list[JoinStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # Ambiguities needing human review

    @property
    def referenced_tables(self) -> list[str]:
        return [self.base_table] + [s.right for s in self.steps]

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "base_table": self.base_table,
            "steps": [
                {
                    "left": s.left,
                    "right": s.right,
                    "left_on": s.left_on,
                    "right_on": s.right_on,
                    "how": s.how,
                    "fan_out_risk": s.fan_out_risk,
                    "fan_out_note": s.fan_out_note,
                }
                for s in self.steps
            ],
            "warnings": self.warnings,
        }


# -----------------------------------------------------------------------------
# Join resolver
# -----------------------------------------------------------------------------

class JoinResolver:
    """
    Infers join graph from approved manifest + schema contract.

    Args:
        schema_contract: Parsed schema contract dict. Expected structure:
            {
              "datasets": {
                "course_df": {
                  "unique_keys": ["student_id", "term_descr", ...],
                  "normalized_columns": {"ORIG_COL": "normalized_col", ...},
                  "dtypes": {"normalized_col": "dtype", ...}
                },
                ...
              }
            }
        target_unique_keys: Per entity type, the target schema unique constraint.
            Used to identify the base table.
            e.g. {
              "cohort": ["student_id"],
              "course": ["student_id", "academic_year", "academic_term",
                         "course_prefix", "course_number"]
            }
    """

    def __init__(
        self,
        schema_contract: dict,
        target_unique_keys: dict[str, list[str]],
    ):
        self.datasets = schema_contract["datasets"]
        self.target_unique_keys = target_unique_keys
        self._alias_map: dict[str, dict[str, str]] = {}  # populated per resolve() call

    # -------------------------------------------------------------------------
    # Public
    # -------------------------------------------------------------------------

    def resolve(
        self,
        manifest: FieldMappingManifest,
    ) -> JoinGraph:
        """
        Infer join graph from approved manifest.

        Args:
            manifest: Approved FieldMappingManifest for one entity type.

        Returns:
            JoinGraph with ordered join steps and fan-out warnings.
        """
        entity_type = manifest.entity_type

        # --- Build alias map FIRST — must happen before any _get_columns /
        #     _get_unique_keys call since both methods apply aliases ---
        self._alias_map = self._build_alias_map(manifest)
        if self._alias_map:
            logger.info(
                f"[{entity_type}] Column aliases loaded: "
                + ", ".join(
                    f"{t}.{s} → {c}"
                    for t, aliases in self._alias_map.items()
                    for s, c in aliases.items()
                )
            )

        # --- 1. Collect referenced tables from manifest ---
        referenced = {
            m.source_table
            for m in manifest.mappings
            if m.source_table is not None and m.source_columns
        }
        logger.info(f"[{entity_type}] Referenced tables from manifest: {referenced}")

        if not referenced:
            raise ValueError(
                f"No source tables found in approved manifest for {entity_type}"
            )

        # Validate all referenced tables exist in schema contract
        missing = referenced - set(self.datasets.keys())
        if missing:
            available = list(self.datasets.keys())
            error_parts = [
                f"Dataset '{t}' not found in schema contract" for t in missing
            ]
            for t in missing:
                suggestions = [
                    d for d in available
                    if t.lower() in d.lower() or d.lower() in t.lower()
                ]
                if suggestions:
                    error_parts.append(f"  '{t}' — did you mean: {suggestions}?")
            error_parts.append(f"Available datasets: {available}")
            raise ValueError(
                "Manifest references tables not in schema contract:\n"
                + "\n".join(error_parts)
            )

        # --- 2. Identify base table ---
        base_table = self._identify_base_table(entity_type, referenced)
        logger.info(f"[{entity_type}] Base table: {base_table}")

        join_candidates = referenced - {base_table}

        # --- 3. Build join steps ---
        graph = JoinGraph(entity_type=entity_type, base_table=base_table)
        resolved_tables = {base_table}

        # available_columns tracks CANONICAL names for key matching
        available_columns = set(self._get_columns(base_table))
        logger.debug(
            f"[{entity_type}] Base table canonical columns: {sorted(available_columns)}"
        )

        # Order candidates — tables whose keys are already available come first
        ordered_candidates = self._order_candidates(join_candidates, available_columns)

        for candidate in ordered_candidates:
            step, warning = self._resolve_join(
                left_table=base_table,
                right_table=candidate,
                available_columns=available_columns,
                entity_type=entity_type,
            )
            if step:
                graph.steps.append(step)
                # Add right table canonical columns to available pool
                available_columns.update(self._get_columns(candidate))
                resolved_tables.add(candidate)
                logger.info(
                    f"[{entity_type}] Join resolved: {base_table} ← {candidate} "
                    f"on {step.left_on} = {step.right_on}"
                    + (" [FAN-OUT RISK]" if step.fan_out_risk else "")
                )
            if warning:
                graph.warnings.append(warning)
                logger.warning(f"[{entity_type}] {warning}")

        # --- 4. Flag unresolved tables ---
        unresolved = referenced - resolved_tables
        for t in unresolved:
            msg = (
                f"Could not resolve join for '{t}' — no shared keys found with "
                f"available columns. Manual join declaration required."
            )
            graph.warnings.append(msg)
            logger.warning(f"[{entity_type}] {msg}")

        return graph

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _build_alias_map(
        self,
        manifest: FieldMappingManifest,
    ) -> dict[str, dict[str, str]]:
        """
        Build a {table: {source_column: canonical_column}} lookup from manifest aliases.
        Called at the start of resolve() before any column/key lookups.
        """
        alias_map: dict[str, dict[str, str]] = {}
        for alias in manifest.column_aliases:
            alias_map.setdefault(alias.table, {})[alias.source_column] = alias.canonical_column
        return alias_map

    def _get_columns(self, table: str) -> list[str]:
        """
        Return normalized column names for a dataset, with aliases applied.
        Returns CANONICAL names — used for key matching, not DataFrame merge.
        """
        if table not in self.datasets:
            available = list(self.datasets.keys())
            suggestions = [
                d for d in available
                if table.lower() in d.lower() or d.lower() in table.lower()
            ]
            msg = f"Dataset '{table}' not found in schema contract. Available: {available}"
            if suggestions:
                msg += f" Did you mean: {suggestions}?"
            raise ValueError(msg)
        cols = list(self.datasets[table].get("normalized_columns", {}).values())
        aliases = self._alias_map.get(table, {})
        return [aliases.get(col, col) for col in cols]

    def _get_unique_keys(self, table: str) -> list[str]:
        """
        Return unique keys for a dataset, with aliases applied.
        Returns CANONICAL names — used for key matching, not DataFrame merge.
        """
        if table not in self.datasets:
            available = list(self.datasets.keys())
            suggestions = [
                d for d in available
                if table.lower() in d.lower() or d.lower() in table.lower()
            ]
            msg = f"Dataset '{table}' not found in schema contract. Available: {available}"
            if suggestions:
                msg += f" Did you mean: {suggestions}?"
            raise ValueError(msg)
        uks = self.datasets[table].get("unique_keys", [])
        aliases = self._alias_map.get(table, {})
        return [aliases.get(uk, uk) for uk in uks]

    def _identify_base_table(
        self,
        entity_type: str,
        referenced: set[str],
    ) -> str:
        """
        Identify the base table for the join graph.

        The base table is the most granular — its canonical unique keys are a
        superset of at least one other referenced table's canonical unique keys.

        Strategy:
        1. Score by how many other tables have unique keys that are subsets of
           this table's unique keys. Highest score = finest grain = base.
        2. Fallback: largest unique key set if no subset relationships found
           (handles column name mismatches that prevent subset detection).

        UCF course example (after aliases applied):
            course_df canonical UKs: ["student_id", "term_desc", "crse_prefix",
                                       "crse_number", "course_section_number",
                                       "course_section_type", "cf_boe_term_id"]
            student_df canonical UKs: ["student_id", "term_desc"]
            stems_def_df canonical UKs: ["cip"]

            student_df UKs ⊆ course_df UKs → course_df scores 1
            stems_def_df UKs ⊄ course_df UKs → course_df still scores highest
            → base = course_df ✓
        """
        if len(referenced) == 1:
            return next(iter(referenced))

        best_table = None
        best_score = -1

        for table in referenced:
            uks = set(self._get_unique_keys(table))
            subset_count = sum(
                1 for other in referenced - {table}
                if set(self._get_unique_keys(other)).issubset(uks)
            )
            if subset_count > best_score:
                best_score = subset_count
                best_table = table

        if best_table and best_score > 0:
            return best_table

        # Fallback — largest unique key set
        largest = max(referenced, key=lambda t: len(self._get_unique_keys(t)))
        logger.warning(
            f"[{entity_type}] No unique key subset relationships found among "
            f"{referenced} — falling back to largest unique key table '{largest}'. "
            f"If this is wrong, check column_aliases in the manifest."
        )
        return largest

    def _order_candidates(
        self,
        candidates: set[str],
        available_columns: set[str],
    ) -> list[str]:
        """
        Order candidates so tables whose canonical keys are already in
        available_columns come first. Handles transitive join dependencies.
        """
        def priority(table: str) -> int:
            uks = set(self._get_unique_keys(table))
            return -len(uks & available_columns)

        return sorted(candidates, key=priority)

    def _resolve_join(
        self,
        left_table: str,
        right_table: str,
        available_columns: set[str],
        entity_type: str,
    ) -> tuple[Optional[JoinStep], Optional[str]]:
        """
        Resolve join keys between left (running DataFrame) and right table.

        Alias handling:
          - available_columns and unique key matching use CANONICAL names
          - left_on / right_on in the returned JoinStep use ORIGINAL names
            so the actual pandas merge operates on real DataFrame columns

        Strategy:
        1. Direct match — all right canonical UKs found in available_columns
        2. Partial match — some right canonical UKs found → fan-out risk
        3. Foreign key scan — scan available_columns for any right canonical UK
        4. Fail — return warning, no step
        """
        # Canonical UKs for matching
        right_uks_canonical = self._get_unique_keys(right_table)
        # Original UKs for execution (no alias applied)
        right_uks_original = self.datasets[right_table].get("unique_keys", [])

        # Reverse alias maps: canonical → original
        left_aliases = self._alias_map.get(left_table, {})
        right_aliases = self._alias_map.get(right_table, {})
        left_reverse = {v: k for k, v in left_aliases.items()}
        right_reverse = {v: k for k, v in right_aliases.items()}

        def to_left_original(canonical: str) -> str:
            return left_reverse.get(canonical, canonical)

        def to_right_original(canonical: str) -> str:
            return right_reverse.get(canonical, canonical)

        # --- Strategy 1 & 2: Direct / partial unique key match ---
        matched_canonical = [k for k in right_uks_canonical if k in available_columns]

        if matched_canonical:
            left_on = [to_left_original(k) for k in matched_canonical]
            right_on = [to_right_original(k) for k in matched_canonical]
            fan_out = len(matched_canonical) < len(right_uks_canonical)
            fan_out_note = (
                f"'{right_table}' unique keys are {right_uks_original} but join keys are "
                f"{right_on} — multiple rows per join key possible. "
                f"Review collapse strategy or pre-join filter."
            ) if fan_out else None

        else:
            # --- Strategy 3: Foreign key scan ---
            fk_canonical = [k for k in right_uks_canonical if k in available_columns]

            if fk_canonical:
                left_on = [to_left_original(k) for k in fk_canonical]
                right_on = [to_right_original(k) for k in fk_canonical]
                fan_out = len(fk_canonical) < len(right_uks_canonical)
                fan_out_note = (
                    f"Join via foreign key scan on {right_on}. "
                    + (
                        f"Fan-out risk: right unique keys are {right_uks_original}."
                        if fan_out else ""
                    )
                ) if fan_out else None
            else:
                # --- Strategy 4: Fail ---
                return None, (
                    f"No shared keys found between running DataFrame and '{right_table}' "
                    f"(right unique keys: {right_uks_original}, "
                    f"sample available canonical cols: {sorted(available_columns)[:20]}). "
                    f"Manual join declaration required."
                )

        if not left_on or not right_on:
            return None, (
                f"Join key resolution produced empty keys for '{right_table}'. "
                f"Manual declaration required."
            )

        return JoinStep(
            left=left_table,
            right=right_table,
            left_on=left_on,
            right_on=right_on,
            how="left",
            fan_out_risk=fan_out,
            fan_out_note=fan_out_note,
        ), None


# -----------------------------------------------------------------------------
# Join graph executor
# -----------------------------------------------------------------------------

def execute_join_graph(
    graph: JoinGraph,
    dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Execute an approved JoinGraph against loaded DataFrames.

    Args:
        graph: Approved JoinGraph from JoinResolver.resolve()
        dataframes: Dict of dataset_name -> DataFrame.
                    Keys must match graph.base_table and step.right values.

    Returns:
        Flat input DataFrame ready for TransformationMap executor.
    """
    if graph.base_table not in dataframes:
        raise ValueError(
            f"Base table '{graph.base_table}' not found in dataframes. "
            f"Available: {list(dataframes.keys())}"
        )

    result = dataframes[graph.base_table].copy()
    logger.info(
        f"[{graph.entity_type}] Starting join from base '{graph.base_table}': "
        f"{result.shape}"
    )

    for step in graph.steps:
        if step.right not in dataframes:
            raise ValueError(
                f"Table '{step.right}' not found in dataframes. "
                f"Available: {list(dataframes.keys())}"
            )

        right_df = dataframes[step.right]

        if step.fan_out_risk:
            logger.warning(
                f"Executing join with fan-out risk: {step.left} ← {step.right} "
                f"on {step.left_on} = {step.right_on}. {step.fan_out_note}"
            )

        pre_shape = result.shape
        result = result.merge(
            right_df,
            left_on=step.left_on,
            right_on=step.right_on,
            how=step.how,
            suffixes=("", f"_{step.right}"),
        )
        logger.info(
            f"Joined '{step.right}' on {step.left_on} = {step.right_on}: "
            f"{pre_shape} → {result.shape}"
        )

    return result