"""
Deterministic join resolver for Agent 2 pipeline.

Runs after 2a (FieldMappingManifest approval) and before 2c (TransformationMap).
Infers join graph from:
  1. Approved manifest — which source_tables are referenced per entity type
  2. Schema contract   — unique_keys + column inventory per dataset

Produces a JoinGraph per entity type (cohort / course) which is:
  - Persisted to schema_contract for human review alongside 2c
  - Executed deterministically by join_resolver.execute() to produce flat input DataFrames

Design:
  - Base table = dataset whose unique keys are a subset of the target schema unique constraint
  - Join candidates = all other referenced tables
  - Join keys = intersection of base table columns and candidate unique keys
  - Foreign key fallback = scan all base table columns for match to candidate unique keys
  - Fan-out detection = flag when candidate grain is finer than base table grain
  - Join order = topological sort by key dependencies (base table first, then dependents)
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
    left: str                          # Left table name (running DataFrame after prior steps)
    right: str                         # Right table name
    left_on: list[str]                 # Join keys on left
    right_on: list[str]                # Join keys on right (may differ from left_on)
    how: str = "left"                  # Join type — always left for safety
    fan_out_risk: bool = False         # True if right table grain is finer than left
    fan_out_note: Optional[str] = None # Human-readable explanation of fan-out risk


@dataclass
class JoinGraph:
    """
    Complete join graph for one entity type.
    Produced by JoinResolver, reviewed by human, executed by execute().
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
                "student_df": {
                  "unique_keys": ["student_id", "term"],
                  "normalized_columns": {"col": "col", ...},
                  "dtypes": {"col": "dtype", ...}
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

        # --- 1. Collect referenced tables from manifest ---
        referenced = {
            m.source_table
            for m in manifest.mappings
            if m.source_table is not None and m.source_columns
        }
        logger.info(f"[{entity_type}] Referenced tables from manifest: {referenced}")

        if not referenced:
            raise ValueError(f"No source tables found in approved manifest for {entity_type}")

        # Validate that all referenced tables exist in schema contract
        missing = referenced - set(self.datasets.keys())
        if missing:
            available = list(self.datasets.keys())
            error_parts = [f"Dataset '{table}' not found in schema contract" for table in missing]
            for table in missing:
                suggestions = [d for d in available if table.lower() in d.lower() or d.lower() in table.lower()]
                if suggestions:
                    error_parts.append(f"  '{table}' - did you mean one of: {suggestions}?")
            error_parts.append(f"Available datasets: {available}")
            raise ValueError(
                f"Manifest references tables not in schema contract:\n" + "\n".join(error_parts)
            )

        # --- 2. Identify base table ---
        # Base table = dataset whose unique key columns most closely match
        # the target schema unique constraint after column normalization.
        # Fallback: dataset with smallest unique keys (closest to target grain).
        base_table = self._identify_base_table(entity_type, referenced)
        logger.info(f"[{entity_type}] Base table: {base_table}")

        join_candidates = referenced - {base_table}

        # --- 3. Build join steps ---
        graph = JoinGraph(entity_type=entity_type, base_table=base_table)
        resolved_tables = {base_table}
        # Track all columns available in the running DataFrame
        available_columns = set(self._get_columns(base_table))

        # Sort candidates for deterministic ordering —
        # tables whose join keys are already in available_columns come first
        ordered_candidates = self._order_candidates(
            join_candidates, available_columns
        )

        for candidate in ordered_candidates:
            step, warning = self._resolve_join(
                left_table=base_table,
                right_table=candidate,
                available_columns=available_columns,
                entity_type=entity_type,
            )
            if step:
                graph.steps.append(step)
                # Add right table columns to available pool for subsequent joins
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

    def _get_columns(self, table: str) -> list[str]:
        """Return normalized column names for a dataset."""
        if table not in self.datasets:
            # Suggest similar dataset names for better error messages
            available = list(self.datasets.keys())
            suggestions = [d for d in available if table.lower() in d.lower() or d.lower() in table.lower()]
            suggestion_msg = f" Available datasets: {available}"
            if suggestions:
                suggestion_msg = f" Did you mean one of: {suggestions}? Available datasets: {available}"
            raise ValueError(f"Dataset '{table}' not found in schema contract.{suggestion_msg}")
        return list(self.datasets[table].get("normalized_columns", {}).values())

    def _get_unique_keys(self, table: str) -> list[str]:
        """Return unique keys for a dataset."""
        if table not in self.datasets:
            # Suggest similar dataset names for better error messages
            available = list(self.datasets.keys())
            suggestions = [d for d in available if table.lower() in d.lower() or d.lower() in table.lower()]
            suggestion_msg = f" Available datasets: {available}"
            if suggestions:
                suggestion_msg = f" Did you mean one of: {suggestions}? Available datasets: {available}"
            raise ValueError(f"Dataset '{table}' not found in schema contract.{suggestion_msg}")
        return self.datasets[table].get("unique_keys", [])

    
    def _build_alias_map(
        self,
        manifest: "FieldMappingManifest",
    ) -> dict[str, dict[str, str]]:
        """
        Build a table -> {source_column: canonical_column} lookup from manifest aliases.
        """
        alias_map: dict[str, dict[str, str]] = {}
        for alias in manifest.column_aliases:
            alias_map.setdefault(alias.table, {})[alias.source_column] = alias.canonical_column
        return alias_map

    def _identify_base_table(
        self,
        entity_type: str,
        referenced: set[str],
    ) -> str:
        """
        Identify the base table for the join graph.

        The base table is the most granular table — the one whose unique keys
        are a superset of at least one other referenced table's unique keys.
        Other tables join INTO the base table, not the other way around.

        Strategy:
        1. Score each table by how many other referenced tables have unique keys that
        are subsets of this table's unique keys. Higher score = finer grain = base.
        2. Fallback: table with largest unique keys (finest grain) if no
        subset relationships exist.

        Example — UCF course:
            course_df unique keys: ["student_id", "term_descr", "crse_prefix",
                        "crse_number", "course_section_number",
                        "course_section_type", "cf_boe_term_id"]
            student_df unique keys: ["student_id", "term_desc"]

            student_df unique keys are NOT a subset of course_df unique keys due to term_desc vs
            term_descr mismatch — fallback fires, course_df wins on largest unique keys.

            Once column aliases are applied (term_descr → term_desc),
            student_df unique keys become a subset of course_df unique keys and score-based
            selection works correctly.
        """
        if len(referenced) == 1:
            return next(iter(referenced))

        best_table = None
        best_score = -1

        for table in referenced:
            uks = set(self._get_unique_keys(table))
            # Count how many other referenced tables have unique keys that are
            # subsets of this table's unique keys
            subset_count = sum(
                1 for other in referenced - {table}
                if set(self._get_unique_keys(other)).issubset(uks)
            )
            if subset_count > best_score:
                best_score = subset_count
                best_table = table

        if best_table and best_score > 0:
            return best_table

        # Fallback: largest unique key (finest grain)
        # Handles cases where column name mismatches prevent subset detection
        # e.g. term_descr vs term_desc across UCF tables before alias resolution
        largest = max(referenced, key=lambda t: len(self._get_unique_keys(t)))
        logger.warning(
            f"[{entity_type}] No unique key subset relationships found among referenced "
            f"tables {referenced} — falling back to largest unique key table '{largest}'. "
            f"If this is wrong, check column aliases in the manifest."
        )
        return largest

    def _order_candidates(
        self,
        candidates: set[str],
        available_columns: set[str],
    ) -> list[str]:
        """
        Order join candidates so tables whose keys are already available come first.
        This handles transitive dependencies — tables that need columns produced
        by a prior join are pushed to the end.
        """
        def priority(table: str) -> int:
            uks = set(self._get_unique_keys(table))
            # Higher priority = more unique key columns already in available_columns
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

        Strategy:
        1. Direct unique key match — right unique key columns exist in available_columns
        2. Foreign key scan — scan all left columns for match to right unique keys
        3. Fail — return warning, no step

        Fan-out detection:
        Right table has fan-out risk if its unique keys have MORE columns than
        the shared join keys — meaning multiple right rows per left row.
        """
        right_uks = self._get_unique_keys(right_table)
        right_cols = set(self._get_columns(right_table))

        # --- Strategy 1: Direct unique key match ---
        # All right unique key columns exist in available running DataFrame columns
        shared_keys = [k for k in right_uks if k in available_columns]
        if len(shared_keys) == len(right_uks):
            # Perfect match — right unique keys fully covered by available columns
            fan_out = False
            fan_out_note = None
        elif shared_keys:
            # Partial match — right unique keys have extra columns beyond shared keys
            # This means multiple right rows per join key = fan-out risk
            fan_out = True
            fan_out_note = (
                f"'{right_table}' unique keys are {right_uks} but join keys are "
                f"{shared_keys} — multiple rows per join key possible. "
                f"Review collapse strategy or pre-join filter."
            )
        else:
            # --- Strategy 2: Foreign key scan ---
            # Look for right unique key columns in all available left columns
            # (handles non-unique-key foreign keys like major_code)
            left_cols = available_columns
            fk_matches = [k for k in right_uks if k in left_cols]

            if fk_matches:
                shared_keys = fk_matches
                fan_out = len(fk_matches) < len(right_uks)
                fan_out_note = (
                    f"Join resolved via foreign key scan — '{right_table}' joins on "
                    f"{fk_matches} (not part of left unique keys). "
                    + (
                        f"Fan-out risk: right unique keys are {right_uks}."
                        if fan_out else ""
                    )
                ) if fan_out else None
            else:
                # --- Strategy 3: Fail ---
                return None, (
                    f"No shared keys found between running DataFrame and '{right_table}' "
                    f"(right unique keys: {right_uks}). Manual join declaration required."
                )

        return JoinStep(
            left=left_table,
            right=right_table,
            left_on=shared_keys,
            right_on=shared_keys,
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
        raise ValueError(f"Base table '{graph.base_table}' not found in dataframes")

    result = dataframes[graph.base_table].copy()
    logger.info(
        f"[{graph.entity_type}] Starting join from base '{graph.base_table}': "
        f"{result.shape}"
    )

    for step in graph.steps:
        if step.right not in dataframes:
            raise ValueError(f"Table '{step.right}' not found in dataframes")

        right_df = dataframes[step.right]

        if step.fan_out_risk:
            logger.warning(
                f"Executing join with fan-out risk: {step.left} ← {step.right} "
                f"on {step.left_on}. {step.fan_out_note}"
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
            f"Joined '{step.right}' on {step.left_on}: "
            f"{pre_shape} → {result.shape}"
        )

    return result
