"""
Backward-compatible re-exports for SMA grain reconciliation.

Implementations live under :mod:`edvise.genai.mapping.schema_mapping_agent.grain_resolution`.
Prefer importing from that package for new code.
"""

from edvise.genai.mapping.schema_mapping_agent.grain_resolution import (
    build_sma_grain_hitl_items,
    detect_grain_scenario,
    profile_against_manifest_grain,
    propose_dedup_policy,
    run_grain_reconciliation_gate,
)

__all__ = [
    "build_sma_grain_hitl_items",
    "detect_grain_scenario",
    "profile_against_manifest_grain",
    "propose_dedup_policy",
    "run_grain_reconciliation_gate",
]
