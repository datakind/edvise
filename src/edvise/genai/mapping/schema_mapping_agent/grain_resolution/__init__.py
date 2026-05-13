"""
SMA grain resolution: mismatch detection, dedup proposals, HITL construction, and job wiring.

Execution orchestration (transformation map reduce, ``GrainReconciliationRequired``) remains in
:mod:`edvise.genai.mapping.schema_mapping_agent.execution.field_executor`; this package supplies
grain-specific resolution and the SMA job bridge in :mod:`.job`.
"""

from edvise.genai.mapping.shared.strip_json_fences import strip_json_fences

from .prompt import (
    DedupProposalLLM,
    propose_dedup_policy,
    proposal_to_grain_resolution,
)
from .hitl import build_sma_grain_hitl_items
from .job import (
    SmaGrainHitlPending,
    SmaSchemaMappingRunPaths,
    execute_transformation_map_for_sma_execute_mode,
    execute_transformation_map_for_sma_run,
    ia_post_clean_primary_key_for_dataset,
    reload_field_manifest_entity,
    run_onboard_gate_2_entity_with_grain_uc,
)
from .reconciliation_gate import (
    detect_grain_scenario,
    profile_against_manifest_grain,
    run_grain_reconciliation_gate,
)

__all__ = [
    "DedupProposalLLM",
    "SmaGrainHitlPending",
    "SmaSchemaMappingRunPaths",
    "build_sma_grain_hitl_items",
    "detect_grain_scenario",
    "execute_transformation_map_for_sma_execute_mode",
    "execute_transformation_map_for_sma_run",
    "ia_post_clean_primary_key_for_dataset",
    "profile_against_manifest_grain",
    "proposal_to_grain_resolution",
    "propose_dedup_policy",
    "reload_field_manifest_entity",
    "run_grain_reconciliation_gate",
    "run_onboard_gate_2_entity_with_grain_uc",
    "strip_json_fences",
]
