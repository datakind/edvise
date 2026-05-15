"""
SMA grain resolution: mismatch detection, dedup proposals, HITL construction, and runner wiring.

Execution orchestration (transformation map reduce, ``GrainReconciliationRequired``) remains in
:mod:`edvise.genai.mapping.schema_mapping_agent.execution.field_executor`; this package supplies
grain-specific resolution and the SMA runner bridge in :mod:`.runner`.
"""

from edvise.genai.mapping.shared.utilities import strip_json_fences

from .hitl import (
    build_sma_grain_hitl_items,
    detect_grain_scenario,
    profile_against_manifest_grain,
    run_grain_reconciliation_gate,
)
from .prompt import (
    DedupProposalLLM,
    propose_dedup_policy,
    proposal_to_grain_resolution,
)
from .runner import (
    MAX_SMA_GRAIN_ROUNDS,
    SmaGrainHitlPending,
    SmaSchemaMappingRunPaths,
    execute_transformation_map_for_sma_execute_mode,
    execute_transformation_map_for_sma_run,
    ia_post_clean_primary_key_for_dataset,
    reload_field_manifest_entity,
    run_onboard_gate_2_entity_with_grain_uc,
)

__all__ = [
    "DedupProposalLLM",
    "MAX_SMA_GRAIN_ROUNDS",
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
