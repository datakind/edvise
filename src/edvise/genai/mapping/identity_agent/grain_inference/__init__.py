"""
Grain inference: profiling prep → prompts, validated LLM output schema, runners, optional dedupe.

Use :func:`build_identity_profiling_run_by_dataset` to go from bronze CSVs to per-dataset user
messages plus :class:`~profiling.schemas.RankedCandidateProfiles`. Step 1 stats come from
``edvise.genai.mapping.identity_agent.profiling``.
"""

from ..dataset_io import load_school_dataset_dataframe
from . import deduplication
from edvise.genai.mapping.shared.databricks_ai_gateway import (
    DEFAULT_GATEWAY_CLAUDE_HAIKU_MODEL_ID,
    DEFAULT_GATEWAY_CLAUDE_SONNET_MODEL_ID,
    build_mlflow_ai_gateway_base_url,
    create_openai_client_for_databricks_gateway,
    make_databricks_gateway_llm_complete,
    require_databricks_token,
    resolve_ai_gateway_base_url,
    resolve_column_roles_gateway_model_id,
    resolve_databricks_workspace_host,
    resolve_databricks_workspace_id,
    resolve_gateway_model_id,
    wrap_llm_complete_with_retries,
)
from .grain_gateway_logging import log_grain_auto_approve, log_grain_hitl_queue
from .hitl_uniqueness_backfill import (
    backfill_hitl_uniqueness_scores,
    backfill_hitl_uniqueness_scores_by_table,
    backfill_hitl_uniqueness_scores_from_measured_keys,
    backfill_hitl_uniqueness_scores_from_key_profile,
)
from .prompt import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    IDENTITY_AGENT_USER_TEMPLATE,
    build_identity_agent_system_prompt,
    build_identity_agent_user_message,
    format_column_list,
    parse_grain_contract,
    parse_grain_contract_with_hitl,
    parse_institution_grain_contracts,
    strip_json_fences,
)
from .runner import (
    run_identity_agent_with_hitl,
    run_identity_agents_for_institution_with_hitl,
)
from .run_by_dataset import (
    IdentityProfilingDatasetResult,
    build_identity_profiling_run_by_dataset,
    identity_profiling_run_to_jsonable,
    write_identity_profiling_artifacts,
)
from edvise.genai.mapping.shared.hitl import PIPELINE_HITL_CONFIDENCE_THRESHOLD
from edvise.genai.mapping.shared.grain.dedup_execution import drop_duplicate_keys
from edvise.genai.mapping.shared.grain.dedup_strategies import (
    DedupPolicyStrategy,
    GrainResolutionDedupStrategy,
    GrainResolutionDedupStrategyAny,
    SmaGrainMultiplicityProposalStrategy,
    SmaOnlyGrainResolutionDedupStrategy,
)
from .schemas import (
    DedupPolicy,
    DedupStrategy,
    GrainContract,
    InstitutionGrainContract,
    build_institution_grain_contracts,
)

__all__ = [
    "DEFAULT_GATEWAY_CLAUDE_HAIKU_MODEL_ID",
    "DEFAULT_GATEWAY_CLAUDE_SONNET_MODEL_ID",
    "build_mlflow_ai_gateway_base_url",
    "PIPELINE_HITL_CONFIDENCE_THRESHOLD",
    "IdentityProfilingDatasetResult",
    "identity_profiling_run_to_jsonable",
    "write_identity_profiling_artifacts",
    "DedupPolicy",
    "DedupPolicyStrategy",
    "DedupStrategy",
    "GrainResolutionDedupStrategy",
    "GrainResolutionDedupStrategyAny",
    "SmaGrainMultiplicityProposalStrategy",
    "SmaOnlyGrainResolutionDedupStrategy",
    "IDENTITY_AGENT_SYSTEM_PROMPT",
    "IDENTITY_AGENT_USER_TEMPLATE",
    "GrainContract",
    "InstitutionGrainContract",
    "build_identity_agent_system_prompt",
    "build_identity_agent_user_message",
    "drop_duplicate_keys",
    "format_column_list",
    "build_institution_grain_contracts",
    "backfill_hitl_uniqueness_scores",
    "backfill_hitl_uniqueness_scores_by_table",
    "backfill_hitl_uniqueness_scores_from_measured_keys",
    "backfill_hitl_uniqueness_scores_from_key_profile",
    "build_identity_profiling_run_by_dataset",
    "create_openai_client_for_databricks_gateway",
    "load_school_dataset_dataframe",
    "log_grain_auto_approve",
    "log_grain_hitl_queue",
    "make_databricks_gateway_llm_complete",
    "wrap_llm_complete_with_retries",
    "parse_grain_contract",
    "parse_grain_contract_with_hitl",
    "parse_institution_grain_contracts",
    "run_identity_agent_with_hitl",
    "run_identity_agents_for_institution_with_hitl",
    "require_databricks_token",
    "resolve_ai_gateway_base_url",
    "resolve_databricks_workspace_host",
    "resolve_databricks_workspace_id",
    "resolve_column_roles_gateway_model_id",
    "resolve_gateway_model_id",
    "strip_json_fences",
]
