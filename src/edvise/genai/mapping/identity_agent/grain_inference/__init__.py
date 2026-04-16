"""
Grain inference: profiling prep → prompts, validated LLM output schema, runners, optional dedupe.

Use :func:`build_identity_profiling_run_by_dataset` to go from bronze CSVs to per-dataset user
messages plus :class:`~profiling.schemas.RankedCandidateProfiles`. Step 1 stats come from
``edvise.genai.mapping.identity_agent.profiling``.
"""

from ..dataset_io import load_school_dataset_dataframe
from . import deduplication
from .databricks_gateway import (
    DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL,
    DEFAULT_GATEWAY_MODEL_ID,
    create_openai_client_for_databricks_gateway,
    log_grain_auto_approve,
    log_grain_hitl_queue,
    make_databricks_gateway_llm_complete,
    require_databricks_token,
    resolve_ai_gateway_base_url,
    resolve_gateway_model_id,
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
)
from .schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    DedupPolicy,
    DedupStrategy,
    GrainContract,
    InstitutionGrainContract,
    build_institution_grain_contracts,
)

__all__ = [
    "DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL",
    "DEFAULT_GATEWAY_MODEL_ID",
    "IDENTITY_CONFIDENCE_HITL_THRESHOLD",
    "IdentityProfilingDatasetResult",
    "DedupPolicy",
    "DedupStrategy",
    "IDENTITY_AGENT_SYSTEM_PROMPT",
    "IDENTITY_AGENT_USER_TEMPLATE",
    "GrainContract",
    "InstitutionGrainContract",
    "build_identity_agent_system_prompt",
    "build_identity_agent_user_message",
    "deduplication",
    "format_column_list",
    "build_institution_grain_contracts",
    "build_identity_profiling_run_by_dataset",
    "create_openai_client_for_databricks_gateway",
    "load_school_dataset_dataframe",
    "log_grain_auto_approve",
    "log_grain_hitl_queue",
    "make_databricks_gateway_llm_complete",
    "parse_grain_contract",
    "parse_grain_contract_with_hitl",
    "parse_institution_grain_contracts",
    "run_identity_agent_with_hitl",
    "run_identity_agents_for_institution_with_hitl",
    "require_databricks_token",
    "resolve_ai_gateway_base_url",
    "resolve_gateway_model_id",
    "strip_json_fences",
]
