"""Orchestration: LLM call + parse per dataset, optional institution-wide loop with HITL routing."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import pandas as pd

from edvise.genai.mapping.identity_agent.hitl.schemas import HITLItem
from edvise.genai.mapping.identity_agent.profiling import (
    RankedCandidateProfiles,
    RawTableProfile,
)
from edvise.genai.mapping.shared.hitl import PIPELINE_HITL_CONFIDENCE_THRESHOLD
from edvise.genai.mapping.shared.token_audit.prompt_token_audit import estimate_tokens
from edvise.utils.data_cleaning import convert_to_snake_case

from edvise.utils.llm_utils import llm_complete_with_parse_retry

from .hitl_uniqueness_backfill import (
    backfill_hitl_uniqueness_scores_from_measured_keys,
    backfill_hitl_uniqueness_scores_from_key_profile,
)
from .prompt import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    build_identity_agent_user_message,
    parse_grain_contract_with_hitl,
)
from edvise.genai.mapping.shared.databricks_ai_gateway import (
    llm_complete_combined_message_content,
)
from .contract_validation import GrainContractVerification, verify_grain_contract
from .schemas import GrainContract

_LOG = logging.getLogger(__name__)


def run_identity_agent_with_hitl(
    *,
    institution_id: str,
    dataset_name: str,
    key_profile: RankedCandidateProfiles,
    df: pd.DataFrame,
    llm_complete: Callable[[str, str], str],
    raw_table_profile: RawTableProfile | None = None,
) -> tuple[GrainContract, list[HITLItem], GrainContractVerification]:
    """
    Build IdentityAgent prompts, call ``llm_complete(system, user)``, parse JSON to a contract
    and optional top-level ``hitl_items``.

    ``llm_complete`` must return raw model text (optionally fenced); it receives the fixed
    system prompt and the per-dataset user message from :func:`build_identity_agent_user_message`.
    If you only need the contract, unpack ``contract, _, _ = run_identity_agent_with_hitl(...)``.

    Pass ``raw_table_profile`` from :func:`edvise.genai.mapping.identity_agent.profiling.profile_candidate_keys`
    (same run as ``key_profile``) so the model sees per-column cardinality in the user message.
    """
    user = build_identity_agent_user_message(
        institution_id,
        dataset_name,
        key_profile,
        df=df,
        raw_table_profile=raw_table_profile,
    )
    _combined = llm_complete_combined_message_content(
        IDENTITY_AGENT_SYSTEM_PROMPT, user
    )
    _LOG.info(
        "[ia grain] dataset=%s gateway_input_chars=%d est_input_tokens~=%d",
        dataset_name,
        len(_combined),
        estimate_tokens(_combined),
    )

    def _parse_backfill_and_verify(
        raw: str,
    ) -> tuple[GrainContract, list[HITLItem], GrainContractVerification]:
        normalized_cols = [convert_to_snake_case(c) for c in df.columns]
        contract, items = parse_grain_contract_with_hitl(
            raw,
            available_columns_normalized=normalized_cols,
        )
        items = backfill_hitl_uniqueness_scores_from_key_profile(items, key_profile)
        contract, verification = verify_grain_contract(contract, df)
        items = backfill_hitl_uniqueness_scores_from_measured_keys(items, df)
        return contract, items, verification

    return llm_complete_with_parse_retry(
        llm_complete,
        IDENTITY_AGENT_SYSTEM_PROMPT,
        user,
        _parse_backfill_and_verify,
        logger=_LOG,
    )


def run_identity_agents_for_institution_with_hitl(
    *,
    institution_id: str,
    institution_profiles: Mapping[str, RankedCandidateProfiles],
    dfs: Mapping[str, pd.DataFrame],
    llm_complete: Callable[[str, str], str],
    raw_table_profiles_by_table: Mapping[str, RawTableProfile] | None = None,
    confidence_threshold: float = PIPELINE_HITL_CONFIDENCE_THRESHOLD,
    queue_for_hitl_review: Callable[[GrainContract], None] | None = None,
    auto_approve_and_apply: Callable[[GrainContract], None] | None = None,
) -> tuple[dict[str, GrainContract], list[HITLItem], dict[str, GrainContractVerification]]:
    """
    For each ``(dataset_name, key_profile)`` in ``institution_profiles``, runs
    :func:`run_identity_agent_with_hitl` with ``dfs[dataset_name]``, merges all per-table
    ``hitl_items`` into one list (for
    :func:`~edvise.genai.mapping.identity_agent.hitl.artifacts.write_identity_grain_artifacts`).

    Optional ``raw_table_profiles_by_table`` maps the same ``dataset_name`` keys to
    :class:`~edvise.genai.mapping.identity_agent.profiling.schemas.RawTableProfile` from the
    same profiling run as each ``key_profile`` (adds cardinality to the user message per table).

    - If ``contract.hitl_flag`` or ``contract.confidence <= confidence_threshold``:
      ``queue_for_hitl_review(contract)`` when provided.
    - Else: ``auto_approve_and_apply(contract)`` when provided.

    Raises ``KeyError`` if a profiled dataset name is missing from ``dfs``.
    """
    q = queue_for_hitl_review or (lambda _c: None)
    a = auto_approve_and_apply or (lambda _c: None)
    contracts: dict[str, GrainContract] = {}
    verifications: dict[str, GrainContractVerification] = {}
    all_hitl: list[HITLItem] = []

    for dataset_name, key_profile in institution_profiles.items():
        if dataset_name not in dfs:
            raise KeyError(
                f"No DataFrame for dataset {dataset_name!r} in dfs "
                f"(have {list(dfs.keys())!r})"
            )
        rtp = None
        if raw_table_profiles_by_table is not None:
            rtp = raw_table_profiles_by_table.get(dataset_name)
        contract, items, verification = run_identity_agent_with_hitl(
            institution_id=institution_id,
            dataset_name=dataset_name,
            key_profile=key_profile,
            df=dfs[dataset_name],
            llm_complete=llm_complete,
            raw_table_profile=rtp,
        )
        contracts[dataset_name] = contract
        verifications[dataset_name] = verification
        all_hitl.extend(items)
        if contract.hitl_flag or contract.confidence <= confidence_threshold:
            q(contract)
        else:
            a(contract)

    return contracts, all_hitl, verifications
