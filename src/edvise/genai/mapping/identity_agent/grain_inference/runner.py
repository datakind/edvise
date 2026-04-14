"""Orchestration: LLM call + parse per dataset, optional institution-wide loop with HITL routing."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from edvise.genai.mapping.identity_agent.hitl.schemas import HITLItem
from edvise.genai.mapping.identity_agent.profiling import RankedCandidateProfiles

from .prompt_builder import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    build_identity_agent_user_message,
    parse_grain_contract_with_hitl,
)
from .schemas import IDENTITY_CONFIDENCE_HITL_THRESHOLD, GrainContract


def run_identity_agent_with_hitl(
    *,
    institution_id: str,
    dataset_name: str,
    key_profile: RankedCandidateProfiles,
    df: pd.DataFrame,
    llm_complete: Callable[[str, str], str],
) -> tuple[GrainContract, list[HITLItem]]:
    """
    Build IdentityAgent prompts, call ``llm_complete(system, user)``, parse JSON to a contract
    and optional top-level ``hitl_items``.

    ``llm_complete`` must return raw model text (optionally fenced); it receives the fixed
    system prompt and the per-dataset user message from :func:`build_identity_agent_user_message`.
    If you only need the contract, unpack ``contract, _ = run_identity_agent_with_hitl(...)``.
    """
    user = build_identity_agent_user_message(
        institution_id, dataset_name, key_profile, df=df
    )
    raw = llm_complete(IDENTITY_AGENT_SYSTEM_PROMPT, user)
    return parse_grain_contract_with_hitl(raw)


def run_identity_agents_for_institution_with_hitl(
    *,
    institution_id: str,
    institution_profiles: Mapping[str, RankedCandidateProfiles],
    dfs: Mapping[str, pd.DataFrame],
    llm_complete: Callable[[str, str], str],
    confidence_threshold: float = IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    queue_for_hitl_review: Callable[[GrainContract], None] | None = None,
    auto_approve_and_apply: Callable[[GrainContract], None] | None = None,
) -> tuple[dict[str, GrainContract], list[HITLItem]]:
    """
    For each ``(dataset_name, key_profile)`` in ``institution_profiles``, runs
    :func:`run_identity_agent_with_hitl` with ``dfs[dataset_name]``, merges all per-table
    ``hitl_items`` into one list (for
    :func:`~edvise.genai.mapping.identity_agent.hitl.artifacts.write_identity_grain_artifacts`).

    - If ``contract.hitl_flag`` or ``contract.confidence < confidence_threshold``:
      ``queue_for_hitl_review(contract)`` when provided.
    - Else: ``auto_approve_and_apply(contract)`` when provided.

    Raises ``KeyError`` if a profiled dataset name is missing from ``dfs``.
    """
    q = queue_for_hitl_review or (lambda _c: None)
    a = auto_approve_and_apply or (lambda _c: None)
    contracts: dict[str, GrainContract] = {}
    all_hitl: list[HITLItem] = []

    for dataset_name, key_profile in institution_profiles.items():
        if dataset_name not in dfs:
            raise KeyError(
                f"No DataFrame for dataset {dataset_name!r} in dfs "
                f"(have {list(dfs.keys())!r})"
            )
        contract, items = run_identity_agent_with_hitl(
            institution_id=institution_id,
            dataset_name=dataset_name,
            key_profile=key_profile,
            df=dfs[dataset_name],
            llm_complete=llm_complete,
        )
        contracts[dataset_name] = contract
        all_hitl.extend(items)
        if contract.hitl_flag or contract.confidence < confidence_threshold:
            q(contract)
        else:
            a(contract)

    return contracts, all_hitl
