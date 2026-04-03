"""Orchestration: LLM call + parse per dataset, optional institution-wide loop with HITL routing."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from edvise.genai.identity_agent.profiling.key_profiler import KeyProfile

from .prompt_builder import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    build_identity_agent_user_message,
    parse_identity_grain_contract,
)
from .schemas import IDENTITY_CONFIDENCE_HITL_THRESHOLD, IdentityGrainContract


def run_identity_agent(
    *,
    institution_id: str,
    dataset_name: str,
    key_profile: KeyProfile,
    df: pd.DataFrame,
    llm_complete: Callable[[str, str], str],
) -> IdentityGrainContract:
    """
    Build IdentityAgent prompts, call ``llm_complete(system, user)``, parse JSON to a contract.

    ``llm_complete`` must return raw model text (optionally fenced); it receives the fixed
    system prompt and the per-dataset user message from :func:`build_identity_agent_user_message`.
    """
    user = build_identity_agent_user_message(
        institution_id, dataset_name, key_profile, df=df
    )
    raw = llm_complete(IDENTITY_AGENT_SYSTEM_PROMPT, user)
    return parse_identity_grain_contract(raw)


def run_identity_agents_for_institution(
    *,
    institution_id: str,
    institution_profiles: Mapping[str, KeyProfile],
    dfs: Mapping[str, pd.DataFrame],
    llm_complete: Callable[[str, str], str],
    confidence_threshold: float = IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    queue_for_hitl_review: Callable[[IdentityGrainContract], None] | None = None,
    auto_approve_and_apply: Callable[[IdentityGrainContract], None] | None = None,
) -> dict[str, IdentityGrainContract]:
    """
    For each ``(dataset_name, key_profile)`` in ``institution_profiles``, run
    :func:`run_identity_agent` with ``dfs[dataset_name]``, then route by HITL / confidence.

    - If ``contract.hitl_flag`` or ``contract.confidence < confidence_threshold``:
      ``queue_for_hitl_review(contract)`` when provided.
    - Else: ``auto_approve_and_apply(contract)`` when provided.

    Raises ``KeyError`` if a profiled dataset name is missing from ``dfs``.
    """
    q = queue_for_hitl_review or (lambda _c: None)
    a = auto_approve_and_apply or (lambda _c: None)
    contracts: dict[str, IdentityGrainContract] = {}

    for dataset_name, key_profile in institution_profiles.items():
        if dataset_name not in dfs:
            raise KeyError(
                f"No DataFrame for dataset {dataset_name!r} in dfs "
                f"(have {list(dfs.keys())!r})"
            )
        contract = run_identity_agent(
            institution_id=institution_id,
            dataset_name=dataset_name,
            key_profile=key_profile,
            df=dfs[dataset_name],
            llm_complete=llm_complete,
        )
        contracts[dataset_name] = contract
        if contract.hitl_flag or contract.confidence < confidence_threshold:
            q(contract)
        else:
            a(contract)

    return contracts
