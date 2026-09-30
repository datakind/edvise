"""
Onboard — resume_from="start"

Profile -> Pass 1 grain LLM -> Pass 2 term LLM -> write HITL -> exit
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from edvise.genai.mapping.state import job_state as _pipeline_job_state

from .paths import IAPaths

LOGGER = logging.getLogger("edvise_ia")


def run_onboard_start(
    institution_id: str,
    paths: IAPaths,
    school_config: Any,
    llm_complete: Callable[[str, str], str],
    *,
    column_roles_llm_complete: Callable[[str, str], str] | None = None,
    catalog: str,
    onboard_run_id: str,
) -> None:
    from edvise.genai.mapping.identity_agent.column_roles import (
        run_column_roles_for_institution,
        write_column_roles_artifacts,
    )
    from edvise.genai.mapping.identity_agent.grain_inference import (
        build_identity_profiling_run_by_dataset,
        write_identity_profiling_artifacts,
    )
    from edvise.genai.mapping.identity_agent.grain_inference.runner import (
        run_identity_agents_for_institution_with_hitl,
    )
    from edvise.genai.mapping.identity_agent.grain_inference import (
        log_grain_auto_approve,
        log_grain_hitl_queue,
    )
    from edvise.genai.mapping.shared.hitl import PIPELINE_HITL_CONFIDENCE_THRESHOLD
    from edvise.genai.mapping.identity_agent.term_normalization.prompt import (
        TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT,
        build_term_normalization_batch_user_message_from_grain_and_profiles,
    )
    from edvise.genai.mapping.identity_agent.term_normalization.validation import (
        build_parse_institution_term_contracts_with_semantic_checks,
    )
    from edvise.genai.mapping.identity_agent.hitl import (
        write_identity_grain_artifacts,
        write_identity_term_artifacts,
    )
    from edvise.genai.mapping.identity_agent.grain_inference import (
        load_school_dataset_dataframe,
    )

    LOGGER.info("[onboard/start] Profiling datasets for %s", institution_id)
    paths.run_root.mkdir(parents=True, exist_ok=True)
    paths.cleaned_datasets.mkdir(parents=True, exist_ok=True)

    # §3a — Column role classification (lightweight LLM; Haiku by default)
    roles_llm = column_roles_llm_complete or llm_complete
    LOGGER.info("[onboard/start] ColumnRolesAgent")
    column_roles_by_dataset = run_column_roles_for_institution(
        institution_id=institution_id,
        school=school_config,
        llm_complete=roles_llm,
    )
    write_column_roles_artifacts(
        paths.profiling_output.parent,
        institution_id,
        column_roles_by_dataset,
        filename="column_roles_run.json",
    )

    # §3b — Profile (semantic keys + combinatorial KeyProfiler)
    run_by_dataset = build_identity_profiling_run_by_dataset(
        institution_id=institution_id,
        school=school_config,
        column_roles_by_dataset=column_roles_by_dataset,
    )
    write_identity_profiling_artifacts(
        paths.profiling_output.parent,
        institution_id,
        run_by_dataset,
    )
    LOGGER.info("[onboard/start] Profiled datasets: %s", list(run_by_dataset.keys()))

    # §4 — Pass 1: Grain LLM
    LOGGER.info("[onboard/start] Pass 1 — Grain LLM")
    institution_profiles = {
        name: run_by_dataset[name]["key_profile"] for name in run_by_dataset
    }
    dfs = {
        name: load_school_dataset_dataframe(school_config, name)
        for name in run_by_dataset
    }
    raw_table_profiles_by_table = {
        name: run_by_dataset[name]["raw_table_profile"] for name in run_by_dataset
    }
    contracts_by_dataset, grain_hitl_items, verifications_by_dataset = (
        run_identity_agents_for_institution_with_hitl(
            institution_id=institution_id,
            institution_profiles=institution_profiles,
            dfs=dfs,
            llm_complete=llm_complete,
            raw_table_profiles_by_table=raw_table_profiles_by_table,
            confidence_threshold=PIPELINE_HITL_CONFIDENCE_THRESHOLD,
            queue_for_hitl_review=lambda c: log_grain_hitl_queue(c, logger=LOGGER),
            auto_approve_and_apply=lambda c: log_grain_auto_approve(c, logger=LOGGER),
        )
    )
    for name, verification in verifications_by_dataset.items():
        run_by_dataset[name]["grain_verification"] = verification.to_jsonable()
    write_identity_profiling_artifacts(
        paths.profiling_output.parent,
        institution_id,
        run_by_dataset,
    )

    # §5 — Pass 2: Term batch LLM
    LOGGER.info("[onboard/start] Pass 2 — Term batch LLM")
    term_batch_user = (
        build_term_normalization_batch_user_message_from_grain_and_profiles(
            institution_id,
            contracts_by_dataset,
            run_by_dataset,
        )
    )
    from edvise.genai.mapping.shared.databricks_ai_gateway import (
        DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS,
        llm_complete_combined_message_content,
    )
    from edvise.genai.mapping.shared.token_audit.prompt_token_audit import (
        estimate_tokens,
    )
    from edvise.utils.llm_utils import llm_complete_with_parse_retry

    _term_combined = llm_complete_combined_message_content(
        TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT,
        term_batch_user,
    )
    _term_est_in = estimate_tokens(_term_combined)
    LOGGER.info(
        "[onboard/start] Pass 2 gateway request (same message shape as chat.completions): "
        "chars=%d est_input_tokens~=%d (len/4 heuristic) max_output_tokens=%d est_total~=%d "
        "(if est_total exceeds the route's context window, some gateways return 403 "
        "PERMISSION_DENIED; confirm with workspace model limits)",
        len(_term_combined),
        _term_est_in,
        DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS,
        _term_est_in + DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS,
    )
    _institution_term, term_hitl_items = llm_complete_with_parse_retry(
        llm_complete,
        TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT,
        term_batch_user,
        build_parse_institution_term_contracts_with_semantic_checks(run_by_dataset),
        logger=LOGGER,
    )
    term_contract_by_dataset = _institution_term.contracts_by_dataset()

    # Write HITL artifacts to run folder
    write_identity_grain_artifacts(
        paths.run_root,
        institution_id,
        contracts_by_dataset,
        grain_hitl_items,
        key_profiles_by_table={
            name: run_by_dataset[name]["key_profile"] for name in run_by_dataset
        },
        dfs_by_table=dfs,
    )
    write_identity_term_artifacts(
        paths.run_root,
        institution_id,
        term_contract_by_dataset,
        term_hitl_items,
    )
    LOGGER.info(
        "[onboard/start] Wrote grain HITL (%d item(s)) and term HITL (%d item(s)). Exiting.",
        len(grain_hitl_items),
        len(term_hitl_items),
    )
    _pipeline_job_state.after_ia_onboard_start(
        catalog,
        institution_id,
        onboard_run_id,
        grain_path=paths.grain_hitl,
        term_path=paths.term_hitl,
    )
