"""
Run local token **estimates** (``len(text) // 4``) across Edvise prompt builders for one snapshot.

Typical use: load a real enriched schema contract + manifests + reference artifacts,
then call :func:`run_prompt_token_audit_bundle` and inspect the ``builders`` map
(per-section estimates, ``buckets``, and ``total_estimated_tokens`` per prompt).

No API calls—see :mod:`edvise.genai.mapping.shared.token_audit.prompt_token_audit`.
"""

from __future__ import annotations

from typing import Any

from edvise.genai.mapping.identity_agent.grain_inference.prompt import (
    audit_identity_agent_prompt,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.prompt import (
    audit_hook_generation_prompt,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLItem
from edvise.genai.mapping.identity_agent.profiling import RankedCandidateProfiles
from edvise.genai.mapping.identity_agent.term_normalization.prompt import (
    audit_term_normalization_batch_user_prompt,
    audit_term_normalization_prompt,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import (
    audit_step2a_prompt,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.prompt import (
    audit_step2b_prompt,
)


def run_prompt_token_audit_bundle(
    *,
    institution_id: str,
    institution_schema_contract: dict,
    institution_mapping_manifest: dict,
    cohort_schema_class: type,
    course_schema_class: type,
    reference_manifests: list[dict],
    reference_transformation_maps: list[dict],
    output_path_manifest: str = "/tmp/edvise_field_mapping_manifest.json",
    output_path_transformation: str = "/tmp/edvise_transformation_map.json",
    reference_institution_ids: list[str] | None = None,
    log: bool = True,
    include_identity_grain: bool = False,
    grain_key_profile: RankedCandidateProfiles | None = None,
    grain_dataset_name: str = "student",
    grain_column_list: str | None = None,
    include_identity_term_single: bool = False,
    term_dataset: str = "student",
    term_row_selection_required: bool = False,
    term_candidates_json: str = "[]",
    term_raw_columns_json: str = "[]",
    term_batch_system: bool = False,
    include_identity_term_batch: bool = False,
    grain_contracts_by_dataset: dict | None = None,
    run_by_dataset: dict | None = None,
    include_hook_generation_sample: bool = False,
    hook_audit_item: HITLItem | None = None,
    hook_config_snippet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Aggregate token audit dicts for Schema Mapping Agent Step 2a/2b and optional IdentityAgent prompts.

    Identity sections are **opt-in** because they need profiling payloads (or synthetic JSON strings).

    Parameters
    ----------
    include_identity_grain
        When true, requires ``grain_key_profile`` and ``grain_column_list`` (or pass a pre-built
        column list text block).
    include_identity_term_batch
        When true, requires ``grain_contracts_by_dataset`` and ``run_by_dataset`` for
        :func:`~edvise.genai.mapping.identity_agent.term_normalization.prompt.build_term_normalization_batch_user_message_from_grain_and_profiles`.
    include_hook_generation_sample
        When true, requires ``hook_audit_item`` and ``hook_config_snippet`` (HITL hook generation).
    """
    builders: dict[str, Any] = {}

    builders["schema_mapping_agent.step2a.single"] = audit_step2a_prompt(
        institution_id,
        output_path_manifest,
        institution_schema_contract,
        reference_manifests,
        cohort_schema_class,
        course_schema_class,
        reference_institution_ids,
        variant="single",
        log=log,
    )
    builders["schema_mapping_agent.step2a.cohort_pass"] = audit_step2a_prompt(
        institution_id,
        output_path_manifest,
        institution_schema_contract,
        reference_manifests,
        cohort_schema_class,
        course_schema_class,
        reference_institution_ids,
        variant="cohort_pass",
        log=log,
    )
    builders["schema_mapping_agent.step2a.course_pass"] = audit_step2a_prompt(
        institution_id,
        output_path_manifest,
        institution_schema_contract,
        reference_manifests,
        cohort_schema_class,
        course_schema_class,
        reference_institution_ids,
        variant="course_pass",
        log=log,
    )
    builders["schema_mapping_agent.step2b"] = audit_step2b_prompt(
        institution_id,
        output_path_transformation,
        institution_mapping_manifest,
        institution_schema_contract,
        cohort_schema_class,
        course_schema_class,
        reference_transformation_maps,
        reference_institution_ids,
        log=log,
    )

    if include_identity_grain:
        if grain_key_profile is None:
            raise ValueError(
                "grain_key_profile is required when include_identity_grain=True"
            )
        if grain_column_list is None:
            raise ValueError(
                "grain_column_list is required when include_identity_grain=True"
            )
        builders["identity_agent.grain_inference"] = audit_identity_agent_prompt(
            institution_id,
            grain_dataset_name,
            grain_key_profile,
            column_list=grain_column_list,
            log=log,
        )

    if include_identity_term_single:
        builders["identity_agent.term_normalization.single_table"] = (
            audit_term_normalization_prompt(
                institution_id,
                term_dataset,
                term_row_selection_required,
                term_candidates_json=term_candidates_json,
                raw_table_profile_columns_json=term_raw_columns_json,
                log=log,
                batch_system=term_batch_system,
            )
        )

    if include_identity_term_batch:
        if grain_contracts_by_dataset is None or run_by_dataset is None:
            raise ValueError(
                "grain_contracts_by_dataset and run_by_dataset are required when "
                "include_identity_term_batch=True"
            )
        builders["identity_agent.term_normalization.batch"] = (
            audit_term_normalization_batch_user_prompt(
                institution_id,
                grain_contracts_by_dataset,
                run_by_dataset,
                log=log,
            )
        )

    if include_hook_generation_sample:
        if hook_audit_item is None or hook_config_snippet is None:
            raise ValueError(
                "hook_audit_item and hook_config_snippet required when "
                "include_hook_generation_sample=True"
            )
        builders[f"identity_agent.hook_generation.{hook_audit_item.domain.value}"] = (
            audit_hook_generation_prompt(
                hook_audit_item,
                hook_config_snippet,
                log=log,
            )
        )

    return {
        "institution_id": institution_id,
        "builders": builders,
    }
