"""
Onboard — resume_from="start"

Load IA outputs -> 2a LLM -> structural validation -> refinement LLM -> write HITL -> exit
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, cast

from edvise.genai.mapping.shared.reference_select import ensure_run_few_shot
from edvise.genai.mapping.state import job_state as _pipeline_job_state
from edvise.utils.llm_utils import llm_complete_with_parse_retry

from .helpers import _load_enriched_contract, _sma_llm_complete_run_once
from .paths import SMAPaths

LOGGER = logging.getLogger("edvise_sma")


def run_onboard_start(
    institution_id: str,
    catalog: str,
    paths: SMAPaths,
    client: Any,
    spark_session: Any,
    *,
    onboard_run_id: str,
    pipeline_version: str,
) -> None:
    from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import (
        build_step2a_batched_prompt,
        load_json,
        run_sma_refinement,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.eval import (
        validate_envelope_dict,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
        MappingManifestEnvelope,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.validation import (
        validate_manifest as validate_manifest_structure,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.hitl import (
        InstitutionSMAHITLItems,
        write_sma_hitl_artifact,
    )
    from edvise.genai.mapping.shared.schema_contract import (
        parse_enriched_schema_contract_for_sma,
    )
    from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
    from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema

    LOGGER.info("[onboard/start] Loading IA outputs for %s", institution_id)
    paths.run_root.mkdir(parents=True, exist_ok=True)

    enriched_contract = _load_enriched_contract(paths.ia_enriched_schema_contract)

    # Auto-select + snapshot few-shot into the run tree once (reuse if already present).
    snap, _selection = ensure_run_few_shot(
        paths.run_root,
        catalog=catalog,
        institution_id=institution_id,
        query_contract=enriched_contract,
        spark=spark_session,
    )
    reference_id = snap.reference_id
    ref_manifest_path = snap.manifest_map
    LOGGER.info(
        "[onboard/start] Reference manifest (run few_shot/ hash=%s id=%s): %s",
        snap.content_hash,
        reference_id,
        ref_manifest_path,
    )
    reference_manifest = load_json(str(ref_manifest_path))

    # Step 2a — mapping manifest LLM
    LOGGER.info("[onboard/start] Step 2a — mapping manifest LLM")
    prompt_2a = build_step2a_batched_prompt(
        institution_id=institution_id,
        output_path=str(paths.manifest_map),
        institution_schema_contract=enriched_contract,
        reference_manifests=[reference_manifest],
        reference_institution_ids=[reference_id],
        cohort_schema_class=RawEdviseStudentDataSchema,
        course_schema_class=RawEdviseCourseDataSchema,
    )

    # cache_system_prompt is a no-op for the Step 2a call below (system="" there); it only
    # activates for the refinement calls further down, which reuse this same llm_sma and send
    # a static, ~2.5-3.4k token system prompt (build_refinement_pass1/2_system_prompt) across
    # up to 4 calls per institution.
    llm_sma = _sma_llm_complete_run_once(client, cache_system_prompt=True)

    def _parse_step2a_envelope(raw: str) -> MappingManifestEnvelope:
        manifest_dict = json.loads(raw)
        # Step 2a agent schema omits envelope-only fields (see MappingManifestEnvelope).
        if isinstance(manifest_dict, dict):
            manifest_dict["institution_id"] = institution_id
            manifest_dict["pipeline_version"] = pipeline_version
        ok, err = validate_envelope_dict(manifest_dict)
        if not ok:
            LOGGER.warning(
                "[onboard/start] Manifest Pydantic validation warning: %s", err
            )
        return MappingManifestEnvelope.model_validate(manifest_dict)

    envelope_2a = llm_complete_with_parse_retry(
        llm_sma,
        "",
        prompt_2a,
        _parse_step2a_envelope,
        logger=LOGGER,
    )
    manifest_2a = envelope_2a.model_dump(mode="json", exclude_none=True)

    # Structural validation
    LOGGER.info("[onboard/start] Structural validation")
    schema_contract_sma = parse_enriched_schema_contract_for_sma(enriched_contract)
    structural_validation_errors: dict[str, list] = {}
    for entity_key, entity_manifest in envelope_2a.manifests.items():
        ek = entity_key.value if hasattr(entity_key, "value") else str(entity_key)
        errs = validate_manifest_structure(entity_manifest, schema_contract_sma)
        structural_validation_errors[ek] = [e.model_dump(mode="json") for e in errs]

    paths.mapping_validation_manifest.write_text(
        json.dumps(structural_validation_errors, indent=2)
    )
    n_struct = sum(len(v) for v in structural_validation_errors.values())
    if n_struct:
        LOGGER.warning("[onboard/start] Structural validation: %d issue(s)", n_struct)
    else:
        LOGGER.info("[onboard/start] Structural validation: 0 issues")

    # Refinement LLM — two-pass per entity (4 calls total)
    LOGGER.info("[onboard/start] Refinement LLM (4 calls)")

    def _refinement_llm_complete(system: str, user: str) -> str:
        return cast(str, llm_sma(system, user))

    for entity_key, entity_manifest in list(envelope_2a.manifests.items()):
        ek = entity_key.value if hasattr(entity_key, "value") else str(entity_key)
        errs = validate_manifest_structure(entity_manifest, schema_contract_sma)
        LOGGER.info(
            "[onboard/start] Refinement: entity=%s (validation errors=%d)",
            ek,
            len(errs),
        )

        refined_fm, hitl_env = run_sma_refinement(
            institution_id=institution_id,
            entity_type=cast(Literal["cohort", "course"], ek),
            manifest=entity_manifest,
            validation_errors=errs,
            schema_contract=schema_contract_sma,
            llm_complete=_refinement_llm_complete,
        )
        envelope_2a.manifests[entity_key] = refined_fm

        hitl_basename = (
            "cohort_hitl_manifest.json"
            if ek == "cohort"
            else "course_hitl_manifest.json"
        )
        write_sma_hitl_artifact(paths.run_root, hitl_env, basename=hitl_basename)
        LOGGER.info(
            "[onboard/start] Refinement wrote %d HITL item(s) -> %s",
            len(hitl_env.items),
            paths.run_root / hitl_basename,
        )

    # Update manifest after refinement
    manifest_2a = envelope_2a.model_dump(mode="json", exclude_none=True)
    paths.manifest_map.write_text(json.dumps(manifest_2a, indent=2))
    LOGGER.info("[onboard/start] Wrote mapping manifest -> %s", paths.manifest_map)

    # Seed empty HITL envelopes if refinement produced none
    for hitl_path, entity_type in [
        (paths.cohort_hitl_manifest, "cohort"),
        (paths.course_hitl_manifest, "course"),
    ]:
        if not hitl_path.is_file():
            write_sma_hitl_artifact(
                paths.run_root,
                InstitutionSMAHITLItems(
                    institution_id=institution_id,
                    entity_type=cast(Literal["cohort", "course"], entity_type),
                    items=[],
                ),
                basename=hitl_path.name,
            )
            LOGGER.info("[onboard/start] Seeded empty HITL envelope -> %s", hitl_path)

    LOGGER.info("[onboard/start] Complete. Awaiting HITL review. Exiting.")
    _pipeline_job_state.after_sma_onboard_start(
        catalog,
        institution_id,
        onboard_run_id,
        cohort_path=paths.cohort_hitl_manifest,
        course_path=paths.course_hitl_manifest,
    )
