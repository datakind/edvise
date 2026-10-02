"""
Onboard — resume_from="gate_2"

Resolve manifest HITL -> 2b LLM -> transformation review HITL (UC) -> hook preview ->
hook_required -> execute.

When ``override_2a_manifest=true``: skip HITL; apply overrides -> 2b ... -> promote.
Overrides also apply on resume_from=start (which skips Step 2A so the manifest is not
regenerated).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from edvise.genai.mapping.shared.active_promotion import (
    promote_genai_mapping_to_active,
)
from edvise.genai.mapping.shared.databricks_ai_gateway import resolve_gateway_model_id
from edvise.genai.mapping.schema_mapping_agent.grain_resolution import (
    reload_field_manifest_entity,
    run_onboard_gate_2_entity_with_grain_uc,
)
from edvise.genai.mapping.shared.reference_select import resolve_run_few_shot_snapshot
from edvise.genai.mapping.state import job_state as _pipeline_job_state
from edvise.genai.mapping.state.hitl_poller import (
    DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
)
from edvise.utils.llm_utils import llm_complete_with_parse_retry

from .helpers import (
    _load_cleaned_dataframes,
    _load_enriched_contract,
    _load_institution_term_config_optional,
    _run_once,
    _run_pandera_validation,
    _sma_llm_complete_run_once,
    _write_output_data,
    apply_gate_2_manifest_overrides,
)
from .paths import SMAPaths

LOGGER = logging.getLogger("edvise_sma")


def run_onboard_gate_2(
    institution_id: str,
    catalog: str,
    paths: SMAPaths,
    client: Any,
    spark_session: Any,
    *,
    onboard_run_id: str,
    pipeline_version: str,
    db_run_id: str | None = None,
    override_2a_manifest: bool = False,
    overrides_json_path: str | None = None,
) -> None:
    from pydantic import ValidationError

    from edvise.genai.mapping.schema_mapping_agent.manifest.hitl import (
        check_sma_hitl_gate,
        resolve_sma_items,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
        FieldMappingManifest,
        MappingManifestEnvelope,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
        TransformationMap,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.prompt import (
        build_step2b_prompt,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.dedupe_plans import (
        dedupe_transformation_plans_in_wrapper,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import load_json
    from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
    from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema

    if override_2a_manifest:
        LOGGER.info(
            "[onboard/gate_2] override_2a_manifest=true — skipping sma_gate_1 HITL; "
            "applying post-gate mapping overrides for %s",
            institution_id,
        )
        apply_gate_2_manifest_overrides(
            paths,
            overrides_json_path or "",
            institution_id=institution_id,
            overridden_by="pipeline",
            original_db_run_id=(db_run_id or onboard_run_id),
        )
    else:
        LOGGER.info("[onboard/gate_2] Resolving HITL for %s", institution_id)

        LOGGER.info(
            "[onboard/gate_2] Waiting for Unity Catalog HITL approval (sma_gate_1)"
        )
        _pipeline_job_state.wait_for_gate(
            _pipeline_job_state.GATE_SMA_1,
            catalog,
            onboard_run_id,
            institution_id=institution_id,
            poll_interval_seconds=DEFAULT_HITL_POLL_INTERVAL_SECONDS,
            timeout_seconds=DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
        )

        # Resolve HITL into mapping manifest
        for hitl_path in (paths.cohort_hitl_manifest, paths.course_hitl_manifest):
            resolve_sma_items(
                hitl_path,
                paths.manifest_map,
                resolved_by="pipeline",
                run_log_path=paths.run_log,
                db_run_id=db_run_id,
            )

        # Gate check — raises HITLBlockingError if any items still pending
        LOGGER.info("[onboard/gate_2] HITL gate check")
        for hitl_path in (paths.cohort_hitl_manifest, paths.course_hitl_manifest):
            check_sma_hitl_gate(hitl_path)

    # Reload manifest after HITL resolve and/or overrides
    manifest_2a = json.loads(paths.manifest_map.read_text())
    MappingManifestEnvelope.model_validate(manifest_2a)

    # Load reference transformation map from the run-local few-shot snapshot only
    # (never re-resolve references/*/current/ — mid-run republish must not change 2b).
    snap = resolve_run_few_shot_snapshot(paths.run_root)
    if snap is None:
        raise FileNotFoundError(
            f"Run few-shot snapshot missing under {paths.few_shot_root}. "
            "SMA onboard start must materialize few_shot/ before gate_2."
        )
    reference_id = snap.reference_id
    ref_tm_path = snap.transformation_map
    LOGGER.info(
        "[onboard/gate_2] Reference transformation map (run few_shot/ hash=%s): %s",
        snap.content_hash,
        ref_tm_path,
    )
    reference_tm = load_json(str(ref_tm_path))

    enriched_contract = _load_enriched_contract(paths.ia_enriched_schema_contract)
    institution_term_config = _load_institution_term_config_optional(
        paths.ia_identity_term_output,
        expected_institution_id=institution_id,
    )

    # Step 2b — transformation map LLM
    LOGGER.info("[onboard/gate_2] Step 2b — transformation map LLM")
    prompt_2b = build_step2b_prompt(
        institution_id=institution_id,
        output_path=str(paths.transformation_map),
        institution_mapping_manifest=manifest_2a,
        institution_schema_contract=enriched_contract,
        cohort_schema_class=RawEdviseStudentDataSchema,
        course_schema_class=RawEdviseCourseDataSchema,
        reference_transformation_maps=[reference_tm],
        reference_institution_ids=[reference_id],
        institution_term_config=institution_term_config,
    )

    llm_sma = _sma_llm_complete_run_once(client, catalog=catalog)

    def _parse_step2b_transformation_wrapper(raw: str) -> dict:
        data = json.loads(raw)
        if not isinstance(data, dict):
            ve = ValueError("Root JSON must be an object")
            raise ValidationError.from_exception_data(
                "Step2bTransformationRoot",
                [
                    {
                        "type": "dict_type",
                        "loc": (),
                        "input": data,
                        "ctx": {"error": ve},
                    }
                ],
            )
        dedupe_transformation_plans_in_wrapper(data, log=LOGGER)
        data["institution_id"] = institution_id
        data["pipeline_version"] = pipeline_version
        tmaps = data.get("transformation_maps")
        if not isinstance(tmaps, dict):
            ve = ValueError("transformation_maps must be an object")
            raise ValidationError.from_exception_data(
                "TransformationMaps",
                [
                    {
                        "type": "dict_type",
                        "loc": ("transformation_maps",),
                        "input": tmaps,
                        "ctx": {"error": ve},
                    }
                ],
            )
        for entity_type in ("cohort", "course"):
            sec = tmaps.get(entity_type)
            if not isinstance(sec, dict):
                ve = ValueError("Expected an object")
                raise ValidationError.from_exception_data(
                    "TransformationSection",
                    [
                        {
                            "type": "dict_type",
                            "loc": ("transformation_maps", entity_type),
                            "input": sec,
                            "ctx": {"error": ve},
                        }
                    ],
                )
            tm_dict = {
                **sec,
                "institution_id": institution_id,
                "pipeline_version": pipeline_version,
                "entity_type": entity_type,
            }
            TransformationMap.model_validate(tm_dict)
        from edvise.genai.mapping.schema_mapping_agent.transformation.validation import (
            raise_pydantic_validation_error_if_any,
            validate_transformation_plans_against_manifest,
        )

        raise_pydantic_validation_error_if_any(
            validate_transformation_plans_against_manifest(data, manifest_2a)
        )
        return data

    transformation_data = llm_complete_with_parse_retry(
        llm_sma,
        "",
        prompt_2b,
        _parse_step2b_transformation_wrapper,
        logger=LOGGER,
    )

    from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.review_hitl import (
        apply_transformation_review_resolutions,
        build_transformation_review_hitl_file_for_entity,
        write_transformation_review_hitl_file,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.gates import (
        check_transformation_review_hitl_gate,
    )

    cohort_tr = build_transformation_review_hitl_file_for_entity(
        transformation_data,
        institution_id=institution_id,
        entity_type="cohort",
        pipeline_version=pipeline_version,
    )
    course_tr = build_transformation_review_hitl_file_for_entity(
        transformation_data,
        institution_id=institution_id,
        entity_type="course",
        pipeline_version=pipeline_version,
    )
    write_transformation_review_hitl_file(paths.cohort_transformation_review, cohort_tr)
    write_transformation_review_hitl_file(paths.course_transformation_review, course_tr)
    LOGGER.info(
        "[onboard/gate_2] Transformation review HITL — cohort_items=%d course_items=%d",
        len(cohort_tr.items),
        len(course_tr.items),
    )
    _pipeline_job_state.register_sma_gate_2_transformation_review_artifacts(
        catalog,
        institution_id,
        onboard_run_id,
        cohort_transformation_review_path=paths.cohort_transformation_review,
        course_transformation_review_path=paths.course_transformation_review,
    )
    LOGGER.info(
        "[onboard/gate_2] Waiting for Unity Catalog HITL approval "
        "(sma_gate_2_transformation_review)"
    )
    _pipeline_job_state.wait_for_gate(
        _pipeline_job_state.GATE_SMA_2_TRANSFORMATION_REVIEW,
        catalog,
        onboard_run_id,
        institution_id=institution_id,
        poll_interval_seconds=DEFAULT_HITL_POLL_INTERVAL_SECONDS,
        timeout_seconds=DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    )
    for _tr_path in (
        paths.cohort_transformation_review,
        paths.course_transformation_review,
    ):
        check_transformation_review_hitl_gate(_tr_path)
    _pipeline_job_state.complete_gate(
        _pipeline_job_state.GATE_SMA_2_TRANSFORMATION_REVIEW,
        catalog,
        institution_id,
        onboard_run_id,
    )
    transformation_data = apply_transformation_review_resolutions(
        transformation_data,
        cohort_review_path=paths.cohort_transformation_review,
        course_review_path=paths.course_transformation_review,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.validation import (
        validate_transformation_plans_against_manifest,
    )

    post_review_plan_errors = validate_transformation_plans_against_manifest(
        transformation_data,
        manifest_2a,
    )
    if post_review_plan_errors:
        details = "; ".join(e.detail for e in post_review_plan_errors)
        raise ValueError(
            "Transformation plan / manifest alignment failed after review: " + details
        )

    _sma_gateway_model_id = resolve_gateway_model_id(catalog)

    def _sma_hook_llm_complete(system: str, user: str) -> str:
        prompt = f"{system.strip()}\n\n---\n\n{user.strip()}"
        result = _run_once(_sma_gateway_model_id, prompt, client)
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "SMA transform hook LLM failed")
        resp = result.get("response")
        if not isinstance(resp, str) or not resp.strip():
            raise RuntimeError("SMA transform hook LLM returned empty response")
        return resp

    from edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize import (
        materialize_hook_specs_to_file,
    )
    from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain
    from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_generation import (
        generate_sma_transform_hook_preview_rows_for_entity,
        load_hook_spec_rows_from_sma_preview_path,
        load_hook_specs_from_sma_preview_path,
        write_sma_transform_hook_preview_json,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_required_hitl import (
        attach_materialized_hook_specs_to_plans,
    )

    LOGGER.info("[onboard/gate_2] Transform hook generation (preview)")
    cohort_preview_rows = generate_sma_transform_hook_preview_rows_for_entity(
        transformation_data,
        manifest_2a,
        institution_id=institution_id,
        entity_type="cohort",
        llm_complete=_sma_hook_llm_complete,
    )
    course_preview_rows = generate_sma_transform_hook_preview_rows_for_entity(
        transformation_data,
        manifest_2a,
        institution_id=institution_id,
        entity_type="course",
        llm_complete=_sma_hook_llm_complete,
    )
    write_sma_transform_hook_preview_json(
        output_path=paths.cohort_transformation_hook_preview,
        institution_id=institution_id,
        domain="schema_mapping_transform_cohort",
        spec_rows=cohort_preview_rows,
    )
    write_sma_transform_hook_preview_json(
        output_path=paths.course_transformation_hook_preview,
        institution_id=institution_id,
        domain="schema_mapping_transform_course",
        spec_rows=course_preview_rows,
    )
    LOGGER.info(
        "[onboard/gate_2] Transform hook preview — cohort_specs=%d course_specs=%d",
        len(cohort_preview_rows),
        len(course_preview_rows),
    )
    _pipeline_job_state.register_sma_gate_2_hook_preview_artifacts(
        catalog,
        institution_id,
        onboard_run_id,
        cohort_transformation_hook_preview_path=paths.cohort_transformation_hook_preview,
        course_transformation_hook_preview_path=paths.course_transformation_hook_preview,
    )
    LOGGER.info(
        "[onboard/gate_2] Waiting for Unity Catalog HITL approval (sma_gate_2_hook_preview)"
    )
    _pipeline_job_state.wait_for_gate(
        _pipeline_job_state.GATE_SMA_2_HOOK_PREVIEW,
        catalog,
        onboard_run_id,
        institution_id=institution_id,
        poll_interval_seconds=DEFAULT_HITL_POLL_INTERVAL_SECONDS,
        timeout_seconds=DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    )
    _pipeline_job_state.complete_gate(
        _pipeline_job_state.GATE_SMA_2_HOOK_PREVIEW,
        catalog,
        institution_id,
        onboard_run_id,
    )
    preview_hook_specs = load_hook_specs_from_sma_preview_path(
        paths.cohort_transformation_hook_preview
    ) + load_hook_specs_from_sma_preview_path(paths.course_transformation_hook_preview)
    if preview_hook_specs:
        materialize_hook_specs_to_file(
            preview_hook_specs,
            repo_root=paths.run_root,
            domain=HITLDomain.TRANSFORM,
        )
        LOGGER.info(
            "[onboard/gate_2] Materialized transform_hooks.py (%d HookSpec(s))",
            len(preview_hook_specs),
        )
        # Read back the (possibly reviewer-edited) preview rows and attach each hook_spec to
        # its matching hook_required plan — without this, hook_required never clears/points
        # anywhere and the executor treats the field as an unresolved gap even though the
        # materialized function above is ready to run.
        cohort_hook_rows = load_hook_spec_rows_from_sma_preview_path(
            paths.cohort_transformation_hook_preview
        )
        course_hook_rows = load_hook_spec_rows_from_sma_preview_path(
            paths.course_transformation_hook_preview
        )
        if cohort_hook_rows:
            transformation_data = attach_materialized_hook_specs_to_plans(
                transformation_data,
                entity_type="cohort",
                preview_rows=cohort_hook_rows,
            )
        if course_hook_rows:
            transformation_data = attach_materialized_hook_specs_to_plans(
                transformation_data,
                entity_type="course",
                preview_rows=course_hook_rows,
            )
        LOGGER.info(
            "[onboard/gate_2] Attached hook_spec to %d plan(s) — cohort=%d course=%d",
            len(cohort_hook_rows) + len(course_hook_rows),
            len(cohort_hook_rows),
            len(course_hook_rows),
        )

    tmaps = transformation_data.get("transformation_maps") or {}
    for _entity in ("cohort", "course"):
        _sec = tmaps.get(_entity)
        if isinstance(_sec, dict):
            _sec["pipeline_version"] = pipeline_version

    paths.transformation_map.write_text(json.dumps(transformation_data, indent=2))
    LOGGER.info(
        "[onboard/gate_2] Wrote transformation map -> %s", paths.transformation_map
    )

    # Load cleaned dataframes from IA run folder
    dataframes = _load_cleaned_dataframes(paths.ia_cleaned_datasets, enriched_contract)

    # Step 2c — execute transformation maps
    LOGGER.info("[onboard/gate_2] Step 2c — executing transformation maps")
    institution_id_from_tm = transformation_data.get("institution_id", institution_id)

    cohort_map_data = {
        **transformation_data["transformation_maps"]["cohort"],
        "institution_id": institution_id_from_tm,
        "pipeline_version": pipeline_version,
    }
    course_map_data = {
        **transformation_data["transformation_maps"]["course"],
        "institution_id": institution_id_from_tm,
        "pipeline_version": pipeline_version,
    }

    cohort_manifest = FieldMappingManifest.model_validate(
        manifest_2a["manifests"]["cohort"]
    )
    course_manifest = FieldMappingManifest.model_validate(
        manifest_2a["manifests"]["course"]
    )
    cohort_map = TransformationMap.model_validate(cohort_map_data)
    course_map = TransformationMap.model_validate(course_map_data)

    cohort_result, _ = run_onboard_gate_2_entity_with_grain_uc(
        catalog=catalog,
        institution_id=institution_id,
        onboard_run_id=onboard_run_id,
        paths=paths,
        db_run_id=db_run_id,
        transformation_map=cohort_map,
        manifest=cohort_manifest,
        entity="cohort",
        dataframes=dataframes,
        schema=RawEdviseStudentDataSchema,
        spark_session=spark_session,
        institution_id_from_tm=institution_id_from_tm,
        enriched_contract=enriched_contract,
        grain_hitl_path=paths.run_root / "cohort_sma_grain_hitl.json",
        poll_interval_seconds=DEFAULT_HITL_POLL_INTERVAL_SECONDS,
        timeout_seconds=DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
        hook_modules_root=paths.run_root,
    )
    course_manifest = reload_field_manifest_entity(paths.manifest_map, "course")
    course_result, _ = run_onboard_gate_2_entity_with_grain_uc(
        catalog=catalog,
        institution_id=institution_id,
        onboard_run_id=onboard_run_id,
        paths=paths,
        db_run_id=db_run_id,
        transformation_map=course_map,
        manifest=course_manifest,
        entity="course",
        dataframes=dataframes,
        schema=RawEdviseCourseDataSchema,
        spark_session=spark_session,
        institution_id_from_tm=institution_id_from_tm,
        enriched_contract=enriched_contract,
        grain_hitl_path=paths.run_root / "course_sma_grain_hitl.json",
        poll_interval_seconds=DEFAULT_HITL_POLL_INTERVAL_SECONDS,
        timeout_seconds=DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
        hook_modules_root=paths.run_root,
    )

    # Step 2d — Pandera validation (report only, does not block)
    LOGGER.info("[onboard/gate_2] Step 2d — Pandera validation")
    _run_pandera_validation(
        cohort_result, course_result, report_path=paths.pandera_validation_errors
    )

    # Write output data
    _write_output_data(paths.output_data, cohort_result, course_result)
    LOGGER.info("[onboard/gate_2] Promoting artifacts to active/")
    promote_genai_mapping_to_active(
        paths,
        institution_id=institution_id,
        onboard_run_id=onboard_run_id,
        pipeline_version=pipeline_version,
        uc_catalog=catalog,
    )
    LOGGER.info("[onboard/gate_2] Complete. Exiting.")
    # Last gate of the onboard run, so the run itself ends here rather than resuming.
    _pipeline_job_state.complete_gate(
        _pipeline_job_state.GATE_SMA_1,
        catalog,
        institution_id,
        onboard_run_id,
        run_status="complete",
    )
