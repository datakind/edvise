"""
edvise_sma.py — SchemaMappingAgent pipeline job entry point.

Usage (Databricks job parameters):
    --institution_id    synthetic_edvise
    --catalog           dev_sst_02
    --mode              onboard | execute
    --resume_from       start | gate_2  (onboard only)
    --reference_id      required for onboard; few-shot from that school's
                        ``.../<ref_id>_silver/silver_volume/genai_mapping/active/{manifest_map,transformation_map}.json``
                        (same ``--catalog`` as the reference institution's volumes).

On Databricks, onboard mode best-effort updates ``{catalog}.genai_mapping`` pipeline state
(see :mod:`edvise.genai.mapping.state.job_state`); table setup and Spark are required.
"""

import os
import random
import sys
import time
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

# Layout: <git_root>/src/edvise/genai/mapping/scripts/<this_file>
# `import edvise` needs <git_root>/src on sys.path (package is <git_root>/src/edvise/).
# Databricks spark_python_task often exec()s this file without defining __file__.
_here = globals().get("__file__")
if _here:
    _script_dir = os.path.dirname(os.path.abspath(_here))
else:
    _argv0 = os.path.abspath(sys.argv[0]) if sys.argv else ""
    if _argv0.endswith(".py") and os.path.isfile(_argv0):
        _script_dir = os.path.dirname(_argv0)
    else:
        _script_dir = os.path.abspath(os.getcwd())
_src_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))
if os.path.isdir(_src_root) and _src_root not in sys.path:
    sys.path.insert(0, _src_root)

# Before any import that loads ``openai`` (Databricks may autolog it otherwise).
from edvise.genai.mapping.shared.mlflow_gateway_bootstrap import (
    disable_mlflow_side_effects_for_openai_gateway,
)

disable_mlflow_side_effects_for_openai_gateway()

from edvise.configs import genai as genai_cfg
from edvise.genai.mapping.shared.active_promotion import promote_genai_mapping_to_active
from edvise.genai.mapping.state import job_state as _pipeline_job_state
from edvise.genai.mapping.state import pipeline_state as _pipeline_state
from edvise.genai.mapping.state.hitl_poller import (
    DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    HITLTimeoutError,
)
from edvise.shared.logger import init_file_logging_at_path

LOGGER = logging.getLogger("edvise_sma")

# AI Gateway route for SMA onboard LLM steps (2a / refinement / 2b). Not a CLI flag.
_DEFAULT_SMA_GATEWAY_MODEL_ID = "claude-sonnet-test-genai-ai-data-cleaning"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass
class SMAPaths:
    # Run folder: ``runs/onboard/{onboard_run_id}/`` or ``runs/execute/{execute_run_id}/``
    run_root: Path
    manifest_map: Path
    mapping_validation_manifest: Path
    cohort_hitl_manifest: Path
    course_hitl_manifest: Path
    transformation_map: Path
    transform_hooks: Path           # optional, placeholder
    run_log: Path

    # IA outputs this job reads from (same execute or onboard run segment)
    ia_enriched_schema_contract: Path
    ia_cleaned_datasets: Path       # directory

    # Active folder (promoted artifacts, what execute mode reads from)
    active_root: Path
    active_manifest_map: Path
    active_transformation_map: Path
    active_transform_hooks: Path
    active_enriched_schema_contract: Path

    # Optional upstream cleaned inputs (volume layout)
    genai_data: Path

    # Output data (written after execution)
    output_data: Path               # directory, one .parquet per entity


def resolve_run_paths(
    institution_id: str,
    catalog: str,
    *,
    mode: str,
    onboard_run_id: str | None = None,
    execute_run_id: str | None = None,
) -> SMAPaths:
    genai = Path(genai_cfg.silver_genai_mapping_root(institution_id, catalog=catalog))
    if mode == "onboard":
        rid = (onboard_run_id or "").strip()
        if not rid:
            raise ValueError("onboard_run_id is required when mode='onboard'")
        segment = ("onboard", rid)
    elif mode == "execute":
        rid = (execute_run_id or "").strip()
        if not rid:
            raise ValueError("execute_run_id is required when mode='execute'")
        segment = ("execute", rid)
    else:
        raise ValueError(f"resolve_run_paths: invalid mode={mode!r}")
    run_root = genai / "runs" / segment[0] / segment[1] / "schema_mapping_agent"
    ia_run_root = genai / "runs" / segment[0] / segment[1] / "identity_agent"
    active_root = genai / "active"

    return SMAPaths(
        run_root=run_root,
        manifest_map=run_root / "manifest_map.json",
        mapping_validation_manifest=run_root / "mapping_validation_manifest.json",
        cohort_hitl_manifest=run_root / "cohort_hitl_manifest.json",
        course_hitl_manifest=run_root / "course_hitl_manifest.json",
        transformation_map=run_root / "transformation_map.json",
        transform_hooks=run_root / "transform_hooks.py",
        run_log=run_root / "run_log.json",
        # IA outputs — same run segment under ``runs/onboard/...`` or ``runs/execute/...``
        ia_enriched_schema_contract=ia_run_root / "enriched_schema_contract.json",
        ia_cleaned_datasets=ia_run_root / "cleaned_datasets",
        # Active folder (flat under genai_mapping)
        active_root=active_root,
        active_manifest_map=active_root / "manifest_map.json",
        active_transformation_map=active_root / "transformation_map.json",
        active_transform_hooks=active_root / "transform_hooks.py",
        active_enriched_schema_contract=active_root / "enriched_schema_contract.json",
        genai_data=genai / "data",
        # Output data
        output_data=run_root / "data",
    )


def resolve_reference_sma_active_paths(
    reference_id: str, *, catalog: str
) -> tuple[Path, Path]:
    """Few-shot reference: promoted SMA artifacts under the reference school's ``genai_mapping/active/``."""
    active = Path(genai_cfg.silver_genai_mapping_root(reference_id, catalog=catalog)) / "active"
    return (
        active / "manifest_map.json",
        active / "transformation_map.json",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_enriched_contract(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Enriched schema contract not found: {path}. "
            "Run edvise_ia onboard/gate_1 first."
        )
    return json.loads(path.read_text())


def _load_cleaned_dataframes(cleaned_datasets_dir: Path, enriched_contract: dict) -> dict:
    import pandas as pd

    dataframes = {}
    for logical_name in enriched_contract.get("datasets", {}):
        pq = cleaned_datasets_dir / f"{logical_name}.parquet"
        if not pq.is_file():
            raise FileNotFoundError(
                f"Missing cleaned Parquet for dataset {logical_name!r}: {pq}. "
                "Run edvise_ia onboard/gate_1 first."
            )
        dataframes[logical_name] = pd.read_parquet(pq)
    LOGGER.info("Loaded cleaned dataframes: %s", list(dataframes.keys()))
    return dataframes


def _write_output_data(output_data_dir: Path, cohort_result, course_result) -> None:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_data_dir / "cohort.parquet"
    course_path = output_data_dir / "course.parquet"
    cohort_result.df.to_parquet(cohort_path, index=False)
    course_result.df.to_parquet(course_path, index=False)
    LOGGER.info("Wrote cohort output -> %s (shape=%s)", cohort_path, cohort_result.df.shape)
    LOGGER.info("Wrote course output -> %s (shape=%s)", course_path, course_result.df.shape)


def _run_pandera_validation(cohort_result, course_result) -> None:
    import time
    import pandera

    from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
    from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema

    def _validate(df, schema, label):
        start = time.perf_counter()
        try:
            schema.validate(df, lazy=True)
            LOGGER.info("Pandera [%s]: PASSED (%.2fs)", label, time.perf_counter() - start)
        except pandera.errors.SchemaErrors as e:
            LOGGER.warning(
                "Pandera [%s]: FAILED — %d case(s) (%.2fs)",
                label,
                len(e.failure_cases),
                time.perf_counter() - start,
            )
        except Exception:
            LOGGER.exception("Pandera [%s]: validation error", label)

    _validate(cohort_result.df, RawEdviseStudentDataSchema, "cohort")
    _validate(course_result.df, RawEdviseCourseDataSchema, "course")


def _build_openai_client(catalog: str):
    """Build OpenAI-compatible client for Databricks AI Gateway."""
    from openai import OpenAI

    from edvise.genai.mapping.identity_agent.grain_inference.databricks_gateway import (
        disable_mlflow_tracing_for_openai_gateway_client,
        require_databricks_token,
        resolve_ai_gateway_base_url,
    )

    disable_mlflow_tracing_for_openai_gateway_client()
    return OpenAI(
        api_key=require_databricks_token(),
        base_url=resolve_ai_gateway_base_url(),
    )


def _run_once(model_id: str, prompt: str, client) -> dict:
    """
    Call :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.eval.run_once` with
    retries for transient gateway / transport failures (same policy as IA ``llm_complete``).
    """
    from edvise.genai.mapping.identity_agent.grain_inference.databricks_gateway import (
        gateway_run_once_error_text_is_retryable,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.eval import run_once

    max_attempts = 5
    initial_backoff_s = 2.0
    max_backoff_s = 60.0
    last: dict = {}
    for attempt in range(max_attempts):
        last = run_once(model_id, prompt, client)
        if last.get("success"):
            return last
        if attempt >= max_attempts - 1:
            return last
        err = last.get("error") or ""
        if not gateway_run_once_error_text_is_retryable(err):
            return last
        delay = min(max_backoff_s, initial_backoff_s * (2**attempt)) * (
            0.5 + random.random() * 0.5
        )
        LOGGER.warning(
            "SMA run_once non-success (attempt %d/%d); retry in %.1fs — %s",
            attempt + 1,
            max_attempts,
            delay,
            err[:300].replace("\n", " "),
        )
        time.sleep(delay)
    return last


# ---------------------------------------------------------------------------
# Onboard — resume_from="start"
# Load IA outputs -> 2a LLM -> structural validation -> refinement LLM -> write HITL -> exit
# ---------------------------------------------------------------------------

def run_onboard_start(
    institution_id: str,
    reference_id: str,
    catalog: str,
    paths: SMAPaths,
    client,
    spark_session,
    *,
    onboard_run_id: str,
):
    from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import (
        build_step2a_batched_prompt,
        load_json,
        run_sma_refinement,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.eval import validate_envelope_dict
    from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
        MappingManifestEnvelope,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.validation import (
        validate_manifest as validate_manifest_structure,
    )
    from edvise.genai.mapping.schema_mapping_agent.hitl import (
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

    # Load reference institution few-shot manifest from reference school's promoted SMA folder
    ref_manifest_path, _ = resolve_reference_sma_active_paths(
        reference_id, catalog=catalog
    )
    if not ref_manifest_path.exists():
        raise FileNotFoundError(
            f"Reference mapping manifest not found (expected promoted SMA active/): {ref_manifest_path}"
        )
    LOGGER.info("[onboard/start] Reference manifest (active): %s", ref_manifest_path)
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
    result_2a = _run_once(_DEFAULT_SMA_GATEWAY_MODEL_ID, prompt_2a, client)
    if not result_2a["success"]:
        raise RuntimeError(result_2a.get("error") or "Step 2a LLM failed")

    manifest_2a = json.loads(result_2a["response"])
    # Step 2a agent schema omits envelope-only fields (see MappingManifestEnvelope).
    if isinstance(manifest_2a, dict):
        manifest_2a["institution_id"] = institution_id
    ok, err = validate_envelope_dict(manifest_2a)
    if not ok:
        LOGGER.warning("[onboard/start] Manifest Pydantic validation warning: %s", err)

    envelope_2a = MappingManifestEnvelope.model_validate(manifest_2a)
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
        combined = system + "\n\n---\n\n" + user
        r = _run_once(_DEFAULT_SMA_GATEWAY_MODEL_ID, combined, client)
        if not r.get("success"):
            raise RuntimeError(r.get("error") or "SMA refinement LLM call failed")
        return r["response"]

    for entity_key, entity_manifest in list(envelope_2a.manifests.items()):
        ek = entity_key.value if hasattr(entity_key, "value") else str(entity_key)
        errs = validate_manifest_structure(entity_manifest, schema_contract_sma)
        LOGGER.info("[onboard/start] Refinement: entity=%s (validation errors=%d)", ek, len(errs))

        refined_fm, hitl_env = run_sma_refinement(
            institution_id=institution_id,
            entity_type=ek,
            manifest=entity_manifest,
            validation_errors=errs,
            schema_contract=schema_contract_sma,
            llm_complete=_refinement_llm_complete,
        )
        envelope_2a.manifests[entity_key] = refined_fm

        hitl_basename = (
            "cohort_hitl_manifest.json" if ek == "cohort" else "course_hitl_manifest.json"
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
                InstitutionSMAHITLItems(institution_id=institution_id, entity_type=entity_type, items=[]),
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


# ---------------------------------------------------------------------------
# Onboard — resume_from="gate_2"
# Resolve HITL -> gate check -> 2b LLM -> execute -> Pandera -> write outputs -> exit
# ---------------------------------------------------------------------------

def run_onboard_gate_2(
    institution_id: str,
    reference_id: str,
    catalog: str,
    paths: SMAPaths,
    client,
    spark_session,
    *,
    onboard_run_id: str,
    db_run_id: str | None = None,
):
    from edvise.genai.mapping.schema_mapping_agent.hitl import (
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
    from edvise.genai.mapping.schema_mapping_agent.transformation.eval import (
        validate_transformation_wrapper,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import load_json
    from edvise.genai.mapping.schema_mapping_agent.execution.field_executor import (
        execute_transformation_map,
    )
    from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
    from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema

    LOGGER.info("[onboard/gate_2] Resolving HITL for %s", institution_id)

    LOGGER.info("[onboard/gate_2] Waiting for Unity Catalog HITL approval (sma_gate_1)")
    _pipeline_job_state.wait_for_sma_gate_1_hitl(
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

    # Reload manifest after resolution
    manifest_2a = json.loads(paths.manifest_map.read_text())
    envelope_2a = MappingManifestEnvelope.model_validate(manifest_2a)

    # Gate check — raises HITLBlockingError if any items still pending
    LOGGER.info("[onboard/gate_2] HITL gate check")
    for hitl_path in (paths.cohort_hitl_manifest, paths.course_hitl_manifest):
        check_sma_hitl_gate(hitl_path)

    # Load reference transformation map for 2b few-shot from reference school's promoted SMA folder
    _, ref_tm_path = resolve_reference_sma_active_paths(reference_id, catalog=catalog)
    if not ref_tm_path.exists():
        raise FileNotFoundError(
            f"Reference transformation map not found (expected promoted SMA active/): {ref_tm_path}"
        )
    LOGGER.info("[onboard/gate_2] Reference transformation map (active): %s", ref_tm_path)
    reference_tm = load_json(str(ref_tm_path))

    enriched_contract = _load_enriched_contract(paths.ia_enriched_schema_contract)

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
    )
    result_2b = _run_once(_DEFAULT_SMA_GATEWAY_MODEL_ID, prompt_2b, client)
    if not result_2b["success"]:
        raise RuntimeError(result_2b.get("error") or "Step 2b LLM failed")

    transformation_data = json.loads(result_2b["response"])
    ok, err = validate_transformation_wrapper(transformation_data)
    if not ok:
        LOGGER.warning("[onboard/gate_2] Transformation map validation warning: %s", err)

    paths.transformation_map.write_text(json.dumps(transformation_data, indent=2))
    LOGGER.info("[onboard/gate_2] Wrote transformation map -> %s", paths.transformation_map)

    # Load cleaned dataframes from IA run folder
    dataframes = _load_cleaned_dataframes(paths.ia_cleaned_datasets, enriched_contract)

    # Step 2c — execute transformation maps
    LOGGER.info("[onboard/gate_2] Step 2c — executing transformation maps")
    institution_id_from_tm = transformation_data.get("institution_id", institution_id)

    cohort_map_data = {
        **transformation_data["transformation_maps"]["cohort"],
        "institution_id": institution_id_from_tm,
    }
    course_map_data = {
        **transformation_data["transformation_maps"]["course"],
        "institution_id": institution_id_from_tm,
    }

    cohort_manifest = FieldMappingManifest.model_validate(manifest_2a["manifests"]["cohort"])
    course_manifest = FieldMappingManifest.model_validate(manifest_2a["manifests"]["course"])
    cohort_map = TransformationMap.model_validate(cohort_map_data)
    course_map = TransformationMap.model_validate(course_map_data)

    cohort_result = execute_transformation_map(
        transformation_map=cohort_map,
        manifest=cohort_manifest,
        dataframes=dataframes,
        schema=RawEdviseStudentDataSchema,
        spark_session=spark_session,
    )
    course_result = execute_transformation_map(
        transformation_map=course_map,
        manifest=course_manifest,
        dataframes=dataframes,
        schema=RawEdviseCourseDataSchema,
        spark_session=spark_session,
    )

    # Step 2d — Pandera validation (report only, does not block)
    LOGGER.info("[onboard/gate_2] Step 2d — Pandera validation")
    _run_pandera_validation(cohort_result, course_result)

    # Write output data
    _write_output_data(paths.output_data, cohort_result, course_result)
    LOGGER.info("[onboard/gate_2] Promoting artifacts to active/")
    promote_genai_mapping_to_active(paths)
    LOGGER.info("[onboard/gate_2] Complete. Exiting.")
    _pipeline_job_state.after_sma_onboard_gate_2_success(
        catalog, institution_id, onboard_run_id
    )


# ---------------------------------------------------------------------------
# Execute
# Load approved artifacts -> execute transformation map -> Pandera -> write outputs
# ---------------------------------------------------------------------------

def run_execute(
    institution_id: str,
    paths: SMAPaths,
    spark_session,
):
    from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
        FieldMappingManifest,
        MappingManifestEnvelope,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
        TransformationMap,
    )
    from edvise.genai.mapping.schema_mapping_agent.execution.field_executor import (
        execute_transformation_map,
    )
    from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
    from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema

    LOGGER.info("[execute] Loading approved artifacts from active/ for %s", institution_id)

    # Validate active artifacts exist
    for p in (paths.active_manifest_map, paths.active_transformation_map, paths.active_enriched_schema_contract):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing active artifact: {p}. "
                "Has this institution been onboarded and activated?"
            )

    # Load approved artifacts
    manifest_data = json.loads(paths.active_manifest_map.read_text())
    transformation_data = json.loads(paths.active_transformation_map.read_text())
    enriched_contract = _load_enriched_contract(paths.active_enriched_schema_contract)

    # Load cleaned dataframes — written by edvise_ia execute mode in this run
    dataframes = _load_cleaned_dataframes(paths.ia_cleaned_datasets, enriched_contract)

    # Execute transformation maps
    LOGGER.info("[execute] Executing transformation maps")
    institution_id_from_tm = transformation_data.get("institution_id", institution_id)

    cohort_map_data = {
        **transformation_data["transformation_maps"]["cohort"],
        "institution_id": institution_id_from_tm,
    }
    course_map_data = {
        **transformation_data["transformation_maps"]["course"],
        "institution_id": institution_id_from_tm,
    }

    envelope = MappingManifestEnvelope.model_validate(manifest_data)
    cohort_manifest = FieldMappingManifest.model_validate(manifest_data["manifests"]["cohort"])
    course_manifest = FieldMappingManifest.model_validate(manifest_data["manifests"]["course"])
    cohort_map = TransformationMap.model_validate(cohort_map_data)
    course_map = TransformationMap.model_validate(course_map_data)

    cohort_result = execute_transformation_map(
        transformation_map=cohort_map,
        manifest=cohort_manifest,
        dataframes=dataframes,
        schema=RawEdviseStudentDataSchema,
        spark_session=spark_session,
    )
    course_result = execute_transformation_map(
        transformation_map=course_map,
        manifest=course_manifest,
        dataframes=dataframes,
        schema=RawEdviseCourseDataSchema,
        spark_session=spark_session,
    )

    # Pandera validation (report only)
    LOGGER.info("[execute] Pandera validation")
    _run_pandera_validation(cohort_result, course_result)

    # Write output data
    _write_output_data(paths.output_data, cohort_result, course_result)
    LOGGER.info("[execute] Complete. Exiting.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    institution_id: str,
    catalog: str,
    mode: str,
    onboard_run_id: str | None = None,
    execute_run_id: str | None = None,
    artifacts_onboard_run_id: str | None = None,
    resume_from: str = "start",
    reference_id: str = "",
    db_run_id: str | None = None,
):
    if mode == "onboard":
        if not (onboard_run_id or "").strip():
            raise ValueError("onboard_run_id is required when mode='onboard'")
        paths = resolve_run_paths(
            institution_id,
            catalog,
            mode="onboard",
            onboard_run_id=onboard_run_id,
        )
        _log_run = onboard_run_id
    elif mode == "execute":
        if not (execute_run_id or "").strip():
            raise ValueError("execute_run_id is required when mode='execute'")
        paths = resolve_run_paths(
            institution_id,
            catalog,
            mode="execute",
            execute_run_id=execute_run_id,
        )
        _log_run = execute_run_id
    else:
        raise ValueError(f"Invalid mode={mode!r}. Must be 'onboard' or 'execute'.")

    init_file_logging_at_path(
        paths.run_root / "sma_pipeline.log",
        logger_name="edvise_sma",
        append=True,
    )
    LOGGER.info(
        "edvise_sma | institution=%s | run=%s | mode=%s | resume_from=%s | artifacts_onboard=%s",
        institution_id,
        _log_run,
        mode,
        resume_from,
        artifacts_onboard_run_id or "",
    )

    # Spark session (optional — graceful degradation outside Databricks runtime)
    try:
        from databricks.connect import DatabricksSession
        spark_session = DatabricksSession.builder.getOrCreate()
    except Exception:
        spark_session = None
        LOGGER.warning("No Databricks Spark session available.")

    if mode == "execute":
        run_execute(institution_id, paths, spark_session)
        try:
            _pipeline_state.update_execute_pipeline_run_status(
                catalog,
                institution_id,
                str(execute_run_id).strip(),
                "complete",
                db_run_id=db_run_id,
            )
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(
                "Could not mark pipeline_runs complete after SMA execute: catalog=%s execute_run_id=%s (%s)",
                catalog,
                execute_run_id,
                e,
            )

    elif mode == "onboard":
        if resume_from not in ("start", "gate_2"):
            raise ValueError(
                f"Invalid resume_from={resume_from!r} for mode='onboard'. Must be 'start' or 'gate_2'."
            )
        if not reference_id:
            raise ValueError("--reference_id is required for onboard mode.")

        _pipeline_job_state.on_sma_onboard_begin(
            catalog, onboard_run_id, resume_from=resume_from
        )

        client = _build_openai_client(catalog)

        try:
            if resume_from == "start":
                run_onboard_start(
                    institution_id=institution_id,
                    reference_id=reference_id,
                    catalog=catalog,
                    paths=paths,
                    client=client,
                    spark_session=spark_session,
                    onboard_run_id=onboard_run_id,
                )
            elif resume_from == "gate_2":
                run_onboard_gate_2(
                    institution_id=institution_id,
                    reference_id=reference_id,
                    catalog=catalog,
                    paths=paths,
                    client=client,
                    spark_session=spark_session,
                    onboard_run_id=onboard_run_id,
                    db_run_id=db_run_id,
                )
        except HITLTimeoutError:
            raise
        except Exception:
            _pipeline_job_state.mark_pipeline_failed(
                catalog, institution_id, onboard_run_id
            )
            raise

    else:
        raise ValueError(f"Invalid mode={mode!r}. Must be 'onboard' or 'execute'.")


if __name__ == "__main__":
    from edvise.genai.mapping.state import pipeline_state

    parser = argparse.ArgumentParser(description="SchemaMappingAgent pipeline job")
    parser.add_argument("--institution_id", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--mode", required=True, choices=["onboard", "execute"])
    parser.add_argument("--resume_from", default="start", choices=["start", "gate_2"])
    parser.add_argument("--reference_id", default="")
    parser.add_argument(
        "--db_run_id",
        default="",
        help="Databricks job run id (orchestration id) stored on pipeline_runs.db_run_id; empty omits.",
    )
    args = parser.parse_args()

    try:
        from pyspark.sql import SparkSession

        _spark_sess = SparkSession.getActiveSession()
        _db_from_spark = (
            _spark_sess.conf.get("spark.databricks.job.runId", None)
            if _spark_sess is not None
            else None
        )
    except Exception:
        _db_from_spark = None

    _db_run_id = (args.db_run_id or "").strip() or (
        (str(_db_from_spark).strip()) if _db_from_spark else ""
    ).strip() or None

    _execute_run_id: str | None = None
    _artifacts_onboard: str | None = None
    _onboard_run_id: str | None = None

    if args.mode == "execute":
        _boot = pipeline_state.bootstrap_execute_run(
            args.catalog,
            args.institution_id,
            db_run_id=_db_run_id,
        )
        _execute_run_id = _boot.execute_run_id
        _artifacts_onboard = _boot.artifacts_onboard_run_id
    else:
        _onboard_run_id = pipeline_state.bootstrap_resolved_onboard_run_id(
            args.catalog,
            args.institution_id,
            None,
        )

    try:
        run(
            institution_id=args.institution_id,
            catalog=args.catalog,
            mode=args.mode,
            onboard_run_id=_onboard_run_id,
            execute_run_id=_execute_run_id,
            artifacts_onboard_run_id=_artifacts_onboard,
            resume_from=args.resume_from,
            reference_id=args.reference_id,
            db_run_id=_db_run_id,
        )
    except BaseException:
        if args.mode == "execute" and _execute_run_id:
            pipeline_state.mark_execute_pipeline_run_status(
                args.catalog,
                args.institution_id,
                _execute_run_id,
                "failed",
            )
        raise