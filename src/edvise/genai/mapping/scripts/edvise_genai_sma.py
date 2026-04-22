"""
edvise_sma.py — SchemaMappingAgent pipeline job entry point.

Usage (Databricks job parameters):
    --institution_id    ucf
    --pipeline_run_id   ucf_20260420_001
    --catalog           dev_sst_02
    --mode              onboard | execute
    --resume_from       start | gate_2  (onboard only)
"""

import os
import sys
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

# Layout: <git_root>/src/edvise/genai/mapping/scripts/<this_file>
# `import edvise` needs <git_root>/src on sys.path (package is <git_root>/src/edvise/).
_script_dir = os.path.dirname(os.path.abspath(__file__))
_src_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))
if os.path.isdir(_src_root) and _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from edvise.shared.logger import init_file_logging_at_path

LOGGER = logging.getLogger("edvise_sma")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SILVER_VOLUME_BASE = "/Volumes/{catalog}/{institution_id}_silver/silver_volume"


@dataclass
class SMAPaths:
    # Run folder (working artifacts, keyed by pipeline_run_id)
    run_root: Path
    mapping_manifest: Path
    mapping_validation_manifest: Path
    cohort_hitl_manifest: Path
    course_hitl_manifest: Path
    transformation_map: Path
    transform_hooks: Path           # optional, placeholder
    run_log: Path

    # IA outputs this job reads from (same pipeline_run_id)
    ia_enriched_schema_contract: Path
    ia_cleaned_datasets: Path       # directory

    # Active folder (promoted artifacts, what execute mode reads from)
    active_root: Path
    active_mapping_manifest: Path
    active_transformation_map: Path
    active_transform_hooks: Path
    active_enriched_schema_contract: Path
    active_cleaned_datasets: Path   # directory

    # Output data (written after execution)
    output_data: Path               # directory, one .parquet per entity


def resolve_run_paths(
    institution_id: str,
    pipeline_run_id: str,
    catalog: str,
) -> SMAPaths:
    silver = Path(
        SILVER_VOLUME_BASE.format(catalog=catalog, institution_id=institution_id)
    )
    genai = silver / "genai_mapping"
    run_root = genai / "runs" / pipeline_run_id / "schema_mapping_agent"
    ia_run_root = genai / "runs" / pipeline_run_id / "identity_agent"
    active_root = genai / "active" / "schema_mapping_agent"

    return SMAPaths(
        run_root=run_root,
        mapping_manifest=run_root / "mapping_manifest.json",
        mapping_validation_manifest=run_root / "mapping_validation_manifest.json",
        cohort_hitl_manifest=run_root / "cohort_hitl_manifest.json",
        course_hitl_manifest=run_root / "course_hitl_manifest.json",
        transformation_map=run_root / "transformation_map.json",
        transform_hooks=run_root / "transform_hooks.py",
        run_log=run_root / "run_log.json",
        # IA outputs — same run_id, identity_agent folder
        ia_enriched_schema_contract=ia_run_root / "enriched_schema_contract.json",
        ia_cleaned_datasets=ia_run_root / "cleaned_datasets",
        # Active folder
        active_root=active_root,
        active_mapping_manifest=active_root / "mapping_manifest.json",
        active_transformation_map=active_root / "transformation_map.json",
        active_transform_hooks=active_root / "transform_hooks.py",
        active_enriched_schema_contract=active_root / "enriched_schema_contract.json",
        active_cleaned_datasets=active_root / "cleaned_datasets",
        # Output data
        output_data=run_root / "data",
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
    import os
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["DATABRICKS_TOKEN"],
        base_url=os.environ.get(
            "AI_GATEWAY_BASE_URL",
            "https://4437281602191762.ai-gateway.gcp.databricks.com/mlflow/v1",
        ),
    )


def _run_once(model_id: str, prompt: str, client) -> dict:
    from edvise.genai.mapping.schema_mapping_agent.manifest.eval import run_once
    return run_once(model_id, prompt, client)


# ---------------------------------------------------------------------------
# Onboard — resume_from="start"
# Load IA outputs -> 2a LLM -> structural validation -> refinement LLM -> write HITL -> exit
# ---------------------------------------------------------------------------

def run_onboard_start(
    institution_id: str,
    institution_name: str,
    reference_id: str,
    reference_name: str,
    model_id: str,
    paths: SMAPaths,
    client,
    spark_session,
):
    from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import (
        build_step2a_batched_prompt,
        load_json,
        run_sma_refinement,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.eval import (
        run_once,
        validate_envelope_dict,
    )
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

    # Load reference institution few-shot manifest
    # Reference manifests live in the silver volume under a shared examples folder
    silver = paths.run_root.parents[3]  # .../genai_mapping/runs/{run_id}/sma -> .../silver/genai_mapping
    ref_manifest_path = (
        silver / "genai_mapping" / "schema_mapping_agent_examples" / reference_id / "final_hitl"
        / f"{reference_id}_mapping_manifest.json"
    )
    if not ref_manifest_path.exists():
        raise FileNotFoundError(f"Reference HITL manifest not found: {ref_manifest_path}")
    reference_manifest = load_json(str(ref_manifest_path))

    # Step 2a — mapping manifest LLM
    LOGGER.info("[onboard/start] Step 2a — mapping manifest LLM")
    prompt_2a = build_step2a_batched_prompt(
        institution_id=institution_id,
        institution_name=institution_name,
        output_path=str(paths.mapping_manifest),
        institution_schema_contract=enriched_contract,
        reference_manifests=[reference_manifest],
        reference_institution_names=[reference_name],
        cohort_schema_class=RawEdviseStudentDataSchema,
        course_schema_class=RawEdviseCourseDataSchema,
    )
    result_2a = run_once(model_id, prompt_2a, client)
    if not result_2a["success"]:
        raise RuntimeError(result_2a.get("error") or "Step 2a LLM failed")

    manifest_2a = json.loads(result_2a["response"])
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
        r = run_once(model_id, combined, client)
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
    paths.mapping_manifest.write_text(json.dumps(manifest_2a, indent=2))
    LOGGER.info("[onboard/start] Wrote mapping manifest -> %s", paths.mapping_manifest)

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


# ---------------------------------------------------------------------------
# Onboard — resume_from="gate_2"
# Resolve HITL -> gate check -> 2b LLM -> execute -> Pandera -> write outputs -> exit
# ---------------------------------------------------------------------------

def run_onboard_gate_2(
    institution_id: str,
    institution_name: str,
    reference_id: str,
    reference_name: str,
    model_id: str,
    paths: SMAPaths,
    client,
    spark_session,
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
    from edvise.genai.mapping.schema_mapping_agent.manifest.eval import run_once
    from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import load_json
    from edvise.genai.mapping.schema_mapping_agent.execution.field_executor import (
        execute_transformation_map,
    )
    from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
    from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema

    LOGGER.info("[onboard/gate_2] Resolving HITL for %s", institution_id)

    # Resolve HITL into mapping manifest
    for hitl_path in (paths.cohort_hitl_manifest, paths.course_hitl_manifest):
        resolve_sma_items(
            hitl_path,
            paths.mapping_manifest,
            resolved_by="pipeline",
            run_log_path=paths.run_log,
        )

    # Reload manifest after resolution
    manifest_2a = json.loads(paths.mapping_manifest.read_text())
    envelope_2a = MappingManifestEnvelope.model_validate(manifest_2a)

    # Gate check — raises HITLBlockingError if any items still pending
    LOGGER.info("[onboard/gate_2] HITL gate check")
    for hitl_path in (paths.cohort_hitl_manifest, paths.course_hitl_manifest):
        check_sma_hitl_gate(hitl_path)

    # Load reference transformation map for 2b few-shot
    silver = paths.run_root.parents[3]
    ref_tm_path = (
        silver / "genai_mapping" / "schema_mapping_agent_examples" / reference_id / "final_hitl"
        / f"{reference_id}_transformation_map.json"
    )
    if not ref_tm_path.exists():
        raise FileNotFoundError(f"Reference transformation map not found: {ref_tm_path}")
    reference_tm = load_json(str(ref_tm_path))

    enriched_contract = _load_enriched_contract(paths.ia_enriched_schema_contract)

    # Step 2b — transformation map LLM
    LOGGER.info("[onboard/gate_2] Step 2b — transformation map LLM")
    prompt_2b = build_step2b_prompt(
        institution_id=institution_id,
        institution_name=institution_name,
        output_path=str(paths.transformation_map),
        institution_mapping_manifest=manifest_2a,
        institution_schema_contract=enriched_contract,
        cohort_schema_class=RawEdviseStudentDataSchema,
        course_schema_class=RawEdviseCourseDataSchema,
        reference_transformation_maps=[reference_tm],
        reference_institution_names=[reference_name],
    )
    result_2b = run_once(model_id, prompt_2b, client)
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
    LOGGER.info("[onboard/gate_2] Complete. Exiting.")


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
    for p in (paths.active_mapping_manifest, paths.active_transformation_map, paths.active_enriched_schema_contract):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing active artifact: {p}. "
                "Has this institution been onboarded and activated?"
            )

    # Load approved artifacts
    manifest_data = json.loads(paths.active_mapping_manifest.read_text())
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
    pipeline_run_id: str,
    catalog: str,
    mode: str,
    resume_from: str = "start",
    institution_name: str = "",
    reference_id: str = "",
    reference_name: str = "",
    model_id: str = "claude-sonnet-test-genai-ai-data-cleaning",
):
    paths = resolve_run_paths(institution_id, pipeline_run_id, catalog)
    init_file_logging_at_path(
        paths.run_root / "sma_pipeline.log",
        logger_name="edvise_sma",
        append=True,
    )
    LOGGER.info(
        "edvise_sma | institution=%s | run=%s | mode=%s | resume_from=%s",
        institution_id, pipeline_run_id, mode, resume_from,
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

    elif mode == "onboard":
        if resume_from not in ("start", "gate_2"):
            raise ValueError(
                f"Invalid resume_from={resume_from!r} for mode='onboard'. Must be 'start' or 'gate_2'."
            )
        if not institution_name:
            raise ValueError("--institution_name is required for onboard mode.")
        if not reference_id or not reference_name:
            raise ValueError("--reference_id and --reference_name are required for onboard mode.")

        client = _build_openai_client(catalog)

        if resume_from == "start":
            run_onboard_start(
                institution_id=institution_id,
                institution_name=institution_name,
                reference_id=reference_id,
                reference_name=reference_name,
                model_id=model_id,
                paths=paths,
                client=client,
                spark_session=spark_session,
            )
        elif resume_from == "gate_2":
            run_onboard_gate_2(
                institution_id=institution_id,
                institution_name=institution_name,
                reference_id=reference_id,
                reference_name=reference_name,
                model_id=model_id,
                paths=paths,
                client=client,
                spark_session=spark_session,
            )

    else:
        raise ValueError(f"Invalid mode={mode!r}. Must be 'onboard' or 'execute'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SchemaMappingAgent pipeline job")
    parser.add_argument("--institution_id", required=True)
    parser.add_argument("--pipeline_run_id", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--mode", required=True, choices=["onboard", "execute"])
    parser.add_argument("--resume_from", default="start", choices=["start", "gate_2"])
    parser.add_argument("--institution_name", default="")
    parser.add_argument("--reference_id", default="")
    parser.add_argument("--reference_name", default="")
    parser.add_argument("--model_id", default="claude-sonnet-test-genai-ai-data-cleaning")
    args = parser.parse_args()

    run(
        institution_id=args.institution_id,
        pipeline_run_id=args.pipeline_run_id,
        catalog=args.catalog,
        mode=args.mode,
        resume_from=args.resume_from,
        institution_name=args.institution_name,
        reference_id=args.reference_id,
        reference_name=args.reference_name,
        model_id=args.model_id,
    )