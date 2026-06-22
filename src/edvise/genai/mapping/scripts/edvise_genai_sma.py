"""
edvise_sma.py — SchemaMappingAgent pipeline job entry point.

Usage (Databricks job parameters):
    --institution_id    synthetic_edvise
    --catalog           dev_sst_02
    --mode              onboard | execute
    --resume_from       start | gate_2  (onboard only)
    --pipeline_version  Release / git tag for manifests and transformation maps (match edvise_ia job).
    --reference_id      required for onboard; few-shot from that school's
                        ``.../<ref_id>_silver/silver_volume/genai_mapping/active/`` (``manifest_map.json``,
                        ``transformation_map.json``, ``enriched_schema_contract.json``, and optional
                        ``sma_grain_resolution_cohort.json`` / ``sma_grain_resolution_course.json`` when promoted)
                        (same ``--catalog`` as the reference institution's volumes).
    --inputs_toml_path  Same resolution as edvise_ia (relative under bronze ``genai_mapping/``).

On Databricks, onboard mode best-effort updates ``{catalog}.genai_mapping`` pipeline state
(see :mod:`edvise.genai.mapping.state.job_state`); table setup and Spark are required.

After Step 2b, ``gate_2`` registers ``sma_gate_2_transformation_review`` when plans need human review
(``cohort_transformation_review.json`` / ``course_transformation_review.json`` — artifact types
``cohort_transformation_review`` / ``course_transformation_review``), merges resolutions, then
``sma_gate_2_hook_preview`` for ``HookSpec`` previews
(``cohort_transformation_hook_preview.json`` / ``course_transformation_hook_preview.json``) for plans
with ``hook_required: true`` (set in Step 2b or via transformation review option 3), then
materializes ``transform_hooks.py`` after UC approval. When manifest grain is stricter than cleaned
row count, ``sma_gate_2_grain`` gates ``cohort_sma_grain_hitl.json`` / ``course_sma_grain_hitl.json``
(see :mod:`edvise.genai.mapping.schema_mapping_agent.grain_resolution`).
"""

import os
import random
import sys
import time
import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, cast
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
from edvise.genai.mapping.shared.utilities import (
    disable_mlflow_side_effects_for_openai_gateway,
)

disable_mlflow_side_effects_for_openai_gateway()

from edvise.configs import genai as genai_cfg
from edvise.genai.mapping.shared.active_promotion import (
    promote_genai_mapping_to_active,
    update_genai_active_registry_execute,
)
from edvise.genai.mapping.shared.databricks_ai_gateway import resolve_gateway_model_id
from edvise.genai.mapping.schema_mapping_agent.grain_resolution import (
    execute_transformation_map_for_sma_execute_mode,
    reload_field_manifest_entity,
    run_onboard_gate_2_entity_with_grain_uc,
)
from edvise.genai.mapping.shared.silver_run_paths import sma_pipeline_input_root
from edvise.genai.mapping.state import job_state as _pipeline_job_state
from edvise.genai.mapping.state import pipeline_state as _pipeline_state
from edvise.genai.mapping.state.hitl_poller import (
    DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    HITLTimeoutError,
)
from edvise.shared.logger import (
    init_file_logging_at_path,
    resolve_genai_segment_log_path,
)
from edvise.utils.llm_utils import llm_complete_with_parse_retry

LOGGER = logging.getLogger("edvise_sma")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass
class SMAPaths:
    # SMA JSON / maps / HITL: ``runs/onboard/{onboard_run_id}/schema_mapping_agent/`` (or execute).
    run_root: Path
    manifest_map: Path
    mapping_validation_manifest: Path
    cohort_hitl_manifest: Path
    course_hitl_manifest: Path
    cohort_transformation_hook_hitl: Path
    course_transformation_hook_hitl: Path
    cohort_transformation_hook_preview: Path
    course_transformation_hook_preview: Path
    cohort_transformation_review: Path
    course_transformation_review: Path
    transformation_map: Path
    transform_hooks: Path  # optional, placeholder
    run_log: Path
    mapping_override_log: Path
    pandera_validation_errors: Path

    # IA outputs this job reads from (same execute or onboard run segment)
    ia_enriched_schema_contract: Path
    ia_identity_term_output: (
        Path  # identity_term_output.json (optional Step 2b context)
    )
    ia_cleaned_datasets: Path  # directory

    # Active folder (promoted artifacts, what execute mode reads from)
    active_root: Path
    active_manifest_map: Path
    active_transformation_map: Path
    active_transform_hooks: Path
    active_enriched_schema_contract: Path

    # Optional upstream cleaned inputs (volume layout)
    genai_data: Path

    # Materialized cohort/course parquet (sibling of ``schema_mapping_agent/`` under the run id)
    output_data: Path


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
    run_segment = genai / "runs" / segment[0] / segment[1]
    run_root = run_segment / "schema_mapping_agent"
    ia_run_root = run_segment / "identity_agent"
    active_root = genai / "active"

    return SMAPaths(
        run_root=run_root,
        manifest_map=run_root / "manifest_map.json",
        mapping_validation_manifest=run_root / "mapping_validation_manifest.json",
        cohort_hitl_manifest=run_root / "cohort_hitl_manifest.json",
        course_hitl_manifest=run_root / "course_hitl_manifest.json",
        cohort_transformation_hook_hitl=run_root
        / "cohort_transformation_hook_hitl.json",
        course_transformation_hook_hitl=run_root
        / "course_transformation_hook_hitl.json",
        cohort_transformation_hook_preview=run_root
        / "cohort_transformation_hook_preview.json",
        course_transformation_hook_preview=run_root
        / "course_transformation_hook_preview.json",
        cohort_transformation_review=run_root / "cohort_transformation_review.json",
        course_transformation_review=run_root / "course_transformation_review.json",
        transformation_map=run_root / "transformation_map.json",
        transform_hooks=run_root / "transform_hooks.py",
        run_log=run_root / "run_log.json",
        mapping_override_log=run_root / "mapping_override_log.json",
        pandera_validation_errors=run_root / "pandera_validation_errors.json",
        # IA outputs — same run segment under ``runs/onboard/...`` or ``runs/execute/...``
        ia_enriched_schema_contract=ia_run_root / "enriched_schema_contract.json",
        ia_identity_term_output=ia_run_root / "identity_term_output.json",
        ia_cleaned_datasets=ia_run_root / "cleaned_datasets",
        # Active folder (flat under genai_mapping)
        active_root=active_root,
        active_manifest_map=active_root / "manifest_map.json",
        active_transformation_map=active_root / "transformation_map.json",
        active_transform_hooks=active_root / "transform_hooks.py",
        active_enriched_schema_contract=active_root / "enriched_schema_contract.json",
        genai_data=genai / "data",
        output_data=sma_pipeline_input_root(genai, mode=segment[0], run_id=segment[1]),
    )


def resolve_reference_sma_active_paths(
    reference_id: str, *, catalog: str
) -> tuple[Path, Path]:
    """Few-shot reference: promoted SMA artifacts under the reference school's ``genai_mapping/active/``."""
    active = (
        Path(genai_cfg.silver_genai_mapping_root(reference_id, catalog=catalog))
        / "active"
    )
    return (
        active / "manifest_map.json",
        active / "transformation_map.json",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_enriched_contract(path: Path) -> dict[Any, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Enriched schema contract not found: {path}. "
            "Run edvise_ia onboard/gate_1 first."
        )
    return cast(dict[Any, Any], json.loads(path.read_text()))


def _load_institution_term_config_optional(
    path: Path, *, expected_institution_id: str
) -> dict | None:
    """
    Load IdentityAgent ``identity_term_output.json`` as a dict for Step 2b optional term context.

    The caller should pass the pipeline ``institution_id`` so we reject a file whose embedded
    ``institution_id`` does not match (path alone is already scoped to that school's volume and
    run segment — see :func:`resolve_run_paths`).

    Returns None when the file is missing, invalid, or mismatched — Step 2b still runs without it.
    """
    if not path.is_file():
        LOGGER.info(
            "[step2b] No identity term output at %s — skipping institution_term_config",
            path,
        )
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        LOGGER.warning(
            "[step2b] Could not read JSON from %s (%s) — skipping institution_term_config",
            path,
            e,
        )
        return None
    from pydantic import ValidationError

    from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
        InstitutionTermContract,
    )

    try:
        inst = InstitutionTermContract.model_validate(raw)
    except ValidationError as e:
        LOGGER.warning(
            "[step2b] Invalid InstitutionTermContract in %s (%s) — skipping institution_term_config",
            path,
            e,
        )
        return None
    if inst.institution_id != expected_institution_id:
        LOGGER.warning(
            "[step2b] identity_term_output institution_id %r != pipeline institution_id %r "
            "(%s) — skipping institution_term_config",
            inst.institution_id,
            expected_institution_id,
            path,
        )
        return None
    return inst.model_dump(mode="json")


def _load_cleaned_dataframes(
    cleaned_datasets_dir: Path, enriched_contract: dict
) -> dict:
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


def _write_output_data(
    output_data_dir: Path, cohort_result: Any, course_result: Any
) -> None:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_data_dir / "cohort.parquet"
    course_path = output_data_dir / "course.parquet"
    cohort_result.df.to_parquet(cohort_path, index=False)
    course_result.df.to_parquet(course_path, index=False)
    LOGGER.info(
        "Wrote cohort output -> %s (shape=%s)", cohort_path, cohort_result.df.shape
    )
    LOGGER.info(
        "Wrote course output -> %s (shape=%s)", course_path, course_result.df.shape
    )


def _run_pandera_validation(
    cohort_result: Any,
    course_result: Any,
    *,
    report_path: Path,
) -> None:
    from edvise.genai.mapping.schema_mapping_agent.execution.pandera_validation_report import (
        write_pandera_validation_errors,
    )

    write_pandera_validation_errors(
        report_path,
        cohort_result.df,
        course_result.df,
        logger=LOGGER,
    )


def _build_openai_client(catalog: str) -> Any:
    """Build OpenAI-compatible client for Databricks AI Gateway."""
    from openai import OpenAI

    from edvise.genai.mapping.shared.databricks_ai_gateway import (
        disable_mlflow_tracing_for_openai_gateway_client,
        require_databricks_token,
        resolve_ai_gateway_base_url,
    )

    disable_mlflow_tracing_for_openai_gateway_client()
    return OpenAI(
        api_key=require_databricks_token(),
        base_url=resolve_ai_gateway_base_url(),
    )


def _run_once(model_id: str, prompt: str, client: Any) -> dict[str, Any]:
    """
    Call :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.eval.run_once` with
    retries for transient gateway / transport failures (same policy as IA ``llm_complete``).
    """
    from edvise.genai.mapping.shared.databricks_ai_gateway import (
        gateway_run_once_error_text_is_retryable,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.eval import run_once

    max_attempts = 5
    initial_backoff_s = 2.0
    max_backoff_s = 60.0
    last: dict[str, Any] = {}
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


def _sma_llm_complete_run_once(client):
    """``(system, user) -> text`` for :func:`llm_complete_with_parse_retry` (combines like refinement)."""
    model_id = resolve_gateway_model_id()

    def llm_complete(system: str, user: str) -> str:
        s = (system or "").strip()
        u = (user or "").strip()
        if s and u:
            combined = f"{s}\n\n---\n\n{u}"
        elif u:
            combined = u
        elif s:
            combined = s
        else:
            raise RuntimeError("SMA LLM call has empty system and user prompts")
        result = _run_once(model_id, combined, client)
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "SMA LLM call failed")
        resp = result.get("response")
        if not isinstance(resp, str) or not resp.strip():
            raise RuntimeError("SMA LLM returned empty response")
        return resp

    return llm_complete


# ---------------------------------------------------------------------------
# Onboard — resume_from="start"
# Load IA outputs -> 2a LLM -> structural validation -> refinement LLM -> write HITL -> exit
# ---------------------------------------------------------------------------


def run_onboard_start(
    institution_id: str,
    reference_id: str,
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

    llm_sma = _sma_llm_complete_run_once(client)

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


# ---------------------------------------------------------------------------
# Onboard — resume_from="gate_2"
# Resolve manifest HITL -> 2b LLM -> transformation review HITL (UC) -> hook preview -> hook_required -> execute
# ---------------------------------------------------------------------------


def run_onboard_gate_2(
    institution_id: str,
    reference_id: str,
    catalog: str,
    paths: SMAPaths,
    client: Any,
    spark_session: Any,
    *,
    onboard_run_id: str,
    pipeline_version: str,
    db_run_id: str | None = None,
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
    LOGGER.info(
        "[onboard/gate_2] Reference transformation map (active): %s", ref_tm_path
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

    llm_sma = _sma_llm_complete_run_once(client)

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
    _pipeline_job_state.wait_for_sma_gate_2_transformation_review_hitl(
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
    _pipeline_job_state.after_sma_gate_2_transformation_review_approved(
        catalog, institution_id, onboard_run_id
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

    _sma_gateway_model_id = resolve_gateway_model_id()

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
        load_hook_specs_from_sma_preview_path,
        write_sma_transform_hook_preview_json,
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
    _pipeline_job_state.wait_for_sma_gate_2_hook_preview_hitl(
        catalog,
        onboard_run_id,
        institution_id=institution_id,
        poll_interval_seconds=DEFAULT_HITL_POLL_INTERVAL_SECONDS,
        timeout_seconds=DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    )
    _pipeline_job_state.after_sma_gate_2_hook_preview_approved(
        catalog, institution_id, onboard_run_id
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
    spark_session: Any,
    *,
    execute_run_id: str,
) -> None:
    from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
        FieldMappingManifest,
        MappingManifestEnvelope,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.dedupe_plans import (
        dedupe_transformation_plans_in_wrapper,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
        TransformationMap,
    )
    from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
    from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema

    LOGGER.info(
        "[execute] Loading approved artifacts from active/ for %s", institution_id
    )

    # Validate active artifacts exist
    for p in (
        paths.active_manifest_map,
        paths.active_transformation_map,
        paths.active_enriched_schema_contract,
    ):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing active artifact: {p}. "
                "Has this institution been onboarded and activated?"
            )

    # Load approved artifacts
    manifest_data = json.loads(paths.active_manifest_map.read_text())
    transformation_data = json.loads(paths.active_transformation_map.read_text())
    dedupe_transformation_plans_in_wrapper(transformation_data, log=LOGGER)
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
    cohort_manifest = FieldMappingManifest.model_validate(
        manifest_data["manifests"]["cohort"]
    )
    course_manifest = FieldMappingManifest.model_validate(
        manifest_data["manifests"]["course"]
    )
    cohort_map = TransformationMap.model_validate(cohort_map_data)
    course_map = TransformationMap.model_validate(course_map_data)

    cohort_result = execute_transformation_map_for_sma_execute_mode(
        transformation_map=cohort_map,
        manifest=cohort_manifest,
        dataframes=dataframes,
        schema=RawEdviseStudentDataSchema,
        spark_session=spark_session,
        institution_id=institution_id_from_tm,
        enriched_contract=enriched_contract,
        manifest_map_path=paths.active_manifest_map,
        grain_hitl_path=paths.run_root / "cohort_sma_grain_hitl.json",
        active_grain_resolution_root=paths.active_root,
    )
    course_result = execute_transformation_map_for_sma_execute_mode(
        transformation_map=course_map,
        manifest=course_manifest,
        dataframes=dataframes,
        schema=RawEdviseCourseDataSchema,
        spark_session=spark_session,
        institution_id=institution_id_from_tm,
        enriched_contract=enriched_contract,
        manifest_map_path=paths.active_manifest_map,
        grain_hitl_path=paths.run_root / "course_sma_grain_hitl.json",
        active_grain_resolution_root=paths.active_root,
    )

    # Pandera validation (report only)
    LOGGER.info("[execute] Pandera validation")
    _run_pandera_validation(
        cohort_result, course_result, report_path=paths.pandera_validation_errors
    )

    # Write output data
    _write_output_data(paths.output_data, cohort_result, course_result)
    LOGGER.info("[execute] Updating genai_active_registry execute pointer")
    update_genai_active_registry_execute(
        paths.active_root,
        execute_run_id=execute_run_id,
    )
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
    inputs_toml_path: str | None = None,
    db_run_id: str | None = None,
    pipeline_version: str | None = None,
    bronze_batch_dir: str | None = None,
) -> None:
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

    _segment_log = resolve_genai_segment_log_path(
        paths.run_root,
        mode=mode,
        resume_from=resume_from,
    )
    init_file_logging_at_path(
        _segment_log,
        logger_name="edvise_sma",
        append=False,
    )
    LOGGER.info(
        "edvise_sma | institution=%s | run=%s | mode=%s | resume_from=%s | artifacts_onboard=%s | log=%s",
        institution_id,
        _log_run,
        mode,
        resume_from,
        artifacts_onboard_run_id or "",
        _segment_log,
    )

    from edvise import configs, dataio
    from edvise.configs.genai import resolve_genai_data_path

    institution_inputs_toml = Path(
        configs.genai.resolve_genai_inputs_toml_path(
            institution_id,
            catalog=catalog,
            inputs_toml_path=(inputs_toml_path or "").strip() or None,
        )
    )
    if not institution_inputs_toml.is_file():
        default_hint = configs.genai.resolve_genai_inputs_toml_path(
            institution_id, catalog=catalog, inputs_toml_path=None
        )
        raise FileNotFoundError(
            f"IdentityAgent inputs.toml not found: {institution_inputs_toml}. "
            "Pass --inputs_toml_path relative to genai_mapping on bronze (e.g. inputs.toml or inputs/inputs.toml), "
            "a full /Volumes/... path, or place the file at "
            f"{default_hint!r}."
        )
    LOGGER.info("Loading SMA school config from %s", institution_inputs_toml)
    _ia = dataio.read.read_config(
        str(institution_inputs_toml),
        schema=configs.genai.IdentityAgentInputsConfig,
    )
    _pv_job = (pipeline_version or "").strip() or None
    school_config = _ia.to_school_mapping_config(
        uc_catalog=catalog,
        pipeline_mode=cast(Literal["onboard", "execute"], mode),
        pipeline_version=_pv_job,
    )
    LOGGER.info("pipeline_version=%s", school_config.pipeline_version)

    if mode == "execute":
        from edvise.genai.mapping.shared.batch_input_paths import (
            apply_bronze_batch_dir_overrides,
        )

        school_config = apply_bronze_batch_dir_overrides(
            school_config,
            bronze_batch_dir=bronze_batch_dir,
        )

    input_file_paths: dict[str, list[str]] = {
        ds_name: [
            str(resolve_genai_data_path(school_config.bronze_volumes_path, f))
            for f in dc.files
        ]
        for ds_name, dc in school_config.datasets.items()
    }
    input_file_paths_json = json.dumps(input_file_paths)

    # Spark session (optional — graceful degradation outside Databricks runtime)
    try:
        from databricks.connect import DatabricksSession

        spark_session = DatabricksSession.builder.getOrCreate()
    except Exception:
        spark_session = None
        LOGGER.warning("No Databricks Spark session available.")

    if mode == "execute":
        try:
            _pipeline_state.update_execute_pipeline_run_input_file_paths(
                catalog,
                institution_id,
                str(execute_run_id).strip(),
                input_file_paths_json,
            )
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(
                "Could not stamp input_file_paths on execute pipeline_runs: catalog=%s execute_run_id=%s (%s)",
                catalog,
                execute_run_id,
                e,
            )

        run_execute(
            institution_id,
            paths,
            spark_session,
            execute_run_id=str(execute_run_id).strip(),
        )
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

        onboard_run_id_s = cast(str, onboard_run_id)
        _pipeline_job_state.on_sma_onboard_begin(
            catalog,
            onboard_run_id_s,
            resume_from=resume_from,
            institution_id=institution_id,
            input_file_paths_json=input_file_paths_json,
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
                    onboard_run_id=onboard_run_id_s,
                    pipeline_version=school_config.pipeline_version,
                )
            elif resume_from == "gate_2":
                run_onboard_gate_2(
                    institution_id=institution_id,
                    reference_id=reference_id,
                    catalog=catalog,
                    paths=paths,
                    client=client,
                    spark_session=spark_session,
                    onboard_run_id=onboard_run_id_s,
                    pipeline_version=school_config.pipeline_version,
                    db_run_id=db_run_id,
                )
        except HITLTimeoutError:
            raise
        except Exception:
            _pipeline_job_state.mark_pipeline_failed(
                catalog, institution_id, onboard_run_id_s
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
        "--pipeline_version",
        default="",
        help=(
            "Edvise/git release id for manifest and transformation artifacts (align with edvise_ia). "
            "Empty falls back to GENAI_GIT_TAG / installed edvise version."
        ),
    )
    parser.add_argument(
        "--inputs_toml_path",
        default="",
        help=(
            "Relative to …/bronze_volume/genai_mapping/ on the institution bronze volume, "
            "or an absolute /Volumes/... path. Empty uses inputs.toml (requires --catalog)."
        ),
    )
    parser.add_argument(
        "--db_run_id",
        default="",
        help="Databricks job run id (orchestration id) stored on pipeline_runs.db_run_id; empty omits.",
    )
    parser.add_argument(
        "--bronze_batch_dir",
        default="",
        help=(
            "ES inference only: batch landing dir from batch_gcs_ingest "
            "(gcs_uploads/{batch_id}/). Resolves inputs.toml filenames inside it."
        ),
    )
    parser.add_argument(
        "--new_onboard_run",
        action="store_true",
        help=(
            "Onboard mode only: mint a fresh opaque onboard_run_id (ignore db_run_id); rare escape "
            "hatch — prefer starting a new job for a new folder; repairs reuse the same db_run_id."
        ),
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

    _db_run_id = (
        (args.db_run_id or "").strip()
        or ((str(_db_from_spark).strip()) if _db_from_spark else "").strip()
        or None
    )

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
            db_run_id=_db_run_id,
            force_new_onboard_run=bool(args.new_onboard_run),
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
            inputs_toml_path=(args.inputs_toml_path or "").strip() or None,
            db_run_id=_db_run_id,
            pipeline_version=(args.pipeline_version or "").strip() or None,
            bronze_batch_dir=(args.bronze_batch_dir or "").strip() or None,
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
