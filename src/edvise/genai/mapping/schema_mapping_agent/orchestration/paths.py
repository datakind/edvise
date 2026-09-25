"""Run-path resolution for the SchemaMappingAgent pipeline entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edvise.configs import genai as genai_cfg
from edvise.genai.mapping.shared.silver_run_paths import sma_pipeline_input_root


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
    # Run-local few-shot snapshot (selected pin copied once at SMA start)
    few_shot_root: Path

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
        few_shot_root=run_root / "few_shot",
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
