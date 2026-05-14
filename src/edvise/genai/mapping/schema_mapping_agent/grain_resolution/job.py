"""
SMA grain job wiring: call :mod:`edvise.genai.mapping.schema_mapping_agent.execution.field_executor`
with grain paths and onboard UC polling.

Domain logic (HITL construction, LLM dedup) lives in sibling modules
:mod:`edvise.genai.mapping.schema_mapping_agent.grain_resolution.reconciliation_gate` and
:mod:`edvise.genai.mapping.schema_mapping_agent.grain_resolution.prompt`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Protocol, cast

LOGGER = logging.getLogger(__name__)


class SmaGrainHitlPending(Exception):
    """Grain HITL file written; onboard gate_2 registers UC, polls, then ``resolve_items``."""

    def __init__(
        self, *, grain_hitl_path: Path, manifest_map_path: Path
    ) -> None:
        self.grain_hitl_path = grain_hitl_path
        self.manifest_map_path = manifest_map_path
        super().__init__(
            f"SMA grain HITL pending UC + resolver: {grain_hitl_path}"
        )


class SmaSchemaMappingRunPaths(Protocol):
    """Paths for one SMA run under ``schema_mapping_agent/`` (matches :class:`SMAPaths` in the job script)."""

    @property
    def run_root(self) -> Path: ...

    @property
    def manifest_map(self) -> Path: ...

    @property
    def run_log(self) -> Path: ...


def ia_post_clean_primary_key_for_dataset(
    enriched_contract: dict[str, Any], dataset_name: str
) -> list[str] | None:
    """IdentityAgent grain (source space) for ``execute_transformation_map(..., ia_source_keys=...)``."""
    ds = (enriched_contract.get("datasets") or {}).get(dataset_name)
    if not isinstance(ds, dict):
        return None
    grain = ds.get("grain_contract") or {}
    if not isinstance(grain, dict):
        return None
    pk = grain.get("post_clean_primary_key")
    if not isinstance(pk, list) or not pk:
        return None
    return [str(x) for x in pk]


def execute_transformation_map_for_sma_run(
    *,
    transformation_map: Any,
    manifest: Any,
    dataframes: dict[str, Any],
    schema: Any,
    spark_session: Any,
    institution_id: str,
    enriched_contract: dict[str, Any],
    manifest_map_path: Path,
    grain_hitl_path: Path,
) -> Any:
    """Run SMA execution with grain paths (HITL, IA keys, optional resolution sidecar)."""
    from edvise.genai.mapping.schema_mapping_agent.execution.field_executor import (
        GrainReconciliationRequired,
        execute_transformation_map,
    )
    from edvise.genai.mapping.schema_mapping_agent.grain_resolution.reconciliation_gate import (
        run_grain_reconciliation_gate,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.validation import (
        infer_manifest_base_table,
    )

    base_table = infer_manifest_base_table(manifest)
    ia_keys = ia_post_clean_primary_key_for_dataset(enriched_contract, base_table)
    raw_et = transformation_map.entity_type
    et_value = raw_et.value if hasattr(raw_et, "value") else str(raw_et)
    if et_value not in ("cohort", "course"):
        raise ValueError(
            f"Unexpected entity_type={et_value!r} for SMA grain job (expected cohort|course)"
        )
    entity_lit = cast(Literal["cohort", "course"], et_value)
    sidecar = grain_hitl_path.parent / f"sma_grain_resolution_{et_value}.json"
    resolution_path = sidecar if sidecar.is_file() else None

    try:
        return execute_transformation_map(
            transformation_map=transformation_map,
            manifest=manifest,
            dataframes=dataframes,
            schema=schema,
            spark_session=spark_session,
            institution_id=institution_id,
            hitl_output_path=grain_hitl_path,
            ia_source_keys=ia_keys,
            sma_grain_resolution_path=resolution_path,
            sma_manifest_path=manifest_map_path,
        )
    except GrainReconciliationRequired as exc:
        run_grain_reconciliation_gate(
            df=exc.base_df,
            institution_id=exc.institution_id,
            dataset=exc.dataset,
            entity_type=entity_lit,
            manifest_source_keys=exc.manifest_source_keys,
            mapped_source_columns=exc.mapped_source_columns,
            ia_source_keys=exc.ia_source_keys,
            hitl_output_path=exc.hitl_output_path,
            sma_manifest_path=exc.sma_manifest_path,
        )
        raise SmaGrainHitlPending(
            grain_hitl_path=Path(exc.hitl_output_path),
            manifest_map_path=manifest_map_path,
        ) from exc


def execute_transformation_map_for_sma_execute_mode(**kwargs: Any) -> Any:
    """``mode=execute`` has no UC grain gate; surface a clear error with the HITL path."""
    try:
        return execute_transformation_map_for_sma_run(**kwargs)
    except SmaGrainHitlPending as e:
        name_lc = e.grain_hitl_path.name.lower()
        et = "cohort" if "cohort" in name_lc else "course"
        sidecar = e.grain_hitl_path.parent / f"sma_grain_resolution_{et}.json"
        raise RuntimeError(
            f"SMA grain reconciliation required at {e.grain_hitl_path}. "
            "Execute mode does not poll Unity Catalog for grain HITL; resolve grain in an onboard "
            f"run (or manually run resolve_items and create {sidecar.name}), then re-execute."
        ) from e


def reload_field_manifest_entity(
    manifest_map_path: Path, entity: Literal["cohort", "course"]
) -> Any:
    from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
        FieldMappingManifest,
    )

    data = json.loads(manifest_map_path.read_text())
    return FieldMappingManifest.model_validate(data["manifests"][entity])


def run_onboard_gate_2_entity_with_grain_uc(
    *,
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    paths: SmaSchemaMappingRunPaths,
    db_run_id: str | None,
    transformation_map: Any,
    manifest: Any,
    entity: Literal["cohort", "course"],
    dataframes: dict[str, Any],
    schema: Any,
    spark_session: Any,
    institution_id_from_tm: str,
    enriched_contract: dict[str, Any],
    grain_hitl_path: Path,
    poll_interval_seconds: int,
    timeout_seconds: int,
) -> tuple[Any, Any]:
    """Execute one entity map; on grain mismatch register ``sma_gate_2_grain``, poll UC, resolve."""
    from edvise.genai.mapping.identity_agent.hitl.resolver import check_gate, resolve_items
    from edvise.genai.mapping.state import job_state as pipeline_job_state

    manifest_cur = manifest
    manifest_map_path = paths.manifest_map
    while True:
        try:
            result = execute_transformation_map_for_sma_run(
                transformation_map=transformation_map,
                manifest=manifest_cur,
                dataframes=dataframes,
                schema=schema,
                spark_session=spark_session,
                institution_id=institution_id_from_tm,
                enriched_contract=enriched_contract,
                manifest_map_path=manifest_map_path,
                grain_hitl_path=grain_hitl_path,
            )
            return result, manifest_cur
        except SmaGrainHitlPending as pend:
            LOGGER.info(
                "[onboard/gate_2] SMA grain mismatch — registering UC phase sma_gate_2_grain (%s)",
                pend.grain_hitl_path,
            )
            pipeline_job_state.register_sma_gate_2_grain_artifacts(
                catalog,
                institution_id,
                onboard_run_id,
                grain_hitl_paths=[pend.grain_hitl_path],
            )
            LOGGER.info(
                "[onboard/gate_2] Waiting for Unity Catalog HITL approval (sma_gate_2_grain)"
            )
            pipeline_job_state.wait_for_sma_gate_2_grain_hitl(
                catalog,
                onboard_run_id,
                institution_id=institution_id,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
            )
            check_gate(pend.grain_hitl_path)
            resolve_items(
                pend.grain_hitl_path,
                resolved_by="pipeline",
                run_log_path=paths.run_log,
                db_run_id=db_run_id,
            )
            pipeline_job_state.after_sma_gate_2_grain_approved(
                catalog, institution_id, onboard_run_id
            )
            manifest_cur = reload_field_manifest_entity(manifest_map_path, entity)
