"""
SMA grain job wiring: call :mod:`edvise.genai.mapping.schema_mapping_agent.execution.field_executor`
with grain paths and onboard UC polling.

Also appends ordered steps to ``sma_grain_resolution_<entity>.json`` (see
:func:`append_sma_grain_resolution_step`); execution applies them in order via
:func:`~edvise.genai.mapping.shared.grain.dedup_execution.apply_sma_grain_resolution_payload`.

Domain logic (HITL construction, LLM dedup) lives in
:mod:`edvise.genai.mapping.schema_mapping_agent.grain_resolution.hitl` and
:mod:`edvise.genai.mapping.schema_mapping_agent.grain_resolution.prompt`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Protocol, cast

LOGGER = logging.getLogger(__name__)

# Onboard allows this many full UC + ``resolve_items`` cycles; the next grain mismatch
# writes ``true_duplicate`` to ``sma_grain_resolution_<entity>.json`` and retries without further HITL (third pass).
MAX_SMA_GRAIN_ROUNDS = 2


def append_sma_grain_resolution_step(
    out: Path,
    *,
    institution_id: str,
    dataset: str,
    entity_type: str,
    manifest_source_keys: list[str],
    grain_resolution: dict[str, Any],
) -> None:
    """
    Append ``grain_resolution`` to ``grain_resolutions`` on disk, migrating a legacy
    single ``grain_resolution`` key if present.
    """
    keys = list(manifest_source_keys)
    chain: list[dict[str, Any]] = []
    inst = institution_id
    ds = dataset
    et = entity_type

    if out.is_file():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            LOGGER.warning(
                "Could not read existing SMA grain resolution file %s (%s) — overwriting",
                out,
                e,
            )
            prev = {}
        pkeys = prev.get("manifest_source_keys")
        if isinstance(pkeys, list) and pkeys and list(pkeys) != keys:
            raise ValueError(
                f"Cannot append SMA grain resolution: manifest_source_keys {list(pkeys)!r} "
                f"!= {keys!r} in {out}"
            )
        prev_i = prev.get("institution_id")
        if prev_i not in (None, "", institution_id) and institution_id:
            if str(prev_i) != str(institution_id):
                raise ValueError(
                    f"Cannot append SMA grain resolution: institution_id mismatch in {out}"
                )
        inst = str(prev.get("institution_id") or institution_id)
        ds = str(prev.get("dataset") or dataset)
        et = str(prev.get("entity_type") or entity_type)

        pr = prev.get("grain_resolutions")
        if isinstance(pr, list) and pr:
            chain = [cast(dict[str, Any], dict(x)) for x in pr if isinstance(x, dict)]
        elif isinstance(prev.get("grain_resolution"), dict):
            chain = [cast(dict[str, Any], dict(prev["grain_resolution"]))]
        chain.append(dict(grain_resolution))
    else:
        chain = [dict(grain_resolution)]

    body: dict[str, Any] = {
        "institution_id": inst,
        "dataset": ds,
        "entity_type": et,
        "manifest_source_keys": keys,
        "grain_resolutions": chain,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    LOGGER.info(
        "Appended SMA grain resolution step %d to %s",
        len(chain),
        out.name,
    )


class SmaGrainHitlPending(Exception):
    """Grain HITL file written; onboard gate_2 registers UC, polls, then ``resolve_items``."""

    def __init__(
        self,
        *,
        grain_hitl_path: Path,
        manifest_map_path: Path,
        institution_id: str,
        dataset: str,
        entity_type: Literal["cohort", "course"],
        manifest_source_keys: list[str],
    ) -> None:
        self.grain_hitl_path = grain_hitl_path
        self.manifest_map_path = manifest_map_path
        self.institution_id = institution_id
        self.dataset = dataset
        self.entity_type = entity_type
        self.manifest_source_keys = list(manifest_source_keys)
        super().__init__(f"SMA grain HITL pending UC + resolver: {grain_hitl_path}")


def _write_sma_grain_true_duplicate_resolution_file(
    grain_hitl_path: Path,
    *,
    institution_id: str,
    dataset: str,
    entity_type: Literal["cohort", "course"],
    manifest_source_keys: list[str],
) -> Path:
    """Append ``true_duplicate`` as the next step on ``sma_grain_resolution_<entity>.json``."""
    out = grain_hitl_path.parent / f"sma_grain_resolution_{entity_type}.json"
    append_sma_grain_resolution_step(
        out,
        institution_id=institution_id,
        dataset=dataset,
        entity_type=entity_type,
        manifest_source_keys=manifest_source_keys,
        grain_resolution={"dedup_strategy": "true_duplicate"},
    )
    LOGGER.info(
        "SMA grain onboard: appended true_duplicate fallback step (post HITL round limit): %s",
        out,
    )
    return out


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
    """Run SMA execution with grain paths (HITL, IA keys, optional ``sma_grain_resolution_*.json``)."""
    from edvise.genai.mapping.schema_mapping_agent.execution.field_executor import (
        GrainReconciliationRequired,
        execute_transformation_map,
    )
    from edvise.genai.mapping.schema_mapping_agent.grain_resolution.hitl import (
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
    resolution_json = grain_hitl_path.parent / f"sma_grain_resolution_{et_value}.json"
    resolution_path = resolution_json if resolution_json.is_file() else None

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
            institution_id=exc.institution_id,
            dataset=exc.dataset,
            entity_type=entity_lit,
            manifest_source_keys=list(exc.manifest_source_keys),
        ) from exc


def execute_transformation_map_for_sma_execute_mode(**kwargs: Any) -> Any:
    """``mode=execute`` has no UC grain gate; surface a clear error with the HITL path."""
    try:
        return execute_transformation_map_for_sma_run(**kwargs)
    except SmaGrainHitlPending as e:
        name_lc = e.grain_hitl_path.name.lower()
        et = "cohort" if "cohort" in name_lc else "course"
        resolution_json = e.grain_hitl_path.parent / f"sma_grain_resolution_{et}.json"
        raise RuntimeError(
            f"SMA grain reconciliation required at {e.grain_hitl_path}. "
            "Execute mode does not poll Unity Catalog for grain HITL; resolve grain in an onboard "
            f"run (or manually run resolve_items and create {resolution_json.name}), then re-execute."
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
    """Execute one entity map; on grain mismatch register ``sma_gate_2_grain``, poll UC, resolve.

    After :data:`MAX_SMA_GRAIN_ROUNDS` completed UC+resolve cycles, a further mismatch applies
    ``true_duplicate`` via ``sma_grain_resolution_<entity>.json`` without registering another HITL gate.
    """
    from edvise.genai.mapping.identity_agent.hitl.resolver import (
        check_gate,
        resolve_items,
    )
    from edvise.genai.mapping.state import job_state as pipeline_job_state

    manifest_cur = manifest
    manifest_map_path = paths.manifest_map
    completed_hitl_rounds = 0
    forced_true_duplicate = False
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
            if completed_hitl_rounds >= MAX_SMA_GRAIN_ROUNDS:
                if forced_true_duplicate:
                    raise RuntimeError(
                        "SMA grain reconciliation still unresolved after "
                        f"{MAX_SMA_GRAIN_ROUNDS} HITL round(s) and an automatic true_duplicate "
                        f"fallback ({pend.grain_hitl_path})."
                    ) from pend
                LOGGER.warning(
                    "[onboard/gate_2] SMA grain mismatch after %d HITL round(s) (limit=%d) — "
                    "writing true_duplicate grain resolution file; skipping further HITL",
                    completed_hitl_rounds,
                    MAX_SMA_GRAIN_ROUNDS,
                )
                _write_sma_grain_true_duplicate_resolution_file(
                    pend.grain_hitl_path,
                    institution_id=pend.institution_id,
                    dataset=pend.dataset,
                    entity_type=pend.entity_type,
                    manifest_source_keys=pend.manifest_source_keys,
                )
                forced_true_duplicate = True
                manifest_cur = reload_field_manifest_entity(manifest_map_path, entity)
                continue

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
            completed_hitl_rounds += 1
