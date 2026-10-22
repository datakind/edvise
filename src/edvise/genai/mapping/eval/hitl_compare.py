"""Compare HITL flags on two Step 2a MLflow runs.

This scores the raw predicted manifests. It does not run the refinement LLM, so
a flag here is a field the pipeline would send into refinement and HITL: confidence
at or below the production threshold, or a deterministic manifest validation error.

Notebook::

    from edvise.genai.mapping.eval.hitl_compare import compare_hitl_runs

    comparison = compare_hitl_runs("<run id a>", "<run id b>")
    comparison.fields
    comparison.by_school
    comparison.by_failure_mode
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas import (
    HITL_CONFIDENCE_THRESHOLD,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.prompts.refine import (
    _failure_mode_for_validation_errors,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.validation import (
    validate_manifest,
)
from edvise.genai.mapping.shared.reference_pin import resolve_sma_few_shot_pin
from edvise.genai.mapping.shared.schema_contract import (
    parse_enriched_schema_contract_for_sma,
)

_CONTRACT_NAME = "enriched_schema_contract.json"
_KEYS = ("institution_id", "entity", "target_field")


@dataclass(frozen=True)
class HitlRunComparison:
    """Field-level flags and the two rollups for a pair of runs."""

    fields: pd.DataFrame
    by_school: pd.DataFrame
    by_failure_mode: pd.DataFrame
    label_a: str
    label_b: str


def compare_hitl_runs(
    run_id_a: str,
    run_id_b: str,
    *,
    catalog: str | None = None,
) -> HitlRunComparison:
    """Download ``manifests/*.json`` for two runs and compare their HITL flags.

    ``catalog`` overrides the ``catalog`` param logged on the runs. The schema
    contract for each school is the pinned reference contract, which is what
    production validation uses.
    """
    params_a, manifests_a = _load_run_manifests(run_id_a)
    params_b, manifests_b = _load_run_manifests(run_id_b)
    resolved_catalog = catalog or params_a.get("catalog") or params_b.get("catalog")
    if not resolved_catalog:
        raise ValueError(
            "Neither run logged a catalog param. Pass catalog= so the schema "
            "contracts can be loaded from the reference pins."
        )
    label_a, label_b = _labels(params_a, run_id_a, params_b, run_id_b)
    contracts = _contracts_for(
        set(manifests_a) | set(manifests_b),
        catalog=resolved_catalog,
    )
    rows_a = _rows_for_run(manifests_a, contracts, run_id=run_id_a, label=label_a)
    rows_b = _rows_for_run(manifests_b, contracts, run_id=run_id_b, label=label_b)
    fields = _compare_rows(rows_a, rows_b, label_a, label_b)
    return HitlRunComparison(
        fields=fields,
        by_school=_summarize_by_school(fields, label_a, label_b),
        by_failure_mode=_summarize_by_failure_mode(fields, label_a, label_b),
        label_a=label_a,
        label_b=label_b,
    )


def manifest_hitl_rows(
    manifest: dict[str, Any],
    *,
    schema_contract: Any | None,
    institution_id: str,
    run_id: str = "",
    label: str = "",
) -> list[dict[str, Any]]:
    """One row per field for a single predicted manifest."""
    if manifest.get("error") and "manifests" not in manifest:
        return [
            _row(
                institution_id=institution_id,
                entity="",
                target_field="",
                confidence=None,
                source_table=None,
                source_column=None,
                rationale=None,
                low_confidence=False,
                error_codes="",
                error_details=str(manifest.get("error")),
                failure_mode="prediction_failed",
                hitl_flagged=False,
                run_id=run_id,
                label=label,
            )
        ]

    entities = manifest.get("manifests")
    if not isinstance(entities, dict):
        return [
            _row(
                institution_id=institution_id,
                entity="",
                target_field="",
                confidence=None,
                source_table=None,
                source_column=None,
                rationale=None,
                low_confidence=False,
                error_codes="",
                error_details="Predicted artifact has no manifests object.",
                failure_mode="invalid_manifest",
                hitl_flagged=False,
                run_id=run_id,
                label=label,
            )
        ]

    rows: list[dict[str, Any]] = []
    for entity, block in entities.items():
        if not isinstance(block, dict):
            continue
        try:
            parsed = FieldMappingManifest.model_validate(block)
        except ValidationError as exc:
            rows.append(
                _row(
                    institution_id=institution_id,
                    entity=str(entity),
                    target_field="",
                    confidence=None,
                    source_table=None,
                    source_column=None,
                    rationale=None,
                    low_confidence=False,
                    error_codes="invalid_manifest",
                    error_details=str(exc),
                    failure_mode="invalid_manifest",
                    hitl_flagged=True,
                    run_id=run_id,
                    label=label,
                )
            )
            continue
        errors = (
            validate_manifest(parsed, schema_contract)
            if schema_contract is not None
            else []
        )
        by_field: dict[str, list[Any]] = defaultdict(list)
        for error in errors:
            by_field[error.target_field].append(error)
        seen: set[str] = set()
        for record in parsed.mappings:
            seen.add(record.target_field)
            rows.append(
                _row_for_record(
                    record=record,
                    errors=by_field.get(record.target_field, []),
                    institution_id=institution_id,
                    entity=str(entity),
                    run_id=run_id,
                    label=label,
                )
            )
        for target_field, field_errors in by_field.items():
            if target_field in seen:
                continue
            rows.append(
                _row(
                    institution_id=institution_id,
                    entity=str(entity),
                    target_field=target_field,
                    confidence=None,
                    source_table=None,
                    source_column=None,
                    rationale=None,
                    low_confidence=False,
                    error_codes=", ".join(_error_code(error) for error in field_errors),
                    error_details=" | ".join(error.detail for error in field_errors),
                    failure_mode=_failure_mode_for_validation_errors(field_errors),
                    hitl_flagged=True,
                    run_id=run_id,
                    label=label,
                )
            )
    return rows


def _rows_for_run(
    manifests: dict[str, dict[str, Any]],
    contracts: dict[str, Any | None],
    *,
    run_id: str,
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for institution_id, manifest in manifests.items():
        found = manifest.get("institution_id")
        school = (
            found.strip()
            if isinstance(found, str) and found.strip()
            else institution_id
        )
        rows.extend(
            manifest_hitl_rows(
                manifest,
                schema_contract=contracts.get(school),
                institution_id=school,
                run_id=run_id,
                label=label,
            )
        )
    return rows


def _row_for_record(
    *,
    record: Any,
    errors: list[Any],
    institution_id: str,
    entity: str,
    run_id: str,
    label: str,
) -> dict[str, Any]:
    low_confidence = record.confidence <= HITL_CONFIDENCE_THRESHOLD
    if errors:
        failure_mode = _failure_mode_for_validation_errors(errors)
    elif low_confidence:
        failure_mode = "low_confidence"
    else:
        failure_mode = ""
    return _row(
        institution_id=institution_id,
        entity=entity,
        target_field=record.target_field,
        confidence=record.confidence,
        source_table=record.source_table,
        source_column=record.source_column,
        rationale=record.rationale,
        low_confidence=low_confidence,
        error_codes=", ".join(_error_code(error) for error in errors),
        error_details=" | ".join(error.detail for error in errors),
        failure_mode=failure_mode,
        hitl_flagged=low_confidence or bool(errors),
        run_id=run_id,
        label=label,
    )


def _row(**values: Any) -> dict[str, Any]:
    return values


def _error_code(error: Any) -> str:
    code = error.error_code
    return str(getattr(code, "value", code))


def _compare_rows(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    left = _indexed(rows_a)
    right = _indexed(rows_b)
    fields = left.merge(right, on=list(_KEYS), how="outer", suffixes=("_a", "_b"))
    flagged_a = _as_bool(fields["hitl_flagged_a"])
    flagged_b = _as_bool(fields["hitl_flagged_b"])
    fields["flag_bucket"] = [
        _bucket(a, b, label_a, label_b) for a, b in zip(flagged_a, flagged_b)
    ]
    fields.insert(0, "label_a", label_a)
    fields.insert(1, "label_b", label_b)
    return fields.sort_values(list(_KEYS), kind="stable").reset_index(drop=True)


def _indexed(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                *_KEYS,
                "confidence",
                "source_table",
                "source_column",
                "rationale",
                "low_confidence",
                "error_codes",
                "error_details",
                "failure_mode",
                "hitl_flagged",
                "run_id",
                "label",
            ]
        )
    return frame.drop(columns=["label"], errors="ignore")


def _as_bool(values: pd.Series) -> pd.Series:
    return values.map(lambda value: bool(value) if pd.notna(value) else False)


def _bucket(flagged_a: bool, flagged_b: bool, label_a: str, label_b: str) -> str:
    if flagged_a and flagged_b:
        return "both"
    if flagged_a:
        return f"{label_a}_only"
    if flagged_b:
        return f"{label_b}_only"
    return "neither"


def _summarize_by_school(
    fields: pd.DataFrame, label_a: str, label_b: str
) -> pd.DataFrame:
    buckets = ("both", f"{label_a}_only", f"{label_b}_only", "neither")
    rows: list[dict[str, Any]] = []
    if fields.empty:
        return pd.DataFrame(columns=["institution_id", *buckets, "n_fields"])
    for institution_id, group in fields.groupby("institution_id", dropna=False):
        counts = group["flag_bucket"].value_counts()
        row: dict[str, Any] = {"institution_id": institution_id}
        for bucket in buckets:
            row[bucket] = int(counts.get(bucket, 0))
        row["n_fields"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_by_failure_mode(
    fields: pd.DataFrame, label_a: str, label_b: str
) -> pd.DataFrame:
    modes = sorted(
        {
            mode
            for column in ("failure_mode_a", "failure_mode_b")
            if column in fields
            for mode in fields[column].fillna("").astype(str)
            if mode
        }
    )
    rows = []
    for mode in modes:
        rows.append(
            {
                "failure_mode": mode,
                label_a: int((fields.get("failure_mode_a", "") == mode).sum()),
                label_b: int((fields.get("failure_mode_b", "") == mode).sum()),
            }
        )
    return pd.DataFrame(rows, columns=["failure_mode", label_a, label_b])


def _labels(
    params_a: dict[str, str],
    run_id_a: str,
    params_b: dict[str, str],
    run_id_b: str,
) -> tuple[str, str]:
    label_a = _label(params_a, run_id_a)
    label_b = _label(params_b, run_id_b)
    if label_a == label_b:
        return f"{label_a}_{run_id_a[:8]}", f"{label_b}_{run_id_b[:8]}"
    return label_a, label_b


def _label(params: dict[str, str], run_id: str) -> str:
    family = params.get("model_family", "").strip()
    version = params.get("model_version", "").strip().replace(".", "_")
    if family and version:
        return f"{family}_{version}"
    return run_id[:8]


def _load_run_manifests(
    run_id: str,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    import mlflow
    from mlflow.tracking import MlflowClient

    params = dict(MlflowClient().get_run(run_id).data.params)
    local = Path(
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="manifests")
    )
    manifests: dict[str, dict[str, Any]] = {}
    for path in sorted(local.glob("*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            manifests[path.stem] = payload
    if not manifests:
        raise ValueError(f"Run {run_id} has no manifests/*.json artifacts.")
    return params, manifests


def _contracts_for(institution_ids: set[str], *, catalog: str) -> dict[str, Any | None]:
    contracts: dict[str, Any | None] = {}
    for institution_id in sorted(institution_ids):
        try:
            pin = resolve_sma_few_shot_pin(institution_id, catalog=catalog)
            path = pin.current_root / _CONTRACT_NAME
            payload = json.loads(path.read_text())
            contracts[institution_id] = parse_enriched_schema_contract_for_sma(payload)
        except (OSError, ValueError, ValidationError):
            contracts[institution_id] = None
    return contracts
