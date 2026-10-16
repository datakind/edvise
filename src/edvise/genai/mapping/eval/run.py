"""Run one Step 2a model comparison through ``mlflow.genai.evaluate``."""

from __future__ import annotations

import logging
import re
from typing import Any

from edvise.genai.mapping.eval.dataset import (
    PROMPT_VARIANT,
    Step2aEvalContext,
    load_step2a_reference_cases,
)
from edvise.genai.mapping.eval.models import (
    ResolvedEvalModel,
    resolve_comparison_models,
)
from edvise.genai.mapping.eval.predict import (
    PredictionObservation,
    make_step2a_predict_fn,
)
from edvise.genai.mapping.eval.scorers import assess_step2a

LOGGER = logging.getLogger(__name__)

EXPERIMENT_SUBPATH = "edvise/genai_mapping/step2a_model_comparison"
_MLFLOW_GENAI_REQUIRED = (
    "mlflow.genai.evaluate requires MLflow 3.3 or newer. "
    "Run this evaluation with the Databricks step 2a eval job, which pins mlflow==3.3.2."
)


def experiment_name_for(ds_run_as: str) -> str:
    """MLflow experiment under the job service principal's workspace folder.

    The principal can create experiments in ``/Users/<ds_run_as>/`` and often
    cannot create them under ``/Shared``.
    """
    account = ds_run_as.strip().strip("/")
    if not account or "/" in account:
        raise ValueError(
            "ds_run_as must be the Databricks service principal application id."
        )
    return f"/Users/{account}/{EXPERIMENT_SUBPATH}"


def run_step2a_evaluation(
    *,
    catalog: str,
    models: str | list[str],
    ds_run_as: str,
    pipeline_version: str = "eval",
    mlflow_module: Any | None = None,
    cases: list[Step2aEvalContext] | None = None,
) -> list[str]:
    """Score every supplied model on the same reference cases.

    ``models`` is a comma-separated string, JSON array, or list of slots such
    as ``sonnet:4.5``. Each slot defaults to the hosted
    ``system.ai.claude-<family>-<version>`` foundation model unless overridden
    (see :mod:`edvise.genai.mapping.eval.models`). Each model is one MLflow
    run under ``/Users/<ds_run_as>/``; run metrics are the mean across the
    reference cases, so the experiment compares the models.
    """
    experiment_name = experiment_name_for(ds_run_as)
    loaded = (
        cases if cases is not None else load_step2a_reference_cases(catalog=catalog)
    )
    if not loaded:
        raise ValueError("Step 2a evaluation has no reference cases.")
    resolved_models = resolve_comparison_models(models)
    mlflow = mlflow_module or _import_mlflow_genai()
    run_ids: list[str] = []
    for resolved in resolved_models:
        run_id = _execute_evaluation(
            mlflow=mlflow,
            cases=loaded,
            resolved=resolved,
            experiment_name=experiment_name,
            pipeline_version=pipeline_version or "eval",
        )
        LOGGER.info(
            "Step 2a evaluation complete | model=%s | gateway=%s | run_id=%s",
            resolved.selection,
            resolved.gateway_model_id,
            run_id,
        )
        run_ids.append(run_id)
    return run_ids


def _execute_evaluation(
    *,
    mlflow: Any,
    cases: list[Step2aEvalContext],
    resolved: ResolvedEvalModel,
    experiment_name: str,
    pipeline_version: str,
) -> str:
    observation = PredictionObservation()
    diagnostics: list[dict[str, Any]] = []
    predict_fn = make_step2a_predict_fn(
        cases=cases,
        gateway_model_id=resolved.gateway_model_id,
        pipeline_version=pipeline_version,
        observation=observation,
    )
    scorers = _mlflow_scorers(mlflow, diagnostics)
    mlflow.set_experiment(experiment_name)
    result = mlflow.genai.evaluate(
        data=[case.evaluation_record() for case in cases],
        predict_fn=predict_fn,
        scorers=scorers,
    )
    run_id = result.run_id
    catalog = cases[0].catalog
    target_ids = ",".join(case.target_institution_id for case in cases)
    few_shot_ids = ",".join(case.reference_institution_id for case in cases)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(
            {
                "model_family": resolved.family,
                "model_version": resolved.version,
                "gateway_model_id": resolved.gateway_model_id,
                "prompt_variant": PROMPT_VARIANT,
                "catalog": catalog,
                "n_reference_cases": str(len(cases)),
                "target_reference_ids": target_ids,
                "few_shot_reference_ids": few_shot_ids,
            }
        )
        _log_operational_metrics(mlflow, observation)
        _log_predicted_manifests(mlflow, observation)
        mlflow.log_table(
            data=overall_by_school_table(diagnostics),
            artifact_file="diagnostics/step2a_overall_by_school.json",
        )
        mlflow.log_dict(
            {
                "prompt_variant": PROMPT_VARIANT,
                "model_family": resolved.family,
                "model_version": resolved.version,
                "gateway_model_id": resolved.gateway_model_id,
                "catalog": catalog,
                "predictions": observation.calls,
                "cases": diagnostics,
            },
            "diagnostics/step2a_field_detail.json",
        )
    return str(run_id)


def _mlflow_scorers(mlflow: Any, diagnostics: list[dict[str, Any]]) -> list[Any]:
    scorer = mlflow.genai.scorers.scorer

    @scorer
    def envelope_valid(outputs: Any, expectations: dict[str, Any]) -> float:
        assessment = assess_step2a(outputs, expectations)
        diagnostics.append(assessment.diagnostic)
        return assessment.envelope_valid

    @scorer
    def source_exact(outputs: Any, expectations: dict[str, Any]) -> float:
        return assess_step2a(outputs, expectations).source_exact

    @scorer
    def execution_ready(outputs: Any, expectations: dict[str, Any]) -> float:
        return assess_step2a(outputs, expectations).execution_ready

    return [envelope_valid, source_exact, execution_ready]


_OVERALL_COLUMNS = (
    "map_decision_accuracy",
    "mappable_precision_strict",
    "mappable_recall_strict",
    "source_exact_accuracy_gold_mappable",
    "row_selection_accuracy",
    "row_selection_accuracy_non_anyrow",
    "join_f1",
    "join_table_accuracy",
    "join_key_accuracy",
    "degree_filter_accuracy",
    "exact_field_match_rate",
    "execution_ready_rate",
)


def overall_by_school_table(diagnostics: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """One row per school, with that case's ``overall`` scores as columns.

    A school that fails before scoring still gets a row. Its metric columns
    are empty and ``validation_error`` says why. A repeated school keeps the
    later row, which is the scored call rather than the pre-flight probe.
    """
    columns = (
        "institution_id",
        "validation_passed",
        "validation_error",
        *_OVERALL_COLUMNS,
    )
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, case in enumerate(diagnostics):
        institution_id = case.get("institution_id")
        school = institution_id.strip() if isinstance(institution_id, str) else ""
        key = school or f"__unidentified_{index}"
        overall = case.get("overall")
        scores = overall if isinstance(overall, dict) else {}
        row: dict[str, Any] = {
            "institution_id": school or None,
            "validation_passed": case.get("validation_passed"),
            "validation_error": case.get("validation_error"),
        }
        for column in _OVERALL_COLUMNS:
            row[column] = scores.get(column)
        if key not in rows:
            order.append(key)
        rows[key] = row
    return {column: [rows[key][column] for key in order] for column in columns}


def _log_predicted_manifests(mlflow: Any, observation: PredictionObservation) -> None:
    """Write one predicted manifest per reference case.

    ``mlflow.genai.evaluate`` probes the first case before scoring, so the
    same institution can appear twice. The later call is the scored one.
    """
    latest: dict[str, Any] = {}
    for record in observation.manifests:
        institution_id = str(record.get("institution_id") or "").strip()
        if institution_id:
            latest[institution_id] = record.get("manifest")
    for institution_id, manifest in latest.items():
        mlflow.log_dict(
            manifest if isinstance(manifest, dict) else {"response": manifest},
            f"manifests/{_artifact_stem(institution_id)}.json",
        )


def _artifact_stem(institution_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", institution_id).strip("._")
    return stem or "manifest"


def _log_operational_metrics(mlflow: Any, observation: PredictionObservation) -> None:
    latencies = [
        float(call["latency_s"])
        for call in observation.calls
        if isinstance(call.get("latency_s"), int | float)
    ]
    sizes = [
        float(call["response_chars"])
        for call in observation.calls
        if isinstance(call.get("response_chars"), int | float)
    ]
    if latencies:
        mlflow.log_metric("latency_s", round(sum(latencies) / len(latencies), 3))
    if sizes:
        mlflow.log_metric("response_chars", round(sum(sizes) / len(sizes), 3))


def _import_mlflow_genai() -> Any:
    import mlflow

    genai = getattr(mlflow, "genai", None)
    evaluate = getattr(genai, "evaluate", None) if genai is not None else None
    if evaluate is None:
        raise RuntimeError(_MLFLOW_GENAI_REQUIRED)
    return mlflow
