"""Run one Step 2a model comparison through ``mlflow.genai.evaluate``."""

from __future__ import annotations

import logging
from typing import Any

from edvise.genai.mapping.eval.dataset import (
    PROMPT_VARIANT,
    Step2aEvalContext,
    load_step2a_eval_context,
)
from edvise.genai.mapping.eval.models import ResolvedEvalModel, resolve_eval_model
from edvise.genai.mapping.eval.predict import (
    PredictionObservation,
    make_step2a_predict_fn,
)
from edvise.genai.mapping.eval.scorers import assess_step2a

LOGGER = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "/Shared/edvise/genai_mapping/step2a_model_comparison"
_MLFLOW_GENAI_REQUIRED = (
    "mlflow.genai.evaluate requires MLflow 3.3 or newer. "
    "Run this evaluation with the Databricks step 2a eval job, which pins mlflow==3.3.2."
)


def run_step2a_evaluation(
    *,
    catalog: str,
    model: str,
    gateway_model_id: str | None = None,
    target_institution_id: str | None = None,
    reference_institution_id: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    pipeline_version: str = "eval",
    mlflow_module: Any | None = None,
) -> str:
    """Evaluate one gateway model and return the MLflow run id."""
    context_kwargs: dict[str, Any] = {"catalog": catalog}
    if target_institution_id:
        context_kwargs["target_institution_id"] = target_institution_id
    if reference_institution_id:
        context_kwargs["reference_institution_id"] = reference_institution_id
    context = load_step2a_eval_context(**context_kwargs)
    resolved = resolve_eval_model(model, gateway_model_id=gateway_model_id)
    mlflow = mlflow_module or _import_mlflow_genai()
    run_id = _execute_evaluation(
        mlflow=mlflow,
        context=context,
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
    return run_id


def _execute_evaluation(
    *,
    mlflow: Any,
    context: Step2aEvalContext,
    resolved: ResolvedEvalModel,
    experiment_name: str,
    pipeline_version: str,
) -> str:
    observation = PredictionObservation()
    diagnostics: list[dict[str, Any]] = []
    predict_fn = make_step2a_predict_fn(
        context=context,
        gateway_model_id=resolved.gateway_model_id,
        pipeline_version=pipeline_version,
        observation=observation,
    )
    scorers = _mlflow_scorers(mlflow, diagnostics)
    mlflow.set_experiment(experiment_name)
    result = mlflow.genai.evaluate(
        data=[context.evaluation_record()],
        predict_fn=predict_fn,
        scorers=scorers,
    )
    run_id = result.run_id
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(
            {
                "model_family": resolved.family,
                "model_version": resolved.version,
                "gateway_model_id": resolved.gateway_model_id,
                "prompt_variant": PROMPT_VARIANT,
                "institution_id": context.target_institution_id,
                "reference_institution_id": context.reference_institution_id,
                "catalog": context.catalog,
            }
        )
        _log_operational_metrics(mlflow, observation)
        mlflow.log_dict(
            {
                "prompt_variant": PROMPT_VARIANT,
                "model_family": resolved.family,
                "model_version": resolved.version,
                "gateway_model_id": resolved.gateway_model_id,
                "institution_id": context.target_institution_id,
                "reference_institution_id": context.reference_institution_id,
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
