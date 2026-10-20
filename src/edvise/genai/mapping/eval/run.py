"""Run registered GenAI mapping evals through ``mlflow.genai.evaluate``.

Each prompt is one step id (``2a``, ``2b``, and further ids such as Identity
Agent). The Databricks eval job passes that id as ``--step``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from edvise.genai.mapping.eval.dataset import (
    Step2aEvalContext,
    Step2bEvalContext,
    load_step2a_reference_cases,
    load_step2b_reference_cases,
)
from edvise.genai.mapping.eval.models import (
    ResolvedEvalModel,
    resolve_comparison_models,
)
from edvise.genai.mapping.eval.predict import (
    PredictionObservation,
    make_step2a_predict_fn,
    make_step2b_predict_fn,
)
from edvise.genai.mapping.eval.scorers import assess_step2a, assess_step2b

LOGGER = logging.getLogger(__name__)


def _experiment_subpath(step_id: str) -> str:
    return f"edvise/genai_mapping/step{step_id}_model_comparison"


EXPERIMENT_SUBPATH = _experiment_subpath("2a")
_MLFLOW_GENAI_REQUIRED = (
    "mlflow.genai.evaluate requires MLflow 3.3 or newer. "
    "Run this evaluation with the Databricks genai mapping eval job, which pins mlflow==3.3.2."
)


def experiment_name_for(ds_run_as: str, *, subpath: str = EXPERIMENT_SUBPATH) -> str:
    """MLflow experiment under the job service principal's workspace folder.

    The principal can create experiments in ``/Users/<ds_run_as>/`` and often
    cannot create them under ``/Shared``.
    """
    account = ds_run_as.strip().strip("/")
    if not account or "/" in account:
        raise ValueError(
            "ds_run_as must be the Databricks service principal application id."
        )
    return f"/Users/{account}/{subpath.strip('/')}"


@dataclass(frozen=True)
class _Bakeoff:
    """What differs between prompt bake-offs. The run loop is shared."""

    step_id: str
    description: str
    predicted_directory: str
    predicted_payload_key: str
    overall_columns: tuple[str, ...]
    headline_metrics: tuple[str, ...]
    assess: Callable[..., Any]
    load_cases: Callable[..., list[Any]]
    make_predict_fn: Callable[..., Any]

    @property
    def label(self) -> str:
        return f"Step {self.step_id}"

    @property
    def artifact_prefix(self) -> str:
        return f"step{self.step_id}"

    @property
    def experiment_subpath(self) -> str:
        return _experiment_subpath(self.step_id)

    @property
    def empty_message(self) -> str:
        return f"Step {self.step_id} evaluation has no reference cases."


def _bakeoff_runner(bakeoff: _Bakeoff) -> Callable[..., list[str]]:
    def run(
        *,
        catalog: str,
        models: str | list[str],
        ds_run_as: str,
        pipeline_version: str = "eval",
        mlflow_module: Any | None = None,
        cases: Sequence[Any] | None = None,
    ) -> list[str]:
        return _run_model_comparison(
            bakeoff,
            catalog=catalog,
            models=models,
            ds_run_as=ds_run_as,
            pipeline_version=pipeline_version,
            mlflow_module=mlflow_module,
            cases=cases,
        )

    return run


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

_STEP2B_OVERALL_COLUMNS = (
    "plan_decision_accuracy",
    "exact_plan_match_rate",
    "function_chain_accuracy",
    "function_chain_precision",
    "function_chain_recall",
    "function_chain_f1",
    "required_step_recall",
    "critical_arg_accuracy",
    "map_values_step_accuracy",
    "dependency_accuracy",
    "fill_constant_accuracy",
    "execution_safe_rate",
    "execution_exact_rate",
)


@dataclass(frozen=True)
class EvalStep:
    """One prompt bake-off the shared eval job can run."""

    step_id: str
    description: str
    run: Callable[..., list[str]]


_BAKEOFFS: dict[str, _Bakeoff] = {
    "2a": _Bakeoff(
        step_id="2a",
        description="Schema Mapping Agent manifest prompt",
        predicted_directory="manifests",
        predicted_payload_key="manifest",
        overall_columns=_OVERALL_COLUMNS,
        headline_metrics=("envelope_valid", "source_exact", "execution_ready"),
        assess=assess_step2a,
        load_cases=load_step2a_reference_cases,
        make_predict_fn=make_step2a_predict_fn,
    ),
    "2b": _Bakeoff(
        step_id="2b",
        description="Schema Mapping Agent transformation-map prompt",
        predicted_directory="transformation_maps",
        predicted_payload_key="transformation_map",
        overall_columns=_STEP2B_OVERALL_COLUMNS,
        headline_metrics=("transformation_valid", "execution_exact", "execution_safe"),
        assess=assess_step2b,
        load_cases=load_step2b_reference_cases,
        make_predict_fn=make_step2b_predict_fn,
    ),
}

_EVAL_STEPS: dict[str, EvalStep] = {
    bakeoff.step_id: EvalStep(
        bakeoff.step_id,
        bakeoff.description,
        _bakeoff_runner(bakeoff),
    )
    for bakeoff in _BAKEOFFS.values()
}


def run_step2a_evaluation(
    *,
    catalog: str,
    models: str | list[str],
    ds_run_as: str,
    pipeline_version: str = "eval",
    mlflow_module: Any | None = None,
    cases: list[Step2aEvalContext] | None = None,
) -> list[str]:
    """Score the manifest prompt. See :func:`run_evaluation` for the shared loop."""
    return run_evaluation(
        step="2a",
        catalog=catalog,
        models=models,
        ds_run_as=ds_run_as,
        pipeline_version=pipeline_version,
        mlflow_module=mlflow_module,
        cases=cases,
    )


def run_step2b_evaluation(
    *,
    catalog: str,
    models: str | list[str],
    ds_run_as: str,
    pipeline_version: str = "eval",
    mlflow_module: Any | None = None,
    cases: list[Step2bEvalContext] | None = None,
) -> list[str]:
    """Score the transformation-map prompt. See :func:`run_evaluation`."""
    return run_evaluation(
        step="2b",
        catalog=catalog,
        models=models,
        ds_run_as=ds_run_as,
        pipeline_version=pipeline_version,
        mlflow_module=mlflow_module,
        cases=cases,
    )


def supported_eval_steps() -> tuple[str, ...]:
    """Step ids the eval job accepts, in registration order."""
    return tuple(_EVAL_STEPS)


def eval_step_help() -> str:
    """Argparse help listing every registered step."""
    details = "; ".join(
        f"{step.step_id} ({step.description})" for step in _EVAL_STEPS.values()
    )
    return f"Registered eval to score: {details}."


def run_evaluation(
    *,
    step: str,
    catalog: str,
    models: str | list[str],
    ds_run_as: str,
    pipeline_version: str = "eval",
    mlflow_module: Any | None = None,
    cases: Sequence[Any] | None = None,
) -> list[str]:
    """Score one registered step for every supplied model.

    ``step`` is looked up in :data:`_EVAL_STEPS`. Add Identity Agent, or
    another prompt, by registering a runner there; this job then accepts
    that id with no new Databricks job.
    """
    key = step.strip().lower()
    try:
        spec = _EVAL_STEPS[key]
    except KeyError as exc:
        known = ", ".join(_EVAL_STEPS)
        raise ValueError(
            f"Unknown eval step {step!r}. Choose one of: {known}."
        ) from exc
    return spec.run(
        catalog=catalog,
        models=models,
        ds_run_as=ds_run_as,
        pipeline_version=pipeline_version,
        mlflow_module=mlflow_module,
        cases=cases,
    )


def _run_model_comparison(
    bakeoff: _Bakeoff,
    *,
    catalog: str,
    models: str | list[str],
    ds_run_as: str,
    pipeline_version: str,
    mlflow_module: Any | None,
    cases: Sequence[Any] | None,
) -> list[str]:
    loaded = list(cases) if cases is not None else bakeoff.load_cases(catalog=catalog)
    if not loaded:
        raise ValueError(bakeoff.empty_message)
    experiment_name = experiment_name_for(ds_run_as, subpath=bakeoff.experiment_subpath)
    resolved_models = resolve_comparison_models(models)
    mlflow = mlflow_module or _import_mlflow_genai()
    run_ids: list[str] = []
    for resolved in resolved_models:
        run_id = _execute_evaluation(
            bakeoff,
            mlflow=mlflow,
            cases=loaded,
            resolved=resolved,
            experiment_name=experiment_name,
            pipeline_version=pipeline_version or "eval",
        )
        LOGGER.info(
            "%s evaluation complete | model=%s | gateway=%s | run_id=%s",
            bakeoff.label,
            resolved.selection,
            resolved.gateway_model_id,
            run_id,
        )
        run_ids.append(run_id)
    return run_ids


def _execute_evaluation(
    bakeoff: _Bakeoff,
    *,
    mlflow: Any,
    cases: Sequence[Any],
    resolved: ResolvedEvalModel,
    experiment_name: str,
    pipeline_version: str,
) -> str:
    observation = PredictionObservation()
    diagnostics: list[dict[str, Any]] = []
    predict_fn = bakeoff.make_predict_fn(
        cases=cases,
        gateway_model_id=resolved.gateway_model_id,
        pipeline_version=pipeline_version,
        observation=observation,
    )
    scorers = _headline_scorers(
        mlflow, diagnostics, bakeoff.assess, bakeoff.headline_metrics
    )
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
                "catalog": catalog,
                "n_reference_cases": str(len(cases)),
                "target_reference_ids": target_ids,
                "few_shot_reference_ids": few_shot_ids,
            }
        )
        _log_operational_metrics(mlflow, observation)
        _log_predicted_manifests(
            mlflow,
            observation,
            directory=bakeoff.predicted_directory,
            payload_key=bakeoff.predicted_payload_key,
        )
        mlflow.log_table(
            data=overall_by_school_table(
                diagnostics, overall_columns=bakeoff.overall_columns
            ),
            artifact_file=f"diagnostics/{bakeoff.artifact_prefix}_overall_by_school.json",
        )
        mlflow.log_dict(
            {
                "model_family": resolved.family,
                "model_version": resolved.version,
                "gateway_model_id": resolved.gateway_model_id,
                "catalog": catalog,
                "predictions": observation.calls,
                "cases": diagnostics,
            },
            f"diagnostics/{bakeoff.artifact_prefix}_field_detail.json",
        )
    return str(run_id)


def _headline_scorers(
    mlflow: Any,
    diagnostics: list[dict[str, Any]],
    assess: Callable[..., Any],
    metric_names: tuple[str, ...],
) -> list[Any]:
    """One MLflow scorer per headline field. The first scorer records the diagnostic.

    MLflow calls each scorer separately. The assessment is cached for the same
    prediction so the full score runs once per case.
    """
    scorer = mlflow.genai.scorers.scorer
    cached: dict[tuple[int, int], Any] = {}

    def assessment_for(outputs: Any, expectations: dict[str, Any]) -> Any:
        key = (id(outputs), id(expectations))
        assessment = cached.get(key)
        if assessment is None:
            assessment = assess(outputs, expectations)
            cached[key] = assessment
        return assessment

    bound: list[Any] = []
    for index, name in enumerate(metric_names):

        def metric(
            outputs: Any,
            expectations: dict[str, Any],
            *,
            _name: str = name,
            _record: bool = index == 0,
        ) -> float:
            assessment = assessment_for(outputs, expectations)
            if _record:
                diagnostics.append(assessment.diagnostic)
            return float(getattr(assessment, _name))

        metric.__name__ = name
        bound.append(scorer(metric))
    return bound


def overall_by_school_table(
    diagnostics: list[dict[str, Any]],
    *,
    overall_columns: tuple[str, ...] = _OVERALL_COLUMNS,
) -> dict[str, list[Any]]:
    """One row per school, with that case's ``overall`` scores as columns.

    A school that fails before scoring still gets a row. Its metric columns
    are empty and ``validation_error`` says why. A repeated school keeps the
    later row, which is the scored call rather than the pre-flight probe.
    """
    columns = (
        "institution_id",
        "validation_passed",
        "validation_error",
        *overall_columns,
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
        for column in overall_columns:
            row[column] = scores.get(column)
        if key not in rows:
            order.append(key)
        rows[key] = row
    return {column: [rows[key][column] for key in order] for column in columns}


def _log_predicted_manifests(
    mlflow: Any,
    observation: PredictionObservation,
    *,
    directory: str = "manifests",
    payload_key: str = "manifest",
) -> None:
    """Write one predicted artifact per reference case.

    ``mlflow.genai.evaluate`` probes the first case before scoring, so the
    same institution can appear twice. The later call is the scored one.
    """
    latest: dict[str, Any] = {}
    for record in observation.manifests:
        institution_id = str(record.get("institution_id") or "").strip()
        if institution_id:
            latest[institution_id] = record.get(payload_key)
    for institution_id, payload in latest.items():
        mlflow.log_dict(
            payload if isinstance(payload, dict) else {"response": payload},
            f"{directory}/{_artifact_stem(institution_id)}.json",
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
