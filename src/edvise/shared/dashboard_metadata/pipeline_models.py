from __future__ import annotations

import json
import logging
import typing as t
from datetime import datetime

LOGGER = logging.getLogger(__name__)


def _get_spark_session():
    try:
        from pyspark.sql import SparkSession  # type: ignore

        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception as e:
        # Best-effort: never fail the pipeline because observability couldn't init.
        LOGGER.warning("pipeline_models: unable to create SparkSession (%s)", e)
        return None


def _get_pipeline_models_schema():
    """Define explicit schema for pipeline_models table to avoid Spark inference issues."""
    try:
        from pyspark.sql.types import (
            StructType,
            StructField,
            StringType,
            TimestampType,
        )

        return StructType(
            [
                StructField("training_run_id", StringType(), nullable=True),
                StructField("institution_id", StringType(), nullable=False),
                StructField(
                    "training_cohort_dataset_name", StringType(), nullable=True
                ),
                StructField(
                    "training_course_dataset_name", StringType(), nullable=True
                ),
                StructField("model_name", StringType(), nullable=False),
                StructField("model_card_path", StringType(), nullable=True),
                StructField("model_version", StringType(), nullable=True),
                StructField("model_run_id", StringType(), nullable=False),
                StructField("summary_metrics", StringType(), nullable=True),
                StructField("payload_json", StringType(), nullable=True),
                StructField("bias_summary", StringType(), nullable=True),
                StructField("logged_ts", TimestampType(), nullable=False),
            ]
        )
    except Exception:
        return None


def _json_dumps(payload: t.Optional[dict[str, t.Any]]) -> t.Optional[str]:
    if payload is None:
        return None
    try:
        return json.dumps(payload, default=str, sort_keys=True)
    except Exception as e:
        return json.dumps({"_error": f"json_dumps_failed: {e}", "_raw": str(payload)})


def _best_effort_fetch_mlflow_run_metrics(*, run_id: str) -> dict[str, float] | None:
    """
    Best-effort fetch of MLflow scalar metrics for a run. Returns None on any error.
    """
    try:
        from mlflow.tracking import MlflowClient  # type: ignore

        client = MlflowClient()
        run = client.get_run(run_id)
        metrics = getattr(getattr(run, "data", None), "metrics", None)
        if not isinstance(metrics, dict):
            return None
        # mypy/typing: coerce keys/values to the expected shapes
        out: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    except Exception as e:
        LOGGER.warning(
            "pipeline_models: failed to fetch MLflow metrics for %s: %s", run_id, e
        )
        return None


def _select_summary_metrics(
    metrics: dict[str, float] | None,
) -> dict[str, float] | None:
    """
    Keep a stable, dashboard-friendly subset of model performance metrics.
    """
    if not metrics:
        return None
    keys = [
        "test_precision",
        "test_recall",
        "test_f1",
        "test_accuracy",
        "test_roc_auc",
        "test_log_loss",
    ]
    out = {k: float(metrics[k]) for k in keys if k in metrics}
    return out or None


def _select_bias_summary(metrics: dict[str, float] | None) -> dict[str, float] | None:
    """
    Keep a stable, dashboard-friendly subset of bias metrics (currently bias scores).
    """
    if not metrics:
        return None
    keys = [
        "test_bias_score_sum",
        "test_bias_score_mean",
        "test_bias_score_max",
        "test_num_bias_flags",
        "test_num_valid_comparisons",
    ]
    out = {k: float(metrics[k]) for k in keys if k in metrics}
    return out or None


def _best_effort_resolve_uc_model_version(
    *,
    catalog: str,
    institution_id: str,
    model_name: str,
    model_run_id: str,
    registry_uri: str = "databricks-uc",
) -> str | None:
    """
    Best-effort resolve of the UC model version associated with a given MLflow run_id.
    Returns the version as a string (to avoid int/str drift across APIs).
    """
    try:
        from mlflow.tracking import MlflowClient  # type: ignore

        client = MlflowClient(registry_uri=registry_uri)
        full_model_name = f"{catalog}.{institution_id}_gold.{model_name}"
        versions = client.search_model_versions(f"name='{full_model_name}'")
        for v in versions or []:
            try:
                if str(getattr(v, "run_id", "")) == str(model_run_id):
                    ver = getattr(v, "version", None)
                    return str(ver) if ver is not None else None
            except Exception:
                continue
        return None
    except Exception as e:
        LOGGER.warning(
            "pipeline_models: failed to resolve model_version for %s/%s/%s (run_id=%s): %s",
            catalog,
            institution_id,
            model_name,
            model_run_id,
            e,
        )
        return None


def upsert_pipeline_model(
    *,
    catalog: t.Optional[str],
    institution_id: t.Optional[str],
    model_name: t.Optional[str],
    model_run_id: t.Optional[str],
    training_run_id: t.Optional[str] = None,
    training_cohort_dataset_name: t.Optional[str] = None,
    training_course_dataset_name: t.Optional[str] = None,
    model_card_path: t.Optional[str] = None,
    summary_metrics: t.Optional[dict[str, float]] = None,
    bias_summary: t.Optional[dict[str, float]] = None,
    payload: t.Optional[dict[str, t.Any]] = None,
    schema: str = "default",
    table: str = "pipeline_models",
) -> bool:
    """
    Upsert a single model row into a UC Delta table.

    Target (default): <catalog>.default.pipeline_models

    Best-effort only: any failure returns False and must not fail pipelines.
    """
    try:
        if not catalog:
            LOGGER.warning("pipeline_models: skipping write because catalog is empty")
            return False
        if not institution_id:
            LOGGER.warning(
                "pipeline_models: skipping write because institution_id is empty"
            )
            return False
        if not model_name:
            LOGGER.warning(
                "pipeline_models: skipping write because model_name is empty"
            )
            return False
        if not model_run_id:
            LOGGER.warning(
                "pipeline_models: skipping write because model_run_id is empty"
            )
            return False

        spark = _get_spark_session()
        if spark is None:
            return False

        table_path = f"{catalog}.{schema}.{table}"

        # Best-effort metrics fetch if caller didn't provide any
        metrics_all = _best_effort_fetch_mlflow_run_metrics(run_id=str(model_run_id))
        summary_metrics2 = (
            summary_metrics
            if summary_metrics is not None
            else _select_summary_metrics(metrics_all)
        )
        bias_summary2 = (
            bias_summary
            if bias_summary is not None
            else _select_bias_summary(metrics_all)
        )

        model_version = _best_effort_resolve_uc_model_version(
            catalog=str(catalog),
            institution_id=str(institution_id),
            model_name=str(model_name),
            model_run_id=str(model_run_id),
        )

        row: dict[str, t.Any] = {
            "training_run_id": training_run_id,
            "institution_id": str(institution_id),
            "training_cohort_dataset_name": training_cohort_dataset_name,
            "training_course_dataset_name": training_course_dataset_name,
            "model_name": str(model_name),
            "model_card_path": model_card_path,
            "model_version": model_version,
            "model_run_id": str(model_run_id),
            "summary_metrics": _json_dumps(summary_metrics2),
            "payload_json": _json_dumps(payload or {}),
            "bias_summary": _json_dumps(bias_summary2),
            "logged_ts": datetime.now(),
        }

        schema = _get_pipeline_models_schema()
        df = (
            spark.createDataFrame([row], schema=schema)
            if schema
            else spark.createDataFrame([row])
        )

        # Prefer an idempotent upsert via Delta merge when available.
        try:
            from delta.tables import DeltaTable  # type: ignore

            try:
                dt = DeltaTable.forName(spark, table_path)
            except Exception:
                (
                    df.write.format("delta")
                    .mode("append")
                    .option("mergeSchema", "true")
                    .saveAsTable(table_path)
                )
                return True

            (
                dt.alias("t")
                .merge(df.alias("s"), "t.model_run_id = s.model_run_id")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
            return True
        except Exception:
            # Fallback: append-only (may create duplicates if called repeatedly).
            (
                df.write.format("delta")
                .mode("append")
                .option("mergeSchema", "true")
                .saveAsTable(table_path)
            )
            return True
    except Exception as e:
        LOGGER.warning(
            "pipeline_models: failed to upsert model row (institution_id=%s, model_name=%s, model_run_id=%s): %s",
            institution_id,
            model_name,
            model_run_id,
            e,
        )
        return False
