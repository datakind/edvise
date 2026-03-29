from __future__ import annotations

from typing import TypeAlias

import pandas as pd


RUN_SEARCH_COLUMNS = [
    "institution_id",
    "run_id",
    "run_type",
    "status",
    "cohort_dataset_name",
    "course_dataset_name",
    "cohort",
    "term_filter",
    "model_run_id",
    "experiment_id",
    "pipeline_version",
    "error_message",
    "model_name",
    "model_version",
]

MODEL_SEARCH_COLUMNS = [
    "institution_id",
    "training_run_id",
    "model_run_id",
    "model_name",
    "model_version",
    "training_cohort_dataset_name",
    "training_course_dataset_name",
    "model_card_path",
    "summary_metrics",
    "bias_summary",
]

INSTITUTION_SEARCH_COLUMNS = [
    "institution_id",
    "latest_run_status",
    "latest_model_name",
    "latest_model_version",
    "freshness_status",
]

FAILURE_SEARCH_COLUMNS = [
    "institution_id",
    "run_id",
    "run_type",
    "pipeline_version",
    "error_category",
    "error_message",
]

LatestActivityRecord: TypeAlias = dict[str, object]


def build_runs_query(
    runs_table: str, models_table: str, start_date_str: str, end_date_exclusive_str: str
) -> str:
    return f"""
    WITH latest_model_per_run AS (
        SELECT *
        FROM (
            SELECT
                pm.*,
                ROW_NUMBER() OVER (
                    PARTITION BY pm.institution_id, pm.training_run_id
                    ORDER BY pm.logged_ts DESC
                ) AS rn
            FROM {models_table} pm
        ) ranked
        WHERE rn = 1
    )
    SELECT
        pr.institution_id,
        pr.run_id,
        pr.run_type,
        pr.status,
        pr.started_at,
        pr.finished_at,
        pr.updated_at,
        COALESCE(pr.started_at, pr.updated_at, pr.finished_at) AS run_ts,
        pr.run_url,
        pr.cohort_dataset_name,
        pr.course_dataset_name,
        pr.dataset_ts,
        pr.cohort,
        pr.term_filter,
        pr.model_run_id,
        pr.experiment_id,
        pr.pipeline_version,
        pr.error_message,
        pr.payload_json,
        CASE
            WHEN pr.started_at IS NOT NULL AND pr.finished_at IS NOT NULL
            THEN unix_timestamp(pr.finished_at) - unix_timestamp(pr.started_at)
            ELSE NULL
        END AS duration_seconds,
        CASE
            WHEN lower(COALESCE(pr.status, '')) IN ('failed', 'error')
            THEN TRUE ELSE FALSE
        END AS is_failed,
        CASE
            WHEN lower(COALESCE(pr.run_type, '')) = 'training'
            THEN TRUE ELSE FALSE
        END AS is_training_run,
        CASE
            WHEN lower(COALESCE(pr.run_type, '')) = 'inference'
            THEN TRUE ELSE FALSE
        END AS is_inference_run,
        lm.model_name,
        lm.model_version,
        lm.model_card_path,
        lm.summary_metrics,
        lm.bias_summary,
        lm.logged_ts AS model_logged_ts
    FROM {runs_table} pr
    LEFT JOIN latest_model_per_run lm
        ON pr.institution_id = lm.institution_id
       AND pr.run_id = lm.training_run_id
    WHERE COALESCE(pr.started_at, pr.updated_at, pr.finished_at) >= timestamp('{start_date_str} 00:00:00')
      AND COALESCE(pr.started_at, pr.updated_at, pr.finished_at) < timestamp('{end_date_exclusive_str} 00:00:00')
    """


def build_models_query(
    models_table: str, start_date_str: str, end_date_exclusive_str: str
) -> str:
    return f"""
    SELECT
        institution_id,
        training_run_id,
        training_cohort_dataset_name,
        training_course_dataset_name,
        model_card_path,
        payload_json,
        bias_summary,
        summary_metrics,
        model_version,
        model_run_id,
        model_name,
        logged_ts
    FROM {models_table}
    WHERE logged_ts >= timestamp('{start_date_str} 00:00:00')
      AND logged_ts < timestamp('{end_date_exclusive_str} 00:00:00')
    """


def has_value(value: object | None) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return str(value).strip() != ""


def to_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(
                out[col], errors="coerce", utc=True
            ).dt.tz_convert(None)
    return out


def prepare_runs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = to_datetime_columns(
        df,
        [
            "run_ts",
            "started_at",
            "finished_at",
            "updated_at",
            "dataset_ts",
            "model_logged_ts",
        ],
    )

    if "duration_seconds" in out.columns:
        out["duration_seconds"] = pd.to_numeric(
            out["duration_seconds"], errors="coerce"
        )

    for col in ["is_failed", "is_training_run", "is_inference_run"]:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)

    for col in ["status", "run_type"]:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str)

    return out


def prepare_models_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = to_datetime_columns(df, ["logged_ts"])

    for col in [
        "institution_id",
        "training_run_id",
        "model_run_id",
        "model_name",
        "model_version",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)

    return out


def get_unique_options(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def search_dataframe(df: pd.DataFrame, term: str, columns: list[str]) -> pd.DataFrame:
    if df.empty or not term or not term.strip():
        return df

    search_term = term.strip().lower()
    mask = pd.Series(False, index=df.index)

    for col in columns:
        if col in df.columns:
            mask = mask | df[col].fillna("").astype(str).str.lower().str.contains(
                search_term, na=False, regex=False
            )

    return df[mask]


def sort_dataframe(df: pd.DataFrame, sort_by: str, descending: bool) -> pd.DataFrame:
    if df.empty or sort_by not in df.columns:
        return df
    return df.sort_values(by=sort_by, ascending=not descending, na_position="last")


def truncate_text(value: object | None, length: int = 140) -> str:
    if not has_value(value):
        return ""
    text = str(value)
    return text if len(text) <= length else text[:length] + "..."


def format_dt(value: object | None) -> str:
    if not has_value(value):
        return "—"
    return str(pd.to_datetime(value).strftime("%Y-%m-%d %H:%M:%S"))


def categorize_error(message: object | None) -> str:
    if not has_value(message):
        return "Unknown"

    text = str(message).lower()

    if any(k in text for k in ["schema", "pandera", "validation", "column", "dtype"]):
        return "Schema / validation"
    if any(
        k in text
        for k in [
            "missing",
            "not found",
            "no such file",
            "does not exist",
            "path",
            "volume",
        ]
    ):
        return "Missing input / path"
    if any(
        k in text
        for k in ["permission", "denied", "unauthorized", "forbidden", "credential"]
    ):
        return "Permissions / auth"
    if any(k in text for k in ["timeout", "timed out", "connection", "socket"]):
        return "Timeout / connectivity"
    if any(k in text for k in ["oom", "out of memory", "memory"]):
        return "Resource / memory"
    if any(k in text for k in ["train", "fit", "model"]):
        return "Training / model"
    if any(k in text for k in ["infer", "predict", "score"]):
        return "Inference / scoring"
    if any(k in text for k in ["write", "append", "delta", "save", "insert"]):
        return "Write / logging"

    return "Other"


def get_freshness_status(days_since_last_run: int | float | None) -> str:
    if days_since_last_run is None or pd.isna(days_since_last_run):
        return "No activity"
    if days_since_last_run <= 1:
        return "Healthy"
    if days_since_last_run <= 7:
        return "Warning"
    return "Stale"


def apply_run_filters(
    df: pd.DataFrame,
    institutions: list[str],
    run_types: list[str],
    statuses: list[str],
    pipeline_versions: list[str],
    global_search: str,
) -> pd.DataFrame:
    out = df.copy()

    if institutions:
        out = out[out["institution_id"].astype(str).isin(institutions)]

    if run_types:
        out = out[out["run_type"].astype(str).isin(run_types)]

    if statuses:
        out = out[out["status"].astype(str).isin(statuses)]

    if pipeline_versions:
        out = out[
            out["pipeline_version"].fillna("").astype(str).isin(pipeline_versions)
        ]

    return search_dataframe(out, global_search, RUN_SEARCH_COLUMNS)


def apply_model_filters(
    df: pd.DataFrame, institutions: list[str], global_search: str
) -> pd.DataFrame:
    out = df.copy()

    if institutions:
        out = out[out["institution_id"].astype(str).isin(institutions)]

    return search_dataframe(out, global_search, MODEL_SEARCH_COLUMNS)


def build_institution_summary(
    runs_df: pd.DataFrame, models_df: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "institution_id",
        "total_runs",
        "training_runs",
        "inference_runs",
        "failed_runs",
        "success_rate",
        "median_duration_seconds",
        "latest_run_at",
        "latest_run_status",
        "latest_successful_run_at",
        "latest_training_run_at",
        "latest_inference_run_at",
        "latest_dataset_ts",
        "latest_model_name",
        "latest_model_version",
        "latest_model_logged_at",
        "days_since_last_run",
        "freshness_status",
    ]

    if runs_df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        runs_df.groupby("institution_id", dropna=False)
        .agg(
            total_runs=("run_id", "count"),
            training_runs=("is_training_run", "sum"),
            inference_runs=("is_inference_run", "sum"),
            failed_runs=("is_failed", "sum"),
            median_duration_seconds=("duration_seconds", "median"),
            latest_run_at=("run_ts", "max"),
            latest_dataset_ts=("dataset_ts", "max"),
        )
        .reset_index()
    )

    summary["success_rate"] = (
        ((summary["total_runs"] - summary["failed_runs"]) / summary["total_runs"]) * 100
    ).round(1)
    summary["median_duration_seconds"] = summary["median_duration_seconds"].round(1)

    latest_run_status = (
        runs_df.sort_values("run_ts")
        .groupby("institution_id", dropna=False)
        .tail(1)[["institution_id", "status"]]
        .rename(columns={"status": "latest_run_status"})
    )

    latest_successful = (
        runs_df[~runs_df["is_failed"]]
        .groupby("institution_id", dropna=False)["run_ts"]
        .max()
        .rename("latest_successful_run_at")
        .reset_index()
    )

    latest_training = (
        runs_df[runs_df["is_training_run"]]
        .groupby("institution_id", dropna=False)["run_ts"]
        .max()
        .rename("latest_training_run_at")
        .reset_index()
    )

    latest_inference = (
        runs_df[runs_df["is_inference_run"]]
        .groupby("institution_id", dropna=False)["run_ts"]
        .max()
        .rename("latest_inference_run_at")
        .reset_index()
    )

    summary = summary.merge(latest_run_status, on="institution_id", how="left")
    summary = summary.merge(latest_successful, on="institution_id", how="left")
    summary = summary.merge(latest_training, on="institution_id", how="left")
    summary = summary.merge(latest_inference, on="institution_id", how="left")

    if not models_df.empty:
        latest_models = (
            models_df.sort_values("logged_ts")
            .groupby("institution_id", dropna=False)
            .tail(1)[["institution_id", "model_name", "model_version", "logged_ts"]]
            .rename(
                columns={
                    "model_name": "latest_model_name",
                    "model_version": "latest_model_version",
                    "logged_ts": "latest_model_logged_at",
                }
            )
        )
        summary = summary.merge(latest_models, on="institution_id", how="left")
    else:
        summary["latest_model_name"] = None
        summary["latest_model_version"] = None
        summary["latest_model_logged_at"] = pd.NaT

    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    summary["days_since_last_run"] = (now - summary["latest_run_at"]).dt.days
    summary["freshness_status"] = summary["days_since_last_run"].apply(
        get_freshness_status
    )

    return summary[columns].sort_values("institution_id")


def build_failures_dataframe(runs_df: pd.DataFrame) -> pd.DataFrame:
    failures_df = runs_df[runs_df["is_failed"]].copy()

    if failures_df.empty:
        failures_df["error_category"] = pd.Series(dtype="object")
        failures_df["error_preview"] = pd.Series(dtype="object")
        return failures_df

    failures_df["error_category"] = failures_df["error_message"].apply(categorize_error)
    failures_df["error_preview"] = failures_df["error_message"].apply(truncate_text)
    return failures_df


def build_inspection_labels(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")
    return df.loc[:, columns].fillna("").astype(str).agg(" | ".join, axis=1)


def build_overview_metrics(
    filtered_runs: pd.DataFrame,
    filtered_models: pd.DataFrame,
    institution_summary: pd.DataFrame,
) -> dict[str, object]:
    total_runs = len(filtered_runs)
    failed_runs = (
        int(filtered_runs["is_failed"].sum()) if not filtered_runs.empty else 0
    )
    training_runs = (
        int(filtered_runs["is_training_run"].sum()) if not filtered_runs.empty else 0
    )
    inference_runs = (
        int(filtered_runs["is_inference_run"].sum()) if not filtered_runs.empty else 0
    )

    return {
        "monitored_institutions": int(institution_summary["institution_id"].nunique())
        if not institution_summary.empty
        else 0,
        "needs_attention": int(
            institution_summary["freshness_status"]
            .isin(["Warning", "Stale", "No activity"])
            .sum()
        )
        if not institution_summary.empty
        else 0,
        "healthy_institutions": int(
            (institution_summary["freshness_status"] == "Healthy").sum()
        )
        if not institution_summary.empty
        else 0,
        "total_runs": total_runs,
        "failed_runs": failed_runs,
        "training_runs": training_runs,
        "inference_runs": inference_runs,
        "success_rate": round(((total_runs - failed_runs) / total_runs) * 100, 1)
        if total_runs
        else 0.0,
        "models_logged": len(filtered_models),
        "latest_run_at": filtered_runs["run_ts"].max()
        if not filtered_runs.empty
        else pd.NaT,
        "latest_model_at": filtered_models["logged_ts"].max()
        if not filtered_models.empty
        else pd.NaT,
    }


def build_day_over_day_metrics(
    filtered_runs: pd.DataFrame, filtered_models: pd.DataFrame
) -> dict[str, object]:
    metrics = {
        "run_reference_day": pd.NaT,
        "previous_run_day": pd.NaT,
        "runs_on_latest_day": 0,
        "runs_delta": None,
        "failures_on_latest_day": 0,
        "failures_delta": None,
        "active_institutions_on_latest_day": 0,
        "active_institutions_delta": None,
        "success_rate_on_latest_day": None,
        "success_rate_delta": None,
        "model_reference_day": pd.NaT,
        "previous_model_day": pd.NaT,
        "models_on_latest_day": 0,
        "models_delta": None,
    }

    if not filtered_runs.empty:
        valid_runs = filtered_runs.dropna(subset=["run_ts"]).copy()
        if not valid_runs.empty:
            daily_runs = (
                valid_runs.assign(activity_day=valid_runs["run_ts"].dt.floor("D"))
                .groupby("activity_day")
                .agg(
                    runs_on_latest_day=("run_id", "count"),
                    failures_on_latest_day=("is_failed", "sum"),
                    active_institutions_on_latest_day=(
                        "institution_id",
                        pd.Series.nunique,
                    ),
                )
                .sort_index()
            )
            daily_runs["success_rate_on_latest_day"] = (
                (
                    (
                        daily_runs["runs_on_latest_day"]
                        - daily_runs["failures_on_latest_day"]
                    )
                    / daily_runs["runs_on_latest_day"]
                )
                * 100
            ).round(1)

            latest_day = daily_runs.index.max()
            previous_day = latest_day - pd.Timedelta(days=1)
            latest_row = daily_runs.loc[latest_day]
            previous_row = (
                daily_runs.loc[previous_day]
                if previous_day in daily_runs.index
                else pd.Series(
                    {
                        "runs_on_latest_day": 0,
                        "failures_on_latest_day": 0,
                        "active_institutions_on_latest_day": 0,
                        "success_rate_on_latest_day": 0.0,
                    }
                )
            )

            metrics.update(
                {
                    "run_reference_day": latest_day,
                    "previous_run_day": previous_day,
                    "runs_on_latest_day": int(latest_row["runs_on_latest_day"]),
                    "runs_delta": int(
                        latest_row["runs_on_latest_day"]
                        - previous_row["runs_on_latest_day"]
                    ),
                    "failures_on_latest_day": int(latest_row["failures_on_latest_day"]),
                    "failures_delta": int(
                        latest_row["failures_on_latest_day"]
                        - previous_row["failures_on_latest_day"]
                    ),
                    "active_institutions_on_latest_day": int(
                        latest_row["active_institutions_on_latest_day"]
                    ),
                    "active_institutions_delta": int(
                        latest_row["active_institutions_on_latest_day"]
                        - previous_row["active_institutions_on_latest_day"]
                    ),
                    "success_rate_on_latest_day": float(
                        latest_row["success_rate_on_latest_day"]
                    ),
                    "success_rate_delta": round(
                        float(
                            latest_row["success_rate_on_latest_day"]
                            - previous_row["success_rate_on_latest_day"]
                        ),
                        1,
                    ),
                }
            )

    if not filtered_models.empty:
        valid_models = filtered_models.dropna(subset=["logged_ts"]).copy()
        if not valid_models.empty:
            daily_models = (
                valid_models.assign(
                    activity_day=valid_models["logged_ts"].dt.floor("D")
                )
                .groupby("activity_day")
                .size()
                .rename("models_on_latest_day")
                .sort_index()
            )

            latest_day = daily_models.index.max()
            previous_day = latest_day - pd.Timedelta(days=1)
            previous_count = (
                int(daily_models.loc[previous_day])
                if previous_day in daily_models.index
                else 0
            )

            metrics.update(
                {
                    "model_reference_day": latest_day,
                    "previous_model_day": previous_day,
                    "models_on_latest_day": int(daily_models.loc[latest_day]),
                    "models_delta": int(daily_models.loc[latest_day] - previous_count),
                }
            )

    return metrics


def build_latest_activity_summary(
    filtered_runs: pd.DataFrame, filtered_models: pd.DataFrame
) -> dict[str, LatestActivityRecord | None]:
    summary: dict[str, LatestActivityRecord | None] = {
        "latest_run": None,
        "latest_model": None,
    }

    if not filtered_runs.empty:
        valid_runs = filtered_runs.dropna(subset=["run_ts"]).sort_values("run_ts")
        if not valid_runs.empty:
            latest_run = valid_runs.iloc[-1]
            summary["latest_run"] = {
                "run_ts": latest_run.get("run_ts"),
                "institution_id": latest_run.get("institution_id"),
                "status": latest_run.get("status"),
                "run_type": latest_run.get("run_type"),
                "run_id": latest_run.get("run_id"),
                "pipeline_version": latest_run.get("pipeline_version"),
                "run_url": latest_run.get("run_url"),
            }

    if not filtered_models.empty:
        valid_models = filtered_models.dropna(subset=["logged_ts"]).sort_values(
            "logged_ts"
        )
        if not valid_models.empty:
            latest_model = valid_models.iloc[-1]
            summary["latest_model"] = {
                "logged_ts": latest_model.get("logged_ts"),
                "institution_id": latest_model.get("institution_id"),
                "model_name": latest_model.get("model_name"),
                "model_version": latest_model.get("model_version"),
                "model_run_id": latest_model.get("model_run_id"),
                "model_card_path": latest_model.get("model_card_path"),
            }

    return summary


def build_attention_table(institution_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "institution_id",
        "freshness_status",
        "latest_run_status",
        "days_since_last_run",
        "failed_runs",
        "latest_model_name",
        "latest_model_version",
    ]

    if institution_summary.empty:
        return pd.DataFrame(columns=columns)

    latest_status = (
        institution_summary["latest_run_status"].fillna("").astype(str).str.lower()
    )
    attention_mask = institution_summary["freshness_status"].isin(
        ["Warning", "Stale", "No activity"]
    ) | latest_status.isin(["failed", "error"])

    if not attention_mask.any():
        return pd.DataFrame(columns=columns)

    return institution_summary.loc[attention_mask, columns].sort_values(
        by=["failed_runs", "days_since_last_run"],
        ascending=[False, False],
        na_position="last",
    )
