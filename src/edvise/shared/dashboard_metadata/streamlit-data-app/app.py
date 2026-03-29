import json
import os
from datetime import date, timedelta
from typing import Any, cast

from databricks import sql as databricks_sql  # type: ignore[attr-defined]
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
import pandas as pd
import streamlit as st

from helpers import (
    FAILURE_SEARCH_COLUMNS,
    INSTITUTION_SEARCH_COLUMNS,
    MODEL_SEARCH_COLUMNS,
    RUN_SEARCH_COLUMNS,
    apply_model_filters,
    apply_run_filters,
    build_day_over_day_metrics,
    build_failures_dataframe,
    build_inspection_labels,
    build_institution_summary,
    build_latest_activity_summary,
    build_models_query,
    build_overview_metrics,
    build_runs_query,
    format_dt,
    get_unique_options,
    has_value,
    prepare_models_dataframe,
    prepare_runs_dataframe,
    search_dataframe,
    sort_dataframe,
)


DEFAULT_DB_WORKSPACE = "dev_sst_02"
DEFAULT_METADATA_SCHEMA = "default"


def get_db_workspace() -> str:
    db_workspace = os.getenv("DB_workspace", DEFAULT_DB_WORKSPACE).strip()
    return db_workspace or DEFAULT_DB_WORKSPACE


RUNS_TABLE = f"{get_db_workspace()}.{DEFAULT_METADATA_SCHEMA}.pipeline_runs"
MODELS_TABLE = f"{get_db_workspace()}.{DEFAULT_METADATA_SCHEMA}.pipeline_models"

HIDE_EXPORT_CSS = """
<style>
div[data-testid="stElementToolbar"] {
    display: none;
}
</style>
"""

RUN_DISPLAY_COLUMNS = [
    "institution_id",
    "run_id",
    "run_type",
    "status",
    "run_ts",
    "started_at",
    "finished_at",
    "duration_seconds",
    "cohort_dataset_name",
    "course_dataset_name",
    "dataset_ts",
    "cohort",
    "term_filter",
    "pipeline_version",
    "model_run_id",
    "experiment_id",
    "model_name",
    "model_version",
    "run_url",
]

MODEL_DISPLAY_COLUMNS = [
    "institution_id",
    "training_run_id",
    "model_run_id",
    "model_name",
    "model_version",
    "training_cohort_dataset_name",
    "training_course_dataset_name",
    "logged_ts",
    "model_card_path",
]

FAILURE_DISPLAY_COLUMNS = [
    "institution_id",
    "run_id",
    "run_type",
    "status",
    "run_ts",
    "pipeline_version",
    "duration_seconds",
    "error_category",
    "error_preview",
    "run_url",
]


st.set_page_config(
    page_title="Pipeline Metadata Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def get_warehouse_id() -> str:
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID must be set in the app configuration."
        )
    return warehouse_id


def run_query(query: str) -> pd.DataFrame:
    cfg = Config()

    with databricks_sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{get_warehouse_id()}",
        credentials_provider=lambda: cfg.authenticate,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


@st.cache_data(ttl=30)
def get_runs_data(start_date_str: str, end_date_exclusive_str: str) -> pd.DataFrame:
    query = build_runs_query(
        RUNS_TABLE, MODELS_TABLE, start_date_str, end_date_exclusive_str
    )
    return prepare_runs_dataframe(run_query(query))


@st.cache_data(ttl=30)
def get_models_data(start_date_str: str, end_date_exclusive_str: str) -> pd.DataFrame:
    query = build_models_query(MODELS_TABLE, start_date_str, end_date_exclusive_str)
    return prepare_models_dataframe(run_query(query))


@st.cache_data(ttl=300)
def resolve_run_link_url(run_id: object | None, fallback_url: object | None) -> str | None:
    fallback = str(fallback_url).strip() if has_value(fallback_url) else None
    if not has_value(run_id):
        return fallback

    try:
        run_id_int = int(str(run_id).strip())
    except (TypeError, ValueError):
        return fallback

    try:
        run = WorkspaceClient(config=Config()).jobs.get_run(run_id=run_id_int)
        run_page_url = getattr(run, "run_page_url", None)
        if has_value(run_page_url):
            return str(run_page_url).strip()
    except Exception:
        return fallback

    return fallback


def render_jsonish(raw_value: object | None) -> None:
    if not has_value(raw_value):
        st.caption("No data")
        return

    text = str(raw_value)
    try:
        st.json(json.loads(text), expanded=False)
    except Exception:
        st.code(text, language="json")


def render_data_table(
    df: pd.DataFrame, columns: list[str], rows_per_table: int, height: int
) -> None:
    st.dataframe(
        df.reindex(columns=columns).head(rows_per_table),
        use_container_width=True,
        height=height,
        hide_index=True,
    )


def render_inspection_selectbox(
    label: str, options: pd.Series, key: str
) -> object | None:
    option_values = [None, *options.index.tolist()]

    selected_option = st.selectbox(
        label,
        options=option_values,
        format_func=lambda idx: "Select an item" if idx is None else options.loc[idx],
        key=key,
    )
    return cast(object | None, selected_option)


def format_count_delta(delta: Any | None) -> str | None:
    if delta is None:
        return None
    return f"{int(delta):+d}"


def format_rate_delta(delta: Any | None) -> str | None:
    if delta is None:
        return None
    return f"{float(delta):+.1f} pts"


def format_day_reference(value: object | None) -> str:
    if pd.isna(value):
        return "No activity"
    return str(pd.to_datetime(value).strftime("%Y-%m-%d"))


def format_delta_reference(
    label: str, current_day: object | None, previous_day: object | None
) -> str:
    if pd.isna(current_day):
        return f"{label}: no activity in selected range"
    return f"{label}: {format_day_reference(current_day)} vs {format_day_reference(previous_day)}"


def display_text(value: object | None) -> str:
    return str(value) if has_value(value) else "—"


def render_latest_update(
    title: str,
    timestamp_label: str,
    timestamp_value: object | None,
    details: list[str],
    link_label: str | None = None,
    link_url: object | None = None,
) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(f"{timestamp_label}: {format_dt(timestamp_value)}")
        for detail in details:
            st.write(detail)
        if has_value(link_url) and link_label:
            st.markdown(f"[{link_label}]({link_url})")


def render_metric_card(
    column: Any,
    label: str,
    value: object,
    delta: object | None = None,
    delta_color: str = "normal",
) -> None:
    with column:
        with st.container(border=True):
            st.metric(label, value, delta=delta, delta_color=delta_color)


def render_run_details(row: pd.Series) -> None:
    st.markdown("### Run details")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Institution", row.get("institution_id", "—"))
    c2.metric("Status", row.get("status", "—"))
    c3.metric("Run type", row.get("run_type", "—"))
    c4.metric(
        "Duration (s)",
        "—"
        if pd.isna(row.get("duration_seconds"))
        else f"{row.get('duration_seconds'):.0f}",
    )

    c5, c6, c7 = st.columns(3)
    c5.metric("Started", format_dt(row.get("started_at")))
    c6.metric("Finished", format_dt(row.get("finished_at")))
    c7.metric("Pipeline version", row.get("pipeline_version", "—"))

    st.write(f"**Run ID:** `{row.get('run_id', '')}`")
    st.write(f"**Model run ID:** `{row.get('model_run_id', '')}`")
    st.write(f"**Experiment ID:** `{row.get('experiment_id', '')}`")
    st.write(f"**Cohort dataset:** {row.get('cohort_dataset_name', '—')}")
    st.write(f"**Course dataset:** {row.get('course_dataset_name', '—')}")
    st.write(f"**Term filter:** {row.get('term_filter', '—')}")

    run_link_url = resolve_run_link_url(row.get("run_id"), row.get("run_url"))
    if has_value(run_link_url):
        st.markdown(f"[Open Databricks run]({run_link_url})")

    if has_value(row.get("model_card_path")):
        st.markdown(f"[Open model card]({row.get('model_card_path')})")

    with st.expander("Error message", expanded=has_value(row.get("error_message"))):
        st.write(row.get("error_message", "No error message"))

    with st.expander("Payload JSON"):
        render_jsonish(row.get("payload_json"))

    with st.expander("Summary metrics"):
        render_jsonish(row.get("summary_metrics"))

    with st.expander("Bias summary"):
        render_jsonish(row.get("bias_summary"))


def render_model_details(row: pd.Series) -> None:
    st.markdown("### Model details")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Institution", row.get("institution_id", "—"))
    c2.metric("Model name", row.get("model_name", "—"))
    c3.metric("Model version", row.get("model_version", "—"))
    c4.metric("Logged at", format_dt(row.get("logged_ts")))

    st.write(f"**Training run ID:** `{row.get('training_run_id', '')}`")
    st.write(f"**Model run ID:** `{row.get('model_run_id', '')}`")
    st.write(
        f"**Training cohort dataset:** {row.get('training_cohort_dataset_name', '—')}"
    )
    st.write(
        f"**Training course dataset:** {row.get('training_course_dataset_name', '—')}"
    )

    if has_value(row.get("model_card_path")):
        st.markdown(f"[Open model card]({row.get('model_card_path')})")

    with st.expander("Payload JSON"):
        render_jsonish(row.get("payload_json"))

    with st.expander("Summary metrics"):
        render_jsonish(row.get("summary_metrics"))

    with st.expander("Bias summary"):
        render_jsonish(row.get("bias_summary"))


def render_overview_tab(
    filtered_runs: pd.DataFrame,
    institution_summary: pd.DataFrame,
    failures_df: pd.DataFrame,
    overview_metrics: dict[str, object],
    day_over_day_metrics: dict[str, object],
    latest_activity_summary: dict[str, dict[str, object] | None],
    rows_per_table: int,
) -> None:
    pulse_top = st.columns(3, gap="large")
    render_metric_card(
        pulse_top[0],
        "Runs",
        overview_metrics["total_runs"],
        delta=format_count_delta(day_over_day_metrics["runs_delta"]),
    )
    render_metric_card(
        pulse_top[1],
        "Success rate",
        f"{overview_metrics['success_rate']}%",
        delta=format_rate_delta(day_over_day_metrics["success_rate_delta"]),
    )
    render_metric_card(
        pulse_top[2],
        "Models logged",
        overview_metrics["models_logged"],
        delta=format_count_delta(day_over_day_metrics["models_delta"]),
    )

    pulse_bottom = st.columns(2, gap="large")
    render_metric_card(
        pulse_bottom[0],
        "Failures",
        overview_metrics["failed_runs"],
        delta=format_count_delta(day_over_day_metrics["failures_delta"]),
        delta_color="inverse",
    )
    render_metric_card(
        pulse_bottom[1],
        "Active institutions",
        overview_metrics["monitored_institutions"],
        delta=format_count_delta(day_over_day_metrics["active_institutions_delta"]),
    )

    latest_run = latest_activity_summary["latest_run"]
    latest_model = latest_activity_summary["latest_model"]
    with st.expander("Recent updates", expanded=False):
        st.caption("Most recent records in the selected range.")
        update1, update2 = st.columns(2, gap="large")

        with update1:
            if latest_run is None:
                st.info("No recent run activity in the selected range.")
            else:
                latest_run_link_url = resolve_run_link_url(
                    latest_run["run_id"], latest_run["run_url"]
                )
                render_latest_update(
                    "Latest run",
                    "Timestamp",
                    latest_run["run_ts"],
                    [
                        f"Institution: {display_text(latest_run['institution_id'])}",
                        f"Run type: {display_text(latest_run['run_type'])}",
                        f"Status: {display_text(latest_run['status'])}",
                        f"Run ID: {display_text(latest_run['run_id'])}",
                    ],
                    link_label="Open Databricks run",
                    link_url=latest_run_link_url,
                )

        with update2:
            if latest_model is None:
                st.info("No recent model activity in the selected range.")
            else:
                render_latest_update(
                    "Latest model log",
                    "Timestamp",
                    latest_model["logged_ts"],
                    [
                        f"Institution: {display_text(latest_model['institution_id'])}",
                        f"Model: {display_text(latest_model['model_name'])}",
                        f"Version: {display_text(latest_model['model_version'])}",
                        f"Model run ID: {display_text(latest_model['model_run_id'])}",
                    ],
                    link_label="Open model card",
                    link_url=latest_model["model_card_path"],
                )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Runs by institution**")
        if filtered_runs.empty:
            st.info("No runs to show.")
        else:
            runs_by_inst = (
                filtered_runs.groupby("institution_id")
                .size()
                .rename("Runs")
                .sort_values(ascending=False)
                .head(15)
                .rename_axis("Institution")
                .reset_index()
            )
            st.bar_chart(runs_by_inst, x="Institution", y="Runs", height=300)

    with c2:
        st.markdown("**Training vs inference**")
        split_df = pd.DataFrame(
            {
                "Run type": ["training", "inference"],
                "Runs": [
                    overview_metrics["training_runs"],
                    overview_metrics["inference_runs"],
                ],
            },
        )
        st.bar_chart(split_df, x="Run type", y="Runs", height=300)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Runs over time**")
        if filtered_runs.empty:
            st.info("No run data in the selected range.")
        else:
            runs_over_time = (
                filtered_runs.dropna(subset=["run_ts"])
                .assign(run_day=lambda d: d["run_ts"].dt.date)
                .groupby("run_day")
                .size()
                .rename("Runs")
                .rename_axis("Date")
                .reset_index()
            )
            st.line_chart(runs_over_time, x="Date", y="Runs", height=300)

    with c4:
        st.markdown("**Failures by institution**")
        if failures_df.empty:
            st.info("No failures in the selected range.")
        else:
            failures_by_inst = (
                failures_df.groupby("institution_id")
                .size()
                .rename("Failures")
                .sort_values(ascending=False)
                .head(15)
                .rename_axis("Institution")
                .reset_index()
            )
            st.bar_chart(
                failures_by_inst, x="Institution", y="Failures", height=300
            )

    recent_runs = sort_dataframe(filtered_runs.copy(), "run_ts", True)
    render_data_table(
        recent_runs,
        [
            "institution_id",
            "run_id",
            "run_type",
            "status",
            "run_ts",
            "duration_seconds",
            "pipeline_version",
            "model_name",
            "model_version",
        ],
        rows_per_table=min(rows_per_table, 20),
        height=320,
    )

def render_institutions_tab(
    institution_summary: pd.DataFrame, rows_per_table: int
) -> None:
    st.caption(
        "Best starting point for users who want to see which institutions are healthy, need help, or require follow-up."
    )

    inst_search = st.text_input(
        "Search institutions",
        placeholder="institution id, latest run status, model version...",
        key="inst_search",
    )

    inst_table = search_dataframe(
        institution_summary.copy(), inst_search, INSTITUTION_SEARCH_COLUMNS
    )

    latest_status_options = get_unique_options(inst_table, "latest_run_status")
    selected_latest_status = st.multiselect(
        "Latest run status", latest_status_options, key="selected_latest_status"
    )

    if selected_latest_status:
        inst_table = inst_table[
            inst_table["latest_run_status"].astype(str).isin(selected_latest_status)
        ]

    inst_table = inst_table.sort_values(
        by=["failed_runs", "date_last_run", "institution_id"],
        ascending=[False, False, True],
        na_position="last",
    )
    render_data_table(
        inst_table, inst_table.columns.tolist(), rows_per_table, height=520
    )


def render_runs_tab(filtered_runs: pd.DataFrame, rows_per_table: int) -> None:
    st.caption("Use this view for run-level debugging, lineage, and execution detail.")

    runs_search = st.text_input(
        "Search runs",
        placeholder="run id, experiment id, dataset name, error text...",
        key="runs_search",
    )

    run_table = search_dataframe(filtered_runs.copy(), runs_search, RUN_SEARCH_COLUMNS)
    render_data_table(run_table, RUN_DISPLAY_COLUMNS, rows_per_table, height=520)

    inspect_run_table = sort_dataframe(run_table.copy(), "run_ts", True)
    available_run_dates = sorted(
        inspect_run_table["run_ts"].dropna().dt.date.unique().tolist()
    )

    selected_run_date: object | None = None
    if available_run_dates:
        selected_run_date = st.date_input(
            "Run date (optional)",
            value=None,
            min_value=available_run_dates[0],
            max_value=available_run_dates[-1],
            key="selected_run_date",
        )
        if selected_run_date is not None:
            inspect_run_table = inspect_run_table[
                inspect_run_table["run_ts"].dt.date == selected_run_date
            ]
            if inspect_run_table.empty:
                st.info("No runs found for the selected date.")

    inspection_labels = build_inspection_labels(
        inspect_run_table, ["institution_id", "run_id"]
    )
    selected_run_index = render_inspection_selectbox(
        "Inspect a run", inspection_labels, "selected_run_index"
    )
    if selected_run_index is not None:
        render_run_details(inspect_run_table.loc[selected_run_index])


def render_models_tab(filtered_models: pd.DataFrame, rows_per_table: int) -> None:
    st.caption(
        "Use this view to trace model versions, training sources, and model artifacts."
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        model_name_options = get_unique_options(filtered_models, "model_name")
        selected_model_names = st.multiselect(
            "Model name", model_name_options, key="selected_model_names"
        )
    with mc2:
        model_version_options = get_unique_options(filtered_models, "model_version")
        selected_model_versions = st.multiselect(
            "Model version", model_version_options, key="selected_model_versions"
        )

    models_search = st.text_input(
        "Search models",
        placeholder="model name, version, training run id, model run id...",
        key="models_search",
    )

    model_table = filtered_models.copy()

    if selected_model_names:
        model_table = model_table[
            model_table["model_name"].astype(str).isin(selected_model_names)
        ]

    if selected_model_versions:
        model_table = model_table[
            model_table["model_version"].astype(str).isin(selected_model_versions)
        ]

    model_table = search_dataframe(model_table, models_search, MODEL_SEARCH_COLUMNS)

    model_sort_options = [
        "logged_ts",
        "institution_id",
        "model_name",
        "model_version",
        "training_run_id",
        "model_run_id",
    ]
    st.markdown("**Sort models**")
    mc3, mc4 = st.columns([3.4, 5.6], gap="small")
    with mc3:
        models_sort_by = st.selectbox(
            "Sort models by",
            model_sort_options,
            index=0,
            label_visibility="collapsed",
        )
    with mc4:
        st.empty()

    model_table = sort_dataframe(model_table, models_sort_by, True)
    render_data_table(model_table, MODEL_DISPLAY_COLUMNS, rows_per_table, height=520)

    inspection_labels = build_inspection_labels(
        model_table,
        ["institution_id", "model_name", "model_version", "model_run_id"],
    )
    selected_model_index = render_inspection_selectbox(
        "Inspect a model", inspection_labels, "selected_model_index"
    )
    if selected_model_index is not None:
        render_model_details(model_table.loc[selected_model_index])


def render_failures_tab(failures_df: pd.DataFrame, rows_per_table: int) -> None:
    st.caption(
        "Use this review failures, identify repeated issues, and inspect the raw errors."
    )

    failure_search = st.text_input(
        "Search failures",
        placeholder="error text, run id, pipeline version...",
        key="failure_search",
    )

    failure_table = failures_df.copy()
    failure_category_options = get_unique_options(failure_table, "error_category")
    selected_error_categories = st.multiselect(
        "Error category",
        failure_category_options,
        key="selected_error_categories",
    )

    if selected_error_categories:
        failure_table = failure_table[
            failure_table["error_category"].astype(str).isin(selected_error_categories)
        ]

    failure_table = search_dataframe(
        failure_table, failure_search, FAILURE_SEARCH_COLUMNS
    )

    fc1, fc2 = st.columns([3, 1])
    with fc1:
        failure_sort_by = st.selectbox(
            "Sort failures by",
            [
                "run_ts",
                "institution_id",
                "pipeline_version",
                "error_category",
                "duration_seconds",
            ],
            index=0,
        )
    with fc2:
        st.empty()

    failure_table = sort_dataframe(failure_table, failure_sort_by, True)
    render_data_table(
        failure_table, FAILURE_DISPLAY_COLUMNS, rows_per_table, height=520
    )

    if failure_table.empty:
        return

    st.markdown("#### Failure breakdown")
    ff1, ff2 = st.columns(2)

    with ff1:
        failure_counts = (
            failure_table.groupby("error_category")
            .size()
            .rename("Failures")
            .sort_values(ascending=False)
            .rename_axis("Error category")
            .reset_index()
        )
        st.bar_chart(failure_counts, x="Error category", y="Failures", height=280)

    with ff2:
        failure_by_inst = (
            failure_table.groupby("institution_id")
            .size()
            .rename("Failures")
            .sort_values(ascending=False)
            .head(15)
            .rename_axis("Institution")
            .reset_index()
        )
        st.bar_chart(failure_by_inst, x="Institution", y="Failures", height=280)

    inspect_failure_table = sort_dataframe(failure_table.copy(), "run_ts", True)
    available_failure_dates = sorted(
        inspect_failure_table["run_ts"].dropna().dt.date.unique().tolist()
    )

    selected_failure_date: object | None = None
    if available_failure_dates:
        selected_failure_date = st.date_input(
            "Failure date (optional)",
            value=None,
            min_value=available_failure_dates[0],
            max_value=available_failure_dates[-1],
            key="selected_failure_date",
        )
        if selected_failure_date is not None:
            inspect_failure_table = inspect_failure_table[
                inspect_failure_table["run_ts"].dt.date == selected_failure_date
            ]
            if inspect_failure_table.empty:
                st.info("No failed runs found for the selected date.")

    inspection_labels = build_inspection_labels(
        inspect_failure_table, ["institution_id", "run_id"]
    )
    selected_failure_index = render_inspection_selectbox(
        "Inspect a failed run",
        inspection_labels,
        "selected_failure_index",
    )
    if selected_failure_index is not None:
        render_run_details(inspect_failure_table.loc[selected_failure_index])


def main() -> None:
    st.markdown(HIDE_EXPORT_CSS, unsafe_allow_html=True)

    st.title("Pipeline Metadata Dashboard")

    with st.expander("How to use this dashboard"):
        st.markdown(
            """
            - **Overview** gives the fastest health check across institutions and recent activity.
            - **Institution health** is the best place to see who needs attention in plain language.
            - **Run diagnostics**, **Model registry**, and **Failures** support deeper technical investigation.
            """
        )

    st.sidebar.header("Global filters")
    st.sidebar.caption(
        "Filters update every view. Tables are view-only and data export is disabled."
    )
    st.sidebar.caption(f"Runs table: `{RUNS_TABLE}`")
    st.sidebar.caption(f"Models table: `{MODELS_TABLE}`")

    today = date.today()
    default_start = today - timedelta(days=90)
    start_date = st.sidebar.date_input("Start date", value=default_start)
    end_date = st.sidebar.date_input("End date", value=today)

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        st.stop()

    end_date_exclusive = end_date + timedelta(days=1)

    try:
        raw_runs = get_runs_data(start_date.isoformat(), end_date_exclusive.isoformat())
        raw_models = get_models_data(
            start_date.isoformat(), end_date_exclusive.isoformat()
        )
    except Exception as exc:
        st.error(f"Could not load the metadata tables: {exc}")
        st.stop()

    institution_options = sorted(
        set(get_unique_options(raw_runs, "institution_id")).union(
            get_unique_options(raw_models, "institution_id")
        )
    )
    run_type_options = get_unique_options(raw_runs, "run_type")
    status_options = get_unique_options(raw_runs, "status")
    pipeline_version_options = get_unique_options(raw_runs, "pipeline_version")

    selected_institutions = st.sidebar.multiselect("Institution", institution_options)
    selected_run_types = st.sidebar.multiselect("Run type", run_type_options)
    selected_statuses = st.sidebar.multiselect("Status", status_options)
    selected_pipeline_versions = st.sidebar.multiselect(
        "Pipeline version", pipeline_version_options
    )
    rows_per_table = st.sidebar.slider(
        "Rows shown per table", min_value=25, max_value=500, value=200, step=25
    )

    filtered_runs = apply_run_filters(
        raw_runs,
        institutions=selected_institutions,
        run_types=selected_run_types,
        statuses=selected_statuses,
        pipeline_versions=selected_pipeline_versions,
    )
    filtered_models = apply_model_filters(
        raw_models,
        institutions=selected_institutions,
    )

    institution_summary = build_institution_summary(filtered_runs, filtered_models)
    failures_df = build_failures_dataframe(filtered_runs)
    overview_metrics = build_overview_metrics(
        filtered_runs, filtered_models, institution_summary
    )
    day_over_day_metrics = build_day_over_day_metrics(filtered_runs, filtered_models)
    latest_activity_summary = build_latest_activity_summary(
        filtered_runs, filtered_models
    )

    tab_overview, tab_institutions, tab_models, tab_failures, tab_runs = st.tabs(
        [
            "Overview",
            "Institution Overview",
            "Model registry",
            "Failures",
            "Run diagnostics",
        ]
    )

    with tab_overview:
        render_overview_tab(
            filtered_runs,
            institution_summary,
            failures_df,
            overview_metrics,
            day_over_day_metrics,
            latest_activity_summary,
            rows_per_table,
        )

    with tab_institutions:
        render_institutions_tab(institution_summary, rows_per_table)

    with tab_models:
        render_models_tab(filtered_models, rows_per_table)

    with tab_failures:
        render_failures_tab(failures_df, rows_per_table)

    with tab_runs:
        render_runs_tab(filtered_runs, rows_per_table)


main()
