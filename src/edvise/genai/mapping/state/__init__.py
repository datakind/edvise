"""
Pipeline state: Delta tables under ``{catalog}.genai_mapping`` for run / phase / HITL tracking.
"""

from edvise.genai.mapping.state.pipeline_state import (
    ExecuteRunBootstrap,
    bootstrap_execute_run,
    check_hitl_resolution,
    count_pipeline_runs_created_today,
    create_execute_pipeline_run,
    create_pipeline_run,
    upsert_onboard_pipeline_run_row,
    get_latest_pipeline_run,
    get_latest_pipeline_run_created_today,
    log_phase_transition,
    reconcile_stale_nonterminal_pipeline_runs,
    register_hitl_artifacts,
    resolve_hitl,
    resolve_onboard_run_id,
    update_execute_pipeline_run_input_file_paths,
    update_execute_pipeline_run_status,
    update_onboard_pipeline_run_input_file_paths,
    update_pipeline_run_status,
)
from edvise.genai.mapping.state.table_setup import create_state_tables

__all__ = [
    "ExecuteRunBootstrap",
    "bootstrap_execute_run",
    "check_hitl_resolution",
    "count_pipeline_runs_created_today",
    "create_execute_pipeline_run",
    "create_pipeline_run",
    "upsert_onboard_pipeline_run_row",
    "create_state_tables",
    "get_latest_pipeline_run",
    "get_latest_pipeline_run_created_today",
    "log_phase_transition",
    "reconcile_stale_nonterminal_pipeline_runs",
    "register_hitl_artifacts",
    "resolve_hitl",
    "resolve_onboard_run_id",
    "update_execute_pipeline_run_input_file_paths",
    "update_execute_pipeline_run_status",
    "update_onboard_pipeline_run_input_file_paths",
    "update_pipeline_run_status",
]
