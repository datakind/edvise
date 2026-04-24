"""
Shared CLI helpers for :mod:`edvise.scripts.es_data_audit` and
:mod:`edvise.scripts.pdp_data_audit` entry points.

Reduces duplicate argparse, Databricks run-path inference, training run events, and log flush.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import typing as t

from edvise.shared.dashboard_metadata.pipeline_runs import append_pipeline_run_event

__all__ = [
    "apply_bronze_training_inputs_sys_path",
    "flush_data_audit_logging",
    "infer_databricks_institution_name",
    "parse_data_audit_args",
    "run_data_audit_with_training_events",
    "training_cohort_json_for_pipeline_event",
]


def parse_data_audit_args() -> argparse.Namespace:
    """
    Argparse for ES and PDP data audit jobs (identical in both former scripts).
    """
    parser = argparse.ArgumentParser(
        description="Data preprocessing for inference in the SST pipeline."
    )
    parser.add_argument(
        "--course_dataset_validated_path",
        required=False,
        help="Name of the course data file during inference with GCS blobs when connected to webapp",
    )
    parser.add_argument(
        "--cohort_dataset_validated_path",
        required=False,
        help="Name of the cohort data file during inference with GCS blobs when connected to webapp",
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--bronze_volume_path", type=str, required=False)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--DB_workspace", type=str, required=True)
    parser.add_argument("--job_type", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    return parser.parse_args()


def apply_bronze_training_inputs_sys_path(args: argparse.Namespace) -> None:
    """If ``--bronze_volume_path`` is set, append its ``training_inputs`` subdir to ``sys.path``."""
    if args.bronze_volume_path:
        sys.path.append(f"{args.bronze_volume_path}/training_inputs")


def infer_databricks_institution_name(silver_volume_path: str) -> t.Optional[str]:
    """
    Best-effort: infer ``databricks_institution_name`` from a path segment ending in
    ``_silver``, e.g. ``/Volumes/<catalog>/<inst>_silver/...``
    """
    try:
        for seg in pathlib.PurePosixPath(silver_volume_path).parts:
            if seg.endswith("_silver"):
                return seg[: -len("_silver")]
    except Exception:
        pass
    return None


def training_cohort_json_for_pipeline_event(task: t.Any) -> t.Optional[str]:
    """
    Build optional JSON string of modeling training cohorts for run events (training jobs only).
    """
    try:
        modeling_cfg = getattr(task.cfg, "modeling", None)
        training_cfg = (
            getattr(modeling_cfg, "training", None)
            if modeling_cfg is not None
            else None
        )
        cohorts = (
            getattr(training_cfg, "cohort", None) if training_cfg is not None else None
        )
        if cohorts:
            return json.dumps(cohorts, default=str)
    except Exception:
        pass
    return None


def _pipeline_event_dataset_names(task: t.Any) -> tuple[t.Optional[str], t.Optional[str]]:
    ds = getattr(task.cfg, "datasets", None)
    return (
        getattr(ds, "raw_cohort", None) if ds is not None else None,
        getattr(ds, "raw_course", None) if ds is not None else None,
    )


def _pipeline_event_payload(
    args: argparse.Namespace,
) -> dict[str, t.Optional[str]]:
    return {
        "bronze_volume_path": getattr(args, "bronze_volume_path", None),
        "silver_volume_path": getattr(args, "silver_volume_path", None),
        "config_file_path": getattr(args, "config_file_path", None),
    }


def run_data_audit_with_training_events(
    args: argparse.Namespace, task: t.Any, *, run: t.Optional[t.Callable[[], None]] = None
) -> None:
    """
    If ``job_type == "training"``, emit ``append_pipeline_run_event`` started, then
    run the task, then completed or failed. Otherwise call ``run()`` once.
    """
    databricks_institution_name = infer_databricks_institution_name(
        str(args.silver_volume_path)
    )
    do_run = run if run is not None else task.run

    if getattr(args, "job_type", None) != "training":
        do_run()
        return

    cohort = training_cohort_json_for_pipeline_event(task)
    raw_cohort_name, raw_course_name = _pipeline_event_dataset_names(task)

    append_pipeline_run_event(
        catalog=args.DB_workspace,
        run_id=args.db_run_id,
        run_type=args.job_type,
        event="started",
        institution_id=getattr(task.cfg, "institution_id", None),
        databricks_institution_name=databricks_institution_name,
        cohort=cohort,
        cohort_dataset_name=raw_cohort_name,
        course_dataset_name=raw_course_name,
        payload=_pipeline_event_payload(args),
    )

    try:
        do_run()
        append_pipeline_run_event(
            catalog=args.DB_workspace,
            run_id=args.db_run_id,
            run_type=args.job_type,
            event="completed",
            institution_id=getattr(task.cfg, "institution_id", None),
            databricks_institution_name=databricks_institution_name,
            cohort=cohort,
            cohort_dataset_name=raw_cohort_name,
            course_dataset_name=raw_course_name,
        )
    except Exception as e:
        append_pipeline_run_event(
            catalog=args.DB_workspace,
            run_id=args.db_run_id,
            run_type=args.job_type,
            event="failed",
            institution_id=getattr(task.cfg, "institution_id", None),
            databricks_institution_name=databricks_institution_name,
            cohort=cohort,
            cohort_dataset_name=raw_cohort_name,
            course_dataset_name=raw_course_name,
            error_message=str(e),
        )
        raise


def flush_data_audit_logging() -> None:
    """Flush all root logger handlers, then :func:`logging.shutdown`."""
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()
