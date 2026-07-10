"""Shared CLI argument parsing for versioned inference launcher tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from pipelines.pdp.launchers.inference_parameters import (
    build_stable_trigger_payload,
    deep_merge_stable_dict,
    parse_stable_trigger_json,
)
from pipelines.pdp.launchers.release_config import resolve_release_base_path


@dataclass(frozen=True)
class LauncherTriggerInputs:
    """Resolved launcher inputs shared by validate and trigger tasks."""

    databricks_institution_name: str
    model_name: str
    db_workspace: str
    release_base_path: str
    git_url: str
    param_overrides: dict[str, str]
    extra_param_overrides: dict[str, str]
    stable_trigger: dict[str, Any]


def _optional_arg(value: str) -> str | None:
    s = (value or "").strip()
    return s if s else None


def add_model_resolution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--databricks_institution_name", default="")
    parser.add_argument("--model_name", default="")
    parser.add_argument("--DB_workspace", default="")
    parser.add_argument(
        "--model_run_id",
        default="",
        help=(
            "Optional training model_run_id (MLflow run id). Skips pipeline_models / "
            "UC lookup when set."
        ),
    )
    parser.add_argument(
        "--release_base_path",
        default="",
        help="UC volume base for edvise_releases; defaults from DB_workspace.",
    )


def optional_model_run_id(args: argparse.Namespace) -> str | None:
    return _optional_arg(getattr(args, "model_run_id", ""))


def add_inference_trigger_args(parser: argparse.ArgumentParser) -> None:
    add_model_resolution_args(parser)
    parser.add_argument(
        "--git_url",
        default="",
        help="Git remote for child inference submit (default: datakind/edvise).",
    )
    parser.add_argument("--cohort_file_name", default="")
    parser.add_argument("--course_file_name", default="")
    parser.add_argument("--gcp_bucket_name", default="")
    parser.add_argument("--datakind_notification_email", default="")
    parser.add_argument("--DK_CC_EMAIL", default="")
    parser.add_argument("--ds_run_as", default="")
    parser.add_argument("--service_account_executer", default="")
    parser.add_argument("--datakind_group_to_manage_workflow", default="")
    parser.add_argument("--viewer_user", default="")
    parser.add_argument("--inference_output_run_id", default="")
    parser.add_argument("--inference_parameters_json", default="")
    parser.add_argument("--stable_trigger_json", default="")


def parse_extra_parameter_overrides(raw: str) -> dict[str, str]:
    """Parse ``inference_parameters_json`` (archived parameter names only)."""
    text = (raw or "").strip()
    if not text:
        return {}
    extra = json.loads(text)
    if not isinstance(extra, dict):
        msg = "inference_parameters_json must be a JSON object"
        raise TypeError(msg)
    return {str(k): str(v) for k, v in extra.items() if v is not None}


def build_launcher_parameter_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides: dict[str, str] = {
        "databricks_institution_name": args.databricks_institution_name.strip(),
        "model_name": args.model_name.strip(),
        "DB_workspace": args.DB_workspace.strip(),
    }
    for key in (
        "cohort_file_name",
        "course_file_name",
        "gcp_bucket_name",
        "datakind_notification_email",
        "DK_CC_EMAIL",
        "ds_run_as",
        "service_account_executer",
        "datakind_group_to_manage_workflow",
        "viewer_user",
    ):
        val = _optional_arg(getattr(args, key, ""))
        if val is not None:
            overrides[key] = val
    out_id = _optional_arg(getattr(args, "inference_output_run_id", ""))
    if out_id is not None:
        overrides["db_run_id"] = out_id
    return overrides


def build_stable_trigger(args: argparse.Namespace) -> dict[str, Any]:
    base = build_stable_trigger_payload(
        institution=args.databricks_institution_name.strip(),
        model_name=args.model_name.strip(),
        workspace=args.DB_workspace.strip(),
        cohort_dataset=args.cohort_file_name.strip(),
        course_dataset=args.course_file_name.strip(),
        output_bucket=args.gcp_bucket_name.strip(),
        notification_to=args.datakind_notification_email.strip(),
        notification_cc=args.DK_CC_EMAIL.strip(),
        inference_output_run_id=args.inference_output_run_id.strip(),
        ds_run_as=args.ds_run_as.strip(),
        service_account_executer=args.service_account_executer.strip(),
    )
    overlay = parse_stable_trigger_json(getattr(args, "stable_trigger_json", ""))
    if overlay:
        return deep_merge_stable_dict(base, overlay)
    return base


def build_launcher_trigger_inputs(
    args: argparse.Namespace,
    *,
    default_git_url: str,
) -> LauncherTriggerInputs:
    db_ws = args.DB_workspace.strip()
    git_url = (_optional_arg(getattr(args, "git_url", "")) or default_git_url).strip()
    return LauncherTriggerInputs(
        databricks_institution_name=args.databricks_institution_name.strip(),
        model_name=args.model_name.strip(),
        db_workspace=db_ws,
        release_base_path=resolve_release_base_path(
            db_ws, getattr(args, "release_base_path", "")
        ),
        git_url=git_url,
        param_overrides=build_launcher_parameter_overrides(args),
        extra_param_overrides=parse_extra_parameter_overrides(
            getattr(args, "inference_parameters_json", "")
        ),
        stable_trigger=build_stable_trigger(args),
    )
