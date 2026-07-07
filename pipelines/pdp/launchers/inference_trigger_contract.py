"""
PDP versioned inference launcher — integration contract for edvise-api and operators.

**Replace** direct ``edvise_github_sourced_pdp_inference_pipeline`` job runs with
``edvise_versioned_inference_launcher``. The launcher resolves the model's trained
``pipeline_version`` (Git SHA in dev, release tag in staging), materializes the archived
DAB bundle, validates parameters, and submits the full inference DAG at that version.

There is no fallback to HEAD/current-deploy inference when validation fails.
"""

from __future__ import annotations

import json
from typing import Any

from pipelines.pdp.launchers.inference_parameters import build_stable_trigger_payload

VERSIONED_INFERENCE_LAUNCHER_JOB_KEY = "edvise_versioned_inference_launcher"

# Job parameters expected by ``github_pdp_versioned_inference_launcher.yml``.
VERSIONED_INFERENCE_LAUNCHER_JOB_PARAMETERS: tuple[str, ...] = (
    "databricks_institution_name",
    "model_name",
    "DB_workspace",
    "release_base_path",
    "github_repo",
    "git_url",
    "cohort_file_name",
    "course_file_name",
    "gcp_bucket_name",
    "datakind_notification_email",
    "DK_CC_EMAIL",
    "ds_run_as",
    "service_account_executer",
    "datakind_group_to_manage_workflow",
    "viewer_user",
    "inference_output_run_id",
    "inference_parameters_json",
    "stable_trigger_json",
)


def build_versioned_inference_job_parameters(
    *,
    databricks_institution_name: str,
    model_name: str,
    db_workspace: str,
    cohort_file_name: str = "",
    course_file_name: str = "",
    gcp_bucket_name: str = "",
    datakind_notification_email: str = "",
    dk_cc_email: str = "",
    ds_run_as: str = "",
    service_account_executer: str = "",
    datakind_group_to_manage_workflow: str = "",
    viewer_user: str = "",
    inference_output_run_id: str = "",
    release_base_path: str = "",
    github_repo: str = "datakind/edvise",
    git_url: str = "https://github.com/datakind/edvise",
    inference_parameters_json: dict[str, str] | None = None,
    stable_trigger: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Build Databricks ``run_now`` job_parameters for ``edvise_versioned_inference_launcher``.

    Prefer ``stable_trigger`` (Layer-1 webapp payload); otherwise flat fields are mapped
  inside the launcher via built-in aliases.
    """
    params: dict[str, str] = {
        "databricks_institution_name": databricks_institution_name.strip(),
        "model_name": model_name.strip(),
        "DB_workspace": db_workspace.strip(),
        "release_base_path": release_base_path.strip(),
        "github_repo": github_repo.strip(),
        "git_url": git_url.strip(),
        "cohort_file_name": cohort_file_name.strip(),
        "course_file_name": course_file_name.strip(),
        "gcp_bucket_name": gcp_bucket_name.strip(),
        "datakind_notification_email": datakind_notification_email.strip(),
        "DK_CC_EMAIL": dk_cc_email.strip(),
        "ds_run_as": ds_run_as.strip(),
        "service_account_executer": service_account_executer.strip(),
        "datakind_group_to_manage_workflow": datakind_group_to_manage_workflow.strip(),
        "viewer_user": viewer_user.strip(),
        "inference_output_run_id": inference_output_run_id.strip(),
    }
    if inference_parameters_json:
        params["inference_parameters_json"] = json.dumps(inference_parameters_json)
    if stable_trigger is None:
        stable_trigger = build_stable_trigger_payload(
            institution=databricks_institution_name,
            model_name=model_name,
            workspace=db_workspace,
            cohort_dataset=cohort_file_name,
            course_dataset=course_file_name,
            output_bucket=gcp_bucket_name,
            notification_to=datakind_notification_email,
            notification_cc=dk_cc_email,
            inference_output_run_id=inference_output_run_id,
            ds_run_as=ds_run_as,
            service_account_executer=service_account_executer,
        )
    if stable_trigger:
        params["stable_trigger_json"] = json.dumps(stable_trigger)
    return params
