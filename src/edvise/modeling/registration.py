import logging
import typing as t

import mlflow
import mlflow.exceptions
import mlflow.tracking

from edvise.shared.utils import (
    format_enrollment_intensity_time_limits,
    normalize_degree,
)

if t.TYPE_CHECKING:
    from edvise.configs.pdp import (
        CheckpointFirstAtNumCreditsEarnedConfig,
        CheckpointFirstConfig,
        CheckpointFirstWithinCohortConfig,
        CheckpointLastConfig,
        CheckpointLastInEnrollmentYearConfig,
        CheckpointNthConfig,
        TargetCreditsEarnedConfig,
        TargetGraduationConfig,
        TargetRetentionConfig,
    )

    TargetConfigType = (
        TargetCreditsEarnedConfig | TargetGraduationConfig | TargetRetentionConfig
    )
    CheckpointConfigType = (
        CheckpointFirstAtNumCreditsEarnedConfig
        | CheckpointFirstConfig
        | CheckpointFirstWithinCohortConfig
        | CheckpointLastConfig
        | CheckpointLastInEnrollmentYearConfig
        | CheckpointNthConfig
    )

__all__ = [
    "normalize_degree",
    "format_enrollment_intensity_time_limits",
]

LOGGER = logging.getLogger(__name__)


def _get_attr(obj: t.Any, key: str, default: t.Any = None) -> t.Any:
    """Get attribute from dict or object (supports both config representations)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _checkpoint_suffix(
    checkpoint: t.Any,
    checkpoint_type: str | None,
) -> str:
    """Build checkpoint suffix for model name from checkpoint config."""
    if checkpoint_type == "nth":
        n_plus_1 = _get_attr(checkpoint, "n") + 1
        core = bool(_get_attr(checkpoint, "exclude_non_core_terms"))
        return f"checkpoint_{n_plus_1}_{'core_terms' if core else 'total_terms'}"

    if checkpoint_type == "first_student_terms":
        return "checkpoint_first_term"

    if checkpoint_type == "first_student_terms_within_cohort":
        return "checkpoint_first_cohort_term"

    if checkpoint_type == "first_at_num_credits_earned":
        creds = _get_attr(checkpoint, "min_num_credits")
        credits = str(int(creds)) if creds == int(creds) else str(creds)
        return f"checkpoint_{credits}_credits"

    if checkpoint_type in (None, ""):
        return ""

    raise ValueError(f"Unsupported checkpoint type: {checkpoint_type!r}")


def register_mlflow_model(
    model_name: str,
    institution_id: str,
    *,
    run_id: str,
    catalog: str,
    registry_uri: str = "databricks-uc",
    model_alias: t.Optional[str] = "Staging",
    mlflow_client: mlflow.tracking.MlflowClient,
) -> None:
    """
    Register an mlflow model from a run, using run-level tags to prevent duplicate registration.

    Args:
        model_name: Unity Catalog compatible model name (lowercase with underscores)
        institution_id
        run_id
        catalog
        registry_uri
        model_alias
        mlflow_client

    References:
        - https://mlflow.org/docs/latest/model-registry.html
    """
    model_path = f"{catalog}.{institution_id}_gold.{model_name}"
    LOGGER.info("Registering model '%s' to Unity Catalog", model_path)
    mlflow.set_registry_uri(registry_uri)

    # Create the registered model if it doesn't exist
    try:
        mlflow_client.create_registered_model(name=model_path)
        LOGGER.info("New registered model '%s' successfully created", model_path)
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            LOGGER.info("Model '%s' already exists in registry", model_path)
        else:
            raise e

    # Check for the "model_registered" tag on the run
    try:
        run_tags = mlflow_client.get_run(run_id).data.tags
        if run_tags.get("model_registered") == "true":
            LOGGER.info("Run ID '%s' has already been registered. Skipping.", run_id)
            return
    except mlflow.exceptions.MlflowException as e:
        LOGGER.warning("Unable to check tags for run_id '%s': %s", run_id, str(e))
        raise

    # Register new model version
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow_client.create_model_version(
        name=model_path,
        source=model_uri,
        run_id=run_id,
    )
    LOGGER.info("Registered new model version %s from run_id '%s'", mv.version, run_id)

    # Mark the run as registered via tag
    mlflow_client.set_tag(run_id, "model_registered", "true")

    # Optionally assign alias
    if model_alias:
        mlflow_client.set_registered_model_alias(model_path, model_alias, mv.version)
        LOGGER.info("Set alias '%s' to version %s", model_alias, mv.version)


def get_model_name(
    *,
    institution_id: t.Optional[str] = None,
    target: str = "",
    checkpoint: str = "",
    extra_info: t.Optional[str] = None,
) -> str:
    """
    Get a standard model name generated from key components, formatted as
    "[{institution_id}_]{target}[_{checkpoint}][_{extra_info}]"

    Args:
        institution_id: Optional institution identifier
        target: Target name
        checkpoint: Optional checkpoint name
        extra_info: Optional additional information

    Returns:
        Formatted model name
    """
    parts = []
    if institution_id:
        parts.append(institution_id)
    if target:
        parts.append(target)
    if checkpoint:
        parts.append(checkpoint)

    model_name = "_".join(parts)

    if extra_info is not None:
        model_name = f"{model_name}_{extra_info}"

    return model_name


def pdp_get_model_name(
    *,
    target: "TargetConfigType | dict[str, t.Any]",
    checkpoint: "CheckpointConfigType | dict[str, t.Any]",
    student_criteria: dict,
    extra_info: t.Optional[str] = None,
) -> str:
    """
    PDP-specific wrapper for get_model_name that derives target and checkpoint names
    from config objects using PDP's standard logic.

    Get a simple, lowercase, underscore-separated model name for Unity Catalog.

    Use Formatting().friendly_case() from reporting.utils.formatting to convert
    to display format for front-end.

    Args:
        target: Target config object (supports both dict and Pydantic model access)
        checkpoint: Checkpoint config object (supports both dict and Pydantic model access)
        student_criteria: Dictionary of student selection criteria
        extra_info: Optional additional information to append to model name

    Returns:
        Simple lowercase model name like "retention_into_year_2_associates"

    Examples:
        >>> # Retention example
        >>> pdp_get_model_name(...)
        'retention_into_year_2_associates'

        >>> # Graduation example
        >>> pdp_get_model_name(...)
        'graduation_in_3y_ft_6y_pt_checkpoint_2_core_terms'
    """
    target_type = _get_attr(target, "type_")
    if not target_type:
        raise ValueError("target.type_ is required to build model name")

    checkpoint_type = _get_attr(checkpoint, "type_") if checkpoint is not None else None

    # --- Retention ---
    if target_type == "retention":
        if "credential_type_sought_year_1" in student_criteria:
            credential = normalize_degree(
                student_criteria["credential_type_sought_year_1"]
            ).lower()
            target_name = f"retention_into_year_2_{credential}"
        else:
            target_name = "retention_into_year_2_all_degrees"
        checkpoint_name = ""

    # --- Graduation / Credits Earned share time limit formatting ---
    elif target_type in {"graduation", "credits_earned"}:
        time_limits = format_enrollment_intensity_time_limits(
            intensity_time_limits=_get_attr(target, "intensity_time_limits"),
            style="underscore",
        )
        if target_type == "graduation":
            base = "graduation_in"
        else:
            creds = _get_attr(target, "min_num_credits")
            credits = str(int(creds)) if creds == int(creds) else str(creds)
            base = f"{credits}_credits_in"
        target_name = f"{base}_{time_limits}"
        checkpoint_name = _checkpoint_suffix(checkpoint, checkpoint_type)

    else:
        raise ValueError(f"Unsupported target type: {target_type!r}")

    return get_model_name(
        target=target_name,
        checkpoint=checkpoint_name,
        extra_info=extra_info,
    )


def get_mlflow_model_uri(
    *,
    model_name: t.Optional[str] = None,
    model_version: t.Optional[int] = None,
    model_alias: t.Optional[str] = None,
    run_id: t.Optional[str] = None,
    model_path: t.Optional[str] = None,
) -> str:
    """
    Get an mlflow model's URI based on its name, version, alias, path, and/or run id.

    References:
        - https://docs.databricks.com/gcp/en/mlflow/models
        - https://www.mlflow.org/docs/latest/concepts.html#artifact-locations
    """
    if run_id is not None and model_path is not None:
        return f"runs:/{run_id}/{model_path}"
    elif model_name is not None and model_version is not None:
        return f"models:/{model_name}/{model_version}"
    elif model_name is not None and model_alias is not None:
        return f"models:/{model_name}@{model_alias}"
    else:
        raise ValueError(
            "unable to determine model URI from inputs: "
            f"{model_name=}, {model_version=}, {model_alias=}, {model_path=}, {run_id=}"
        )
