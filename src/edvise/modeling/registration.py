import logging
import typing as t
import re
import mlflow
import mlflow.exceptions
import mlflow.tracking

LOGGER = logging.getLogger(__name__)


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


def normalize_degree(text: str) -> str:
    """
    Normalize degree text by removing the word 'degree' and standardizing capitalization.

    Removes trailing 'degree' (case-insensitive) and converts text to title case
    (lowercase with first letter capitalized).

    Args:
        text: Degree text to normalize (e.g., "ASSOCIATE'S DEGREE", "Bachelor's degree")

    Returns:
        Normalized degree text (e.g., "Associate's", "Bachelor's")

    Examples:
        normalize_degree("ASSOCIATE'S DEGREE")
        "Associates"
    """
    # remove the word "degree" (case-insensitive) at the end
    text = text.strip()

    # remove the word "degree" (case-insensitive) at the end
    text = re.sub(r"\s*degree\s*$", "", text, flags=re.IGNORECASE)

    # normalize possessive degrees: bachelor's/master's/associate's -> bachelors/masters/associates
    # handles straight and curly apostrophes
    text = re.sub(r"[’']s\b", "s", text, flags=re.IGNORECASE)

    # normalize capitalization
    return text.lower().capitalize()


def extract_time_limits(intensity_time_limits: dict) -> str:
    """Transform intensity time limits into compact string like '3Y FT, 6Y PT'"""
    # Define order
    order = ["FULL-TIME", "PART-TIME"]

    parts = []
    for enroll_intensity in order:
        if enroll_intensity not in intensity_time_limits:
            continue

        duration, unit = intensity_time_limits[enroll_intensity]
        duration_str = (
            str(int(duration)) if duration == int(duration) else str(duration)
        )
        unit_abbrev = unit[0].upper()
        intensity_abbrev = "".join(word[0] for word in enroll_intensity.split("-"))

        parts.append(f"{duration_str}{unit_abbrev} {intensity_abbrev}")

    return ", ".join(parts)


def get_model_name(
    *,
    institution_id: str,
    target: t.Any,
    checkpoint: t.Any,
    student_criteria: dict,
    extra_info: t.Optional[str] = None,
) -> str:
    """
    Get a simple, lowercase, underscore-separated model name for Unity Catalog.

    Use Formatting().friendly_case() from reporting.utils.formatting to convert
    to display format for front-end.

    Args:
        target: Target config object (supports both dict and Pydantic model access)
        checkpoint: Checkpoint config object (supports both dict and Pydantic model access)
        student_criteria: Dictionary of student selection criteria

    Returns:
        Simple lowercase model name like "retention_into_year_2_associates"

    Examples:
        >>> # Retention example
        >>> get_model_name(...)
        'retention_into_year_2_associates'

        >>> # Graduation example
        >>> get_model_name(...)
        'graduation_in_3y_ft_6y_pt_checkpoint_2_core_terms'
    """

    # Helper to get attribute that works with both dicts and objects
    def get_attr(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Helper to format time limits as simple underscore string
    def format_time_limits(intensity_time_limits):
        parts = []
        for intensity in ["FULL-TIME", "PART-TIME"]:
            if intensity not in intensity_time_limits:
                continue
            duration, unit = intensity_time_limits[intensity]
            duration_str = (
                str(int(duration)) if duration == int(duration) else str(duration)
            )
            unit_abbrev = unit[0].lower()
            intensity_abbrev = "".join(word[0] for word in intensity.split("-")).lower()
            parts.append(f"{duration_str}{unit_abbrev}_{intensity_abbrev}")
        return "_".join(parts)

    target_type = get_attr(target, "type_")

    if target_type == "retention":
        if "credential_type_sought_year_1" in student_criteria:
            credential_type = normalize_degree(
                student_criteria["credential_type_sought_year_1"]
            ).lower()
            model_name = f"retention_into_year_2_{credential_type}"
        else:
            model_name = "retention_into_year_2_all_degrees"
    elif target_type == "graduation":
        time_limits = format_time_limits(get_attr(target, "intensity_time_limits"))
        checkpoint_type = get_attr(checkpoint, "type_")
        if checkpoint_type == "nth":
            n_plus_1 = get_attr(checkpoint, "n") + 1
            if get_attr(checkpoint, "exclude_non_core_terms") == True:
                model_name = (
                    f"graduation_in_{time_limits}_checkpoint_{n_plus_1}_core_terms"
                )
            else:
                model_name = (
                    f"graduation_in_{time_limits}_checkpoint_{n_plus_1}_total_terms"
                )
        elif checkpoint_type == "first_student_terms":
            model_name = f"graduation_in_{time_limits}_checkpoint_first_term"
        elif checkpoint_type == "first_student_terms_within_cohort":
            model_name = f"graduation_in_{time_limits}_checkpoint_first_cohort_term"
        elif checkpoint_type == "first_at_num_credits_earned":
            creds = get_attr(checkpoint, "min_num_credits")
            credits = str(int(creds)) if creds == int(creds) else str(creds)
            model_name = f"graduation_in_{time_limits}_checkpoint_{credits}_credits"
    elif target_type == "credits_earned":
        time_limits = format_time_limits(get_attr(target, "intensity_time_limits"))
        checkpoint_type = get_attr(checkpoint, "type_")
        creds = get_attr(target, "min_num_credits")
        credits = str(int(creds)) if creds == int(creds) else str(creds)
        if checkpoint_type == "nth":
            n_plus_1 = get_attr(checkpoint, "n") + 1
            if get_attr(checkpoint, "exclude_non_core_terms") == True:
                model_name = f"{credits}_credits_in_{time_limits}_checkpoint_{n_plus_1}_core_terms"
            else:
                model_name = f"{credits}_credits_in_{time_limits}_checkpoint_{n_plus_1}_total_terms"
        elif checkpoint_type == "first_student_terms":
            model_name = f"{credits}_credits_in_{time_limits}_checkpoint_first_term"
        elif checkpoint_type == "first_student_terms_within_cohort":
            model_name = (
                f"{credits}_credits_in_{time_limits}_checkpoint_first_cohort_term"
            )
        elif checkpoint_type == "first_at_num_credits_earned":
            chk_creds = get_attr(checkpoint, "min_num_credits")
            checkpoint_credits = (
                str(int(chk_creds)) if chk_creds == int(chk_creds) else str(chk_creds)
            )
            model_name = f"{credits}_credits_in_{time_limits}_checkpoint_{checkpoint_credits}_credits"

    if extra_info is not None:
        model_name = f"{model_name}_{extra_info}"
    return model_name


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
