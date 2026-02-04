import logging
import typing as t
import re
import mlflow
import mlflow.exceptions
import mlflow.tracking

LOGGER = logging.getLogger(__name__)


def sanitize_model_name_for_uc(model_name: str) -> str:
    """
    Sanitize a model name for Unity Catalog compliance.

    Unity Catalog does not allow spaces, forward slashes, or periods in object names.
    This function replaces these characters to create a valid UC name while keeping
    the name human-readable.

    Args:
        model_name: The human-readable model name (e.g., "retention into Year 2: Associate's")

    Returns:
        A sanitized name suitable for Unity Catalog (e.g., "retention_into_Year_2_Associates")

    Examples:
        >>> sanitize_model_name_for_uc("retention into Year 2: Associate's")
        'retention_into_Year_2_Associates'
        >>> sanitize_model_name_for_uc("Graduation in 3Y FT, 6Y PT (Checkpoint: 2 Core Terms)")
        'Graduation_in_3Y_FT_6Y_PT_Checkpoint_2_Core_Terms'
    """
    # Replace spaces with underscores
    sanitized = model_name.replace(" ", "_")

    # Remove colons, parentheses, commas, and apostrophes
    sanitized = sanitized.replace(":", "")
    sanitized = sanitized.replace("(", "")
    sanitized = sanitized.replace(")", "")
    sanitized = sanitized.replace(",", "")
    sanitized = sanitized.replace("'", "")

    # Remove any other non-alphanumeric characters except underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "", sanitized)

    # Remove any double underscores that might have been created
    sanitized = re.sub(r"_+", "_", sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    return sanitized


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
        model_name: Human-readable model name (will be sanitized for Unity Catalog)
        institution_id
        run_id
        catalog
        registry_uri
        model_alias
        mlflow_client

    References:
        - https://mlflow.org/docs/latest/model-registry.html
    """
    # Sanitize model name for Unity Catalog compliance (no spaces, slashes, or periods)
    sanitized_model_name = sanitize_model_name_for_uc(model_name)
    model_path = f"{catalog}.{institution_id}_gold.{sanitized_model_name}"
    LOGGER.info(
        "Model name '%s' sanitized to '%s' for Unity Catalog",
        model_name,
        sanitized_model_name,
    )
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
        "Associate's"
    """
    # remove the word "degree" (case-insensitive)
    text = re.sub(r"\s*degree\s*$", "", text, flags=re.IGNORECASE)
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
    Get a standard model name generated from key components, depending on target type and student criteria.
    """
    if target["_type"] == "retention":
        if "credential_type_sought_year_1" in student_criteria:
            credential_type = normalize_degree(
                student_criteria["credential_type_sought_year_1"]
            )
            model_name = f"{target['_type']} into Year 2: {credential_type}"
        else:
            # we can keep or remove All Degrees here, but just to make it more clear to schools who go with an "all-in" approach
            model_name = f"{target['_type']} into Year 2: All Degrees"
    elif target["_type"] == "graduation":
        time_limits = extract_time_limits(target["intensity_time_limits"])
        if checkpoint["type_"] == "nth":
            if checkpoint["exclude_non_core_terms"] == True:
                model_name = f"{target['type_']} in {time_limits} (Checkpoint: {checkpoint['n'] + 1} Core Terms)"
            else:
                model_name = f"{target['type_']} in {time_limits} (Checkpoint:{checkpoint['n'] + 1} Total Terms)"
        elif checkpoint["type_"] == "first_student_terms":
            model_name = f"{target['type_']} in {time_limits} (Checkpoint: First Term)"
        elif checkpoint["type_"] == "first_student_terms_within_cohort":
            model_name = (
                f"{target['type_']} in {time_limits} (Checkpoint: First Cohort Term)"
            )
        elif checkpoint["type_"] == "first_at_num_credits_earned":
            model_name = f"{target['type_']} in {time_limits} (Checkpoint: {str(checkpoint['min_num_credits'])} Credits)"
    elif target["_type"] == "credits_earned":
        time_limits = extract_time_limits(target["intensity_time_limits"])
        if checkpoint["type_"] == "nth":
            if checkpoint["exclude_non_core_terms"] == True:
                model_name = f"{str(target['min_num_credits'])} Credits in {time_limits} (Checkpoint: {checkpoint['n'] + 1} Core Terms)"
            else:
                model_name = f"{str(target['min_num_credits'])} Credits in {time_limits} (Checkpoint: {checkpoint['n'] + 1} Total Terms)"
        elif checkpoint["type_"] == "first_student_terms":
            model_name = f"{str(target['min_num_credits'])} Credits in {time_limits} (Checkpoint: First Term)"
        elif checkpoint["type_"] == "first_student_terms_within_cohort":
            model_name = f"{str(target['min_num_credits'])} Credits in {time_limits} (Checkpoint: First Cohort Term)"
        elif checkpoint["type_"] == "first_at_num_credits_earned":
            model_name = f"{str(target['min_num_credits'])} Credits in {time_limits} (Checkpoint: {str(checkpoint['min_num_credits'])} Credits)"
    # do we still need this extra info section? what's it for?
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
