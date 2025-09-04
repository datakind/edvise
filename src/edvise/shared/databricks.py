import logging
import typing as t


LOGGER = logging.getLogger(__name__)


# TODO: Check with Vish these are Databricks specific or just  mlflow?
def get_model_name(
    *,
    institution_id: str,
    target: str,
    checkpoint: str,
    extra_info: t.Optional[str] = None,
) -> str:
    """
    Get a standard model name generated from key components, formatted as
    "{institution_id}_{target}_{checkpoint}[_{extra_info}]"
    """
    model_name = f"{institution_id}_{target}_{checkpoint}"
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


def get_experiment_name(
    *,
    institution_id: str,
    job_run_id: str,
    primary_metric: str,
    timeout_minutes: int,
    exclude_frameworks: t.Optional[list[str]] = None,
) -> str:
    """
    Get a descriptive experiment name based on more important input parameters.

    See Also:
        - :func:`run_automl_classification()`

    References:
        - https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html#classify
    """
    name_components = [
        institution_id,
        f"{job_run_id=}",
        f"{primary_metric=}",
        f"{timeout_minutes=}",
    ]
    if exclude_frameworks:
        name_components.append(f"exclude_frameworks={','.join(exclude_frameworks)}")
    name_components.append(time.strftime("%Y-%m-%dT%H:%M:%S"))

    name = "__".join(name_components)
    if len(name) > 500:
        LOGGER.warning("truncating long experiment name '%s' to first 500 chars", name)
        name = name[:500]
    return name
