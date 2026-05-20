"""
Resolve registered-model metadata and on-volume model card PDF paths.

Used by notebooks/jobs that need the MLflow run id for the latest (or named)
model in an institution's ``{institution_id}_gold`` schema, then the PDF under
``gold_volume/model_cards/<run_id>/``.
"""

from __future__ import annotations

import os
import typing as t

if t.TYPE_CHECKING:
    from mlflow.entities.model_registry import ModelVersion
    from mlflow.entities.model_registry import RegisteredModel

__all__ = [
    "gold_model_cards_run_dir",
    "gold_registered_model_name_prefix",
    "find_model_card_pdf_under_run_dir",
    "get_latest_registered_model_run_id",
    "pick_latest_model_version",
    "pick_newest_registered_model_by_created_at",
    "search_institution_registered_models",
    "short_model_name",
]


def local_fs_path(p: str) -> str:
    return p.replace("dbfs:/", "/dbfs/") if p and p.startswith("dbfs:/") else p


def gold_registered_model_name_prefix(catalog: str, institution_id: str) -> str:
    return f"{catalog}.{institution_id}_gold."


def gold_model_cards_run_dir(catalog: str, institution_id: str, run_id: str) -> str:
    return (
        f"/Volumes/{catalog}/{institution_id}_gold/gold_volume/model_cards/{run_id}"
    )


def short_model_name(full_model_name: str, catalog: str, institution_id: str) -> str:
    prefix = gold_registered_model_name_prefix(catalog, institution_id)
    if full_model_name.startswith(prefix):
        return full_model_name[len(prefix) :]
    parts = full_model_name.split(".")
    return parts[-1] if parts else full_model_name


def _timestamp_ms(obj: t.Any, *attr_names: str) -> int:
    for name in attr_names:
        value = getattr(obj, name, None)
        if value is not None:
            return int(value)
    return 0


def search_institution_registered_models(
    client: t.Any,
    *,
    catalog: str,
    institution_id: str,
    model_name: str | None = None,
) -> list[RegisteredModel]:
    prefix = gold_registered_model_name_prefix(catalog, institution_id)
    models = [
        rm
        for rm in client.search_registered_models()
        if (rm.name or "").startswith(prefix)
    ]
    if model_name:
        full = f"{prefix}{model_name}"
        models = [rm for rm in models if rm.name == full]
    return models


def pick_newest_registered_model_by_created_at(
    models: list[RegisteredModel],
) -> RegisteredModel | None:
    if not models:
        return None
    return max(
        models,
        key=lambda m: _timestamp_ms(m, "creation_timestamp", "last_updated_timestamp"),
    )


def pick_latest_model_version(versions: list[ModelVersion]) -> ModelVersion | None:
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def get_latest_registered_model_run_id(
    client: t.Any,
    *,
    catalog: str,
    institution_id: str,
    model_name: str | None = None,
) -> tuple[str, str, str]:
    """
    Resolve the MLflow source run id for an institution's latest registered model.

    When ``model_name`` is omitted, picks the registered model with the greatest
    ``creation_timestamp`` in ``{catalog}.{institution_id}_gold``, then its
    highest version number. When ``model_name`` is set, uses that model only.

    Returns:
        (run_id, full_uc_model_name, version_str)
    """
    models = search_institution_registered_models(
        client,
        catalog=catalog,
        institution_id=institution_id,
        model_name=model_name,
    )
    if not models:
        scope = (
            f"name='{catalog}.{institution_id}_gold.{model_name}'"
            if model_name
            else f"schema {catalog}.{institution_id}_gold"
        )
        raise ValueError(f"No registered models found for {scope}")

    if model_name:
        registered = models[0]
    else:
        registered = pick_newest_registered_model_by_created_at(models)
        if registered is None:
            raise ValueError(
                f"No registered models found for {catalog}.{institution_id}_gold"
            )

    full_name = registered.name
    versions = client.search_model_versions(f"name='{full_name}'")
    latest = pick_latest_model_version(list(versions or []))
    if latest is None:
        raise ValueError(f"No versions found for registered model: {full_name}")

    run_id = getattr(latest, "run_id", None)
    if not run_id:
        raise ValueError(f"Model version has no run_id: {full_name} v{latest.version}")

    return str(run_id), full_name, str(latest.version)


def find_model_card_pdf_under_run_dir(run_dir: str) -> str:
    """
    Return the path to a model card PDF under ``model_cards/<run_id>/``.

    Prefers ``model-card-*.pdf`` when multiple PDFs exist.
    """
    base = local_fs_path(run_dir)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Model card run directory does not exist: {run_dir}")

    pdfs = sorted(
        os.path.join(base, name)
        for name in os.listdir(base)
        if name.lower().endswith(".pdf")
    )
    if not pdfs:
        raise FileNotFoundError(f"No PDF found under: {run_dir}")

    preferred = [p for p in pdfs if os.path.basename(p).startswith("model-card-")]
    return preferred[0] if preferred else pdfs[0]


def resolve_latest_model_card_pdf(
    client: t.Any,
    *,
    catalog: str,
    institution_id: str,
    model_name: str | None = None,
) -> tuple[str, str, str, str]:
    """
    Full resolution: UC registry -> run id -> PDF path on gold volume.

    Returns:
        (pdf_path, run_id, full_uc_model_name, version_str)
    """
    run_id, full_name, version = get_latest_registered_model_run_id(
        client,
        catalog=catalog,
        institution_id=institution_id,
        model_name=model_name,
    )
    run_dir = gold_model_cards_run_dir(catalog, institution_id, run_id)
    pdf_path = find_model_card_pdf_under_run_dir(run_dir)
    return pdf_path, run_id, full_name, version
