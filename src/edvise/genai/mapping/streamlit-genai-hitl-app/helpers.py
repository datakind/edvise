from __future__ import annotations

import io
import json
import os
from typing import Any, Literal

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config


DEFAULT_DB_WORKSPACE = "dev_sst_02"

HITLFileKey = Literal["identity_grain", "identity_term", "sma"]


def get_workspace_client() -> WorkspaceClient:
    return WorkspaceClient(config=Config())


def bronze_volume_path(institution_id: str, catalog: str) -> str:
    inst = institution_id.strip()
    cat = catalog.strip()
    if not inst or not cat:
        raise ValueError("institution_id and catalog are required")
    return f"/Volumes/{cat}/{inst}_bronze/bronze_volume"


def suggested_relative_path(hitl_file: HITLFileKey) -> str:
    """
    Default path segments under the institution bronze volume (``.../bronze_volume/<this>``).

    Adjust to match your layout (e.g. insert a ``genai_pipeline/<folder>/`` segment when you use one).
    """
    if hitl_file == "identity_grain":
        return "genai_pipeline/identity_hitl/identity_grain_hitl.json"
    if hitl_file == "identity_term":
        return "genai_pipeline/identity_hitl/identity_term_hitl.json"
    if hitl_file == "sma":
        return "genai_pipeline/sma_hitl.json"
    raise ValueError(f"Unknown hitl_file {hitl_file!r}")


def hitl_volume_path(
    *,
    catalog: str,
    institution_id: str,
    relative_path_under_bronze: str,
) -> str:
    """Absolute UC volume path: bronze root + relative path to the HITL JSON file."""
    rel = relative_path_under_bronze.strip().lstrip("/")
    if not rel:
        raise ValueError("Path under bronze volume must be non-empty")
    bronze = bronze_volume_path(institution_id, catalog)
    return f"{bronze.rstrip('/')}/{rel}"


def read_volume_json(path: str) -> dict[str, Any]:
    w = get_workspace_client()
    resp = w.files.download(path)
    if resp.contents is None:
        raise RuntimeError(f"Empty response downloading {path!r}")
    raw = resp.contents.read()
    return json.loads(raw.decode("utf-8"))


def write_volume_json(path: str, data: dict[str, Any]) -> None:
    w = get_workspace_client()
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    w.files.upload(path, io.BytesIO(payload.encode("utf-8")), overwrite=True)


def catalog_from_env() -> str:
    return os.getenv("DB_workspace", DEFAULT_DB_WORKSPACE).strip() or DEFAULT_DB_WORKSPACE


def detect_envelope_kind(data: dict[str, Any]) -> str:
    """Return ``identity``, ``sma``, or ``unknown`` based on top-level keys."""
    if "domain" in data and data.get("domain") in ("grain", "term"):
        return "identity"
    if "entity_type" in data and "items" in data:
        return "sma"
    return "unknown"
