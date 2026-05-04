"""
Constants for NSC SFTP ingestion pipeline.

Unity Catalog name must match the job's DB_workspace parameter (see
configure_nsc_catalog / resolve_nsc_catalog). Other values here are fixed or
scoped to default schema.
"""

from __future__ import annotations

import os
import sys

# Unity Catalog name — set by configure_nsc_catalog (usually from job parameter DB_workspace).
DEFAULT_CATALOG_FOR_LOCAL = "dev_sst_02"
DEFAULT_SCHEMA = "default"

# Table names (without catalog.schema prefix)
MANIFEST_TABLE = "ingestion_manifest"
QUEUE_TABLE = "pending_ingest_queue"
PLAN_TABLE = "institution_ingest_plan"

SFTP_TMP_VOLUME_NAME = "tmp"

CATALOG: str
MANIFEST_TABLE_PATH: str
QUEUE_TABLE_PATH: str
PLAN_TABLE_PATH: str
SFTP_TMP_VOLUME_FQN: str
SFTP_TMP_DIR: str


def configure_nsc_catalog(catalog: str) -> None:
    """Set Unity Catalog name and derived table/volume paths (once per process)."""
    global CATALOG, MANIFEST_TABLE_PATH, QUEUE_TABLE_PATH, PLAN_TABLE_PATH
    global SFTP_TMP_VOLUME_FQN, SFTP_TMP_DIR
    cat = str(catalog).strip()
    if not cat:
        raise ValueError(
            "NSC ingestion catalog is empty. Pass job parameter DB_workspace "
            "(Unity Catalog name), set widget DB_workspace, or NSC_DB_WORKSPACE."
        )
    CATALOG = cat
    MANIFEST_TABLE_PATH = f"{CATALOG}.{DEFAULT_SCHEMA}.{MANIFEST_TABLE}"
    QUEUE_TABLE_PATH = f"{CATALOG}.{DEFAULT_SCHEMA}.{QUEUE_TABLE}"
    PLAN_TABLE_PATH = f"{CATALOG}.{DEFAULT_SCHEMA}.{PLAN_TABLE}"
    SFTP_TMP_VOLUME_FQN = f"{CATALOG}.{DEFAULT_SCHEMA}.{SFTP_TMP_VOLUME_NAME}"
    SFTP_TMP_DIR = (
        f"/Volumes/{CATALOG}/{DEFAULT_SCHEMA}/{SFTP_TMP_VOLUME_NAME}"
    )


def parse_spark_python_task_params(argv: list[str] | None = None) -> dict[str, str]:
    """Parse ``--key value`` pairs from ``spark_python_task.parameters``."""
    if argv is None:
        argv = sys.argv
    out: dict[str, str] = {}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and i + 1 < len(argv):
            out[a[2:].replace("-", "_")] = argv[i + 1]
            i += 2
        else:
            i += 1
    return out


def resolve_nsc_catalog(argv: list[str] | None = None) -> str:
    """
    Resolve Unity Catalog name in order: task argv ``--DB_workspace``, notebook widget
    ``DB_workspace``, env ``NSC_DB_WORKSPACE``, else DEFAULT_CATALOG_FOR_LOCAL.
    """
    argv = sys.argv if argv is None else argv
    pairs = parse_spark_python_task_params(argv)
    raw = pairs.get("DB_workspace", "").strip()
    if raw:
        return raw
    try:
        from edvise.utils.databricks import get_db_widget_param

        w = get_db_widget_param("DB_workspace", default="")
        if str(w).strip():
            return str(w).strip()
    except Exception:
        pass
    env = os.environ.get("NSC_DB_WORKSPACE", "").strip()
    if env:
        return env
    return DEFAULT_CATALOG_FOR_LOCAL


# SFTP settings
SFTP_REMOTE_FOLDER = "./receive"
SFTP_SOURCE_SYSTEM = "NSC"
SFTP_PORT = 22
SFTP_DOWNLOAD_CHUNK_MB = 150
SFTP_VERIFY_DOWNLOAD = "size"  # Options: "size", "sha256", "md5", "none"

# Edvise API settings
SST_BASE_URL = "https://staging-sst.datakind.org"
SST_TOKEN_ENDPOINT = f"{SST_BASE_URL}/api/v1/token-from-api-key"
INSTITUTION_LOOKUP_PATH = "/api/v1/institutions/pdp-id/{pdp_id}"
SST_API_KEY_SECRET_KEY = "sst_staging_api_key"  # Key name in Databricks secrets

# File processing settings
INSTITUTION_COLUMN_PATTERN = r"(?=.*institution)(?=.*id)"

# Column name mappings (mangled -> normalized)
# Applied after snake_case conversion
COLUMN_RENAMES = {
    # NOTE: convert_to_snake_case splits trailing digit groups with an underscore,
    # e.g. "attemptedgatewaymathyear1" -> "attemptedgatewaymathyear_1".
    "attemptedgatewaymathyear_1": "attempted_gateway_math_year_1",
    "attemptedgatewayenglishyear_1": "attempted_gateway_english_year_1",
    "completedgatewaymathyear_1": "completed_gateway_math_year_1",
    "completedgatewayenglishyear_1": "completed_gateway_english_year_1",
    "gatewaymathgradey_1": "gateway_math_grade_y_1",
    "gatewayenglishgradey_1": "gateway_english_grade_y_1",
    "attempteddevmathy_1": "attempted_dev_math_y_1",
    "attempteddevenglishy_1": "attempted_dev_english_y_1",
    "completeddevmathy_1": "completed_dev_math_y_1",
    "completeddevenglishy_1": "completed_dev_english_y_1",
}

configure_nsc_catalog(DEFAULT_CATALOG_FOR_LOCAL)
