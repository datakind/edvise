"""
Constants for NSC SFTP ingestion pipeline.

Unity Catalog name must match the job's DB_workspace parameter
(see runtime.bootstrap_catalog). Secret scope/key names are job params.
"""

from __future__ import annotations

# Unity Catalog name — set by configure_nsc_catalog.
DEFAULT_CATALOG_FOR_LOCAL = "dev_sst_02"
DEFAULT_SCHEMA = "default"

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
    SFTP_TMP_DIR = f"/Volumes/{CATALOG}/{DEFAULT_SCHEMA}/{SFTP_TMP_VOLUME_NAME}"


# SFTP settings
SFTP_REMOTE_FOLDER = "./receive"
SFTP_SOURCE_SYSTEM = "NSC"
SFTP_PORT = 22
SFTP_DOWNLOAD_CHUNK_MB = 150
SFTP_VERIFY_DOWNLOAD = "size"  # Options: "size", "sha256", "md5", "none"
SFTP_SECRET_KEY_HOST = "nsc-sftp-host"
SFTP_SECRET_KEY_USER = "nsc-sftp-user"
SFTP_SECRET_KEY_PASSWORD = "nsc-sftp-password"

# Edvise API path templates (base URL + secret scope/key come from job params).
SST_TOKEN_PATH = "/api/v1/token-from-api-key"
INSTITUTION_LOOKUP_PATH = "/api/v1/institutions/pdp-id/{pdp_id}"

INSTITUTION_COLUMN_PATTERN = r"(?=.*institution)(?=.*id)"

# Applied after snake_case conversion
COLUMN_RENAMES = {
    # convert_to_snake_case splits trailing digit groups with an underscore,
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
