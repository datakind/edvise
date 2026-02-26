"""
Constants for NSC SFTP ingestion pipeline.

These values are fixed and don't vary between runs or environments.
For environment-specific values (like secret scope names), see gcp_config.yaml.
"""

# Databricks catalog and schema
try:
    dbutils
    workspace_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().workspaceId().get()
    if workspace_id == "4437281602191762":
        CATALOG = "dev_sst_02"
    elif workspace_id == "2052166062819251":
        CATALOG = "staging_sst_01"
except:
    from unittest.mock import MagicMock
    dbutils = MagicMock()
    CATALOG = "staging_sst_01"
DEFAULT_SCHEMA = "default"

# Table names (without catalog.schema prefix)
MANIFEST_TABLE = "ingestion_manifest"
QUEUE_TABLE = "pending_ingest_queue"
PLAN_TABLE = "institution_ingest_plan"

# Full table paths
MANIFEST_TABLE_PATH = f"{CATALOG}.{DEFAULT_SCHEMA}.{MANIFEST_TABLE}"
QUEUE_TABLE_PATH = f"{CATALOG}.{DEFAULT_SCHEMA}.{QUEUE_TABLE}"
PLAN_TABLE_PATH = f"{CATALOG}.{DEFAULT_SCHEMA}.{PLAN_TABLE}"

# SFTP settings
SFTP_REMOTE_FOLDER = "./receive"
SFTP_SOURCE_SYSTEM = "NSC"
SFTP_PORT = 22
SFTP_TMP_DIR = "/tmp/pdp_sftp_stage"
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
    "attemptedgatewaymathyear1": "attempted_gateway_math_year_1",
    "attemptedgatewayenglishyear1": "attempted_gateway_english_year_1",
    "completedgatewaymathyear1": "completed_gateway_math_year_1",
    "completedgatewayenglishyear1": "completed_gateway_english_year_1",
    "gatewaymathgradey1": "gateway_math_grade_y_1",
    "gatewayenglishgradey1": "gateway_english_grade_y_1",
    "attempteddevmathy1": "attempted_dev_math_y_1",
    "attempteddevenglishy1": "attempted_dev_english_y_1",
    "completeddevmathy1": "completed_dev_math_y_1",
    "completeddevenglishy1": "completed_dev_english_y_1",
}
