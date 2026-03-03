# Databricks notebook source
# MAGIC %md
# MAGIC # Sync Institutions Metadata (WebApp -> UC Delta)
# MAGIC
# MAGIC This notebook pulls institution metadata from the SST WebApp API and writes it to:
# MAGIC
# MAGIC `"<DB_workspace>.default.institutions"`
# MAGIC
# MAGIC Columns:
# MAGIC - `institution_id`
# MAGIC - `institution_name`
# MAGIC - `pdp_id`
# MAGIC - `databricks_institution_name`

# COMMAND ----------

import os

from py4j.protocol import Py4JJavaError

from edvise.shared.dashboard_metadata.sync_institutions_table import (
    sync_institutions_table,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC
# MAGIC Prefer using Secrets for the token. The cell below supports either widgets or env vars.

# COMMAND ----------

try:
    # Optional: widgets (works when run as a Databricks job / notebook with widgets configured)
    DB_workspace = dbutils.widgets.get("DB_workspace")  # noqa: F821
    SST_ACCESS_TOKEN = dbutils.widgets.get("SST_ACCESS_TOKEN")  # noqa: F821
    SST_BASE_URL = dbutils.widgets.get("SST_BASE_URL")  # noqa: F821
    NAME_QUERY = dbutils.widgets.get("NAME_QUERY")  # noqa: F821
except Py4JJavaError:
    # Interactive fallback
    DB_workspace = os.environ.get("DB_workspace", "")
    SST_ACCESS_TOKEN = os.environ.get("SST_ACCESS_TOKEN", "")
    SST_BASE_URL = os.environ.get("SST_BASE_URL", "https://staging-sst.datakind.org")
    NAME_QUERY = os.environ.get("NAME_QUERY", "")

DB_workspace, SST_BASE_URL

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Sync

# COMMAND ----------

table_path = sync_institutions_table(
    DB_workspace=DB_workspace,
    sst_access_token=SST_ACCESS_TOKEN,
    sst_base_url=SST_BASE_URL,
    name_query=NAME_QUERY,
    schema="default",
    table="institutions",
    write_mode="overwrite",
    spark=spark,  # noqa: F821
)

table_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview

# COMMAND ----------

spark.table(table_path).orderBy("institution_name").limit(50).display()  # noqa: F821

