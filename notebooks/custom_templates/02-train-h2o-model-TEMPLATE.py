# Databricks notebook source
# MAGIC %md
# MAGIC # SST Train and Evaluate H2O Model
# MAGIC
# MAGIC Third step in the process of transforming raw data into actionable, data-driven insights for advisors: load a prepared modeling dataset, configure experiment tracking framework, then train and evaluate a predictive model.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks Classification with AutoML](https://docs.databricks.com/en/machine-learning/automl/classification.html)
# MAGIC - [Databricks AutoML Python API reference](https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# WARNING: AutoML/mlflow expect particular packages with version constraints
# that directly conflicts with dependencies in our SST repo. As a temporary fix,
# we need to manually install a certain version of pandas and scikit-learn in order
# for our models to load and run properly.

# %pip install git+https://github.com/datakind/edvise.git@v0.1.9
# %restart_python

# COMMAND ----------

import logging
import os

import mlflow
from mlflow.tracking import MlflowClient
from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils
from pyspark.dbutils import DBUtils

dbutils = DBUtils(spark)
client = MlflowClient()

from edvise import configs, dataio, modeling, utils
from edvise.modeling import h2o_ml


h2o_ml.utils.safe_h2o_init()

# HACK: Disable the mlflow widget template otherwise it freaks out
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

# COMMAND ----------

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

try:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    user_email = ctx.tags().get("user").get()
    if user_email:
        workspace_path = f"/Users/{user_email}"
        logging.info(f"retrieved workspace path at {workspace_path}")
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# Get job run id for automl run
job_run_id = utils.databricks.get_db_widget_param("job_run_id", default="interactive")

# COMMAND ----------

# MAGIC %md
# MAGIC ## configuration

# COMMAND ----------

# project configuration stored as a config file in TOML format
cfg = dataio.read.read_config(
    "./config-TEMPLATE.toml", schema=configs.custom.CustomProjectConfig
)
cfg

# COMMAND ----------

# MAGIC %md
# MAGIC # read preprocessed dataset

# COMMAND ----------

df = dataio.read.from_delta_table(
    cfg.datasets.silver["preprocessed"].table_path,
    spark_session=spark,
)
df.head()

# COMMAND ----------

# delta tables not great about maintaining dtypes; this may be needed
# df = df.convert_dtypes()

# COMMAND ----------

print(f"target proportions:\n{df[cfg.target_col].value_counts(normalize=True)}")

# COMMAND ----------

if cfg.split_col:
    print(f"split proportions:\n{df[cfg.split_col].value_counts(normalize=True)}")

# COMMAND ----------

if cfg.sample_weight_col:
    print(f"sample weights: {df[cfg.sample_weight_col].unique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # feature selection

# COMMAND ----------

# databricks freaks out during feature selection if autologging isn't disabled :shrug:
mlflow.autolog(disable=True)

# COMMAND ----------

# load feature selection params from the project config
# HACK: set non-feature cols in params since it's computed outside
# of feature selection config
selection_params = cfg.modeling.feature_selection.model_dump()
selection_params["non_feature_cols"] = cfg.non_feature_cols
logging.info("selection params = %s", selection_params)

# COMMAND ----------

df_selected = modeling.feature_selection.select_features(
    (df.loc[df[cfg.split_col].eq("train"), :] if cfg.split_col else df),
    **selection_params,
)
print(f"rows x cols = {df_selected.shape}")
df_selected.head()

# COMMAND ----------

# HACK: we want to use selected columns for *all* splits, not just train
df = df.loc[:, df_selected.columns]

# COMMAND ----------

# save modeling dataset with all splits
dataio.write.to_delta_table(
    df, cfg.datasets.silver["modeling"].table_path, spark_session=spark
)