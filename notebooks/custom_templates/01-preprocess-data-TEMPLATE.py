# Databricks notebook source
# MAGIC %md
# MAGIC # SST Preprocess Custom Data
# MAGIC
# MAGIC First step in the process of transforming raw data into actionable, data-driven insights for advisors: load raw data, build a schema contract to enhance data & pipeline reliability, and ensure limited training-inference skew.
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

# %pip install git+https://github.com/datakind/edvise.git@v0.1.7
# %restart_python

# COMMAND ----------

import logging
import typing as t

import pandas as pd
from databricks.connect import DatabricksSession
from py4j.protocol import Py4JJavaError

from edvise import dataio, configs
from edvise.data_audit import custom_cleaning

# NOTE: You may want to add term order here
# from TODO.helpers import create_term_order

try:
    # Get the pipeline type from job definition
    run_type = dbutils.widgets.get("run_type")  # noqa: F821
except Py4JJavaError:
    # Run script interactively
    run_type = "train"

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

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
# MAGIC ## read raw datasets

# COMMAND ----------

student_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze["raw_students"].file_path,
    spark_session=spark,
)
course_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze["raw_course"].file_path,
    spark_session=spark,
)
semester_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze["raw_semester"].file_path,
    spark_session=spark,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## clean datasets & build schema contract

# COMMAND ----------

# Map logical dataset names -> (bronze_config_key, raw_dataframe)
DF_MAP: t.Dict[str, t.Tuple[str, pd.DataFrame]] = {
    "student_df": ("raw_student", student_raw_df),
    "course_df": ("raw_course", course_raw_df),
    "semester_df": ("raw_semester", semester_raw_df),
    # Add more as needed...
}

# Optional per-dataset term order + dedupe hooks
term_order_by_dataset: dict[str, tuple[custom_cleaning.TermOrderFn, str]] = {
    # "student_df":  (create_student_term_order, "term"),
    # "semester_df": (create_semester_term_order, "term_code"),
}
dedupe_fn_by_dataset: dict[str, custom_cleaning.DedupeFn] = {
    # "course_df": dedupe_course_rows,
}

# Pull cleaning config
cleaning_cfg = getattr(getattr(cfg, "preprocessing", None), "cleaning", None)
cleaning_cfg

# COMMAND ----------

# Clean bronze datasets
# Either use one term_order_fn/dedupe_fn for all datasets
# OR use per-dataset hooks from above
cleaned = custom_cleaning.clean_bronze_datasets(
    cfg=cfg,
    df_map=DF_MAP,
    run_type=run_type,  # "train" or "predict"
    cleaning_cfg=cleaning_cfg,
    term_order_fn=None,
    term_col="term",
    dedupe_fn=None,
    term_order_by_dataset=term_order_by_dataset,
    dedupe_fn_by_dataset=dedupe_fn_by_dataset,
)

# Build specs
datasets_spec = custom_cleaning.build_datasets_from_bronze(cfg, DF_MAP)

# Build or load schema contract
schema_contract = custom_cleaning.load_or_build_schema_contract(
    cfg=cfg,
    run_type=run_type,
    cleaned=cleaned,
    specs=datasets_spec,
)

# Enforce schema at inference (or reuse cleaned at train-time)
if run_type == "train":
    aligned = cleaned
else:
    aligned = custom_cleaning.enforce_schema_contract(cleaned, schema_contract)

student_df = aligned["student_df"]
course_df = aligned["course_df"]
semester_df = aligned["semester_df"]
