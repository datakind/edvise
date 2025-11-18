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

# %pip install git+https://github.com/datakind/edvise.git@v0.1.6
# %restart_python

# COMMAND ----------

import logging
import sys

import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
from databricks.connect import DatabricksSession
from py4j.protocol import Py4JJavaError

from edvise import dataio, configs, data_audit

try:
  # Get the pipeline type from job definition
  run_type = dbutils.widgets.get("run_type") # noqa: F821
except Py4JJavaError:
  # Run script interactively
  run_type = 'train'

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
# MAGIC # read raw datasets

# COMMAND ----------

student_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze['raw_cohort'].file_path,
    spark_session=spark,
)
course_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze['raw_course'].file_path,
    spark_session=spark,
)
semester_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze['raw_semester'].file_path,
    spark_session=spark,
)
