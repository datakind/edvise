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

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from databricks.connect import DatabricksSession
from py4j.protocol import Py4JJavaError
from databricks import dbutils 

from edvise import dataio, configs

