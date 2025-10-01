# Databricks notebook source
# MAGIC %md
# MAGIC # SST Make and Explain Predictions
# MAGIC
# MAGIC Fourth step in the process of transforming raw (PDP) data into actionable, data-driven insights for advisors: generate predictions and feature importances for new (unlabeled) data.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
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

# %pip install git+https://github.com/datakind/edvise.git@v0.1.4
# %restart_python

# COMMAND ----------

import logging

import mlflow
import pandas as pd
from databricks.connect import DatabricksSession

from edvise import configs, dataio, modeling
from edvise.modeling import h2o_ml
from py4j.protocol import Py4JJavaError

import h2o

h2o_ml.utils.safe_h2o_init()
h2o.display.toggle_user_tips(False)

# COMMAND ----------

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger
logging.getLogger("h2o").setLevel(
    logging.WARN
)  # ignore h2o logger since it gets verbose

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

try:
    # Get the pipeline type from job definition
    run_type = dbutils.widgets.get("run_type")  # noqa: F821
except Py4JJavaError:
    # Run script interactively
    run_type = "predict"

# COMMAND ----------

# Databricks logs every instance that uses sklearn or other modelling libraries
# to MLFlow experiments... which we don't want
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## import school-specific code

# COMMAND ----------

# project configuration stored as a config file in TOML format
cfg = dataio.read.read_config(
    "./config-TEMPLATE.toml", schema=configs.custom.CustomProjectConfig
)
cfg

# COMMAND ----------

# Load human-friendly PDP feature names
features_table = dataio.read_features_table("./features_table.toml")

# COMMAND ----------

# MAGIC %md
# MAGIC # load artifacts

# COMMAND ----------

df = dataio.read.from_delta_table(
    cfg.datasets.silver["modeling"].table_path,
    spark_session=spark,
)
df.head()

# COMMAND ----------

model = h2o_ml.utils.load_h2o_model(cfg.model.run_id)
model

# COMMAND ----------

model_feature_names = h2o_ml.inference.get_h2o_used_features(model)
logging.info(
    "model uses %s features: %s", len(model_feature_names), model_feature_names
)

# COMMAND ----------

df_train = h2o_ml.evaluation.extract_training_data_from_model(cfg.model.experiment_id)
if cfg.split_col:
    df_train = df_train.loc[df_train[cfg.split_col].eq("train"), :]

# Load and transform using sklearn imputer
imputer = h2o_ml.imputation.SklearnImputerWrapper.load(run_id=cfg.model.run_id)
df_train = imputer.transform(df_train)

# COMMAND ----------

# MAGIC %md
# MAGIC # preprocess data

# COMMAND ----------

if cfg.split_col and cfg.split_col in df.columns:
    df_test = df.loc[df[cfg.split_col].eq("test"), :]
else:
    df_test = df.copy(deep=True)

if run_type == "train":
    df_test = df_test.sample(n=min(200, len(df_test)), random_state=cfg.random_state)

# transform with imputer
df_test = imputer.transform(df_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # make predictions

# COMMAND ----------

features_df = df_test.loc[:, model_feature_names]
unique_ids = df_test[cfg.student_id_col]

# COMMAND ----------

pred_labels, pred_probs = h2o_ml.inference.predict_h2o(
    features_df,
    model=model,
    feature_names=model_feature_names,
    pos_label=cfg.pos_label,
)
print(pred_probs.shape)
pd.Series(pred_probs).describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # explain predictions

# COMMAND ----------

# Sample background data for performance optimization
df_bd = df_train.sample(
    n=min(cfg.inference.background_data_sample, len(df_test)),
    random_state=cfg.random_state,
)

contribs_df = h2o_ml.inference.compute_h2o_shap_contributions(
    model=model,
    df=features_df,
    background_data=df_bd,
)
contribs_df

# COMMAND ----------

# Group one-hot encoding and missing value flags
grouped_contribs_df = h2o_ml.inference.group_shap_values(contribs_df)
grouped_features = h2o_ml.inference.group_feature_values(features_df)

if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_id=cfg.model.run_id):
    # Create & log SHAP summary plot (default to group missing flags)
    h2o_ml.inference.plot_grouped_shap(
        contribs_df=contribs_df,
        features_df=features_df,
        original_dtypes=imputer.input_dtypes,
    )

    # Create & log ranked features by SHAP magnitude
    selected_features_df = modeling.automl.inference.generate_ranked_feature_table(
        grouped_features,
        grouped_contribs_df.to_numpy(),
        features_table=features_table,
        metadata=False,
    )

selected_features_df

# COMMAND ----------

# MAGIC %md
# MAGIC # finalize results

# COMMAND ----------

# Provide output using top features, SHAP values, and support scores
result = modeling.automl.inference.select_top_features_for_display(
    grouped_features,
    unique_ids,
    pred_probs,
    grouped_contribs_df.to_numpy(),
    n_features=10,
    features_table=features_table,
    needs_support_threshold_prob=cfg.inference.min_prob_pos_label,
)
result

# COMMAND ----------

# save sample advisor output dataset
dataio.write.to_delta_table(
    result, cfg.datasets.gold.advisor_output.table_path, spark_session=spark
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Front-End Tables

# COMMAND ----------

# Log MLFlow confusion matrix & roc table figures in silver schema
with mlflow.start_run(run_id=cfg.model.run_id):
    confusion_matrix = modeling.evaluation.log_confusion_matrix(
        institution_id=cfg.institution_id,
        automl_run_id=cfg.model.run_id,
    )

    # Log roc curve table for front-end
    roc_logs = h2o_ml.evaluation.log_roc_table(
        institution_id=cfg.institution_id,
        automl_run_id=cfg.model.run_id,
        modeling_dataset_name=cfg.datasets.silver["modeling"].train_table_path,
    )

# COMMAND ----------

shap_feature_importance = modeling.automl.inference.generate_ranked_feature_table(
    features=grouped_features,
    shap_values=grouped_contribs_df.to_numpy(),
    features_table=features_table,
    metadata=False,
)
shap_feature_importance

# COMMAND ----------

# save sample advisor output dataset
dataio.write.to_delta_table(
    shap_feature_importance,
    f"staging_sst_01.{cfg.institution_id}_silver.training_{cfg.model.run_id}_shap_feature_importance",
    spark_session=spark,
)

# COMMAND ----------

support_score_distribution = modeling.automl.inference.support_score_distribution_table(
    df_serving=grouped_features,
    unique_ids=unique_ids,
    pred_probs=pred_probs,
    shap_values=grouped_contribs_df,
    inference_params=cfg.inference.dict(),
)
support_score_distribution

# COMMAND ----------

# save sample advisor output dataset
dataio.write.to_delta_table(
    support_score_distribution,
    f"staging_sst_01.{cfg.institution_id}_silver.training_{cfg.model.run_id}_support_overview",
    spark_session=spark,
)
