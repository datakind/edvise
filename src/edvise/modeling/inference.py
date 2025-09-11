import logging

import pandas as pd
import numpy as np
import numpy.typing as npt

import edvise.dataio as dataio
import edvise.modeling as modeling


def top_n_features(
        grouped_features: pd.DataFrame,
        unique_ids: pd.Series,
        grouped_shap_values: npt.NDArray[np.float64] | pd.DataFrame,  # relax input
        features_table_path: str,
        n: int = 10,
    ) -> pd.DataFrame:
        features_table = dataio.read.read_features_table(features_table_path)
        try:
            top_n_shap_features = modeling.automl.inference.top_shap_features(
                features=grouped_features,
                unique_ids=unique_ids,
                shap_values=(
                    grouped_shap_values.values
                    if isinstance(grouped_shap_values, pd.DataFrame)
                    else grouped_shap_values
                ),
                top_n=n,
                features_table=features_table,
            )
            return top_n_shap_features
        except Exception as e:
            logging.error("Error computing top %d shap features table: %s", n, e)
            raise  # keep the signature honest

def features_box_whiskers_table(
        features: pd.DataFrame,
        shap_values: npt.NDArray[np.float64],
        features_table_path: str,
    ) -> pd.DataFrame:
        features_table = dataio.read.read_features_table(features_table_path)
        try:
            feature_boxstats = modeling.automl.inference.top_feature_boxstats(
                features=features,
                shap_values=shap_values,
                features_table=features_table,
            )
            return feature_boxstats

        except Exception as e:
            logging.error("Error computing box features %d shap features table: %s", e)
            return None