import pandas as pd
import numpy as np
import os
import pytest
from unittest import mock
from edvise.modeling.h2o_ml import imputation
from sklearn.pipeline import Pipeline
import logging


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "num_low_skew": [1.0, 2.0, np.nan, 4.0],
            "num_high_skew": [1, 1000, np.nan, 3],
            "bool_col": [True, False, None, True],
            "cat_col": ["a", "b", None, "a"],
            "text_col": ["x", None, "y", "x"],
            "complete_col": [10, 20, 30, 40],
        },
        index=["s1", "s2", "s3", "s4"],
    )


def test_fit_and_transform_shapes_and_columns(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(sample_df)
    result = imputer.transform(sample_df)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == sample_df.shape[0]

    original_cols = set(sample_df.columns)
    result_cols = set(result.columns)

    assert original_cols.issubset(result_cols)

    extra_cols = result_cols - original_cols
    assert all(col.endswith("_missing_flag") for col in extra_cols)


def test_transform_raises_if_not_fitted(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    with pytest.raises(ValueError, match="Pipeline not fitted"):
        imputer.transform(sample_df)


def test_missing_values_filled(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(sample_df)
    result = imputer.transform(sample_df)

    assert result.isnull().sum().sum() == 0  # No missing values remain


def test_pipeline_instance_and_step_names(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    pipeline = imputer.fit(sample_df)
    assert isinstance(pipeline, Pipeline)
    assert "imputer" in dict(pipeline.named_steps)


def test_missing_flags_added_only_for_missing_columns(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(sample_df)
    result = imputer.transform(sample_df)

    expected_flags = {
        "num_low_skew_missing_flag",
        "num_high_skew_missing_flag",
        "bool_col_missing_flag",
        "cat_col_missing_flag",
        "text_col_missing_flag",
    }

    for flag_col in expected_flags:
        assert flag_col in result.columns
        assert set(result[flag_col].unique()).issubset({0, 1})

    assert "complete_col_missing_flag" not in result.columns


@mock.patch("mlflow.active_run")
@mock.patch("mlflow.start_run")
@mock.patch("mlflow.log_artifact")
@mock.patch("mlflow.end_run")
def test_pipeline_logged_to_mlflow(
    mock_end_run,
    mock_log_artifact,
    mock_start_run,
    mock_active_run,
    sample_df,
):
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(sample_df)

    imputer.log_pipeline(artifact_path="test_artifact_path")

    # Expect pipeline, input_dtypes, input_feature_names, missing_flag_cols to be logged
    assert mock_log_artifact.call_count == 4

    artifact_paths = [
        call.kwargs["artifact_path"] for call in mock_log_artifact.call_args_list
    ]
    assert all(path == "test_artifact_path" for path in artifact_paths)

    logged_filenames = [
        os.path.basename(call.args[0]) for call in mock_log_artifact.call_args_list
    ]
    assert "imputer_pipeline.joblib" in logged_filenames
    assert "input_dtypes.json" in logged_filenames
    assert "input_feature_names.json" in logged_filenames


def test_newly_missing_warn_logs_and_no_new_flag(sample_df, caplog):
    """
    Columns that were clean at fit (complete_col) become missing at inference:
    - should NOT raise when on_new_missing='warn'
    - should log a WARNING mentioning the column
    - imputer should fill the NaN
    - should NOT add a *_missing_flag for complete_col (we only keep fit-time flags)
    """
    df_train = sample_df.copy()  # complete_col is clean at fit
    df_infer = sample_df.copy()
    df_infer.loc["s2", "complete_col"] = np.nan  # introduce new missing

    imputer = imputation.SklearnImputerWrapper(on_new_missing="warn")
    imputer.fit(df_train)

    # be explicit about which logger we capture
    caplog.set_level(logging.WARNING, logger=imputation.LOGGER.name)

    result = imputer.transform(df_infer)

    # log message and column name present
    msgs = " ".join(r.message for r in caplog.records)
    assert "New missingness" in msgs and "complete_col" in msgs

    # still no NaNs
    assert result.isnull().sum().sum() == 0

    # no new flag persisted for complete_col
    assert "complete_col_missing_flag" not in result.columns


@mock.patch("mlflow.log_artifact")
@mock.patch("mlflow.active_run", return_value=True)
def test_newly_missing_error_raises_and_logs_artifact(
    mock_active_run, mock_log_artifact, sample_df
):
    """
    With on_new_missing='error', newly-missing on a clean-at-fit column should:
    - log (attempt) an MLflow artifact with unique new_missing_report_*.json
    - raise ValueError mentioning the column
    """
    df_train = sample_df.copy()
    df_infer = sample_df.copy()
    df_infer.loc["s2", "complete_col"] = np.nan

    imputer = imputation.SklearnImputerWrapper(
        on_new_missing="error", log_drift_to_mlflow=True
    )
    imputer.fit(df_train)

    with pytest.raises(ValueError, match="complete_col"):
        imputer.transform(df_infer)

    # artifact was attempted
    assert mock_log_artifact.called
    # first positional arg is a file path; basename should look like new_missing_report_*.json
    basenames = [
        os.path.basename(call.args[0]) for call in mock_log_artifact.call_args_list
    ]
    assert any(
        name.startswith("new_missing_report_") and name.endswith(".json")
        for name in basenames
    )


def test_impute_clean_at_fit_numeric_uses_trained_stat(sample_df):
    """
    A column that was clean at fit but missing at inference should be imputed
    using the trained statistic (mean/median). For complete_col=[10,20,30,40], mean=25.
    """
    df_train = sample_df.copy()
    df_infer = sample_df.copy()
    df_infer.loc["s3", "complete_col"] = np.nan  # create a single missing

    imputer = imputation.SklearnImputerWrapper(on_new_missing="warn")
    imputer.fit(df_train)
    result = imputer.transform(df_infer)

    # Expect mean fill (skew likely small): 25.0
    assert pytest.approx(result.loc["s3", "complete_col"], rel=0, abs=1e-9) == 25.0


def test_suffix_handling_when_flag_like_names_present():
    """
    Ensure suffix parsing is robust: a base feature whose name contains the token
    '_missing_flag' internally should not confuse drift detection.
    """
    df_train = pd.DataFrame(
        {
            "feature_missing_flag_base": [
                1,
                2,
                np.nan,
                4,
            ],  # has missing at fit -> gets flag
            "clean_col": [10, 20, 30, 40],  # clean at fit
        },
        index=["s1", "s2", "s3", "s4"],
    )
    df_infer = df_train.copy()
    df_infer.loc["s2", "clean_col"] = np.nan  # newly missing on clean col

    imputer = imputation.SklearnImputerWrapper(on_new_missing="warn")
    imputer.fit(df_train)
    out = imputer.transform(df_infer)

    # flag for the feature with internal token exists (from fit)
    assert "feature_missing_flag_base_missing_flag" in out.columns

    # we should NOT accidentally treat 'feature_missing_flag_base' as clean
    # (i.e., no spurious drift error/raise since warn mode, just ensure result has no NaNs)
    assert out.isnull().sum().sum() == 0
