import pandas as pd
import pytest

from edvise.modeling import evaluation


@pytest.fixture
def patch_mlflow(monkeypatch):
    def _patch(mock_df, target="mlflow.search_runs"):
        monkeypatch.setattr(target, lambda *args, **kwargs: mock_df)

    return _patch


def test_check_array_of_arrays_true():
    input_array = pd.Series([[1, 0, 1], [0, 1, 0]])
    assert evaluation._check_array_of_arrays(input_array)


def test_check_array_of_arrays_false():
    input_array = pd.Series([1, 0, 1])
    assert not evaluation._check_array_of_arrays(input_array)


@pytest.fixture
def mock_runs_df():
    """Mock MLflow run data for testing."""
    return pd.DataFrame(
        {
            "run_id": ["r1", "r2", "r3"],
            "tags.mlflow.runName": ["run_1", "run_2", "run_3"],
            "metrics.test_roc_auc": [0.80, 0.60, 0.90],
            "metrics.test_recall_score": [0.70, 0.95, 0.60],
            "metrics.val_log_loss": [0.25, 0.20, 0.30],
        }
    )


@pytest.mark.parametrize(
    "metrics, expected_run_name",
    [
        (["test_roc_auc"], "run_3"),
        (["test_recall_score"], "run_2"),
        (["val_log_loss"], "run_2"),
        (["test_roc_auc", "val_log_loss"], "run_1"),
    ],
)
def test_get_top_runs_balanced(metrics, expected_run_name, mock_runs_df, patch_mlflow):
    mock_df = pd.DataFrame(
        {
            "run_id": ["r1", "r2", "r3"],
            "tags.mlflow.runName": ["run_1", "run_2", "run_3"],
            "metrics.test_roc_auc": [0.80, 0.60, 0.90],
            "metrics.test_recall_score": [0.70, 0.95, 0.60],
            "metrics.val_log_loss": [0.25, 0.20, 0.30],
        }
    )

    patch_mlflow(mock_df)

    top = evaluation.get_top_runs(
        experiment_id="dummy",
        optimization_metrics=metrics,
        topn_runs_included=1,
    )

    assert list(top.keys())[0] == expected_run_name


@pytest.mark.parametrize(
    "metrics, expected_top",
    [
        (["test_recall_score"], "run_2"),
        (["val_log_loss"], "run_2"),
        (["test_roc_auc"], "run_3"),
        (["test_roc_auc", "val_log_loss"], "run_1"),
    ],
)
def test_get_top_runs_parametrized(metrics, expected_top, mock_runs_df, patch_mlflow):
    patch_mlflow(mock_runs_df)

    top = evaluation.get_top_runs(
        experiment_id="dummy",
        optimization_metrics=metrics,
        topn_runs_included=1,
    )

    assert list(top.keys())[0] == expected_top
