import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless test env
import pytest
from sklearn.metrics import roc_auc_score

from edvise.modeling.h2o_ml import evaluation, training


def test_build_roc_curve_table_score_passes_own_rounded_threshold():
    # If only thresholds are rounded, 0.12345 >= round(0.12345, 4) is often False.
    y_true = np.array([1, 0])
    y_scores = np.array([0.12345, 0.0])

    roc_df = evaluation.build_roc_curve_table(y_true, y_scores)
    own_thresh = float(np.round(0.12345, 4))
    row = roc_df.loc[roc_df["threshold"] == own_thresh].iloc[0]
    assert row["true_positive"] == 1


def test_build_roc_curve_table_matches_sklearn_auc_on_random_scores():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=400)
    y_scores = rng.random(400)

    roc_df = evaluation.build_roc_curve_table(y_true, y_scores).sort_values(
        "false_positive_rate"
    )
    auc_table = float(
        np.trapz(roc_df["true_positive_rate"], roc_df["false_positive_rate"])
    )
    assert auc_table == pytest.approx(roc_auc_score(y_true, y_scores), abs=0.01)


def test_create_and_log_h2o_model_comparison(monkeypatch, tmp_path):
    # Fake leaderboard with only GBM models
    fake_lb = pd.DataFrame(
        {
            "model_id": [
                "GBM_lr_annealing_selection_AutoML_2_20250823_00331_select_model",
                "GBM_grid_1_AutoML_2_20250823_00331_model_119",
                "GBM_grid_1_AutoML_2_20250823_00331_model_167",
            ],
            "logloss": [0.5538, 0.5539, 0.5544],
            "auc": [0.7906, 0.7900, 0.7896],
        }
    )

    class DummyAML:
        leaderboard = fake_lb

    # monkeypatch utils._to_pandas to return our fake lb
    monkeypatch.setattr(evaluation.utils, "_to_pandas", lambda _: fake_lb)

    # monkeypatch mlflow.log_figure so it doesn’t try to actually log
    called = {}

    def fake_log_figure(fig, artifact_path):
        called["artifact_path"] = artifact_path

    monkeypatch.setattr(evaluation.mlflow, "log_figure", fake_log_figure)
    monkeypatch.setattr(evaluation.mlflow, "active_run", lambda: True)

    # Run the function
    best = evaluation.create_and_log_h2o_model_comparison(
        DummyAML(), artifact_path="model_comparison.png"
    )

    # Assertions
    assert "framework" in best.columns
    assert set(best["framework"]) <= training.VALID_H2O_FRAMEWORKS
    # Best logloss should be the first row (lowest value)
    assert best.iloc[0]["logloss"] == pytest.approx(min(fake_lb["logloss"]))
    # MLflow log was called with expected path
    assert called["artifact_path"] == "model_comparison.png"
