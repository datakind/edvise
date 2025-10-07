import os
import io
import json
import tempfile
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from edvise.modeling.h2o_ml.calibration import SklearnCalibratorWrapper


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def probs_small(rng):
    # 200 samples -> should select Platt by default (min_n=1000)
    p = np.clip(rng.beta(2, 5, size=200), 1e-6, 1 - 1e-6)
    y = (p + rng.normal(0, 0.15, size=p.size) > 0.5).astype(int)
    return p, y


@pytest.fixture
def probs_large_balanced(rng):
    # 5000 samples, ~balanced -> should select Isotonic by default
    p = np.clip(rng.beta(2.5, 2.5, size=5000), 1e-6, 1 - 1e-6)
    y = (p + rng.normal(0, 0.1, size=p.size) > 0.5).astype(int)
    # force near-balance
    return p, y


def test_choose_method_small_defaults_to_platt(probs_small):
    _, y = probs_small
    assert SklearnCalibratorWrapper._choose_method(y) == "platt"


def test_choose_method_large_defaults_to_isotonic(probs_large_balanced):
    _, y = probs_large_balanced
    assert SklearnCalibratorWrapper._choose_method(y) == "isotonic"


def test_choose_method_edge_thresholds():
    n = 1000
    y = np.array([0, 1] * (n // 2))
    assert (
        SklearnCalibratorWrapper._choose_method(y, min_n=1000, min_pos_neg=200)
        == "isotonic"
    )

    y2 = np.r_[np.ones(199, int), np.zeros(3000, int)]
    assert (
        SklearnCalibratorWrapper._choose_method(y2, min_n=1000, min_pos_neg=200)
        == "platt"
    )


def test_transform_raises_if_unfitted():
    cal = SklearnCalibratorWrapper()
    with pytest.raises(RuntimeError):
        cal.transform(np.array([0.1, 0.9]))


def test_save_raises_if_unfitted(tmp_path):
    cal = SklearnCalibratorWrapper()
    with pytest.raises(RuntimeError):
        cal.save(artifact_path=str(tmp_path))


def test_fit_sets_method_and_model_isotonic(probs_large_balanced, monkeypatch):
    import mlflow

    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)

    p, y = probs_large_balanced
    cal = SklearnCalibratorWrapper().fit(p, y)
    assert cal.method == "isotonic"
    assert isinstance(cal.model, IsotonicRegression)


def test_fit_sets_method_and_model_platt(probs_small, monkeypatch):
    import mlflow

    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)

    p, y = probs_small
    cal = SklearnCalibratorWrapper().fit(p, y)
    assert cal.method == "platt"
    assert isinstance(cal.model, LogisticRegression)


def test_transform_shape_and_bounds(probs_large_balanced, monkeypatch):
    import mlflow

    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)

    p, y = probs_large_balanced
    cal = SklearnCalibratorWrapper().fit(p, y)
    out = cal.transform(p)
    assert out.shape == p.shape
    assert out.dtype.kind == "f"
    assert np.all((out >= 0.0) & (out <= 1.0))  # clipped & valid probs


def test_isotonic_monotonicity(probs_large_balanced, monkeypatch):
    """Isotonic regression should be non-decreasing in input probability."""
    import mlflow

    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)

    p, y = probs_large_balanced
    cal = SklearnCalibratorWrapper().fit(p, y)
    if cal.method != "isotonic":
        pytest.skip("This test validates isotonic monotonicity only.")
    order = np.argsort(p)
    cal_p = cal.transform(p[order])
    diffs = np.diff(cal_p)
    assert np.all(diffs >= -1e-12)  # numerical tolerance


# ---------- MLflow artifact roundtrip ----------


def test_save_and_load_roundtrip_tmpdir(probs_small, tmp_path, monkeypatch):
    """Ensure save() writes a bundle and load() reconstructs the calibrator."""
    # --- Fit (Platt on small data) ---
    import mlflow

    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)

    p, y = probs_small
    cal = SklearnCalibratorWrapper().fit(p, y)
    assert cal.method == "platt"

    # --- Intercept mlflow.log_artifact to copy file to a temp store we control ---
    saved_dir = tmp_path / "mlflow_artifacts" / "calibration"
    saved_dir.mkdir(parents=True, exist_ok=True)

    def fake_log_artifact(local_path, artifact_path="calibration"):
        # local_path is the joblib file written in a tmpdir; copy it to saved_dir
        dest = saved_dir / os.path.basename(local_path)
        with open(local_path, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())

    monkeypatch.setattr(mlflow, "log_artifact", fake_log_artifact)

    # --- Save bundle ---
    cal.save(artifact_path="calibration")
    bundle_path = saved_dir / "calibrator.joblib"
    assert bundle_path.exists()

    # --- Monkeypatch mlflow.artifacts.download_artifacts so .load() can find our file ---
    class FakeArtifacts:
        @staticmethod
        def download_artifacts(run_id, artifact_path, dst_path=None):
            # Return the path to the saved bundle regardless of args
            return str(bundle_path)

    monkeypatch.setattr(mlflow, "artifacts", FakeArtifacts)

    # --- Load and compare ---
    loaded = SklearnCalibratorWrapper.load(run_id="dummy", artifact_path="sklearn_calibration")
    assert loaded is not None
    assert loaded.method == "platt"
    assert isinstance(loaded.model, LogisticRegression)

    # sanity: applying loaded transform works and is bounded
    out = loaded.transform(np.array([0.05, 0.5, 0.95]))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_load_returns_none_when_missing(monkeypatch):
    """If artifact is missing or raises, load() should return None (not raise)."""
    import mlflow

    class FakeArtifacts:
        @staticmethod
        def download_artifacts(*a, **k):
            raise FileNotFoundError("no such artifact")

    monkeypatch.setattr(mlflow, "artifacts", FakeArtifacts)

    loaded = SklearnCalibratorWrapper.load(
        run_id="whatever", artifact_path="calibration"
    )
    assert loaded is None


# ---------- Integration-ish sanity check ----------


def test_calibrated_threshold_application(probs_small, monkeypatch):
    """Applying calibrated probs at 0.5 produces valid 0/1 decisions (no crash path)."""
    import mlflow

    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: None)

    p, y = probs_small
    cal = SklearnCalibratorWrapper().fit(p, y)
    p_cal = cal.transform(p)
    y_hat = (p_cal >= 0.5).astype(int)
    assert set(np.unique(y_hat)).issubset({0, 1})
    assert y_hat.shape == y.shape
