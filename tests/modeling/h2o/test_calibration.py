import os
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from edvise.modeling.h2o_ml.calibration import SklearnCalibratorWrapper


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def probs_small(rng):
    # 200 samples with noisy labels → calibration should usually help (λ>0)
    p = np.clip(rng.beta(2, 5, size=1000), 1e-6, 1 - 1e-6)
    y = (p + rng.normal(0, 0.15, size=p.size) > 0.5).astype(int)
    return p, y


@pytest.fixture
def probs_large_balanced(rng):
    # Larger set—still only used for general behavior now (no isotonic path anymore)
    p = np.clip(rng.beta(2.5, 2.5, size=5000), 1e-6, 1 - 1e-6)
    y = (p + rng.normal(0, 0.1, size=p.size) > 0.5).astype(int)
    return p, y


def test_transform_identity_when_unfitted():
    # Unfitted calibrator returns raw probabilities (no-op)
    cal = SklearnCalibratorWrapper()
    p = np.array([0.1, 0.9, 0.3])
    out = cal.transform(p)
    assert np.allclose(out, p)


def test_save_raises_if_unfitted(tmp_path, monkeypatch):
    # Save requires a fitted calibrator
    cal = SklearnCalibratorWrapper()
    with pytest.raises(RuntimeError):
        cal.save(artifact_path=str(tmp_path))


def test_fit_sets_method_model_and_lambda_when_helpful(probs_small):
    p, y = probs_small
    cal = SklearnCalibratorWrapper().fit(p, y)
    assert cal.method in ("platt_logits", "passthrough")
    # For this synthetic setting we expect calibration to help:
    assert cal.method == "platt_logits"
    assert isinstance(cal.model, LogisticRegression)
    assert 0.0 < cal.lam <= 1.0


def test_transform_shape_and_bounds(probs_small):
    p, y = probs_small
    cal = SklearnCalibratorWrapper().fit(p, y)
    out = cal.transform(p)
    assert out.shape == p.shape
    assert out.dtype.kind == "f"
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_no_improvement_means_no_calibration(rng):
    """
    Construct an almost-perfectly calibrated case: y is a Bernoulli draw from p,
    which makes Platt unlikely to improve Brier score by > min_improve.
    Expect method='passthrough', lam=0, and identity transform.
    """
    n = 2000
    p = np.clip(rng.beta(2.5, 2.5, size=n), 1e-6, 1 - 1e-6)
    y = rng.binomial(1, p, size=n)  # labels drawn from p -> already calibrated

    cal = SklearnCalibratorWrapper().fit(p, y)

    assert cal.lam == 0.0
    assert cal.method == "passthrough"

    out = cal.transform(p)
    assert np.allclose(out, p)


# ---------- MLflow artifact roundtrip ----------


def test_save_and_load_roundtrip_tmpdir(probs_small, tmp_path, monkeypatch):
    """Ensure save() writes a bundle and load() reconstructs the calibrator."""
    import mlflow
    import joblib

    p, y = probs_small
    cal = SklearnCalibratorWrapper().fit(p, y)
    assert cal.method == "platt_logits"

    # Intercept mlflow.log_artifact to copy the joblib into our temp dir
    saved_dir = tmp_path / "mlflow_artifacts" / "sklearn_calibration"
    saved_dir.mkdir(parents=True, exist_ok=True)

    def fake_log_artifact(local_path, artifact_path="sklearn_calibration"):
        dest = saved_dir / os.path.basename(local_path)
        with open(local_path, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())

    monkeypatch.setattr(mlflow, "log_artifact", fake_log_artifact)

    # Save bundle
    cal.save(artifact_path="sklearn_calibration")
    bundle_path = saved_dir / "calibrator.joblib"
    assert bundle_path.exists()

    # Monkeypatch download_artifacts to return our saved bundle
    class FakeArtifacts:
        @staticmethod
        def download_artifacts(run_id, artifact_path, dst_path=None):
            return str(bundle_path)

    monkeypatch.setattr(mlflow, "artifacts", FakeArtifacts)

    # Load and sanity-check
    loaded = SklearnCalibratorWrapper.load(
        run_id="dummy", artifact_path="sklearn_calibration"
    )
    assert loaded is not None
    assert loaded.method == "platt_logits"
    assert isinstance(loaded.model, LogisticRegression)

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
        run_id="whatever", artifact_path="sklearn_calibration"
    )
    assert loaded is None


# ---------- Integration-ish sanity check ----------


def test_calibrated_threshold_application(probs_small):
    """Applying calibrated probs at 0.5 produces valid 0/1 decisions."""
    p, y = probs_small
    cal = SklearnCalibratorWrapper().fit(p, y)
    p_cal = cal.transform(p)
    y_hat = (p_cal >= 0.5).astype(int)
    assert set(np.unique(y_hat)).issubset({0, 1})
    assert y_hat.shape == y.shape
