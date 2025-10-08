import os
import tempfile
import typing as t
import joblib
import mlflow
import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)


class SklearnCalibratorWrapper:
    """Post modeling probability calibrator; also, auto-selects isotonic or Platt (logistic regression) based on validation size."""

    def __init__(self):
        self.method: str | None = None
        self.model = None
        LOGGER.info("Initiating calibrator for H2O")

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> "SklearnCalibratorWrapper":
        """
        Auto-select calibration method (Platt vs Isotonic) based on val size/imbalance,
        then fit the appropriate model.
        """
        p = np.asarray(p_raw, float).ravel()
        y = np.asarray(y_true, int).ravel()

        # infer method
        self.method = self._choose_method(y)
        mlflow.log_param("calibration_method", self.method)

        # fit model
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        else:
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(p.reshape(-1, 1), y)
            self.model = lr

        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        """Apply fitted calibration mapping to new probabilities."""
        if self.model is None:
            raise RuntimeError("Calibrator not fitted")
        p = np.asarray(p_raw, float).ravel()
        if self.method == "isotonic":
            return np.array(self.model.transform(p))
        return np.array(self.model.predict_proba(p.reshape(-1, 1))[:, 1])

    def save(self, artifact_path: str = "calibration") -> None:
        """Save calibration model + metadata as MLflow artifact."""
        if self.model is None:
            raise RuntimeError("Calibrator not fitted")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "calibrator.joblib")
            joblib.dump({"method": self.method, "model": self.model}, path)
            mlflow.log_artifact(path, artifact_path=artifact_path)

    @staticmethod
    def _choose_method(
        y_val: np.ndarray, *, min_n: int = 1000, min_pos_neg: int = 200
    ) -> str:
        """Platt for small/imbalanced validation, Isotonic otherwise."""
        y = np.asarray(y_val).astype(int).ravel()
        n = y.size
        pos = int(y.sum())
        neg = n - pos
        if n < min_n or min(pos, neg) < min_pos_neg:
            return "platt"
        return "platt"

    @classmethod
    def load(
        cls, run_id: str, artifact_path: str = "calibration"
    ) -> t.Optional["SklearnCalibratorWrapper"]:
        """
        Load a fitted calibrator from MLflow if it exists.
        Returns a SklearnCalibratorWrapper instance or None if not found.
        """
        try:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"{artifact_path}/calibrator.joblib"
            )
            bundle = joblib.load(local)
            inst = cls()
            inst.method = bundle["method"]
            inst.model = bundle["model"]
            LOGGER.info(f"Loaded calibrator from run {run_id} (method={inst.method})")
            return inst
        except Exception as e:
            # Handles missing artifacts or joblib load errors
            LOGGER.info(
                f"No calibrator found for run {run_id}: {e}. Model calibration was not performed."
            )
            return None
