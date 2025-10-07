import os
import tempfile
import joblib
import mlflow
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class SklearnCalibrationWrapper:
    """Post-hoc probability calibrator: auto-selects isotonic or Platt (logistic) based on validation size."""

    def __init__(self):
        self.method: str | None = None
        self.model = None

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> "SklearnCalibrationWrapper":
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
            return self.model.transform(p)
        return self.model.predict_proba(p.reshape(-1, 1))[:, 1]

    def save(self, artifact_path: str = "calibration") -> None:
        """Save calibration model + metadata as MLflow artifact."""
        if self.model is None:
            raise RuntimeError("Calibrator not fitted")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "calibrator.joblib")
            joblib.dump({"method": self.method, "model": self.model}, path)
            mlflow.log_artifact(path, artifact_path=artifact_path)

    @staticmethod
    def _choose_method(y_val: np.ndarray, *, min_n: int = 1000, min_pos_neg: int = 200) -> str:
        """Heuristic: Platt for small/imbalanced validation, Isotonic otherwise."""
        y = np.asarray(y_val).astype(int).ravel()
        n = y.size
        pos = int(y.sum())
        neg = n - pos
        if n < min_n or min(pos, neg) < min_pos_neg:
            return "platt"
        return "isotonic"

    @classmethod
    def load(cls, run_id: str, artifact_path: str = "calibration") -> "SklearnCalibrationWrapper":
        """Load a fitted calibrator from MLflow."""
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"{artifact_path}/calibrator.joblib"
        )
        bundle = joblib.load(local)
        inst = cls()
        inst.method = bundle["method"]
        inst.model = bundle["model"]
        return inst
