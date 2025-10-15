import os
import tempfile
import typing as t
import logging
import joblib
import mlflow
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from scipy.special import logit, expit

LOGGER = logging.getLogger(__name__)


class SklearnCalibratorWrapper:
    """
    Lightweight Platt (logistic) probability calibrator with automatic tuning (lambda-tuning).

    We apply calibration when our model is underpredicting (calibration curve sits under the 1:1 slope line).
    The intent behind adding calibration is to help improve recall. We avoid calibration when our model
    is overpredicting, since we want to capture as many students as possible for Edvise.

    This class fits a logistic regression model on the logits of predicted probabilities
    from a base classifier (i.e., Platt scaling). It then automatically tunes a tuning
    parameter lambda ∈ {0.25, 0.5, 0.75, 1.0} to avoid overcorrection by mixing calibrated and
    original probabilities. The equation is the following:

        p_final = (1 - lambda) * p_raw + lambda * p_platt

    The best lambda minimizes the Brier score on the validation set.
    Calibration is applied only if the best lambda improves the Brier score by a small margin. This
    is a guardrail, so that we only apply calibrator if it actually improves calibration compared
    to our raw probabilities from our model.

    ---
    Key behaviors:
    - Always uses Platt calibration. We avoid isotonic, so that probabilities stay smooth and not discrete.
    - Automatically tunes lambda to avoid overcorrection.
    - Skips calibration if it doesn't improve Brier score meaningfully.

    ---
    Attributes:
    - method: str | None — "platt_logits" or "none".
    - model: fitted LogisticRegression or None.
    - lam: float — mixing coefficient (1.0 = full calibration (full Platt), 0.0 = no calibration).
    """

    def __init__(self):
        self.method: str | None = None
        self.model = None
        self.lam: float = 0.0
        self._min_improve = 1e-3

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> "SklearnCalibratorWrapper":
        """
        Fit a logistic (Platt) calibration model and tune lambda to minimize Brier score.

        Parameters:
            p_raw: Raw predicted probabilities from the base model.
            y_true: True binary labels (0/1).

        Returns:
            Self, fitted in place.
        """
        p = np.asarray(p_raw, float).ravel()
        y = np.asarray(y_true, int).ravel()

        # Minimum data guard -> avoids noisy fits on tiny or imbalanced validation datasets
        n = y.size
        pos = int(y.sum())
        neg = n - pos
        if n < 100 or min(pos, neg) < 50:
            self.method = "passthrough"
            self.model = None
            self.lam = 0.0
            LOGGER.info(
                f"Calibrator skipped (n={n}, pos={pos}, neg={neg} too small). "
                f"method={self.method}, lambda={self.lam:.2f}"
            )
            return self

        # Compute base score (lower = better)
        base_brier = brier_score_loss(y, p)

        # Adaptive lambda grid & C regularization based on sample size
        lam_grid: tuple[float, ...] | None = None
        if n < 1000:
            lam_grid = (0.25, 0.5)
            C_reg = 0.01
        elif n < 5000:
            lam_grid = (0.25, 0.5, 0.75)
            C_reg = 0.1
        else:
            lam_grid = (0.2, 0.4, 0.6, 0.8, 1.0)
            C_reg = 1.0

        # Fit Platt on logits
        z = self._safe_logit(p)
        lr = LogisticRegression(solver="lbfgs", C=C_reg)
        lr.fit(z.reshape(-1, 1), y)

        self.method = "platt_logits"
        self.model = lr

        # Get calibrated probabilities at lambda = 1
        p_cal = expit(lr.decision_function(z.reshape(-1, 1)))

        # Tune lambda
        lam_best = lam_grid[0]
        score_best = self._score(y, self._tune(p, p_cal, lam_best))
        for lam in lam_grid[1:]:
            score = self._score(y, self._tune(p, p_cal, lam))
            if score < score_best:
                score_best, lam_best = score, lam

        # Only apply if Brier score improves meaningfully
        if (base_brier - score_best) < self._min_improve:
            self.method = "passthrough"
            self.model = None
            self.lam = 0.0
        else:
            self.lam = float(lam_best)

        # Log summary
        LOGGER.info(
            f"Calibrator method={self.method}, lambda={self.lam:.2f}, "
            f"Brier score improvement={(base_brier - score_best):.6f}"
        )

        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        """
        Apply the fitted calibrator to new predicted probabilities.

        Parameters:
            p_raw: Array of predicted probabilities.

        Returns:
            Calibrated probabilities (or raw if no calibration was applied).
        """
        p = np.asarray(p_raw, float).ravel()

        if self.method != "platt_logits" or self.model is None or self.lam == 0.0:
            return p

        z = self._safe_logit(p)
        p_cal = expit(self.model.decision_function(z.reshape(-1, 1)))
        return self._tune(p, p_cal, self.lam)

    def save(self, artifact_path: str = "sklearn_calibrator") -> None:
        """Save calibration model, lambda, and metadata as an MLflow artifact."""
        if self.method is None:
            raise RuntimeError("Calibrator not fitted")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "calibrator.joblib")
            joblib.dump(
                {"method": self.method, "model": self.model, "lam": self.lam},
                path,
            )
            mlflow.log_artifact(path, artifact_path=artifact_path)

    @classmethod
    def load(
        cls, run_id: str, artifact_path: str = "sklearn_calibrator"
    ) -> t.Optional["SklearnCalibratorWrapper"]:
        """Load a saved calibrator from MLflow. Returns None if not found."""
        try:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"{artifact_path}/calibrator.joblib"
            )
            bundle = joblib.load(local)
            inst = cls()
            inst.method = bundle.get("method")
            inst.model = bundle.get("model")
            inst.lam = float(bundle.get("lam", 0.0))
            LOGGER.info(
                f"Loaded calibrator (method={inst.method}, lambda={inst.lam:.2f}) from run {run_id}"
            )
            return inst
        except Exception as e:
            LOGGER.info(f"No calibrator for run {run_id}: {e}")
            return None

    @staticmethod
    def _score(y: np.ndarray, p: np.ndarray) -> float:
        """Compute Brier score (lower is better)."""
        return float(brier_score_loss(y, p))

    @staticmethod
    def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Avoid infinities for probabilities near 0 or 1."""
        p_clip = np.clip(p, eps, 1 - eps)
        return np.array(logit(p_clip))

    @staticmethod
    def _tune(p_raw: np.ndarray, p_cal: np.ndarray, lam: float) -> np.ndarray:
        """Linear tune between raw and calibrated probabilities using lambda."""
        return (1.0 - lam) * p_raw + lam * p_cal
