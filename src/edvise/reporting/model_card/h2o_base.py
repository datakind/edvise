import typing as t
from abc import abstractmethod

from edvise.modeling import h2o_ml
from edvise.reporting.model_card.base import ModelCard
from edvise.reporting.utils.types import ModelCardConfig
import edvise.reporting.utils as reporting_utils
from edvise.shared.utils import as_percent


C = t.TypeVar("C", bound=ModelCardConfig)


class H2OModelCard(ModelCard[C]):
    """
    Base class for H2O-based model cards.

    This intermediate class handles all H2O-specific logic that is common
    across different project types (Custom, PDP, etc.). Subclasses should
    only implement project-specific customizations.
    """

    REQUIRED_PLOT_ARTIFACTS = [
        "model_comparison.png",
        "test_calibration_curve.png",
        "test_roc_curve.png",
        "test_confusion_matrix.png",
        "preds/test_hist.png",
        "h2o_feature_importances_by_shap_plot.png",
    ]

    def load_model(self):
        """
        Loads the H2O model from MLflow using the run ID from config.
        Also assigns the run ID and experiment ID.
        """
        model_cfg = self.cfg.model
        if not model_cfg:
            raise ValueError(f"Model configuration for '{self.model_name}' is missing.")
        if not all([model_cfg.run_id, model_cfg.experiment_id]):
            raise ValueError(
                f"Incomplete model config for '{self.model_name}': "
                f"run_id or experiment_id missing."
            )

        self.model = h2o_ml.utils.load_h2o_model(model_cfg.run_id)
        self.run_id = model_cfg.run_id
        self.experiment_id = model_cfg.experiment_id

    def extract_training_data(self):
        """
        Extracts the training data from the MLflow run using H2O utilities.
        """
        self.modeling_data = h2o_ml.evaluation.extract_training_data_from_model(
            self.experiment_id
        )
        self.training_data = self.modeling_data
        if self.cfg.split_col:
            if self.cfg.split_col not in self.modeling_data.columns:
                raise ValueError(
                    f"Configured split_col '{self.cfg.split_col}' is not present in modeling data columns: "
                    f"{list(self.modeling_data.columns)}"
                )
        self.context["training_dataset_size"] = self.modeling_data.shape[0]
        self.context["num_runs_in_experiment"] = (
            h2o_ml.evaluation.extract_number_of_runs_from_model_training(
                self.experiment_id
            )
        )

    def get_feature_metadata(self) -> dict[str, str]:
        """
        Collects feature count from the H2O model and feature selection config.

        Returns:
            A dictionary with feature metadata for the template.
        """
        feature_count = len(h2o_ml.inference.get_h2o_used_features(self.model))
        if not self.cfg.modeling or not self.cfg.modeling.feature_selection:
            raise ValueError(
                "Modeling configuration or feature selection config is missing."
            )

        fs_cfg = self.cfg.modeling.feature_selection

        return {
            "number_of_features": str(feature_count),
            "collinearity_threshold": str(fs_cfg.collinear_threshold),
            "low_variance_threshold": str(fs_cfg.low_variance_threshold),
            "incomplete_threshold": as_percent(fs_cfg.incomplete_threshold),
        }

    def get_model_plots(self) -> dict[str, str]:
        """
        Collects model plots from the MLflow run and downloads them locally.

        Subclasses can override this to customize plot artifact paths.

        Returns:
            A dictionary with plot names and their inline HTML representations.
        """
        plots = self._get_plot_config()
        return {
            key: reporting_utils.utils.download_artifact(
                run_id=self.run_id,
                description=description,
                artifact_path=path,
                local_folder=self.assets_folder,
                fixed_width=width,
                caption=caption,
            )
            or ""
            for key, (description, path, width, caption) in plots.items()
        }

    @abstractmethod
    def _get_plot_config(self) -> dict[str, tuple[str, str, str, str]]:
        """
        Returns plot configuration as a dictionary.

        Each key is the plot name, and the value is a tuple of:
        (description, artifact_path, width, caption)

        Subclasses must implement this to provide project-specific plot paths.
        """
        pass
