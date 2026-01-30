import typing as t
from mlflow.tracking import MlflowClient

from edvise.configs.custom import CustomProjectConfig
from edvise.reporting.model_card.h2o_base import H2OModelCard
from edvise.reporting.sections.custom import (
    register_sections as register_custom_sections,
)


class H2OCustomModelCard(H2OModelCard[CustomProjectConfig]):
    REQUIRED_PLOT_ARTIFACTS = [
        "model_comparison.png",
        "test_calibration_curve.png",
        "test_roc_curve.png",
        "test_confusion_matrix.png",
        "preds/test_hist.png",
        "h2o_feature_importances_by_shap_plot.png",
    ]

    def __init__(
        self,
        config: CustomProjectConfig,
        catalog: str,
        model_name: str,
        assets_path: t.Optional[str] = None,
        mlflow_client: t.Optional[MlflowClient] = None,
    ):
        """
        Initializes custom model card with a custom project config.
        """
        super().__init__(config, catalog, model_name, assets_path, mlflow_client)

    def _get_plot_config(self) -> dict[str, tuple[str, str, str, str]]:
        """
        Returns Custom project-specific plot configuration.
        """
        return {
            "model_comparison_plot": (
                "Model Comparison",
                "model_comparison.png",
                "125mm",
                "Model Comparison by Architecture",
            ),
            "test_calibration_curve": (
                "Test Calibration Curve",
                "test_calibration_curve.png",
                "125mm",
                "Test Calibration Curve",
            ),
            "test_roc_curve": (
                "Test ROC Curve",
                "test_roc_curve.png",
                "125mm",
                "Test ROC Curve",
            ),
            "test_confusion_matrix": (
                "Test Confusion Matrix",
                "test_confusion_matrix.png",
                "175mm",
                "Test Confusion Matrix",
            ),
            "test_histogram": (
                "Test Histogram",
                "preds/test_hist.png",
                "125mm",
                "Test Support Score Histogram",
            ),
            "feature_importances_by_shap_plot": (
                "Feature Importances",
                "h2o_feature_importances_by_shap_plot.png",
                "150mm",
                "Feature Importances by SHAP on Test Data",
            ),
        }

    def _register_sections(self):
        """
        Register cusom-specific sections.
        """
        # Clearing registry for overrides
        self.section_registry.clear()

        # Register custom-specific sections
        register_custom_sections(self, self.section_registry)
