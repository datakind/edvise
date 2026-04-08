import typing as t
from mlflow.tracking import MlflowClient

from edvise.configs.legacy import LegacyProjectConfig
from edvise.reporting.model_card.h2o_base import H2OModelCard
from edvise.reporting.sections.legacy import (
    register_sections as register_legacy_sections,
)


class H2OLegacyModelCard(H2OModelCard[LegacyProjectConfig]):
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
        config: LegacyProjectConfig,
        catalog: str,
        model_name: str,
        assets_path: t.Optional[str] = None,
        mlflow_client: t.Optional[MlflowClient] = None,
    ):
        """
        Initializes a legacy (non-PDP) institution model card with a LegacyProjectConfig.
        """
        super().__init__(config, catalog, model_name, assets_path, mlflow_client)

    def _get_plot_config(self) -> dict[str, tuple[str, str, str, str]]:
        """
        Returns Legacy project-specific plot configuration.
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
        Register legacy (non-PDP) institution-specific sections.
        """
        # Clearing registry for overrides
        self.section_registry.clear()

        # Register legacy-specific sections
        register_legacy_sections(self, self.section_registry)
