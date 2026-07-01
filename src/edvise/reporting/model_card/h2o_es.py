import typing as t
from mlflow.tracking import MlflowClient

from edvise.configs.es import ESProjectConfig
from edvise.reporting.model_card.h2o_base import H2OModelCard
from edvise.reporting.sections.es import register_sections as register_es_sections


class H2OESModelCard(H2OModelCard[ESProjectConfig]):
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
        config: ESProjectConfig,
        catalog: str,
        model_name: str,
        assets_path: t.Optional[str] = None,
        mlflow_client: t.Optional[MlflowClient] = None,
    ):
        """
        Initializes Edvise model card with an ES project config.
        """
        super().__init__(config, catalog, model_name, assets_path, mlflow_client)

    def _get_plot_config(self) -> dict[str, tuple[str, str, str, str]]:
        """
        Returns Edvise project-specific plot configuration.
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
        Register Edvise-specific sections.
        """
        self.section_registry.clear()
        register_es_sections(self, self.section_registry)
