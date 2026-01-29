import pytest
from unittest.mock import MagicMock

from edvise.reporting.model_card.h2o_pdp import H2OPDPModelCard


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.model = MagicMock(run_id="123", experiment_id="456")
    cfg.institution_id = "inst"
    cfg.institution_name = "TestInstitution"
    cfg.split_col = None
    return cfg


@pytest.fixture
def mock_client():
    return MagicMock()


def test_init_with_pdp_config(mock_config, mock_client):
    """Test initialization with PDPProjectConfig."""
    card = H2OPDPModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="my_model",
        mlflow_client=mock_client,
    )
    assert card.model_name == "my_model"
    assert isinstance(card.context, dict)


def test_get_plot_config(mock_config, mock_client):
    """Test that plot configuration is correctly defined."""
    card = H2OPDPModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="my_model",
        mlflow_client=mock_client,
    )

    plot_config = card._get_plot_config()

    # Verify expected plots are present
    expected_plots = [
        "model_comparison_plot",
        "test_calibration_curve",
        "test_roc_curve",
        "test_confusion_matrix",
        "test_histogram",
        "feature_importances_by_shap_plot",
    ]

    for plot_name in expected_plots:
        assert plot_name in plot_config, f"Missing plot: {plot_name}"
        assert isinstance(plot_config[plot_name], tuple)
        assert len(plot_config[plot_name]) == 4  # (description, path, width, caption)


def test_required_plot_artifacts_list():
    """Test that REQUIRED_PLOT_ARTIFACTS is correctly defined."""
    expected_artifacts = [
        "model_comparison.png",
        "test_calibration_curve.png",
        "test_roc_curve.png",
        "test_confusion_matrix.png",
        "preds/test_hist.png",
        "h2o_feature_importances_by_shap_plot.png",
    ]

    assert H2OPDPModelCard.REQUIRED_PLOT_ARTIFACTS == expected_artifacts
