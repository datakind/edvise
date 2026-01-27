import pytest
from unittest.mock import MagicMock

from edvise.reporting.model_card.h2o_custom import H2OCustomModelCard
from edvise.configs.custom import CustomProjectConfig
from edvise.configs.pdp import PDPProjectConfig


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.__class__ = CustomProjectConfig
    cfg.model = MagicMock(run_id="123", experiment_id="456")
    cfg.institution_id = "inst"
    cfg.institution_name = "TestInstitution"
    cfg.split_col = None
    return cfg


@pytest.fixture
def mock_client():
    return MagicMock()


def test_init_with_custom_config(mock_config, mock_client):
    """Test initialization with CustomProjectConfig."""
    card = H2OCustomModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="my_model",
        mlflow_client=mock_client,
    )
    assert card.model_name == "my_model"
    assert isinstance(card.context, dict)


def test_init_rejects_wrong_config_type(mock_client):
    """Test that H2OCustomModelCard rejects non-Custom configs."""
    pdp_config = MagicMock()
    pdp_config.__class__ = PDPProjectConfig
    pdp_config.institution_id = "inst"

    with pytest.raises(TypeError, match="Expected config to be of type CustomProjectConfig"):
        H2OCustomModelCard(
            config=pdp_config,
            catalog="catalog",
            model_name="my_model",
            mlflow_client=mock_client,
        )


def test_get_plot_config(mock_config, mock_client):
    """Test that plot configuration is correctly defined."""
    card = H2OCustomModelCard(
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

    assert H2OCustomModelCard.REQUIRED_PLOT_ARTIFACTS == expected_artifacts
