import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from edvise.reporting.model_card.h2o_base import H2OModelCard
from edvise.configs.custom import CustomProjectConfig


# Minimal concrete implementation for testing
class TestableH2OModelCard(H2OModelCard[CustomProjectConfig]):
    """Concrete implementation for testing H2O base functionality."""

    def _get_plot_config(self) -> dict[str, tuple[str, str, str, str]]:
        return {
            "test_plot": ("Test Plot", "test.png", "100mm", "Test Caption")
        }

    def _register_sections(self):
        pass  # No sections needed for base tests


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.model = MagicMock(mlflow_model_uri="uri", run_id="123", experiment_id="456")
    cfg.institution_id = "inst"
    cfg.institution_name = "TestInstitution"
    cfg.modeling.feature_selection.collinear_threshold = 0.9
    cfg.modeling.feature_selection.low_variance_threshold = 0.01
    cfg.modeling.feature_selection.incomplete_threshold = 0.05
    cfg.split_col = None
    return cfg


@pytest.fixture
def mock_client():
    return MagicMock()


def test_init_defaults(mock_config, mock_client):
    card = TestableH2OModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    assert card.model_name == "inst_my_model"
    assert card.uc_model_name == "catalog.inst_gold.inst_my_model"
    assert card.assets_folder == "card_assets"
    assert isinstance(card.context, dict)


@patch("edvise.reporting.model_card.h2o_base.h2o_ml.utils.load_h2o_model")
def test_load_model_success(mock_load_model, mock_config, mock_client):
    card = TestableH2OModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.load_model()
    mock_load_model.assert_called_once_with("123")
    assert card.run_id == "123"
    assert card.experiment_id == "456"


def test_load_model_missing_config(mock_config, mock_client):
    mock_config.model = None
    card = TestableH2OModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    with pytest.raises(ValueError, match="Model configuration.*missing"):
        card.load_model()


def test_load_model_incomplete_config(mock_config, mock_client):
    mock_config.model.run_id = None
    card = TestableH2OModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    with pytest.raises(ValueError, match="Incomplete model config"):
        card.load_model()


@patch("edvise.reporting.model_card.h2o_base.h2o_ml.inference.get_h2o_used_features")
def test_get_feature_metadata_success(mock_get_features, mock_config, mock_client):
    mock_get_features.return_value = ["a", "b", "c"]
    card = TestableH2OModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.model = MagicMock()
    metadata = card.get_feature_metadata()
    assert metadata["number_of_features"] == "3"
    assert metadata["collinearity_threshold"] == "0.9"
    assert metadata["low_variance_threshold"] == "0.01"
    assert metadata["incomplete_threshold"] == "5"


@patch(
    "edvise.reporting.model_card.h2o_base.h2o_ml.evaluation.extract_number_of_runs_from_model_training"
)
@patch(
    "edvise.reporting.model_card.h2o_base.h2o_ml.evaluation.extract_training_data_from_model"
)
def test_extract_training_data(
    mock_extract_data, mock_extract_runs, mock_config, mock_client
):
    mock_config.split_col = "split"
    df = pd.DataFrame(
        {"feature": [1, 2, 3, 4], "split": ["train", "test", "train", "val"]}
    )
    mock_extract_data.return_value = df
    mock_extract_runs.return_value = 2

    card = TestableH2OModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.experiment_id = "456"
    card.extract_training_data()

    assert card.context["training_dataset_size"] == 4
    assert card.context["num_runs_in_experiment"] == 2


@patch(
    "edvise.reporting.model_card.h2o_base.h2o_ml.evaluation.extract_training_data_from_model"
)
def test_extract_training_data_invalid_split_col(
    mock_extract_data, mock_config, mock_client
):
    mock_config.split_col = "invalid_col"
    df = pd.DataFrame({"feature": [1, 2, 3, 4]})
    mock_extract_data.return_value = df

    card = TestableH2OModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.experiment_id = "456"

    with pytest.raises(ValueError, match="split_col.*not present"):
        card.extract_training_data()


@patch("edvise.reporting.model_card.h2o_base.reporting_utils.utils.download_artifact")
def test_get_model_plots(mock_download, mock_config, mock_client):
    mock_download.return_value = "<img>plot</img>"
    card = TestableH2OModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.run_id = "123"

    plots = card.get_model_plots()

    assert "test_plot" in plots
    assert plots["test_plot"] == "<img>plot</img>"
    mock_download.assert_called_once()
