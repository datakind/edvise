import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from edvise.reporting.model_card.h2o_custom import H2OCustomModelCard
from edvise.configs.custom import CustomProjectConfig


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    # Set the class to pass isinstance check
    cfg.__class__ = CustomProjectConfig
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
    card = H2OCustomModelCard(
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
    card = H2OCustomModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.load_model()
    mock_load_model.assert_called_once_with("123")
    assert card.run_id == "123"
    assert card.experiment_id == "456"


def test_find_model_version_found(mock_config, mock_client):
    card = H2OCustomModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.run_id = "123"
    mock_version = MagicMock(run_id="123", version="5")
    mock_client.search_model_versions.return_value = [mock_version]
    card.find_model_version()
    assert card.context["version_number"] == "5"


def test_find_model_version_not_found(mock_config, mock_client):
    mock_client.search_model_versions.return_value = []
    card = H2OCustomModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.run_id = "999"
    card.find_model_version()
    assert card.context["version_number"] is None


@patch("edvise.reporting.model_card.h2o_base.h2o_ml.inference.get_h2o_used_features")
def test_get_feature_metadata_success(mock_get_features, mock_config, mock_client):
    mock_get_features.return_value = ["a", "b", "c"]
    card = H2OCustomModelCard(
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


@patch("edvise.reporting.model_card.base.utils.download_static_asset")
def test_get_basic_context(mock_download, mock_config, mock_client):
    mock_download.return_value = "<img>Logo</img>"
    card = H2OCustomModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    result = card.get_basic_context()
    assert result["institution_name"] == "TestInstitution"
    assert "logo" in result


def test_build_calls_all_steps(mock_config, mock_client):
    card = H2OCustomModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    for method in [
        "load_model",
        "find_model_version",
        "extract_training_data",
        "_register_sections",
        "collect_metadata",
        "render",
    ]:
        setattr(card, method, MagicMock())

    card.build()

    for method in [
        card.load_model,
        card.find_model_version,
        card.extract_training_data,
        card._register_sections,
        card.collect_metadata,
        card.render,
    ]:
        method.assert_called_once()


@patch(
    "edvise.reporting.model_card.h2o_base.h2o_ml.evaluation.extract_number_of_runs_from_model_training"
)
@patch(
    "edvise.reporting.model_card.h2o_base.h2o_ml.evaluation.extract_training_data_from_model"
)
@patch("edvise.reporting.model_card.h2o_base.h2o_ml.utils.load_h2o_model")
def test_extract_training_data_with_split_call_load_model(
    mock_load_model, mock_extract_data, mock_extract_runs, mock_config, mock_client
):
    mock_config.split_col = "split"
    df = pd.DataFrame(
        {"feature": [1, 2, 3, 4], "split": ["train", "test", "train", "val"]}
    )
    mock_extract_data.return_value = df
    mock_extract_runs.return_value = 2  # simulate 2 runs in experiment

    card = H2OCustomModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.load_model()
    card.extract_training_data()

    assert card.context["training_dataset_size"] == 4
    assert card.context["num_runs_in_experiment"] == 2


@patch("builtins.open", new_callable=MagicMock)
def test_render_template_and_output(mock_open, mock_config, mock_client):
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    mock_file.read.return_value = "Model: {institution_name}"

    card = H2OCustomModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.template_path = "template.md"
    card.output_path = "output.md"
    card.context = {"institution_name": "TestInstitution"}
    card.render()

    mock_open.assert_any_call("template.md", "r")
    mock_open.assert_any_call("output.md", "w")
    mock_file.write.assert_called_once_with("Model: TestInstitution")
