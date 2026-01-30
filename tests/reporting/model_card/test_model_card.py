import pytest
from unittest.mock import patch, MagicMock

from edvise.reporting.model_card.base import ModelCard


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


def test_find_model_version_found(mock_config, mock_client):
    """Test finding model version when it exists."""

    # Create a minimal concrete implementation for testing
    class ConcreteModelCard(ModelCard):
        def load_model(self):
            pass

        def extract_training_data(self):
            pass

        def get_feature_metadata(self):
            return {}

        def get_model_plots(self):
            return {}

        def _register_sections(self):
            pass

    card = ConcreteModelCard(
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
    """Test when model version is not found."""

    class ConcreteModelCard(ModelCard):
        def load_model(self):
            pass

        def extract_training_data(self):
            pass

        def get_feature_metadata(self):
            return {}

        def get_model_plots(self):
            return {}

        def _register_sections(self):
            pass

    mock_client.search_model_versions.return_value = []
    card = ConcreteModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    card.run_id = "999"
    card.find_model_version()
    assert card.context["version_number"] is None


@patch("edvise.reporting.model_card.base.utils.download_static_asset")
def test_get_basic_context(mock_download, mock_config, mock_client):
    """Test basic context retrieval."""

    class ConcreteModelCard(ModelCard):
        def load_model(self):
            pass

        def extract_training_data(self):
            pass

        def get_feature_metadata(self):
            return {}

        def get_model_plots(self):
            return {}

        def _register_sections(self):
            pass

    mock_download.return_value = "<img>Logo</img>"
    card = ConcreteModelCard(
        config=mock_config,
        catalog="catalog",
        model_name="inst_my_model",
        mlflow_client=mock_client,
    )
    result = card.get_basic_context()
    assert result["institution_name"] == "TestInstitution"
    assert "logo" in result


def test_build_calls_all_steps(mock_config, mock_client):
    """Test that build() calls all required steps in sequence."""

    class ConcreteModelCard(ModelCard):
        def load_model(self):
            pass

        def extract_training_data(self):
            pass

        def get_feature_metadata(self):
            return {}

        def get_model_plots(self):
            return {}

        def _register_sections(self):
            pass

    card = ConcreteModelCard(
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


@patch("builtins.open", new_callable=MagicMock)
def test_render_template_and_output(mock_open, mock_config, mock_client):
    """Test template rendering and output writing."""

    class ConcreteModelCard(ModelCard):
        def load_model(self):
            pass

        def extract_training_data(self):
            pass

        def get_feature_metadata(self):
            return {}

        def get_model_plots(self):
            return {}

        def _register_sections(self):
            pass

    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    mock_file.read.return_value = "Model: {institution_name}"

    card = ConcreteModelCard(
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
