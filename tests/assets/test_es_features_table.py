import os

import pytest

from edvise.dataio.read import from_toml_file
from edvise.modeling.features_table_mapping import ES_ONLY_FEATURES_TABLE_COLUMNS
from edvise.modeling.inference import is_feature_defined_in_table


@pytest.fixture
def feature_table_data():
    project_root = os.getcwd()
    toml_path = os.path.join(
        project_root,
        "src",
        "edvise",
        "shared",
        "assets",
        "features_table.toml",
    )
    return from_toml_file(toml_path)


@pytest.mark.parametrize("feature_col", ES_ONLY_FEATURES_TABLE_COLUMNS)
def test_es_only_columns_defined_in_features_table(feature_table_data, feature_col):
    assert is_feature_defined_in_table(feature_col, feature_table_data)
