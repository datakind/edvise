import os

import pytest

from edvise.dataio.read import from_toml_file
from edvise.modeling.features_table_mapping import (
    ES_MAPPED_FEATURES_TABLE_COLUMNS,
    ES_ONLY_FEATURES_TABLE_COLUMNS,
)
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


@pytest.mark.parametrize("feature_col", ES_MAPPED_FEATURES_TABLE_COLUMNS)
def test_es_mapped_columns_defined_in_features_table(feature_table_data, feature_col):
    assert is_feature_defined_in_table(feature_col, feature_table_data)


@pytest.mark.parametrize(
    ("es_col", "schema_type"),
    [
        ("learner_age", "edvise"),
        ("first_generation_status", "edvise"),
    ],
)
def test_es_renamed_columns_resolve_via_mapping(
    feature_table_data, es_col, schema_type
):
    assert is_feature_defined_in_table(
        es_col, feature_table_data, schema_type=schema_type
    )
