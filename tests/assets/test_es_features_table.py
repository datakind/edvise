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


# ES dummy names from the training failure, resolved via PDP features-table keys.
ES_DUMMY_FEATURES_MAPPED_TO_PDP = (
    "num_courses_course_grade_s",
    "num_courses_course_grade_u",
    "frac_courses_course_grade_nr",
    "frac_courses_course_grade_s",
    "frac_courses_course_grade_u",
    "cumfrac_num_courses_course_grade_ng",
    "cumfrac_num_courses_course_grade_nr",
    "cumfrac_num_courses_course_grade_s",
    "cumfrac_num_courses_course_grade_u",
    "frac_courses_gateway_or_developmental_flag_gateway_english",
)


@pytest.mark.parametrize("feature_col", ES_DUMMY_FEATURES_MAPPED_TO_PDP)
def test_es_dummy_features_resolve_to_pdp_features_table(
    feature_table_data, feature_col
):
    assert is_feature_defined_in_table(
        feature_col, feature_table_data, schema_type="edvise"
    )
