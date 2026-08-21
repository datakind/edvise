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


# Status-grade dummies from ALLOWED_LETTER_GRADES that were missing from
# features_table, plus ES gateway dummy values that failed training validation.
_STATUS_GRADE_SUFFIXES = ("s", "u", "nr", "ng", "pass", "sat", "unsat", "wd", "ip")
_STATUS_GRADE_PREFIXES = (
    "num_courses_course_grade_",
    "frac_courses_course_grade_",
    "cumfrac_num_courses_course_grade_",
)
STATUS_GRADE_AND_GATEWAY_FEATURES = tuple(
    f"{prefix}{suffix}"
    for suffix in _STATUS_GRADE_SUFFIXES
    for prefix in _STATUS_GRADE_PREFIXES
) + (
    "frac_courses_gateway_or_developmental_flag_gateway_english",
    "num_courses_gateway_or_developmental_flag_gateway_math",
)


@pytest.mark.parametrize("feature_col", STATUS_GRADE_AND_GATEWAY_FEATURES)
def test_status_grade_and_es_gateway_features_defined(feature_table_data, feature_col):
    assert is_feature_defined_in_table(
        feature_col, feature_table_data, schema_type="edvise"
    )
