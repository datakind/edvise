import pytest

from edvise.modeling.features_table_mapping import (
    build_es_to_pdp_feature_token_map,
    map_feature_col_for_features_table,
)
from edvise.modeling.inference import (
    _get_mapped_feature_name,
    is_feature_defined_in_table,
)


def test_build_es_to_pdp_feature_token_map_includes_course_and_cohort_tokens():
    token_map = build_es_to_pdp_feature_token_map()
    assert token_map["instructional_modality"] == "delivery_method"
    assert token_map["gen_ed_flag"] == "core_course"
    assert token_map["gateway_or_developmental_flag"] == "math_or_english_gateway"
    assert (
        token_map["instructor_appointment_status"]
        == "course_instructor_employment_status"
    )
    assert token_map["entry_term"] == "cohort_term"
    assert token_map["entry_year"] == "cohort"


@pytest.mark.parametrize(
    ("feature_col", "schema_type", "exp"),
    [
        (
            "num_courses_instructional_modality_f",
            "edvise",
            "num_courses_delivery_method_f",
        ),
        (
            "frac_courses_gen_ed_flag_y",
            "edvise",
            "frac_courses_core_course_y",
        ),
        ("academic_term", "pdp", "academic_term"),
        ("num_courses_delivery_method_f", "edvise", "num_courses_delivery_method_f"),
        (
            "pell_recipient_year_1",
            "edvise",
            "student_is_pell_recipient_first_year",
        ),
    ],
)
def test_map_feature_col_for_features_table(feature_col, schema_type, exp):
    assert map_feature_col_for_features_table(feature_col, schema_type) == exp


def test_is_feature_defined_in_table_with_es_pell_snake_case_alias() -> None:
    features_table = {
        "student_is_pell_recipient_first_year": {
            "name": "student is a Pell grant recipient in year 1"
        }
    }
    assert is_feature_defined_in_table(
        "pell_recipient_year_1",
        features_table,
        schema_type="edvise",
    )


def test_is_feature_defined_in_table_with_es_mapping():
    features_table = {
        "num_courses_delivery_method_f": {
            "name": "number of face to face courses taken this term"
        }
    }
    assert is_feature_defined_in_table(
        "num_courses_instructional_modality_f",
        features_table,
        schema_type="edvise",
    )
    assert not is_feature_defined_in_table(
        "num_courses_instructional_modality_f",
        features_table,
        schema_type="pdp",
    )


def test_get_mapped_feature_name_with_es_mapping():
    features_table = {
        "cohort_term": {
            "name": "student's cohort (enrollment) term",
            "short_desc": "short",
            "long_desc": "long",
        }
    }
    assert (
        _get_mapped_feature_name(
            "entry_term",
            features_table,
            metadata=False,
            schema_type="edvise",
        )
        == "student's cohort (enrollment) term"
    )
