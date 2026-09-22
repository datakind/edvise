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
            "num_courses_instructional_modality_hybrid",
            "edvise",
            "num_courses_delivery_method_hybrid",
        ),
        (
            "num_courses_instructional_modality_online",
            "edvise",
            "num_courses_delivery_method_online",
        ),
        (
            "num_courses_instructional_modality_face_to_face_and_online",
            "edvise",
            "num_courses_delivery_method_face_to_face_and_online",
        ),
        (
            "num_courses_instructional_modality_online_internet_or_web",
            "edvise",
            "num_courses_delivery_method_online_internet_or_web",
        ),
        (
            "num_courses_instructional_modality_web_based",
            "edvise",
            "num_courses_delivery_method_web_based",
        ),
        (
            "num_courses_instructional_modality_fully_online",
            "edvise",
            "num_courses_delivery_method_fully_online",
        ),
        (
            "num_courses_instructional_modality_hybrid_asynchronous",
            "edvise",
            "num_courses_delivery_method_hybrid_asynchronous",
        ),
        (
            "num_courses_instructional_modality_hybrid_synchronous",
            "edvise",
            "num_courses_delivery_method_hybrid_synchronous",
        ),
        (
            "num_courses_instructional_modality_online_asynchronous",
            "edvise",
            "num_courses_delivery_method_online_asynchronous",
        ),
        (
            "num_courses_instructional_modality_online_mix",
            "edvise",
            "num_courses_delivery_method_online_mix",
        ),
        (
            "num_courses_instructional_modality_online_synchronous",
            "edvise",
            "num_courses_delivery_method_online_synchronous",
        ),
        (
            "cumfrac_num_courses_instructional_modality_online_that_is_hybrid",
            "edvise",
            "cumfrac_num_courses_delivery_method_online_that_is_hybrid",
        ),
        (
            "num_courses_instructional_modality_web_enhanced",
            "edvise",
            "num_courses_delivery_method_web_enhanced",
        ),
        (
            "num_courses_gateway_or_developmental_flag_no",
            "edvise",
            "num_courses_math_or_english_gateway_no",
        ),
        (
            "frac_courses_gen_ed_flag_no",
            "edvise",
            "frac_courses_core_course_no",
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
        (
            "frac_courses_gateway_or_developmental_flag_gateway_english",
            "edvise",
            "frac_courses_math_or_english_gateway_gateway_english",
        ),
        (
            "num_courses_gateway_or_developmental_flag_gateway_math",
            "edvise",
            "num_courses_math_or_english_gateway_gateway_math",
        ),
        (
            "num_courses_course_grade_s",
            "edvise",
            "num_courses_course_grade_s",
        ),
        (
            "cumfrac_num_courses_course_grade_ng",
            "edvise",
            "cumfrac_num_courses_course_grade_ng",
        ),
        (
            "frac_courses_course_grade_u",
            "edvise",
            "frac_courses_course_grade_u",
        ),
        (
            "num_courses_course_grade_s",
            "pdp",
            "num_courses_course_grade_s",
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


def test_unseen_dummy_value_descriptions_name_the_field_and_value():
    features_table = {
        r"^num_courses_(delivery_method)_(.*)$": {
            "name": "number of courses with {} '{}' this term",
            "short_desc": "Number of courses this term whose {} was '{}'",
            "long_desc": "The number of courses whose {} was reported as '{}'.",
            "capture_format": "words",
        }
    }

    name, short_desc, long_desc = _get_mapped_feature_name(
        "num_courses_instructional_modality_hyflex_synchronous",
        features_table,
        metadata=True,
        schema_type="edvise",
    )

    assert (
        name == "number of courses with delivery method 'hyflex synchronous' this term"
    )
    assert short_desc == (
        "Number of courses this term whose delivery method was 'hyflex synchronous'"
    )
    assert long_desc == (
        "The number of courses whose delivery method was reported as "
        "'hyflex synchronous'."
    )


def test_untemplated_descriptions_are_left_unchanged():
    features_table = {
        r"^num_courses_course_id_(.*)$": {
            "name": "number of times course '{}' taken this term",
            "short_desc": "If a student took this specific course in that term",
        }
    }

    name, short_desc, _ = _get_mapped_feature_name(
        "num_courses_course_id_engl_101",
        features_table,
        metadata=True,
    )

    assert name == "number of times course 'engl_101' taken this term"
    assert short_desc == "If a student took this specific course in that term"


def test_get_mapped_feature_name_humanizes_unseen_dummy_value():
    features_table = {
        r"^num_courses_(delivery_method)_(.*)$": {
            "name": "number of courses where {} is '{}'",
            "capture_format": "words",
        }
    }

    assert (
        _get_mapped_feature_name(
            "num_courses_instructional_modality_hyflex_synchronous",
            features_table,
            schema_type="edvise",
        )
        == "number of courses where delivery method is 'hyflex synchronous'"
    )


def test_exact_feature_description_wins_over_dummy_family_fallback():
    features_table = {
        "num_courses_delivery_method_o": {"name": "curated online description"},
        r"^num_courses_(delivery_method)_(.*)$": {
            "name": "generic {} {} description",
            "capture_format": "words",
        },
    }

    assert (
        _get_mapped_feature_name(
            "num_courses_instructional_modality_o",
            features_table,
            schema_type="edvise",
        )
        == "curated online description"
    )
