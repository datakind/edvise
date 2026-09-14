"""Unit tests for ES course categorical → PDP closed-set mapping."""

import numpy as np
import pandas as pd

from edvise.data_audit.es_categorical_map import (
    canonicalize_es_course_categoricals,
    gateway_flag_series_to_pdp,
    instructor_appointment_series_to_pdp,
    yes_no_series_to_pdp,
)
from edvise.data_audit.schemas._edvise_shared import _apply_course_schema_transforms
from edvise.feature_generation.column_names import (
    CourseFeatureSpec,
    ES_COURSE_INPUT_COLUMNS,
)
from edvise.feature_generation.course import add_features
from edvise.feature_generation.student_term import sum_dummy_cols_by_group


def test_instructor_appointment_synonyms() -> None:
    series = pd.Series(["FT", "Full Time", "adjunct", "part-time", "Tenured"])
    assert instructor_appointment_series_to_pdp(series).tolist() == [
        "FT",
        "FT",
        "PT",
        "PT",
        "FT",
    ]


def test_gateway_flag_synonyms() -> None:
    series = pd.Series(
        ["E", "Gateway English", "gateway_math", "developmental", "NA"]
    )
    assert gateway_flag_series_to_pdp(series).tolist() == ["E", "E", "M", "NA", "NA"]


def test_yes_no_synonyms() -> None:
    series = pd.Series(["Y", "Yes", "true", "N", "false", "0"])
    assert yes_no_series_to_pdp(series).tolist() == ["Y", "Y", "Y", "N", "N", "N"]


def test_unmapped_become_na() -> None:
    series = pd.Series(["visiting", "unknown", None, np.nan])
    assert instructor_appointment_series_to_pdp(series).isna().all()


def test_canonicalize_rewrites_all_es_course_categoricals() -> None:
    df = pd.DataFrame(
        {
            "instructional_modality": ["Web Based"],
            "instructor_appointment_status": ["Full Time"],
            "gateway_or_developmental_flag": ["Gateway English"],
            "gen_ed_flag": ["Yes"],
            "prerequisite_flag": ["true"],
            "intent_to_transfer_flag": ["No"],
        }
    )
    out = canonicalize_es_course_categoricals(df)
    assert out["instructional_modality"].iloc[0] == "O"
    assert out["instructor_appointment_status"].iloc[0] == "FT"
    assert out["gateway_or_developmental_flag"].iloc[0] == "E"
    assert out["gen_ed_flag"].iloc[0] == "Y"
    assert out["prerequisite_flag"].iloc[0] == "Y"
    assert out["intent_to_transfer_flag"].iloc[0] == "N"


def test_course_schema_transforms_canonicalize_all_es_categoricals() -> None:
    df = pd.DataFrame(
        {
            "instructor_appointment_status": ["adjunct"],
            "gateway_or_developmental_flag": ["developmental"],
            "gen_ed_flag": ["Yes"],
        }
    )
    out = _apply_course_schema_transforms(df)
    assert out["instructor_appointment_status"].iloc[0] == "PT"
    assert out["gateway_or_developmental_flag"].iloc[0] == "NA"
    assert out["gen_ed_flag"].iloc[0] == "Y"


def test_add_features_normalizes_instructor_and_gateway_before_dummies() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s1"],
            "term_id": ["t1", "t1", "t1"],
            "instructor_appointment_status": ["Full Time", "adjunct", "PT"],
            "gateway_or_developmental_flag": [
                "Gateway English",
                "gateway_math",
                "developmental",
            ],
            "course_prefix": ["MATH"] * 3,
            "course_number": ["101"] * 3,
            "grade": ["B"] * 3,
        }
    )
    featured = add_features(
        df,
        cols=ES_COURSE_INPUT_COLUMNS,
        spec=CourseFeatureSpec(
            course_id=False,
            course_subject_area=False,
            course_passed=False,
            course_completed=False,
            course_level=False,
            course_grade_numeric=False,
            course_grade=False,
        ),
        grade_semantics="es",
    )
    inst = sum_dummy_cols_by_group(
        featured,
        grp_cols=["student_id", "term_id"],
        agg_cols=["instructor_appointment_status"],
    )
    gate = sum_dummy_cols_by_group(
        featured,
        grp_cols=["student_id", "term_id"],
        agg_cols=["gateway_or_developmental_flag"],
    )
    assert set(c for c in inst.columns if c.startswith("num_courses_")) == {
        "num_courses_instructor_appointment_status_FT",
        "num_courses_instructor_appointment_status_PT",
    }
    assert set(c for c in gate.columns if c.startswith("num_courses_")) == {
        "num_courses_gateway_or_developmental_flag_E",
        "num_courses_gateway_or_developmental_flag_M",
        "num_courses_gateway_or_developmental_flag_NA",
    }
