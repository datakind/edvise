"""Unit tests for ES instructional_modality → PDP delivery_method mapping."""

import numpy as np
import pandas as pd

from edvise.data_audit.instructional_modality_map import (
    instructional_modality_series_to_pdp,
    instructional_modality_to_pdp_val,
)
from edvise.data_audit.schemas._edvise_shared import _apply_course_schema_transforms
from edvise.feature_generation.column_names import (
    CourseFeatureSpec,
    ES_COURSE_INPUT_COLUMNS,
)
from edvise.feature_generation.course import add_features
from edvise.feature_generation.student_term import sum_dummy_cols_by_group


def test_known_synonyms_map_to_pdp_codes() -> None:
    series = pd.Series(
        [
            "F",
            "face-to-face",
            "In Person",
            "O",
            "web-based",
            "Online Internet or Web",
            "H",
            "hybrid",
            "Face to Face and Online",
        ]
    )
    result = instructional_modality_series_to_pdp(series)
    assert result.tolist() == ["F", "F", "F", "O", "O", "O", "H", "H", "H"]


def test_unmapped_and_nulls_become_na() -> None:
    series = pd.Series(["Correspondence", "Unknown", None, np.nan, "n/a", ""])
    result = instructional_modality_series_to_pdp(series)
    assert result.isna().all()


def test_already_canonical_codes_are_idempotent() -> None:
    assert instructional_modality_to_pdp_val("F") == "F"
    assert instructional_modality_to_pdp_val("o") == "O"
    assert instructional_modality_to_pdp_val("h") == "H"


def test_course_schema_transforms_canonicalize_modality() -> None:
    df = pd.DataFrame(
        {
            "instructional_modality": [
                "Face to Face and Online",
                "Online Internet or Web",
                "Web Based",
                "Correspondence",
            ]
        }
    )
    out = _apply_course_schema_transforms(df)
    assert out["instructional_modality"].tolist()[:3] == ["H", "O", "O"]
    assert pd.isna(out["instructional_modality"].iloc[3])


def test_course_add_features_normalizes_before_dummies() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1", "s1", "s1", "s1"],
            "term_id": ["t1", "t1", "t1", "t1"],
            "instructional_modality": [
                "Face to Face and Online",
                "Online Internet or Web",
                "Web Based",
                "in-person",
            ],
            "course_prefix": ["MATH"] * 4,
            "course_number": ["101"] * 4,
            "grade": ["B"] * 4,
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
    dummies = sum_dummy_cols_by_group(
        featured,
        grp_cols=["student_id", "term_id"],
        agg_cols=["instructional_modality"],
    )
    dummy_cols = [c for c in dummies.columns if c.startswith("num_courses_")]
    assert set(dummy_cols) == {
        "num_courses_instructional_modality_F",
        "num_courses_instructional_modality_O",
        "num_courses_instructional_modality_H",
    }
    assert int(dummies["num_courses_instructional_modality_H"].iloc[0]) == 1
    assert int(dummies["num_courses_instructional_modality_O"].iloc[0]) == 2
    assert int(dummies["num_courses_instructional_modality_F"].iloc[0]) == 1
