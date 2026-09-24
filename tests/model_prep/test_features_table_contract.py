"""Post-cleanup modeling columns must be defined in features_table.toml.

Inputs are schema-validated (PDP/ES Faker synth) then standardized — the same
boundary feature generation consumes. Feature gen runs once per schema; cleanup
is parametrized over checkpoint families that change keep/drop. Does not
enumerate leftover names — any undocumented keep is a failure.
"""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import faker
import pandas as pd
import pytest

from edvise.data_audit.raw_course_grade_map import (
    apply_raw_course_grade_map,
    resolve_es_grade_map,
)
from edvise.data_audit.schemas import (
    RawEdviseCourseDataSchema,
    RawEdviseStudentDataSchema,
    RawPDPCohortDataSchema,
    RawPDPCourseDataSchema,
)
from edvise.data_audit.standardizer import (
    ESCohortStandardizer,
    ESCourseStandardizer,
    PDPCohortStandardizer,
    PDPCourseStandardizer,
)
from edvise.dataio.read import read_features_table
from edvise.feature_generation.assemble_student_terms import (
    make_student_term_dataset,
    student_level_merge_keys,
)
from edvise.feature_generation.column_names import (
    ES_COHORT_INPUT_COLUMNS,
    ES_COURSE_INPUT_COLUMNS,
    PDP_COHORT_INPUT_COLUMNS,
    PDP_COURSE_INPUT_COLUMNS,
)
from edvise.feature_generation.es_feature_specs import build_edvise_feature_specs
from edvise.modeling.inference import is_feature_defined_in_table
from edvise.shared.utils import feature_cleanup_for_schema
from edvise.synth_generation.es import raw_course as es_raw_course
from edvise.synth_generation.es import raw_student as es_raw_student
from edvise.synth_generation.pdp import raw_cohort, raw_course
from edvise.utils.data_cleaning import convert_to_snake_case

_FEATURES_TABLE = read_features_table("shared/assets/features_table.toml")

_PDP_FAKER = faker.Faker()
_PDP_FAKER.seed_instance(20260924)
_PDP_FAKER.add_provider(raw_cohort.Provider)
_PDP_FAKER.add_provider(raw_course.Provider)

_ES_FAKER = faker.Faker()
_ES_FAKER.seed_instance(20260924)
_ES_FAKER.add_provider(es_raw_student.Provider)
_ES_FAKER.add_provider(es_raw_course.Provider)

# Mirrors ProjectConfig.non_feature_cols (id/target/split/weight + student_group_cols).
_NON_FEATURE_COLS_BY_SCHEMA: dict[str, frozenset[str]] = {
    "pdp": frozenset(
        {
            "student_id",
            "learner_id",
            "target",
            "split",
            "sample_weight",
            "student_age",
            "race",
            "ethnicity",
            "gender",
            "first_gen",
        }
    ),
    "edvise": frozenset(
        {
            "student_id",
            "learner_id",
            "target",
            "split",
            "sample_weight",
            "learner_age",
            "race",
            "ethnicity",
            "gender",
            "first_generation_status",
        }
    ),
}

# Checkpoint families that change extra_snapshot_drop_cols keep/drop.
_CHECKPOINT_FAMILIES: tuple[tuple[str, dict[str, object]], ...] = (
    ("first_term", {"type_": "first_within_cohort"}),
    (
        "second_core_term",
        {"type_": "nth", "n": 1, "exclude_non_core_terms": True},
    ),
    ("later_or_credits", {"type_": "first_at_num_credits_earned"}),
)


def _checkpoint_cfg(**ckpt_fields: object) -> SimpleNamespace:
    return SimpleNamespace(
        preprocessing=SimpleNamespace(checkpoint=SimpleNamespace(**ckpt_fields))
    )


def _snake_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {convert_to_snake_case(key): val for key, val in rec.items()} for rec in records
    ]


def _pdp_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw PDP synth → schema validate → standardize (feature-gen silver)."""
    institution_id = "999900"
    # Title-case keys so course synth can copy overlapping fields from cohort.
    cohort_records = [
        _PDP_FAKER.raw_cohort_record(
            normalize_col_names=False, institution_id=institution_id
        )
        for _ in range(3)
    ]
    course_records: list[dict[str, object]] = []
    for rec in cohort_records:
        for _ in range(2):
            course_records.append(
                _PDP_FAKER.raw_course_record(
                    cohort_record=rec, normalize_col_names=False
                )
            )
    cohort = RawPDPCohortDataSchema.validate(
        pd.DataFrame(_snake_records(cohort_records)), lazy=True
    )
    course = RawPDPCourseDataSchema.validate(
        pd.DataFrame(_snake_records(course_records)), lazy=True
    )
    return (
        PDPCohortStandardizer().standardize(cohort),
        PDPCourseStandardizer().standardize(course),
    )


def _es_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw ES synth → grade map → schema validate → standardize."""
    student_records = [
        _ES_FAKER.raw_student_record(include_optionals=True) for _ in range(3)
    ]
    for i, rec in enumerate(student_records):
        rec["learner_id"] = f"{10000 + i}"
    course_records: list[dict[str, object]] = []
    for rec in student_records:
        for _ in range(2):
            course_records.append(
                _ES_FAKER.raw_course_record(student_record=rec, include_optionals=True)
            )
    cohort = pd.DataFrame(student_records)
    course = apply_raw_course_grade_map(
        pd.DataFrame(course_records), resolve_es_grade_map(None)
    )
    cohort = RawEdviseStudentDataSchema.validate(cohort, lazy=True)
    course = RawEdviseCourseDataSchema.validate(course, lazy=True)
    return (
        ESCohortStandardizer().standardize(cohort),
        ESCourseStandardizer().standardize(course),
    )


def _generate_pdp() -> pd.DataFrame:
    df_cohort, df_course = _pdp_inputs()
    merge_on = student_level_merge_keys(
        df_cohort, df_course, cohort_cols=PDP_COHORT_INPUT_COLUMNS
    )
    return make_student_term_dataset(
        df_cohort=df_cohort,
        df_course=df_course,
        merge_on=merge_on,
        cohort_input_columns=PDP_COHORT_INPUT_COLUMNS,
        course_input_columns=PDP_COURSE_INPUT_COLUMNS,
    )


def _generate_es() -> pd.DataFrame:
    df_cohort, df_course = _es_inputs()
    spec_bundle = build_edvise_feature_specs(
        df_cohort,
        df_course,
        cohort_cols=ES_COHORT_INPUT_COLUMNS,
        course_cols=ES_COURSE_INPUT_COLUMNS,
    )
    merge_on = student_level_merge_keys(
        df_cohort, df_course, cohort_cols=ES_COHORT_INPUT_COLUMNS
    )
    return make_student_term_dataset(
        df_cohort=df_cohort,
        df_course=df_course,
        merge_on=merge_on,
        cohort_input_columns=ES_COHORT_INPUT_COLUMNS,
        course_input_columns=ES_COURSE_INPUT_COLUMNS,
        student_feature_spec=spec_bundle.student,
        course_feature_spec=spec_bundle.course,
        term_feature_spec=spec_bundle.term,
        section_feature_spec=spec_bundle.section,
        student_term_aggregate_spec=spec_bundle.student_term_aggregate,
        student_term_add_feature_spec=spec_bundle.student_term_add,
        cumulative_feature_spec=spec_bundle.cumulative,
        grade_semantics="es",
    )


@lru_cache(maxsize=2)
def _student_terms(schema_type: str) -> pd.DataFrame:
    if schema_type == "pdp":
        return _generate_pdp()
    if schema_type == "edvise":
        return _generate_es()
    raise ValueError(f"unsupported schema_type {schema_type!r}")


@pytest.mark.parametrize("schema_type", ["pdp", "edvise"])
@pytest.mark.parametrize(
    ("_family", "ckpt_fields"),
    _CHECKPOINT_FAMILIES,
    ids=[name for name, _ in _CHECKPOINT_FAMILIES],
)
def test_post_cleanup_features_are_defined_in_features_table(
    schema_type: str,
    _family: str,
    ckpt_fields: dict[str, object],
) -> None:
    df = _student_terms(schema_type)
    assert not df.empty

    cleaner = feature_cleanup_for_schema(schema_type)
    cleaned = cleaner.clean_up_labeled_dataset_cols_and_vals(
        df.copy(), cfg=_checkpoint_cfg(**ckpt_fields)
    )
    feature_cols = [
        c for c in cleaned.columns if c not in _NON_FEATURE_COLS_BY_SCHEMA[schema_type]
    ]
    undefined = [
        col
        for col in feature_cols
        if not is_feature_defined_in_table(
            col, _FEATURES_TABLE, schema_type=schema_type
        )
    ]
    assert undefined == [], (
        f"{schema_type} leftover columns not in features_table: {undefined}"
    )
