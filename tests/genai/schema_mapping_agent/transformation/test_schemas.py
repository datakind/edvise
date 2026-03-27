"""Unit tests for edvise.genai.schema_mapping_agent.transformation.schemas."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from edvise.genai.schema_mapping_agent.manifest.schemas import (
    EntityType,
    ReviewStatus,
)
from edvise.genai.schema_mapping_agent.transformation.schemas import (
    FieldTransformationPlan,
    MapValuesStep,
    NewUtilityNeededStep,
    TransformationMap,
    TransformationStep,
)


def test_transformation_step_discriminated_union_cast_string():
    adapter = TypeAdapter(TransformationStep)
    step = adapter.validate_python(
        {"function_name": "cast_string", "column": "raw_id"}
    )
    assert step.function_name == "cast_string"
    assert step.column == "raw_id"


def test_transformation_step_map_values_default_and_mapping():
    adapter = TypeAdapter(TransformationStep)
    step = adapter.validate_python(
        {
            "function_name": "map_values",
            "column": "status",
            "mapping": {"1": "enrolled", "0": "left"},
        }
    )
    assert isinstance(step, MapValuesStep)
    assert step.default == "passthrough"


def test_transformation_step_coerce_datetime_optional_fmt():
    adapter = TypeAdapter(TransformationStep)
    step = adapter.validate_python(
        {"function_name": "coerce_datetime", "column": "d", "fmt": "%Y-%m-%d"}
    )
    assert step.fmt == "%Y-%m-%d"


def test_transformation_step_new_utility_needed_gap_property():
    adapter = TypeAdapter(TransformationStep)
    step = adapter.validate_python(
        {
            "function_name": "NEW_UTILITY_NEEDED",
            "description": "Need fuzzy date parser",
        }
    )
    assert isinstance(step, NewUtilityNeededStep)
    assert step.is_gap is True


def test_transformation_step_invalid_discriminator():
    adapter = TypeAdapter(TransformationStep)
    with pytest.raises(ValidationError):
        adapter.validate_python({"function_name": "not_a_real_step", "column": "x"})


def test_field_transformation_plan_steps_typed():
    plan = FieldTransformationPlan(
        target_field="grade",
        steps=[
            MapValuesStep(
                function_name="map_values",
                column="_",
                mapping={"A": "A"},
            )
        ],
        review_status=ReviewStatus.approved,
    )
    assert plan.steps[0].function_name == "map_values"


def test_transformation_map_unique_target_fields():
    with pytest.raises(ValidationError, match="only once"):
        TransformationMap(
            institution_id="x",
            entity_type=EntityType.course,
            target_schema="RawEdviseCourseDataSchema",
            plans=[
                FieldTransformationPlan(target_field="grade", steps=[]),
                FieldTransformationPlan(target_field="grade", steps=[]),
            ],
        )


def test_transformation_map_minimal_valid():
    tm = TransformationMap(
        institution_id="x",
        entity_type=EntityType.cohort,
        target_schema="RawEdviseStudentDataSchema",
        plans=[FieldTransformationPlan(target_field="learner_id", steps=[])],
    )
    assert tm.schema_version == "0.1.0"
    assert len(tm.plans) == 1
