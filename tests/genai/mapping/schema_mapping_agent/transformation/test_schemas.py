"""Unit tests for edvise.genai.mapping.schema_mapping_agent.transformation.schemas."""

from __future__ import annotations

import importlib.metadata

import pytest
from pydantic import TypeAdapter, ValidationError

from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    EntityType,
    ReviewStatus,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
    AcademicYearAndCanonicalSeasonToConferralDateStep,
    FieldTransformationPlan,
    FlaggedStep,
    MapValuesStep,
    TermComponentsToDatetimeStep,
    TransformationHITLItem,
    TransformationHITLOption,
    TransformationMap,
    TransformationStep,
)


def test_transformation_step_discriminated_union_cast_string():
    adapter = TypeAdapter(TransformationStep)
    step = adapter.validate_python({"function_name": "cast_string", "column": "raw_id"})
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


def test_transformation_step_term_components_to_datetime():
    adapter = TypeAdapter(TransformationStep)
    step = adapter.validate_python(
        {
            "function_name": "term_components_to_datetime",
            "column": "_edvise_term_academic_year",
            "extra_columns": {"season_series": "_edvise_term_season"},
        }
    )
    assert isinstance(step, TermComponentsToDatetimeStep)
    assert step.extra_columns["season_series"] == "_edvise_term_season"


def test_transformation_step_academic_year_and_canonical_season_to_conferral_date():
    adapter = TypeAdapter(TransformationStep)
    step = adapter.validate_python(
        {
            "function_name": "academic_year_and_canonical_season_to_conferral_date",
            "column": "_year_str",
            "extra_columns": {"season_series": "_season_canon"},
        }
    )
    assert isinstance(step, AcademicYearAndCanonicalSeasonToConferralDateStep)
    assert step.extra_columns["season_series"] == "_season_canon"


def test_transformation_step_compact_term_code_to_conferral_date():
    adapter = TypeAdapter(TransformationStep)
    step = adapter.validate_python(
        {
            "function_name": "compact_term_code_to_conferral_date",
            "column": "degree_term",
        }
    )
    assert step.function_name == "compact_term_code_to_conferral_date"


def test_field_plan_rejects_extract_year_before_compact_term_conferral():
    with pytest.raises(ValidationError, match="extract_year"):
        FieldTransformationPlan.model_validate(
            {
                "target_field": "associates_degree_conferral_date",
                "output_dtype": "datetime64[ns]",
                "confidence": 0.9,
                "steps": [
                    {"function_name": "strip_whitespace", "column": "term"},
                    {"function_name": "extract_year", "column": "term"},
                    {
                        "function_name": "compact_term_code_to_conferral_date",
                        "column": "term",
                    },
                ],
            }
        )


def test_field_plan_allows_compact_term_conferral_without_extract_year():
    plan = FieldTransformationPlan.model_validate(
        {
            "target_field": "associates_degree_conferral_date",
            "output_dtype": "datetime64[ns]",
            "confidence": 0.9,
            "steps": [
                {"function_name": "strip_whitespace", "column": "term"},
                {
                    "function_name": "compact_term_code_to_conferral_date",
                    "column": "term",
                },
            ],
        }
    )
    assert len(plan.steps) == 2


def test_field_plan_rejects_map_values_on_source_preserving_raw_edvise_field():
    with pytest.raises(ValidationError, match="map_values"):
        FieldTransformationPlan.model_validate(
            {
                "target_field": "enrollment_type",
                "output_dtype": "string",
                "confidence": 0.9,
                "steps": [
                    {
                        "function_name": "map_values",
                        "column": "raw_enroll",
                        "mapping": {"FT": "FIRST-TIME"},
                    },
                ],
            }
        )


def test_field_plan_allows_strip_and_cast_on_enrollment_type():
    plan = FieldTransformationPlan.model_validate(
        {
            "target_field": "enrollment_type",
            "output_dtype": "string",
            "confidence": 0.9,
            "steps": [
                {"function_name": "strip_whitespace", "column": "raw_enroll"},
                {"function_name": "cast_string", "column": "raw_enroll"},
            ],
        }
    )
    assert [s.function_name for s in plan.steps] == ["strip_whitespace", "cast_string"]


def test_hitl_item_rejects_map_values_on_conferred_credential_type():
    with pytest.raises(ValidationError, match="map_values"):
        TransformationHITLItem(
            item_id="lee_col_cohort_conferred_credential_type",
            institution_id="lee_col",
            entity_type=EntityType.cohort,
            target_field="conferred_credential_type",
            confidence=0.7,
            flagged_steps=[
                FlaggedStep(
                    step_index=0,
                    function_name="map_values",
                    reason="inferred_value_mapping",
                    context={},
                )
            ],
            steps=[
                {
                    "function_name": "map_values",
                    "column": "award",
                    "mapping": {"Assoc": "Associate's"},
                },
            ],
            options=[
                TransformationHITLOption(
                    option_id="approve",
                    label="A",
                    description="a",
                    resolution={"approved": True},
                ),
                TransformationHITLOption(
                    option_id="correct",
                    label="C",
                    description="c",
                    resolution=None,
                ),
                TransformationHITLOption(
                    option_id="hook_required",
                    label="H",
                    description="h",
                    resolution={"hook_required": True},
                ),
            ],
        )


def test_field_plan_rejects_map_values_on_term_degree():
    with pytest.raises(ValidationError, match="map_values"):
        FieldTransformationPlan.model_validate(
            {
                "target_field": "term_degree",
                "output_dtype": "string",
                "confidence": 0.9,
                "steps": [
                    {
                        "function_name": "map_values",
                        "column": "lvl",
                        "mapping": {"UG": "Undergraduate"},
                    },
                ],
            }
        )


def test_hitl_item_rejects_extract_year_before_compact_term_code_conferral():
    with pytest.raises(ValidationError, match="extract_year"):
        TransformationHITLItem(
            item_id="lee_col_cohort_associates_degree_conferral_date",
            institution_id="lee_col",
            entity_type=EntityType.cohort,
            target_field="associates_degree_conferral_date",
            confidence=0.7,
            flagged_steps=[
                FlaggedStep(
                    step_index=1,
                    function_name="compact_term_code_to_conferral_date",
                    reason="ambiguous_format",
                    context={},
                )
            ],
            steps=[
                {"function_name": "extract_year", "column": "term"},
                {
                    "function_name": "compact_term_code_to_conferral_date",
                    "column": "term",
                },
            ],
            options=[
                TransformationHITLOption(
                    option_id="approve",
                    label="A",
                    description="a",
                    resolution={"approved": True},
                ),
                TransformationHITLOption(
                    option_id="correct",
                    label="C",
                    description="c",
                    resolution=None,
                ),
                TransformationHITLOption(
                    option_id="hook_required",
                    label="H",
                    description="h",
                    resolution={"hook_required": True},
                ),
            ],
        )


def test_transformation_step_rejects_retired_new_utility_discriminator():
    adapter = TypeAdapter(TransformationStep)
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "function_name": "NEW_UTILITY_NEEDED",
                "description": "Need fuzzy date parser",
            }
        )


def test_transformation_step_rejects_legacy_conferral_function_names():
    adapter = TypeAdapter(TransformationStep)
    for legacy in ("term_season_to_conferral_date", "raw_term_token_to_conferral_date"):
        with pytest.raises(ValidationError):
            adapter.validate_python(
                {
                    "function_name": legacy,
                    "column": "x",
                    **(
                        {"extra_columns": {"season_series": "y"}}
                        if legacy == "term_season_to_conferral_date"
                        else {}
                    ),
                }
            )


def test_field_transformation_plan_hook_required():
    plan = FieldTransformationPlan(
        target_field="completion_term",
        output_dtype="category",
        hook_required=True,
        reviewer_notes="YYYYMM custom season split not covered by utilities.",
        steps=[],
    )
    assert plan.hook_required is True


def test_field_transformation_plan_rejects_hook_and_review_required():
    with pytest.raises(ValidationError, match="hook_required and review_required"):
        FieldTransformationPlan(
            target_field="completion_term",
            output_dtype="category",
            confidence=0.5,
            hook_required=True,
            review_required=True,
            steps=[],
            flagged_steps=[
                {
                    "step_index": 0,
                    "function_name": "cast_string",
                    "reason": "low_confidence_utility_chain",
                    "context": {},
                }
            ],
            hitl_options=[
                {"option_id": "approve", "label": "a", "description": "d"},
                {"option_id": "correct", "label": "c", "description": "d"},
                {"option_id": "hook_required", "label": "h", "description": "d"},
            ],
        )


def test_field_transformation_plan_review_required_requires_hitl_fields():
    opts = [
        TransformationHITLOption(
            option_id="approve",
            label="Approve mapping for deg_comp_term",
            description="Confirm proposed steps as-is for deg_comp_term.",
            resolution={"approved": True},
        ),
        TransformationHITLOption(
            option_id="correct",
            label="Correct steps",
            description="Supply corrected transformation steps.",
            resolution=None,
        ),
        TransformationHITLOption(
            option_id="hook_required",
            label="Hook required",
            description="Custom hook for field.",
            resolution={"hook_required": True},
        ),
    ]
    FieldTransformationPlan(
        target_field="bachelors_degree_conferral_date",
        output_dtype="datetime64[ns]",
        confidence=0.65,
        review_required=True,
        steps=[
            MapValuesStep(
                function_name="map_values",
                column="term_code",
                mapping={"01": "SPRING"},
            )
        ],
        flagged_steps=[
            FlaggedStep(
                step_index=0,
                function_name="map_values",
                reason="inferred_season_mapping",
                context={
                    "sample_values": ["202301.0"],
                    "inferred_mapping": {"01": "SPRING"},
                },
            )
        ],
        hitl_options=opts,
    )
    with pytest.raises(ValidationError, match="flagged_steps must be omitted"):
        FieldTransformationPlan(
            target_field="x",
            steps=[],
            flagged_steps=[
                FlaggedStep(
                    step_index=0,
                    function_name="cast_string",
                    reason="ambiguous_format",
                    context=None,
                )
            ],
        )


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
        review_status=ReviewStatus.auto_approved,
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
    assert tm.pipeline_version == importlib.metadata.version("edvise")
    assert len(tm.plans) == 1


def test_transformation_map_legacy_schema_version_key():
    tm = TransformationMap.model_validate(
        {
            "schema_version": "9.9.9",
            "institution_id": "x",
            "entity_type": "cohort",
            "target_schema": "RawEdviseStudentDataSchema",
            "plans": [{"target_field": "learner_id", "steps": []}],
        }
    )
    assert tm.pipeline_version == "9.9.9"
