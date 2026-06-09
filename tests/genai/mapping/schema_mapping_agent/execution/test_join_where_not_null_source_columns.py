"""Join + where_not_null: condition_col lives on lookup, not on manifest base_df."""

import pandas as pd

from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
from edvise.genai.mapping.schema_mapping_agent.execution.field_executor import (
    execute_transformation_map,
    validate_source_columns_for_execute,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    JoinConfig,
    ReviewStatus,
    RowSelectionConfig,
    RowSelectionStrategy,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
    FieldTransformationPlan,
    NormalizePellStep,
    TransformationMap,
)


def test_validate_does_not_require_lookup_condition_col_on_base_table() -> None:
    """Regression: Odessa-style pell_recipient_year1 from terms.pell_awarded + join."""
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[
            FieldMappingRecord(
                target_field="learner_id",
                source_column="learner_id",
                source_table="student",
                join=None,
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=1.0,
                rationale="",
            ),
            FieldMappingRecord(
                target_field="pell_recipient_year1",
                source_column="pell_awarded",
                source_table="terms",
                join=JoinConfig(
                    base_table="student",
                    lookup_table="terms",
                    join_keys=["learner_id"],
                ),
                row_selection=RowSelectionConfig(
                    strategy=RowSelectionStrategy.where_not_null,
                    condition_col="pell_awarded",
                ),
                confidence=0.85,
                rationale="",
            ),
        ],
        column_aliases=[],
    )
    tm = TransformationMap(
        institution_id="test_inst",
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        plans=[
            FieldTransformationPlan(target_field="learner_id", steps=[]),
            FieldTransformationPlan(
                target_field="pell_recipient_year1",
                steps=[
                    NormalizePellStep(
                        function_name="normalize_pell",
                        column="pell_awarded",
                    )
                ],
                confidence=0.95,
                review_status=ReviewStatus.corrected_by_hitl,
            ),
        ],
    )
    student = pd.DataFrame({"learner_id": ["a", "b"]})
    terms = pd.DataFrame(
        {
            "learner_id": ["a", "a", "b"],
            "pell_awarded": [True, False, False],
        }
    )
    validate_source_columns_for_execute(
        tm,
        manifest,
        RawEdviseStudentDataSchema,
        {"student": student, "terms": terms},
    )


def test_execute_join_where_not_null_without_condition_on_base() -> None:
    manifest = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[
            FieldMappingRecord(
                target_field="learner_id",
                source_column="learner_id",
                source_table="student",
                join=None,
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=1.0,
                rationale="",
            ),
            FieldMappingRecord(
                target_field="pell_recipient_year1",
                source_column="pell_awarded",
                source_table="terms",
                join=JoinConfig(
                    base_table="student",
                    lookup_table="terms",
                    join_keys=["learner_id"],
                ),
                row_selection=RowSelectionConfig(
                    strategy=RowSelectionStrategy.where_not_null,
                    condition_col="pell_awarded",
                ),
                confidence=0.85,
                rationale="",
            ),
        ],
        column_aliases=[],
    )
    tm = TransformationMap(
        institution_id="test_inst",
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        plans=[
            FieldTransformationPlan(target_field="learner_id", steps=[]),
            FieldTransformationPlan(
                target_field="pell_recipient_year1",
                steps=[
                    NormalizePellStep(
                        function_name="normalize_pell",
                        column="pell_awarded",
                    )
                ],
                confidence=0.95,
                review_status=ReviewStatus.corrected_by_hitl,
            ),
        ],
    )
    student = pd.DataFrame({"learner_id": ["a", "b"]})
    terms = pd.DataFrame(
        {
            "learner_id": ["a", "a", "b"],
            "pell_awarded": [True, False, False],
        }
    )
    out = execute_transformation_map(
        tm,
        manifest,
        {"student": student, "terms": terms},
        RawEdviseStudentDataSchema,
        institution_id="test_inst",
    )
    assert len(out.df) == 2
    assert "pell_recipient_year1" in out.df.columns
