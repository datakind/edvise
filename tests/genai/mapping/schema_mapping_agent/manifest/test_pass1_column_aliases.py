"""Pass 1 ``column_aliases_to_add`` merges into the refined manifest."""

from __future__ import annotations

from edvise.genai.mapping.schema_mapping_agent.manifest.prompts.refine import (
    _apply_pass1_result,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    JoinConfig,
    RowSelectionConfig,
    RowSelectionStrategy,
)


def test_apply_pass1_result_merges_column_aliases_to_add():
    input_manifest = FieldMappingManifest(
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        mappings=[
            FieldMappingRecord(
                target_field="learner_id",
                source_column="learner_id",
                source_table="course",
                join=None,
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=1.0,
                rationale="",
            ),
            FieldMappingRecord(
                target_field="term_degree",
                source_column="ugrd_grad_flag",
                source_table="student",
                join=JoinConfig(
                    base_table="course",
                    lookup_table="student",
                    join_keys=["learner_id", "term_desc"],
                ),
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=0.9,
                rationale="",
            ),
        ],
        column_aliases=[],
    )
    pass1 = {
        "field_statuses": {
            "learner_id": "auto_approved",
            "term_degree": "refined_by_llm",
        },
        "refined_corrections": {
            "term_degree": {
                "join": {
                    "base_table": "course",
                    "lookup_table": "student",
                    "join_keys": ["learner_id", "term_descr"],
                },
                "validation_notes": "Join key corrected; alias bridges student term_desc.",
            }
        },
        "column_aliases_to_add": [
            {
                "table": "student",
                "source_column": "term_desc",
                "canonical_column": "term_descr",
                "rationale": "student physical term column",
            }
        ],
        "hitl_flags": [],
    }

    refined, flags = _apply_pass1_result("inst", input_manifest, pass1)

    assert flags == []
    assert len(refined.column_aliases) == 1
    alias = refined.column_aliases[0]
    assert alias.table == "student"
    assert alias.source_column == "term_desc"
    assert alias.canonical_column == "term_descr"
    td = next(m for m in refined.mappings if m.target_field == "term_degree")
    assert td.join is not None
    assert td.join.join_keys == ["learner_id", "term_descr"]
    assert td.review_status == "refined_by_llm"


def test_apply_pass1_result_dedupes_existing_alias():
    from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import ColumnAlias

    input_manifest = FieldMappingManifest(
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        mappings=[
            FieldMappingRecord(
                target_field="learner_id",
                source_column="learner_id",
                source_table="course",
                join=None,
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=1.0,
                rationale="",
            ),
        ],
        column_aliases=[
            ColumnAlias(
                table="student",
                source_column="term_desc",
                canonical_column="term_descr",
            )
        ],
    )
    pass1 = {
        "field_statuses": {"learner_id": "auto_approved"},
        "refined_corrections": {},
        "column_aliases_to_add": [
            {
                "table": "student",
                "source_column": "term_desc",
                "canonical_column": "term_descr",
            }
        ],
        "hitl_flags": [],
    }
    refined, _ = _apply_pass1_result("inst", input_manifest, pass1)
    assert len(refined.column_aliases) == 1
