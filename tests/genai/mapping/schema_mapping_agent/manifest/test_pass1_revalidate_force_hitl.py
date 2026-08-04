"""Post–Pass 1 validate_manifest re-check forces incomplete refined_by_llm onto HITL."""

from __future__ import annotations

from edvise.genai.mapping.schema_mapping_agent.manifest.prompts.refine import (
    _revalidate_pass1_and_force_hitl,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    JoinConfig,
    ReviewStatus,
    RowSelectionConfig,
    RowSelectionStrategy,
)
from edvise.genai.mapping.shared.schema_contract import (
    parse_enriched_schema_contract_for_sma,
)


def _cd(name: str) -> dict:
    return {
        "original_name": name,
        "normalized_name": name,
        "null_count": 0,
        "null_percentage": 0.0,
        "unique_count": 1,
        "sample_values": [],
    }


def _contract():
    return parse_enriched_schema_contract_for_sma(
        {
            "school_id": "x",
            "school_name": "X",
            "datasets": {
                "course": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": [],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/c.csv",
                        "num_rows": 1,
                        "num_columns": 3,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("term_descr"),
                            _cd("course_grade"),
                        ],
                    },
                },
                "student": {
                    "normalized_columns": {},
                    "dtypes": {},
                    "non_null_columns": [],
                    "unique_keys": [],
                    "null_tokens": [],
                    "boolean_map": {},
                    "training": {
                        "file_path": "/s.csv",
                        "num_rows": 1,
                        "num_columns": 3,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            _cd("learner_id"),
                            _cd("term_desc"),
                            _cd("ugrd_grad_flag"),
                        ],
                    },
                },
            },
        }
    )


def test_revalidate_forces_incomplete_join_fix_to_hitl():
    """Production footgun: Pass 1 renames join key only, stamps refined_by_llm."""
    manifest = FieldMappingManifest(
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
                review_status=ReviewStatus.auto_approved,
            ),
            FieldMappingRecord(
                target_field="term_degree",
                source_column="ugrd_grad_flag",
                source_table="student",
                join=JoinConfig(
                    base_table="course",
                    lookup_table="student",
                    join_keys=["learner_id", "term_descr"],
                ),
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=0.75,
                rationale="",
                review_status=ReviewStatus.refined_by_llm,
            ),
        ],
        column_aliases=[],
    )

    flags, warnings = _revalidate_pass1_and_force_hitl(
        "inst",
        "course",
        manifest,
        hitl_flags=[],
        schema_contract=_contract(),
        refined_corrections={
            "term_degree": {
                "join": {
                    "base_table": "course",
                    "lookup_table": "student",
                    "join_keys": ["learner_id", "term_descr"],
                }
            }
        },
    )

    td = next(m for m in manifest.mappings if m.target_field == "term_degree")
    assert td.review_status == ReviewStatus.refined_and_proposed_for_hitl
    assert len(flags) == 1
    assert flags[0]["target_field"] == "term_degree"
    assert flags[0]["failure_mode"] == "join_structure"
    assert any("refined_by_llm" in w for w in warnings)


def test_revalidate_noop_when_manifest_clean():
    manifest = FieldMappingManifest(
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
                review_status=ReviewStatus.auto_approved,
            ),
            FieldMappingRecord(
                target_field="course_grade",
                source_column="course_grade",
                source_table="course",
                join=None,
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=1.0,
                rationale="",
                review_status=ReviewStatus.auto_approved,
            ),
        ],
        column_aliases=[],
    )
    flags, warnings = _revalidate_pass1_and_force_hitl(
        "inst",
        "course",
        manifest,
        hitl_flags=[],
        schema_contract=_contract(),
        refined_corrections={},
    )
    # Grain keys beyond learner_id may still flag ENTITY_GRAIN — only assert
    # that fields we marked auto_approved without join issues stay clean when
    # no JOIN/COLUMN errors remain for those targets.
    join_flags = [f for f in flags if f.get("failure_mode") == "join_structure"]
    assert join_flags == []
    assert not any("term_degree" in (w or "") for w in warnings)
