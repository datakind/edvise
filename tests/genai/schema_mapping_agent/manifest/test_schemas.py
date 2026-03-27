"""Unit tests for edvise.genai.schema_mapping_agent.manifest.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edvise.genai.schema_mapping_agent.manifest.schemas import (
    ColumnAlias,
    FieldMappingManifest,
    FieldMappingRecord,
    JoinConfig,
    JoinFilter,
    RowSelectionConfig,
    RowSelectionStrategy,
    get_manifest_schema_context,
)


def _minimal_mapping_record(**overrides) -> dict:
    base = {
        "target_field": "learner_id",
        "source_column": "student_id",
        "source_table": "cohort",
        "confidence": 1.0,
        "row_selection": {"strategy": "any_row"},
    }
    base.update(overrides)
    return base


def _minimal_manifest(**overrides) -> dict:
    base = {
        "institution_id": "test_cc",
        "entity_type": "cohort",
        "target_schema": "RawEdviseStudentDataSchema",
        "mappings": [_minimal_mapping_record()],
    }
    base.update(overrides)
    return base


def test_join_filter_isin_requires_list_value():
    with pytest.raises(ValidationError):
        JoinFilter(column="c", operator="isin", value="single")


def test_join_filter_non_isin_rejects_list_value():
    with pytest.raises(ValidationError):
        JoinFilter(column="c", operator="equals", value=["a", "b"])


def test_join_filter_isin_accepts_list():
    f = JoinFilter(column="c", operator="isin", value=["a", "b"])
    assert f.value == ["a", "b"]


def test_row_selection_first_by_requires_order_by():
    with pytest.raises(ValidationError, match="order_by"):
        RowSelectionConfig(strategy=RowSelectionStrategy.first_by)


def test_row_selection_where_not_null_requires_condition_col():
    with pytest.raises(ValidationError, match="condition_col"):
        RowSelectionConfig(strategy=RowSelectionStrategy.where_not_null)


def test_row_selection_nth_requires_n_and_order_by():
    with pytest.raises(ValidationError, match="n is required"):
        RowSelectionConfig(strategy=RowSelectionStrategy.nth, order_by="dt")
    with pytest.raises(ValidationError, match="order_by"):
        RowSelectionConfig(strategy=RowSelectionStrategy.nth, n=1)


def test_row_selection_fan_out_risk():
    assert (
        RowSelectionConfig(strategy=RowSelectionStrategy.any_row).fan_out_risk is False
    )
    assert (
        RowSelectionConfig(
            strategy=RowSelectionStrategy.first_by, order_by="x"
        ).fan_out_risk
        is True
    )


def test_field_mapping_record_source_column_requires_source_table():
    with pytest.raises(ValidationError, match="source_table must be set"):
        FieldMappingRecord(
            target_field="x",
            source_column="col",
            source_table=None,
            confidence=1.0,
            row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        )


def test_field_mapping_record_join_requires_source_table():
    with pytest.raises(ValidationError, match="source_table must be set when join"):
        FieldMappingRecord(
            target_field="x",
            source_column=None,
            source_table=None,
            confidence=1.0,
            join=JoinConfig(
                base_table="a",
                lookup_table="b",
                join_keys=["id"],
            ),
        )


def test_field_mapping_record_row_selection_without_source_only_constant_allowed():
    with pytest.raises(ValidationError, match="row_selection requires source_column"):
        FieldMappingRecord(
            target_field="x",
            source_column=None,
            source_table=None,
            confidence=1.0,
            row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
        )
    rec = FieldMappingRecord(
        target_field="x",
        source_column=None,
        source_table=None,
        confidence=1.0,
        row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.constant),
    )
    assert rec.source_column is None


def test_field_mapping_record_strict_forbids_extra_fields():
    with pytest.raises(ValidationError):
        FieldMappingRecord(
            **_minimal_mapping_record(),
            not_a_field=123,
        )


def test_field_mapping_manifest_unique_target_fields():
    with pytest.raises(ValidationError, match="only once"):
        FieldMappingManifest(
            **_minimal_manifest(
                mappings=[
                    _minimal_mapping_record(target_field="a"),
                    _minimal_mapping_record(target_field="a"),
                ]
            )
        )


def test_field_mapping_manifest_round_trip_minimal():
    data = _minimal_manifest(
        column_aliases=[
            {
                "table": "t",
                "source_column": "sid",
                "canonical_column": "student_id",
            }
        ]
    )
    m = FieldMappingManifest.model_validate(data)
    assert m.institution_id == "test_cc"
    assert len(m.mappings) == 1
    assert isinstance(m.column_aliases[0], ColumnAlias)


def test_get_manifest_schema_context_non_empty():
    text = get_manifest_schema_context()
    assert "JoinFilter" in text
    assert "FieldMappingManifest" in text
