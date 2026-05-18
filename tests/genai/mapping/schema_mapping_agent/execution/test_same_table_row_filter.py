"""Same-table row_selection.filter masks source values without dropping base rows."""

import pandas as pd

from edvise.genai.mapping.schema_mapping_agent.execution.field_executor import (
    _apply_grain_reduction,
    _resolve_same_table_series,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingRecord,
    JoinFilter,
    RowSelectionConfig,
    RowSelectionStrategy,
)


def test_same_table_filter_masks_non_matching_rows() -> None:
    base_df = pd.DataFrame(
        {
            "learner_id": [1, 2, 3],
            "degree_term": ["2020 Spring", "2019 Fall", "2021 Spring"],
            "primary_degree": [
                "Associate of Arts",
                "Bachelor of Science",
                "Associate of Science",
            ],
        }
    )
    record = FieldMappingRecord(
        target_field="associates_degree_conferral_date",
        source_column="degree_term",
        source_table="student",
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.any_row,
            filter=JoinFilter(
                column="primary_degree",
                operator="contains",
                value="Associate",
            ),
        ),
        confidence=0.7,
    )
    out = _resolve_same_table_series(record, base_df, "student")
    assert out.tolist() == ["2020 Spring", pd.NA, "2021 Spring"]


def test_same_table_filter_isin() -> None:
    base_df = pd.DataFrame(
        {
            "id": [1, 2],
            "code": ["A", "B"],
            "tier": ["Associate of Arts", "Bachelor of Arts"],
        }
    )
    record = FieldMappingRecord(
        target_field="x",
        source_column="code",
        source_table="student",
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.any_row,
            filter=JoinFilter(
                column="tier",
                operator="isin",
                value=["Associate of Arts", "Associate of Science"],
            ),
        ),
        confidence=0.8,
    )
    out = _resolve_same_table_series(record, base_df, "student")
    assert out.tolist() == ["A", pd.NA]


def test_first_by_with_same_table_filter_sorts_only_passing_rows() -> None:
    """UCF-style: earliest row overall may fail the filter; first_by must use earliest passing row."""
    base_df = pd.DataFrame(
        {
            "learner_id": [1, 1, 1],
            "_term_order": [100, 200, 300],
            "highest_deg_earn": ["Associate", "Baccalaureate", "Baccalaureate"],
        }
    )
    # Simulates masked + transformed series: non-bacc rows would be NA from resolve;
    # values here are aligned to rows for clarity.
    s = pd.Series([pd.NA, "earliest_bacc_term", "later_bacc_term"])
    entity_keys = ["learner_id"]
    entity_index = base_df.drop_duplicates(subset=entity_keys)[entity_keys].reset_index(
        drop=True
    )
    record = FieldMappingRecord(
        target_field="bachelors_degree_conferral_date",
        source_column="degree_earned_term",
        source_table="student",
        row_selection=RowSelectionConfig(
            strategy=RowSelectionStrategy.first_by,
            order_by="_term_order",
            filter=JoinFilter(
                column="highest_deg_earn",
                operator="equals",
                value="Baccalaureate",
            ),
        ),
        confidence=0.85,
    )
    out = _apply_grain_reduction(s, record, base_df, entity_keys, entity_index)
    assert out.tolist() == ["earliest_bacc_term"]
