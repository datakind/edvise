"""HITL flag comparison over raw Step 2a manifests."""

from __future__ import annotations

from edvise.genai.mapping.eval.hitl_compare import (
    _compare_rows,
    _summarize_by_failure_mode,
    _summarize_by_school,
    manifest_hitl_rows,
)
from edvise.genai.mapping.shared.schema_contract import (
    parse_enriched_schema_contract_for_sma,
)


def _contract():
    return parse_enriched_schema_contract_for_sma(
        {
            "school_id": "school_a",
            "school_name": "School A",
            "datasets": {
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
                        "num_columns": 1,
                        "column_normalization": {"original_to_normalized": {}},
                        "column_details": [
                            {
                                "original_name": "learner_id",
                                "normalized_name": "learner_id",
                                "null_count": 0,
                                "null_percentage": 0.0,
                                "unique_count": 1,
                                "sample_values": [],
                            }
                        ],
                    },
                }
            },
        }
    )


def _manifest(confidence: float, source_column: str) -> dict:
    return {
        "institution_id": "school_a",
        "manifests": {
            "cohort": {
                "entity_type": "cohort",
                "target_schema": "CustomEvalSchema",
                "mappings": [
                    {
                        "target_field": "student_id",
                        "source_column": source_column,
                        "source_table": "student",
                        "row_selection": {"strategy": "any_row"},
                        "confidence": confidence,
                        "rationale": "mapped from learner id",
                    }
                ],
            }
        },
    }


def test_low_confidence_and_missing_column_are_separate_flags():
    contract = _contract()
    low = manifest_hitl_rows(
        _manifest(0.4, "learner_id"),
        schema_contract=contract,
        institution_id="school_a",
        label="sonnet_4_5",
    )
    missing = manifest_hitl_rows(
        _manifest(0.95, "not_a_column"),
        schema_contract=contract,
        institution_id="school_a",
        label="sonnet_5",
    )

    assert low[0]["hitl_flagged"] is True
    assert low[0]["failure_mode"] == "low_confidence"
    assert low[0]["error_codes"] == ""

    assert missing[0]["hitl_flagged"] is True
    assert missing[0]["failure_mode"] == "column_not_found"
    assert "SOURCE_COLUMN_NOT_FOUND" in missing[0]["error_codes"]


def test_high_confidence_valid_mapping_is_not_flagged():
    rows = manifest_hitl_rows(
        _manifest(0.95, "learner_id"),
        schema_contract=_contract(),
        institution_id="school_a",
    )
    assert rows[0]["hitl_flagged"] is False
    assert rows[0]["failure_mode"] == ""


def test_comparison_buckets_and_summaries():
    contract = _contract()
    left = manifest_hitl_rows(
        _manifest(0.4, "learner_id"),
        schema_contract=contract,
        institution_id="school_a",
    )
    right_payload = _manifest(0.95, "learner_id")
    right_payload["manifests"]["cohort"]["mappings"].append(
        {
            "target_field": "credits",
            "source_column": "not_a_column",
            "source_table": "student",
            "row_selection": {"strategy": "any_row"},
            "confidence": 0.9,
            "rationale": "guess",
        }
    )
    right = manifest_hitl_rows(
        right_payload,
        schema_contract=contract,
        institution_id="school_a",
    )
    fields = _compare_rows(left, right, "sonnet_4_5", "sonnet_5")
    buckets = dict(zip(fields["target_field"], fields["flag_bucket"]))
    assert buckets["student_id"] == "sonnet_4_5_only"
    assert buckets["credits"] == "sonnet_5_only"

    by_school = _summarize_by_school(fields, "sonnet_4_5", "sonnet_5")
    assert int(by_school.loc[0, "sonnet_4_5_only"]) == 1
    assert int(by_school.loc[0, "sonnet_5_only"]) == 1

    by_mode = _summarize_by_failure_mode(fields, "sonnet_4_5", "sonnet_5")
    modes = dict(
        zip(by_mode["failure_mode"], zip(by_mode["sonnet_4_5"], by_mode["sonnet_5"]))
    )
    assert modes["low_confidence"] == (1, 0)
    assert modes["column_not_found"] == (0, 1)
