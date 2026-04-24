"""Pure helpers for the GenAI HITL Streamlit bundle (no Streamlit runtime)."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_HITL_APP = _REPO / "src/edvise/genai/mapping/streamlit-genai-hitl-app"
if str(_HITL_APP) not in sys.path:
    sys.path.insert(0, str(_HITL_APP))

from hitl_reviewer.sma import enriched_schema_contract as sc


def test_enriched_schema_contract_path_from_manifest():
    p = (
        "/Volumes/dev_cat/uni_of_central_florida_silver/silver_volume/"
        "genai_mapping/runs/onboard/run-abc/schema_mapping_agent/cohort_hitl_manifest.json"
    )
    assert sc.enriched_schema_contract_path_from_manifest(p, "run-abc") == (
        "/Volumes/dev_cat/uni_of_central_florida_silver/silver_volume/"
        "genai_mapping/runs/onboard/run-abc/identity_agent/enriched_schema_contract.json"
    )
    assert sc.enriched_schema_contract_path_from_manifest(p, "wrong") is None


def test_silver_relative_path():
    p = "/Volumes/c/inst_silver/silver_volume/genai_mapping/x.json"
    assert sc.silver_relative_path(p) == "silver_volume/genai_mapping/x.json"
    assert sc.silver_relative_path("/tmp/nope") is None


def test_extract_column_panel_fields_unique_vs_sample():
    contract = {
        "null_tokens": ["(Blank)"],
        "datasets": {
            "courses": {
                "dtypes": {"col_a": "string"},
                "null_tokens": [],
                "training": {
                    "column_details": [
                        {
                            "original_name": "Col A",
                            "normalized_name": "col_a",
                            "null_count": 1,
                            "null_percentage": 10.0,
                            "unique_count": 2,
                            "sample_values": ["x", "y"],
                            "unique_values": ["u1", "u2"],
                        }
                    ],
                },
            }
        },
    }
    panel = sc.extract_column_panel_fields(
        contract, dataset_name="courses", source_column="col_a"
    )
    assert panel is not None
    assert panel["chip_mode"] == "unique"
    assert panel["chip_values"] == ["u1", "u2"]
    assert "(Blank)" in panel["inst_null_tokens"]


def test_format_institution_display_name():
    from edvise.utils.institution_naming import format_institution_display_name

    assert "Community College" in format_institution_display_name(
        "fixture_alpha_state_cc"
    )
