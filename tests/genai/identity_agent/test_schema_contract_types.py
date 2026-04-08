"""Canonical schema contract types: SMA frozen JSON + combined identity handoff."""

import pytest

from edvise.genai.identity_agent.grain_inference.schemas import (
    DedupPolicy,
    GrainContract,
    InstitutionGrainContract,
)
from edvise.genai.identity_agent.identity_bundle import (
    InstitutionIdentityContract,
    institution_identity_contract_from_parts,
)
from edvise.genai.identity_agent.term_normalization.schemas import (
    InstitutionTermContract,
    TermContract,
)
from edvise.genai.identity_agent.execution.sma_schema_contract import (
    EnrichedSchemaContractForSMA,
    parse_enriched_schema_contract_for_sma,
)


def _grain(inst: str, table: str) -> GrainContract:
    return GrainContract(
        institution_id=inst,
        table=table,
        student_id_alias=None,
        post_clean_primary_key=["sid"],
        dedup_policy=DedupPolicy(
            strategy="no_dedup",
            sort_by=None,
            keep=None,
            notes="",
        ),
        row_selection_required=True,
        join_keys_for_2a=["sid"],
        confidence=0.9,
        hitl_flag=False,
        hitl_question=None,
        reasoning="",
        notes="",
    )


def _term(inst: str, table: str) -> TermContract:
    return TermContract(
        institution_id=inst,
        table=table,
        term_config=None,
        confidence=0.9,
        hitl_flag=False,
        hitl_question=None,
        reasoning="ok",
    )


def test_institution_identity_contract_roundtrip():
    inst = "school_a"
    grain = InstitutionGrainContract(
        institution_id=inst,
        datasets={"t1": _grain(inst, "t1")},
    )
    term = InstitutionTermContract(
        institution_id=inst,
        datasets={"t1": _term(inst, "t1")},
    )
    bundle = institution_identity_contract_from_parts(grain, term)
    assert bundle.institution_id == inst
    assert set(bundle.grain.datasets) == {"t1"}

    again = InstitutionIdentityContract.model_validate(bundle.model_dump(mode="json"))
    assert again.model_dump(mode="json") == bundle.model_dump(mode="json")


def test_institution_identity_contract_rejects_mismatched_dataset_keys():
    inst = "school_a"
    grain = InstitutionGrainContract(
        institution_id=inst,
        datasets={"t1": _grain(inst, "t1")},
    )
    term = InstitutionTermContract(
        institution_id=inst,
        datasets={"t2": _term(inst, "t2")},
    )
    with pytest.raises(ValueError, match="same keys"):
        institution_identity_contract_from_parts(grain, term)


def test_enriched_schema_contract_for_sma_minimal():
    raw = {
        "school_id": "s1",
        "school_name": "School One",
        "datasets": {
            "students": {
                "normalized_columns": {"A": "a"},
                "dtypes": {"a": "Int64"},
                "non_null_columns": [],
                "unique_keys": ["a"],
                "null_tokens": ["(Blank)"],
                "boolean_map": {"true": True, "false": False},
                "training": {
                    "file_path": "/x.csv",
                    "num_rows": 1,
                    "num_columns": 1,
                    "column_normalization": {"original_to_normalized": {}},
                    "column_details": [
                        {
                            "original_name": "A",
                            "normalized_name": "a",
                            "null_count": 0,
                            "null_percentage": 0.0,
                            "unique_count": 1,
                            "sample_values": ["1"],
                        }
                    ],
                },
            }
        },
    }
    m = parse_enriched_schema_contract_for_sma(raw)
    assert isinstance(m, EnrichedSchemaContractForSMA)
    assert m.school_id == "s1"
    assert "students" in m.datasets
    assert m.datasets["students"].training.num_rows == 1
