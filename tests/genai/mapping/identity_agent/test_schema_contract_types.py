"""Canonical schema contract types: SMA frozen JSON + combined identity handoff."""

import pytest

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    DedupPolicy,
    GrainContract,
    InstitutionGrainContract,
)
from edvise.genai.mapping.identity_agent.identity_bundle import (
    InstitutionIdentityContract,
    institution_identity_contract_from_parts,
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    InstitutionTermContract,
    TermContract,
)
from edvise.genai.mapping.schema_contract import (
    BaseFrozenSchemaContract,
    EnrichedSchemaContractForSMA,
    parse_base_frozen_schema_contract,
    parse_enriched_schema_contract_for_sma,
)


def _grain(inst: str, table: str) -> GrainContract:
    return GrainContract(
        institution_id=inst,
        table=table,
        learner_id_alias=None,
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


def test_enriched_legacy_student_id_alias_maps_to_learner_id_alias():
    raw = {
        "school_id": "s1",
        "school_name": "School One",
        "student_id_alias": "raw_sid_col",
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
    assert m.learner_id_alias == "raw_sid_col"


def test_base_frozen_maps_build_schema_contract_student_id_alias_to_learner():
    """Raw dict uses ``student_id_alias`` (data audit); GenAI model exposes ``learner_id_alias``."""
    raw = {
        "created_at": "2024-01-01T00:00:00Z",
        "null_tokens": ["(Blank)"],
        "student_id_alias": "from_audit",
        "datasets": {
            "t": {
                "normalized_columns": {"A": "a"},
                "dtypes": {"a": "Int64"},
                "non_null_columns": [],
                "unique_keys": ["a"],
                "null_tokens": ["(Blank)"],
                "boolean_map": {"true": True},
            }
        },
    }
    m = parse_base_frozen_schema_contract(raw)
    assert m.learner_id_alias == "from_audit"


def test_base_frozen_schema_contract_minimal():
    raw = {
        "created_at": "2024-01-01T00:00:00Z",
        "null_tokens": ["(Blank)"],
        "learner_id_alias": "x",
        "datasets": {
            "t": {
                "normalized_columns": {"A": "a"},
                "dtypes": {"a": "Int64"},
                "non_null_columns": [],
                "unique_keys": ["a"],
                "null_tokens": ["(Blank)"],
                "boolean_map": {"true": True},
            }
        },
    }
    m = parse_base_frozen_schema_contract(raw)
    assert isinstance(m, BaseFrozenSchemaContract)
    assert m.learner_id_alias == "x"


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
