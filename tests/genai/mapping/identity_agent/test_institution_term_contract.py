"""Batch term stage: :class:`InstitutionTermContract` and user payload builder."""

import json

import pytest

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    DedupPolicy,
    GrainContract,
)
from edvise.genai.mapping.identity_agent.profiling.schemas import (
    RawColumnProfile,
    RawTableProfile,
)
from edvise.genai.mapping.identity_agent.term_normalization.prompt import (
    build_term_normalization_batch_user_payload,
    parse_institution_term_contracts,
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    InstitutionTermContract,
    TermContract,
)


def _grain(inst: str, table: str, *, row_sel: bool = True) -> GrainContract:
    return GrainContract(
        institution_id=inst,
        table=table,
        learner_id_alias=None,
        post_clean_primary_key=["sid", "term_code"],
        dedup_policy=DedupPolicy(
            strategy="no_dedup",
            sort_by=None,
            keep=None,
            notes="",
        ),
        row_selection_required=row_sel,
        join_keys_for_2a=["sid", "term_code"],
        confidence=0.9,
        hitl_flag=False,
        hitl_question=None,
        reasoning="",
        notes="",
    )


def _rtp(inst: str, dataset: str) -> RawTableProfile:
    col = RawColumnProfile(
        name="term_code",
        dtype="object",
        null_rate=0.0,
        unique_count=2,
        unique_values=["2018FA", "2019SP"],
        sample_values=["2018FA"],
        is_term_candidate=True,
    )
    return RawTableProfile(
        institution_id=inst,
        dataset=dataset,
        row_count=10,
        column_count=1,
        columns=[col],
    )


def _term(inst: str, table: str) -> TermContract:
    return TermContract(
        institution_id=inst,
        table=table,
        term_config=None,
        confidence=0.9,
        hitl_flag=False,
        hitl_question=None,
        reasoning="no term order needed",
    )


def test_parse_institution_term_contracts_fenced_json():
    env = InstitutionTermContract(
        institution_id="school_a",
        datasets={
            "enroll": _term("school_a", "enroll"),
            "grades": _term("school_a", "grades"),
        },
    )
    text = "```json\n" + env.model_dump_json(indent=2) + "\n```"
    got = parse_institution_term_contracts(text)
    assert got.institution_id == "school_a"
    assert set(got.datasets) == {"enroll", "grades"}


def test_institution_term_contract_table_key_must_match():
    bad = {
        "institution_id": "school_a",
        "datasets": {
            "enroll": _term("school_a", "wrong_name").model_dump(mode="json"),
        },
    }
    with pytest.raises(ValueError, match="map key"):
        InstitutionTermContract.model_validate(bad)


def test_build_term_normalization_batch_user_payload():
    inst = "school_a"
    grains = {"t1": _grain(inst, "t1"), "t2": _grain(inst, "t2", row_sel=False)}
    run = {
        "t1": {"raw_table_profile": _rtp(inst, "t1")},
        "t2": {"raw_table_profile": _rtp(inst, "t2")},
    }
    payload = build_term_normalization_batch_user_payload(inst, grains, run)
    assert payload["institution_id"] == inst
    assert set(payload["datasets"]) == {"t1", "t2"}
    inner = payload["datasets"]["t1"]
    assert inner["row_selection_required"] is True
    assert inner["grain_post_clean_primary_key"] == ["sid", "term_code"]
    assert len(inner["term_candidates"]) == 1
    assert inner["term_candidates"][0]["name"] == "term_code"
    # JSON-serializable
    json.dumps(payload)
