"""HITL file envelopes and resolver-shaped config builders."""

import json

from edvise.genai.mapping.identity_agent.grain_inference.prompt_builder import parse_grain_contract_with_hitl
from edvise.genai.mapping.identity_agent.grain_inference.schemas import DedupPolicy, GrainContract
from edvise.genai.mapping.identity_agent.hitl.artifacts import (
    build_grain_config_for_resolver,
    write_identity_grain_artifacts,
)
from edvise.genai.mapping.identity_agent.term_normalization.prompt_builder import (
    parse_institution_term_contracts,
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import InstitutionTermContract, TermContract


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
        row_selection_required=False,
        join_keys_for_2a=["sid"],
        confidence=0.9,
        hitl_flag=False,
        hitl_question=None,
        reasoning="",
        notes="",
    )


def test_build_grain_config_for_resolver_shape():
    cfg = build_grain_config_for_resolver("u1", {"t1": _grain("u1", "t1")})
    assert cfg["institution_id"] == "u1"
    assert "grain_contract" in cfg["datasets"]["t1"]
    assert cfg["datasets"]["t1"]["grain_contract"]["table"] == "t1"


def test_write_identity_grain_artifacts_roundtrip(tmp_path):
    gc = _grain("u1", "t1")
    cfg_p, hitl_p = write_identity_grain_artifacts(tmp_path, "u1", {"t1": gc}, [])
    assert cfg_p.exists() and hitl_p.exists()
    env = json.loads(hitl_p.read_text())
    assert env["institution_id"] == "u1"
    assert env["domain"] == "grain"
    assert env["items"] == []


def test_parse_grain_contract_with_hitl_no_items():
    gc = _grain("u1", "t1")
    d = gc.model_dump(mode="json")
    out, items = parse_grain_contract_with_hitl(d)
    assert out.table == "t1"
    assert items == []


def test_parse_institution_term_contracts_strips_empty_hitl_items():
    inst = InstitutionTermContract(
        institution_id="school_a",
        datasets={"enroll": TermContract(
            institution_id="school_a",
            table="enroll",
            term_config=None,
            confidence=0.9,
            hitl_flag=False,
            hitl_question=None,
            reasoning="x",
        )},
    )
    payload = json.loads(inst.model_dump_json())
    payload["hitl_items"] = []
    payload["datasets"]["enroll"]["hitl_items"] = []
    got = parse_institution_term_contracts(json.dumps(payload))
    assert got.institution_id == "school_a"
