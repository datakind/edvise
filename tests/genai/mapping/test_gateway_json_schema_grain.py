"""IA grain gateway JSON schema vs :class:`GrainContract` (Pydantic)."""

import json

import jsonschema
import pytest

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    DedupPolicy,
    GrainContract,
)
from edvise.genai.mapping.shared.gateway_json_schema import (
    identity_grain_contract_response_format,
)


def _grain_schema() -> dict:
    rf = identity_grain_contract_response_format()
    return rf["json_schema"]["schema"]  # type: ignore[return-value, index]


def test_grain_schema_accepts_minimal_valid_grain_contract():
    gc = GrainContract(
        institution_id="u1",
        table="student",
        learner_id_alias=None,
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
        reasoning="r",
    )
    data = json.loads(gc.model_dump_json())
    jsonschema.validate(data, _grain_schema())


def test_grain_schema_rejects_extra_top_level_key():
    gc = GrainContract(
        institution_id="u1",
        table="student",
        post_clean_primary_key=["sid"],
        dedup_policy=DedupPolicy(strategy="no_dedup", notes=""),
        row_selection_required=False,
        join_keys_for_2a=["sid"],
        confidence=0.9,
        hitl_flag=False,
        reasoning="r",
    )
    data = json.loads(gc.model_dump_json())
    data["not_a_grain_field"] = True
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(data, _grain_schema())


def test_grain_schema_rejects_loose_dedup_policy_key():
    payload = {
        "institution_id": "u1",
        "table": "t",
        "learner_id_alias": None,
        "post_clean_primary_key": ["k"],
        "dedup_policy": {
            "strategy": "no_dedup",
            "sort_by": None,
            "sort_ascending": None,
            "keep": None,
            "notes": "",
            "made_up_field": 1,
        },
        "row_selection_required": False,
        "join_keys_for_2a": ["k"],
        "confidence": 0.5,
        "hitl_flag": True,
        "reasoning": "r",
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(payload, _grain_schema())


def test_grain_schema_rejects_invalid_strategy():
    payload = {
        "institution_id": "u1",
        "table": "t",
        "learner_id_alias": None,
        "post_clean_primary_key": ["k"],
        "dedup_policy": {"strategy": "invalid_strategy"},
        "row_selection_required": False,
        "join_keys_for_2a": ["k"],
        "confidence": 0.5,
        "hitl_flag": True,
        "reasoning": "r",
    }
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(payload, _grain_schema())
