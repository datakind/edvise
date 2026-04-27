"""Term batch gateway JSON schema accepts :class:`InstitutionTermContract` output, not user profile JSON."""

import json

import jsonschema
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
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    InstitutionTermContract,
    TermContract,
)
from edvise.genai.mapping.shared.schema_utils import (
    identity_term_batch_envelope_response_format,
)


def _term_batch_response_schema() -> dict:
    rf = identity_term_batch_envelope_response_format()
    return rf["json_schema"]["schema"]  # type: ignore[return-value, index]


def _grain(inst: str, table: str) -> GrainContract:
    return GrainContract(
        institution_id=inst,
        table=table,
        learner_id_alias=None,
        post_clean_primary_key=["sid", "t"],
        dedup_policy=DedupPolicy(
            strategy="no_dedup", sort_by=None, keep=None, notes=""
        ),
        row_selection_required=True,
        join_keys_for_2a=["sid", "t"],
        confidence=0.9,
        hitl_flag=False,
        reasoning="",
        notes="",
    )


def _rtp(inst: str, dataset: str) -> RawTableProfile:
    c = RawColumnProfile(
        name="term_code",
        dtype="object",
        null_rate=0.0,
        null_rate_including_tokens=0.0,
        unique_count=1,
        unique_values=["1"],
        sample_values=["1"],
        is_term_candidate=True,
    )
    return RawTableProfile(
        institution_id=inst,
        dataset=dataset,
        row_count=1,
        column_count=1,
        columns=[c],
    )


def test_term_batch_schema_accepts_institution_output():
    inst = "school_i"
    env = InstitutionTermContract(
        institution_id=inst,
        datasets={"enroll": TermContract(
            institution_id=inst,
            table="enroll",
            term_config=None,
            confidence=0.9,
            hitl_flag=False,
            reasoning="n/a",
        )},
    )
    data = json.loads(env.model_dump_json())
    # Raw model output may include a top-level hitl_items; parser strips it.
    data["hitl_items"] = []
    jsonschema.validate(data, _term_batch_response_schema())


def test_term_batch_schema_rejects_user_profile_shape():
    inst = "school_i"
    grains = {"t1": _grain(inst, "t1")}
    run = {"t1": {"raw_table_profile": _rtp(inst, "t1")}}
    user_payload = build_term_normalization_batch_user_payload(inst, grains, run)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(user_payload, _term_batch_response_schema())
