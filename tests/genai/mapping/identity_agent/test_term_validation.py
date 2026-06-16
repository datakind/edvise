"""Emit-time validation for term hook_group_tables vs term_config column shapes."""

import json

import pytest

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    HookFunctionSpec,
    HookSpec,
)
from edvise.genai.mapping.identity_agent.hitl.artifacts import (
    write_identity_term_artifacts,
)
from edvise.genai.mapping.identity_agent.hitl.resolver import (
    HITLValidationError,
    validate_term_hook_hitl_covers_hook_required,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import (
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLTarget,
    InstitutionHITLItems,
    ReentryDepth,
    TermResolution,
)
from edvise.genai.mapping.identity_agent.term_normalization.prompt import (
    parse_institution_term_contracts_with_hitl,
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    InstitutionTermContract,
    TermContract,
    TermOrderConfig,
)
from edvise.genai.mapping.identity_agent.term_normalization.validation import (
    assert_term_hook_groups_compatible,
)

INST = "indiana_institute_of_technology"


def _hook_spec() -> HookSpec:
    return HookSpec(
        file=f"identity_hooks/{INST}/term_hooks.py",
        functions=[
            HookFunctionSpec(
                name="year_extractor_shared",
                signature="def year_extractor_shared(term: str) -> int",
                description="year",
                draft="int(str(term).split('-')[0]) if '-' in str(term) else None",
            ),
            HookFunctionSpec(
                name="season_extractor_shared",
                signature="def season_extractor_shared(term: str) -> str",
                description="season",
                draft=("str(term).split('-')[1] if '-' in str(term) else str(term)"),
            ),
        ],
    )


def _hook_term_contract(table: str, *, term_col: str) -> TermContract:
    return TermContract(
        institution_id=INST,
        table=table,
        term_config=TermOrderConfig(
            term_col=term_col,
            season_map=[],
            term_extraction="hook_required",
            hook_spec=_hook_spec(),
        ),
        confidence=0.75,
        hitl_flag=True,
        reasoning="opaque combined term encoding",
    )


def _split_standard_student() -> TermContract:
    return TermContract(
        institution_id=INST,
        table="student",
        term_config=TermOrderConfig(
            year_col="year_col",
            season_col="term_code",
            season_map=[],
            term_extraction="standard",
        ),
        confidence=0.85,
        hitl_flag=True,
        reasoning="split year and opaque season codes",
    )


def _shared_hitl_item() -> HITLItem:
    resolution = TermResolution(
        hook_spec=_hook_spec(),
        season_map_replace=[
            {"raw": "10", "canonical": "FALL"},
            {"raw": "20", "canonical": "SPRING"},
        ],
    )
    return HITLItem(
        item_id="shared_term_encoding_season_map_confirmation",
        institution_id=INST,
        table="student",
        domain=HITLDomain.IDENTITY_TERM,
        hook_group_id="shared_term_encoding_iit",
        hook_group_tables=["student", "course", "semester"],
        hitl_question="Confirm season mapping",
        hitl_context="student uses 10/20; course uses YYYY-TT",
        options=[
            HITLOption(
                option_id="confirm_10_fall_20_spring",
                label="10→FALL, 20→SPRING",
                description="confirm",
                resolution=resolution.model_dump(mode="json"),
                reentry=ReentryDepth.GENERATE_HOOK,
            ),
            HITLOption(
                option_id="custom",
                label="Custom",
                description="custom",
                resolution=TermResolution(
                    season_map_replace=resolution.season_map_replace
                ).model_dump(mode="json"),
                reentry=ReentryDepth.GENERATE_HOOK,
            ),
        ],
        target=HITLTarget(
            institution_id=INST,
            table="student",
            config="term_config",
            field="season_map",
        ),
    )


def test_assert_term_hook_groups_compatible_valid_group():
    inst = InstitutionTermContract(
        institution_id=INST,
        datasets={
            "student": _hook_term_contract("student", term_col="term_col"),
            "course": _hook_term_contract("course", term_col="semester"),
        },
    )
    item = _shared_hitl_item()
    item.hook_group_tables = ["student", "course"]
    assert_term_hook_groups_compatible(inst, [item])


def test_assert_term_hook_groups_compatible_rejects_split_student_in_group():
    inst = InstitutionTermContract(
        institution_id=INST,
        datasets={
            "student": _split_standard_student(),
            "course": _hook_term_contract("course", term_col="semester"),
            "semester": _hook_term_contract("semester", term_col="semester"),
        },
    )
    with pytest.raises(ValueError, match="split year_col/season_col"):
        assert_term_hook_groups_compatible(inst, [_shared_hitl_item()])


def test_parse_institution_term_contracts_with_hitl_rejects_bad_hook_group(
    tmp_path,
):
    payload = {
        "institution_id": INST,
        "datasets": {
            "student": _split_standard_student().model_dump(mode="json"),
            "course": _hook_term_contract("course", term_col="semester").model_dump(
                mode="json"
            ),
            "semester": _hook_term_contract("semester", term_col="semester").model_dump(
                mode="json"
            ),
        },
        "hitl_items": [_shared_hitl_item().model_dump(mode="json")],
    }
    with pytest.raises(ValueError, match="split year_col/season_col"):
        parse_institution_term_contracts_with_hitl(json.dumps(payload))


def test_write_identity_term_artifacts_rejects_bad_hook_group(tmp_path):
    contracts = {
        "student": _split_standard_student(),
        "course": _hook_term_contract("course", term_col="semester"),
        "semester": _hook_term_contract("semester", term_col="semester"),
    }
    with pytest.raises(ValueError, match="split year_col/season_col"):
        write_identity_term_artifacts(
            tmp_path,
            INST,
            contracts,
            [_shared_hitl_item()],
        )


def test_validate_term_hook_hitl_covers_hook_required_rejects_bad_group(tmp_path):
    hitl_path = tmp_path / "identity_term_hitl.json"
    item = _shared_hitl_item()
    item.choice = 1
    hitl_path.write_text(
        InstitutionHITLItems(
            institution_id=INST,
            domain="term",
            items=[item],
        ).model_dump_json(indent=2)
    )
    contracts = {
        "student": _split_standard_student(),
        "course": _hook_term_contract("course", term_col="semester"),
        "semester": _hook_term_contract("semester", term_col="semester"),
    }
    with pytest.raises(HITLValidationError, match="split year_col/season_col"):
        validate_term_hook_hitl_covers_hook_required(
            term_hitl_path=hitl_path,
            term_contract_by_dataset=contracts,
        )
