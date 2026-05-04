"""Guardrails: hook_required term_config vs GENERATE_HOOK HITL items."""

import json

import pytest

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    HookFunctionSpec,
    HookSpec,
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
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    TermContract,
    TermOrderConfig,
)


def _hs() -> HookSpec:
    return HookSpec(
        file="identity_hooks/u/term_hooks.py",
        functions=[
            HookFunctionSpec(
                name="year_x",
                description="y",
                draft="def year_x(t): return 2020",
            ),
            HookFunctionSpec(
                name="season_x",
                description="s",
                draft="def season_x(t): return '01'",
            ),
        ],
    )


def _term_item_generate_hook(item_id: str = "hook_a") -> HITLItem:
    return HITLItem(
        item_id=item_id,
        institution_id="u",
        table="student",
        domain=HITLDomain.IDENTITY_TERM,
        hitl_question="q",
        hitl_context=None,
        options=[
            HITLOption(
                option_id="ok",
                label="Ok",
                description="d",
                resolution=TermResolution(exclude_tokens=[]).model_dump(mode="json"),
                reentry=ReentryDepth.TERMINAL,
            ),
            HITLOption(
                option_id="custom",
                label="Custom",
                description="c",
                resolution=None,
                reentry=ReentryDepth.GENERATE_HOOK,
            ),
        ],
        target=HITLTarget(
            institution_id="u",
            table="student",
            config="term_config",
            field="hook_spec",
        ),
        choice=2,
    )


def test_validate_term_hook_hitl_skips_when_no_hook_required(tmp_path) -> None:
    tc = TermContract(
        institution_id="u",
        table="student",
        term_config=TermOrderConfig(
            term_col="T",
            season_map=[{"raw": "Fall", "canonical": "FALL"}],
            term_extraction="standard",
        ),
        confidence=0.9,
        hitl_flag=False,
        reasoning="r",
    )
    p = tmp_path / "hitl.json"
    p.write_text(
        json.dumps(
            InstitutionHITLItems(
                institution_id="u",
                domain="term",
                items=[],
            ).model_dump(mode="json")
        )
    )
    validate_term_hook_hitl_covers_hook_required(
        term_hitl_path=p,
        term_contract_by_dataset={"student": tc},
    )


def test_validate_term_hook_hitl_passes_when_streams_match_generate_hook_items(
    tmp_path,
) -> None:
    """One hook_required primary stream and one GENERATE_HOOK item."""
    tc = TermContract(
        institution_id="u",
        table="student",
        term_config=TermOrderConfig(
            term_col="T",
            season_map=[],
            term_extraction="hook_required",
            hook_spec=_hs(),
        ),
        confidence=0.9,
        hitl_flag=True,
        reasoning="r",
    )
    p = tmp_path / "hitl.json"
    p.write_text(
        json.dumps(
            InstitutionHITLItems(
                institution_id="u",
                domain="term",
                items=[_term_item_generate_hook()],
            ).model_dump(mode="json")
        )
    )
    validate_term_hook_hitl_covers_hook_required(
        term_hitl_path=p,
        term_contract_by_dataset={"student": tc},
    )


def test_validate_term_hook_hitl_raises_when_hook_tables_exceed_hitl_items(
    tmp_path,
) -> None:
    """Two hook_required datasets but only one GENERATE_HOOK item (after dedupe)."""
    tc_student = TermContract(
        institution_id="u",
        table="student",
        term_config=TermOrderConfig(
            term_col="TERM_DESC",
            season_map=[],
            term_extraction="hook_required",
            hook_spec=_hs(),
        ),
        confidence=0.75,
        hitl_flag=True,
        reasoning="r",
    )
    tc_course = TermContract(
        institution_id="u",
        table="course",
        term_config=TermOrderConfig(
            term_col="DEG",
            season_map=[],
            term_extraction="hook_required",
            hook_spec=_hs(),
        ),
        confidence=0.75,
        hitl_flag=True,
        reasoning="r",
    )
    p = tmp_path / "hitl.json"
    p.write_text(
        json.dumps(
            InstitutionHITLItems(
                institution_id="u",
                domain="term",
                items=[_term_item_generate_hook("only_one")],
            ).model_dump(mode="json")
        )
    )
    with pytest.raises(HITLValidationError, match="course"):
        validate_term_hook_hitl_covers_hook_required(
            term_hitl_path=p,
            term_contract_by_dataset={"student": tc_student, "course": tc_course},
        )


def test_validate_term_hook_hitl_passes_when_one_generate_hook_covers_group_tables(
    tmp_path,
) -> None:
    """Two hook_required datasets sharing one GENERATE_HOOK item via hook_group_tables."""
    tc_student = TermContract(
        institution_id="u",
        table="student",
        term_config=TermOrderConfig(
            term_col="TERM_DESC",
            season_map=[],
            term_extraction="hook_required",
            hook_spec=_hs(),
        ),
        confidence=0.75,
        hitl_flag=True,
        reasoning="r",
    )
    tc_course = TermContract(
        institution_id="u",
        table="course",
        term_config=TermOrderConfig(
            term_col="DEG",
            season_map=[],
            term_extraction="hook_required",
            hook_spec=_hs(),
        ),
        confidence=0.75,
        hitl_flag=True,
        reasoning="r",
    )
    shared = _term_item_generate_hook("shared_term_hook")
    shared.hook_group_id = "shared_nsc_encoding"
    shared.hook_group_tables = ["student", "course"]
    p = tmp_path / "hitl.json"
    p.write_text(
        json.dumps(
            InstitutionHITLItems(
                institution_id="u",
                domain="term",
                items=[shared],
            ).model_dump(mode="json")
        )
    )
    validate_term_hook_hitl_covers_hook_required(
        term_hitl_path=p,
        term_contract_by_dataset={"student": tc_student, "course": tc_course},
    )


def test_validate_term_hook_hitl_raises_when_no_generate_hook_items(tmp_path) -> None:
    """hook_required in contract but reviewer chose terminal option only."""
    tc = TermContract(
        institution_id="u",
        table="student",
        term_config=TermOrderConfig(
            term_col="T",
            season_map=[],
            term_extraction="hook_required",
            hook_spec=_hs(),
        ),
        confidence=0.9,
        hitl_flag=True,
        reasoning="r",
    )
    terminal_only = HITLItem(
        item_id="t",
        institution_id="u",
        table="student",
        domain=HITLDomain.IDENTITY_TERM,
        hitl_question="q",
        hitl_context=None,
        options=[
            HITLOption(
                option_id="ok",
                label="Ok",
                description="d",
                resolution=TermResolution(exclude_tokens=[]).model_dump(mode="json"),
                reentry=ReentryDepth.TERMINAL,
            ),
            HITLOption(
                option_id="custom",
                label="Custom",
                description="c",
                resolution=None,
                reentry=ReentryDepth.GENERATE_HOOK,
            ),
        ],
        target=HITLTarget(
            institution_id="u",
            table="student",
            config="term_config",
            field="hook_spec",
        ),
        choice=1,
    )
    p = tmp_path / "hitl.json"
    p.write_text(
        json.dumps(
            InstitutionHITLItems(
                institution_id="u",
                domain="term",
                items=[terminal_only],
            ).model_dump(mode="json")
        )
    )
    with pytest.raises(HITLValidationError, match="zero HookSpecs"):
        validate_term_hook_hitl_covers_hook_required(
            term_hitl_path=p,
            term_contract_by_dataset={"student": tc},
        )
