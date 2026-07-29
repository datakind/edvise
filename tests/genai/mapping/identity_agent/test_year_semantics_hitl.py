"""Tests for year_semantics HITL detection and validation."""

import json

import pytest
from pydantic import ValidationError

from edvise.genai.mapping.identity_agent.hitl.artifacts import (
    write_identity_term_artifacts,
)
from edvise.genai.mapping.identity_agent.hitl.resolver import (
    HITLValidationError,
    validate_term_year_semantics_resolved,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import (
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLTarget,
    ReentryDepth,
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
    build_parse_institution_term_contracts_with_semantic_checks,
)
from edvise.genai.mapping.identity_agent.term_normalization.year_semantics_hitl import (
    assert_term_year_semantics_hitl_coverage,
    collect_term_year_semantics_hitl_coverage_errors,
    term_config_needs_year_semantics_review,
)

INST = "test_university"


def _yyyypp_term_config(*, year_semantics: str | None = None) -> dict:
    return {
        "term_col": "academic_period",
        "year_col": None,
        "season_col": None,
        "season_map": [
            {"raw": "15", "canonical": "SPRING"},
            {"raw": "20", "canonical": "SUMMER"},
            {"raw": "30", "canonical": "FALL"},
        ],
        "term_extraction": "hook_required",
        "year_semantics": year_semantics,
        "hook_spec": {
            "file": "identity_hooks/u/term_hooks.py",
            "functions": [
                {
                    "name": "year_extractor_shared",
                    "description": "year",
                    "draft": "int(str(term)[:4])",
                },
                {
                    "name": "season_extractor_shared",
                    "description": "season",
                    "draft": "str(term)[4:6]",
                },
            ],
        },
    }


def test_yyyypp_needs_year_semantics_when_unset():
    assert term_config_needs_year_semantics_review(_yyyypp_term_config()) is True


def test_yyyypp_satisfied_when_year_semantics_set():
    cfg = _yyyypp_term_config(year_semantics="academic_year_prefix")
    assert term_config_needs_year_semantics_review(cfg) is False


def test_datetime_hook_does_not_need_year_semantics():
    cfg = {
        "term_col": "first_enr",
        "season_map": [{"raw": "june", "canonical": "SUMMER"}],
        "term_extraction": "hook_required",
        "year_semantics": None,
        "hook_spec": {
            "file": "identity_hooks/u/term_hooks.py",
            "functions": [
                {
                    "name": "year_extractor_student",
                    "draft": "pd.to_datetime(term).year",
                },
                {
                    "name": "season_extractor_student",
                    "draft": "pd.to_datetime(term).strftime('%B').lower()",
                },
            ],
        },
    }
    assert term_config_needs_year_semantics_review(cfg) is False


def test_yyyymm_month_codes_do_not_need_year_semantics():
    cfg = {
        "term_col": "term_code",
        "season_map": [
            {"raw": "01", "canonical": "SPRING"},
            {"raw": "08", "canonical": "FALL"},
        ],
        "term_extraction": "hook_required",
        "year_semantics": None,
        "hook_spec": {
            "file": "identity_hooks/u/term_hooks.py",
            "functions": [
                {"name": "year_extractor", "draft": "int(str(term)[:4])"},
                {"name": "season_extractor", "draft": "str(term)[4:6]"},
            ],
        },
    }
    assert term_config_needs_year_semantics_review(cfg) is False


def test_split_year_period_columns_need_year_semantics():
    cfg = {
        "term_col": None,
        "year_col": "yr",
        "season_col": "term",
        "season_map": [
            {"raw": "10", "canonical": "FALL"},
            {"raw": "20", "canonical": "SPRING"},
        ],
        "term_extraction": "standard",
        "year_semantics": None,
        "hook_spec": None,
    }
    assert term_config_needs_year_semantics_review(cfg) is True


def test_validate_term_year_semantics_raises_for_unresolved_yyyypp():
    contracts = {
        "course": TermContract(
            institution_id="u",
            table="course",
            term_config=TermOrderConfig.model_validate(_yyyypp_term_config()),
            confidence=0.8,
            hitl_flag=True,
            reasoning="r",
        ),
        "student": TermContract(
            institution_id="u",
            table="student",
            term_config=TermOrderConfig.model_validate(
                {
                    "term_col": "first_enr",
                    "season_map": [],
                    "term_extraction": "hook_required",
                    "hook_spec": {
                        "file": "identity_hooks/u/term_hooks.py",
                        "functions": [
                            {
                                "name": "year_extractor_student",
                                "description": "year",
                                "draft": "pd.to_datetime(term).year",
                            },
                            {
                                "name": "season_extractor_student",
                                "description": "season",
                                "draft": "pd.to_datetime(term).strftime('%B').lower()",
                            },
                        ],
                    },
                }
            ),
            confidence=0.8,
            hitl_flag=True,
            reasoning="r",
        ),
    }
    with pytest.raises(HITLValidationError, match="year_semantics"):
        validate_term_year_semantics_resolved(term_contract_by_dataset=contracts)


def test_validate_term_year_semantics_passes_when_resolved():
    contracts = {
        "course": TermContract(
            institution_id="u",
            table="course",
            term_config=TermOrderConfig.model_validate(
                _yyyypp_term_config(year_semantics="calendar_literal")
            ),
            confidence=0.8,
            hitl_flag=True,
            reasoning="r",
        ),
    }
    validate_term_year_semantics_resolved(term_contract_by_dataset=contracts)


def _split_student_contract() -> TermContract:
    return TermContract(
        institution_id=INST,
        table="student",
        term_config=TermOrderConfig(
            year_col="year_code",
            season_col="term_code",
            season_map=[
                {"raw": "20", "canonical": "SPRING"},
                {"raw": "10", "canonical": "FALL"},
            ],
            term_extraction="standard",
            year_semantics=None,
        ),
        confidence=0.75,
        hitl_flag=True,
        reasoning="split year_code + term_code period codes",
    )


def _season_map_only_hitl() -> HITLItem:
    return HITLItem(
        item_id="student_term_code_season_mapping",
        institution_id=INST,
        table="student",
        domain=HITLDomain.IDENTITY_TERM,
        hitl_question="Confirm season mapping for term_code 10/20.",
        hitl_context="year_code is a separate column.",
        options=[
            HITLOption(
                option_id="fall_spring_mapping",
                label="10=Fall, 20=Spring",
                description="confirm",
                resolution={
                    "season_map_replace": [
                        {"raw": "20", "canonical": "SPRING"},
                        {"raw": "10", "canonical": "FALL"},
                    ]
                },
                reentry=ReentryDepth.TERMINAL,
            ),
            HITLOption(
                option_id="custom",
                label="Custom",
                description="custom",
                resolution=None,
                reentry=ReentryDepth.TERMINAL,
            ),
        ],
        target=HITLTarget(
            institution_id=INST,
            table="student",
            config="term_config",
            field="season_map",
        ),
    )


def _year_semantics_hitl(*, table: str = "student") -> HITLItem:
    return HITLItem(
        item_id=f"{table}_year_semantics",
        institution_id=INST,
        table=table,
        domain=HITLDomain.IDENTITY_TERM,
        hitl_question="Confirm year_semantics for coded year prefix.",
        hitl_context="Independent of season mapping.",
        options=[
            HITLOption(
                option_id="calendar_literal",
                label="Calendar year",
                description="calendar",
                resolution={"year_semantics": "calendar_literal"},
                reentry=ReentryDepth.TERMINAL,
            ),
            HITLOption(
                option_id="academic_year_prefix",
                label="Academic-year start",
                description="academic",
                resolution={"year_semantics": "academic_year_prefix"},
                reentry=ReentryDepth.TERMINAL,
            ),
            HITLOption(
                option_id="custom",
                label="Custom",
                description="custom",
                resolution=None,
                reentry=ReentryDepth.TERMINAL,
            ),
        ],
        target=HITLTarget(
            institution_id=INST,
            table=table,
            config="term_config",
            field="year_semantics",
        ),
    )


def test_collect_year_semantics_hitl_coverage_errors_for_season_only_item():
    inst = InstitutionTermContract(
        institution_id=INST,
        datasets={"student": _split_student_contract()},
    )
    errors = collect_term_year_semantics_hitl_coverage_errors(
        inst, [_season_map_only_hitl()]
    )
    assert len(errors) == 1
    assert "student" in errors[0]
    assert "year_semantics" in errors[0]


def test_collect_year_semantics_hitl_coverage_passes_when_item_present():
    inst = InstitutionTermContract(
        institution_id=INST,
        datasets={"student": _split_student_contract()},
    )
    errors = collect_term_year_semantics_hitl_coverage_errors(
        inst, [_season_map_only_hitl(), _year_semantics_hitl()]
    )
    assert errors == []


def test_assert_term_year_semantics_hitl_coverage_raises():
    inst = InstitutionTermContract(
        institution_id=INST,
        datasets={"student": _split_student_contract()},
    )
    with pytest.raises(ValueError, match="year_semantics"):
        assert_term_year_semantics_hitl_coverage(inst, [_season_map_only_hitl()])


def test_parse_rejects_missing_year_semantics_hitl_with_validation_error():
    payload = {
        "institution_id": INST,
        "datasets": {"student": _split_student_contract().model_dump(mode="json")},
        "hitl_items": [_season_map_only_hitl().model_dump(mode="json")],
    }
    with pytest.raises(ValidationError, match="year_semantics"):
        parse_institution_term_contracts_with_hitl(json.dumps(payload))


def test_build_parse_rejects_missing_year_semantics_hitl():
    payload = {
        "institution_id": INST,
        "datasets": {"student": _split_student_contract().model_dump(mode="json")},
        "hitl_items": [_season_map_only_hitl().model_dump(mode="json")],
    }
    parse_fn = build_parse_institution_term_contracts_with_semantic_checks()
    with pytest.raises(ValidationError, match="year_semantics"):
        parse_fn(json.dumps(payload))


def test_parse_accepts_split_student_with_year_semantics_hitl():
    payload = {
        "institution_id": INST,
        "datasets": {"student": _split_student_contract().model_dump(mode="json")},
        "hitl_items": [
            _season_map_only_hitl().model_dump(mode="json"),
            _year_semantics_hitl().model_dump(mode="json"),
        ],
    }
    inst, items = parse_institution_term_contracts_with_hitl(json.dumps(payload))
    assert "student" in inst.datasets
    assert len(items) == 2


def test_write_identity_term_artifacts_rejects_missing_year_semantics_hitl(tmp_path):
    with pytest.raises(ValueError, match="year_semantics"):
        write_identity_term_artifacts(
            tmp_path,
            INST,
            {"student": _split_student_contract()},
            [_season_map_only_hitl()],
        )
