"""Tests for year_semantics HITL detection and validation."""

import pytest

from edvise.genai.mapping.identity_agent.hitl.resolver import (
    HITLValidationError,
    validate_term_year_semantics_resolved,
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    TermContract,
    TermOrderConfig,
)
from edvise.genai.mapping.identity_agent.term_normalization.year_semantics_hitl import (
    term_config_needs_year_semantics_review,
)


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
