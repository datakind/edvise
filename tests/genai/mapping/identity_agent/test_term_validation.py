"""Emit-time validation for term hook_group_tables vs term_config column shapes."""

import json

import pytest
from pydantic import ValidationError

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
    season_map_chronology_error,
)
from edvise.genai.mapping.identity_agent.term_normalization.validation import (
    assert_term_hook_groups_compatible,
    build_parse_institution_term_contracts_with_semantic_checks,
    collect_term_semantic_validation_errors,
    raise_term_semantic_validation_error_if_any,
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
            {"raw": "20", "canonical": "SPRING"},
            {"raw": "10", "canonical": "FALL"},
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


def _st_thomas_bad_course_contract() -> TermContract:
    """Model drift case: season-only term_col + hooks that cannot read startdate."""
    return TermContract(
        institution_id="st_thomas_uni",
        table="course",
        term_config=TermOrderConfig(
            term_col="semester",
            season_map=[
                {"raw": "spring", "canonical": "SPRING"},
                {"raw": "summer", "canonical": "SUMMER"},
                {"raw": "fall", "canonical": "FALL"},
            ],
            term_extraction="hook_required",
            hook_spec=HookSpec(
                file="identity_hooks/st_thomas_uni/term_hooks.py",
                functions=[
                    HookFunctionSpec(
                        name="year_extractor_course_semester_year_extraction",
                        signature=(
                            "def year_extractor_course_semester_year_extraction(term: str) -> int"
                        ),
                        description="Extract year from startdate",
                        draft="int(pd.to_datetime(term).year)",
                    ),
                    HookFunctionSpec(
                        name="season_extractor_course_semester_year_extraction",
                        signature=(
                            "def season_extractor_course_semester_year_extraction(term: str) -> str"
                        ),
                        description="Season token from semester",
                        draft="term.strip().lower()",
                    ),
                ],
            ),
        ),
        confidence=0.75,
        hitl_flag=True,
        reasoning="semester has season only; year from startdate",
    )


def test_collect_term_semantic_validation_errors_st_thomas_course_drift():
    inst = InstitutionTermContract(
        institution_id="st_thomas_uni",
        datasets={"course": _st_thomas_bad_course_contract()},
    )
    errors = collect_term_semantic_validation_errors(inst)
    assert len(errors) == 1
    assert "term_col='semester'" in errors[0]
    assert "year_col" in errors[0]
    assert "hook_required" in errors[0] or "standard" in errors[0]


def test_raise_term_semantic_validation_error_if_any_raises_validation_error():
    with pytest.raises(ValidationError, match="TermConfigSemanticValidation"):
        raise_term_semantic_validation_error_if_any(
            ["dataset course: use split year_col/season_col"]
        )


def test_build_parse_rejects_st_thomas_bad_course_with_profile():
    from edvise.genai.mapping.identity_agent.profiling.schemas import (
        RawColumnProfile,
        RawTableProfile,
    )

    inst = InstitutionTermContract(
        institution_id="st_thomas_uni",
        datasets={"course": _st_thomas_bad_course_contract()},
    )
    run_by_dataset = {
        "course": {
            "raw_table_profile": RawTableProfile(
                institution_id="st_thomas_uni",
                dataset="course",
                row_count=100,
                column_count=2,
                columns=[
                    RawColumnProfile(
                        name="semester",
                        dtype="string",
                        null_rate=0.0,
                        null_rate_including_tokens=0.0,
                        unique_count=3,
                        unique_values=["Fall", "Spring", "Summer"],
                        sample_values=["Fall", "Spring", "Summer"],
                    ),
                    RawColumnProfile(
                        name="startdate",
                        dtype="datetime64[ns]",
                        null_rate=0.0,
                        null_rate_including_tokens=0.0,
                        unique_count=3,
                        unique_values=None,
                        sample_values=["8/15/19", "1/10/20", "8/20/21"],
                    ),
                ],
            )
        }
    }
    payload = {
        "institution_id": "st_thomas_uni",
        "datasets": {
            "course": {
                **inst.datasets["course"].model_dump(mode="json"),
                "hitl_flag": False,
            },
        },
        "hitl_items": [],
    }
    parse_fn = build_parse_institution_term_contracts_with_semantic_checks(
        run_by_dataset
    )
    with pytest.raises(ValidationError):
        parse_fn(json.dumps(payload))


def _minimal_term_config(*, season_map: list[dict[str, str]]) -> TermOrderConfig:
    return TermOrderConfig(
        term_col="semester",
        season_map=season_map,
        term_extraction="standard",
    )


def test_season_map_chronology_allows_duplicate_summer():
    cfg = _minimal_term_config(
        season_map=[
            {"raw": "SR", "canonical": "SPRING"},
            {"raw": "UL", "canonical": "SUMMER"},
            {"raw": "UR", "canonical": "SUMMER"},
            {"raw": "FR", "canonical": "FALL"},
        ]
    )
    assert cfg.season_map[2].canonical == "SUMMER"


def test_season_map_chronology_rejects_fall_before_spring():
    with pytest.raises(ValidationError, match="calendar-chronological"):
        _minimal_term_config(
            season_map=[
                {"raw": "10", "canonical": "FALL"},
                {"raw": "20", "canonical": "SPRING"},
            ]
        )


def test_season_map_chronology_error_detects_bad_order():
    err = season_map_chronology_error(
        [
            {"raw": "10", "canonical": "FALL"},
            {"raw": "20", "canonical": "SPRING"},
        ]
    )
    assert err is not None
    assert "FALL (position 1) precedes SPRING (position 2)" in err


def test_collect_term_semantic_validation_errors_passes_valid_season_map():
    inst = InstitutionTermContract(
        institution_id=INST,
        datasets={
            "course": TermContract(
                institution_id=INST,
                table="course",
                term_config=_minimal_term_config(
                    season_map=[
                        {"raw": "20", "canonical": "SPRING"},
                        {"raw": "10", "canonical": "FALL"},
                    ]
                ),
                confidence=0.75,
                hitl_flag=False,
                reasoning="ok",
            )
        },
    )
    assert collect_term_semantic_validation_errors(inst) == []


def test_season_map_chronology_allows_empty_map():
    cfg = TermOrderConfig(
        term_col="semester",
        season_map=[],
        term_extraction="hook_required",
        hook_spec=_hook_spec(),
    )
    assert cfg.season_map == []
