"""Tests for :mod:`edvise.genai.mapping.schema_mapping_agent.transformation.dedupe_plans`."""

import copy
import logging

import pytest

from edvise.genai.mapping.schema_mapping_agent.transformation.dedupe_plans import (
    dedupe_plans_in_section,
    dedupe_transformation_plans_in_wrapper,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
    FieldTransformationPlan,
)

_LOG = logging.getLogger("test_dedupe_plans")


def _step(name: str = "cast_string", col: str = "c") -> dict:
    return {"function_name": name, "column": col, "rationale": "r"}


def test_merges_identical_duplicates_and_joins_distinct_notes() -> None:
    plans = [
        {
            "target_field": "x",
            "output_dtype": "string",
            "steps": [_step()],
            "validation_notes": "one",
            "reviewer_notes": "r1",
        },
        {
            "target_field": "x",
            "output_dtype": "string",
            "steps": [_step()],
            "validation_notes": "two",
            "reviewer_notes": "r1",
        },
    ]
    out, n = dedupe_plans_in_section(copy.deepcopy(plans), entity="course", log=_LOG)
    assert n == 1
    assert len(out) == 1
    p = out[0]
    assert p["output_dtype"] == "string"
    assert p["steps"] == [_step()]
    # First plan's note first, then the distinct second
    assert p["validation_notes"] == "one; two"
    assert p["reviewer_notes"] == "r1"


def test_merge_keeps_hitl_when_first_duplicate_omits_review_required() -> None:
    """Second duplicate row often carries review_required; merge must not drop it."""
    flagged = [
        {
            "step_index": 1,
            "function_name": "map_values",
            "reason": "inferred_season_mapping",
            "context": {},
        }
    ]
    hitl_opts = [
        {
            "option_id": "approve",
            "label": "a",
            "description": "d",
            "resolution": {"approved": True},
        },
        {"option_id": "correct", "label": "c", "description": "d", "resolution": None},
        {
            "option_id": "hook_required",
            "label": "h",
            "description": "d",
            "resolution": {"hook_required": True},
        },
    ]
    steps = [_step()]
    plans = [
        {
            "target_field": "bachelors_degree_conferral_date",
            "output_dtype": "datetime64[ns]",
            "steps": steps,
            "confidence": 0.7,
        },
        {
            "target_field": "bachelors_degree_conferral_date",
            "output_dtype": "datetime64[ns]",
            "steps": steps,
            "confidence": 0.7,
            "review_required": True,
            "flagged_steps": flagged,
            "hitl_options": hitl_opts,
        },
    ]
    out, n = dedupe_plans_in_section(copy.deepcopy(plans), entity="cohort", log=_LOG)
    assert n == 1
    merged = out[0]
    FieldTransformationPlan.model_validate(merged)


def test_raises_on_conflicting_duplicates() -> None:
    plans = [
        {
            "target_field": "x",
            "output_dtype": "string",
            "steps": [_step("cast_string", "a")],
        },
        {
            "target_field": "x",
            "output_dtype": "string",
            "steps": [_step("cast_string", "b")],
        },
    ]
    with pytest.raises(ValueError) as e:
        dedupe_plans_in_section(plans, entity="cohort", log=_LOG)
    assert "conflicting" in str(e.value).lower() or "Duplicate" in str(e.value)


def test_dedupe_wrapper_both_entities() -> None:
    w = {
        "transformation_maps": {
            "cohort": {
                "entity_type": "cohort",
                "target_schema": "S",
                "plans": [
                    {
                        "target_field": "a",
                        "output_dtype": None,
                        "steps": [],
                        "validation_notes": "m1",
                    },
                    {
                        "target_field": "a",
                        "output_dtype": None,
                        "steps": [],
                        "validation_notes": "m2",
                    },
                ],
            },
            "course": {"entity_type": "course", "target_schema": "C", "plans": []},
        }
    }
    n = dedupe_transformation_plans_in_wrapper(w, log=_LOG)
    assert n == 1
    assert len(w["transformation_maps"]["cohort"]["plans"]) == 1
    assert "m1" in w["transformation_maps"]["cohort"]["plans"][0]["validation_notes"]
    assert "m2" in w["transformation_maps"]["cohort"]["plans"][0]["validation_notes"]
