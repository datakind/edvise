"""Tests for SMA grain dedup proposal heuristics (``grain_resolution.prompt``)."""

from __future__ import annotations

from edvise.genai.mapping.schema_mapping_agent.grain_resolution.prompt import (
    DedupProposalLLM,
    _measure_variance_suffix_first_signal,
    _normalize_proposals_after_llm,
    _pick_manifest_suffix_key_column,
)
from edvise.genai.mapping.shared.profiling.variance import ColumnVarianceProfile


def test_pick_manifest_suffix_key_prefers_course_like() -> None:
    assert (
        _pick_manifest_suffix_key_column(["student_id", "course_section_id"])
        == "course_section_id"
    )
    assert _pick_manifest_suffix_key_column(["only_key"]) == "only_key"


def test_measure_variance_suffix_first_detects_grade_column() -> None:
    profiles = [
        ColumnVarianceProfile("final_grade", 0.9, []),
        ColumnVarianceProfile("foo", 0.1, []),
    ]
    assert _measure_variance_suffix_first_signal(profiles, ["final_grade", "foo"])


def test_measure_variance_suffix_first_respects_min_pct() -> None:
    profiles = [ColumnVarianceProfile("credits_earned", 0.1, [])]
    assert not _measure_variance_suffix_first_signal(profiles, ["credits_earned"])


def test_normalize_forces_suffix_first_for_measure_policy() -> None:
    props = [
        DedupProposalLLM(
            strategy="true_duplicate",
            label="Drop dupes",
            description="x",
            sort_by=None,
            sort_ascending=None,
            suffix_column=None,
            reasoning="r",
        ),
        DedupProposalLLM(
            strategy="temporal_collapse",
            label="Time",
            description="y",
            sort_by="seq",
            sort_ascending=True,
            suffix_column=None,
            reasoning="r2",
        ),
    ]
    keys = ["student_id", "course_section_id"]
    out = _normalize_proposals_after_llm(
        props,
        manifest_source_keys=keys,
        measure_variance_suffix_first=True,
        suffix_second_required=False,
    )
    assert out[0].strategy == "suffix_identifier"
    assert out[0].suffix_column == "course_section_id"
    assert out[1].strategy == "temporal_collapse"


def test_normalize_suffix_second_uses_manifest_key() -> None:
    props = [
        DedupProposalLLM(
            strategy="true_duplicate",
            label="a",
            description="d",
            sort_by=None,
            sort_ascending=None,
            suffix_column=None,
            reasoning="r",
        ),
        DedupProposalLLM(
            strategy="true_duplicate",
            label="b",
            description="d2",
            sort_by=None,
            sort_ascending=None,
            suffix_column=None,
            reasoning="r2",
        ),
    ]
    keys = ["sid", "course_ref"]
    out = _normalize_proposals_after_llm(
        props,
        manifest_source_keys=keys,
        measure_variance_suffix_first=False,
        suffix_second_required=True,
    )
    assert out[1].strategy == "suffix_identifier"
    assert out[1].suffix_column in keys
