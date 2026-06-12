"""Tests for :mod:`edvise.genai.mapping.shared.grain.dedup_execution`."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from edvise.genai.mapping.identity_agent.hitl.resolver import (
    HITLValidationError,
    resolve_items,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import (
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLTarget,
    InstitutionHITLItems,
    ReentryDepth,
)
from edvise.genai.mapping.schema_mapping_agent.grain_resolution.hitl import (
    build_sma_grain_hitl_items,
)
from edvise.genai.mapping.schema_mapping_agent.grain_resolution.prompt import (
    DedupProposalLLM,
)
from edvise.genai.mapping.shared.grain.dedup_execution import (
    apply_sma_grain_resolution_payload,
    assert_suffix_column_in_entity_keys,
)


def test_true_duplicate_dedupes_on_entity_keys_not_full_row() -> None:
    df = pd.DataFrame(
        {
            "sid": [1, 1],
            "term": ["A", "A"],
            "extra": [10, 20],
        }
    )
    keys = ["sid", "term"]
    payload = {"grain_resolution": {"dedup_strategy": "true_duplicate"}}
    out = apply_sma_grain_resolution_payload(
        df, keys, payload, log=logging.getLogger("test")
    )
    assert len(out) == 1
    assert list(out.columns) == list(df.columns)


def test_temporal_collapse_sort_then_keep_first() -> None:
    df = pd.DataFrame(
        {
            "sid": [1, 1, 1],
            "term": ["A", "A", "A"],
            "seq": [3, 1, 2],
        }
    )
    keys = ["sid", "term"]
    payload = {
        "grain_resolution": {
            "dedup_strategy": "temporal_collapse",
            "dedup_sort_by": "seq",
            "dedup_sort_ascending": True,
            "dedup_keep": "first",
        }
    }
    out = apply_sma_grain_resolution_payload(df, keys, payload)
    assert len(out) == 1
    assert int(out["seq"].iloc[0]) == 1


def test_categorical_priority_keeps_best_row() -> None:
    df = pd.DataFrame(
        {
            "sid": [1, 1],
            "term": ["A", "A"],
            "honors": ["Cum Laude", "Summa Cum Laude"],
        }
    )
    keys = ["sid", "term"]
    payload = {
        "grain_resolution": {
            "dedup_strategy": "categorical_priority",
            "priority_column": "honors",
            "priority_order": ["Summa Cum Laude", "Magna Cum Laude", "Cum Laude"],
        }
    }
    out = apply_sma_grain_resolution_payload(df, keys, payload)
    assert len(out) == 1
    assert "Summa" in str(out["honors"].iloc[0])


def test_suffix_identifier_requires_suffix_in_entity_keys() -> None:
    df = pd.DataFrame({"sid": [1, 1], "course": ["X", "X"], "term": ["A", "A"]})
    keys = ["sid", "term"]
    payload = {
        "grain_resolution": {
            "dedup_strategy": "suffix_identifier",
            "suffix_column": "course",
        }
    }
    out = apply_sma_grain_resolution_payload(df, keys, payload)
    # course not in entity_keys — skip mutation
    assert len(out) == 2


def test_suffix_identifier_applies_when_suffix_in_keys() -> None:
    df = pd.DataFrame({"sid": [1, 1], "course": ["X", "X"], "term": ["A", "A"]})
    keys = ["sid", "course", "term"]
    payload = {
        "grain_resolution": {
            "dedup_strategy": "suffix_identifier",
            "suffix_column": "course",
        }
    }
    out = apply_sma_grain_resolution_payload(df, keys, payload)
    assert len(out) == 2
    vals = set(out["course"].astype(str))
    assert any("-" in v for v in vals)


def test_no_dedup_returns_copy() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    out = apply_sma_grain_resolution_payload(
        df, ["a"], {"grain_resolution": {"dedup_strategy": "no_dedup"}}
    )
    assert len(out) == 2
    assert out is not df


def test_assert_suffix_column_in_entity_keys_validates() -> None:
    assert assert_suffix_column_in_entity_keys("a", ["a", "b"]) == "a"
    with pytest.raises(ValueError, match="suffix_column"):
        assert_suffix_column_in_entity_keys("c", ["a", "b"])


def test_resolve_sma_grain_suffix_identifier_writes_sidecar_not_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest_map.json"
    before = json.dumps(
        {
            "manifests": {
                "course": {
                    "mappings": [
                        {"target_field": "source_term_key", "source_column": "old_col"}
                    ]
                }
            }
        }
    )
    manifest.write_text(before)
    hitl_path = tmp_path / "course_sma_grain_hitl.json"
    envelope = InstitutionHITLItems(
        institution_id="u",
        domain="sma_grain",
        items=[
            HITLItem(
                item_id="i1",
                institution_id="u",
                table="course",
                domain=HITLDomain.SMA_GRAIN,
                hitl_question="q",
                options=[
                    HITLOption(
                        option_id="suffix",
                        label="Suffix",
                        description="d",
                        resolution={
                            "dedup_strategy": "suffix_identifier",
                            "suffix_column": "course_prefix",
                        },
                        reentry=ReentryDepth.TERMINAL,
                    ),
                    HITLOption(
                        option_id="custom",
                        label="Custom",
                        description="c",
                        resolution=None,
                        reentry=ReentryDepth.TERMINAL,
                    ),
                ],
                target=HITLTarget(
                    institution_id="u", table="course", config="x", field="y"
                ),
                choice=1,
                metadata={
                    "manifest_source_keys": [
                        "learner_id",
                        "course_prefix",
                        "course_number",
                    ],
                    "dataset": "course",
                    "entity_type": "course",
                },
            )
        ],
    )
    hitl_path.write_text(envelope.model_dump_json(indent=2))
    resolve_items(hitl_path)
    sidecar = tmp_path / "sma_grain_resolution_course.json"
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text())
    assert payload["grain_resolutions"][0]["dedup_strategy"] == "suffix_identifier"
    assert payload["grain_resolutions"][0]["suffix_column"] == "course_prefix"
    assert manifest.read_text() == before


def test_resolve_sma_grain_suffix_rejects_column_not_in_grain(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest_map.json"
    manifest.write_text("{}\n")
    hitl_path = tmp_path / "course_sma_grain_hitl.json"
    envelope = InstitutionHITLItems(
        institution_id="u",
        domain="sma_grain",
        items=[
            HITLItem(
                item_id="i1",
                institution_id="u",
                table="course",
                domain=HITLDomain.SMA_GRAIN,
                hitl_question="q",
                options=[
                    HITLOption(
                        option_id="suffix",
                        label="Suffix",
                        description="d",
                        resolution={
                            "dedup_strategy": "suffix_identifier",
                            "suffix_column": "stc_title",
                        },
                        reentry=ReentryDepth.TERMINAL,
                    ),
                    HITLOption(
                        option_id="custom",
                        label="Custom",
                        description="c",
                        resolution=None,
                        reentry=ReentryDepth.TERMINAL,
                    ),
                ],
                target=HITLTarget(
                    institution_id="u", table="course", config="x", field="y"
                ),
                choice=1,
                metadata={
                    "manifest_source_keys": ["learner_id", "course_prefix"],
                    "dataset": "course",
                    "entity_type": "course",
                },
            )
        ],
    )
    hitl_path.write_text(envelope.model_dump_json(indent=2))
    with pytest.raises(HITLValidationError, match="suffix_column"):
        resolve_items(hitl_path)


def test_ia_post_clean_primary_key_prefers_dataset_unique_keys() -> None:
    from edvise.genai.mapping.schema_mapping_agent.grain_resolution.runner import (
        ia_post_clean_primary_key_for_dataset,
    )

    contract = {
        "datasets": {
            "student": {
                "unique_keys": ["learner_id", "term_desc"],
                "grain_contract": {"post_clean_primary_key": ["learner_id"]},
            }
        }
    }
    assert ia_post_clean_primary_key_for_dataset(contract, "student") == [
        "learner_id",
        "term_desc",
    ]


def test_ia_post_clean_primary_key_falls_back_to_grain_contract() -> None:
    from edvise.genai.mapping.schema_mapping_agent.grain_resolution.runner import (
        ia_post_clean_primary_key_for_dataset,
    )

    contract = {
        "datasets": {
            "student": {
                "grain_contract": {"post_clean_primary_key": ["pidm", "term"]},
            }
        }
    }
    assert ia_post_clean_primary_key_for_dataset(contract, "student") == [
        "pidm",
        "term",
    ]


def test_build_sma_dedup_proposals_without_llm_measure_columns_suffix_first() -> None:
    from edvise.genai.mapping.schema_mapping_agent.grain_resolution.prompt import (
        build_sma_dedup_proposals_without_llm,
    )
    from edvise.genai.mapping.shared.profiling.variance import (
        ColumnVarianceProfile,
        WithinGroupVarianceResult,
    )

    variance = WithinGroupVarianceResult(
        non_unique_rows=20,
        affected_groups=5,
        group_size_distribution={4: 5},
        column_profiles=[
            ColumnVarianceProfile(
                column="term_gpa",
                pct_groups_with_variance=0.6,
                sample_values=[3.1, 2.0],
            ),
        ],
        sampled=False,
    )
    props = build_sma_dedup_proposals_without_llm(
        manifest_source_keys=["learner_id", "course_prefix"],
        variance=variance,
        mapped_source_columns=["term_gpa"],
    )
    assert len(props) == 2
    assert props[0].strategy == "suffix_identifier"
    assert props[0].suffix_column in ("course_prefix", "learner_id")
    assert props[1].strategy == "true_duplicate"


def test_build_sma_grain_hitl_rejects_suffix_not_in_manifest_grain() -> None:
    proposals = [
        DedupProposalLLM(
            strategy="true_duplicate",
            label="x",
            description="y",
            reasoning="r",
        ),
        DedupProposalLLM(
            strategy="suffix_identifier",
            label="s",
            description="d",
            suffix_column="stc_title",
            reasoning="r",
        ),
    ]
    with pytest.raises(ValueError, match="suffix_column"):
        build_sma_grain_hitl_items(
            institution_id="i",
            dataset="course",
            entity_type="course",
            scenario="within_grain_multiplicity",
            base_rows=10,
            entity_rows=5,
            manifest_source_keys=["a", "b"],
            mapped_source_columns=["c"],
            ia_source_keys=None,
            proposals=proposals,
            sma_manifest_path=None,
            variance=None,
        )


def test_build_sma_grain_hitl_items_serializes_timestamp_variance_context() -> None:
    from edvise.genai.mapping.shared.profiling.variance import (
        ColumnVarianceProfile,
        WithinGroupVarianceResult,
    )

    variance = WithinGroupVarianceResult(
        non_unique_rows=4,
        affected_groups=2,
        group_size_distribution={2: 2},
        column_profiles=[
            ColumnVarianceProfile(
                column="course_begin_date",
                pct_groups_with_variance=1.0,
                sample_values=["2024-08-19T00:00:00", "2024-08-20T00:00:00"],
            ),
        ],
        sampled=False,
    )
    proposals = [
        DedupProposalLLM(
            strategy="true_duplicate",
            label="Drop identical duplicates",
            description="Keep one row per manifest key.",
            reasoning="Fallback.",
        ),
        DedupProposalLLM(
            strategy="suffix_identifier",
            label="Suffix key ties",
            description="Append -1, -2 suffixes.",
            suffix_column="course_section",
            reasoning="Fallback.",
        ),
    ]
    items = build_sma_grain_hitl_items(
        institution_id="indiana_institute_of_technology",
        dataset="course",
        entity_type="course",
        scenario="within_grain_multiplicity",
        base_rows=72669,
        entity_rows=65176,
        manifest_source_keys=["learner_id", "course_section"],
        mapped_source_columns=["course_begin_date"],
        ia_source_keys=["learner_id", "course_name", "semester"],
        proposals=proposals,
        sma_manifest_path=None,
        variance=variance,
    )
    assert items[0].hitl_context is not None
    ctx = json.loads(items[0].hitl_context)
    assert ctx["top_column_profiles"][0]["sample_values"] == [
        "2024-08-19T00:00:00",
        "2024-08-20T00:00:00",
    ]


def test_write_sma_grain_true_duplicate_resolution_file_shape(tmp_path: Path) -> None:
    from edvise.genai.mapping.schema_mapping_agent.grain_resolution.runner import (
        _write_sma_grain_true_duplicate_resolution_file,
    )

    hitl = tmp_path / "course_sma_grain_hitl.json"
    hitl.write_text("{}")
    out = _write_sma_grain_true_duplicate_resolution_file(
        hitl,
        institution_id="inst",
        dataset="course",
        entity_type="course",
        manifest_source_keys=["learner_id", "course_prefix"],
    )
    assert out.name == "sma_grain_resolution_course.json"
    data = json.loads(out.read_text())
    assert data["institution_id"] == "inst"
    assert data["manifest_source_keys"] == ["learner_id", "course_prefix"]
    assert data["grain_resolutions"][0]["dedup_strategy"] == "true_duplicate"


def test_grain_resolutions_applies_steps_in_order() -> None:
    df = pd.DataFrame({"sid": [1, 1, 1], "term": ["A", "A", "A"], "seq": [3, 1, 2]})
    keys = ["sid", "term"]
    payload = {
        "grain_resolutions": [
            {"dedup_strategy": "no_dedup"},
            {
                "dedup_strategy": "first_by_column",
                "dedup_sort_by": "seq",
                "dedup_sort_ascending": True,
                "dedup_keep": "first",
            },
        ],
    }
    out = apply_sma_grain_resolution_payload(df, keys, payload)
    assert len(out) == 1
    assert int(out["seq"].iloc[0]) == 1


def test_append_grain_resolution_file_chains_steps(tmp_path: Path) -> None:
    from edvise.genai.mapping.schema_mapping_agent.grain_resolution.runner import (
        append_sma_grain_resolution_step,
    )

    out = tmp_path / "sma_grain_resolution_course.json"
    append_sma_grain_resolution_step(
        out,
        institution_id="i",
        dataset="course",
        entity_type="course",
        manifest_source_keys=["sid"],
        grain_resolution={
            "dedup_strategy": "suffix_identifier",
            "suffix_column": "sid",
        },
    )
    append_sma_grain_resolution_step(
        out,
        institution_id="i",
        dataset="course",
        entity_type="course",
        manifest_source_keys=["sid"],
        grain_resolution={"dedup_strategy": "true_duplicate"},
    )
    data = json.loads(out.read_text())
    assert len(data["grain_resolutions"]) == 2
    assert data["grain_resolutions"][0]["dedup_strategy"] == "suffix_identifier"
    assert data["grain_resolutions"][1]["dedup_strategy"] == "true_duplicate"


def test_append_migrates_legacy_single_grain_resolution(tmp_path: Path) -> None:
    from edvise.genai.mapping.schema_mapping_agent.grain_resolution.runner import (
        append_sma_grain_resolution_step,
    )

    out = tmp_path / "sma_grain_resolution_course.json"
    out.write_text(
        json.dumps(
            {
                "institution_id": "i",
                "dataset": "course",
                "entity_type": "course",
                "manifest_source_keys": ["sid"],
                "grain_resolution": {"dedup_strategy": "no_dedup"},
            }
        )
    )
    append_sma_grain_resolution_step(
        out,
        institution_id="i",
        dataset="course",
        entity_type="course",
        manifest_source_keys=["sid"],
        grain_resolution={"dedup_strategy": "true_duplicate"},
    )
    data = json.loads(out.read_text())
    assert len(data["grain_resolutions"]) == 2
    assert data["grain_resolutions"][0]["dedup_strategy"] == "no_dedup"
    assert data["grain_resolutions"][1]["dedup_strategy"] == "true_duplicate"
