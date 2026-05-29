"""
Construct SMA grain ``InstitutionHITLItems`` for step-down vs within-grain multiplicity.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Literal

from edvise.genai.mapping.identity_agent.hitl.schemas import (
    GrainResolution,
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLTarget,
    ReentryDepth,
)
from edvise.genai.mapping.schema_mapping_agent.grain_resolution.prompt import (
    DedupProposalLLM,
    proposal_to_grain_resolution,
)
from edvise.genai.mapping.shared.grain.dedup_execution import (
    assert_suffix_column_in_entity_keys,
)
from edvise.genai.mapping.shared.profiling.variance import WithinGroupVarianceResult


def build_sma_grain_hitl_items(
    *,
    institution_id: str,
    dataset: str,
    entity_type: Literal["cohort", "course"],
    scenario: Literal["step_down", "within_grain_multiplicity"],
    base_rows: int,
    entity_rows: int,
    manifest_source_keys: list[str],
    mapped_source_columns: list[str],
    ia_source_keys: list[str] | None,
    proposals: list[DedupProposalLLM] | None,
    sma_manifest_path: Path | None,
    variance: WithinGroupVarianceResult | None = None,
    aligned_ia_manifest_entity_keys: bool = False,
) -> list[HITLItem]:
    """Construct HITL items for ``sma_grain_hitl.json``."""
    delta = base_rows - entity_rows
    meta_base: dict[str, Any] = {
        "base_rows": base_rows,
        "entity_rows": entity_rows,
        "delta_rows": delta,
        "scenario": scenario,
        "dataset": dataset,
        "ia_source_keys": list(ia_source_keys or []),
        "manifest_source_keys": list(manifest_source_keys),
        "entity_type": entity_type,
    }
    if sma_manifest_path is not None:
        meta_base["sma_manifest_path"] = str(sma_manifest_path)
    if aligned_ia_manifest_entity_keys:
        meta_base["aligned_ia_manifest_entity_keys"] = True

    target = HITLTarget(
        institution_id=institution_id,
        table=dataset,
        config="sma_execution_grain",
        field="base_df_reduction",
    )

    if scenario == "step_down":
        q = (
            f"Dataset '{dataset}': manifest entity grain ({manifest_source_keys}) is coarser than "
            f"IdentityAgent post-clean keys ({ia_source_keys or []}). "
            f"Multiple base rows ({base_rows}) map to {entity_rows} entities — confirm this "
            "intentional step-down (sanctioned collapse) or flag as unexpected."
        )
        return [
            HITLItem(
                item_id=f"{institution_id}_sma_grain_step_down_{dataset}_{uuid.uuid4().hex[:8]}",
                institution_id=institution_id,
                table=dataset,
                domain=HITLDomain.SMA_GRAIN,
                hitl_question=q,
                hitl_context=None,
                options=[
                    HITLOption(
                        option_id="confirm_step_down",
                        label="Confirm intentional step-down",
                        description=(
                            "Collapse finer source rows to manifest entity grain; "
                            "no manifest change."
                        ),
                        resolution=GrainResolution(
                            dedup_strategy="intentional_step_down"
                        ).model_dump(mode="json"),
                        reentry=ReentryDepth.TERMINAL,
                    ),
                    HITLOption(
                        option_id="custom",
                        label="Unexpected — needs follow-up",
                        description="Escalate: this collapse was not expected for this dataset.",
                        resolution=None,
                        reentry=ReentryDepth.TERMINAL,
                    ),
                ],
                target=target,
                choice=None,
                severity="warning",
                metadata=meta_base,
            )
        ]

    ctx_json: str | None = None
    if variance is not None:
        ctx_json = json.dumps(
            {
                "top_column_profiles": [
                    {
                        "column": p.column,
                        "pct_groups_with_variance": p.pct_groups_with_variance,
                        "sample_values": p.sample_values,
                    }
                    for p in variance.column_profiles[:12]
                ],
                "group_size_distribution": variance.group_size_distribution,
                "sampled": variance.sampled,
            },
            indent=2,
        )

    assert proposals is not None and len(proposals) >= 2
    if aligned_ia_manifest_entity_keys:
        q = (
            f"Dataset '{dataset}': IdentityAgent and this manifest both use entity keys "
            f"{manifest_source_keys}. The cleaned base still has {base_rows} rows vs "
            f"{entity_rows} unique key groups (within-grain collapse / history, not a finer "
            "IA grain than the manifest). Pick a dedup strategy informed by the variance profile."
        )
    else:
        q = (
            f"Dataset '{dataset}': rows do not reduce cleanly to manifest keys "
            f"{manifest_source_keys} ({base_rows} base rows vs {entity_rows} unique key groups). "
            "Pick a dedup / grain strategy informed by the variance profile."
        )
    opt_models: list[HITLOption] = []
    for i, prop in enumerate(proposals[:2]):
        if prop.strategy == "suffix_identifier":
            assert_suffix_column_in_entity_keys(
                prop.suffix_column, manifest_source_keys
            )
        opt_models.append(
            HITLOption(
                option_id=f"proposal_{i + 1}_{prop.strategy}",
                label=prop.label[:80],
                description=prop.description,
                resolution=proposal_to_grain_resolution(prop).model_dump(mode="json"),
                reentry=ReentryDepth.TERMINAL,
            )
        )
    opt_models.append(
        HITLOption(
            option_id="custom",
            label="Custom handling",
            description="None of the proposals fit; follow up manually.",
            resolution=None,
            reentry=ReentryDepth.TERMINAL,
        )
    )
    return [
        HITLItem(
            item_id=f"{institution_id}_sma_grain_multiplicity_{dataset}_{uuid.uuid4().hex[:8]}",
            institution_id=institution_id,
            table=dataset,
            domain=HITLDomain.SMA_GRAIN,
            hitl_question=q,
            hitl_context=ctx_json,
            options=opt_models,
            target=target,
            choice=None,
            severity="error",
            metadata=meta_base,
        )
    ]
