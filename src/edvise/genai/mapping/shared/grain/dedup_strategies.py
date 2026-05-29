"""
Shared grain dedup strategy literals.

IdentityAgent ``GrainContract.dedup_policy.strategy`` and reviewer resolutions share
``DedupPolicyStrategy`` / ``GrainResolutionDedupStrategy``. SMA-only resolution values are
listed separately so IdentityAgent prompts and docs do not treat them as first-class IA options.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

# Persisted on ``GrainContract.dedup_policy.strategy`` (includes ``policy_required`` flag state).
DedupPolicyStrategy = Literal[
    "true_duplicate",
    "temporal_collapse",
    "first_by_column",
    "categorical_priority",
    "suffix_identifier",
    "no_dedup",
    "policy_required",
]

# Reviewer resolutions for Identity grain HITL (and SMA strategies that also apply to IA contracts).
# Excludes ``policy_required`` (flag state) and ``intentional_step_down`` (SMA grain gate only).
GrainResolutionDedupStrategy = Literal[
    "true_duplicate",
    "temporal_collapse",
    "first_by_column",
    "categorical_priority",
    "suffix_identifier",
    "no_dedup",
]

# SMA grain step-down option only (``domain='sma_grain'``); never written to IA ``dedup_policy``.
SmaOnlyGrainResolutionDedupStrategy: TypeAlias = Literal["intentional_step_down"]

# ``GrainResolution.dedup_strategy`` on disk may be IA strategies or SMA-only ``intentional_step_down``.
GrainResolutionDedupStrategyAny: TypeAlias = (
    GrainResolutionDedupStrategy | SmaOnlyGrainResolutionDedupStrategy
)

# SMA within-grain multiplicity LLM proposals (subset of ``DedupPolicyStrategy``).
SmaGrainMultiplicityProposalStrategy = Literal[
    "true_duplicate",
    "temporal_collapse",
    "first_by_column",
    "suffix_identifier",
]

# Back-compat: historical export name in ``identity_agent.grain_inference``.
DedupStrategy = DedupPolicyStrategy

__all__ = [
    "DedupPolicyStrategy",
    "DedupStrategy",
    "GrainResolutionDedupStrategy",
    "GrainResolutionDedupStrategyAny",
    "SmaGrainMultiplicityProposalStrategy",
    "SmaOnlyGrainResolutionDedupStrategy",
]
