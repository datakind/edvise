"""
Per-dataset prep for grain LLM prompts: load bronze tables, profile keys, build user messages.

Sits in :mod:`grain_inference` because output feeds :func:`build_identity_agent_user_message`
and Pass 1 runners (:mod:`runner`).
"""

from __future__ import annotations

from typing import TypedDict

from edvise.configs.genai import SchoolMappingConfig

from edvise.genai.mapping.identity_agent.dataset_io import load_school_dataset_dataframe

from ..profiling.candidate_keys import profile_candidate_keys
from ..profiling.schemas import RankedCandidateProfiles, RawTableProfile
from .prompt_builder import build_identity_agent_user_message


class IdentityProfilingDatasetResult(TypedDict):
    """One dataset's profiling output for grain prompt prep (Pass 1)."""

    n_rows: int
    n_cols: int
    key_profile: RankedCandidateProfiles
    raw_table_profile: RawTableProfile
    user_message: str


def build_identity_profiling_run_by_dataset(
    *,
    institution_id: str,
    school: SchoolMappingConfig,
) -> dict[str, IdentityProfilingDatasetResult]:
    """
    For each dataset on ``school``, load CSVs, run ``profile_candidate_keys``,
    and build the grain user message via ``build_identity_agent_user_message``.

    Returns a mapping ``dataset_name ->`` :class:`IdentityProfilingDatasetResult` (same keys
    as the ``ia_dev`` notebook ``run_by_dataset``).
    """
    results: dict[str, IdentityProfilingDatasetResult] = {}
    for name in school.datasets.keys():
        df = load_school_dataset_dataframe(school, name)
        kp_result = profile_candidate_keys(
            df, institution_id=institution_id, dataset=name
        )
        key_profile = kp_result.key_profile
        user_msg = build_identity_agent_user_message(
            institution_id,
            name,
            key_profile,
            df=df,
        )
        rtp = kp_result.raw_table_profile
        results[name] = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "key_profile": key_profile,
            "raw_table_profile": rtp,
            "user_message": user_msg,
        }
    return results


__all__ = [
    "IdentityProfilingDatasetResult",
    "build_identity_profiling_run_by_dataset",
]
