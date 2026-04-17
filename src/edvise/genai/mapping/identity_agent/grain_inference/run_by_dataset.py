"""
Per-dataset prep for grain LLM prompts: load bronze tables, profile keys, build user messages.

Sits in :mod:`grain_inference` because output feeds :func:`build_identity_agent_user_message`
and Pass 1 runners (:mod:`runner`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from edvise.configs.genai import SchoolMappingConfig

from edvise.genai.mapping.identity_agent.dataset_io import load_school_dataset_dataframe

from ..profiling.candidate_keys import profile_candidate_keys
from ..profiling.schemas import RankedCandidateProfiles, RawTableProfile
from .prompt import build_identity_agent_user_message


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
    cleaning = school.cleaning

    for name in school.datasets.keys():
        df = load_school_dataset_dataframe(school, name)
        kp_result = profile_candidate_keys(
            df,
            institution_id=institution_id,
            dataset=name,
            cleaning=cleaning,
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


def identity_profiling_run_to_jsonable(
    institution_id: str,
    run_by_dataset: Mapping[str, IdentityProfilingDatasetResult],
) -> dict[str, object]:
    """
    Serialize a profiling run to a JSON-compatible dict: per-dataset ``raw_table_profile``,
    ``key_profile`` (candidate keys + variance), row/col counts, and the grain ``user_message``.
    """
    datasets: dict[str, object] = {}
    for name, row in run_by_dataset.items():
        datasets[name] = {
            "n_rows": row["n_rows"],
            "n_cols": row["n_cols"],
            "key_profile": row["key_profile"].model_dump(mode="json"),
            "raw_table_profile": row["raw_table_profile"].model_dump(mode="json"),
            "user_message": row["user_message"],
        }
    return {"institution_id": institution_id, "datasets": datasets}


def write_identity_profiling_artifacts(
    output_dir: str | Path,
    institution_id: str,
    run_by_dataset: Mapping[str, IdentityProfilingDatasetResult],
    *,
    filename: str = "identity_profiling_run.json",
) -> Path:
    """
    Write IdentityAgent profiling output to disk (e.g. Unity Catalog
    ``/Volumes/<catalog>/<inst>_bronze/bronze_volume/identity_agent/identity_profiling/``) for audit and
    cross-checks against HITL ``hitl_context`` strings.

    Parameters
    ----------
    output_dir:
        Directory to create; typically ``{bronze_volume_root}/identity_agent/identity_profiling``.
    institution_id:
        Institution identifier (top-level ``institution_id`` in the JSON).
    run_by_dataset:
        Mapping from :func:`build_identity_profiling_run_by_dataset`.
    filename:
        Output file name (single JSON envelope with all datasets).

    Returns
    -------
    Path
        Path written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = identity_profiling_run_to_jsonable(institution_id, run_by_dataset)
    path = output_dir / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "IdentityProfilingDatasetResult",
    "build_identity_profiling_run_by_dataset",
    "identity_profiling_run_to_jsonable",
    "write_identity_profiling_artifacts",
]
