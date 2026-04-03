"""
Step 1 — Key profiling: deterministic candidate keys and variance facts for LLM consumption.
"""

from .key_profiler import (
    CandidateKey,
    CandidateKeyProfile,
    ColumnVarianceProfile,
    KeyProfile,
    profile_candidate_keys,
)

__all__ = [
    "CandidateKey",
    "CandidateKeyProfile",
    "ColumnVarianceProfile",
    "KeyProfile",
    "profile_candidate_keys",
]
