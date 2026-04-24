"""Shared thresholds, regex patterns, and name heuristics for table profiling."""

from __future__ import annotations

import re

# --- Raw column / snapshot profiling ---
UNIQUE_VALUES_MAX_CARDINALITY = 50
SAMPLE_VALUES_TOP_N = 5

TERM_COLUMN_PATTERNS = re.compile(
    r"(?:^term|_term|term_|semester|session|acad[_\s]|enroll.*date|strm)",
    re.IGNORECASE,
)

# --- Candidate key detection and variance profiling ---
TIER1_MIN_NUNIQUE_ABS = 50
MIN_DISCRIMINATOR_NUNIQUE = 2
MAX_NULL_RATE_TIER1 = 0.05
MAX_NULL_RATE_TIER2 = 0.30
EARLY_STOP_UNIQUENESS = 0.995
MAX_CANDIDATE_POOL = 8
MAX_KEY_SIZE = 6
TOP_K_CANDIDATES = 10
LARGE_TABLE_ROW_THRESHOLD = 500_000
MAX_COMBINATION_EVALS = 3_000
MAX_KEY_SIZE_LARGE_TABLE = 3
MAX_COMBINATION_EVALS_LARGE_TABLE = 500
NEAR_BEST_STOP_DELTA = 0.003
LARGE_TABLE_KEY_SEARCH_SAMPLE_ROWS = 100_000
TOP_K_CANDIDATES_LARGE_TABLE = 5

SAMPLE_GROUP_SIZE = 500
PROFILE_MAX_WORK_ROWS = 150_000
WITHIN_GROUP_SAMPLE_VALUES = 5

INDEX_COLUMN_PATTERNS = re.compile(
    r"^(unnamed:\s*[\d.]+|index|row_number|row_num|rownum|row_id|__index_level_\d+__)$",
    re.IGNORECASE,
)

STUDENT_ANCHOR_NAME_PATTERN = re.compile(
    r"(?:student[_\s]?id|learner[_\s]?id|person[_\s]?id|enrollment[_\s]?id|sis[_\s]?id|"
    r"member[_\s]?id|participant[_\s]?id)",
    re.IGNORECASE,
)
