"""Tests for Step 2b conferral datetime utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from edvise.genai.mapping.schema_mapping_agent.transformation.utilities import (
    compact_term_code_to_conferral_date,
    spelled_season_year_to_conferral_date,
)


@pytest.mark.parametrize(
    "token,expected",
    [
        ("2019 Spring", "2019-05-31"),
        ("Fall 2023", "2023-12-31"),
        ("2020-Summer", "2020-08-31"),
        ("Winter/2021", "2021-03-31"),
        ("2025SP", pd.NaT),  # compact codes belong on compact_term_code_to_conferral_date
        ("not-a-term", pd.NaT),
    ],
)
def test_spelled_season_year_to_conferral_date(token, expected):
    out = spelled_season_year_to_conferral_date(pd.Series([token]))
    if pd.isna(expected):
        assert pd.isna(out.iloc[0])
    else:
        assert out.iloc[0] == pd.Timestamp(expected)


def test_spelled_season_year_open_ended_future_years():
    """Future years must parse without a finite map_values table."""
    out = spelled_season_year_to_conferral_date(
        pd.Series(["2025 Spring", "2026 Fall", "Spring 2027"])
    )
    assert list(out) == [
        pd.Timestamp("2025-05-31"),
        pd.Timestamp("2026-12-31"),
        pd.Timestamp("2027-05-31"),
    ]


def test_compact_term_code_still_handles_contiguous():
    out = compact_term_code_to_conferral_date(pd.Series(["2025SP", "2019 Spring"]))
    assert out.iloc[0] == pd.Timestamp("2025-05-31")
    assert pd.isna(out.iloc[1])
