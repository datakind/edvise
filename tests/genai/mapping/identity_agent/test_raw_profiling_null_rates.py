"""RawColumnProfile null_rate vs null_rate_including_tokens (CleaningConfig-aligned)."""

import pandas as pd
import pytest

from edvise.configs.custom import CleaningConfig
from edvise.genai.mapping.identity_agent.profiling.raw_snapshot import profile_raw_table


def test_null_rate_including_tokens_counts_blank_sentinel():
    df = pd.DataFrame({"t": ["(Blank)", "2001FA", "(Blank)"]})
    rtp = profile_raw_table(df, "inst", "student", cleaning=CleaningConfig())
    col = next(c for c in rtp.columns if c.name == "t")
    assert col.null_rate == 0.0
    assert col.null_rate_including_tokens == pytest.approx(2 / 3, rel=1e-4)


def test_null_rate_including_tokens_default_without_cleaning_config():
    """Same defaults as clean_dataset when cleaning_cfg is None."""
    df = pd.DataFrame({"t": ["(Blank)", "x"]})
    rtp = profile_raw_table(df, "inst", "student", cleaning=None)
    col = next(c for c in rtp.columns if c.name == "t")
    assert col.null_rate == 0.0
    assert col.null_rate_including_tokens == pytest.approx(0.5, rel=1e-4)


def test_null_rate_including_tokens_custom_tokens():
    df = pd.DataFrame({"t": ["N/A", "ok", "N/A"]})
    rtp = profile_raw_table(
        df,
        "inst",
        "student",
        cleaning=CleaningConfig(null_tokens=["(Blank)", "N/A"]),
    )
    col = next(c for c in rtp.columns if c.name == "t")
    assert col.null_rate == 0.0
    assert col.null_rate_including_tokens == pytest.approx(2 / 3, rel=1e-4)


def test_whitespace_only_counts_when_treat_empty_strings_as_null():
    df = pd.DataFrame({"t": ["  ", "a"]})
    rtp = profile_raw_table(df, "inst", "student", cleaning=CleaningConfig())
    col = next(c for c in rtp.columns if c.name == "t")
    assert col.null_rate == 0.0
    assert col.null_rate_including_tokens == pytest.approx(0.5, rel=1e-4)


def test_native_null_in_both_rates():
    df = pd.DataFrame({"t": [None, "2001FA"]})
    rtp = profile_raw_table(df, "inst", "student", cleaning=CleaningConfig())
    col = next(c for c in rtp.columns if c.name == "t")
    assert col.null_rate == pytest.approx(0.5, rel=1e-4)
    assert col.null_rate_including_tokens == pytest.approx(0.5, rel=1e-4)
