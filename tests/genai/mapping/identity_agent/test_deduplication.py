import pandas as pd
import pytest

from edvise.genai.mapping.identity_agent.grain_inference.deduplication import (
    drop_duplicate_keys,
)


def test_drop_duplicate_keys_sort_prefers_row():
    df = pd.DataFrame(
        {
            "k": [1, 1, 1],
            "score": [10, 30, 20],
        }
    )
    out = drop_duplicate_keys(df, ["k"], sort_by=["score"], ascending=False)
    assert len(out) == 1
    assert int(out["score"].iloc[0]) == 30


def test_drop_duplicate_keys_missing_key_column():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="key_cols"):
        drop_duplicate_keys(df, ["missing"])


def test_drop_duplicate_keys_missing_sort_by_column():
    df = pd.DataFrame({"k": [1, 1], "score": [1, 2]})
    with pytest.raises(ValueError, match="sort_by"):
        drop_duplicate_keys(df, ["k"], sort_by=["nope"])
