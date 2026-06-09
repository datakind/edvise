import json

import pandas as pd

from edvise.genai.mapping.shared.profiling.variance import compute_within_group_variance


def test_compute_within_group_variance_matches_manual_dup_profile() -> None:
    df = pd.DataFrame(
        {
            "k": [1, 1, 2],
            "a": [10, 20, 30],
            "b": ["x", "x", "y"],
        }
    )
    r = compute_within_group_variance(df, ["k"], profile_cols=["a", "b"])
    assert r.non_unique_rows == 2
    assert r.affected_groups == 1
    assert r.group_size_distribution[2] == 1
    assert r.column_profiles[0].column == "a"
    assert r.column_profiles[0].pct_groups_with_variance == 1.0
    assert r.column_profiles[1].column == "b"
    assert r.column_profiles[1].pct_groups_with_variance == 0.0


def test_compute_within_group_variance_unique_key_empty_profiles() -> None:
    df = pd.DataFrame({"k": [1, 2, 3], "a": [1, 2, 3]})
    r = compute_within_group_variance(df, ["k"])
    assert r.non_unique_rows == 0
    assert r.column_profiles == []


def test_compute_within_group_variance_sample_values_are_json_serializable() -> None:
    df = pd.DataFrame(
        {
            "k": [1, 1, 2],
            "started": pd.to_datetime(["2024-08-19", "2024-08-20", "2025-01-12"]),
        }
    )
    r = compute_within_group_variance(df, ["k"], profile_cols=["started"])
    assert r.column_profiles[0].column == "started"
    for val in r.column_profiles[0].sample_values:
        assert isinstance(val, str)
    json.dumps(
        {
            "top_column_profiles": [
                {
                    "column": p.column,
                    "pct_groups_with_variance": p.pct_groups_with_variance,
                    "sample_values": p.sample_values,
                }
                for p in r.column_profiles
            ],
            "group_size_distribution": r.group_size_distribution,
            "sampled": r.sampled,
        }
    )
