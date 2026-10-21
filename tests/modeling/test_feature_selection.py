import numpy as np
import pandas as pd
import pytest

from edvise.modeling import feature_selection as fs


@pytest.fixture(scope="module")
def df():
    return pd.DataFrame(
        {
            "id": ["01", "02", "03", "04", "05"],
            "categorical_feature": ["a", "a", "b", "c", "b"],
            "include_feature": [3.142, 1.414, 1.618, 2.718, 0.693],
            "good_feature": [1, 2, 3, 4, 5],
            "partially_incomplete_feature": [np.nan, np.nan, 8, 5, 3],
            "incomplete_feature": [np.nan, 0.0, np.nan, np.nan, np.nan],
            "low_variance_feature": [1, 1, 1, 1, 2],
            "zero_variance_feature": [1, 1, 1, 1, 1],
            "high_collinear_feature": [6, 7, 8, 9, 10],
            "low_collinear_feature": [6.5, 1.5, 4, 3.5, 2],
        }
    ).astype({"id": "string", "categorical_feature": "category"})


# TODO: figure out why collinaer features col keeps raising errors / gives bad results
# @pytest.mark.parametrize(
#     ["non_feature_cols", "force_include_cols", "exp_dropped_cols"],
#     [
#         (
#             ["id"],
#             ["include_feature"],
#             ["incomplete_feature", "zero_variance_feature", "high_collinear_feature"],
#         ),
#     ],
# )
# def test_select_features(df, non_feature_cols, force_include_cols, exp_dropped_cols):
#     df_result = fs.select_features(
#         df, non_feature_cols=non_feature_cols, force_include_cols=force_include_cols
#     )
#     assert df.columns.difference(df_result.columns).tolist() == exp_dropped_cols
#     assert df_result.shape[0] == df.shape[0]


@pytest.mark.parametrize(
    ["threshold", "exp_dropped_cols"],
    [
        (0.5, ["incomplete_feature"]),
        (0.4, ["incomplete_feature", "partially_incomplete_feature"]),
    ],
)
def test_drop_incomplete_features(df, threshold, exp_dropped_cols):
    df_result = fs.drop_incomplete_features(df, threshold=threshold)
    assert df.columns.difference(df_result.columns).tolist() == exp_dropped_cols
    assert df_result.shape[0] == df.shape[0]


def test_select_features_caps_incomplete_threshold(df, caplog):
    """Values above MAX_INCOMPLETE_THRESHOLD are capped before dropping incompletes."""
    # partially_incomplete_feature is 40% null; incomplete_feature is 80% null.
    # Cap at 0.2 => both drop. Without the cap (0.5), only incomplete_feature drops.
    with caplog.at_level("WARNING"):
        df_result = fs.select_features(
            df,
            non_feature_cols=["id"],
            force_include_cols=["include_feature"],
            incomplete_threshold=0.5,
            low_variance_threshold=0.0,
            collinear_threshold=None,
        )
    dropped = df.columns.difference(df_result.columns).tolist()
    assert "incomplete_feature" in dropped
    assert "partially_incomplete_feature" in dropped
    assert any("capping for feature selection" in msg for msg in caplog.messages)


def test_select_features_allows_stricter_incomplete_threshold(df):
    df_result = fs.select_features(
        df,
        non_feature_cols=["id"],
        force_include_cols=["include_feature"],
        incomplete_threshold=0.1,
        low_variance_threshold=0.0,
        collinear_threshold=None,
    )
    dropped = df.columns.difference(df_result.columns).tolist()
    assert "incomplete_feature" in dropped
    assert "partially_incomplete_feature" in dropped


def test_drop_low_variance_features_drops_exact_constants(df):
    """tol=1 drops a single non-null value and keeps a near-constant numeric."""
    df_result = fs.drop_low_variance_features(df)
    dropped = set(df.columns.difference(df_result.columns))
    assert dropped == {"incomplete_feature", "zero_variance_feature"}
    assert "low_variance_feature" in df_result.columns
    assert df_result.shape[0] == df.shape[0]


def test_drop_low_variance_features_drops_constant_labels():
    features = pd.DataFrame(
        {
            "const_label": pd.Series(["fall", "fall", "fall"], dtype="category"),
            "const_bool": [True, True, True],
            "grade": pd.Series(["A", "B", "A"], dtype="string"),
        }
    )
    returned = fs.drop_low_variance_features(features)
    assert returned.columns.tolist() == ["grade"]


def test_drop_low_variance_features_ignores_nulls_in_constant_column():
    features = pd.DataFrame(
        {"mostly_null_constant": [1.0, np.nan, 1.0], "other": [1, 2, 3]}
    )
    returned = fs.drop_low_variance_features(features)
    assert returned.columns.tolist() == ["other"]


def test_drop_low_variance_features_flags_id_like_categoricals(df, caplog):
    with caplog.at_level("WARNING"):
        returned = fs.drop_low_variance_features(df)
    assert "id" in returned.columns
    assert any(
        "ID-like categorical columns" in message and "id" in message
        for message in caplog.messages
    )


def test_select_features_does_not_apply_variance_threshold(df, caplog):
    with caplog.at_level("WARNING"):
        result = fs.select_features(
            df,
            non_feature_cols=["id"],
            force_include_cols=["include_feature"],
            low_variance_threshold=0.2,
            collinear_threshold=None,
        )
    assert "low_variance_feature" in result.columns
    assert "zero_variance_feature" not in result.columns
    assert any("not applied" in message for message in caplog.messages)


def test_drop_collinear_features_iteratively():
    related_features = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10], "C": [6.5, 1.5, 4, 3.5, 2]}
    )
    returned_df = fs.drop_collinear_features_iteratively(
        related_features, force_include_cols=[]
    )
    assert returned_df.columns.tolist() == ["A", "C"]
    assert returned_df.shape[0] == related_features.shape[0]


def test_drop_duplicate_features_bool_matches_numeric_twin():
    related_features = pd.DataFrame(
        {
            "A": [2, 12, 6.5, 4, 3.5],
            "bool": [True, False, True, False, True],
            "numeric": [1, 0, 1, 0, 1],
        }
    )
    returned_df = fs.drop_duplicate_features(related_features, force_include_cols=[])
    assert returned_df.columns.tolist() == ["A", "bool"]
    assert returned_df.shape[0] == related_features.shape[0]


def test_drop_duplicate_features_drops_every_twin_set(caplog):
    """Every identical pair is removed, including numeric-only frames and categoricals."""
    features = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "a_copy": [1, 2, 3, 4],
            "b": [8, 1, 3, 5],
            "b_copy": [8, 1, 3, 5],
            "color": pd.Series(["red", "blue", "red", "green"], dtype="category"),
            "color_copy": pd.Series(["red", "blue", "red", "green"], dtype="string"),
        }
    )
    with caplog.at_level("INFO"):
        returned_df = fs.drop_duplicate_features(features)
    assert returned_df.columns.tolist() == ["a", "b", "color"]
    summary = next(
        message for message in caplog.messages if message.startswith("dropping")
    )
    assert "a_copy (duplicate of a)" in summary
    assert "b_copy (duplicate of b)" in summary
    assert "color_copy (duplicate of color)" in summary


def test_drop_duplicate_features_distinguishes_null_positions():
    features = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "same": [1.0, np.nan, 3.0],
            "different": [1.0, 2.0, 3.0],
        }
    )
    returned_df = fs.drop_duplicate_features(features)
    assert returned_df.columns.tolist() == ["a", "different"]


def test_drop_duplicate_features_prefers_force_include_over_earlier_twin():
    features = pd.DataFrame(
        {"copy": [1, 2, 3], "original": [1, 2, 3], "other": [3, 1, 2]}
    )
    returned_df = fs.drop_duplicate_features(features, force_include_cols=["original"])
    assert returned_df.columns.tolist() == ["original", "other"]


def test_drop_duplicate_features_warns_when_force_includes_match(caplog):
    features = pd.DataFrame(
        {
            "keep_me": [1, 2, 3, 4],
            "also_forced": [1, 2, 3, 4],
            "other": [4, 1, 3, 2],
        }
    )
    with caplog.at_level("WARNING"):
        returned_df = fs.drop_duplicate_features(
            features, force_include_cols=["also_forced", "keep_me"]
        )
    assert returned_df.columns.tolist() == ["keep_me", "other"]
    assert any(
        "Duplicate force-included columns" in message for message in caplog.messages
    )


def test_select_features_drops_identical_columns_before_vif():
    """Identical numerics are removed even when the frame has no boolean column.

    A copy that reached VIF would be dropped together with its twin, because both
    have infinite VIF. Keeping the earlier column shows the duplicate step ran first.
    """
    features = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "a_copy": [1, 2, 3, 4, 5],
            "b": [5, 1, 4, 2, 3],
            "b_copy": [5, 1, 4, 2, 3],
            "c": [0, 1, 0, 1, 0],
        }
    )
    returned_df = fs.select_features(
        features,
        force_include_cols=["b_copy"],
        collinear_threshold=1e12,
    )
    assert returned_df.columns.tolist() == ["a", "b_copy", "c"]


def test_drop_collinear_features_iteratively_force_include():
    related_features = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10], "C": [6.5, 1.5, 4, 3.5, 2]}
    )
    returned_df = fs.drop_collinear_features_iteratively(
        related_features, force_include_cols=["B"]
    )
    assert returned_df.columns.tolist() == ["B", "C"]
    assert returned_df.shape[0] == related_features.shape[0]


def test_drop_collinear_features_iteratively_fails_with_one_feature_left():
    related_features = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    with pytest.raises(Exception):
        fs.drop_collinear_features_iteratively(related_features)
