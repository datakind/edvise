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
    assert not any("low_variance_threshold" in msg for msg in caplog.messages)


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
    """tol=0.99 drops a single value and keeps a column that is only 80% one value."""
    df_result = fs.drop_low_variance_features(df)
    dropped = set(df.columns.difference(df_result.columns))
    assert "zero_variance_feature" in dropped
    assert "low_variance_feature" in df_result.columns
    # One non-null value in five rows is under 99% of the row total.
    assert "incomplete_feature" in df_result.columns
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


def test_drop_low_variance_features_counts_nulls_in_the_row_total():
    """A repeated value must cover 99% of rows. Nulls stay in that total."""
    mostly_null = pd.DataFrame(
        {"mostly_null_constant": [1.0, np.nan, 1.0], "other": [1, 2, 3]}
    )
    assert fs.drop_low_variance_features(mostly_null).columns.tolist() == [
        "mostly_null_constant",
        "other",
    ]
    with_one_null = pd.DataFrame(
        {"almost_constant": [1.0] * 99 + [np.nan], "other": range(100)}
    )
    assert fs.drop_low_variance_features(with_one_null).columns.tolist() == ["other"]


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
    assert any(
        "deprecated" in message and "ignored" in message for message in caplog.messages
    )


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


def test_drop_collinear_features_iteratively_stops_when_one_feature_remains():
    """Equal keep-scores drop the later tied column, then VIF stops."""
    related_features = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    returned_df = fs.drop_collinear_features_iteratively(related_features)
    assert returned_df.columns.tolist() == ["A"]
    assert returned_df.shape[0] == related_features.shape[0]


def test_drop_collinear_features_iteratively_drops_worst_keep_score():
    """A lower AUC is dropped even when another column has the higher VIF."""
    related_features = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10], "C": [6.5, 1.5, 4, 3.5, 2]}
    )
    # B has the higher VIF. A is the weaker separator, so A is the one removed.
    returned_df = fs.drop_collinear_features_iteratively(
        related_features,
        aucs=pd.Series({"A": 0.55, "B": 0.80, "C": 0.60}),
        mis=pd.Series({"A": 0.10, "B": 0.20, "C": 0.05}),
        corrs=pd.Series({"A": 0.10, "B": 0.40, "C": 0.00}),
    )
    assert returned_df.columns.tolist() == ["B", "C"]


def test_repair_smart_correlated_groups_restores_force_include():
    drop = fs.repair_smart_correlated_groups(
        {"frac_courses_passed": ["num_courses_passed"]},
        aucs=pd.Series({"frac_courses_passed": 0.71, "num_courses_passed": 0.64}),
        force_include_cols=["num_courses_passed"],
    )
    assert drop == ["frac_courses_passed"]


def test_repair_smart_correlated_groups_breaks_auc_ties_only():
    groups = {"kept_by_scs": ["better_mi", "weaker"]}
    tied = fs.repair_smart_correlated_groups(
        groups,
        aucs=pd.Series({"kept_by_scs": 0.70, "better_mi": 0.70, "weaker": 0.70}),
        mis=pd.Series({"kept_by_scs": 0.10, "better_mi": 0.40, "weaker": 0.05}),
        corrs=pd.Series({"kept_by_scs": 0.20, "better_mi": 0.10, "weaker": 0.90}),
    )
    assert tied == ["kept_by_scs", "weaker"]

    distinct = fs.repair_smart_correlated_groups(
        groups,
        aucs=pd.Series({"kept_by_scs": 0.80, "better_mi": 0.70, "weaker": 0.60}),
        mis=pd.Series({"kept_by_scs": 0.01, "better_mi": 0.90, "weaker": 0.01}),
    )
    assert distinct == ["better_mi", "weaker"]


def test_drop_redundant_categoricals_keeps_higher_mi():
    program = ["stem"] * 20 + ["arts"] * 20
    area = program.copy()
    area[0] = "arts"
    area[20] = "stem"
    features = pd.DataFrame(
        {
            "program": pd.Series(program, dtype="string"),
            "area": pd.Series(area, dtype="string"),
            "unrelated": pd.Series(["yes", "no"] * 20, dtype="category"),
        }
    )
    returned = fs.drop_redundant_categoricals(
        features,
        mis=pd.Series({"program": 0.80, "area": 0.20, "unrelated": 0.10}),
    )
    assert returned.columns.tolist() == ["program", "unrelated"]


def test_drop_redundant_categoricals_keeps_force_include():
    labels = ["stem"] * 20 + ["arts"] * 20
    features = pd.DataFrame(
        {
            "program": pd.Series(labels, dtype="string"),
            "area": pd.Series(labels, dtype="string"),
        }
    )
    returned = fs.drop_redundant_categoricals(
        features,
        mis=pd.Series({"program": 0.90, "area": 0.10}),
        force_include_cols=["area"],
    )
    assert returned.columns.tolist() == ["area"]


def test_apply_feature_family_quotas_leaves_absent_families_empty():
    source = pd.DataFrame({"noise": [1, 0, 1]})
    restored = fs.apply_feature_family_quotas(source, source, candidate_cols=["noise"])
    assert restored.columns.tolist() == ["noise"]


def test_apply_feature_family_quotas_restores_best_dropped_member():
    source = pd.DataFrame(
        {
            "frac_courses_passed": [0.2, 0.8, 0.4, 0.9],
            "num_courses_passed": [1, 4, 2, 5],
            "noise": [1, 0, 1, 0],
        }
    )
    selected = source.loc[:, ["noise"]]
    restored = fs.apply_feature_family_quotas(
        selected,
        source,
        candidate_cols=source.columns,
        aucs=pd.Series({"frac_courses_passed": 0.71, "num_courses_passed": 0.64}),
        mis=pd.Series({"frac_courses_passed": 0.20, "num_courses_passed": 0.40}),
    )
    assert restored.columns.tolist() == ["frac_courses_passed", "noise"]


def test_select_features_repairs_force_include_after_smart_correlation():
    base = np.linspace(-3, 3, 90)
    features = pd.DataFrame(
        {
            "target": (base > 0).astype(int),
            "frac_courses_passed": base,
            "num_courses_passed": base * 2.0 + 1.0,
            "low_collinear_feature": np.sin(np.linspace(0, 20, 90)),
        }
    )
    returned = fs.select_features(
        features,
        non_feature_cols=["target"],
        force_include_cols=["num_courses_passed"],
        target_col="target",
        random_state=0,
        collinear_threshold=1e12,
    )
    assert returned.columns.tolist() == [
        "target",
        "num_courses_passed",
        "low_collinear_feature",
    ]
