import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from edvise.modeling.inference import (
    _get_mapped_feature_name,
    select_top_features_for_display,
    generate_ranked_feature_table,
    top_shap_features,
    support_score_distribution_table,
)


@pytest.mark.parametrize(
    [
        "features",
        "unique_ids",
        "predicted_probabilities",
        "shap_values",
        "n_features",
        "needs_support_threshold_prob",
        "features_table",
        "exp",
    ],
    [
        (
            pd.DataFrame(
                {
                    "x1": ["val1", "val2", "val3"],
                    "x2": [True, False, True],
                    "x3": [2.0, 1.0001, 0.5],
                    "x4": [1, 2, 3],
                }
            ),
            pd.Series([1, 2, 3]),
            [0.9, 0.1, 0.5],
            np.array(
                [[1.0, 0.9, 0.8, 0.7], [0.0, -1.0, 0.9, -0.8], [0.25, 0.0, -0.5, 0.75]]
            ),
            3,
            0.5,
            {
                "x1": {"name": "feature #1"},
                "x2": {"name": "feature #2"},
                "x3": {"name": "feature #3"},
            },
            pd.DataFrame(
                {
                    "Student ID": [1, 2, 3],
                    "Support Score": [0.9, 0.1, 0.5],
                    "Support Needed": [True, False, True],
                    "Feature_1_Name": ["feature #1", "feature #2", "x4"],
                    "Feature_1_Value": ["val1", "False", "3"],
                    "Feature_1_Importance": [1.0, -1.0, 0.75],
                    "Feature_2_Name": ["feature #2", "feature #3", "feature #3"],
                    "Feature_2_Value": ["True", "1.0", "0.5"],
                    "Feature_2_Importance": [0.9, 0.9, -0.5],
                    "Feature_3_Name": ["feature #3", "x4", "feature #1"],
                    "Feature_3_Value": ["2.0", "2", "val3"],
                    "Feature_3_Importance": [0.8, -0.8, 0.25],
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "x1": ["val1", "val2", "val3"],
                    "x2": [True, False, True],
                    "x3": [2.0, 1.0, 0.5],
                    "x4": [1, 2, 3],
                }
            ),
            pd.Series([1, 2, 3]),
            [0.9, 0.1, 0.5],
            np.array(
                [[1.0, 0.9, 0.8, 0.7], [0.0, -1.0, 0.9, -0.8], [0.25, 0.0, -0.5, 0.75]]
            ),
            1,
            None,
            None,
            pd.DataFrame(
                {
                    "Student ID": [1, 2, 3],
                    "Support Score": [0.9, 0.1, 0.5],
                    "Feature_1_Name": ["x1", "x2", "x4"],
                    "Feature_1_Value": ["val1", "False", "3"],
                    "Feature_1_Importance": [1.0, -1.0, 0.75],
                }
            ),
        ),
    ],
)
def test_select_top_features_for_display(
    features,
    unique_ids,
    predicted_probabilities,
    shap_values,
    n_features,
    needs_support_threshold_prob,
    features_table,
    exp,
):
    obs = select_top_features_for_display(
        features,
        unique_ids,
        predicted_probabilities,
        shap_values,
        n_features=n_features,
        needs_support_threshold_prob=needs_support_threshold_prob,
        features_table=features_table,
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert pd.testing.assert_frame_equal(obs, exp) is None


@pytest.fixture
def ranked_feature_table_data():
    features = pd.DataFrame(
        {
            "pell_status": [False, True, False],
            "english_math_gateway": ["E", "M", "M"],
            "term_gpa": [4.0, 3.0, 2.0],
        }
    )
    shap_values = np.array(
        [
            [0.1, -0.2, 0.05],
            [0.3, 0.1, -0.05],
            [0.2, -0.1, 0.15],
        ]
    )
    features_table = {
        "pell_status": {"name": "Pell Status"},
        "english_math_gateway": {"name": "English or Math Gateway"},
        "term_gpa": {"name": "Term GPA"},
    }
    return features, shap_values, features_table


# --- generate_ranked_feature_table tests ---


@pytest.mark.parametrize("use_features_table", [True, False])
@pytest.mark.parametrize("metadata", [True, False])
def test_generate_ranked_feature_table_columns_and_sorting(
    ranked_feature_table_data, use_features_table, metadata
):
    """
    Verifies:
    - returned columns match expectation based on metadata flag
    - sorting is descending by numeric average |SHAP|
    - readable names resolve correctly with and without features_table
    """
    features, shap_values, features_table = ranked_feature_table_data
    selected_features_table = features_table if use_features_table else None

    result = generate_ranked_feature_table(
        features=features,
        shap_values=shap_values,
        features_table=selected_features_table,
        metadata=metadata,
    )

    assert isinstance(result, pd.DataFrame) and not result.empty

    # Columns expected
    base_cols = {
        "feature_name",
        "readable_feature_name",
        "data_type",
        "average_shap_magnitude",
    }
    if metadata:
        expected_cols = base_cols | {"short_feature_desc", "long_feature_desc"}
    else:
        expected_cols = base_cols

    assert set(result.columns) == expected_cols

    # Check sorting numerically by recomputing avg |SHAP|
    avg_abs = np.mean(np.abs(shap_values), axis=0)  # shape: (n_features,)

    # Map the dataframe values back to numeric for comparison
    def to_float(val: str) -> float:
        return 0.0 if val == "<0.0000" else float(val)

    df_values = result["average_shap_magnitude"].map(to_float).to_numpy()
    # Ensure descending
    assert np.all(df_values[:-1] >= df_values[1:])

    # Ensure values match the inputs (after rounding to 4 decimals as in the function)
    # Build a mapping from feature -> avg |SHAP|
    feat_to_avg = {col: round(float(v), 4) for col, v in zip(features.columns, avg_abs)}
    # Compare per-row by feature_name (original raw column)
    for _, row in result.iterrows():
        raw_feat = row["feature_name"]
        expected_val = feat_to_avg[raw_feat]
        observed_val = to_float(row["average_shap_magnitude"])
        assert observed_val == expected_val or (
            expected_val == 0.0 and observed_val == 0.0
        )

    # Name mapping expectations
    if use_features_table:
        # With a features table, we should see at least one human-friendly name
        assert (result["readable_feature_name"] == "English or Math Gateway").any()
    else:
        # Without a features table, readable names should fall back to raw column names
        assert (result["readable_feature_name"] == "term_gpa").any()


@pytest.mark.parametrize("metadata", [True, False])
def test_generate_ranked_feature_table_dtype_labels(
    ranked_feature_table_data, metadata
):
    """
    Spot-check that data_type is one of the three categories and that booleans/numerics/categoricals
    are labeled consistently.
    """
    features, shap_values, features_table = ranked_feature_table_data

    # Coerce one column to boolean and one to category for the test (if they exist)
    if "is_first_term" in features.columns:
        features = features.copy()
        features["is_first_term"] = features["is_first_term"].astype(bool)
    if "major" in features.columns:
        features = features.copy()
        features["major"] = features["major"].astype("category")

    result = generate_ranked_feature_table(
        features=features,
        shap_values=shap_values,
        features_table=features_table,
        metadata=metadata,
    )

    assert set(result["data_type"].unique()).issubset(
        {"Boolean", "Continuous", "Categorical"}
    )


def test_generate_ranked_feature_table_original_dtypes():
    """
    Test that original_dtypes correctly identifies boolean features that were
    converted to numeric types during sklearn processing.
    """
    # Create a feature that's float64 (simulating sklearn conversion)
    # but was originally boolean
    features = pd.DataFrame(
        {
            "cummax_in_12_creds_took_course_subject_area_math": [0.0, 1.0, 0.0],
            "term_gpa": [4.0, 3.0, 2.0],
            "course_grade_numeric_mean_cumstd": [1.5, 2.0, 1.8],
        }
    )
    shap_values = np.array(
        [
            [0.1, 0.05, 0.02],
            [0.3, -0.05, 0.01],
            [0.2, 0.15, 0.03],
        ]
    )

    # original_dtypes uses string values as they would be stored in JSON
    # This matches the actual format from MLflow: {"term_in_peak_covid": "bool", ...}
    original_dtypes = {
        "cummax_in_12_creds_took_course_subject_area_math": "bool",
        "term_gpa": "float64",
        "course_grade_numeric_mean_cumstd": "float64",
    }

    result = generate_ranked_feature_table(
        features=features,
        shap_values=shap_values,
        features_table=None,
        metadata=False,
        original_dtypes=original_dtypes,
    )

    # The boolean feature should be classified as "Boolean" even though it's float64
    bool_feature = result[
        result["feature_name"] == "cummax_in_12_creds_took_course_subject_area_math"
    ]
    assert len(bool_feature) == 1
    assert bool_feature.iloc[0]["data_type"] == "Boolean", (
        f"Expected 'Boolean' but got '{bool_feature.iloc[0]['data_type']}'"
    )

    # The numeric features should be classified as "Continuous"
    numeric_feature = result[result["feature_name"] == "term_gpa"]
    assert len(numeric_feature) == 1
    assert numeric_feature.iloc[0]["data_type"] == "Continuous"

    # Test that None original_dtypes works (backward compatibility)
    result_no_original = generate_ranked_feature_table(
        features=features,
        shap_values=shap_values,
        features_table=None,
        metadata=False,
        original_dtypes=None,
    )

    # Without original_dtypes, the float64 feature should be classified as "Continuous"
    bool_feature_no_original = result_no_original[
        result_no_original["feature_name"]
        == "cummax_in_12_creds_took_course_subject_area_math"
    ]
    assert len(bool_feature_no_original) == 1
    assert bool_feature_no_original.iloc[0]["data_type"] == "Continuous"


# --- _get_mapped_feature_name tests ---


@pytest.mark.parametrize(
    ["feature_col", "features_table", "exp"],
    [
        (
            "academic_term",
            {"academic_term": {"name": "academic term"}},
            "academic term",
        ),
        ("foo_bar", {"academic_term": {"name": "academic term"}}, "foo_bar"),
        (
            "num_courses_course_subject_area_24",
            {
                r"num_courses_course_subject_area_(\d+)": {
                    "name": "number of courses taken in subject area {} this term"
                }
            },
            "number of courses taken in subject area 24 this term",
        ),
        (
            "num_courses_course_id_engl_101",
            {
                r"num_courses_course_id_(.*)": {
                    "name": "number of times course '{}' taken this term"
                }
            },
            "number of times course 'engl_101' taken this term",
        ),
        (
            "num_courses_course_id_engl_101_cumfrac",
            {
                r"num_courses_course_id_(.*)_cumfrac": {
                    "name": "fraction of times course '{}' taken so far"
                }
            },
            "fraction of times course 'engl_101' taken so far",
        ),
    ],
)
def test_get_mapped_feature_name_no_metadata(feature_col, features_table, exp):
    obs = _get_mapped_feature_name(feature_col, features_table, metadata=False)
    assert isinstance(obs, str)
    assert obs == exp


@pytest.mark.parametrize(
    ["feature_col", "features_table", "exp_name"],
    [
        (
            "academic_term",
            {
                "academic_term": {
                    "name": "academic term",
                    "short_desc": "sd",
                    "long_desc": "ld",
                }
            },
            "academic term",
        ),
        (
            "foo_bar",
            {},
            "foo_bar",
        ),
        (
            "num_courses_course_subject_area_24",
            {
                r"num_courses_course_subject_area_(\d+)": {
                    "name": "number of courses taken in subject area {} this term",
                    "short_feature_desc": "short",
                    "long_feature_desc": "long",
                }
            },
            "number of courses taken in subject area 24 this term",
        ),
    ],
)
def test_get_mapped_feature_name_with_metadata(feature_col, features_table, exp_name):
    """
    When metadata=True, expect a 3-tuple: (name, short_desc|None, long_desc|None).
    """
    obs = _get_mapped_feature_name(feature_col, features_table, metadata=True)
    assert isinstance(obs, tuple) and len(obs) == 3
    name, short_desc, long_desc = obs
    assert name == exp_name
    # short/long may be None depending on table; just ensure tuple structure


@pytest.fixture
def sample_data():
    features = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "feature3": [7, 8, 9],
            "feature4": [0, 1, 0],
            "feature5": [2, 2, 2],
            "feature6": [1, 1, 1],
            "feature7": [0, 0, 1],
            "feature8": [3, 3, 3],
            "feature9": [4, 4, 4],
            "feature10": [5, 5, 5],
            "feature11": [6, 6, 6],
        }
    )
    unique_ids = pd.Series([101, 102, 103])
    shap_values = np.array(
        [
            [0.1, 0.3, 0.2, 0.0, 0.4, 0.1, 0.0, 0.3, 0.2, 0.5, 0.6],
            [0.2, 0.2, 0.1, 0.0, 0.3, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.3, 0.1, 0.3, 0.0, 0.2, 0.0, 0.0, 0.1, 0.4, 0.3, 0.4],
        ]
    )
    features_table = {
        "feature1": {
            "name": "Feature 1 Name",
            "short_desc": "A short description of feature 1",
            "long_desc": "A long description of feature 1",
        },
        "feature2": {
            "name": "Feature 2 Name",
            "short_desc": "A short description of feature 2",
            "long_desc": "A long description of feature 2",
        },
        "feature3": {
            "name": "Feature 3 Name",
        },
    }
    return features, unique_ids, shap_values, features_table


def test_top_shap_features_behavior(sample_data):
    features, unique_ids, shap_values, features_table = sample_data
    result = top_shap_features(
        features, unique_ids, shap_values, features_table=features_table
    )

    # Check output shape and columns
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "student_id",
        "feature_name",
        "shap_value",
        "feature_value",
        "feature_readable_name",
        "feature_short_desc",
        "feature_long_desc",
    }

    # Check top 10 feature selection
    top_features = result["feature_name"].unique()
    assert len(top_features) == 10

    grouped = result.groupby("feature_readable_name")["shap_value"].apply(
        lambda x: np.mean(np.abs(x))
    )
    shap_values = grouped.sort_values(ascending=False).values
    assert all(
        shap_values[i] >= shap_values[i + 1] for i in range(len(shap_values) - 1)
    )

    assert len(grouped) == 10
    print(grouped)
    assert grouped.index[0] == "Feature 1 Name"
    assert grouped.index[1] == "Feature 2 Name"

    assert (
        result["feature_short_desc"]
        .apply(lambda x: isinstance(x, str) or x is None)
        .all()
    )
    assert (
        result["feature_long_desc"]
        .apply(lambda x: isinstance(x, str) or x is None)
        .all()
    )


def test_handles_fewer_than_10_features():
    features = pd.DataFrame(
        {
            "feature1": [1, 2],
            "feature2": [3, 4],
        }
    )
    unique_ids = pd.Series([1, 2])
    shap_values = np.array([[0.5, 0.1], [0.3, 0.4]])

    result = top_shap_features(features, unique_ids, shap_values)
    assert set(result["feature_name"].unique()) == {"feature1", "feature2"}
    assert len(result) == 4  # 2 students × 2 features


def test_empty_input():
    features = pd.DataFrame()
    unique_ids = pd.Series(dtype=int)
    shap_values = np.empty((0, 0))

    with pytest.raises(ValueError):
        top_shap_features(features, unique_ids, shap_values)


@patch("edvise.modeling.inference.select_top_features_for_display")
@pytest.mark.parametrize(
    [
        "features",
        "unique_ids",
        "predicted_probabilities",
        "shap_values",
        "n_features",
        "needs_support_threshold_prob",
        "features_table",
        "exp",
    ],
    [
        (
            pd.DataFrame(
                {
                    "x1": ["val1", "val2", "val3"],
                    "x2": [True, False, True],
                    "x3": [2.0, 1.0001, 0.5],
                    "x4": [1, 2, 3],
                }
            ),
            pd.Series([1, 2, 3]),
            [0.9, 0.1, 0.5],
            np.array(
                [
                    [1.0, 0.9, 0.8, 0.7],
                    [0.0, -1.0, 0.9, -0.8],
                    [0.25, 0.0, -0.5, 0.75],
                ]
            ),
            3,
            0.5,
            {
                "x1": {"name": "feature #1"},
                "x2": {"name": "feature #2"},
                "x3": {"name": "feature #3"},
            },
            pd.DataFrame(
                {
                    "Student ID": [1, 2, 3],
                    "Support Score": [0.9, 0.1, 0.5],
                    "Support Needed": [True, False, True],
                    "Feature_1_Name": ["feature #1", "feature #2", "x4"],
                    "Feature_1_Value": ["val1", "False", "3"],
                    "Feature_1_Importance": [1.0, -1.0, 0.75],
                    "Feature_2_Name": ["feature #2", "feature #3", "feature #3"],
                    "Feature_2_Value": ["True", "1.0", "0.5"],
                    "Feature_2_Importance": [0.9, 0.9, -0.5],
                    "Feature_3_Name": ["feature #3", "x4", "feature #1"],
                    "Feature_3_Value": ["2.0", "2", "val3"],
                    "Feature_3_Importance": [0.8, -0.8, 0.25],
                }
            ),
        )
    ],
)
def test_support_score_distribution_table(
    mock_select_top_features_for_display,
    features,
    unique_ids,
    predicted_probabilities,
    shap_values,
    n_features,
    needs_support_threshold_prob,
    features_table,
    exp,
):
    inference_params = {
        "num_top_features": n_features,
        "min_prob_pos_label": needs_support_threshold_prob or 0.0,
    }

    mock_select_top_features_for_display.return_value = exp

    result = support_score_distribution_table(
        df_serving=features,
        unique_ids=unique_ids,
        pred_probs=predicted_probabilities,
        shap_values=pd.DataFrame(shap_values),
        inference_params=inference_params,
        features_table=features_table,
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "bin_lower",
        "bin_upper",
        "support_score",
        "count_of_students",
        "pct",
    }
    assert result["count_of_students"].sum() == len(unique_ids)
    assert np.isclose(result["pct"].sum(), 1.0, atol=1e-6)

    bin_width = 0.2 / 5
    # Binning logic checks
    for _, row in result.iterrows():
        expected_midpoint = (row["bin_lower"] + row["bin_upper"]) / 2.0
        assert np.isclose(row["support_score"], expected_midpoint, atol=1e-8)
        assert np.isclose(row["bin_upper"] - row["bin_lower"], bin_width, atol=1e-8)
        assert 0.0 <= row["bin_lower"] < row["bin_upper"] <= 1.0
