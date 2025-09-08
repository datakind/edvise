import pandas as pd
import pytest
import tomlkit
import numpy as np

from edvise.model_prep import training_params


@pytest.mark.parametrize(
    ["df", "label_fracs", "shuffle", "seed"],
    [
        (
            pd.DataFrame({"x": range(1000)}),
            {"train": 0.6, "validate": 0.2, "test": 0.2},
            True,
            10,
        ),
        (
            pd.DataFrame({"x": range(1000)}),
            {"train": 0.6, "validate": 0.2, "test": 0.2},
            False,
            11,
        ),
        (
            pd.DataFrame({"x": range(1000)}),
            {"train": 0.5, "validate": 0.25, "test": 0.25},
            True,
            42,
        ),
    ],
)
def test_compute_dataset_splits(df, label_fracs, shuffle, seed):
    obs = training_params.compute_dataset_splits(
        df,
        stratify_col=None,
        label_fracs=label_fracs,
        shuffle=shuffle,
        seed=seed,
    )
    assert isinstance(obs, pd.Series)
    assert len(obs) == len(df)
    assert obs.isna().sum() == 0
    assert set(obs.unique()) <= {"train", "validate", "test"}

    labels = ["train", "validate", "test"]
    fracs = [label_fracs[l] for l in labels]

    obs_value_counts = (
        obs.value_counts(normalize=True)
        .reindex(labels, fill_value=0.0)
        .astype("float64")
    )
    # make expected index dtype match observed
    exp_index = pd.Index(labels, dtype=obs_value_counts.index.dtype, name="split")
    exp_value_counts = pd.Series(
        data=np.array(fracs) / np.sum(fracs),
        index=exp_index,
        name="proportion",
        dtype="float64",
    )

    pd.testing.assert_series_equal(
        obs_value_counts,
        exp_value_counts,
        rtol=0.15,
        check_exact=False,
        check_names=True,
    )

    # Reproducibility with same seed
    if seed is not None:
        obs2 = training_params.compute_dataset_splits(
            df,
            stratify_col=None,
            label_fracs=label_fracs,
            shuffle=shuffle,
            seed=seed,
        )
        assert obs.equals(obs2)


def test_compute_dataset_splits_stratified_preserves_prevalence():
    # Imbalanced target
    n = 3000
    rng = np.random.default_rng(7)
    y = pd.Series(rng.choice([0, 1], size=n, p=[0.8, 0.2]), name="target")
    df = pd.DataFrame({"x": rng.normal(size=n), "target": y})

    splits = training_params.compute_dataset_splits(
        df,
        stratify_col="target",
        label_fracs={"train": 0.6, "validate": 0.2, "test": 0.2},
        seed=123,
        shuffle=True,
    )

    prev_overall = df["target"].mean()
    prev_by_split = df.groupby(splits)["target"].mean()

    # Each split’s prevalence should be close to overall (within a few points)
    for split_name, prev in prev_by_split.items():
        assert abs(prev - prev_overall) < 0.03  # 3 percentage points


@pytest.mark.parametrize(
    ["df", "target_col", "class_weight", "exp"],
    [
        (
            pd.DataFrame({"target": [1, 1, 1, 0]}),
            "target",
            "balanced",
            pd.Series(
                [0.667, 0.667, 0.667, 2.0], dtype="float32", name="sample_weight"
            ),
        ),
        (
            pd.DataFrame({"target": [1, 1, 1, 0]}),
            "target",
            {1: 2, 0: 0.5},
            pd.Series([2.0, 2.0, 2.0, 0.5], dtype="float32", name="sample_weight"),
        ),
    ],
)
def test_compute_sample_weights(df, target_col, class_weight, exp):
    obs = training_params.compute_sample_weights(
        df, target_col=target_col, class_weight=class_weight
    )
    assert isinstance(obs, pd.Series)
    assert len(obs) == len(df)
    assert pd.testing.assert_series_equal(obs, exp, rtol=0.01) is None
