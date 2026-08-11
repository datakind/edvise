import numpy as np
import pandas as pd

from edvise.model_prep.target_associations import (
    compute_mutual_infos,
    compute_spearman_corrs,
    compute_univariate_aucs,
    feature_family,
    suggest_force_include_cols,
)


def test_feature_family_collapses_frac_num_cum_prefixes() -> None:
    assert feature_family("frac_courses_course_grade_f") == "course_grade_f"
    assert feature_family("num_courses_course_grade_f") == "course_grade_f"
    assert feature_family("cumfrac_num_courses_course_grade_f") == "course_grade_f"
    assert feature_family("cummin_num_credits_earned") == "credits_earned"


def test_univariate_auc_and_suggest_prefer_discriminative_feature() -> None:
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    # Higher signal => more likely positive
    target = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
    df = pd.DataFrame(
        {
            "target": target,
            "frac_courses_passed": signal,
            "num_courses_passed": signal + 0.01,
            "noise_feature": noise,
            "student_id": np.arange(n),
        }
    )
    exclude = {"target", "student_id"}
    corrs = compute_spearman_corrs(df, target_col="target", exclude=exclude)
    aucs = compute_univariate_aucs(df, target_col="target", exclude=exclude)
    mis = compute_mutual_infos(df, target_col="target", exclude=exclude, random_state=0)

    assert aucs["frac_courses_passed"] > 0.6
    assert aucs["frac_courses_passed"] >= aucs["noise_feature"]
    assert mis["frac_courses_passed"] >= mis["noise_feature"]

    suggest = suggest_force_include_cols(
        corrs,
        aucs,
        mis=mis,
        exclude=exclude,
        n_each=3,
    )
    assert suggest
    assert suggest[0] in {"frac_courses_passed", "num_courses_passed"}
    # Family collapse: only one of frac_/num_ courses_passed
    assert len([c for c in suggest if feature_family(c) == "passed"]) == 1
