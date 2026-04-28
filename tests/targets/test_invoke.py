"""Tests for :mod:`edvise.targets.invoke`."""

from unittest.mock import patch

import pandas as pd

from edvise.configs.pdp import TargetGraduationConfig
from edvise.targets.invoke import compute_target_from_config


def test_compute_target_from_config_passes_project_student_id_col() -> None:
    """``student_id_col`` is forwarded as ``student_id_cols`` to ``compute_target`` ."""
    target = TargetGraduationConfig(
        name="g",
        type_="graduation",
        intensity_time_limits={"*": [4.0, "year"]},
        years_to_degree_col="y",
        max_term_rank="infer",
    )
    df_st = pd.DataFrame()
    df_ckpt = pd.DataFrame()
    with patch("edvise.targets.graduation.compute_target") as m:
        m.return_value = pd.Series(dtype=bool)
        compute_target_from_config(
            target, df_st, df_ckpt, student_id_col="custom_learner_id"
        )
    assert m.call_args.kwargs["student_id_cols"] == "custom_learner_id"
