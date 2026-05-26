"""Call :func:`compute_target` for a validated preprocessing target config (PDP or Edvise)."""

from __future__ import annotations

import typing as t

import pandas as pd

from . import credits_earned, graduation, retention


def compute_target_from_config(
    target_cfg: t.Any,
    df_student_terms: pd.DataFrame,
    df_checkpoint: pd.DataFrame,
    *,
    student_id_col: str,
) -> pd.Series:
    if target_cfg is None:
        raise ValueError("target config is required (cfg.preprocessing.target).")

    target_type = target_cfg.type_
    target_modules = {
        "credits_earned": credits_earned,
        "graduation": graduation,
        "retention": retention,
    }
    if target_type not in target_modules:
        raise ValueError(f"Unknown target type: {target_type}")

    compute_func = target_modules[target_type].compute_target
    kwargs = target_cfg.model_dump()
    kwargs.pop("name", None)
    kwargs.pop("type_", None)
    kwargs["student_id_cols"] = student_id_col
    if target_type == "credits_earned":
        kwargs["checkpoint"] = df_checkpoint

    s = compute_func(df_student_terms, **kwargs)
    if not isinstance(s, pd.Series):
        raise TypeError(f"compute_target must return pd.Series, got {type(s)}")
    return s.astype(bool) if s.dtype != "bool" else s
