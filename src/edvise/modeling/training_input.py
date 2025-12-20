from __future__ import annotations

import os
import logging
import typing as t

import pandas as pd

from edvise.shared.logger import local_fs_path
from edvise.shared.validation import require

SchemaType = t.Literal["pdp", "custom"]

def load_or_build_modeling_df(
    *,
    schema_type: SchemaType,
    current_run_path: str,
    cfg: t.Any,
    feature_selection_fn: t.Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Returns the modeling dataset used for training.

    PDP:
      - prefer modeling.parquet if present
      - else read preprocessed.parquet -> feature selection -> write modeling.parquet

    Custom:
      - require modeling.parquet (created by the custom FE notebook)
      - do NOT run feature selection here
    """
    modeling_path = os.path.join(current_run_path, "modeling.parquet")
    modeling_path_local = local_fs_path(modeling_path)

    if os.path.exists(modeling_path_local):
        logging.info("Loading modeling.parquet: %s", modeling_path)
        df_modeling = pd.read_parquet(modeling_path_local)
        validate_modeling_contract(df_modeling=df_modeling, cfg=cfg, schema_type=schema_type)
        return df_modeling

    if schema_type == "custom":
        raise FileNotFoundError(
            "Custom training expects modeling.parquet to already exist "
            "(produced by the custom feature engineering notebook). "
            f"Missing: {modeling_path}"
        )

    # PDP fallback path
    require(feature_selection_fn is not None, "PDP requires feature_selection_fn")

    preproc_path = os.path.join(current_run_path, "preprocessed.parquet")
    preproc_path_local = local_fs_path(preproc_path)
    if not os.path.exists(preproc_path_local):
        raise FileNotFoundError(
            f"Missing preprocessed.parquet at: {preproc_path} (local: {preproc_path_local})"
        )

    logging.info("PDP: loading preprocessed.parquet: %s", preproc_path)
    df_preprocessed = pd.read_parquet(preproc_path_local)

    logging.info("PDP: running feature selection")
    df_modeling = feature_selection_fn(df_preprocessed)

    os.makedirs(local_fs_path(current_run_path), exist_ok=True)
    df_modeling.to_parquet(modeling_path_local, index=False)
    logging.info("Saved modeling.parquet: %s", modeling_path)

    validate_modeling_contract(df_modeling=df_modeling, cfg=cfg, schema_type=schema_type)
    return df_modeling


def validate_modeling_contract(*, df_modeling: pd.DataFrame, cfg: t.Any, schema_type: SchemaType) -> None:
    """
    Enforce the modeling dataset contract so training/inference/model cards stay consistent.
    """
    require(df_modeling is not None and not df_modeling.empty, "modeling df is empty")
    require(df_modeling.columns.is_unique, "modeling df has duplicate columns")

    required_cols = [cfg.student_id_col, cfg.target_col]
    if cfg.split_col:
        required_cols.append(cfg.split_col)

    missing = [c for c in required_cols if c not in df_modeling.columns]
    require(not missing, f"modeling df missing required columns: {missing}")

    if cfg.split_col:
        allowed = {"train", "test", "validate"}
        bad = sorted(set(df_modeling[cfg.split_col].dropna().unique()) - allowed)
        require(not bad, f"Unexpected split values in '{cfg.split_col}': {bad}")

    non_feature_cols = set(cfg.non_feature_cols)
    feature_cols = [c for c in df_modeling.columns if c not in non_feature_cols]
    require(len(feature_cols) > 0, "No feature columns found after excluding non_feature_cols")

    if schema_type == "custom":
        forbidden = set(cfg.student_group_cols or [])
        leak = sorted([c for c in feature_cols if c in forbidden])
        require(not leak, f"Student-group columns leaked into features: {leak}")
