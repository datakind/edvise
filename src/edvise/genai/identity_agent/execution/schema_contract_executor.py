"""Merge grain into school config and build frozen GenAI schema contracts via SMA preprocessing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Optional

import pandas as pd

from edvise.configs.custom import CleaningConfig
from edvise.configs.genai import DatasetConfig, SchoolMappingConfig
from edvise.data_audit.custom_cleaning import DtypeGenerationOptions, TermOrderFn
from edvise.genai.identity_agent.grain_inference.schemas import IdentityGrainContract

logger = logging.getLogger(__name__)


def merge_grain_contracts_into_school_config(
    school_config: SchoolMappingConfig,
    grain_contracts_by_dataset: dict[str, IdentityGrainContract],
    *,
    dataset_name_suffix: str = "",
) -> SchoolMappingConfig:
    """
    Return a copy of ``school_config`` with ``primary_keys`` overridden from grain contracts.

    Only datasets **present** in ``grain_contracts_by_dataset`` are updated; others keep
    ``inputs.toml`` primary keys.

    Args:
        school_config: Loaded :class:`~edvise.configs.genai.SchoolMappingConfig`.
        grain_contracts_by_dataset: Map **dataset name** (same keys as
            ``school_config.datasets``, i.e. inputs.toml table names) to the approved
            :class:`IdentityGrainContract` for that table.
        dataset_name_suffix: Same suffix you pass to ``build_schema_contract_from_config``
            (used only to log a warning if ``contract.table`` does not match the logical name).

    Returns:
        New ``SchoolMappingConfig`` with updated ``DatasetConfig.primary_keys`` where provided.
    """
    unknown = set(grain_contracts_by_dataset) - set(school_config.datasets)
    if unknown:
        raise KeyError(
            "grain_contracts_by_dataset has unknown dataset names (not in school_config): "
            f"{sorted(unknown)}"
        )

    datasets: dict[str, DatasetConfig] = {}
    for name, dc in school_config.datasets.items():
        gc = grain_contracts_by_dataset.get(name)
        if gc is None:
            datasets[name] = dc
            continue

        if gc.institution_id != school_config.institution_id:
            logger.warning(
                "Grain contract institution_id %r != school_config %r for dataset %r",
                gc.institution_id,
                school_config.institution_id,
                name,
            )
        logical = f"{name}{dataset_name_suffix}" if dataset_name_suffix else name
        if gc.table not in (name, logical):
            logger.warning(
                "Grain contract table %r does not match dataset name %r or logical %r",
                gc.table,
                name,
                logical,
            )

        uks = list(gc.unique_keys)
        if not uks:
            raise ValueError(
                f"Grain contract for dataset {name!r} has empty unique_keys / post_clean_primary_key"
            )
        datasets[name] = dc.model_copy(update={"primary_keys": uks})

    return school_config.model_copy(update={"datasets": datasets})


def build_schema_contract_from_grain_contracts(
    school_config: SchoolMappingConfig,
    grain_contracts_by_dataset: dict[str, IdentityGrainContract],
    *,
    dtype_opts: Optional[DtypeGenerationOptions] = None,
    spark_session: Optional[Any] = None,
    dataset_name_suffix: str = "",
    sample_size: Optional[int] = None,
    cleaning_cfg: Optional[CleaningConfig] = None,
    term_order_fn: Optional[TermOrderFn] = None,
    term_col_by_dataset: Optional[dict[str, str]] = None,
    dedupe_fn_by_dataset: Optional[
        dict[str, Callable[[pd.DataFrame], pd.DataFrame]]
    ] = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Build cleaned frames and a frozen schema contract, with primary keys taken from grain contracts.

    Applies :func:`merge_grain_contracts_into_school_config` then
    :func:`edvise.genai.schema_mapping_agent.preprocessing.build_schema_contract_from_config`.

    Args:
        school_config: School mapping config (paths, cleaning, baseline primary_keys).
        grain_contracts_by_dataset: Per-dataset grain contracts. Keys are **dataset names**
            matching ``school_config.datasets``.
        Remaining kwargs: forwarded to preprocessing.

    Returns:
        ``(cleaned_dataframes_by_logical_name, schema_contract_dict)`` — same as preprocessing.
    """
    from edvise.genai.schema_mapping_agent.preprocessing import (
        build_schema_contract_from_config,
    )

    merged = merge_grain_contracts_into_school_config(
        school_config,
        grain_contracts_by_dataset,
        dataset_name_suffix=dataset_name_suffix,
    )
    return build_schema_contract_from_config(
        merged,
        dtype_opts=dtype_opts,
        spark_session=spark_session,
        dataset_name_suffix=dataset_name_suffix,
        sample_size=sample_size,
        cleaning_cfg=cleaning_cfg,
        term_order_fn=term_order_fn,
        term_col_by_dataset=term_col_by_dataset,
        dedupe_fn_by_dataset=dedupe_fn_by_dataset,
    )
