"""Orchestration: one lightweight LLM call per dataset table."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import pandas as pd

from edvise.configs.genai import SchoolMappingConfig
from edvise.data_audit.custom_cleaning import CleaningConfig
from edvise.utils.llm_utils import LLMRetryExhausted, llm_complete_with_parse_retry

from edvise.genai.mapping.identity_agent.dataset_io import load_school_dataset_dataframe
from edvise.genai.mapping.identity_agent.profiling.constants import (
    INDEX_COLUMN_PATTERNS,
)
from edvise.genai.mapping.identity_agent.profiling.raw_snapshot import profile_raw_table

from .fallback import apply_column_role_fallbacks
from .prompt import (
    COLUMN_ROLES_SYSTEM_PROMPT,
    build_column_roles_user_message,
    parse_column_roles_response,
)
from .schemas import ColumnRolesResult

logger = logging.getLogger(__name__)


def _prepare_dataframe_for_column_roles(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    index_cols = [c for c in work.columns if INDEX_COLUMN_PATTERNS.match(c)]
    if index_cols:
        work = work.drop(columns=index_cols)
    return work.drop_duplicates().reset_index(drop=True)


def run_column_roles_for_dataset(
    *,
    institution_id: str,
    dataset: str,
    df: pd.DataFrame,
    llm_complete: Callable[[str, str], str],
    cleaning: CleaningConfig | None = None,
) -> ColumnRolesResult:
    """
    Classify columns via LLM using a raw table profile, then apply deterministic fallbacks.
    """
    prepared = _prepare_dataframe_for_column_roles(df)
    raw_table_profile = profile_raw_table(
        prepared,
        institution_id=institution_id,
        dataset=dataset,
        cleaning=cleaning,
    )
    user = build_column_roles_user_message(institution_id, dataset, raw_table_profile)
    expected_columns = [c.name for c in raw_table_profile.columns]

    def _parse(
        raw: str,
        *,
        validate_completeness: bool = True,
    ) -> ColumnRolesResult:
        parsed = parse_column_roles_response(
            raw,
            institution_id=institution_id,
            dataset=dataset,
            expected_columns=expected_columns,
            validate_completeness=validate_completeness,
        )
        return apply_column_role_fallbacks(parsed, columns=expected_columns)

    def _parse_strict(raw: str) -> ColumnRolesResult:
        return _parse(raw, validate_completeness=True)

    try:
        result = llm_complete_with_parse_retry(
            llm_complete,
            COLUMN_ROLES_SYSTEM_PROMPT,
            user,
            _parse_strict,
            logger=logger,
        )
    except LLMRetryExhausted as exc:
        logger.warning(
            "[column_roles] dataset=%s LLM retries exhausted (%s); "
            "assigning missing columns to other",
            dataset,
            exc.last_error,
        )
        result = _parse(exc.last_raw_response, validate_completeness=False)
    logger.info(
        "[column_roles] dataset=%s file_kind=%s learner_id=%s low_confidence=%s warnings=%d",
        dataset,
        result.file_kind.value,
        result.learner_id_column(),
        result.low_confidence_columns,
        len(result.profiler_warnings),
    )
    return result


def run_column_roles_for_institution(
    *,
    institution_id: str,
    school: SchoolMappingConfig,
    llm_complete: Callable[[str, str], str],
) -> dict[str, ColumnRolesResult]:
    """Run :func:`run_column_roles_for_dataset` for each configured dataset."""
    results: dict[str, ColumnRolesResult] = {}
    for name in school.datasets.keys():
        df = load_school_dataset_dataframe(school, name)
        results[name] = run_column_roles_for_dataset(
            institution_id=institution_id,
            dataset=name,
            df=df,
            llm_complete=llm_complete,
            cleaning=school.cleaning,
        )
    return results


def column_roles_by_dataset_to_jsonable(
    institution_id: str,
    roles_by_dataset: Mapping[str, ColumnRolesResult],
) -> dict[str, object]:
    return {
        "institution_id": institution_id,
        "datasets": {name: r.to_jsonable() for name, r in roles_by_dataset.items()},
    }
