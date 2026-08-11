"""
High-level validation module for our training & inference pipelines.
- Validate expected tables exist
- Validate required columns, no nulls, or any other needed assertion.
- Raise ValueError for hard validation stops.
"""

import logging
import typing as t
import pandas as pd
from pyspark.sql import SparkSession
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedTable:
    path: str
    label: str
    min_rows: t.Optional[int] = 1
    required: bool = True


def require(cond: bool, msg: str, *, exc: type[Exception] = ValueError) -> None:
    """
    Always-on validation guard in our pipeline. We utilize this over 'assert'
    since assert statements are skipped when Python is run with optimization (python -O).
    """
    if not cond:
        raise exc(msg)


def require_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    require(not missing, f"{label}: missing required columns: {missing}")


def require_no_nulls(df: pd.DataFrame, cols: list[str], label: str) -> None:
    for c in cols:
        require(c in df.columns, f"{label}: missing required column: {c}")
        nulls = int(df[c].isna().sum())
        require(nulls == 0, f"{label}: {c} has {nulls} null values.")


def require_attr(obj: t.Any, attr: str, msg: str) -> t.Any:
    value = getattr(obj, attr, None)
    require(value is not None, msg)
    return value


def warn_if(cond: bool, msg: str, logger: logging.Logger | None = None) -> None:
    """Soft validation; logs a warning."""
    if cond:
        (logger or logging.getLogger(__name__)).warning(msg)


def is_boolean_like(series: pd.Series) -> bool:
    """True if non-null values are boolean or in {0, 1, True, False}."""
    if pd.api.types.is_bool_dtype(series):
        return True
    valid = series.dropna()
    return (not valid.empty) and bool(valid.isin([0, 1, True, False]).all())


def validate_tables_exist(spark: SparkSession, tables: list[ExpectedTable]) -> None:
    for table in tables:
        ok = False
        try:
            ok = bool(spark.catalog.tableExists(table.path))
        except Exception:
            ok = False

        msg_missing = f"Missing expected table [{table.label}]: {table.path}"
        (require if table.required else warn_if)(ok, msg_missing)

        if not ok:
            continue

        try:
            if table.min_rows is None:
                spark.sql(f"SELECT 1 FROM {table.path} LIMIT 1").collect()
            else:
                n = spark.sql(f"SELECT COUNT(1) AS n FROM {table.path}").collect()[0][
                    "n"
                ]
                msg_rows = (
                    f"Table [{table.label}] has {n} rows (<{table.min_rows}): "
                    f"{table.path}"
                )
                (require if table.required else warn_if)(n >= table.min_rows, msg_rows)
        except Exception as e:
            raise RuntimeError(
                f"Table exists but is not queryable [{table.label}]: {table.path}. "
                f"Error: {e}"
            )
