"""
High-level validation module for our training & inference pipelines.
- Validate expected tables exist
- Validate required columns, no nulls, or any other needed assertion.
- Raise ValueError for hard validation stops.
"""

import logging
import typing as t
import pandas as pd

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


def warn_if(cond: bool, msg: str, logger: logging.Logger | None = None) -> None:
    """Soft validation; logs a warning."""
    if cond:
        (logger or logging.getLogger(__name__)).warning(msg)


def validate_tables_exist(spark, tables: list[ExpectedTable]) -> None:
    for t in tables:
        ok = False
        try:
            ok = bool(spark.catalog.tableExists(t.path))
        except Exception:
            ok = False

        msg_missing = f"Missing expected table [{t.label}]: {t.path}"
        (require if t.required else warn_if)(ok, msg_missing)

        if not ok:
            continue

        try:
            if t.min_rows is None:
                spark.sql(f"SELECT 1 FROM {t.path} LIMIT 1").collect()
            else:
                n = spark.sql(f"SELECT COUNT(1) AS n FROM {t.path}").collect()[0]["n"]
                msg_rows = f"Table [{t.label}] has {n} rows (<{t.min_rows}): {t.path}"
                (require if t.required else warn_if)(n >= t.min_rows, msg_rows)
        except Exception as e:
            raise RuntimeError(
                f"Table exists but is not queryable [{t.label}]: {t.path}. Error: {e}"
            )
