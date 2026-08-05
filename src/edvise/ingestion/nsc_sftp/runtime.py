"""Shared bootstrap for NSC SFTP spark_python_task scripts."""

from __future__ import annotations

import logging
import sys
from typing import Any
from unittest.mock import MagicMock

from edvise.ingestion.nsc_sftp.constants import (
    configure_nsc_catalog,
    parse_spark_python_task_params,
    resolve_nsc_catalog,
)


def bootstrap_catalog(argv: list[str] | None = None) -> None:
    """Configure UC catalog paths from job argv / widgets / env."""
    configure_nsc_catalog(resolve_nsc_catalog(sys.argv if argv is None else argv))


def get_dbutils() -> Any:
    try:
        return dbutils  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        return MagicMock()


def get_spark():
    from databricks.connect import DatabricksSession

    return DatabricksSession.builder.getOrCreate()


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def job_param(name: str, default: str = "", *, argv: list[str] | None = None) -> str:
    """Resolve a job/widget parameter as a stripped string."""
    from edvise import utils

    pairs = parse_spark_python_task_params(sys.argv if argv is None else argv)
    return str(
        utils.databricks.get_db_widget_param(name, default=pairs.get(name, default))
    ).strip()


def workflow_run_id(dbutils_obj: Any) -> str | None:
    try:
        ctx = dbutils_obj.notebook.entry_point.getDbutils().notebook().getContext()
        tags = ctx.tags()
        for key in ("jobRunId", "runId"):
            try:
                value = tags.apply(key)
                if value:
                    return str(value)
            except Exception:
                pass
        try:
            value = ctx.currentRunId().get()
            if value:
                return str(value)
        except Exception:
            pass
    except Exception:
        pass
    return None
