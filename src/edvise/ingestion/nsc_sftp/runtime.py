"""Shared bootstrap for NSC SFTP spark_python_task scripts."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any
from unittest.mock import MagicMock

from edvise.ingestion.nsc_sftp.constants import configure_nsc_catalog
from edvise.utils.databricks import (
    get_db_widget_param,
    get_dbutils_or_none,
    get_spark_session,
)


def parse_spark_python_task_params(argv: list[str] | None = None) -> dict[str, str]:
    """Parse ``--key value`` pairs from ``spark_python_task.parameters``."""
    if argv is None:
        argv = sys.argv
    out: dict[str, str] = {}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and i + 1 < len(argv):
            out[a[2:].replace("-", "_")] = argv[i + 1]
            i += 2
        else:
            i += 1
    return out


def resolve_nsc_catalog(argv: list[str] | None = None) -> str:
    """
    Resolve Unity Catalog name: ``--DB_workspace`` → widget → ``NSC_DB_WORKSPACE``
    → default catalog for local imports.
    """
    from edvise.ingestion.nsc_sftp.constants import DEFAULT_CATALOG_FOR_LOCAL

    pairs = parse_spark_python_task_params(argv)
    raw = pairs.get("DB_workspace", "").strip()
    if raw:
        return raw
    try:
        w = get_db_widget_param("DB_workspace", default="")
        if str(w).strip():
            return str(w).strip()
    except Exception:
        pass
    return os.environ.get("NSC_DB_WORKSPACE", "").strip() or DEFAULT_CATALOG_FOR_LOCAL


def bootstrap_catalog(argv: list[str] | None = None) -> None:
    """Configure UC catalog paths from job argv / widgets / env."""
    configure_nsc_catalog(resolve_nsc_catalog(sys.argv if argv is None else argv))


def get_dbutils() -> Any:
    """Return real dbutils on Databricks; MagicMock locally so scripts import cleanly."""
    return get_dbutils_or_none() or MagicMock()


def get_spark():
    return get_spark_session()


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def job_param(name: str, default: str = "", *, argv: list[str] | None = None) -> str:
    """Resolve a job parameter: ``spark_python_task`` argv, then notebook widget."""
    pairs = parse_spark_python_task_params(sys.argv if argv is None else argv)
    if name in pairs:
        return str(pairs[name]).strip()
    try:
        return str(get_db_widget_param(name, default=default)).strip()
    except Exception:
        return str(default).strip()


def require_job_param(name: str, *, argv: list[str] | None = None) -> str:
    value = job_param(name, "", argv=argv)
    if not value:
        raise ValueError(
            f"Missing required job parameter {name}. "
            "Pass it via DAB var / job parameter at deploy or run time."
        )
    return value


def notebook_exit(dbutils_obj: Any, message: str) -> None:
    """Exit a Databricks task; raise SystemExit locally for testability."""
    try:
        dbutils_obj.notebook.exit(message)
    except Exception:
        raise SystemExit(message) from None


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
