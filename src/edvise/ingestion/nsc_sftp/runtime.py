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


def ensure_src_on_path(start_dir: str | None = None) -> str | None:
    """Insert repo ``src/`` on ``sys.path`` (needed before ``import edvise`` in jobs)."""
    current = os.path.abspath(start_dir or os.getcwd())
    for _ in range(8):
        if os.path.isdir(os.path.join(current, "edvise")):
            if current not in sys.path:
                sys.path.insert(0, current)
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def parse_spark_python_task_params(argv: list[str] | None = None) -> dict[str, str]:
    """Parse ``--key value`` pairs from ``spark_python_task.parameters``."""
    argv = sys.argv if argv is None else argv
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


_LOGGING_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Enable INFO/WARNING/ERROR on Databricks via shared console logging setup."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    from edvise.shared.logger import configure_console_logging

    configure_console_logging(level=level)
    _LOGGING_CONFIGURED = True


def bootstrap_catalog(argv: list[str] | None = None) -> None:
    """Configure UC catalog paths from job argv / widgets / env."""
    configure_logging()
    configure_nsc_catalog(resolve_nsc_catalog(sys.argv if argv is None else argv))


def get_dbutils() -> Any:
    """Return real dbutils on Databricks; MagicMock locally so scripts import cleanly."""
    return get_dbutils_or_none() or MagicMock()


def get_spark():
    return get_spark_session()


def get_logger(name: str) -> logging.Logger:
    configure_logging()
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


def job_param_bool(
    name: str, default: bool = False, *, argv: list[str] | None = None
) -> bool:
    """Parse a job parameter as a boolean (true/false/1/0/yes/no)."""
    raw = job_param(name, "true" if default else "false", argv=argv).lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Invalid boolean job parameter {name}={raw!r}. Use true/false.")


def require_edvise_api_client(dbutils_obj: Any) -> Any:
    """Build SST API client from parameterized job secret scope/key + DB_workspace."""
    from edvise.ingestion.nsc_sftp.helpers import job_edvise_api_client

    return job_edvise_api_client(
        dbutils_obj,
        db_workspace=require_job_param("DB_workspace"),
        secret_scope=require_job_param("nsc_sftp_secret_scope"),
        sst_api_key_secret_key=require_job_param("sst_api_key_secret_key"),
    )


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
