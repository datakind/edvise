"""Runtime compatibility checks for archived versioned inference bundles."""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)

BUNDLE_ARCHIVED_DAB_HINT = (
    "Versioned inference cannot run on this cluster/runtime. "
    "Use the archived Databricks bundle for this pipeline_version."
)

_DBR_MAJOR_MINOR = re.compile(r"^(\d+)\.(\d+)")


def dbr_major_minor(version: str) -> tuple[int, int] | None:
    m = _DBR_MAJOR_MINOR.match(version.strip().lower())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def databricks_runtime_compatible(required: str, current: str) -> bool:
    req = required.strip().lower()
    cur = current.strip().lower()
    if req == cur:
        return True
    req_mm = dbr_major_minor(req)
    cur_mm = dbr_major_minor(cur)
    if req_mm and cur_mm and req_mm == cur_mm:
        return True
    return req.startswith(f"{cur}.") or req.startswith(f"{cur}-")


def parse_python_xy(spec: str) -> tuple[int, int] | None:
    s = spec.strip()
    parts = s.replace(" ", "").split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def current_databricks_runtime_version() -> str | None:
    v = os.environ.get("DATABRICKS_RUNTIME_VERSION")
    return v.strip() if isinstance(v, str) and v.strip() else None


def current_spark_version(spark: Any) -> str | None:
    try:
        v = spark.version
        return str(v).strip() if v else None
    except Exception:
        return None


def check_runtime_bundle_compatibility(
    effective: Mapping[str, Any],
    *,
    spark: Any,
    logger: logging.Logger = LOGGER,
) -> tuple[bool, str]:
    mode = effective.get("execution_mode")
    if isinstance(mode, str) and mode.strip().lower() == "dab":
        return (
            False,
            "This bundle declares execution_mode=dab; use archived DAB job for this "
            "release. " + BUNDLE_ARCHIVED_DAB_HINT,
        )

    rr = effective.get("required_runtime")
    if not isinstance(rr, dict):
        return True, ""

    req_py = rr.get("python")
    if isinstance(req_py, str) and req_py.strip():
        want = parse_python_xy(req_py)
        got = sys.version_info[:2]
        if want and got != want:
            return (
                False,
                f"Bundle requires Python {req_py}; driver is {got[0]}.{got[1]}. "
                + BUNDLE_ARCHIVED_DAB_HINT,
            )

    req_dbr = rr.get("databricks_runtime")
    if isinstance(req_dbr, str) and req_dbr.strip():
        cur = current_databricks_runtime_version()
        if not cur:
            logger.warning(
                "Bundle requires databricks_runtime=%r but DATABRICKS_RUNTIME_VERSION "
                "is unset; skipping DBR check.",
                req_dbr,
            )
        elif not databricks_runtime_compatible(req_dbr, cur):
            return (
                False,
                f"Bundle requires DBR {req_dbr!r}; current cluster is {cur!r}. "
                + BUNDLE_ARCHIVED_DAB_HINT,
            )
        logger.info("DBR compatibility OK (bundle=%r, cluster=%r)", req_dbr, cur)

    req_spark = rr.get("spark")
    if isinstance(req_spark, str) and req_spark.strip():
        cur_sp = current_spark_version(spark)
        if cur_sp and cur_sp.strip() != req_spark.strip():
            return (
                False,
                f"Bundle requires Spark {req_spark!r}; active Spark is {cur_sp!r}. "
                + BUNDLE_ARCHIVED_DAB_HINT,
            )

    return True, ""
