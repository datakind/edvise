import logging
import os
import sys
import typing as t
from urllib.parse import quote

import requests

# Ensure repo src/ is on sys.path so `import edvise.*` works in Databricks Jobs.
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

LOGGER = logging.getLogger(__name__)


try:
    # Preferred: reuse the canonical transformation logic already in the repo.
    from edvise.utils.databricks import databricksify_inst_name  # type: ignore
except Exception as e:  # pragma: no cover
    # Fallback for environments where importing the full edvise.utils.databricks module
    # is not possible (e.g., missing databricks-connect runtime deps).
    LOGGER.warning(
        "Unable to import edvise.utils.databricks.databricksify_inst_name; using "
        "fallback implementation (%s)",
        e,
    )
    raise ImportError(
        "databricksify_inst_name is required but could not be imported. Ensure that "
        "the edvise.utils.databricks module is available and that databricks-connect is "
        "properly configured if running outside of Databricks."
    ) from e


def _get_spark_session():
    try:
        from pyspark.sql import SparkSession  # type: ignore

        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception as e:
        raise RuntimeError(
            "SparkSession is not available. Run this script in a Databricks environment "
            "or in an environment with pyspark configured."
        ) from e


def _headers(access_token: str) -> dict[str, str]:
    return {
        "accept": "application/json",
        "authorization": f"Bearer {access_token}",
    }


def _list_institutions_by_name(
    *,
    session: requests.Session,
    base_url: str,
    access_token: str,
    name_query: str,
    timeout_s: float = 30.0,
) -> list[dict[str, t.Any]]:
    normalized = (name_query or "").strip().lower()
    encoded_name = quote(normalized, safe="")
    url = f"{base_url.rstrip('/')}/api/v1/institutions/name/{encoded_name}"
    resp = session.get(url, headers=_headers(access_token), timeout=timeout_s)
    resp.raise_for_status()

    data = resp.json()
    if isinstance(data, list):
        return t.cast(list[dict[str, t.Any]], data)
    raise TypeError(f"Expected list from {url}; got {type(data).__name__}: {data}")


def _fetch_institution(
    *,
    session: requests.Session,
    base_url: str,
    access_token: str,
    institution_id: str,
    timeout_s: float = 30.0,
) -> dict[str, t.Any]:
    url = f"{base_url.rstrip('/')}/api/v1/institutions/{institution_id}"
    resp = session.get(url, headers=_headers(access_token), timeout=timeout_s)
    resp.raise_for_status()

    data = resp.json()
    if isinstance(data, dict):
        return t.cast(dict[str, t.Any], data)
    raise TypeError(f"Expected dict from {url}; got {type(data).__name__}: {data}")


def sync_institutions_table(
    *,
    DB_workspace: str,
    sst_access_token: str,
    sst_base_url: str = "https://staging-sst.datakind.org",
    name_query: str = "",
    schema: str = "default",
    table: str = "institutions",
    write_mode: str = "overwrite",
    spark=None,
) -> str:
    """
    Fetch institutions from the SST API and write them as a UC Delta table.

    Returns:
        The fully-qualified UC table name written to: "<DB_workspace>.<schema>.<table>"
    """
    if not isinstance(DB_workspace, str) or not DB_workspace.strip():
        raise ValueError("DB_workspace must be a non-empty string")
    if not isinstance(sst_access_token, str) or not sst_access_token.strip():
        raise ValueError(
            "sst_access_token must be a non-empty string (Bearer token value)"
        )

    session = requests.Session()

    inst_list = _list_institutions_by_name(
        session=session,
        base_url=sst_base_url,
        access_token=sst_access_token,
        name_query=name_query,
    )

    rows_by_id: dict[str, dict[str, t.Any]] = {}
    for inst in inst_list:
        inst_id = inst.get("inst_id")
        if not inst_id:
            continue

        # We rely on the details endpoint for pdp_id (and treat it as authoritative).
        details = _fetch_institution(
            session=session,
            base_url=sst_base_url,
            access_token=sst_access_token,
            institution_id=str(inst_id),
        )
        name = details.get("name") or inst.get("name")
        if not name:
            continue
        pdp_id = details.get("pdp_id")

        rows_by_id[str(inst_id)] = {
            "institution_id": str(inst_id),
            "institution_name": str(name),
            "pdp_id": None if pdp_id is None else str(pdp_id),
            "databricks_institution_name": databricksify_inst_name(str(name)),
        }

    rows = list(rows_by_id.values())
    LOGGER.info("Fetched %s institutions; writing %s rows", len(inst_list), len(rows))

    if spark is None:
        spark = _get_spark_session()
    df = spark.createDataFrame(rows)

    table_path = f"{DB_workspace}.{schema}.{table}"
    (
        df.write.format("delta")
        .mode(write_mode)
        .option("overwriteSchema", "true")
        .saveAsTable(table_path)
    )
    LOGGER.info("Wrote institutions table: %s", table_path)
    return table_path
