# Cleanup UC tables & volumes for synthetic environments.
# Safe-by-default: dry_run=true unless overridden.

from __future__ import annotations

import argparse
import json
import typing as t
from datetime import datetime, timedelta, timezone

import os
import sys

from pyspark.sql import SparkSession

import mlflow
from mlflow.tracking import MlflowClient


# -------------------------
# Utilities & helpers
# -------------------------

def log(msg: str) -> None:
    print(f"[cleanup] {msg}", flush=True)


def to_bool(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def path_exists(spark: SparkSession, path: str) -> bool:
    """Return True if the given DBFS/Volumes path exists."""
    from pyspark.dbutils import DBUtils  # type: ignore
    dbutils = DBUtils(spark)
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


def list_dir(spark: SparkSession, path: str) -> list[str]:
    """List immediate children (paths) under a DBFS/Volumes directory."""
    from pyspark.dbutils import DBUtils  # type: ignore
    dbutils = DBUtils(spark)
    try:
        entries = dbutils.fs.ls(path)
        # entries have .path attributes; strip trailing slash for consistency
        return [e.path[:-1] if e.path.endswith("/") else e.path for e in entries]
    except Exception:
        return []


def list_tables_in_schema(spark: SparkSession, fq_schema: str) -> list[str]:
    """Return fully qualified table names in a given schema."""
    df = spark.sql(f"SHOW TABLES IN {fq_schema}")
    rows = df.select("tableName").collect()
    return [f"{fq_schema}.{r['tableName']}" for r in rows]


def list_tables_older_than(
    spark: SparkSession, fq_schema: str, days: int
) -> list[str]:
    """
    Return fully-qualified table names in `fq_schema` filtered by creation time if available.
    If information_schema isn't accessible, fall back to returning all tables in the schema.
    """
    try:
        cat, sch = fq_schema.split(".", 1)
        q = f"""
        SELECT CONCAT(table_catalog, '.', table_schema, '.', table_name) AS fqtn,
               created
        FROM {cat}.information_schema.tables
        WHERE table_schema = '{sch}'
        """
        df = spark.sql(q).select("fqtn", "created")
        if days <= 0:
            return [r["fqtn"] for r in df.collect()]
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        out: list[str] = []
        for r in df.collect():
            created = r["created"]
            if created and created < cutoff:
                out.append(r["fqtn"])
        return out
    except Exception:
        # info schema not available — drop everything in that schema
        return list_tables_in_schema(spark, fq_schema)


def drop_table(spark: SparkSession, fqtn: str, dry_run: bool) -> None:
    cmd = f"DROP TABLE IF EXISTS {fqtn}"
    if dry_run:
        log(f"DRY-RUN: {cmd}")
    else:
        log(f"EXEC: {cmd}")
        spark.sql(cmd)


def rm_path(spark: SparkSession, path: str, recurse: bool, dry_run: bool) -> None:
    """Remove a path (file or directory). Does NOT check existence beforehand."""
    from pyspark.dbutils import DBUtils  # type: ignore
    dbutils = DBUtils(spark)
    if dry_run:
        log(f"DRY-RUN: dbutils.fs.rm('{path}', recurse={recurse})")
    else:
        log(f"EXEC: dbutils.fs.rm('{path}', recurse={recurse})")
        dbutils.fs.rm(path, recurse=recurse)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cleanup synthetic UC tables & volumes")
    p.add_argument("--DB_workspace", default="dev_sst_02", required=True, help="UC catalog (e.g., dev_sst_02)")
    p.add_argument(
        "--databricks_institution_name", default="synthetic_integration", required=True, help="Institution slug (e.g., synthetic)"
    )
    p.add_argument(
        "--retention_days", type=int, default=0,
        help="Only drop tables older than N days (via information_schema if available); 0 = drop all"
    )
    p.add_argument("--dry_run", type=str, default="true", help="true/false")
    p.add_argument("--clean_volumes", type=str, default="true", help="true/false")
    p.add_argument("--delete_models", type=str, default="false", help="true/false")
    return p.parse_args()


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    args = parse_args()

    # --- HARD SAFETY ASSERTIONS ---
    inst = args.databricks_institution_name.lower()
    assert "synthetic" in inst, (
        f"❌ Unable to run cleanup: databricks_institution_name='{args.databricks_institution_name}' "
        "does not contain 'synthetic'. Cleanup is restricted to synthetic schemas only."
    )

    catalog = args.DB_workspace.lower()
    assert "dev" in catalog, (
        f"❌ Unable to run cleanup: DB_workspace='{args.DB_workspace}' "
        "does not contain 'dev'. Cleanup is restricted to dev instances only."
    )
    # -------------------------------

    dry_run: bool = to_bool(args.dry_run)
    clean_volumes: bool = to_bool(args.clean_volumes)
    delete_models: bool = to_bool(args.delete_models)

    try:
        allowlist: set[str] = set(json.loads(args.allowlist_tables_json))
    except Exception:
        allowlist = set()

    spark: SparkSession = SparkSession.builder.getOrCreate()

    # 1) Drop UC tables (respect retention & allowlist)
    schemas: list[str] = [
        f"{catalog}.{inst}_bronze",
        f"{catalog}.{inst}_silver",
        f"{catalog}.{inst}_gold",
    ]

    for fq_schema in schemas:
        log(f"Scanning schema: {fq_schema}")
        candidates: list[str] = list_tables_older_than(spark, fq_schema, args.retention_days)
        for fqtn in sorted(set(candidates)):
            if fqtn in allowlist:
                log(f"SKIP allowlisted: {fqtn}")
                continue
            drop_table(spark, fqtn, dry_run)

    # 2) Clean Volumes (optional)
    if clean_volumes:
        # SILVER: delete each immediate subfolder (UUID run IDs, logs, parquet batches) entirely
        silver_root = f"/Volumes/{catalog}/{inst}_silver/silver_volume"
        if path_exists(spark, silver_root):
            children = list_dir(spark, silver_root)
            if not children:
                log(f"Silver volume is already empty: {silver_root}")
            for child in children:
                rm_path(spark, child, recurse=True, dry_run=dry_run)  # deletes child + contents
        else:
            log(f"SKIP (not found): {silver_root}")

        # GOLD: empty known subfolders but keep the folders themselves
        gold_root = f"/Volumes/{catalog}/{inst}_gold/gold_volume"
        gold_subdirs = ["inference_jobs", "model_cards", "training_jobs"]
        if path_exists(spark, gold_root):
            for sub in gold_subdirs:
                subdir = f"{gold_root}/{sub}"
                if not path_exists(spark, subdir):
                    log(f"SKIP (missing): {subdir}")
                    continue
                for child in list_dir(spark, subdir):
                    rm_path(spark, child, recurse=True, dry_run=dry_run)
                log(f"Emptied: {subdir}")
        else:
            log(f"SKIP (not found): {gold_root}")

    # 3) (Optional) Delete models by prefix (dangerous; off by default)
    if delete_models:
        try:
            # In Databricks jobs, tracking URI is preconfigured. If not, you can set:
            # mlflow.set_tracking_uri("databricks")
            client: MlflowClient = MlflowClient()
            found_any = False
            for rm in client.search_registered_models():
                name: str = rm.name
                if name.startswith(f"{catalog}.{inst}_gold.{inst}"):
                    found_any = True
                    log(f"{'DRY-RUN' if dry_run else 'DELETE'} model: {name}")
                    if not dry_run:
                        # Delete versions first (archive stage to be safe)
                        for v in client.search_model_versions(f"name='{name}'"):
                            try:
                                client.transition_model_version_stage(name, v.version, stage="Archived")
                            except Exception as e:
                                log(f"Warn: transition stage failed for {name} v{v.version}: {e}")
                            try:
                                client.delete_model_version(name, v.version)
                            except Exception as e:
                                log(f"Warn: delete version failed for {name} v{v.version}: {e}")
                        # Delete the registered model
                        client.delete_registered_model(name)
            if not found_any:
                log(f"No registered models found with prefix '{inst}'.")
        except Exception as e:
            log(f"Model deletion skipped due to error: {e}")

    log("Cleanup complete" + (" (dry-run)" if dry_run else ""))
