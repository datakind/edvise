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


def list_tables_older_than(spark: SparkSession, fq_schema: str, days: int) -> list[str]:
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


# --- UC-safe model deletion (replaces stage transition logic) ---
def delete_uc_model_completely(client: MlflowClient, name: str, dry_run: bool) -> None:
    """
    Delete a UC-registered model:
      - remove any aliases
      - delete all versions
      - delete the registered model
    """

    def _log(msg: str) -> None:
        print(f"[cleanup] {msg}", flush=True)

    try:
        rm = client.get_registered_model(name)
    except Exception as e:
        _log(f"Skip (cannot load model '{name}'): {e}")
        return

    # 1) Remove aliases (if the SDK exposes them)
    # MLflow >= 2.9: RegisteredModel has `aliases` (dict[str, str]) mapping alias -> version
    aliases = getattr(rm, "aliases", None)
    if aliases:
        for alias, ver in list(aliases.items()):
            _log(
                f"{'DRY-RUN' if dry_run else 'DELETE'} alias: {name}@{alias} -> v{ver}"
            )
            if not dry_run:
                # MLflow >= 2.9
                try:
                    client.delete_registered_model_alias(name, alias)
                except Exception as e:
                    _log(f"Warn: failed to delete alias {alias} for {name}: {e}")

    # 2) Delete all versions
    try:
        versions = list(client.search_model_versions(f"name='{name}'"))
    except Exception as e:
        _log(f"Warn: cannot list versions for {name}: {e}")
        versions = []

    for v in versions:
        _log(f"{'DRY-RUN' if dry_run else 'DELETE'} model version: {name} v{v.version}")
        if not dry_run:
            try:
                client.delete_model_version(name, v.version)
            except Exception as e:
                _log(f"Warn: delete_model_version failed for {name} v{v.version}: {e}")

    # 3) Delete the registered model
    _log(f"{'DRY-RUN' if dry_run else 'DELETE'} registered model: {name}")
    if not dry_run:
        try:
            client.delete_registered_model(name)
        except Exception as e:
            _log(f"Warn: delete_registered_model failed for {name}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cleanup synthetic UC tables & volumes")
    p.add_argument(
        "--DB_workspace",
        default="dev_sst_02",
        required=True,
        help="UC catalog (e.g., dev_sst_02)",
    )
    p.add_argument(
        "--databricks_institution_name",
        default="synthetic_integration",
        required=True,
        help="Institution slug (e.g., synthetic)",
    )
    p.add_argument(
        "--retention_days",
        type=int,
        default=0,
        help="Only drop tables older than N days (via information_schema if available); 0 = drop all",
    )
    p.add_argument("--dry_run", type=str, default="true", help="true/false")
    p.add_argument("--clean_volumes", type=str, default="true", help="true/false")
    p.add_argument("--delete_models", type=str, default="false", help="true/false")
    p.add_argument("--delete_experiments", type=str, default="false", help="true/false")
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
        candidates: list[str] = list_tables_older_than(
            spark, fq_schema, args.retention_days
        )
        for fqtn in sorted(set(candidates)):
            if fqtn in allowlist:
                log(f"SKIP allowlisted: {fqtn}")
                continue
            drop_table(spark, fqtn, dry_run)

    # 2) (Optional) Clean Volumes
    if clean_volumes:
        # SILVER: delete each immediate subfolder (UUID run IDs, logs, parquet batches) entirely
        silver_root = f"/Volumes/{catalog}/{inst}_silver/silver_volume"
        if path_exists(spark, silver_root):
            children = list_dir(spark, silver_root)
            if not children:
                log(f"Silver volume is already empty: {silver_root}")
            for child in children:
                rm_path(
                    spark, child, recurse=True, dry_run=dry_run
                )  # deletes child + contents
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
            client = MlflowClient()
            for rm in client.search_registered_models():
                name = rm.name  # In UC this is "<catalog>.<schema>.<model_name>"
                if name.startswith(f"{catalog}.{inst}_gold.{inst}"):
                    log(f"DELETE model: {name}")
                    delete_uc_model_completely(client, name, dry_run)
        except Exception as e:
            log(f"Model deletion skipped due to error: {e}")

    # 4) (Optional) Delete experiments by prefix

    delete_experiments: bool = to_bool(getattr(args, "delete_experiments", "false"))

    if delete_experiments and mlflow is not None and MlflowClient is not None:
        try:
            client = MlflowClient()
            contains = (args.experiment_name_contains or "").lower()
            now_millis_utc = int(datetime.now(timezone.utc).timestamp() * 1000)
            cutoff_ms = now_millis_utc - (args.experiment_retention_days * 24 * 60 * 60 * 1000)

            # List experiments (Mlflow may return many; Databricks caps page size—client handles paging)
            exps = client.search_experiments()  # type: ignore[attr-defined]  # Databricks supports this

            for exp in exps:
                name = (exp.name or "")
                if contains not in name.lower():
                    continue

                log(f"Scanning experiment: {name} (id={exp.experiment_id})")

                # Delete old runs first
                try:
                    # Filter on attributes.start_time (ms since epoch)
                    runs = client.search_runs(
                        [exp.experiment_id],
                        filter_string=f"attributes.start_time < {cutoff_ms}",
                        max_results=10000,
                    )
                except Exception:
                    # Fallback: list without filter, then filter in client
                    runs = client.search_runs([exp.experiment_id], max_results=10000)
                    runs = [r for r in runs if r.info.start_time and r.info.start_time < cutoff_ms]

                if not runs:
                    log(f"No runs older than {args.experiment_retention_days}d in: {name}")
                for r in runs:
                    if dry_run:
                        log(f"DRY-RUN: delete run {r.info.run_id} in {name}")
                    else:
                        try:
                            client.delete_run(r.info.run_id)
                        except Exception as e:
                            log(f"Warn: delete_run failed for {r.info.run_id}: {e}")

                # If the experiment has no remaining runs, optionally delete the experiment
                try:
                    remaining = client.search_runs([exp.experiment_id], max_results=1)
                except Exception:
                    remaining = []
                if not remaining:
                    if dry_run:
                        log(f"DRY-RUN: delete experiment {name} (id={exp.experiment_id})")
                    else:
                        try:
                            # Soft delete
                            client.delete_experiment(exp.experiment_id)
                            log(f"Deleted experiment: {name} (soft)")
                        except Exception as e:
                            log(f"Warn: delete_experiment failed for {name}: {e}")

        except Exception as e:
            log(f"Experiment cleanup skipped due to error: {e}")


    log("Cleanup complete" + (" (dry-run)" if dry_run else ""))
