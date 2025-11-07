# Cleanup UC tables & volumes for synthetic environments.
# Safe-by-default: dry_run=true unless overridden.

import argparse
import json
from datetime import datetime, timedelta

from pyspark.sql import SparkSession

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None


def log(msg: str) -> None:
    print(f"[cleanup] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Cleanup synthetic UC tables & volumes")
    p.add_argument("--DB_workspace", required=True, help="UC catalog (e.g., dev_sst_02)")
    p.add_argument("--databricks_institution_name", required=True, help="Institution slug (e.g., synthetic)")
    p.add_argument("--retention_days", type=int, default=0, help="Only drop objects older than N days; 0 = drop all")
    p.add_argument("--dry_run", type=str, default="true", help="true/false")
    p.add_argument("--clean_volumes", type=str, default="true", help="true/false")
    p.add_argument("--delete_models", type=str, default="false", help="true/false (dangerous)")
    p.add_argument("--model_name_prefix", type=str, default="synthetic_", help="prefix for model deletion")
    p.add_argument("--allowlist_tables_json", type=str, default="[]", help='JSON list of fully qualified tables to KEEP')
    return p.parse_args()


def to_bool(s: str) -> bool:
    return str(s).lower() in {"1", "true", "yes", "y"}


def list_tables_in_schema(spark: SparkSession, fq_schema: str):
    df = spark.sql(f"SHOW TABLES IN {fq_schema}")
    rows = df.select("tableName").collect()
    return [f"{fq_schema}.{r['tableName']}" for r in rows]


def list_tables_older_than(spark: SparkSession, fq_schema: str, days: int):
    # Prefer information_schema to filter by created timestamp; fallback otherwise
    try:
        q = f"""
        SELECT CONCAT(table_catalog, '.', table_schema, '.', table_name) AS fqtn,
               created
        FROM {fq_schema.split('.')[0]}.information_schema.tables
        WHERE table_schema = '{fq_schema.split('.')[1]}'
        """
        df = spark.sql(q)
        if days <= 0:
            return [r["fqtn"] for r in df.collect()]
        cutoff = datetime.utcnow() - timedelta(days=days)
        return [r["fqtn"] for r in df.collect() if r["created"] and r["created"] < cutoff]
    except Exception:
        # info schema not available — drop everything in that schema (or rely on SHOW TABLES)
        return list_tables_in_schema(spark, fq_schema)


def drop_table(spark: SparkSession, fqtn: str, dry_run: bool):
    cmd = f"DROP TABLE IF EXISTS {fqtn}"
    if dry_run:
        log(f"DRY-RUN: {cmd}")
    else:
        log(f"EXEC: {cmd}")
        spark.sql(cmd)


def rm_path(spark: SparkSession, path: str, recursive: bool, dry_run: bool):
    from pyspark.dbutils import DBUtils  # type: ignore
    dbutils = DBUtils(spark)
    if dry_run:
        log(f"DRY-RUN: dbutils.fs.rm('{path}', recursive={recursive})")
    else:
        log(f"EXEC: dbutils.fs.rm('{path}', recursive={recursive})")
        dbutils.fs.rm(path, recursive=recursive)


def main():
    args = parse_args()

    # --- HARD SAFETY ASSERTION ---
    inst = args.databricks_institution_name.lower()
    assert "synthetic" in inst, (
        f"❌ Unable to run cleanup: databricks_institution_name='{args.databricks_institution_name}' "
        "does not contain 'synthetic'. Cleanup is restricted to synthetic schemas only."
    )
    # ------------------------------

    catalog = args.DB_workspace.lower()
    assert "dev" in inst, (
        f"❌ Unable to run cleanup: DB_workspace='{args.databricks_institution_name}' "
        "does not contain 'dev'. Cleanup is restricted to dev instances only."
    )

    dry_run = to_bool(args.dry_run)
    clean_volumes = to_bool(args.clean_volumes)
    delete_models = to_bool(args.delete_models)

    try:
        allowlist = set(json.loads(args.allowlist_tables_json))
    except Exception:
        allowlist = set()

    spark = SparkSession.builder.getOrCreate()

    schemas = [
        f"{catalog}.{inst}_bronze",
        f"{catalog}.{inst}_silver",
        f"{catalog}.{inst}_gold",
    ]

    # 1) Drop tables (respect retention & allowlist)
    for fq_schema in schemas:
        log(f"Scanning schema: {fq_schema}")
        candidates = list_tables_older_than(spark, fq_schema, args.retention_days)
        for fqtn in sorted(set(candidates)):
            if fqtn in allowlist:
                log(f"SKIP allowlisted: {fqtn}")
                continue
            drop_table(spark, fqtn, dry_run)

    # 2) Clean Volumes (optional)
    if clean_volumes:
        volume_paths = [
            f"/Volumes/{catalog}/{inst}_bronze/bronze_volume",
            f"/Volumes/{catalog}/{inst}_silver/silver_volume",
            f"/Volumes/{catalog}/{inst}_gold/gold_volume/inference_jobs",
            f"/Volumes/{catalog}/{inst}_gold/gold_volume/training_jobs",
        ]
        for p in volume_paths:
            rm_path(spark, p, recursive=True, dry_run=dry_run)

    # 3) (Optional) Delete registered models by prefix (dangerous; off by default)
    if delete_models and mlflow is not None and MlflowClient is not None:
        try:
            client = MlflowClient()
            for rm in client.search_registered_models():
                name = rm.name
                if name.startswith(args.model_name_prefix):
                    log(f"{'DRY-RUN' if dry_run else 'DELETE'} model: {name}")
                    if not dry_run:
                        for v in client.search_model_versions(f"name='{name}'"):
                            # archive & remove versions first
                            client.transition_model_version_stage(name, v.version, stage="Archived")
                            client.delete_model_version(name, v.version)
                        client.delete_registered_model(name)
        except Exception as e:
            log(f"Model deletion skipped due to error: {e}")

    log("Cleanup complete" + (" (dry-run)" if dry_run else ""))


if __name__ == "__main__":
    main()
