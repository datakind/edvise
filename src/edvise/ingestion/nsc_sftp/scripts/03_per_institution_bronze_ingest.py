"""
Ingest NEW plan rows to per-institution bronze volumes; update ingestion_manifest.
"""

from __future__ import annotations

import os
from collections import defaultdict

from edvise.ingestion.nsc_sftp import runtime

runtime.bootstrap_catalog()

from pyspark.sql import functions as F

from edvise.ingestion.nsc_sftp.constants import (
    CATALOG,
    COLUMN_RENAMES,
    INSTITUTION_LOOKUP_PATH,
    MANIFEST_TABLE_PATH,
    PLAN_TABLE_PATH,
    SST_TOKEN_PATH,
)
from edvise.ingestion.nsc_sftp.helpers import (
    build_edvise_api_client,
    load_staged_csv,
    process_and_save_file,
    resolve_bronze_volume_dir,
    summarize_file_metrics,
    update_manifest,
)
from edvise.utils.api_requests import fetch_institution_by_pdp_id
from edvise.utils.institution_naming import databricksify_inst_name
from edvise.utils.sftp import output_file_name_from_sftp

dbutils = runtime.get_dbutils()
spark = runtime.get_spark()
logger = runtime.get_logger(__name__)

db_workspace = runtime.require_job_param("DB_workspace")
secret_scope = runtime.require_job_param("nsc_sftp_secret_scope")
sst_api_key_secret_key = runtime.require_job_param("sst_api_key_secret_key")

api_key = dbutils.secrets.get(scope=secret_scope, key=sst_api_key_secret_key).strip()
if not api_key:
    raise RuntimeError(
        f"Empty SST API key: scope={secret_scope} key={sst_api_key_secret_key}"
    )

api_client = build_edvise_api_client(
    api_key=api_key,
    db_workspace=db_workspace,
    token_path=SST_TOKEN_PATH,
    institution_lookup_path=INSTITUTION_LOOKUP_PATH,
)


def _school_check_log(file_name: str, inst_id: str, filtered) -> None:
    if {"cohort", "cohort_term"}.issubset(filtered.columns):
        latest = filtered["cohort"].max()
        terms = (
            filtered.loc[filtered["cohort"] == latest, "cohort_term"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        logger.info(
            "School check file=%s inst=%s rows=%s latest_cohort=%s terms=%s",
            file_name,
            inst_id,
            len(filtered),
            latest,
            terms,
        )
    else:
        logger.info(
            "School check file=%s inst=%s rows=%s", file_name, inst_id, len(filtered)
        )


if not spark.catalog.tableExists(PLAN_TABLE_PATH):
    runtime.notebook_exit(dbutils, "NO_PLAN_TABLE")
if not spark.catalog.tableExists(MANIFEST_TABLE_PATH):
    raise RuntimeError(f"Manifest table missing: {MANIFEST_TABLE_PATH}")

plan_new_df = (
    spark.table(PLAN_TABLE_PATH)
    .join(
        spark.table(MANIFEST_TABLE_PATH).select("file_fingerprint", "status"),
        on="file_fingerprint",
        how="inner",
    )
    .where(F.col("status") == F.lit("NEW"))
)
if plan_new_df.limit(1).count() == 0:
    runtime.notebook_exit(dbutils, "NO_NEW_TO_INGEST")

plan_rows = plan_new_df.select(
    "file_fingerprint", "file_name", "local_path", "inst_col", "institution_id"
).collect()
by_file: dict[str, dict] = {}
inst_ids_by_fp: dict[str, list[str]] = defaultdict(list)
for row in plan_rows:
    fp = row["file_fingerprint"]
    inst_ids_by_fp[fp].append(row["institution_id"])
    by_file.setdefault(
        fp,
        {
            "file_name": row["file_name"],
            "local_path": row["local_path"],
            "inst_col": row["inst_col"],
        },
    )

run_id = runtime.workflow_run_id(dbutils)
counts: dict[str, int] = defaultdict(int)
bronze_dir_cache: dict[str, str] = {}

for fp, meta in by_file.items():
    file_name = meta["file_name"]
    local_path = meta["local_path"]
    inst_col = meta["inst_col"]
    inst_ids = sorted(set(inst_ids_by_fp[fp]))

    def _fail(msg: str, **manifest_kwargs) -> None:
        update_manifest(
            spark,
            MANIFEST_TABLE_PATH,
            fp,
            status="FAILED",
            error_message=msg[:8000],
            run_id=run_id,
            **manifest_kwargs,
        )
        counts["failed_files"] += 1

    if not local_path or not os.path.exists(local_path):
        _fail(f"Staged local file missing: {local_path}")
        continue

    try:
        df_full = load_staged_csv(local_path, renames=COLUMN_RENAMES, inst_col=inst_col)
        student_count, file_cohort, cohort_term_pairs = summarize_file_metrics(df_full)
        metrics = dict(
            cohort=file_cohort,
            cohort_term_pairs=cohort_term_pairs,
            student_count=student_count,
        )
        logger.info(
            "file=%s fp=%s students=%s cohorts=%s institutions=%s",
            file_name,
            fp,
            student_count,
            len(file_cohort or []),
            len(inst_ids),
        )

        if inst_col not in df_full.columns:
            _fail(f"Missing institution column '{inst_col}'", **metrics)
            continue

        if not inst_ids:
            update_manifest(
                spark,
                MANIFEST_TABLE_PATH,
                fp,
                status="BRONZE_WRITTEN",
                error_message=None,
                run_id=run_id,
                **metrics,
            )
            counts["skipped_files"] += 1
            continue

        wanted = set(map(str, inst_ids))
        grouped = {
            str(k): g.reset_index(drop=True)
            for k, g in df_full.groupby(inst_col, sort=False)
            if str(k) in wanted
        }
        file_errors: list[str] = []
        out_name = output_file_name_from_sftp(file_name)

        for inst_id in inst_ids:
            try:
                filtered = grouped.get(str(inst_id))
                if filtered is None or filtered.empty:
                    counts["institutions_empty"] += 1
                    continue

                _school_check_log(file_name, inst_id, filtered)

                try:
                    info = fetch_institution_by_pdp_id(api_client, inst_id)
                except Exception as api_err:
                    counts["institutions_unresolved"] += 1
                    raise ValueError(f"SST API lookup failed: {api_err}") from api_err

                inst_name = info.get("name")
                if not inst_name:
                    counts["institutions_unresolved"] += 1
                    raise ValueError(f"SST API returned no name for pdp_id={inst_id}")

                prefix = databricksify_inst_name(inst_name)
                if prefix not in bronze_dir_cache:
                    try:
                        bronze_dir_cache[prefix] = resolve_bronze_volume_dir(
                            spark, CATALOG, prefix
                        )
                    except ValueError as bronze_err:
                        counts["institutions_no_bronze"] += 1
                        raise ValueError(
                            f"Bronze missing for {inst_name!r} ({prefix}): {bronze_err}"
                        ) from bronze_err

                volume_dir = bronze_dir_cache[prefix]
                full_path = os.path.join(volume_dir, out_name)
                if os.path.exists(full_path):
                    counts["institutions_skipped_existing"] += 1
                    logger.info(
                        "Skip existing file=%s inst=%s path=%s",
                        file_name,
                        inst_id,
                        full_path,
                    )
                    continue

                logger.info(
                    "Write file=%s inst=%s name=%r -> %s/%s",
                    file_name,
                    inst_id,
                    inst_name,
                    volume_dir,
                    out_name,
                )
                process_and_save_file(
                    volume_dir=volume_dir, file_name=out_name, df=filtered
                )
                counts["institutions_written"] += 1
            except Exception as exc:
                msg = (
                    f"inst_ingest_failed file={file_name} fp={fp} inst={inst_id}: {exc}"
                )
                logger.exception(msg)
                file_errors.append(msg)

        if file_errors:
            _fail(" | ".join(file_errors), **metrics)
        else:
            update_manifest(
                spark,
                MANIFEST_TABLE_PATH,
                fp,
                status="BRONZE_WRITTEN",
                error_message=None,
                run_id=run_id,
                **metrics,
            )
            counts["processed_files"] += 1

    except Exception as exc:
        logger.exception("fatal_file_error file=%s fp=%s: %s", file_name, fp, exc)
        _fail(f"fatal_file_error file={file_name} fp={fp}: {exc}")

logger.info("Done counts=%s", dict(counts))
runtime.notebook_exit(
    dbutils,
    "PROCESSED={processed_files};FAILED={failed_files};SKIPPED={skipped_files};"
    "WRITTEN={institutions_written};EXISTING={institutions_skipped_existing};"
    "UNRESOLVED={institutions_unresolved};NO_BRONZE={institutions_no_bronze}".format_map(
        defaultdict(int, counts)
    ),
)
