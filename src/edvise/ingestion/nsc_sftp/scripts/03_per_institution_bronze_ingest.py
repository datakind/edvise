"""
Ingest NEW plan rows to per-institution bronze volumes; update ingestion_manifest.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Optional

# Ensure repo src/ is on sys.path so `import edvise.*` works in Databricks Jobs.
# Layout: <git_root>/src/edvise/ingestion/nsc_sftp/scripts/<this_file>
_here = globals().get("__file__")
if _here:
    _script_dir = os.path.dirname(os.path.abspath(_here))
else:
    _argv0 = os.path.abspath(sys.argv[0]) if sys.argv else ""
    if _argv0.endswith(".py") and os.path.isfile(_argv0):
        _script_dir = os.path.dirname(_argv0)
    else:
        _script_dir = os.path.abspath(os.getcwd())
_current = _script_dir
for _ in range(8):
    if os.path.isdir(os.path.join(_current, "edvise")):
        if _current not in sys.path:
            sys.path.insert(0, _current)
        break
    _parent = os.path.dirname(_current)
    if _parent == _current:
        break
    _current = _parent

import pandas as pd
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


def _school_check_log(file_name: str, inst_id: str, filtered: pd.DataFrame) -> None:
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
by_file: dict[str, dict[str, str]] = {}
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
# Per-institution outcomes for end-of-run summary logging.
ingested_rows: list[str] = []
skipped_rows: list[str] = []
failed_rows: list[str] = []

logger.info(
    "Starting bronze ingest: %s file(s), %s institution-file plan row(s), run_id=%s",
    len(by_file),
    len(plan_rows),
    run_id,
)

for fp, meta in by_file.items():
    file_name = meta["file_name"]
    local_path = meta["local_path"]
    inst_col = meta["inst_col"]
    inst_ids = sorted(set(inst_ids_by_fp[fp]))

    def _fail(
        msg: str,
        *,
        cohort: Optional[list[str]] = None,
        cohort_term_pairs: Optional[list[dict[str, str]]] = None,
        student_count: Optional[int] = None,
    ) -> None:
        update_manifest(
            spark,
            MANIFEST_TABLE_PATH,
            fp,
            status="FAILED",
            error_message=msg[:8000],
            run_id=run_id,
            cohort=cohort,
            cohort_term_pairs=cohort_term_pairs,
            student_count=student_count,
        )
        counts["failed_files"] += 1

    if not local_path or not os.path.exists(local_path):
        _fail(f"Staged local file missing: {local_path}")
        failed_rows.append(f"file={file_name} reason=staged_file_missing")
        continue

    try:
        df_full = load_staged_csv(local_path, renames=COLUMN_RENAMES, inst_col=inst_col)
        student_count, file_cohort, cohort_term_pairs = summarize_file_metrics(df_full)
        logger.info(
            "Processing file=%s fp=%s students=%s cohorts=%s planned_institutions=%s",
            file_name,
            fp,
            student_count,
            len(file_cohort or []),
            inst_ids,
        )

        if inst_col not in df_full.columns:
            _fail(
                f"Missing institution column '{inst_col}'",
                cohort=file_cohort,
                cohort_term_pairs=cohort_term_pairs,
                student_count=student_count,
            )
            failed_rows.append(
                f"file={file_name} reason=missing_institution_column col={inst_col}"
            )
            continue

        if not inst_ids:
            update_manifest(
                spark,
                MANIFEST_TABLE_PATH,
                fp,
                status="BRONZE_WRITTEN",
                error_message=None,
                run_id=run_id,
                cohort=file_cohort,
                cohort_term_pairs=cohort_term_pairs,
                student_count=student_count,
            )
            counts["skipped_files"] += 1
            skipped_rows.append(f"file={file_name} reason=no_institutions_in_plan")
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
                    skipped_rows.append(
                        f"file={file_name} pdp_id={inst_id} reason=no_rows_for_institution"
                    )
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
                    skipped_rows.append(
                        f"file={file_name} pdp_id={inst_id} institution={inst_name!r} "
                        f"reason=already_exists path={full_path}"
                    )
                    logger.info(
                        "Skip existing file=%s pdp_id=%s institution=%r path=%s",
                        file_name,
                        inst_id,
                        inst_name,
                        full_path,
                    )
                    continue

                logger.info(
                    "Writing file=%s pdp_id=%s institution=%r rows=%s -> %s",
                    file_name,
                    inst_id,
                    inst_name,
                    len(filtered),
                    full_path,
                )
                process_and_save_file(
                    volume_dir=volume_dir, file_name=out_name, df=filtered
                )
                counts["institutions_written"] += 1
                ingested_rows.append(
                    f"file={file_name} pdp_id={inst_id} institution={inst_name!r} "
                    f"rows={len(filtered)} path={full_path}"
                )
            except Exception as exc:
                msg = (
                    f"inst_ingest_failed file={file_name} fp={fp} inst={inst_id}: {exc}"
                )
                logger.exception(msg)
                file_errors.append(msg)
                failed_rows.append(
                    f"file={file_name} pdp_id={inst_id} reason={exc}"
                )

        if file_errors:
            _fail(
                " | ".join(file_errors),
                cohort=file_cohort,
                cohort_term_pairs=cohort_term_pairs,
                student_count=student_count,
            )
        else:
            update_manifest(
                spark,
                MANIFEST_TABLE_PATH,
                fp,
                status="BRONZE_WRITTEN",
                error_message=None,
                run_id=run_id,
                cohort=file_cohort,
                cohort_term_pairs=cohort_term_pairs,
                student_count=student_count,
            )
            counts["processed_files"] += 1

    except Exception as exc:
        logger.exception("fatal_file_error file=%s fp=%s: %s", file_name, fp, exc)
        _fail(f"fatal_file_error file={file_name} fp={fp}: {exc}")
        failed_rows.append(f"file={file_name} reason=fatal_file_error error={exc}")

logger.info("=== INGESTION SUMMARY (written=%s) ===", len(ingested_rows))
for line in ingested_rows:
    logger.info("INGESTED %s", line)
logger.info("=== SKIPPED (%s) ===", len(skipped_rows))
for line in skipped_rows:
    logger.info("SKIPPED %s", line)
logger.info("=== FAILED (%s) ===", len(failed_rows))
for line in failed_rows:
    logger.info("FAILED %s", line)
logger.info("Done counts=%s", dict(counts))
runtime.notebook_exit(
    dbutils,
    "PROCESSED={processed_files};FAILED={failed_files};SKIPPED={skipped_files};"
    "WRITTEN={institutions_written};EXISTING={institutions_skipped_existing};"
    "UNRESOLVED={institutions_unresolved};NO_BRONZE={institutions_no_bronze}".format_map(
        defaultdict(int, counts)
    ),
)
