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
    ensure_plan_table,
    job_edvise_api_client,
    load_staged_csv,
    process_and_save_file,
    resolve_bronze_volume_dir,
    resolve_sst_institution,
    summarize_file_metrics,
    update_manifest,
)
from edvise.utils.institution_naming import databricksify_inst_name
from edvise.utils.sftp import output_file_name_from_sftp

dbutils = runtime.get_dbutils()
spark = runtime.get_spark()
logger = runtime.get_logger(__name__)

api_client = job_edvise_api_client(
    dbutils,
    db_workspace=runtime.require_job_param("DB_workspace"),
    secret_scope=runtime.require_job_param("nsc_sftp_secret_scope"),
    sst_api_key_secret_key=runtime.require_job_param("sst_api_key_secret_key"),
    token_path=SST_TOKEN_PATH,
    institution_lookup_path=INSTITUTION_LOOKUP_PATH,
)


def _school_check_log(file_name: str, pdp_id: str, filtered: pd.DataFrame) -> None:
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
            "School check file=%s pdp_id=%s rows=%s latest_cohort=%s terms=%s",
            file_name,
            pdp_id,
            len(filtered),
            latest,
            terms,
        )
    else:
        logger.info(
            "School check file=%s pdp_id=%s rows=%s", file_name, pdp_id, len(filtered)
        )


def _sst_identity(pdp_id: str, planned: Optional[tuple[str, str]]) -> tuple[str, str]:
    if planned and planned[0] and planned[1]:
        return planned
    return resolve_sst_institution(api_client, pdp_id)


if not spark.catalog.tableExists(PLAN_TABLE_PATH):
    runtime.notebook_exit(dbutils, "NO_PLAN_TABLE")
if not spark.catalog.tableExists(MANIFEST_TABLE_PATH):
    raise RuntimeError(f"Manifest table missing: {MANIFEST_TABLE_PATH}")

ensure_plan_table(spark, PLAN_TABLE_PATH)
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
    "file_fingerprint",
    "file_name",
    "local_path",
    "inst_col",
    "institution_id",
    "inst_id",
    "institution_name",
).collect()
by_file: dict[str, dict[str, str]] = {}
inst_ids_by_fp: dict[str, list[str]] = defaultdict(list)
identity_by_fp: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
for row in plan_rows:
    d = row.asDict()
    fp = d["file_fingerprint"]
    pdp_id = str(d["institution_id"])
    inst_ids_by_fp[fp].append(pdp_id)
    by_file.setdefault(
        fp,
        {
            "file_name": d["file_name"],
            "local_path": d["local_path"],
            "inst_col": d["inst_col"],
        },
    )
    sst_id = str(d.get("inst_id") or "").strip()
    sst_name = str(d.get("institution_name") or "").strip()
    if sst_id and sst_name:
        identity_by_fp[fp][pdp_id] = (sst_id, sst_name)

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

        for pdp_id in inst_ids:
            try:
                filtered = grouped.get(str(pdp_id))
                if filtered is None or filtered.empty:
                    counts["institutions_empty"] += 1
                    skipped_rows.append(
                        f"file={file_name} pdp_id={pdp_id} reason=no_rows_for_institution"
                    )
                    continue

                _school_check_log(file_name, pdp_id, filtered)

                try:
                    sst_inst_id, inst_name = _sst_identity(
                        pdp_id, identity_by_fp[fp].get(str(pdp_id))
                    )
                except Exception as api_err:
                    counts["institutions_unresolved"] += 1
                    raise ValueError(f"SST API lookup failed: {api_err}") from api_err

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
                        f"file={file_name} pdp_id={pdp_id} inst_id={sst_inst_id} "
                        f"institution={inst_name!r} reason=already_exists path={full_path}"
                    )
                    logger.info(
                        "Skip existing file=%s pdp_id=%s inst_id=%s institution=%r path=%s",
                        file_name,
                        pdp_id,
                        sst_inst_id,
                        inst_name,
                        full_path,
                    )
                    continue

                logger.info(
                    "Writing file=%s pdp_id=%s inst_id=%s institution=%r rows=%s -> %s",
                    file_name,
                    pdp_id,
                    sst_inst_id,
                    inst_name,
                    len(filtered),
                    full_path,
                )
                process_and_save_file(
                    volume_dir=volume_dir, file_name=out_name, df=filtered
                )
                counts["institutions_written"] += 1
                ingested_rows.append(
                    f"file={file_name} pdp_id={pdp_id} inst_id={sst_inst_id} "
                    f"institution={inst_name!r} rows={len(filtered)} path={full_path}"
                )
            except Exception as exc:
                msg = f"inst_ingest_failed file={file_name} fp={fp} pdp_id={pdp_id}: {exc}"
                logger.exception(msg)
                file_errors.append(msg)
                failed_rows.append(f"file={file_name} pdp_id={pdp_id} reason={exc}")

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

for label, lines in (
    ("INGESTED", ingested_rows),
    ("SKIPPED", skipped_rows),
    ("FAILED", failed_rows),
):
    logger.info("=== %s (%s) ===", label, len(lines))
    for line in lines:
        logger.info("%s %s", label, line)
logger.info("Done counts=%s", dict(counts))
runtime.notebook_exit(
    dbutils,
    "PROCESSED={processed_files};FAILED={failed_files};SKIPPED={skipped_files};"
    "WRITTEN={institutions_written};EXISTING={institutions_skipped_existing};"
    "UNRESOLVED={institutions_unresolved};NO_BRONZE={institutions_no_bronze}".format_map(
        defaultdict(int, counts)
    ),
)
