"""
Ingest NEW plan rows to per-institution bronze volumes; update ingestion_manifest.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Optional

_cur = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0] or "."))
for _ in range(8):
    if os.path.isdir(os.path.join(_cur, "edvise")):
        if _cur not in sys.path:
            sys.path.insert(0, _cur)
        break
    _nxt = os.path.dirname(_cur)
    if _nxt == _cur:
        break
    _cur = _nxt

from edvise.ingestion.nsc_sftp import runtime  # noqa: E402

runtime.bootstrap_catalog()

from pyspark.sql import functions as F

from edvise.ingestion.nsc_sftp.constants import (
    CATALOG,
    COLUMN_RENAMES,
    MANIFEST_TABLE_PATH,
    PLAN_TABLE_PATH,
)
from edvise.ingestion.nsc_sftp.helpers import (
    ensure_plan_table,
    group_dataframe_by_institution_id,
    group_plan_rows_by_file,
    load_staged_csv,
    log_section,
    process_and_save_file,
    resolve_bronze_volume_dir,
    sst_identity_or_resolve,
    summarize_file_metrics,
    update_manifest,
)
from edvise.utils.institution_naming import databricksify_inst_name
from edvise.utils.sftp import output_file_name_from_sftp

dbutils = runtime.get_dbutils()
spark = runtime.get_spark()
logger = runtime.get_logger(__name__)

api_client = runtime.require_edvise_api_client(dbutils)
force_reingest = runtime.job_param_bool("force_reingest", False)

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
by_file, inst_ids_by_fp, identity_by_fp = group_plan_rows_by_file(plan_rows)

run_id = runtime.workflow_run_id(dbutils)
counts: dict[str, int] = defaultdict(int)
bronze_dir_cache: dict[str, str] = {}
ingested_rows: list[str] = []
skipped_rows: list[str] = []
failed_rows: list[str] = []

expected_unique: dict[str, str] = {}
for r in plan_rows:
    d = r.asDict()
    expected_unique[str(d.get("institution_id") or "")] = str(
        d.get("institution_name") or ""
    )
expected_unique.pop("", None)

logger.info(
    "Stage 03 — bronze ingest: %s file(s), %s institution(s), force_reingest=%s",
    len(by_file),
    len(expected_unique),
    force_reingest,
)
log_section(
    logger,
    "Expected institutions",
    [f"pdp_id={pdp_id} {name}" for pdp_id, name in sorted(expected_unique.items())],
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
        failed_rows.append(f"{file_name}: staged file missing")
        continue

    try:
        df_full = load_staged_csv(local_path, renames=COLUMN_RENAMES, inst_col=inst_col)
        student_count, file_cohort, cohort_term_pairs = summarize_file_metrics(df_full)
        logger.info(
            "Processing %s (%s students, %s planned institution(s))",
            file_name,
            student_count,
            len(inst_ids),
        )

        if inst_col not in df_full.columns:
            _fail(
                f"Missing institution column '{inst_col}'",
                cohort=file_cohort,
                cohort_term_pairs=cohort_term_pairs,
                student_count=student_count,
            )
            failed_rows.append(f"{file_name}: missing institution column {inst_col!r}")
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
            skipped_rows.append(f"{file_name}: no institutions in plan")
            continue

        wanted = set(map(str, inst_ids))
        grouped = group_dataframe_by_institution_id(df_full, inst_col, inst_ids)
        present_ids = sorted(
            {
                str(v)
                for v in df_full[inst_col].dropna().unique().tolist()
                if v is not None and str(v).strip() != ""
            }
        )
        missing_from_file = sorted(wanted - set(grouped))
        if missing_from_file:
            logger.warning(
                "File %s: planned PDP id(s) with no rows after normalize: %s "
                "(sample ids in file: %s)",
                file_name,
                ",".join(missing_from_file),
                ",".join(present_ids[:20]) or "none",
            )
        file_errors: list[str] = []
        out_name = output_file_name_from_sftp(file_name)

        for pdp_id in inst_ids:
            try:
                filtered = grouped.get(str(pdp_id))
                if filtered is None or filtered.empty:
                    counts["institutions_empty"] += 1
                    skipped_rows.append(
                        f"pdp_id={pdp_id} in {file_name}: no matching rows in file"
                    )
                    continue

                try:
                    sst_inst_id, inst_name = sst_identity_or_resolve(
                        api_client, pdp_id, identity_by_fp[fp].get(str(pdp_id))
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
                if os.path.exists(full_path) and not force_reingest:
                    counts["institutions_skipped_existing"] += 1
                    skipped_rows.append(
                        f"pdp_id={pdp_id} {inst_name}: already exists "
                        f"(set force_reingest=true to overwrite)"
                    )
                    continue

                process_and_save_file(
                    volume_dir=volume_dir, file_name=out_name, df=filtered
                )
                counts["institutions_written"] += 1
                ingested_rows.append(
                    f"pdp_id={pdp_id} {inst_name}: wrote {len(filtered)} rows -> {full_path}"
                )
            except Exception as exc:
                msg = f"inst_ingest_failed file={file_name} fp={fp} pdp_id={pdp_id}: {exc}"
                logger.exception(msg)
                file_errors.append(msg)
                failed_rows.append(f"pdp_id={pdp_id} in {file_name}: {exc}")

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
        failed_rows.append(f"{file_name}: fatal error — {exc}")

log_section(logger, "Written", ingested_rows)
log_section(logger, "Skipped", skipped_rows)
log_section(logger, "Failed", failed_rows)
logger.info(
    "Stage 03 done — written=%s existing=%s empty=%s unresolved=%s "
    "no_bronze=%s failed_files=%s force_reingest=%s",
    counts["institutions_written"],
    counts["institutions_skipped_existing"],
    counts["institutions_empty"],
    counts["institutions_unresolved"],
    counts["institutions_no_bronze"],
    counts["failed_files"],
    force_reingest,
)
# notebook_exit is what Databricks shows in the task Output panel
expected_summary = (
    "|".join(f"{pdp_id}:{name}" for pdp_id, name in sorted(expected_unique.items()))
    or "None"
)
counts_msg = (
    "PROCESSED={processed_files};FAILED={failed_files};SKIPPED={skipped_files};"
    "WRITTEN={institutions_written};EXISTING={institutions_skipped_existing};"
    "EMPTY={institutions_empty};UNRESOLVED={institutions_unresolved};"
    "NO_BRONZE={institutions_no_bronze}"
).format_map(defaultdict(int, counts))
runtime.notebook_exit(
    dbutils,
    f"{counts_msg};FORCE_REINGEST={force_reingest};EXPECTED={expected_summary};"
    f"INGESTED_N={len(ingested_rows)};SKIPPED_N={len(skipped_rows)};"
    f"FAILED_N={len(failed_rows)}",
)
