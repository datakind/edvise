"""
Consume institution_ingest_plan for manifest status=NEW; resolve institutions via SST API,
write filtered CSVs to per-institution bronze volumes, and update ingestion_manifest.

No SFTP — uses staged local paths from prior steps.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

from edvise.ingestion.constants import (
    CATALOG,
    COLUMN_RENAMES,
    INSTITUTION_LOOKUP_PATH,
    MANIFEST_TABLE_PATH,
    PLAN_TABLE_PATH,
    SST_API_KEY_SECRET_KEY,
    SST_BASE_URL,
    SST_TOKEN_ENDPOINT,
)
from edvise.ingestion.nsc_sftp_helpers import (
    process_and_save_file,
    update_manifest,
)
from edvise.utils.api_requests import (
    EdviseAPIClient,
    fetch_institution_by_pdp_id,
)
from edvise.utils.data_cleaning import convert_to_snake_case
from edvise.utils.databricks import (
    databricksify_inst_name,
    find_bronze_schema,
    find_bronze_volume_name,
)
from edvise.utils.sftp import output_file_name_from_sftp

try:
    dbutils  # noqa: F821
except NameError:
    from unittest.mock import MagicMock

    dbutils = MagicMock()

try:
    display  # noqa: F821
except NameError:

    def display(x):
        return x


spark = DatabricksSession.builder.getOrCreate()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

asset_scope = "nsc-sftp-asset"
SST_API_KEY = dbutils.secrets.get(scope=asset_scope, key=SST_API_KEY_SECRET_KEY).strip()
if not SST_API_KEY:
    raise RuntimeError(
        f"Empty SST API key from secrets: scope={asset_scope} key={SST_API_KEY_SECRET_KEY}"
    )

api_client = EdviseAPIClient(
    api_key=SST_API_KEY,
    base_url=SST_BASE_URL,
    token_endpoint=SST_TOKEN_ENDPOINT,
    institution_lookup_path=INSTITUTION_LOOKUP_PATH,
)


def _get_workflow_run_id():
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        tags = ctx.tags()
        for k in ("jobRunId", "runId"):
            try:
                v = tags.apply(k)
                if v:
                    return str(v)
            except Exception:
                pass
        try:
            v = ctx.currentRunId().get()
            if v:
                return str(v)
        except Exception:
            pass
    except Exception:
        pass
    return None


if not spark.catalog.tableExists(PLAN_TABLE_PATH):
    logger.info(f"Plan table not found: {PLAN_TABLE_PATH}. Exiting (no-op).")
    dbutils.notebook.exit("NO_PLAN_TABLE")

if not spark.catalog.tableExists(MANIFEST_TABLE_PATH):
    raise RuntimeError(f"Manifest table missing: {MANIFEST_TABLE_PATH}")

plan_df = spark.table(PLAN_TABLE_PATH)
if plan_df.limit(1).count() == 0:
    logger.info("institution_ingest_plan is empty. Exiting (no-op).")
    dbutils.notebook.exit("NO_WORK_ITEMS")

manifest_df = spark.table(MANIFEST_TABLE_PATH).select("file_fingerprint", "status")
plan_new_df = plan_df.join(manifest_df, on="file_fingerprint", how="inner").where(
    F.col("status") == F.lit("NEW")
)
if plan_new_df.limit(1).count() == 0:
    logger.info("No planned work items where manifest status=NEW. Exiting (no-op).")
    dbutils.notebook.exit("NO_NEW_TO_INGEST")

plan_summary_df = (
    plan_new_df.groupBy("file_name", "inst_col", "local_path")
    .agg(F.countDistinct("institution_id").alias("institution_count"))
    .orderBy("file_name")
)
logger.info("Planned work summary (manifest status=NEW):")
display(plan_summary_df)

file_groups = (
    plan_new_df.select(
        "file_fingerprint",
        "file_name",
        "local_path",
        "inst_col",
        "file_size",
        "file_modified_time",
    )
    .distinct()
    .collect()
)

logger.info(f"Preparing to ingest {len(file_groups)} NEW file(s).")

workflow_run_id = _get_workflow_run_id()
logger.info(f"Workflow run_id: {workflow_run_id}")

processed_files = 0
failed_files = 0
skipped_files = 0

for fg in file_groups:
    fp = fg["file_fingerprint"]
    sftp_file_name = fg["file_name"]
    local_path = fg["local_path"]
    inst_col = fg["inst_col"]

    if not local_path or not os.path.exists(local_path):
        err = f"Staged local file missing for fp={fp}: {local_path}"
        logger.error(err)
        update_manifest(
            spark,
            MANIFEST_TABLE_PATH,
            fp,
            status="FAILED",
            error_message=err[:8000],
            run_id=workflow_run_id,
        )
        failed_files += 1
        continue

    try:
        header_cols = pd.read_csv(local_path, nrows=0).columns.tolist()
        raw_inst_col = next(
            (
                c
                for c in header_cols
                if COLUMN_RENAMES.get(
                    convert_to_snake_case(c), convert_to_snake_case(c)
                )
                == inst_col
            ),
            None,
        )
        dtype = {raw_inst_col: str} if raw_inst_col else None
        df_full = pd.read_csv(local_path, on_bad_lines="warn", dtype=dtype)
        df_full = df_full.rename(
            columns={c: convert_to_snake_case(c) for c in df_full.columns}
        )
        df_full = df_full.rename(columns=COLUMN_RENAMES)

        file_student_count = None
        try:
            student_col = next(
                (
                    c
                    for c in ("student_id", "study_id", "student_guid")
                    if c in df_full.columns
                ),
                None,
            )
            if student_col:
                file_student_count = int(df_full[student_col].nunique(dropna=True))
        except Exception:
            file_student_count = None

        file_cohort = None
        try:
            if "cohort" in df_full.columns:
                vals = (
                    df_full["cohort"]
                    .dropna()
                    .astype(str)
                    .map(lambda x: x.strip())
                    .tolist()
                )
                vals = [
                    v for v in vals if v and v.lower() not in {"nan", "none", "null"}
                ]
                file_cohort = sorted(set(vals)) or None
        except Exception:
            file_cohort = None

        file_cohort_term_pairs = None
        try:
            if {"cohort", "cohort_term"}.issubset(df_full.columns):
                tmp = df_full[["cohort", "cohort_term"]].dropna()
                tmp = tmp.assign(
                    cohort=tmp["cohort"].astype(str).map(lambda x: x.strip()),
                    cohort_term=tmp["cohort_term"]
                    .astype(str)
                    .map(lambda x: x.strip().upper()),
                )
                tmp = tmp[
                    (tmp["cohort"] != "")
                    & (tmp["cohort_term"] != "")
                    & (~tmp["cohort"].str.lower().isin({"nan", "none", "null"}))
                    & (~tmp["cohort_term"].str.lower().isin({"nan", "none", "null"}))
                ]
                tmp = tmp.drop_duplicates().sort_values(by=["cohort", "cohort_term"])
                pairs = [
                    {"cohort": r.cohort, "cohort_term": r.cohort_term}
                    for r in tmp.itertuples(index=False)
                ]
                file_cohort_term_pairs = pairs or None
        except Exception:
            file_cohort_term_pairs = None

        logger.info(
            "file=%s fp=%s: student_count=%s cohort_count=%s",
            sftp_file_name,
            fp,
            file_student_count,
            (len(file_cohort) if file_cohort else 0),
        )

        if inst_col not in df_full.columns:
            err = f"Expected institution column '{inst_col}' not found after normalization/renames for file={sftp_file_name} fp={fp}"
            logger.error(err)
            update_manifest(
                spark,
                MANIFEST_TABLE_PATH,
                fp,
                status="FAILED",
                error_message=err[:8000],
                run_id=workflow_run_id,
                cohort=file_cohort,
                cohort_term_pairs=file_cohort_term_pairs,
                student_count=file_student_count,
            )
            failed_files += 1
            continue

        inst_ids = (
            plan_new_df.where(F.col("file_fingerprint") == fp)
            .select("institution_id")
            .distinct()
            .collect()
        )
        inst_ids = [r["institution_id"] for r in inst_ids]

        if not inst_ids:
            logger.info(
                f"No institution_ids in plan for file={sftp_file_name} fp={fp}. Marking BRONZE_WRITTEN (no-op)."
            )
            update_manifest(
                spark,
                MANIFEST_TABLE_PATH,
                fp,
                status="BRONZE_WRITTEN",
                error_message=None,
                run_id=workflow_run_id,
                cohort=file_cohort,
                cohort_term_pairs=file_cohort_term_pairs,
                student_count=file_student_count,
            )
            skipped_files += 1
            continue

        preview_inst_ids = inst_ids[:10]
        logger.info(
            f"file={sftp_file_name} fp={fp}: ingesting {len(inst_ids)} institution(s) "
            f"using inst_col='{inst_col}'. Preview first 10 IDs={preview_inst_ids}"
        )

        file_errors = []

        for inst_id in inst_ids:
            try:
                target_inst_id = str(inst_id)
                filtered_df = df_full[df_full[inst_col] == target_inst_id].reset_index(
                    drop=True
                )

                if filtered_df.empty:
                    logger.info(
                        f"file={sftp_file_name} fp={fp}: institution {inst_id} has 0 rows; skipping."
                    )
                    continue

                inst_info = fetch_institution_by_pdp_id(api_client, inst_id)
                inst_name = inst_info.get("name")
                if not inst_name:
                    raise ValueError(
                        f"SST API returned no 'name' for pdp_id={inst_id}. Response={inst_info}"
                    )

                inst_prefix = databricksify_inst_name(inst_name)

                bronze_schema = find_bronze_schema(spark, CATALOG, inst_prefix)
                bronze_volume_name = find_bronze_volume_name(
                    spark, CATALOG, bronze_schema
                )
                volume_dir = f"/Volumes/{CATALOG}/{bronze_schema}/{bronze_volume_name}"

                out_file_name = output_file_name_from_sftp(sftp_file_name)
                full_path = os.path.join(volume_dir, out_file_name)

                if os.path.exists(full_path):
                    logger.info(
                        f"file={sftp_file_name} inst={inst_id}: already exists in {volume_dir}; skipping write."
                    )
                    continue

                logger.info(
                    f"file={sftp_file_name} inst={inst_id}: writing to {volume_dir} as {out_file_name}"
                )
                process_and_save_file(
                    volume_dir=volume_dir, file_name=out_file_name, df=filtered_df
                )
                logger.info(f"file={sftp_file_name} inst={inst_id}: write complete.")

            except Exception as e:
                msg = f"inst_ingest_failed file={sftp_file_name} fp={fp} inst={inst_id}: {e}"
                logger.exception(msg)
                file_errors.append(msg)

        if file_errors:
            err = " | ".join(file_errors)[:8000]
            update_manifest(
                spark,
                MANIFEST_TABLE_PATH,
                fp,
                status="FAILED",
                error_message=err,
                run_id=workflow_run_id,
                cohort=file_cohort,
                cohort_term_pairs=file_cohort_term_pairs,
                student_count=file_student_count,
            )
            failed_files += 1
        else:
            update_manifest(
                spark,
                MANIFEST_TABLE_PATH,
                fp,
                status="BRONZE_WRITTEN",
                error_message=None,
                run_id=workflow_run_id,
                cohort=file_cohort,
                cohort_term_pairs=file_cohort_term_pairs,
                student_count=file_student_count,
            )
            processed_files += 1

    except Exception as e:
        msg = f"fatal_file_error file={sftp_file_name} fp={fp}: {e}"
        logger.exception(msg)
        update_manifest(
            spark,
            MANIFEST_TABLE_PATH,
            fp,
            status="FAILED",
            error_message=msg[:8000],
            run_id=workflow_run_id,
        )
        failed_files += 1

logger.info(
    f"Done. processed_files={processed_files}, failed_files={failed_files}, skipped_files={skipped_files}"
)
dbutils.notebook.exit(
    f"PROCESSED={processed_files};FAILED={failed_files};SKIPPED={skipped_files}"
)
