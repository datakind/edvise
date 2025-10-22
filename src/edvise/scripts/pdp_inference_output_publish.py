"""Task for publishing data that has been output by the inference task."""

import os
import sys
import argparse
import logging
import json
from email.headerregistry import Address

script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)
print("Script dir:", script_dir)
print("Repo root:", repo_root)
print("src_path:", src_path)
print("sys.path:", sys.path)

from edvise.utils.gcs import publish_inference_output_files
from edvise.utils.emails import send_inference_completion_email

# GCS error classes for precise handling
from google.api_core.exceptions import Forbidden, NotFound


def in_databricks() -> bool:
    """Best-effort detection of a DBR environment."""
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION") or os.getenv("DB_IS_DRIVER"))


def get_dbutils():
    """Lazy import: only available on Databricks."""
    try:
        from databricks.sdk.runtime import dbutils  # type: ignore

        return dbutils
    except Exception:
        return None


def _logging(code: str, message: str, extra: dict | None = None) -> None:
    """Log a single structured line that you can search in logs."""
    payload = {"code": code, "message": message, "extra": (extra or {})}
    logging.error("AUDIT %s", json.dumps(payload))


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Publish inference outputs to GCS and notify user (Databricks-aware)."
    )
    parser.add_argument("--DB_workspace", required=True)
    parser.add_argument("--databricks_institution_name", required=True)
    parser.add_argument("--gcp_bucket_name", required=True)
    parser.add_argument("--db_run_id", required=True)
    parser.add_argument("--datakind_notification_email", required=True)
    parser.add_argument(
        "--job_type",
        choices=["inference"],
        default="inference",
        help="This task publishes inference outputs only.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on errors even on Databricks (default strict off on DBR, on elsewhere).",
    )
    args = parser.parse_args()

    # Enforce inference mode explicitly
    if args.job_type != "inference":
        raise ValueError("This publish task only supports --job_type inference.")

    dbx = in_databricks()
    strict = args.strict or (not dbx)  # lenient on DBR by default, strict elsewhere
    dbutils = get_dbutils()

    # 1) Publish files to GCS
    try:
        logging.info(
            "Publishing inference outputs to GCS bucket %s (run_id=%s)",
            args.gcp_bucket_name,
            args.db_run_id,
        )
        publish_inference_output_files(
            args.DB_workspace,
            args.databricks_institution_name,
            args.gcp_bucket_name,
            args.db_run_id,
            False,
        )
        logging.info("Publish complete.")
    except Forbidden as e:
        if strict:
            raise
        _logging(
            "gcs_forbidden",
            f"GCS 403 during publish: {e}",
            {"bucket": args.gcp_bucket_name, "run_id": args.db_run_id},
        )
    except NotFound as e:
        if strict:
            raise
        _logging(
            "gcs_not_found",
            f"Target path not found: {e}",
            {"bucket": args.gcp_bucket_name, "run_id": args.db_run_id},
        )
    except Exception as e:
        if strict:
            raise
        _logging(
            "publish_error",
            f"{e}",
            {"bucket": args.gcp_bucket_name, "run_id": args.db_run_id},
        )

    # 2) Send email notification
    try:
        if not dbutils:
            raise RuntimeError("dbutils unavailable (not running on Databricks).")

        username = dbutils.secrets.get(scope="sst", key="MANDRILL_USERNAME")
        password = dbutils.secrets.get(scope="sst", key="MANDRILL_PASSWORD")

        sender = Address("Datakind Info", "help", "datakind.org")
        to_list = [args.datakind_notification_email]

        logging.info("Sending email notification to %s", to_list)
        send_inference_completion_email(sender, to_list, [], username, password)

        logging.info("Email sent.")
    except Exception as e:
        if strict:
            raise
        _logging(
            "email_error",
            f"Failed to send completion email: {e}",
            {"to": args.datakind_notification_email},
        )


if __name__ == "__main__":
    main()
