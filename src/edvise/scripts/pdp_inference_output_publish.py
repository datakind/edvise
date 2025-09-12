"""Task for publishing data that has been output by the inference task."""

import os
import sys
import argparse
import logging

# Go up 3 levels from the current file's directory to reach repo root
script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")

if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

# Debug info
print("Script dir:", script_dir)
print("Repo root:", repo_root)
print("src_path:", src_path)
print("sys.path:", sys.path)

from edvise.utils.gcs import publish_inference_output_files
from edvise.utils.emails import send_inference_completion_email
from databricks.sdk import WorkspaceClient
from email.headerregistry import Address


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DB_workspace", required=True, help="Databricks workspace of the task."
    )
    parser.add_argument(
        "--databricks_institution_name",
        required=True,
        help="IThe Databricks institution name whose output we want to publish",
    )
    parser.add_argument(
        "--gcp_bucket_name",
        required=True,
        help="Name of the bucket we want to publish results to (in format [env]_[inst_id]).",
    )
    parser.add_argument(
        "--db_run_id",
        required=True,
        help="The job run id of the current pipeline execution.",
    )

    parser.add_argument(
        "--datakind_notification_email",
        required=True,
        help="User's email who triggered the inference run.",
    )
    args = parser.parse_args()
    w = WorkspaceClient()
    logging.info("Publishing files to GCP bucket")
    publish_inference_output_files(
        args.DB_workspace,
        args.databricks_institution_name,
        args.gcp_bucket_name,
        args.db_run_id,
        False,  # Set approved = False since this task will run immediately after inference.
    )
    username = w.dbutils.secrets.get(scope="sst", key="MANDRILL_USERNAME")
    sender_email = Address("Datakind Info", "help", "datakind.org")
    cc_email_list = ["education@datakind.org"]
    password = w.dbutils.secrets.get(scope="sst", key="MANDRILL_PASSWORD")
    logging.info("Sending email notification")
    send_inference_completion_email(
        sender_email,
        [args.datakind_notification_email],
        cc_email_list,
        username,
        password,
    )


if __name__ == "__main__":
    main()
