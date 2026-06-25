"""Resolve ES inference ``config_file_path`` from the registered model (standalone entry)."""

from __future__ import annotations

import argparse
import logging
import sys

from edvise.dataio.inference_model_artifacts import (
    resolve_es_inference_artifacts,
    set_inference_config_task_value,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve ES inference config from the trained model's silver artifacts."
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--DB_workspace", required=True)
    parser.add_argument("--databricks_institution_name", required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()
    artifacts = resolve_es_inference_artifacts(
        model_name=args.model_name,
        db_workspace=args.DB_workspace,
        databricks_institution_name=args.databricks_institution_name,
    )
    set_inference_config_task_value(artifacts.config_file_path)


if __name__ == "__main__":
    main()
