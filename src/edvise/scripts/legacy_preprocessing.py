"""
Run school-specific legacy preprocessing before ``training_h2o``.

The school module (e.g. ``pipelines.john_jay_col_transfer.preprocessing``) is **always**
expected to be importable from a **Unity Catalog–distributed artifact**: register a wheel
in UC (or equivalent) and attach it as a **cluster/job library**. Do not rely on copying
source onto a volume for imports.

Institution **data** paths in ``config.toml`` still use UC volume URIs
(e.g. ``/Volumes/<catalog>/..._bronze/bronze_volume/...``).

Disable preprocessing with job parameter ``legacy_preprocessing_enabled`` set to ``false``
when modeling tables are produced elsewhere.
"""

from __future__ import annotations

import argparse
import importlib
import logging

logging.basicConfig(level=logging.INFO, force=True)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy H2O preprocessing step (school-specific Python module from UC wheel)."
    )
    parser.add_argument(
        "--config_file_path",
        required=True,
        help="Path to institution config.toml (same as training_h2o).",
    )
    parser.add_argument(
        "--run_type",
        default="train",
        choices=("train", "predict"),
        help="Forwarded to the school preprocessing module.",
    )
    parser.add_argument(
        "--preprocessing_module",
        default="pipelines.john_jay_col_transfer.preprocessing",
        help="Import path for the school preprocessing package module.",
    )
    parser.add_argument(
        "--legacy_preprocessing_enabled",
        default="true",
        help='If "false", exit without running (for schools with pre-built tables).',
    )
    args = parser.parse_args()

    enabled = str(args.legacy_preprocessing_enabled).strip().lower()
    if enabled in ("false", "0", "no", "off"):
        LOGGER.info(
            "legacy_preprocessing_enabled=%s — skipping preprocessing.",
            args.legacy_preprocessing_enabled,
        )
        return

    LOGGER.info(
        "Importing school preprocessing from installed package (UC wheel): %s",
        args.preprocessing_module,
    )
    mod = importlib.import_module(args.preprocessing_module)
    run = getattr(mod, "run", None)
    if run is None:
        raise RuntimeError(
            f"Module {args.preprocessing_module!r} must define a run() function."
        )
    LOGGER.info(
        "Running %s.run(config_file_path=%r, run_type=%r)",
        args.preprocessing_module,
        args.config_file_path,
        args.run_type,
    )
    run(config_file_path=args.config_file_path, run_type=args.run_type)


if __name__ == "__main__":
    main()
