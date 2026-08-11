#!/usr/bin/env python3
"""
Publish or pull a gold GenAI mapping reference under ``genai_mapping.references``.

Action is selected by ``--catalog`` (not a separate mode flag):

* ``dev_sst_02`` — **publish** from that institution's ``genai_mapping/active/``
* ``staging_sst_01`` — **pull** pinned bytes from ``dev_sst_02`` (ignores local ``active/``)

Examples::

    # Dev — publish from active/
    python .../edvise_genai_pin_reference.py \\
      --catalog dev_sst_02 --reference_id my_school

    # Staging — pull from fixed canonical catalog
    python .../edvise_genai_pin_reference.py \\
      --catalog staging_sst_01 --reference_id my_school
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Layout: <git_root>/src/edvise/genai/mapping/scripts/<this_file>
# `import edvise` needs <git_root>/src on sys.path (package is <git_root>/src/edvise/).
# Databricks spark_python_task often exec()s this file without defining __file__.
_here = globals().get("__file__")
if _here:
    _script_dir = os.path.dirname(os.path.abspath(_here))
else:
    _argv0 = os.path.abspath(sys.argv[0]) if sys.argv else ""
    if _argv0.endswith(".py") and os.path.isfile(_argv0):
        _script_dir = os.path.dirname(_argv0)
    else:
        _script_dir = os.path.abspath(os.getcwd())
_src_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))
if os.path.isdir(_src_root) and _src_root not in sys.path:
    sys.path.insert(0, _src_root)

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    from edvise.genai.mapping.shared.reference_pin import (
        DEFAULT_CANONICAL_REFERENCE_CATALOG,
        pin_reference_snapshot,
        pull_reference_snapshot,
        resolve_pin_mode,
        upsert_reference_pin_row,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Publish or pull GenAI mapping reference few-shot artifacts. "
            f"catalog={DEFAULT_CANONICAL_REFERENCE_CATALOG} publishes; "
            "catalog=staging_sst_01 pulls from the canonical catalog."
        )
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help=(
            f"UC catalog for this job: {DEFAULT_CANONICAL_REFERENCE_CATALOG} "
            "(publish) or staging_sst_01 (pull)"
        ),
    )
    parser.add_argument(
        "--reference_id",
        required=True,
        help="Institution id / library slot under references/",
    )
    parser.add_argument(
        "--pipeline_version",
        default="",
        help=(
            "publish only: optional override; default from "
            "genai_active_registry.json under active/"
        ),
    )
    parser.add_argument(
        "--no_history_copy",
        action="store_true",
        help="Skip writing references/<id>/vYYYYMMDD_<hash12>/ history snapshot",
    )
    parser.add_argument(
        "--skip_uc_registry",
        action="store_true",
        help="Only write volume files + pin JSON; skip reference_pins Delta upsert",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    mode = resolve_pin_mode(args.catalog)
    write_history = not args.no_history_copy
    if mode == "publish":
        pin = pin_reference_snapshot(
            catalog=args.catalog,
            reference_id=args.reference_id,
            pipeline_version=args.pipeline_version or None,
            write_history_copy=write_history,
        )
        LOGGER.info("Published reference:\n%s", json.dumps(pin, indent=2))
    else:
        pin = pull_reference_snapshot(
            catalog=args.catalog,
            reference_id=args.reference_id,
            source_catalog=DEFAULT_CANONICAL_REFERENCE_CATALOG,
            write_history_copy=write_history,
        )
        LOGGER.info(
            "Pulled reference from %s:\n%s",
            DEFAULT_CANONICAL_REFERENCE_CATALOG,
            json.dumps(pin, indent=2),
        )

    if args.skip_uc_registry:
        LOGGER.warning("Skipped UC reference_pins upsert (--skip_uc_registry)")
        return 0

    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception as e:  # noqa: BLE001
        LOGGER.error(
            "Spark unavailable (%s); re-run on Databricks or pass --skip_uc_registry", e
        )
        return 1

    upsert_reference_pin_row(args.catalog, pin, spark=spark)
    LOGGER.info(
        "Registry updated: %s.genai_mapping.reference_pins reference_id=%r hash=%r mode=%s",
        args.catalog,
        pin["reference_id"],
        pin["content_hash"],
        mode,
    )
    return 0


if __name__ == "__main__":
    # Databricks exec()s this file in an IPython kernel, where even SystemExit(0)
    # is reported as a failed workload; only surface non-zero exits.
    _rc = main()
    if _rc:
        raise SystemExit(_rc)
