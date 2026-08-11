#!/usr/bin/env python3
"""
Publish or pull a gold GenAI mapping reference under ``genai_mapping.references``.

Modes
-----
``publish`` (canonical catalog, typically ``staging_sst_01``)
    Copy that institution's ``genai_mapping/active/`` →
    ``references/<reference_id>/current/`` + ``reference_pins`` row.
    Use when blessing a new/updated gold few-shot set.

``pull`` (replica catalog, typically ``dev_sst_02``)
    Copy ``references/<reference_id>/current/`` from ``source_catalog`` into this
    catalog. Does **not** read local ``active/`` (avoids divergent active overwriting
    the library). Requires volume read access to the source catalog.

Examples::

    # Staging — publish from active/
    python .../edvise_genai_pin_reference.py \\
      --mode publish --catalog staging_sst_01 --reference_id my_school

    # Dev — pull staging's pinned bytes (parity)
    python .../edvise_genai_pin_reference.py \\
      --mode pull --catalog dev_sst_02 --reference_id my_school \\
      --source_catalog staging_sst_01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    from edvise.genai.mapping.shared.reference_pin import (
        DEFAULT_CANONICAL_REFERENCE_CATALOG,
        pin_reference_snapshot,
        pull_reference_snapshot,
        upsert_reference_pin_row,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Publish (from active/) or pull (from canonical catalog) "
            "GenAI mapping reference few-shot artifacts"
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["publish", "pull"],
        help=(
            "publish: pin from local active/ (canonical). "
            "pull: copy references/current/ from source_catalog (replica; no local active/)."
        ),
    )
    parser.add_argument("--catalog", required=True, help="UC catalog for this job run")
    parser.add_argument(
        "--reference_id",
        required=True,
        help="Institution id / library slot under references/",
    )
    parser.add_argument(
        "--source_catalog",
        default=DEFAULT_CANONICAL_REFERENCE_CATALOG,
        help=(
            "pull only: catalog to copy references/ from "
            f"(default: {DEFAULT_CANONICAL_REFERENCE_CATALOG})"
        ),
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

    write_history = not args.no_history_copy
    if args.mode == "publish":
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
            source_catalog=args.source_catalog,
            write_history_copy=write_history,
        )
        LOGGER.info("Pulled reference:\n%s", json.dumps(pin, indent=2))

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
        args.mode,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
