#!/usr/bin/env python3
"""
Pin a gold GenAI mapping reference into the shared ``genai_mapping.references`` volume.

Copies few-shot artifacts from that institution's ``genai_mapping/active/`` into::

    /Volumes/<catalog>/genai_mapping/references/<reference_id>/current/

``reference_id`` is the institution id (library slot = school). Provenance
(``source_onboard_run_id``, ``pipeline_version``) is read from
``genai_active_registry.json`` under ``active/``.

Writes ``genai_reference_pin.json`` (content hash) and upserts
``{catalog}.genai_mapping.reference_pins``.

Example::

    python src/edvise/genai/mapping/scripts/edvise_genai_pin_reference.py \\
      --catalog dev_sst_02 \\
      --reference_id my_school
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pin GenAI mapping few-shot artifacts from institution active/ "
            "to shared references/<reference_id>/current/"
        )
    )
    parser.add_argument("--catalog", required=True, help="UC catalog (e.g. dev_sst_02)")
    parser.add_argument(
        "--reference_id",
        required=True,
        help="Institution id to pin (slot under references/ and source of active/)",
    )
    parser.add_argument(
        "--pipeline_version",
        default="",
        help=(
            "Optional override; default is pipeline_version from "
            "genai_active_registry.json under that school's active/"
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

    from edvise.genai.mapping.shared.reference_pin import (
        pin_reference_snapshot,
        upsert_reference_pin_row,
    )

    pin = pin_reference_snapshot(
        catalog=args.catalog,
        reference_id=args.reference_id,
        pipeline_version=args.pipeline_version or None,
        write_history_copy=not args.no_history_copy,
    )
    LOGGER.info("Pinned reference:\n%s", json.dumps(pin, indent=2))

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
        "Registry updated: %s.genai_mapping.reference_pins reference_id=%r hash=%r",
        args.catalog,
        pin["reference_id"],
        pin["content_hash"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
