#!/usr/bin/env python3
"""
Pin a gold GenAI mapping reference into the shared ``genai_mapping.references`` volume.

Copies few-shot artifacts from a school's ``active/`` (default) or a specific onboard run
into::

    /Volumes/<catalog>/genai_mapping/references/<reference_id>/current/

Writes ``genai_reference_pin.json`` (content hash) and upserts
``{catalog}.genai_mapping.reference_pins``.

Does **not** change SMA few-shot resolution (still reads school ``active/`` until a follow-up
change). Safe to run after HITL + promote for each gold reference school.

Example::

    python src/edvise/genai/mapping/scripts/edvise_genai_pin_reference.py \\
      --catalog dev_sst_02 \\
      --reference_id ref_cc_student_term_01 \\
      --source_institution_id my_school \\
      --source active \\
      --archetype cc_student_term \\
      --pinned_by vishakh
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pin GenAI mapping few-shot artifacts to shared references/ volume"
    )
    parser.add_argument("--catalog", required=True, help="UC catalog (e.g. dev_sst_02)")
    parser.add_argument(
        "--reference_id",
        required=True,
        help="Library slot id (opaque ok), used as folder name under references/",
    )
    parser.add_argument(
        "--source_institution_id",
        required=True,
        help="Institution whose silver volume provides the artifacts to pin",
    )
    parser.add_argument(
        "--source",
        default="active",
        choices=["active", "onboard_run"],
        help="Copy from genai_mapping/active/ (default) or a specific onboard run tree",
    )
    parser.add_argument(
        "--onboard_run_id",
        default="",
        help="Required when --source=onboard_run; optional override when source=active",
    )
    parser.add_argument(
        "--archetype",
        default="",
        help="Optional label (e.g. cc_student_term, 4yr_banner) stored on the pin",
    )
    parser.add_argument(
        "--pipeline_version",
        default="",
        help="Optional; defaults from genai_active_registry.json when pinning from active/",
    )
    parser.add_argument(
        "--pinned_by",
        required=True,
        help="Operator identity recorded on the pin (required)",
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
        source_institution_id=args.source_institution_id,
        pinned_by=args.pinned_by,
        source=args.source,  # type: ignore[arg-type]
        onboard_run_id=args.onboard_run_id or None,
        archetype=args.archetype or None,
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
