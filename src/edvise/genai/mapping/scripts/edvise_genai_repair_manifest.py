"""
edvise_genai_repair_manifest.py — Apply a post-gate SMA (2a) manifest mapping repair.

Usage (local paths or Unity Catalog ``/Volumes/…`` paths):

    python edvise_genai_repair_manifest.py \\
        --manifest-path /path/to/manifest_map.json \\
        --repair-log-path /path/to/repair_log.json \\
        --entity-type cohort \\
        --target-field learner_id \\
        --correction-json /path/to/correction.json \\
        --repaired-by ops@example.org \\
        --original-db-run-id db-run-123 \\
        [--institution-id u9] \\
        [--reviewer-notes "fixed column name"]

To leave a field unmapped instead of supplying a correction file:

    python edvise_genai_repair_manifest.py ... --unmap

After repair, re-run SMA step 2b for the institution/run to regenerate
``transformation_map.json`` (see ``rerun_scope`` in ``repair_log.json``).
"""

from __future__ import annotations

import argparse
import os
import sys

# Layout: <git_root>/src/edvise/genai/mapping/scripts/<this_file>
_here = globals().get("__file__")
if _here:
    _script_dir = os.path.dirname(os.path.abspath(_here))
else:
    _argv0 = os.path.abspath(sys.argv[0]) if sys.argv else ""
    _script_dir = (
        os.path.dirname(_argv0)
        if _argv0.endswith(".py") and os.path.isfile(_argv0)
        else os.path.abspath(os.getcwd())
    )
_src_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))
if os.path.isdir(_src_root) and _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.repair import (  # noqa: E402
    ManifestRepairError,
    load_correction_json,
    repair_manifest_mapping_at_path,
    unmapped_field_mapping_record,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply a post-gate SMA (2a) manifest mapping repair.",
    )
    p.add_argument(
        "--manifest-path",
        required=True,
        help="Path to manifest_map.json (local or /Volumes/…).",
    )
    p.add_argument(
        "--repair-log-path",
        required=True,
        help="Path to repair_log.json (local or /Volumes/…).",
    )
    p.add_argument(
        "--entity-type",
        required=True,
        choices=("cohort", "course"),
        help="Entity slice to update inside the manifest envelope.",
    )
    p.add_argument(
        "--target-field",
        required=True,
        help="Edvise schema target field to repair.",
    )
    corr = p.add_mutually_exclusive_group(required=True)
    corr.add_argument(
        "--correction-json",
        help="JSON file with a FieldMappingRecord-shaped correction.",
    )
    corr.add_argument(
        "--unmap",
        action="store_true",
        help="Mark the target field as unmapped (null source_column/source_table).",
    )
    p.add_argument(
        "--repaired-by",
        required=True,
        help="Reviewer or operator identifier for the repair audit log.",
    )
    p.add_argument(
        "--original-db-run-id",
        required=True,
        help="Databricks pipeline run id that produced the manifest being repaired.",
    )
    p.add_argument(
        "--original-task-run-id",
        default=None,
        help="Optional task run id for audit correlation.",
    )
    p.add_argument(
        "--institution-id",
        default=None,
        help=(
            "Required when manifest_path is a standalone FieldMappingManifest on a volume "
            "(not a MappingManifestEnvelope with institution_id)."
        ),
    )
    p.add_argument(
        "--reviewer-notes",
        default=None,
        help="Optional free-text notes stored on the repaired mapping row.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.unmap:
        corrected = unmapped_field_mapping_record(args.target_field)
    else:
        corrected = load_correction_json(args.correction_json)

    try:
        repair_manifest_mapping_at_path(
            args.manifest_path,
            args.entity_type,
            args.target_field,
            corrected,
            repair_log_path=args.repair_log_path,
            repaired_by=args.repaired_by,
            original_db_run_id=args.original_db_run_id,
            original_task_run_id=args.original_task_run_id,
            reviewer_notes=args.reviewer_notes,
            institution_id=args.institution_id,
        )
    except (ManifestRepairError, Exception) as exc:
        print(f"Repair failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"✓ Repair applied for {args.entity_type}.{args.target_field}. "
        "Re-run SMA step 2b to regenerate transformation_map.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
