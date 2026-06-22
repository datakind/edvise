"""
edvise_genai_override_manifest.py — Apply post-gate SMA manifest mapping override(s).

Single field (local paths or Unity Catalog ``/Volumes/…`` paths):

    python edvise_genai_override_manifest.py \\
        --manifest-path /path/to/manifest_map.json \\
        --override-log-path /path/to/mapping_override_log.json \\
        --entity-type cohort \\
        --target-field learner_id \\
        --correction-json /path/to/override.json \\
        --overridden-by ops@example.org \\
        --original-db-run-id db-run-123

Multiple fields in one run (one manifest write / volume round-trip):

    python edvise_genai_override_manifest.py \\
        --manifest-path /path/to/manifest_map.json \\
        --override-log-path /path/to/mapping_override_log.json \\
        --overrides-json /path/to/overrides.json \\
        --overridden-by ops@example.org \\
        --original-db-run-id db-run-123

See :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.hitl.override.load_overrides_json`
for the batch JSON shape.

After overrides, re-run SMA step 2b to regenerate ``transformation_map.json``.
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

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.override import (  # noqa: E402
    ManifestOverrideError,
    load_correction_json,
    load_overrides_json,
    override_manifest_mapping_at_path,
    override_manifest_mappings_at_path,
    unmapped_field_mapping_record,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply post-gate SMA manifest mapping override(s).",
    )
    p.add_argument(
        "--manifest-path",
        required=True,
        help="Path to manifest_map.json (local or /Volumes/…).",
    )
    p.add_argument(
        "--override-log-path",
        required=True,
        help="Path to mapping_override_log.json (local or /Volumes/…).",
    )
    p.add_argument(
        "--overrides-json",
        help=(
            "Batch overrides file (JSON array or {\"overrides\": [...]}). "
            "When set, single-field flags below are not used."
        ),
    )
    p.add_argument(
        "--entity-type",
        choices=("cohort", "course"),
        help="Entity slice for a single-field override.",
    )
    p.add_argument(
        "--target-field",
        help="Edvise schema target field for a single-field override.",
    )
    corr = p.add_mutually_exclusive_group(required=False)
    corr.add_argument(
        "--correction-json",
        help="JSON file with a FieldMappingRecord-shaped override (single field).",
    )
    corr.add_argument(
        "--unmap",
        action="store_true",
        help="Mark the target field as unmapped (single field).",
    )
    p.add_argument(
        "--overridden-by",
        required=True,
        help="Reviewer or operator identifier for the override audit log.",
    )
    p.add_argument(
        "--original-db-run-id",
        required=True,
        help="Databricks pipeline run id that produced the manifest being overridden.",
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
        help="Optional notes for a single-field override.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.overrides_json:
            overrides = load_overrides_json(args.overrides_json)
            count = override_manifest_mappings_at_path(
                args.manifest_path,
                overrides,
                override_log_path=args.override_log_path,
                overridden_by=args.overridden_by,
                original_db_run_id=args.original_db_run_id,
                original_task_run_id=args.original_task_run_id,
                institution_id=args.institution_id,
            )
            print(
                f"✓ Applied {count} override(s). "
                "Re-run SMA step 2b to regenerate transformation_map.json."
            )
            return 0

        if not args.entity_type or not args.target_field:
            print(
                "Single-field mode requires --entity-type and --target-field, "
                "or use --overrides-json for batch mode.",
                file=sys.stderr,
            )
            return 1
        if not args.unmap and not args.correction_json:
            print(
                "Single-field mode requires --correction-json or --unmap.",
                file=sys.stderr,
            )
            return 1

        if args.unmap:
            corrected = unmapped_field_mapping_record(args.target_field)
        else:
            corrected = load_correction_json(args.correction_json)

        override_manifest_mapping_at_path(
            args.manifest_path,
            args.entity_type,
            args.target_field,
            corrected,
            override_log_path=args.override_log_path,
            overridden_by=args.overridden_by,
            original_db_run_id=args.original_db_run_id,
            original_task_run_id=args.original_task_run_id,
            reviewer_notes=args.reviewer_notes,
            institution_id=args.institution_id,
        )
    except (ManifestOverrideError, Exception) as exc:
        print(f"Override failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"✓ Override applied for {args.entity_type}.{args.target_field}. "
        "Re-run SMA step 2b to regenerate transformation_map.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
