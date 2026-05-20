#!/usr/bin/env python3
"""
Publish a runtime bundle directory for one ``pipeline_version`` (git SHA in dev).

Copies ``databricks_bundle_snapshot/`` from this repo. Optionally copy a built wheel into
the same directory for ad-hoc local experiments (the MVP Databricks flow does not use wheels).

Example::

  python -m build --wheel
  python pipelines/pdp/scripts/materialize_release_bundle.py \\
    --pipeline-version b855c29c1b1fe7f1507872c62cffa765e95b97ac \\
    --output-dir /tmp/edvise_releases/b855c29c1b1fe7f1507872c62cffa765e95b97ac \\
    --wheel dist/edvise-0.2.1-py3-none-any.whl
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.pdp.launchers.bundle_materialize import materialize_runtime_bundle_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize edvise runtime bundle.")
    parser.add_argument(
        "--pipeline-version",
        required=True,
        help="Git SHA or release id (used as UC volume subdirectory name).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Bundle directory to create (e.g. .../edvise_releases/<sha>).",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="Built .whl to copy into the bundle.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Edvise repository root (for DAB YAML sources).",
    )
    args = parser.parse_args()
    pipeline_version = args.pipeline_version.strip()
    output_dir = args.output_dir.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()

    materialize_runtime_bundle_dir(
        output_dir,
        pipeline_version,
        repo_root=repo_root,
        skip_snapshot_if_present=False,
    )

    if args.wheel is not None:
        wheel_dest = output_dir / args.wheel.name
        shutil.copy2(args.wheel.expanduser().resolve(), wheel_dest)
        print(f"  wheel: {args.wheel.name}")

    print(f"Wrote bundle to {output_dir}")
    print(f"  pipeline_version (folder name): {pipeline_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
