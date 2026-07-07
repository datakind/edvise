#!/usr/bin/env python3
"""Dry-run compatibility checks for versioned inference (local / CI)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers.bundle_from_dab import load_inference_job_definition
from pipelines.pdp.launchers.inference_job_submit import build_submit_run_body
from pipelines.pdp.launchers.inference_parameters import resolve_versioned_job_parameters
from pipelines.pdp.launchers.pipeline_version_ref import build_git_source, git_ref_kind

_FIXTURE = (
    _REPO_ROOT / "tests/pdp/fixtures/inference_job_parameter_contract.yml"
)


def _check_case(
    *,
    label: str,
    pipeline_version: str,
    workspace: str,
    bundle_dir: Path,
) -> None:
    job = load_inference_job_definition(
        bundle_dir / "databricks_bundle_snapshot/resources/github_pdp_inference.yml"
    )
    resolved = resolve_versioned_job_parameters(
        job,
        bundle_dir,
        launcher_overrides={
            "databricks_institution_name": "miles_cc",
            "model_name": "retention_into_year_2_associates",
            "DB_workspace": workspace,
            "cohort_file_name": "cohort.csv",
            "course_file_name": "course.csv",
            "gcp_bucket_name": "bucket",
            "datakind_notification_email": "ops@example.com",
        },
    )
    body = build_submit_run_body(
        job,
        pipeline_version=pipeline_version,
        git_url="https://github.com/datakind/edvise",
        run_name=f"verify-{label}",
        parameter_overrides=resolved,
    )
    git_src = build_git_source("https://github.com/datakind/edvise", pipeline_version)
    print(
        json.dumps(
            {
                "case": label,
                "pipeline_version": pipeline_version,
                "git_ref_kind": git_ref_kind(pipeline_version),
                "git_source": git_src,
                "task_count": len(body.get("tasks") or []),
                "resolved_params": sorted(resolved.keys()),
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-yml",
        type=Path,
        default=_FIXTURE,
        help="Inference job YAML fixture for dry-run checks.",
    )
    args = parser.parse_args(argv)
    if not args.fixture_yml.is_file():
        print(f"Fixture not found: {args.fixture_yml}", file=sys.stderr)
        return 1

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for label, pv, ws in (
            ("dev_sha", "87b641939205110d03ce8c300e68980327dd6732", "dev_sst_02"),
            ("staging_tag", "v2.4.0", "staging_sst_01"),
        ):
            bundle_dir = base / label
            snap = bundle_dir / "databricks_bundle_snapshot" / "resources"
            snap.mkdir(parents=True)
            (snap / "github_pdp_inference.yml").write_text(
                args.fixture_yml.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (bundle_dir / "databricks_bundle_snapshot/databricks.yml").write_text(
                'variables:\n  schema_type:\n    default: "pdp"\n',
                encoding="utf-8",
            )
            _check_case(label=label, pipeline_version=pv, workspace=ws, bundle_dir=bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
