"""
MVP smoke-test driver for versioned inference (``python -m edvise.runtime.inference_driver``).

Future wheels can own real orchestration here; this module only validates payload wiring.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import edvise

LOGGER = logging.getLogger(__name__)


def _load_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = "Payload root must be a JSON object"
        raise TypeError(msg)
    return data


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(
        description="MVP versioned inference smoke driver."
    )
    parser.add_argument(
        "--payload",
        required=True,
        type=Path,
        help="Path to JSON payload from the versioned inference launcher.",
    )
    args = parser.parse_args(argv)

    try:
        payload = _load_payload(args.payload)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        LOGGER.error("Failed to load payload: %s", exc)
        return 1

    model_run_id = payload.get("model_run_id")
    pipeline_version = payload.get("pipeline_version")
    institution = payload.get("databricks_institution_name")
    model_name = payload.get("model_name")
    release_raw = payload.get("release")
    if isinstance(release_raw, dict):
        release: dict[str, Any] = release_raw
    else:
        manifest = payload.get("manifest")
        release = manifest if isinstance(manifest, dict) else {}

    LOGGER.info("model_run_id=%r", model_run_id)
    LOGGER.info("pipeline_version=%r", pipeline_version)
    LOGGER.info("databricks_institution_name=%r", institution)
    LOGGER.info("model_name=%r", model_name)
    LOGGER.info("edvise.__file__=%r", edvise.__file__)

    LOGGER.info("release.expected_steps=%r", release.get("expected_steps"))
    LOGGER.info("release.job_name=%r", release.get("job_name"))
    LOGGER.info(
        "release.required_runtime=%r",
        release.get("required_runtime"),
    )
    pkgs = release.get("pypi_packages")
    if isinstance(pkgs, list):
        LOGGER.info("release.pypi_packages count=%s", len(pkgs))

    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
