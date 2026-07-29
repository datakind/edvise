"""Smoke tests for ``edvise.runtime.inference_driver``."""

from __future__ import annotations

import json
from pathlib import Path

from edvise.runtime import inference_driver


def test_inference_driver_main(tmp_path: Path) -> None:
    payload = {
        "model_run_id": "rid",
        "pipeline_version": "sha1",
        "release": {
            "expected_steps": ["feature_generation", "inference_h2o"],
            "required_runtime": {"databricks_runtime": "15.4.x-cpu-ml-scala2.12"},
        },
    }
    p = tmp_path / "p.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert inference_driver.main(["--payload", str(p)]) == 0
