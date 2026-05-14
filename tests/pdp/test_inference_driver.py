"""Smoke tests for ``edvise.runtime.inference_driver``."""

from __future__ import annotations

import json
from pathlib import Path

from edvise.runtime import inference_driver


def test_inference_driver_main(tmp_path: Path) -> None:
    payload = {
        "model_run_id": "rid",
        "pipeline_version": "v0",
        "manifest": {"expected_steps": ["smoke_test"]},
    }
    p = tmp_path / "p.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert inference_driver.main(["--payload", str(p)]) == 0
