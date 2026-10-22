"""Import-order coverage for the Step 2b scoring cycle."""

from __future__ import annotations

import subprocess
import sys


def test_identity_agent_import_does_not_cycle_through_step2b_eval() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from edvise.genai.mapping.identity_agent.orchestration.onboard_start import run_onboard_start",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_transformation_eval_import_defines_score_t2b_result() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from edvise.genai.mapping.schema_mapping_agent.transformation.eval import score_t2b_result",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
