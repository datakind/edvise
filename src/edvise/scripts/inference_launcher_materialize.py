#!/usr/bin/env python3
"""Launcher task: resolve pipeline_version and materialize runtime bundle on UC volume."""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[2]
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from edvise.runtime.versioned_inference.tasks.materialize import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
