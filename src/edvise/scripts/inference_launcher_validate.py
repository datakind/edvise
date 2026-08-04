#!/usr/bin/env python3
"""Launcher task: submit versioned inference from archived bundle YAML."""

from __future__ import annotations

import os
import sys

# Ensure repo src/ is on sys.path so `import edvise.*` works in Databricks Jobs.
# Layout: <git_root>/src/edvise/scripts/<this_file>
# Databricks spark_python_task often exec()s this file without defining __file__.
_here = globals().get("__file__")
if _here:
    _script_dir = os.path.dirname(os.path.abspath(_here))
else:
    _argv0 = os.path.abspath(sys.argv[0]) if sys.argv else ""
    if _argv0.endswith(".py") and os.path.isfile(_argv0):
        _script_dir = os.path.dirname(_argv0)
    else:
        _script_dir = os.path.abspath(os.getcwd())
_src_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if os.path.isdir(_src_root) and os.path.isdir(os.path.join(_src_root, "edvise")):
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

from edvise.runtime.versioned_inference.tasks.validate import main  # noqa: E402

if __name__ == "__main__":
    _rc = main()
    if _rc:
        raise SystemExit(_rc)
