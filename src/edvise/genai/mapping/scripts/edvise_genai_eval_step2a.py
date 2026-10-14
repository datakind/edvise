#!/usr/bin/env python3
"""Compare Step 2a gateway models on the pinned reference library.

Each slot defaults to the hosted ``system.ai.claude-<family>-<version>``
foundation model (e.g. ``sonnet:4.5`` -> ``system.ai.claude-sonnet-4-5``).
Override a slot with ``EDVISE_EVAL_GATEWAY_MODELS`` (JSON object keyed by
slot) or a per-slot ``EDVISE_EVAL_MODEL_<FAMILY>_<VERSION>`` env var if that
naming doesn't match your workspace.

Example::

    python src/edvise/genai/mapping/scripts/edvise_genai_eval_step2a.py \\
      --catalog dev_sst_02 \\
      --models sonnet:4.5,sonnet:5,haiku:4.5,haiku:5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Layout: <git_root>/src/edvise/genai/mapping/scripts/<this_file>
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
_src_root = os.path.abspath(os.path.join(_script_dir, "..", "..", "..", ".."))
if os.path.isdir(_src_root) and _src_root not in sys.path:
    sys.path.insert(0, _src_root)


def main(argv: list[str] | None = None) -> int:
    from edvise.genai.mapping.eval.run import run_step2a_evaluation
    from edvise.genai.mapping.shared.utilities import (
        disable_openai_autolog_for_gateway,
    )

    # Leave MLflow tracing enabled for mlflow.genai.evaluate. Disable only
    # OpenAI autologging before the gateway client is imported.
    disable_openai_autolog_for_gateway()

    parser = argparse.ArgumentParser(
        description=(
            "Score the production Step 2a prompt for each supplied Sonnet or "
            "Haiku version against the active reference pins, and log one "
            "MLflow run per model."
        )
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Unity Catalog that holds the active reference pins",
    )
    parser.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated or JSON array of slots to compare, for example "
            "sonnet:4.5,sonnet:5,haiku:4.5. Include at least two."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_step2a_evaluation(catalog=args.catalog, models=args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
