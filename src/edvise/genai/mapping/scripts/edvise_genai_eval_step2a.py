#!/usr/bin/env python3
"""Evaluate one Step 2a gateway model with ``mlflow.genai.evaluate``.

Example::

    python src/edvise/genai/mapping/scripts/edvise_genai_eval_step2a.py \\
      --catalog dev_sst_02 \\
      --model sonnet:4.5 \\
      --gateway_model_id databricks-claude-sonnet-4-5
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
    from edvise.genai.mapping.eval.dataset import (
        DEFAULT_REFERENCE_INSTITUTION_ID,
        DEFAULT_TARGET_INSTITUTION_ID,
    )
    from edvise.genai.mapping.eval.run import (
        DEFAULT_EXPERIMENT_NAME,
        run_step2a_evaluation,
    )
    from edvise.genai.mapping.shared.utilities import (
        disable_openai_autolog_for_gateway,
    )

    # Leave MLflow tracing enabled for mlflow.genai.evaluate. Disable only
    # OpenAI autologging before the gateway client is imported.
    disable_openai_autolog_for_gateway()

    parser = argparse.ArgumentParser(
        description=(
            "Score the production Step 2a batched prompt for one Sonnet or Haiku "
            "version and log the run to MLflow."
        )
    )
    parser.add_argument(
        "--catalog",
        required=True,
        help="Unity Catalog name used to resolve gold and reference volumes",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Bake-off slot, for example sonnet:4.5 or haiku:5.5",
    )
    parser.add_argument(
        "--gateway_model_id",
        default="",
        help="Databricks gateway endpoint id. Overrides the slot's environment id.",
    )
    parser.add_argument(
        "--institution_id",
        default=DEFAULT_TARGET_INSTITUTION_ID,
        help="Target institution whose gold manifest is scored",
    )
    parser.add_argument(
        "--reference_institution_id",
        default=DEFAULT_REFERENCE_INSTITUTION_ID,
        help="Few-shot reference institution",
    )
    parser.add_argument(
        "--experiment_name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment that collects model-comparison runs",
    )
    parser.add_argument(
        "--pipeline_version",
        default="eval",
        help="Release id injected into the predicted manifest envelope",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_step2a_evaluation(
        catalog=args.catalog,
        model=args.model,
        gateway_model_id=args.gateway_model_id or None,
        target_institution_id=args.institution_id,
        reference_institution_id=args.reference_institution_id,
        experiment_name=args.experiment_name,
        pipeline_version=args.pipeline_version or "eval",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
