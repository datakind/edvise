"""
edvise_ia.py — IdentityAgent pipeline job entry point.

Usage (Databricks job parameters):
    --institution_id    synthetic_edvise
    --catalog           dev_sst_02
    --mode              onboard | execute
    --resume_from       start | gate_1  (onboard only)
    --pipeline_version  Release / git tag for artifacts (e.g. from ``git describe``); same idea as PDP jobs.
    --inputs_toml_path  Relative to ``…/bronze_volume/genai_mapping/`` or absolute ``/Volumes/…``.
                        If omitted or empty, uses ``inputs.toml`` under genai_mapping (requires ``--catalog``).

On Databricks, onboard mode best-effort updates ``{catalog}.genai_mapping`` pipeline state
(see :mod:`edvise.genai.mapping.state.job_state`). Gate ``ia_gate_1`` covers grain/term HITL JSON;
``ia_gate_1_hooks`` covers generated hook specs (preview JSON) before apply/materialize. Table setup
and Spark are required.

Pipeline-stage implementations live under
:mod:`edvise.genai.mapping.identity_agent.orchestration` — this module is the thin CLI/job
entry point that wires them together.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Literal, cast

# Layout: <git_root>/src/edvise/genai/mapping/scripts/<this_file>
# `import edvise` needs <git_root>/src on sys.path (package is <git_root>/src/edvise/).
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

# Before any import that loads ``openai`` (Databricks may autolog it otherwise).
from edvise.genai.mapping.shared.utilities import (
    disable_mlflow_side_effects_for_openai_gateway,
)

disable_mlflow_side_effects_for_openai_gateway()

from edvise.genai.mapping.identity_agent.orchestration.paths import resolve_run_paths
from edvise.genai.mapping.identity_agent.orchestration.onboard_start import (
    run_onboard_start,
)
from edvise.genai.mapping.identity_agent.orchestration.onboard_gate_1 import (
    run_onboard_gate_1,
)
from edvise.genai.mapping.identity_agent.orchestration.execute import run_execute
from edvise.genai.mapping.state import job_state as _pipeline_job_state
from edvise.genai.mapping.state.hitl_poller import HITLTimeoutError
from edvise.shared.logger import (
    init_file_logging_at_path,
    resolve_genai_segment_log_path,
)

LOGGER = logging.getLogger("edvise_ia")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    institution_id: str,
    catalog: str,
    mode: str,
    onboard_run_id: str | None = None,
    execute_run_id: str | None = None,
    artifacts_onboard_run_id: str | None = None,
    resume_from: str = "start",
    inputs_toml_path: str | None = None,
    db_run_id: str | None = None,
    pipeline_version: str | None = None,
    bronze_batch_dir: str | None = None,
) -> None:
    if mode == "onboard":
        if not (onboard_run_id or "").strip():
            raise ValueError("onboard_run_id is required when mode='onboard'")
        paths = resolve_run_paths(
            institution_id,
            catalog,
            mode="onboard",
            onboard_run_id=onboard_run_id,
        )
        _log_run = onboard_run_id
    elif mode == "execute":
        if not (execute_run_id or "").strip():
            raise ValueError("execute_run_id is required when mode='execute'")
        paths = resolve_run_paths(
            institution_id,
            catalog,
            mode="execute",
            execute_run_id=execute_run_id,
        )
        _log_run = execute_run_id
    else:
        raise ValueError(f"Invalid mode={mode!r}. Must be 'onboard' or 'execute'.")

    _segment_log = resolve_genai_segment_log_path(
        paths.run_root,
        mode=mode,
        resume_from=resume_from,
    )
    init_file_logging_at_path(
        _segment_log,
        logger_name="edvise_ia",
        append=False,
    )
    LOGGER.info(
        "edvise_ia | institution=%s | run=%s | mode=%s | resume_from=%s | artifacts_onboard=%s | log=%s",
        institution_id,
        _log_run,
        mode,
        resume_from,
        artifacts_onboard_run_id or "",
        _segment_log,
    )

    # Load school config (shared across all modes)
    from edvise import configs, dataio
    from edvise.genai.mapping.identity_agent.grain_inference import (
        create_openai_client_for_databricks_gateway,
        make_databricks_gateway_llm_complete,
        resolve_column_roles_gateway_model_id,
        resolve_gateway_model_id,
        wrap_llm_complete_with_retries,
    )

    institution_inputs_toml = Path(
        configs.genai.resolve_genai_inputs_toml_path(
            institution_id,
            catalog=catalog,
            inputs_toml_path=(inputs_toml_path or "").strip() or None,
        )
    )
    if not institution_inputs_toml.is_file():
        default_hint = configs.genai.resolve_genai_inputs_toml_path(
            institution_id, catalog=catalog, inputs_toml_path=None
        )
        raise FileNotFoundError(
            f"IdentityAgent inputs.toml not found: {institution_inputs_toml}. "
            "Pass --inputs_toml_path relative to genai_mapping on bronze (e.g. inputs.toml or inputs/inputs.toml), "
            "a full /Volumes/... path, or place the file at "
            f"{default_hint!r}."
        )
    LOGGER.info("Loading IA school config from %s", institution_inputs_toml)
    _ia = dataio.read.read_config(
        str(institution_inputs_toml),
        schema=configs.genai.IdentityAgentInputsConfig,
    )
    _pv_job = (pipeline_version or "").strip() or None
    school_config = _ia.to_school_mapping_config(
        uc_catalog=catalog,
        pipeline_mode=cast(Literal["onboard", "execute"], mode),
        pipeline_version=_pv_job,
    )
    LOGGER.info("pipeline_version=%s", school_config.pipeline_version)

    if mode == "execute":
        from edvise.genai.mapping.shared.batch_input_paths import (
            apply_bronze_batch_dir_overrides,
        )

        school_config = apply_bronze_batch_dir_overrides(
            school_config,
            bronze_batch_dir=bronze_batch_dir,
        )

    from edvise.configs.genai import resolve_genai_data_path

    input_file_paths: dict[str, list[str]] = {
        ds_name: [
            str(resolve_genai_data_path(school_config.bronze_volumes_path, f))
            for f in dc.files
        ]
        for ds_name, dc in school_config.datasets.items()
    }
    input_file_paths_json = json.dumps(input_file_paths)

    if mode == "execute":
        from edvise.genai.mapping.state import pipeline_state as _pipeline_state

        try:
            _pipeline_state.update_execute_pipeline_run_input_file_paths(
                catalog,
                institution_id,
                str(execute_run_id).strip(),
                input_file_paths_json,
            )
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(
                "Could not stamp input_file_paths on execute pipeline_runs: catalog=%s execute_run_id=%s (%s)",
                catalog,
                execute_run_id,
                e,
            )

        run_execute(institution_id, paths, school_config)
        # Leave execute pipeline_runs status non-terminal (initial ``running``) until SMA execute
        # finishes; otherwise ``bootstrap_execute_run`` in the SMA task mints a new execute_run_id
        # and cleaned Parquet paths no longer match IA's output.

    elif mode == "onboard":
        if resume_from not in ("start", "gate_1"):
            raise ValueError(
                f"Invalid resume_from={resume_from!r} for mode='onboard'. Must be 'start' or 'gate_1'."
            )

        onboard_run_id_s = cast(str, onboard_run_id)
        _pipeline_job_state.ensure_ia_run_row(
            catalog,
            institution_id,
            onboard_run_id_s,
            create_run=(resume_from == "start"),
            db_run_id=db_run_id,
            input_file_paths_json=input_file_paths_json,
        )
        _pipeline_job_state.on_ia_onboard_begin(
            catalog,
            onboard_run_id_s,
            resume_from=resume_from,
            institution_id=institution_id,
            input_file_paths_json=input_file_paths_json,
        )

        # LLM client only needed for onboard
        gateway_client = create_openai_client_for_databricks_gateway()
        # cache_system_prompt=True caches the static IA grain-inference system prompt
        # (~24k tokens, identical across every dataset in a run) and the static IA
        # term-normalization system prompt, which both flow through this same
        # llm_complete (see grain_inference/runner.py and the term_normalization call
        # below). Anthropic/Databricks skip caching for prompts under
        # _CACHE_CONTROL_MIN_CHARS, so this is a safe no-op for any other caller that
        # might reuse this instance with a short/empty system prompt.
        llm_complete = wrap_llm_complete_with_retries(
            make_databricks_gateway_llm_complete(
                gateway_client, cache_system_prompt=True
            ),
            log=LOGGER,
        )
        column_roles_llm_complete = wrap_llm_complete_with_retries(
            make_databricks_gateway_llm_complete(
                gateway_client,
                model=resolve_column_roles_gateway_model_id(),
            ),
            log=LOGGER,
        )
        LOGGER.info(
            "[onboard] column_roles model=%s grain/term model=%s",
            resolve_column_roles_gateway_model_id(),
            resolve_gateway_model_id(),
        )

        try:
            if resume_from == "start":
                run_onboard_start(
                    institution_id,
                    paths,
                    school_config,
                    llm_complete,
                    column_roles_llm_complete=column_roles_llm_complete,
                    catalog=catalog,
                    onboard_run_id=onboard_run_id_s,
                )
            elif resume_from == "gate_1":
                run_onboard_gate_1(
                    institution_id,
                    paths,
                    school_config,
                    llm_complete,
                    catalog=catalog,
                    onboard_run_id=onboard_run_id_s,
                    db_run_id=db_run_id,
                )
        except HITLTimeoutError:
            raise
        except Exception:
            _pipeline_job_state.mark_pipeline_failed(
                catalog, institution_id, onboard_run_id_s
            )
            raise

    else:
        raise ValueError(f"Invalid mode={mode!r}. Must be 'onboard' or 'execute'.")


if __name__ == "__main__":
    from edvise.genai.mapping.state import pipeline_state

    parser = argparse.ArgumentParser(description="IdentityAgent pipeline job")
    parser.add_argument("--institution_id", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--mode", required=True, choices=["onboard", "execute"])
    parser.add_argument("--resume_from", default="start", choices=["start", "gate_1"])
    parser.add_argument(
        "--inputs_toml_path",
        default="",
        help=(
            "Relative to …/bronze_volume/genai_mapping/ on the institution bronze volume, "
            "or an absolute /Volumes/... path. Empty uses inputs.toml (requires --catalog)."
        ),
    )
    parser.add_argument(
        "--pipeline_version",
        default="",
        help=(
            "Edvise/git release id stamped on mapping artifacts (set from job parameters / CI, "
            "e.g. git tag). Empty falls back to GENAI_GIT_TAG / installed edvise version."
        ),
    )
    parser.add_argument(
        "--db_run_id",
        default="",
        help="Databricks job run id (orchestration id) stored on pipeline_runs.db_run_id; empty omits.",
    )
    parser.add_argument(
        "--bronze_batch_dir",
        default="",
        help=(
            "ES inference only: batch landing dir from batch_gcs_ingest "
            "(gcs_uploads/{batch_id}/). Resolves inputs.toml filenames inside it."
        ),
    )
    parser.add_argument(
        "--new_onboard_run",
        action="store_true",
        help=(
            "Onboard mode only: mint a fresh opaque onboard_run_id (ignore db_run_id); rare escape "
            "hatch — prefer starting a new job for a new folder; repairs reuse the same db_run_id."
        ),
    )
    args = parser.parse_args()

    try:
        from pyspark.sql import SparkSession

        _spark_sess = SparkSession.getActiveSession()
        _db_from_spark = (
            _spark_sess.conf.get("spark.databricks.job.runId", None)
            if _spark_sess is not None
            else None
        )
    except Exception:
        _db_from_spark = None

    _db_run_id = (
        (args.db_run_id or "").strip()
        or ((str(_db_from_spark).strip()) if _db_from_spark else "").strip()
        or None
    )

    _execute_run_id: str | None = None
    _artifacts_onboard: str | None = None
    _onboard_run_id: str | None = None

    if args.mode == "execute":
        _boot = pipeline_state.bootstrap_execute_run(
            args.catalog,
            args.institution_id,
            db_run_id=_db_run_id,
        )
        _execute_run_id = _boot.execute_run_id
        _artifacts_onboard = _boot.artifacts_onboard_run_id
    else:
        _onboard_run_id = pipeline_state.bootstrap_resolved_onboard_run_id(
            args.catalog,
            args.institution_id,
            None,
            db_run_id=_db_run_id,
            force_new_onboard_run=bool(args.new_onboard_run),
        )

    try:
        run(
            institution_id=args.institution_id,
            catalog=args.catalog,
            mode=args.mode,
            onboard_run_id=_onboard_run_id,
            execute_run_id=_execute_run_id,
            artifacts_onboard_run_id=_artifacts_onboard,
            resume_from=args.resume_from,
            inputs_toml_path=(args.inputs_toml_path or "").strip() or None,
            db_run_id=_db_run_id,
            pipeline_version=(args.pipeline_version or "").strip() or None,
            bronze_batch_dir=(args.bronze_batch_dir or "").strip() or None,
        )
    except BaseException:
        if args.mode == "execute" and _execute_run_id:
            pipeline_state.mark_execute_pipeline_run_status(
                args.catalog,
                args.institution_id,
                _execute_run_id,
                "failed",
            )
        raise
