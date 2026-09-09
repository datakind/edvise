"""
edvise_sma.py — SchemaMappingAgent pipeline job entry point.

Usage (Databricks job parameters):
    --institution_id    synthetic_edvise
    --catalog           dev_sst_02
    --mode              onboard | execute
    --resume_from       start | gate_2  (onboard only)
    --pipeline_version  Release / git tag for manifests and transformation maps (match edvise_ia job).
    --inputs_toml_path  Same resolution as edvise_ia (relative under bronze ``genai_mapping/``).
                        Onboard SMA start auto-selects one pinned reference (rules-based) and
                        snapshots it under the run ``few_shot/`` tree for Step 2a and gate 2.
    --override_2a_manifest   false (default) | true — apply post-gate manifest overrides whenever
                             this flag is set (independent of ``resume_from``). Skips Step 2A /
                             sma_gate_1 HITL; ``start`` applies overrides then returns so the
                             ``gate_2`` task can run Step 2b → promote. Requires
                             ``--overrides_json_path``.
    --overrides_json_path    Batch overrides JSON (absolute ``/Volumes/...`` or relative to the
                             SMA run root). Required when ``--override_2a_manifest=true``.

On Databricks, onboard mode best-effort updates ``{catalog}.genai_mapping`` pipeline state
(see :mod:`edvise.genai.mapping.state.job_state`); table setup and Spark are required.

After Step 2b, ``gate_2`` registers ``sma_gate_2_transformation_review`` when plans need human review
(``cohort_transformation_review.json`` / ``course_transformation_review.json`` — artifact types
``cohort_transformation_review`` / ``course_transformation_review``), merges resolutions, then
``sma_gate_2_hook_preview`` for ``HookSpec`` previews
(``cohort_transformation_hook_preview.json`` / ``course_transformation_hook_preview.json``) for plans
with ``hook_required: true`` (set in Step 2b or via transformation review option 3), then
materializes ``transform_hooks.py`` after UC approval and attaches each materialized
``hook_spec`` back onto its plan (``attach_materialized_hook_specs_to_plans``) so Step 2c's
executor can dynamically import and call the generated function instead of treating the field
as a gap. When manifest grain is stricter than cleaned row count, ``sma_gate_2_grain`` gates
``cohort_sma_grain_hitl.json`` / ``course_sma_grain_hitl.json``
(see :mod:`edvise.genai.mapping.schema_mapping_agent.grain_resolution`).

Pipeline-stage implementations live under
:mod:`edvise.genai.mapping.schema_mapping_agent.orchestration` — this module is the thin
CLI/job entry point that wires them together.
"""

import os
import sys
import argparse
import json
import logging
from typing import Literal, cast
from pathlib import Path

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

from edvise.genai.mapping.schema_mapping_agent.orchestration.paths import (
    resolve_run_paths,
)
from edvise.genai.mapping.schema_mapping_agent.orchestration.helpers import (
    _as_bool_flag,
    _build_openai_client,
    _load_enriched_contract,
    apply_gate_2_manifest_overrides,
)
from edvise.genai.mapping.schema_mapping_agent.orchestration.onboard_start import (
    run_onboard_start,
)
from edvise.genai.mapping.schema_mapping_agent.orchestration.onboard_gate_2 import (
    run_onboard_gate_2,
)
from edvise.genai.mapping.schema_mapping_agent.orchestration.execute import run_execute
from edvise.genai.mapping.shared.reference_select import ensure_run_few_shot
from edvise.genai.mapping.state import job_state as _pipeline_job_state
from edvise.genai.mapping.state import pipeline_state as _pipeline_state
from edvise.genai.mapping.state.hitl_poller import HITLTimeoutError
from edvise.shared.logger import (
    init_file_logging_at_path,
    resolve_genai_segment_log_path,
)

LOGGER = logging.getLogger("edvise_sma")


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
    override_2a_manifest: bool = False,
    overrides_json_path: str | None = None,
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
        logger_name="edvise_sma",
        append=False,
    )
    LOGGER.info(
        "edvise_sma | institution=%s | run=%s | mode=%s | resume_from=%s | "
        "override_2a_manifest=%s | artifacts_onboard=%s | log=%s",
        institution_id,
        _log_run,
        mode,
        resume_from,
        override_2a_manifest,
        artifacts_onboard_run_id or "",
        _segment_log,
    )

    from edvise import configs, dataio
    from edvise.configs.genai import resolve_genai_data_path

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
    LOGGER.info("Loading SMA school config from %s", institution_inputs_toml)
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

    input_file_paths: dict[str, list[str]] = {
        ds_name: [
            str(resolve_genai_data_path(school_config.bronze_volumes_path, f))
            for f in dc.files
        ]
        for ds_name, dc in school_config.datasets.items()
    }
    input_file_paths_json = json.dumps(input_file_paths)

    # Spark session (optional — graceful degradation outside Databricks runtime)
    try:
        from databricks.connect import DatabricksSession

        spark_session = DatabricksSession.builder.getOrCreate()
    except Exception:
        spark_session = None
        LOGGER.warning("No Databricks Spark session available.")

    if mode == "execute":
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

        run_execute(
            institution_id,
            paths,
            spark_session,
            execute_run_id=str(execute_run_id).strip(),
        )
        try:
            _pipeline_state.update_execute_pipeline_run_status(
                catalog,
                institution_id,
                str(execute_run_id).strip(),
                "complete",
                db_run_id=db_run_id,
            )
        except Exception as e:  # noqa: BLE001
            LOGGER.warning(
                "Could not mark pipeline_runs complete after SMA execute: catalog=%s execute_run_id=%s (%s)",
                catalog,
                execute_run_id,
                e,
            )

    elif mode == "onboard":
        if resume_from not in ("start", "gate_2"):
            raise ValueError(
                f"Invalid resume_from={resume_from!r} for mode='onboard'. Must be 'start' or 'gate_2'."
            )

        overrides_path = (overrides_json_path or "").strip() or None
        if override_2a_manifest and not overrides_path:
            raise ValueError(
                "--overrides_json_path is required when --override_2a_manifest=true"
            )

        onboard_run_id_s = cast(str, onboard_run_id)
        client = _build_openai_client(catalog)

        try:
            if resume_from == "start":
                # Auto-select + materialize run few_shot/ before any Step 2A skip path
                # so gate_2 can read the snapshot without touching library current/.
                enriched_contract = _load_enriched_contract(
                    paths.ia_enriched_schema_contract
                )
                snap, _selection = ensure_run_few_shot(
                    paths.run_root,
                    catalog=catalog,
                    institution_id=institution_id,
                    query_contract=enriched_contract,
                    spark=spark_session,
                )
                _pipeline_job_state.on_sma_onboard_begin(
                    catalog,
                    onboard_run_id_s,
                    resume_from=resume_from,
                    institution_id=institution_id,
                    input_file_paths_json=input_file_paths_json,
                    reference_id=snap.reference_id,
                    reference_content_hash=snap.content_hash,
                )
                if override_2a_manifest:
                    # Apply immediately (do not wait for gate_2). Skip Step 2A so we do
                    # not regenerate and overwrite the manifest; gate_2 continues 2b.
                    LOGGER.info(
                        "[onboard/start] override_2a_manifest=true — applying mapping "
                        "overrides and skipping Step 2A for %s",
                        institution_id,
                    )
                    apply_gate_2_manifest_overrides(
                        paths,
                        overrides_path or "",
                        institution_id=institution_id,
                        overridden_by="pipeline",
                        original_db_run_id=(db_run_id or onboard_run_id_s),
                    )
                else:
                    run_onboard_start(
                        institution_id=institution_id,
                        catalog=catalog,
                        paths=paths,
                        client=client,
                        spark_session=spark_session,
                        onboard_run_id=onboard_run_id_s,
                        pipeline_version=school_config.pipeline_version,
                    )
            elif resume_from == "gate_2":
                # Do not re-select or re-stamp reference hash; reuse run few_shot/.
                _pipeline_job_state.on_sma_onboard_begin(
                    catalog,
                    onboard_run_id_s,
                    resume_from=resume_from,
                    institution_id=institution_id,
                    input_file_paths_json=input_file_paths_json,
                )
                run_onboard_gate_2(
                    institution_id=institution_id,
                    catalog=catalog,
                    paths=paths,
                    client=client,
                    spark_session=spark_session,
                    onboard_run_id=onboard_run_id_s,
                    pipeline_version=school_config.pipeline_version,
                    db_run_id=db_run_id,
                    override_2a_manifest=override_2a_manifest,
                    overrides_json_path=overrides_path,
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

    parser = argparse.ArgumentParser(description="SchemaMappingAgent pipeline job")
    parser.add_argument("--institution_id", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--mode", required=True, choices=["onboard", "execute"])
    parser.add_argument("--resume_from", default="start", choices=["start", "gate_2"])
    parser.add_argument(
        "--pipeline_version",
        default="",
        help=(
            "Edvise/git release id for manifest and transformation artifacts (align with edvise_ia). "
            "Empty falls back to GENAI_GIT_TAG / installed edvise version."
        ),
    )
    parser.add_argument(
        "--inputs_toml_path",
        default="",
        help=(
            "Relative to …/bronze_volume/genai_mapping/ on the institution bronze volume, "
            "or an absolute /Volumes/... path. Empty uses inputs.toml (requires --catalog)."
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
    parser.add_argument(
        "--override_2a_manifest",
        default="false",
        help=(
            "Onboard only: when true, apply post-gate mapping overrides whenever this task runs "
            "(independent of resume_from). Skips Step 2A / sma_gate_1 HITL; start applies "
            "overrides then returns so the gate_2 task can run Step 2b → promote. "
            "Requires --overrides_json_path. Default: false."
        ),
    )
    parser.add_argument(
        "--overrides_json_path",
        default="",
        help=(
            "Path to batch overrides JSON (absolute /Volumes/... or relative to the SMA run root). "
            "Required when --override_2a_manifest=true."
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
            override_2a_manifest=_as_bool_flag(args.override_2a_manifest),
            overrides_json_path=(args.overrides_json_path or "").strip() or None,
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
