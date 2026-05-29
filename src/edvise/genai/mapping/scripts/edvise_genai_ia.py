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
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast

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

from edvise.configs import genai as genai_cfg
from edvise.genai.mapping.state import job_state as _pipeline_job_state
from edvise.genai.mapping.state.hitl_poller import (
    DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    HITLTimeoutError,
)
from edvise.shared.logger import (
    init_file_logging_at_path,
    resolve_genai_segment_log_path,
)

LOGGER = logging.getLogger("edvise_ia")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass
class IAPaths:
    # Run folder: ``runs/onboard/{onboard_run_id}/`` or ``runs/execute/{execute_run_id}/``
    run_root: Path
    grain_output: Path
    grain_hitl: Path
    term_output: Path
    term_hitl: Path
    term_hooks: Path
    grain_hooks: Path
    enriched_schema_contract: Path
    profiling_output: Path
    cleaned_datasets: Path  # directory, one .parquet per logical dataset
    run_log: Path
    grain_hook_preview: (
        Path  # UC ``ia_gate_1_hooks`` — generated HookSpecs before apply
    )
    term_hook_preview: Path

    # Active folder (promoted artifacts, what execute mode reads from)
    active_root: Path
    active_grain_output: Path
    active_term_output: Path
    active_term_hooks: Path
    active_grain_hooks: Path
    active_enriched_schema_contract: Path

    # Optional upstream cleaned inputs (volume layout)
    genai_data: Path


def resolve_run_paths(
    institution_id: str,
    catalog: str,
    *,
    mode: str,
    onboard_run_id: str | None = None,
    execute_run_id: str | None = None,
) -> IAPaths:
    genai = Path(genai_cfg.silver_genai_mapping_root(institution_id, catalog=catalog))
    if mode == "onboard":
        rid = (onboard_run_id or "").strip()
        if not rid:
            raise ValueError("onboard_run_id is required when mode='onboard'")
        run_root = genai / "runs" / "onboard" / rid / "identity_agent"
    elif mode == "execute":
        rid = (execute_run_id or "").strip()
        if not rid:
            raise ValueError("execute_run_id is required when mode='execute'")
        run_root = genai / "runs" / "execute" / rid / "identity_agent"
    else:
        raise ValueError(f"resolve_run_paths: invalid mode={mode!r}")
    active_root = genai / "active"

    return IAPaths(
        run_root=run_root,
        grain_output=run_root / "identity_grain_output.json",
        grain_hitl=run_root / "identity_grain_hitl.json",
        term_output=run_root / "identity_term_output.json",
        term_hitl=run_root / "identity_term_hitl.json",
        term_hooks=run_root / "term_hooks.py",
        grain_hooks=run_root / "grain_hooks.py",
        enriched_schema_contract=run_root / "enriched_schema_contract.json",
        profiling_output=run_root / "profiling_output.json",
        cleaned_datasets=run_root / "cleaned_datasets",
        run_log=run_root / "run_log.json",
        grain_hook_preview=run_root / "identity_grain_hook_preview.json",
        term_hook_preview=run_root / "identity_term_hook_preview.json",
        active_root=active_root,
        active_grain_output=active_root / "grain_output.json",
        active_term_output=active_root / "term_output.json",
        active_term_hooks=active_root / "term_hooks.py",
        active_grain_hooks=active_root / "grain_hooks.py",
        active_enriched_schema_contract=active_root / "enriched_schema_contract.json",
        genai_data=genai / "data",
    )


# ---------------------------------------------------------------------------
# Onboard — resume_from="start"
# Profile -> Pass 1 grain LLM -> Pass 2 term LLM -> write HITL -> exit
# ---------------------------------------------------------------------------


def run_onboard_start(
    institution_id: str,
    paths: IAPaths,
    school_config: Any,
    llm_complete: Callable[[str, str], str],
    *,
    catalog: str,
    onboard_run_id: str,
) -> None:
    from edvise.genai.mapping.identity_agent.grain_inference import (
        build_identity_profiling_run_by_dataset,
        write_identity_profiling_artifacts,
    )
    from edvise.genai.mapping.identity_agent.grain_inference.runner import (
        run_identity_agents_for_institution_with_hitl,
    )
    from edvise.genai.mapping.identity_agent.grain_inference import (
        log_grain_auto_approve,
        log_grain_hitl_queue,
    )
    from edvise.genai.mapping.shared.hitl import PIPELINE_HITL_CONFIDENCE_THRESHOLD
    from edvise.genai.mapping.identity_agent.term_normalization.prompt import (
        TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT,
        build_term_normalization_batch_user_message_from_grain_and_profiles,
        parse_institution_term_contracts_with_hitl,
    )
    from edvise.genai.mapping.identity_agent.hitl import (
        write_identity_grain_artifacts,
        write_identity_term_artifacts,
    )
    from edvise.genai.mapping.identity_agent.grain_inference import (
        load_school_dataset_dataframe,
    )

    LOGGER.info("[onboard/start] Profiling datasets for %s", institution_id)
    paths.run_root.mkdir(parents=True, exist_ok=True)
    paths.cleaned_datasets.mkdir(parents=True, exist_ok=True)

    # §3 — Profile
    run_by_dataset = build_identity_profiling_run_by_dataset(
        institution_id=institution_id,
        school=school_config,
    )
    write_identity_profiling_artifacts(
        paths.profiling_output.parent,
        institution_id,
        run_by_dataset,
    )
    LOGGER.info("[onboard/start] Profiled datasets: %s", list(run_by_dataset.keys()))

    # §4 — Pass 1: Grain LLM
    LOGGER.info("[onboard/start] Pass 1 — Grain LLM")
    institution_profiles = {
        name: run_by_dataset[name]["key_profile"] for name in run_by_dataset
    }
    dfs = {
        name: load_school_dataset_dataframe(school_config, name)
        for name in run_by_dataset
    }
    raw_table_profiles_by_table = {
        name: run_by_dataset[name]["raw_table_profile"] for name in run_by_dataset
    }
    contracts_by_dataset, grain_hitl_items, verifications_by_dataset = (
        run_identity_agents_for_institution_with_hitl(
            institution_id=institution_id,
            institution_profiles=institution_profiles,
            dfs=dfs,
            llm_complete=llm_complete,
            raw_table_profiles_by_table=raw_table_profiles_by_table,
            confidence_threshold=PIPELINE_HITL_CONFIDENCE_THRESHOLD,
            queue_for_hitl_review=lambda c: log_grain_hitl_queue(c, logger=LOGGER),
            auto_approve_and_apply=lambda c: log_grain_auto_approve(c, logger=LOGGER),
        )
    )
    for name, verification in verifications_by_dataset.items():
        run_by_dataset[name]["grain_verification"] = verification.to_jsonable()
    write_identity_profiling_artifacts(
        paths.profiling_output.parent,
        institution_id,
        run_by_dataset,
    )

    # §5 — Pass 2: Term batch LLM
    LOGGER.info("[onboard/start] Pass 2 — Term batch LLM")
    term_batch_user = (
        build_term_normalization_batch_user_message_from_grain_and_profiles(
            institution_id,
            contracts_by_dataset,
            run_by_dataset,
        )
    )
    from edvise.genai.mapping.shared.databricks_ai_gateway import (
        DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS,
        llm_complete_combined_message_content,
    )
    from edvise.genai.mapping.shared.token_audit.prompt_token_audit import (
        estimate_tokens,
    )
    from edvise.utils.llm_utils import llm_complete_with_parse_retry

    _term_combined = llm_complete_combined_message_content(
        TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT,
        term_batch_user,
    )
    _term_est_in = estimate_tokens(_term_combined)
    LOGGER.info(
        "[onboard/start] Pass 2 gateway request (same message shape as chat.completions): "
        "chars=%d est_input_tokens~=%d (len/4 heuristic) max_output_tokens=%d est_total~=%d "
        "(if est_total exceeds the route's context window, some gateways return 403 "
        "PERMISSION_DENIED; confirm with workspace model limits)",
        len(_term_combined),
        _term_est_in,
        DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS,
        _term_est_in + DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS,
    )
    _institution_term, term_hitl_items = llm_complete_with_parse_retry(
        llm_complete,
        TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT,
        term_batch_user,
        parse_institution_term_contracts_with_hitl,
        logger=LOGGER,
    )
    term_contract_by_dataset = _institution_term.contracts_by_dataset()

    # Write HITL artifacts to run folder
    write_identity_grain_artifacts(
        paths.run_root,
        institution_id,
        contracts_by_dataset,
        grain_hitl_items,
        key_profiles_by_table={
            name: run_by_dataset[name]["key_profile"] for name in run_by_dataset
        },
    )
    write_identity_term_artifacts(
        paths.run_root,
        institution_id,
        term_contract_by_dataset,
        term_hitl_items,
    )
    LOGGER.info(
        "[onboard/start] Wrote grain HITL (%d item(s)) and term HITL (%d item(s)). Exiting.",
        len(grain_hitl_items),
        len(term_hitl_items),
    )
    _pipeline_job_state.after_ia_onboard_start(
        catalog,
        institution_id,
        onboard_run_id,
        grain_path=paths.grain_hitl,
        term_path=paths.term_hitl,
    )


# ---------------------------------------------------------------------------
# Onboard — resume_from="gate_1"
# Gate check -> resolve HITL -> hook gen LLM -> schema contract + cleaned Parquet -> exit
# ---------------------------------------------------------------------------


def _iter_term_order_configs_with_hooks(t_contract: Any) -> Iterator[Any]:
    """
    Primary ``term_config`` when it uses hooks.

    Used when merging HookSpecs for ``term_hooks.py``.
    """
    from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
        TermContract,
    )

    if not isinstance(t_contract, TermContract):
        raise TypeError(f"expected TermContract, got {type(t_contract)!r}")
    tcfg = t_contract.term_config
    if (
        tcfg is not None
        and tcfg.term_extraction == "hook_required"
        and tcfg.hook_spec is not None
    ):
        yield tcfg


def run_onboard_gate_1(
    institution_id: str,
    paths: IAPaths,
    school_config: Any,
    llm_complete: Callable[[str, str], str],
    *,
    catalog: str,
    onboard_run_id: str,
    db_run_id: str | None = None,
) -> None:
    from collections import defaultdict

    from edvise.genai.mapping.identity_agent.hitl import (
        HITLBlockingError,
        check_gate,
        load_grain_contracts_from_resolver_config,
        load_term_contracts_from_resolver_config,
        resolve_items,
        apply_hook_spec,
        generate_hook_specs_for_hook_items,
        materialize_hook_specs_to_file,
        normalized_column_names_from_raw_headers,
        validate_hook,
        validate_term_hook_hitl_covers_hook_required,
    )
    from edvise.genai.mapping.identity_agent.hitl.hook_generation import (
        apply_term_hook_spec_names_from_item_id,
        ensure_hook_spec_file,
        write_identity_hook_preview_json,
    )
    from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain
    from edvise.genai.mapping.identity_agent.execution.contract_builder import (
        build_enriched_schema_contract_for_institution,
        merge_grain_learner_id_alias_into_school_config,
        save_enriched_schema_contract,
    )
    from edvise.genai.mapping.identity_agent.term_normalization import (
        term_order_column_for_clean_dataset,
        term_order_fn_from_term_order_config,
    )
    from edvise.configs.genai import resolve_genai_data_path
    import pandas as pd

    LOGGER.info("[onboard/gate_1] Checking HITL gates for %s", institution_id)

    LOGGER.info("[onboard/gate_1] Waiting for Unity Catalog HITL approval (ia_gate_1)")
    _pipeline_job_state.wait_for_ia_gate_1_hitl(
        catalog,
        onboard_run_id,
        institution_id=institution_id,
        poll_interval_seconds=DEFAULT_HITL_POLL_INTERVAL_SECONDS,
        timeout_seconds=DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    )

    # Gate check — raises HITLBlockingError if any items still pending
    try:
        check_gate(paths.grain_hitl)
        check_gate(paths.term_hitl)
    except HITLBlockingError as e:
        LOGGER.error("[onboard/gate_1] HITL gate blocked: %s", e)
        raise

    # Resolve HITL items into output configs
    resolve_items(
        paths.grain_hitl,
        paths.grain_output,
        resolved_by="pipeline",
        run_log_path=paths.run_log,
        db_run_id=db_run_id,
    )
    resolve_items(
        paths.term_hitl,
        paths.term_output,
        resolved_by="pipeline",
        run_log_path=paths.run_log,
        db_run_id=db_run_id,
    )

    _term_after_resolve = load_term_contracts_from_resolver_config(
        paths.term_output, expected_institution_id=institution_id
    )
    validate_term_hook_hitl_covers_hook_required(
        term_hitl_path=paths.term_hitl,
        term_contract_by_dataset=_term_after_resolve,
    )

    # §6b — Hook generation LLM (grain + term), then UC ia_gate_1_hooks before apply/materialize
    LOGGER.info("[onboard/gate_1] Hook generation (preview)")
    norm_cols_by_table: dict[str, list[str]] = {}
    for ds_name, dc in school_config.datasets.items():
        csv_path = resolve_genai_data_path(
            school_config.bronze_volumes_path, dc.files[0]
        )
        hdr = pd.read_csv(csv_path, nrows=0)
        norm_cols_by_table[ds_name] = normalized_column_names_from_raw_headers(
            hdr.columns
        )

    grain_pairs = generate_hook_specs_for_hook_items(
        hitl_path=paths.grain_hitl,
        config_path=paths.grain_output,
        llm_complete=llm_complete,
        normalized_columns_by_table=norm_cols_by_table,
    )
    term_pairs = generate_hook_specs_for_hook_items(
        hitl_path=paths.term_hitl,
        config_path=paths.term_output,
        llm_complete=llm_complete,
        normalized_columns_by_table=norm_cols_by_table,
    )
    term_pairs = [
        (
            item_id,
            apply_term_hook_spec_names_from_item_id(
                spec, item_id, institution_id=institution_id
            ),
        )
        for item_id, spec in term_pairs
    ]

    write_identity_hook_preview_json(
        output_path=paths.grain_hook_preview,
        institution_id=institution_id,
        domain="identity_grain",
        specs=grain_pairs,
        hitl_path=paths.grain_hitl,
        config_path=paths.grain_output,
    )
    write_identity_hook_preview_json(
        output_path=paths.term_hook_preview,
        institution_id=institution_id,
        domain="identity_term",
        specs=term_pairs,
        hitl_path=paths.term_hitl,
        config_path=paths.term_output,
    )

    LOGGER.info(
        "[onboard/gate_1] Registering hook preview artifacts; waiting for UC (ia_gate_1_hooks)"
    )
    _pipeline_job_state.register_ia_gate_1_hook_preview_artifacts(
        catalog,
        institution_id,
        onboard_run_id,
        grain_hook_preview_path=paths.grain_hook_preview,
        term_hook_preview_path=paths.term_hook_preview,
    )
    _pipeline_job_state.wait_for_ia_gate_1_hooks_hitl(
        catalog,
        onboard_run_id,
        institution_id=institution_id,
        poll_interval_seconds=DEFAULT_HITL_POLL_INTERVAL_SECONDS,
        timeout_seconds=DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    )
    _pipeline_job_state.after_ia_onboard_gate_1_hooks_approved(
        catalog, institution_id, onboard_run_id
    )

    LOGGER.info("[onboard/gate_1] Applying hook specs (grain)")
    # Grain hooks
    for item_id, spec in grain_pairs:
        apply_hook_spec(
            paths.grain_hitl,
            paths.grain_output,
            item_id=item_id,
            hook_spec=spec,
            apply_to_group=True,
            resolved_by="pipeline",
            run_log_path=paths.run_log,
            materialize=True,
            repo_root=paths.run_root,
            db_run_id=db_run_id,
        )
        validate_hook(
            paths.grain_output,
            paths.grain_hitl,
            item_id=item_id,
            hook_file_root=paths.run_root,
        )

    # Term hooks — merge-materialize per shared term_hooks.py
    LOGGER.info("[onboard/gate_1] Applying hook specs (term)")
    term_specs_by_file: dict[str, list] = defaultdict(list)
    for item_id, spec in term_pairs:
        canonical = ensure_hook_spec_file(
            spec, institution_id=institution_id, domain=HITLDomain.IDENTITY_TERM
        )
        if not canonical.file:
            raise ValueError(f"Term hook spec missing file for item_id={item_id!r}")
        term_specs_by_file[canonical.file].append(canonical)
    for item_id, spec in term_pairs:
        apply_hook_spec(
            paths.term_hitl,
            paths.term_output,
            item_id=item_id,
            hook_spec=spec,
            apply_to_group=True,
            resolved_by="pipeline",
            run_log_path=paths.run_log,
            materialize=False,
            db_run_id=db_run_id,
        )

    # Resolver JSON lists hook_spec on term_config. Merge embedded specs with term_pairs before
    # a single materialize.
    term_contract_by_dataset = load_term_contracts_from_resolver_config(
        paths.term_output, expected_institution_id=institution_id
    )
    for _ds, t_contract in term_contract_by_dataset.items():
        for tcfg in _iter_term_order_configs_with_hooks(t_contract):
            spec_embedded = ensure_hook_spec_file(
                tcfg.hook_spec,
                institution_id=institution_id,
                domain=HITLDomain.IDENTITY_TERM,
            )
            if not spec_embedded.file:
                raise ValueError(
                    f"Embedded term hook spec missing file for dataset={_ds!r}"
                )
            term_specs_by_file[spec_embedded.file].append(spec_embedded)

    for specs in term_specs_by_file.values():
        materialize_hook_specs_to_file(
            specs, repo_root=paths.run_root, domain=HITLDomain.IDENTITY_TERM
        )
    for item_id, _ in term_pairs:
        validate_hook(
            paths.term_output,
            paths.term_hitl,
            item_id=item_id,
            hook_file_root=paths.run_root,
        )

    contracts_by_dataset = load_grain_contracts_from_resolver_config(
        paths.grain_output, expected_institution_id=institution_id
    )
    grain_map = dict(contracts_by_dataset)

    # §7 — Build enriched schema contract + cleaned Parquet
    LOGGER.info("[onboard/gate_1] Building enriched schema contract")
    term_column_by_dataset: dict[str, str] = {}
    term_order_fn_by_dataset: dict[str, Callable[[Any, str], Any] | None] = {}
    for ds, tp in term_contract_by_dataset.items():
        if ds not in grain_map:
            continue
        tcfg = tp.term_config
        if tcfg is None:
            continue
        term_column_by_dataset[ds] = term_order_column_for_clean_dataset(tcfg)
        fn_kw = (
            {"hook_modules_root": paths.run_root}
            if tcfg.term_extraction == "hook_required"
            else {}
        )
        term_order_fn_by_dataset[ds] = term_order_fn_from_term_order_config(
            tcfg, **fn_kw
        )

    school_effective = merge_grain_learner_id_alias_into_school_config(
        school_config, grain_map
    )
    enc, cleaned = build_enriched_schema_contract_for_institution(
        school_effective,
        grain_contracts_by_dataset=grain_map,
        term_column_by_dataset=term_column_by_dataset or None,
        term_order_fn_by_dataset=term_order_fn_by_dataset or None,
        hook_modules_root=paths.run_root,
    )

    # Write cleaned Parquet
    paths.cleaned_datasets.mkdir(parents=True, exist_ok=True)
    for logical_name, df in cleaned.items():
        pq_path = paths.cleaned_datasets / f"{logical_name}.parquet"
        df.to_parquet(pq_path, index=False)
        LOGGER.info("[onboard/gate_1] Wrote cleaned %s -> %s", logical_name, pq_path)

    # Write enriched schema contract
    save_enriched_schema_contract(enc, paths.enriched_schema_contract)
    LOGGER.info(
        "[onboard/gate_1] Wrote enriched schema contract -> %s",
        paths.enriched_schema_contract,
    )
    LOGGER.info("[onboard/gate_1] Complete. Exiting.")
    _pipeline_job_state.after_ia_onboard_gate_1_success(
        catalog, institution_id, onboard_run_id
    )


# ---------------------------------------------------------------------------
# Execute
# Identity clean_dataset (inference) -> drift vs active frozen contract ->
# enforce_schema_contract -> write cleaned Parquet -> exit
#
# Matches :func:`~edvise.data_audit.custom_cleaning.clean_dataset` inference guidance:
# ``generate_dtypes=False`` then apply the **onboard-frozen** JSON via
# :func:`~edvise.data_audit.custom_cleaning.enforce_schema_contract` (no re-freeze).
# ---------------------------------------------------------------------------


def _ia_hook_modules_root_for_execute(
    paths: IAPaths, school_config: Any
) -> Path | None:
    """
    Root directory for :func:`~edvise.genai.mapping.shared.hitl.hook_spec.paths.resolve_hook_module_path`.

    After SMA promotion, materialized modules live under silver
    ``genai_mapping/active/identity_hooks/<institution_id>/`` (see
    :func:`~edvise.genai.mapping.shared.active_promotion.promote_genai_mapping_to_active`).
    Fall back to bronze ``identity_agent`` when that subtree is absent (legacy / local dev).
    """
    from edvise.genai.mapping.shared.hitl.hook_spec.paths import (
        hook_modules_root_from_bronze_volume,
    )

    active_hooks = paths.active_root / "identity_hooks"
    if active_hooks.is_dir():
        LOGGER.info(
            "[execute] Hook modules root: %s (promoted identity_hooks under active/)",
            paths.active_root,
        )
        return paths.active_root
    bv = school_config.bronze_volumes_path
    if bv and str(bv).strip():
        root = hook_modules_root_from_bronze_volume(bv)
        LOGGER.warning(
            "[execute] No promoted %s; using bronze hook root %s",
            active_hooks,
            root,
        )
        return root
    LOGGER.warning(
        "[execute] No identity_hooks under active and no bronze_volumes_path; "
        "using active root %s — hook imports may fail",
        paths.active_root,
    )
    return paths.active_root


def run_execute(
    institution_id: str,
    paths: IAPaths,
    school_config: Any,
) -> None:
    import json
    from typing import Literal

    from edvise.data_audit.custom_cleaning import (
        enforce_schema_contract,
        load_schema_contract,
        normalize_columns,
    )
    from edvise.genai.mapping.identity_agent.dataset_io import (
        load_school_dataset_dataframe,
    )
    from edvise.genai.mapping.identity_agent.execution.contract_builder import (
        build_schema_contract_from_grain_contracts,
        merge_grain_learner_id_alias_into_school_config,
    )
    from edvise.genai.mapping.identity_agent.hitl import (
        load_grain_contracts_from_resolver_config,
        load_term_contracts_from_resolver_config,
    )
    from edvise.genai.mapping.identity_agent.term_normalization import (
        term_order_column_for_clean_dataset,
        term_order_fn_from_term_order_config,
    )

    LOGGER.info(
        "[execute] Loading approved artifacts from active/ for %s", institution_id
    )

    if not paths.active_enriched_schema_contract.is_file():
        raise FileNotFoundError(
            f"No active schema contract found at {paths.active_enriched_schema_contract}. "
            "Has this institution been onboarded and activated?"
        )
    if not paths.active_grain_output.is_file():
        raise FileNotFoundError(
            f"No active grain resolver output at {paths.active_grain_output}. "
            "Promote onboard artifacts to active/ before execute."
        )
    if not paths.active_term_output.is_file():
        raise FileNotFoundError(
            f"No active term resolver output at {paths.active_term_output}. "
            "Promote onboard artifacts to active/ before execute."
        )

    schema_contract = load_schema_contract(str(paths.active_enriched_schema_contract))
    expected_datasets = set(schema_contract.get("datasets", {}).keys())

    _clc = schema_contract.get("canonical_learner_column") or "learner_id"
    canonical_learner_column: Literal["student_id", "learner_id"] = (
        "student_id" if _clc == "student_id" else "learner_id"
    )

    # Soft, non-blocking: compare raw headers to onboard ``normalized_columns`` using the same
    # ``normalize_columns`` as ``clean_dataset`` step 1 — so Title Case keys match snake_case files.
    soft_raw_header_warnings: list[str] = []
    LOGGER.info(
        "[execute] Soft raw-header check vs contract normalized_columns "
        "(normalize_columns, before cleaning)"
    )
    for ds_name in school_config.datasets:
        if ds_name not in schema_contract.get("datasets", {}):
            continue
        ds_schema = schema_contract["datasets"][ds_name]
        nc = ds_schema.get("normalized_columns")
        if not isinstance(nc, dict) or not nc:
            LOGGER.debug(
                "[execute] %s: no normalized_columns; skip soft header check", ds_name
            )
            continue
        try:
            raw_df = load_school_dataset_dataframe(school_config, ds_name)
        except Exception as e:
            w = f"{ds_name}: could not load raw data for header check: {e}"
            soft_raw_header_warnings.append(w)
            LOGGER.warning("[execute] Soft header check — %s", w)
            continue
        raw_cols = [str(c).strip() for c in raw_df.columns]
        norm_raw_idx, _ = normalize_columns(raw_cols)
        raw_tokens = {str(t) for t in norm_raw_idx}

        orig_keys = list(nc.keys())
        norm_key_idx, _ = normalize_columns(orig_keys)
        key_tokens = {str(t) for t in norm_key_idx}

        frozen_vals = [str(v).strip() for v in nc.values()]
        norm_val_idx, _ = normalize_columns(frozen_vals)
        value_tokens = {str(t) for t in norm_val_idx}

        allowlist = key_tokens | value_tokens

        missing_key_tokens = key_tokens - raw_tokens
        missing_orig = sorted(
            orig_keys[i]
            for i, tok in enumerate(norm_key_idx)
            if str(tok) in missing_key_tokens
        )
        unexpected_raw = sorted(
            raw_cols[i]
            for i, tok in enumerate(norm_raw_idx)
            if str(tok) not in allowlist
        )
        if missing_orig:
            w = (
                f"{ds_name}: after normalize_columns, raw file missing onboard source column(s) "
                f"(normalized_columns keys): {missing_orig}"
            )
            soft_raw_header_warnings.append(w)
            LOGGER.warning("[execute] Soft header drift — %s", w)
        if unexpected_raw:
            preview = (
                unexpected_raw if len(unexpected_raw) <= 40 else unexpected_raw[:40]
            )
            suffix = (
                ""
                if len(unexpected_raw) <= 40
                else f" … (+{len(unexpected_raw) - 40} more)"
            )
            w = (
                f"{ds_name}: raw file has column(s) not explained by onboard "
                f"(keys/values after normalize_columns): {preview}{suffix}"
            )
            soft_raw_header_warnings.append(w)
            LOGGER.info("[execute] Soft header note — %s", w)

    grain_map = dict(
        load_grain_contracts_from_resolver_config(
            paths.active_grain_output, expected_institution_id=institution_id
        )
    )
    term_contract_by_dataset = load_term_contracts_from_resolver_config(
        paths.active_term_output, expected_institution_id=institution_id
    )

    hook_modules_root_resolved = _ia_hook_modules_root_for_execute(paths, school_config)

    term_column_by_dataset: dict[str, str] = {}
    term_order_fn_by_dataset: dict[str, Callable[[Any, str], Any] | None] = {}
    for ds, tp in term_contract_by_dataset.items():
        if ds not in grain_map:
            continue
        tcfg = tp.term_config
        if tcfg is None:
            continue
        term_column_by_dataset[ds] = term_order_column_for_clean_dataset(tcfg)
        fn_kw = (
            {"hook_modules_root": hook_modules_root_resolved}
            if tcfg.term_extraction == "hook_required"
            else {}
        )
        term_order_fn_by_dataset[ds] = term_order_fn_from_term_order_config(
            tcfg, **fn_kw
        )

    school_effective = merge_grain_learner_id_alias_into_school_config(
        school_config, grain_map
    )

    LOGGER.info(
        "[execute] Identity cleaning (clean_dataset, generate_dtypes=False); "
        "then enforce active frozen contract per custom_cleaning inference pattern"
    )
    cleaned_pre_enforce, _ = build_schema_contract_from_grain_contracts(
        school_effective,
        grain_map,
        term_column_by_dataset=term_column_by_dataset or None,
        term_order_fn_by_dataset=term_order_fn_by_dataset or None,
        canonical_learner_column=canonical_learner_column,
        hook_modules_root=hook_modules_root_resolved,
        generate_dtypes=False,
        build_frozen_contract=False,
    )

    incoming_datasets = set(cleaned_pre_enforce.keys())
    drift_issues: list[str] = []
    missing_datasets = expected_datasets - incoming_datasets
    extra_datasets = incoming_datasets - expected_datasets
    if missing_datasets:
        drift_issues.append(f"Missing datasets: {sorted(missing_datasets)}")
    if extra_datasets:
        drift_issues.append(f"Unexpected datasets: {sorted(extra_datasets)}")

    LOGGER.info(
        "[execute] Running schema drift check (post clean vs active frozen dtypes)"
    )
    has_missing_cols = False
    for ds_name, df in cleaned_pre_enforce.items():
        if ds_name not in schema_contract.get("datasets", {}):
            continue
        expected_cols = set(
            schema_contract["datasets"][ds_name].get("dtypes", {}).keys()
        )
        actual_cols = set(df.columns)
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        if missing_cols:
            has_missing_cols = True
            drift_issues.append(f"{ds_name}: missing columns {sorted(missing_cols)}")
        if extra_cols:
            LOGGER.warning(
                "[execute] %s: extra columns (will drop at enforce): %s",
                ds_name,
                sorted(extra_cols),
            )

    if drift_issues or soft_raw_header_warnings:
        paths.run_root.mkdir(parents=True, exist_ok=True)
        drift_report_path = paths.run_root / "schema_drift_report.json"
        drift_report_path.write_text(
            json.dumps(
                {
                    "institution_id": institution_id,
                    "issues": drift_issues,
                    "soft_raw_header_warnings": soft_raw_header_warnings,
                },
                indent=2,
            )
        )
        LOGGER.info(
            "[execute] Drift report written to %s (strict issues=%d, soft header notes=%d)",
            drift_report_path,
            len(drift_issues),
            len(soft_raw_header_warnings),
        )

    if drift_issues:
        LOGGER.warning(
            "[execute] Schema drift detected for %s:\n%s",
            institution_id,
            "\n".join(f"  - {i}" for i in drift_issues),
        )
        if missing_datasets or has_missing_cols:
            LOGGER.warning(
                "[execute] Schema drift (missing datasets/columns) for %s — "
                "continuing; see schema_drift_report.json. "
                "SMA execute will fail if active mapping source columns are absent.",
                institution_id,
            )

    LOGGER.info(
        "[execute] Enforcing active frozen schema contract (enforce_schema_contract)"
    )
    cleaned = enforce_schema_contract(cleaned_pre_enforce, schema_contract)
    paths.cleaned_datasets.mkdir(parents=True, exist_ok=True)
    for logical_name, df in cleaned.items():
        pq_path = paths.cleaned_datasets / f"{logical_name}.parquet"
        df.to_parquet(pq_path, index=False)
        LOGGER.info("[execute] Wrote cleaned %s -> %s", logical_name, pq_path)

    LOGGER.info("[execute] Complete. Exiting.")


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
        llm_complete = wrap_llm_complete_with_retries(
            make_databricks_gateway_llm_complete(gateway_client),
            log=LOGGER,
        )

        try:
            if resume_from == "start":
                run_onboard_start(
                    institution_id,
                    paths,
                    school_config,
                    llm_complete,
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
