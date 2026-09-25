"""
Onboard — resume_from="gate_1"

Gate check -> resolve HITL -> hook gen LLM -> schema contract + cleaned Parquet -> exit
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from edvise.genai.mapping.state import job_state as _pipeline_job_state
from edvise.genai.mapping.state.hitl_poller import (
    DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
)

from .paths import IAPaths

LOGGER = logging.getLogger("edvise_ia")


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
        validate_term_year_semantics_resolved,
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
    validate_term_year_semantics_resolved(
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
