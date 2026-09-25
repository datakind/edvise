"""
Execute

Identity clean_dataset (inference) -> drift vs active frozen contract ->
enforce_schema_contract -> write cleaned Parquet -> exit

``generate_dtypes=False`` + ``frozen_schema_contract`` (cast before dedupe), then
``enforce_schema_contract`` for column alignment and final contract enforcement.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .paths import IAPaths

LOGGER = logging.getLogger("edvise_ia")


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
        "[execute] Identity cleaning (clean_dataset, generate_dtypes=False, frozen schema); "
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
        frozen_schema_contract=schema_contract,
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
