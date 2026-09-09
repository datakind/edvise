"""Run-path resolution for the IdentityAgent pipeline entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edvise.configs import genai as genai_cfg


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
