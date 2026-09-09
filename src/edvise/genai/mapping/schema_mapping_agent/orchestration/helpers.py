"""Shared helpers for the SchemaMappingAgent pipeline entry point stages."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from edvise.genai.mapping.shared.databricks_ai_gateway import (
    build_gateway_message_content,
    resolve_gateway_model_id,
)

from .paths import SMAPaths

LOGGER = logging.getLogger("edvise_sma")


def _load_enriched_contract(path: Path) -> dict[Any, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Enriched schema contract not found: {path}. "
            "Run edvise_ia onboard/gate_1 first."
        )
    return cast(dict[Any, Any], json.loads(path.read_text()))


def _load_institution_term_config_optional(
    path: Path, *, expected_institution_id: str
) -> dict | None:
    """
    Load IdentityAgent ``identity_term_output.json`` as a dict for Step 2b optional term context.

    The caller should pass the pipeline ``institution_id`` so we reject a file whose embedded
    ``institution_id`` does not match (path alone is already scoped to that school's volume and
    run segment — see :func:`~edvise.genai.mapping.schema_mapping_agent.orchestration.paths.resolve_run_paths`).

    Returns None when the file is missing, invalid, or mismatched — Step 2b still runs without it.
    """
    if not path.is_file():
        LOGGER.info(
            "[step2b] No identity term output at %s — skipping institution_term_config",
            path,
        )
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        LOGGER.warning(
            "[step2b] Could not read JSON from %s (%s) — skipping institution_term_config",
            path,
            e,
        )
        return None
    from pydantic import ValidationError

    from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
        InstitutionTermContract,
    )

    try:
        inst = InstitutionTermContract.model_validate(raw)
    except ValidationError as e:
        LOGGER.warning(
            "[step2b] Invalid InstitutionTermContract in %s (%s) — skipping institution_term_config",
            path,
            e,
        )
        return None
    if inst.institution_id != expected_institution_id:
        LOGGER.warning(
            "[step2b] identity_term_output institution_id %r != pipeline institution_id %r "
            "(%s) — skipping institution_term_config",
            inst.institution_id,
            expected_institution_id,
            path,
        )
        return None
    return inst.model_dump(mode="json")


def _load_cleaned_dataframes(
    cleaned_datasets_dir: Path, enriched_contract: dict
) -> dict:
    import pandas as pd

    dataframes = {}
    for logical_name in enriched_contract.get("datasets", {}):
        pq = cleaned_datasets_dir / f"{logical_name}.parquet"
        if not pq.is_file():
            raise FileNotFoundError(
                f"Missing cleaned Parquet for dataset {logical_name!r}: {pq}. "
                "Run edvise_ia onboard/gate_1 first."
            )
        dataframes[logical_name] = pd.read_parquet(pq)
    LOGGER.info("Loaded cleaned dataframes: %s", list(dataframes.keys()))
    return dataframes


def _write_output_data(
    output_data_dir: Path, cohort_result: Any, course_result: Any
) -> None:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_data_dir / "cohort.parquet"
    course_path = output_data_dir / "course.parquet"
    cohort_result.df.to_parquet(cohort_path, index=False)
    course_result.df.to_parquet(course_path, index=False)
    LOGGER.info(
        "Wrote cohort output -> %s (shape=%s)", cohort_path, cohort_result.df.shape
    )
    LOGGER.info(
        "Wrote course output -> %s (shape=%s)", course_path, course_result.df.shape
    )


def _run_pandera_validation(
    cohort_result: Any,
    course_result: Any,
    *,
    report_path: Path,
) -> None:
    from edvise.genai.mapping.schema_mapping_agent.execution.pandera_validation_report import (
        write_pandera_validation_errors,
    )

    write_pandera_validation_errors(
        report_path,
        cohort_result.df,
        course_result.df,
        logger=LOGGER,
    )


def _build_openai_client(catalog: str) -> Any:
    """Build OpenAI-compatible client for Databricks AI Gateway."""
    from openai import OpenAI

    from edvise.genai.mapping.shared.databricks_ai_gateway import (
        disable_mlflow_tracing_for_openai_gateway_client,
        require_databricks_token,
        resolve_ai_gateway_base_url,
    )

    disable_mlflow_tracing_for_openai_gateway_client()
    return OpenAI(
        api_key=require_databricks_token(),
        base_url=resolve_ai_gateway_base_url(),
    )


def _run_once(
    model_id: str,
    prompt: str | list[dict[str, Any]],
    client: Any,
    *,
    log_cache_usage: bool = False,
) -> dict[str, Any]:
    """
    Call :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.eval.run_once` with
    retries for transient gateway / transport failures (same policy as IA ``llm_complete``).

    ``prompt`` may be a plain string or a list of content blocks (see
    :func:`~edvise.genai.mapping.shared.databricks_ai_gateway.build_gateway_message_content`)
    when the caller wants Anthropic/Databricks prompt caching on a static block.
    """
    from edvise.genai.mapping.shared.databricks_ai_gateway import (
        gateway_run_once_error_text_is_retryable,
    )
    from edvise.genai.mapping.schema_mapping_agent.manifest.eval import run_once

    max_attempts = 5
    initial_backoff_s = 2.0
    max_backoff_s = 60.0
    last: dict[str, Any] = {}
    for attempt in range(max_attempts):
        last = run_once(model_id, prompt, client, log_cache_usage=log_cache_usage)
        if last.get("success"):
            return last
        if attempt >= max_attempts - 1:
            return last
        err = last.get("error") or ""
        if not gateway_run_once_error_text_is_retryable(err):
            return last
        delay = min(max_backoff_s, initial_backoff_s * (2**attempt)) * (
            0.5 + random.random() * 0.5
        )
        LOGGER.warning(
            "SMA run_once non-success (attempt %d/%d); retry in %.1fs — %s",
            attempt + 1,
            max_attempts,
            delay,
            err[:300].replace("\n", " "),
        )
        time.sleep(delay)
    return last


def _sma_llm_complete_run_once(
    client: Any,
    *,
    cache_system_prompt: bool = False,
) -> Callable[[str, str], str]:
    """
    ``(system, user) -> text`` for :func:`~edvise.utils.llm_utils.llm_complete_with_parse_retry`
    (combines like refinement).

    ``cache_system_prompt`` opts into Anthropic/Databricks prompt caching (see
    :func:`~edvise.genai.mapping.shared.databricks_ai_gateway.build_gateway_message_content`)
    on the ``system`` block whenever both ``system`` and ``user`` are non-empty and ``system``
    is long enough to be cacheable. It's a safe no-op for callers that pass an empty
    ``system`` (e.g. Step 2a/2b, which send the whole prompt as ``user``).
    """
    model_id = resolve_gateway_model_id()

    def llm_complete(system: str, user: str) -> str:
        s = (system or "").strip()
        u = (user or "").strip()
        content: str | list[dict[str, Any]]
        if s and u:
            content = build_gateway_message_content(
                s, u, cache_system_prompt=cache_system_prompt, cache_ttl="5m"
            )
        elif u:
            content = u
        elif s:
            content = s
        else:
            raise RuntimeError("SMA LLM call has empty system and user prompts")
        result = _run_once(
            model_id, content, client, log_cache_usage=isinstance(content, list)
        )
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "SMA LLM call failed")
        resp = result.get("response")
        if not isinstance(resp, str) or not resp.strip():
            raise RuntimeError("SMA LLM returned empty response")
        return resp

    return llm_complete


def _as_bool_flag(value: Any) -> bool:
    """Parse Databricks job / CLI boolean-ish values (``true``/``false`` strings)."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "y", "on")


def resolve_overrides_json_path(
    overrides_json_path: str,
    *,
    run_root: Path,
) -> Path:
    """
    Resolve ``overrides_json_path`` to an existing file.

    Absolute paths (including ``/Volumes/...``) are used as-is; relative paths are
    resolved under the SMA ``run_root``.
    """
    raw = str(overrides_json_path or "").strip()
    if not raw:
        raise ValueError("overrides_json_path must be non-empty")
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    under_run = run_root / raw
    if under_run.is_file():
        return under_run
    raise FileNotFoundError(
        f"Overrides JSON not found at {candidate} or {under_run}. "
        "Pass an absolute path or a path relative to the SMA run root."
    )


def apply_gate_2_manifest_overrides(
    paths: SMAPaths,
    overrides_json_path: str,
    *,
    institution_id: str,
    overridden_by: str,
    original_db_run_id: str,
) -> int:
    """
    Apply batch mapping overrides to ``paths.manifest_map`` before Step 2b.

    Returns the number of overrides applied.
    """
    from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.override import (
        load_overrides_json,
        override_manifest_mappings,
    )

    if not paths.manifest_map.is_file():
        raise FileNotFoundError(
            f"Cannot apply overrides — manifest_map.json missing: {paths.manifest_map}. "
            "Override mode requires an existing post-2A onboard run."
        )
    resolved = resolve_overrides_json_path(overrides_json_path, run_root=paths.run_root)
    overrides = load_overrides_json(resolved)
    count = override_manifest_mappings(
        paths.manifest_map,
        overrides,
        override_log_path=paths.mapping_override_log,
        overridden_by=overridden_by,
        original_db_run_id=original_db_run_id,
        institution_id=institution_id,
    )
    LOGGER.info(
        "[onboard/gate_2] Applied %d mapping override(s) from %s -> %s",
        count,
        resolved,
        paths.manifest_map,
    )
    return count
