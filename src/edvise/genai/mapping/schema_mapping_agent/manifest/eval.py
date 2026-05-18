import time
import json
import logging
import os
import shutil
from collections import Counter
from collections.abc import Callable
from typing import Any, Literal, cast
from copy import deepcopy
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from dotenv import load_dotenv
from pydantic import ValidationError

from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import (
    build_step2a_prompt,
    build_step2a_prompt_cohort_pass,
    build_step2a_prompt_course_pass,
    load_json,
    merge_step2a_entity_manifests,
    run_sma_refinement,
)
from edvise.genai.mapping.shared.utilities import strip_json_fences

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl import (
    check_sma_hitl_gate,
    resolve_sma_items,
    write_sma_hitl_artifact,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas import (
    InstitutionSMAHITLItems,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    MappingManifestEnvelope,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.validation import (
    validate_manifest,
)
from edvise.genai.mapping.shared.schema_contract import (
    parse_enriched_schema_contract_for_sma,
)
from edvise.configs import genai as genai_cfg
from edvise.data_audit.schemas.raw_edvise_student import (
    RawEdviseStudentDataSchema,
)
from edvise.data_audit.schemas.raw_edvise_course import (
    RawEdviseCourseDataSchema,
)
from edvise.genai.mapping.shared.databricks_ai_gateway import (
    create_openai_client_for_databricks_gateway,
)
from edvise.genai.mapping.shared.pipeline_artifacts import coerce_pipeline_version
from edvise.utils.llm_utils import LLMRetryExhausted, llm_complete_with_parse_retry

load_dotenv()

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── config ───────────────────────────────────────────────────────────────────
# Institution configuration - swap these to change target/reference roles
TARGET_INSTITUTION = {
    "id": "synthetic_coastal_cc",
    "name": "Synthetic Coastal Community College",
}

REFERENCE_INSTITUTION = {
    "id": "synthetic_metro_research_uni",
    "name": "Synthetic Metro Research University",
}

# To swap roles (e.g., evaluate reference as target), swap the above:
# TARGET_INSTITUTION = {
#     "id": "synthetic_metro_research_uni",
#     "name": "Synthetic Metro Research University",
# }
# REFERENCE_INSTITUTION = {
#     "id": "synthetic_coastal_cc",
#     "name": "Synthetic Coastal Community College",
# }

MODELS = [
    "claude-opus-test-genai-ai-data-cleaning",
    "claude-sonnet-test-genai-ai-data-cleaning",
    "claude-haiku-test-genai-data-cleaning",
]

# Short label per gateway model → output folders like {base}_2a_{SHOT_TAG}, {base}_2b_{SHOT_TAG}
MODEL_BASE_SLUG = {
    "claude-opus-test-genai-ai-data-cleaning": "opus",
    "claude-sonnet-test-genai-ai-data-cleaning": "sonnet",
    "claude-haiku-test-genai-data-cleaning": "haiku",
}

# Set to "2shot" (or another tag) before run() when evaluating a multi-turn / 2-shot prompt variant
SHOT_TAG = "1shot"

# Two LLM calls (cohort then course), merged to the same full-manifest JSON shape as single-pass.
# Set False to use build_step2a_prompt + one run_once per model (legacy).
STEP2A_TWO_PASS = True


def _eval_step2a_max_envelope_attempts() -> int:
    """Upper bound on Step 2a envelope parse/validate retries (single- and two-pass)."""
    raw = os.environ.get("EDVISE_EVAL_STEP2A_MAX_ENVELOPE_ATTEMPTS", "").strip()
    if not raw:
        return 3
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _round_trip_validate_saved_envelope_json(out_json: str) -> None:
    """Ensure manifest JSON written for eval re-parses as a valid envelope (matches scoring)."""
    pred = json.loads(out_json)
    MappingManifestEnvelope.model_validate(pred)


# After Step 2a, run the same SMA refinement as onboard (``run_sma_refinement`` per entity),
# write ``*_mapping_manifest_refined.json``, score vs gold as ``refined_*`` columns, and extend
# field_detail CSV with ``stage`` = ``2a`` / ``after_refinement``.
RUN_REFINEMENT_AFTER_2A = True


def _eval_write_hitl_artifacts_to_workspace() -> bool:
    """When true, refinement also writes ``cohort_hitl_manifest.json`` / ``course_hitl_manifest.json``."""
    raw = os.environ.get("EDVISE_EVAL_WRITE_HITL_ARTIFACTS", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def folder_slug_2a(model: str) -> str:
    """Per-model artifact folder name under ``…/genai_mapping/eval/`` for Step 2a."""
    base = MODEL_BASE_SLUG.get(model, model.replace("-", "_"))
    return f"{base}_2a_{SHOT_TAG}"


def folder_slug_2b(model: str) -> str:
    """Per-model artifact folder name under ``…/genai_mapping/eval/`` for Step 2b."""
    base = MODEL_BASE_SLUG.get(model, model.replace("-", "_"))
    return f"{base}_2b_{SHOT_TAG}"


# Models that support assistant message prefill (for JSON output formatting)
MODELS_WITH_PREFILL = {
    "claude-sonnet-test-genai-ai-data-cleaning",
    "claude-haiku-test-genai-data-cleaning",
}


# ── inference ─────────────────────────────────────────────────────────────────
def run_once(
    model: str,
    prompt: str,
    client: OpenAI,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        full_text = ""

        # Build messages - only add assistant prefill for models that support it
        messages_list: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        if model in MODELS_WITH_PREFILL:
            messages_list.append(
                {"role": "assistant", "content": "{"}
            )  # prefill for JSON
        messages = cast(list[ChatCompletionMessageParam], messages_list)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 16000,
            "stream": True,
        }
        resp = client.chat.completions.create(**create_kwargs)
        for chunk in resp:
            # Safely extract content from chunk - handle cases where choices might be empty
            try:
                if not isinstance(chunk, ChatCompletionChunk):
                    continue
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    print(delta, end="", flush=True)
                    full_text += delta
            except (IndexError, AttributeError) as e:
                # Skip chunks with malformed structure - this can happen with some models
                logger.debug(f"Skipping malformed chunk: {e}")
                continue

        latency_s = time.perf_counter() - start

        # For models with prefill, reattach the brace if needed; for others, just strip fences
        if model in MODELS_WITH_PREFILL:
            # The model's response continues from the prefill "{", so it may already start with "{"
            # Only prepend "{" if the response doesn't already start with it (after stripping whitespace)
            full_text_stripped = full_text.lstrip()
            if full_text_stripped.startswith("{"):
                raw = strip_json_fences(full_text)
            else:
                raw = strip_json_fences("{" + full_text)
        else:
            raw = strip_json_fences(full_text)
        return {
            "model": model,
            "prompt": prompt,
            "success": True,
            "latency_s": round(latency_s, 3),
            "response_chars": len(raw),
            "response": raw,
            "error": None,
        }
    except Exception as e:
        # Handle all errors - extract as much detail as possible
        latency_s = time.perf_counter() - start
        error_msg = str(e)
        error_type = type(e).__name__

        # Try to extract HTTP status code and response body
        status_code = getattr(e, "status_code", None)
        error_details = error_msg

        # Check for response body in various formats
        response_body = None
        if hasattr(e, "response"):
            if hasattr(e.response, "text"):
                try:
                    response_body = e.response.text
                except:
                    pass
            elif hasattr(e.response, "json"):
                try:
                    response_body = str(e.response.json())
                except:
                    pass

        if not response_body and hasattr(e, "body"):
            try:
                response_body = str(e.body)
            except:
                pass

        # Build detailed error message
        if status_code:
            error_details = f"HTTP {status_code}: {error_msg}"
        else:
            error_details = f"{error_type}: {error_msg}"

        if response_body:
            error_details = f"{error_details}\nResponse body: {response_body}"

        # Also check for additional error attributes
        if hasattr(e, "message") and e.message != error_msg:
            error_details = f"{error_details}\nMessage: {e.message}"

        logger.error(f"Error calling model {model}: {error_details}")

        return {
            "model": model,
            "prompt": prompt,
            "success": False,
            "latency_s": round(latency_s, 3),
            "response_chars": None,
            "response": None,
            "error": error_details,
        }


def _make_eval_step2a_llm_complete(
    model: str,
    client: OpenAI,
    latencies: list[float],
) -> Callable[[str, str], str]:
    """``(system, user) -> text`` for Step 2a parse-retry (same composition as onboard SMA)."""

    def llm_complete(system: str, user: str) -> str:
        s = (system or "").strip()
        u = (user or "").strip()
        if s and u:
            combined = f"{s}\n\n---\n\n{u}"
        elif u:
            combined = u
        elif s:
            combined = s
        else:
            raise RuntimeError(
                "SMA eval Step 2a LLM call has empty system and user prompts"
            )
        out = run_once(model, combined, client)
        latencies.append(float(out.get("latency_s") or 0.0))
        if not out["success"]:
            raise RuntimeError(out.get("error") or "eval Step 2a LLM call failed")
        resp = out.get("response")
        if not isinstance(resp, str) or not resp.strip():
            raise RuntimeError("eval Step 2a LLM returned empty response")
        return resp

    return llm_complete


def _eval_step2a_parse_envelope(
    raw: str,
    *,
    institution_id: str,
    pipeline_version: str,
) -> MappingManifestEnvelope:
    manifest_dict = json.loads(raw)
    if isinstance(manifest_dict, dict):
        manifest_dict["institution_id"] = institution_id
        manifest_dict["pipeline_version"] = pipeline_version
    return MappingManifestEnvelope.model_validate(manifest_dict)


def run_step2a_single_pass(
    model: str,
    client: OpenAI,
    prompt: str,
    *,
    institution_id: str,
    pipeline_version: str | None = None,
) -> dict[str, Any]:
    """
    Step 2a single-pass with :func:`~edvise.utils.llm_utils.llm_complete_with_parse_retry`
    (same pattern as ``edvise_genai_sma`` onboard): retry on JSON / Pydantic envelope errors.
    """
    max_attempts = _eval_step2a_max_envelope_attempts()
    logger.info(
        "Eval Step 2a single-pass: parse-retry (max LLM calls=%s) — "
        "override with EDVISE_EVAL_STEP2A_MAX_ENVELOPE_ATTEMPTS",
        max_attempts,
    )
    pv = coerce_pipeline_version(pipeline_version)
    latencies: list[float] = []
    llm = _make_eval_step2a_llm_complete(model, client, latencies)

    def parse_fn(raw: str) -> MappingManifestEnvelope:
        return _eval_step2a_parse_envelope(
            raw, institution_id=institution_id, pipeline_version=pv
        )

    try:
        envelope = llm_complete_with_parse_retry(
            llm,
            "",
            prompt,
            parse_fn,
            max_retries=max_attempts,
            logger=logger,
        )
    except LLMRetryExhausted as e:
        last_raw = e.last_raw_response
        return {
            "model": model,
            "prompt": prompt,
            "success": False,
            "latency_s": round(sum(latencies), 3),
            "response_chars": len(last_raw) if last_raw else None,
            "response": last_raw,
            "error": str(e),
            "step2a_llm_calls": len(latencies),
            "step2a_envelope_attempts_used": max_attempts,
            "step2a_parse_retry_enabled": True,
        }

    out = envelope.model_dump_json(indent=2, exclude_none=True)
    try:
        _round_trip_validate_saved_envelope_json(out)
    except ValidationError as e:
        logger.error(
            "Eval Step 2a single-pass: round-trip envelope validation failed: %s", e
        )
        return {
            "model": model,
            "prompt": prompt,
            "success": False,
            "latency_s": round(sum(latencies), 3),
            "response_chars": len(out),
            "response": out,
            "error": f"envelope round-trip validation failed: {e}",
            "step2a_llm_calls": len(latencies),
            "step2a_envelope_attempts_used": len(latencies),
            "step2a_parse_retry_enabled": True,
        }

    logger.info(
        "Eval Step 2a single-pass: success after %s LLM call(s) (envelope OK)",
        len(latencies),
    )
    return {
        "model": model,
        "prompt": prompt,
        "success": True,
        "latency_s": round(sum(latencies), 3),
        "response_chars": len(out),
        "response": out,
        "error": None,
        "step2a_llm_calls": len(latencies),
        "step2a_envelope_attempts_used": len(latencies),
        "step2a_parse_retry_enabled": True,
    }


def run_step2a_two_pass(
    model: str,
    client: OpenAI,
    prompt_cohort: str,
    prompt_course: str,
    *,
    institution_id: str | None = None,
    pipeline_version: str | None = None,
) -> dict:
    """
    Step 2a: two gateway calls (cohort, then course), merge, then
    :class:`~.schemas.MappingManifestEnvelope` validation.

    On JSON / merge / Pydantic failures, both passes are re-run with a correction hint
    (up to 3 attempts total, same default as :func:`~edvise.utils.llm_utils.call_with_retry`).

    ``institution_id`` is required for merge when fragments omit it (eval always passes it).
    """
    max_attempts = _eval_step2a_max_envelope_attempts()
    logger.info(
        "Eval Step 2a two-pass: parse-retry (max envelope attempt(s)=%s, up to %s LLM calls) — "
        "override count with EDVISE_EVAL_STEP2A_MAX_ENVELOPE_ATTEMPTS",
        max_attempts,
        max_attempts * 2,
    )
    sep = "\n\n=== STEP 2a PASS 2 (course) ===\n\n"
    prompt_combined = (
        f"=== STEP 2a PASS 1 (cohort) ===\n\n{prompt_cohort}{sep}{prompt_course}"
    )
    if not institution_id:
        return {
            "model": model,
            "prompt": prompt_combined,
            "success": False,
            "latency_s": 0.0,
            "response_chars": None,
            "response": None,
            "error": "institution_id is required for two-pass Step 2a merge",
            "step2a_llm_calls": 0,
            "step2a_envelope_attempts_used": 0,
            "step2a_parse_retry_enabled": True,
        }

    pv = coerce_pipeline_version(pipeline_version)
    latencies: list[float] = []
    llm = _make_eval_step2a_llm_complete(model, client, latencies)

    correction_hint: str | None = None
    last_merged_raw = ""
    last_err: str | None = None

    for attempt in range(1, max_attempts + 1):
        logger.info(
            "Eval Step 2a two-pass: envelope attempt %s/%s (LLM calls so far: %s)",
            attempt,
            max_attempts,
            len(latencies),
        )
        try:
            suffix = f"\n\n{correction_hint}" if correction_hint else ""
            cohort_raw = llm("", prompt_cohort + suffix)
            course_raw = llm("", prompt_course + suffix)
            cohort_parsed = json.loads(cohort_raw)
            course_parsed = json.loads(course_raw)
            merged = merge_step2a_entity_manifests(
                cohort_parsed,
                course_parsed,
                institution_id=institution_id,
                pipeline_version=pv,
            )
            last_merged_raw = json.dumps(merged)
            envelope = MappingManifestEnvelope.model_validate(merged)
            out = envelope.model_dump_json(indent=2, exclude_none=True)
            _round_trip_validate_saved_envelope_json(out)
        except json.JSONDecodeError as e:
            last_err = str(e)
            logger.warning(
                "Eval Step 2a two-pass attempt %s/%s failed: JSONDecodeError: %s",
                attempt,
                max_attempts,
                e,
            )
            if attempt == max_attempts:
                break
            correction_hint = (
                f"The cohort and/or course response was not valid JSON: {e}\n"
                "Return corrected JSON for **both** the cohort-only and the course-only manifest."
            )
        except ValueError as e:
            last_err = str(e)
            logger.warning(
                "Eval Step 2a two-pass attempt %s/%s failed: ValueError: %s",
                attempt,
                max_attempts,
                e,
            )
            if attempt == max_attempts:
                break
            correction_hint = (
                f"Merging cohort and course fragments failed: {e}\n"
                "Fix the structure so each pass matches the expected entity shape, then try again."
            )
        except ValidationError as e:
            last_err = str(e)
            logger.warning(
                "Eval Step 2a two-pass attempt %s/%s failed: ValidationError: %s",
                attempt,
                max_attempts,
                e,
            )
            if attempt == max_attempts:
                break
            correction_hint = (
                f"The merged manifest from your two passes was:\n\n{last_merged_raw}\n\n"
                f"It failed validation with this error:\n\n{e}\n\n"
                "Return corrected cohort JSON and course JSON so the merged envelope validates."
            )
        else:
            logger.info(
                "Eval Step 2a two-pass: success on envelope attempt %s/%s "
                "(%s LLM calls, latency sum=%.3fs)",
                attempt,
                max_attempts,
                len(latencies),
                sum(latencies),
            )
            return {
                "model": model,
                "prompt": prompt_combined,
                "success": True,
                "latency_s": round(sum(latencies), 3),
                "response_chars": len(out),
                "response": out,
                "error": None,
                "step2a_llm_calls": len(latencies),
                "step2a_envelope_attempts_used": attempt,
                "step2a_parse_retry_enabled": True,
            }

    logger.error(
        "Eval Step 2a two-pass: exhausted %s envelope attempt(s) (%s LLM calls)",
        max_attempts,
        len(latencies),
    )
    return {
        "model": model,
        "prompt": prompt_combined,
        "success": False,
        "latency_s": round(sum(latencies), 3),
        "response_chars": len(last_merged_raw) if last_merged_raw else None,
        "response": last_merged_raw or None,
        "error": last_err or "Step 2a two-pass retries exhausted",
        "step2a_llm_calls": len(latencies),
        "step2a_envelope_attempts_used": max_attempts,
        "step2a_parse_retry_enabled": True,
    }


def _make_refinement_llm_complete(
    model: str,
    client: OpenAI,
    latencies: list[float],
) -> Callable[[str, str], str]:
    """Match onboard SMA: single user blob with system and user sections (gateway pattern)."""

    def llm_complete(system: str, user: str) -> str:
        s = (system or "").strip()
        u = (user or "").strip()
        if s and u:
            combined = f"{s}\n\n---\n\n{u}"
        elif u:
            combined = u
        elif s:
            combined = s
        else:
            raise RuntimeError(
                "SMA refinement LLM call has empty system and user prompts"
            )
        out = run_once(model, combined, client)
        latencies.append(float(out.get("latency_s") or 0.0))
        if not out["success"]:
            raise RuntimeError(out.get("error") or "refinement LLM call failed")
        resp = out.get("response")
        if not isinstance(resp, str) or not resp.strip():
            raise RuntimeError("refinement LLM returned empty response")
        return resp

    return llm_complete


def run_refine_on_envelope(
    envelope: MappingManifestEnvelope,
    *,
    institution_id: str,
    schema_contract_sma: Any,
    model: str,
    client: OpenAI,
) -> tuple[MappingManifestEnvelope, dict[str, Any]]:
    """Apply production ``run_sma_refinement`` per entity (same order as onboard)."""
    refined = envelope.model_copy(deep=True)
    latencies: list[float] = []
    llm_complete = _make_refinement_llm_complete(model, client, latencies)
    hitl_items: dict[str, int] = {}
    hitl_by_entity: dict[str, Any] = {}
    struct_before: dict[str, int] = {}
    struct_after: dict[str, int] = {}

    for entity_key, entity_manifest in list(refined.manifests.items()):
        ek = entity_key.value if hasattr(entity_key, "value") else str(entity_key)
        errs = validate_manifest(entity_manifest, schema_contract_sma)
        struct_before[ek] = len(errs)
        refined_fm, hitl_env = run_sma_refinement(
            institution_id=institution_id,
            entity_type=cast(Literal["cohort", "course"], ek),
            manifest=entity_manifest,
            validation_errors=errs,
            schema_contract=schema_contract_sma,
            llm_complete=llm_complete,
        )
        refined.manifests[entity_key] = refined_fm
        hitl_items[ek] = len(hitl_env.items)
        hitl_by_entity[ek] = hitl_env
        errs2 = validate_manifest(refined_fm, schema_contract_sma)
        struct_after[ek] = len(errs2)

    return refined, {
        "refinement_latency_s": round(sum(latencies), 3),
        "hitl_items_by_entity": hitl_items,
        "hitl_by_entity": hitl_by_entity,
        "structural_errors_by_entity_2a": struct_before,
        "structural_errors_by_entity_after_refinement": struct_after,
    }


# ── metric calculation ────────────────────────────────────────────────────────
def _get_mappings(manifest: dict, entity: str) -> dict:
    """Return {target_field: mapping_dict} for a given entity type."""
    return {
        m["target_field"]: m
        for m in manifest.get("manifests", {}).get(entity, {}).get("mappings", [])
    }


def _get_alias_map(manifest: dict, entity: str) -> dict:
    aliases = (
        manifest.get("manifests", {}).get(entity, {}).get("column_aliases", []) or []
    )
    return {
        (a.get("table"), a.get("source_column")): a.get("canonical_column")
        for a in aliases
    }


def _safe_div(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 3)


def _canonicalize_column(
    table: str | None, column: str | None, alias_map: dict
) -> str | None:
    if column is None:
        return None
    return cast(
        str | None,
        alias_map.get((table, column), column),
    )


def _lookup_table_unique_keys(
    schema_contract: dict | None, lookup_table: str | None
) -> list[str] | None:
    """Return declared unique_keys for a dataset table from a schema contract, if any."""
    if not schema_contract or not lookup_table:
        return None
    ds = (schema_contract.get("datasets") or {}).get(lookup_table) or {}
    uks = ds.get("unique_keys")
    if not uks:
        return None
    return list(uks)


def _is_null_mapping(mapping: dict | None) -> bool:
    mapping = mapping or {}
    return mapping.get("source_column") is None and mapping.get("source_table") is None


def _normalize_join(
    join: dict | None, schema_contract: dict | None = None
) -> dict | None:
    if not join:
        return None
    keys = list(join.get("join_keys", []))
    lookup = join.get("lookup_table")
    uks = _lookup_table_unique_keys(schema_contract, lookup)
    if uks:
        uk_set = set(uks)
        # Drop keys not in the lookup table's declared PK (e.g. spurious section_number
        # or section_id when the contract unique_keys are a shorter list such as term +
        # course_reference_number only).
        keys = [k for k in keys if k in uk_set]
    return {
        "base_table": join.get("base_table"),
        "lookup_table": join.get("lookup_table"),
        "join_keys": sorted(keys),
    }


def _normalize_row_selection(mapping: dict | None) -> tuple:
    mapping = mapping or {}
    rs = mapping.get("row_selection") or {}
    return (
        rs.get("strategy"),
        rs.get("order_by"),
        rs.get("condition_col"),
        json.dumps(rs.get("condition"), sort_keys=True, default=str),
        json.dumps(rs.get("filter"), sort_keys=True, default=str),
    )


def _has_valid_join(mapping: dict | None) -> bool:
    join = (mapping or {}).get("join")
    return bool(
        join and all(k in join for k in ("base_table", "lookup_table", "join_keys"))
    )


def _has_degree_filter(mapping: dict | None) -> bool:
    rs = (mapping or {}).get("row_selection") or {}
    return rs.get("strategy") == "where_not_null" or "condition" in rs or "filter" in rs


def _needs_row_selection_check(gold_mapping: dict) -> bool:
    """True when gold specifies any row_selection strategy (including any_row)."""
    rs = (gold_mapping or {}).get("row_selection") or {}
    return rs.get("strategy") is not None


def _needs_row_selection_check_non_anyrow(gold_mapping: dict) -> bool:
    """True when gold row_selection is a non-trivial strategy (excludes any_row)."""
    rs = (gold_mapping or {}).get("row_selection") or {}
    return rs.get("strategy") not in (None, "any_row")


def _is_degree_like_field(field: str) -> bool:
    return any(
        x in field for x in ("grad_date", "certificate", "degree_grad", "major_grad")
    )


def _normalize_mapping(
    mapping: dict | None,
    alias_map: dict,
    schema_contract: dict | None = None,
) -> dict:
    """
    Canonical representation for strict equality checks.
    Ignores rationale/confidence/review metadata.
    """
    m = deepcopy(mapping or {})
    # TODO: review_status is telemetry only — do not use for pipeline routing
    for k in ("confidence", "rationale", "review_status", "validation_notes"):
        m.pop(k, None)

    m["source_column"] = _canonicalize_column(
        m.get("source_table"),
        m.get("source_column"),
        alias_map,
    )

    if "join" in m:
        m["join"] = _normalize_join(m.get("join"), schema_contract)

    return m


def _prepare_mapping(
    mapping: dict | None,
    alias_map: dict,
    schema_contract: dict | None = None,
) -> dict:
    """
    Compact normalized view used for all comparisons.
    """
    mapping = mapping or {}
    source_table = mapping.get("source_table")
    source_column = mapping.get("source_column")

    return {
        "raw": mapping,
        "is_null": _is_null_mapping(mapping),
        "source_table": source_table,
        "source_column": _canonicalize_column(source_table, source_column, alias_map),
        "join": _normalize_join(mapping.get("join"), schema_contract),
        "row_selection": _normalize_row_selection(mapping),
        "normalized": _normalize_mapping(mapping, alias_map, schema_contract),
    }


def _compare_field(
    field: str,
    gold_mapping: dict,
    pred_mapping: dict,
    gold_alias_map: dict,
    pred_alias_map: dict,
    schema_contract: dict | None = None,
) -> dict:
    """
    Compare one field and return all field-level booleans/flags.
    """
    g = _prepare_mapping(gold_mapping, gold_alias_map, schema_contract)
    p = _prepare_mapping(pred_mapping, pred_alias_map, schema_contract)

    g_mappable = not g["is_null"]
    p_mappable = not p["is_null"]

    map_decision_correct = g["is_null"] == p["is_null"]

    source_column_correct = None
    source_table_correct = None
    source_exact = None

    if g_mappable and p_mappable:
        source_column_correct = int(p["source_column"] == g["source_column"])
        source_table_correct = int(p["source_table"] == g["source_table"])
        source_exact = int(source_column_correct and source_table_correct)

    g_has_join = g["join"] is not None
    p_has_join = p["join"] is not None

    join_presence_correct = int(g_has_join == p_has_join)
    join_tables_correct = None
    join_keys_correct = None

    if g_has_join and p_has_join:
        join_tables_correct = int(
            p["join"]["base_table"] == g["join"]["base_table"]
            and p["join"]["lookup_table"] == g["join"]["lookup_table"]
        )
        join_keys_correct = int(p["join"]["join_keys"] == g["join"]["join_keys"])

    row_selection_correct = None
    if _needs_row_selection_check(gold_mapping):
        row_selection_correct = int(p["row_selection"] == g["row_selection"])

    row_selection_correct_non_anyrow = None
    if _needs_row_selection_check_non_anyrow(gold_mapping):
        row_selection_correct_non_anyrow = int(p["row_selection"] == g["row_selection"])

    degree_filter_correct = None
    if _is_degree_like_field(field) and g_mappable:
        degree_filter_correct = int(
            _has_degree_filter(pred_mapping) == _has_degree_filter(gold_mapping)
        )

    exact_field_match = int(p["normalized"] == g["normalized"])

    execution_ready = False
    if g["is_null"] and p["is_null"]:
        execution_ready = True
    elif g_mappable and p_mappable and source_exact == 1:
        row_ok = (
            row_selection_correct == 1 if row_selection_correct is not None else True
        )
        join_ok = True
        if g_has_join or p_has_join:
            join_ok = (
                g_has_join
                and p_has_join
                and join_tables_correct == 1
                and join_keys_correct == 1
            )
        degree_ok = (
            degree_filter_correct == 1 if degree_filter_correct is not None else True
        )
        execution_ready = row_ok and join_ok and degree_ok

    return {
        "field": field,
        "gold_mappable": g_mappable,
        "pred_mappable": p_mappable,
        "map_decision_correct": map_decision_correct,
        "source_column_correct": source_column_correct,
        "source_table_correct": source_table_correct,
        "source_exact": source_exact,
        "gold_has_join": g_has_join,
        "pred_has_join": p_has_join,
        "join_presence_correct": join_presence_correct,
        "join_tables_correct": join_tables_correct,
        "join_keys_correct": join_keys_correct,
        "row_selection_checked": _needs_row_selection_check(gold_mapping),
        "row_selection_correct": row_selection_correct,
        "row_selection_checked_non_anyrow": _needs_row_selection_check_non_anyrow(
            gold_mapping
        ),
        "row_selection_correct_non_anyrow": row_selection_correct_non_anyrow,
        "degree_filter_checked": _is_degree_like_field(field) and g_mappable,
        "degree_filter_correct": degree_filter_correct,
        "exact_field_match": exact_field_match,
        "execution_ready": execution_ready,
    }


def _update_counts(counts: Counter, cmp: dict) -> None:
    """
    Single place where field comparisons roll up into counters.
    """
    counts["n_fields"] += 1
    counts["map_decision_correct"] += int(cmp["map_decision_correct"])

    g_mappable = cmp["gold_mappable"]
    p_mappable = cmp["pred_mappable"]

    if g_mappable:
        counts["gold_mappable"] += 1
    else:
        counts["gold_unmappable"] += 1

    if p_mappable:
        counts["pred_mappable"] += 1

    if g_mappable and p_mappable and cmp["source_exact"] == 1:
        counts["strict_tp"] += 1
    elif g_mappable and not p_mappable:
        counts["strict_fn"] += 1
    elif not g_mappable and p_mappable:
        counts["strict_fp"] += 1
    elif not g_mappable and not p_mappable:
        counts["strict_tn"] += 1
    elif g_mappable and p_mappable and cmp["source_exact"] == 0:
        counts["strict_fp"] += 1

    if p_mappable:
        counts["attempted_mappable"] += 1
        if cmp["source_column_correct"] == 1:
            counts["attempted_correct_col"] += 1
        if cmp["source_table_correct"] == 1:
            counts["attempted_correct_tbl"] += 1

    if cmp["gold_has_join"]:
        counts["join_gold_positive"] += 1
    if cmp["pred_has_join"]:
        counts["join_pred_positive"] += 1
    if cmp["gold_has_join"] and cmp["pred_has_join"]:
        counts["join_true_positive"] += 1
        if cmp["join_tables_correct"] == 1:
            counts["join_table_correct"] += 1
        if cmp["join_keys_correct"] == 1:
            counts["join_key_correct"] += 1
    if (not cmp["gold_has_join"]) and cmp["pred_has_join"]:
        counts["join_hallucinations"] += 1

    if cmp["row_selection_checked"]:
        counts["row_selection_needed"] += 1
        if cmp["row_selection_correct"] == 1:
            counts["row_selection_correct"] += 1

    if cmp["row_selection_checked_non_anyrow"]:
        counts["row_selection_non_anyrow_needed"] += 1
        if cmp["row_selection_correct_non_anyrow"] == 1:
            counts["row_selection_non_anyrow_correct"] += 1

    if cmp["degree_filter_checked"]:
        counts["degree_filter_needed"] += 1
        if cmp["degree_filter_correct"] == 1:
            counts["degree_filter_correct"] += 1

    counts["exact_field_match"] += int(cmp["exact_field_match"])
    counts["execution_ready"] += int(cmp["execution_ready"])


def _finalize_counts(entity: str, counts: Counter, field_scores: list[dict]) -> dict:
    join_precision = _safe_div(
        counts["join_true_positive"], counts["join_pred_positive"]
    )
    join_recall = _safe_div(counts["join_true_positive"], counts["join_gold_positive"])

    return {
        "entity": entity,
        "n_fields": counts["n_fields"],
        "map_decision_accuracy": _safe_div(
            counts["map_decision_correct"], counts["n_fields"]
        ),
        "strict_tp": counts["strict_tp"],
        "strict_fp": counts["strict_fp"],
        "strict_fn": counts["strict_fn"],
        "strict_tn": counts["strict_tn"],
        "mappable_precision_strict": _safe_div(
            counts["strict_tp"],
            counts["strict_tp"] + counts["strict_fp"],
        ),
        "mappable_recall_strict": _safe_div(
            counts["strict_tp"],
            counts["strict_tp"] + counts["strict_fn"],
        ),
        "unmappable_precision_strict": _safe_div(
            counts["strict_tn"],
            counts["strict_tn"] + counts["strict_fn"],
        ),
        "unmappable_recall_strict": _safe_div(
            counts["strict_tn"],
            counts["strict_tn"] + counts["strict_fp"],
        ),
        "source_column_accuracy_attempted": _safe_div(
            counts["attempted_correct_col"],
            counts["attempted_mappable"],
        ),
        "source_table_accuracy_attempted": _safe_div(
            counts["attempted_correct_tbl"],
            counts["attempted_mappable"],
        ),
        "source_exact_accuracy_gold_mappable": _safe_div(
            counts["strict_tp"],
            counts["gold_mappable"],
        ),
        "row_selection_accuracy": _safe_div(
            counts["row_selection_correct"],
            counts["row_selection_needed"],
        ),
        "row_selection_accuracy_non_anyrow": _safe_div(
            counts["row_selection_non_anyrow_correct"],
            counts["row_selection_non_anyrow_needed"],
        ),
        "join_precision": join_precision,
        "join_recall": join_recall,
        "join_f1": _f1(join_precision, join_recall),
        "join_table_accuracy": _safe_div(
            counts["join_table_correct"],
            counts["join_true_positive"],
        ),
        "join_key_accuracy": _safe_div(
            counts["join_key_correct"],
            counts["join_true_positive"],
        ),
        "join_hallucination_rate": _safe_div(
            counts["join_hallucinations"],
            counts["n_fields"],
        ),
        "degree_filter_accuracy": _safe_div(
            counts["degree_filter_correct"],
            counts["degree_filter_needed"],
        ),
        "exact_field_match_rate": _safe_div(
            counts["exact_field_match"],
            counts["n_fields"],
        ),
        "execution_ready_rate": _safe_div(
            counts["execution_ready"],
            counts["n_fields"],
        ),
        "field_scores": field_scores,
    }


def score_manifest_v2(
    pred: dict,
    gold: dict,
    entity: str,
    schema_contract: dict | None = None,
) -> dict:
    pred_mappings = _get_mappings(pred, entity)
    gold_mappings = _get_mappings(gold, entity)

    pred_alias_map = _get_alias_map(pred, entity)
    gold_alias_map = _get_alias_map(gold, entity)

    counts: Counter[str] = Counter()
    field_scores = []

    for field, gold_mapping in gold_mappings.items():
        pred_mapping = pred_mappings.get(field, {})
        cmp = _compare_field(
            field=field,
            gold_mapping=gold_mapping,
            pred_mapping=pred_mapping,
            gold_alias_map=gold_alias_map,
            pred_alias_map=pred_alias_map,
            schema_contract=schema_contract,
        )
        cmp["entity"] = entity
        field_scores.append(cmp)
        _update_counts(counts, cmp)

    return _finalize_counts(entity, counts, field_scores)


def score_result(
    result: dict,
    gold_manifest: dict,
    schema_contract: dict | None = None,
) -> dict | None:
    """Parse response JSON and score against gold. Returns None on parse failure."""
    if not result["success"]:
        return None
    try:
        pred = json.loads(result["response"])
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed for {result['model']}: {e}")
        return None

    is_valid, validation_error = validate_envelope_dict(pred)
    if not is_valid:
        logger.warning(
            f"Pydantic validation failed for {result['model']}: {validation_error}"
        )

    cohort_scores = score_manifest_v2(pred, gold_manifest, "cohort", schema_contract)
    course_scores = score_manifest_v2(pred, gold_manifest, "course", schema_contract)

    def _avg(*vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "validation_passed": is_valid,
        "validation_error": validation_error,
        "cohort": cohort_scores,
        "course": course_scores,
        "overall": {
            "map_decision_accuracy": _avg(
                cohort_scores["map_decision_accuracy"],
                course_scores["map_decision_accuracy"],
            ),
            "mappable_precision_strict": _avg(
                cohort_scores["mappable_precision_strict"],
                course_scores["mappable_precision_strict"],
            ),
            "mappable_recall_strict": _avg(
                cohort_scores["mappable_recall_strict"],
                course_scores["mappable_recall_strict"],
            ),
            "source_exact_accuracy_gold_mappable": _avg(
                cohort_scores["source_exact_accuracy_gold_mappable"],
                course_scores["source_exact_accuracy_gold_mappable"],
            ),
            "row_selection_accuracy": _avg(
                cohort_scores["row_selection_accuracy"],
                course_scores["row_selection_accuracy"],
            ),
            "row_selection_accuracy_non_anyrow": _avg(
                cohort_scores["row_selection_accuracy_non_anyrow"],
                course_scores["row_selection_accuracy_non_anyrow"],
            ),
            "join_f1": _avg(cohort_scores["join_f1"], course_scores["join_f1"]),
            "join_table_accuracy": _avg(
                cohort_scores["join_table_accuracy"],
                course_scores["join_table_accuracy"],
            ),
            "join_key_accuracy": _avg(
                cohort_scores["join_key_accuracy"], course_scores["join_key_accuracy"]
            ),
            "degree_filter_accuracy": _avg(
                cohort_scores["degree_filter_accuracy"],
                course_scores["degree_filter_accuracy"],
            ),
            "exact_field_match_rate": _avg(
                cohort_scores["exact_field_match_rate"],
                course_scores["exact_field_match_rate"],
            ),
            "execution_ready_rate": _avg(
                cohort_scores["execution_ready_rate"],
                course_scores["execution_ready_rate"],
            ),
        },
    }


def validate_envelope_dict(manifest_dict: dict) -> tuple[bool, str | None]:
    """
    Attempt to parse a raw manifest dict against MappingManifestEnvelope.
    Returns (is_valid, error_message).
    """
    try:
        MappingManifestEnvelope.model_validate(manifest_dict)
        return True, None
    except ValidationError as e:
        return False, str(e)


def rescore_step2a_after_hitl_resolution(
    *,
    refined_envelope_path: str | Path,
    gold_manifest: dict,
    target_schema_contract: dict | None,
    model_label: str = "post_hitl",
    output_post_hitl_path: str | Path | None = None,
    resolved_by: str = "eval_hitl_notebook",
    require_hitl_gate_clear: bool = True,
) -> tuple[dict | None, Path]:
    """
    Copy the refined mapping envelope, apply cleared manifest HITL selections, score vs gold.

    Expects ``cohort_hitl_manifest.json`` and ``course_hitl_manifest.json`` in the same
    directory as ``refined_envelope_path`` (same layout as SMA onboard / Step 2a eval).

    Parameters
    ----------
    require_hitl_gate_clear
        When True, runs :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.hitl.check_sma_hitl_gate`
        on both HITL files before resolving (raises if any item is still pending).
    """
    refined_path = Path(refined_envelope_path).resolve()
    manifest_dir = refined_path.parent
    cohort_hitl = manifest_dir / "cohort_hitl_manifest.json"
    course_hitl = manifest_dir / "course_hitl_manifest.json"
    if not cohort_hitl.is_file() or not course_hitl.is_file():
        raise FileNotFoundError(
            "Missing eval HITL files (expected next to refined manifest): "
            f"{cohort_hitl} and/or {course_hitl}"
        )

    raw_env = json.loads(refined_path.read_text())
    institution_id = str(raw_env.get("institution_id") or "").strip()
    if not institution_id:
        raise ValueError(f"Missing institution_id in {refined_path}")

    out_path = (
        Path(output_post_hitl_path).resolve()
        if output_post_hitl_path is not None
        else manifest_dir / f"{institution_id}_mapping_manifest_post_hitl.json"
    )

    if require_hitl_gate_clear:
        check_sma_hitl_gate(cohort_hitl)
        check_sma_hitl_gate(course_hitl)

    shutil.copy2(refined_path, out_path)
    resolve_sma_items(
        cohort_hitl,
        out_path,
        resolved_by=resolved_by,
    )
    resolve_sma_items(
        course_hitl,
        out_path,
        resolved_by=resolved_by,
    )

    result = {
        "model": model_label,
        "success": True,
        "response": out_path.read_text(),
    }
    scores = score_result(
        result,
        gold_manifest,
        schema_contract=target_schema_contract,
    )
    return scores, out_path


def _write_eval_hitl_files_for_refinement(
    manifest_dir: Path,
    *,
    institution_id: str,
    hitl_by_entity: dict[str, Any],
) -> None:
    """Write per-entity HITL JSON next to refined manifest; seed empty files if missing."""
    for ek, hitl_env in hitl_by_entity.items():
        basename = (
            "cohort_hitl_manifest.json"
            if ek == "cohort"
            else "course_hitl_manifest.json"
        )
        hp = write_sma_hitl_artifact(manifest_dir, hitl_env, basename=basename)
        logger.info(
            "→ eval HITL envelope (%s, %d item(s)) → %s",
            ek,
            len(hitl_env.items),
            hp,
        )

    for entity_type, basename in (
        ("cohort", "cohort_hitl_manifest.json"),
        ("course", "course_hitl_manifest.json"),
    ):
        path = manifest_dir / basename
        if not path.is_file():
            write_sma_hitl_artifact(
                manifest_dir,
                InstitutionSMAHITLItems(
                    institution_id=institution_id,
                    entity_type=cast(Literal["cohort", "course"], entity_type),
                    items=[],
                ),
                basename=basename,
            )
            logger.info("→ seeded empty HITL envelope → %s", path)


def _eval_uc_catalog() -> str | None:
    """Workspace catalog for ``/Volumes/<catalog>/…`` (matches job ``DB_workspace``)."""
    for key in ("EDVISE_EVAL_UC_CATALOG", "GENAI_HITL_CATALOG", "DB_workspace"):
        v = os.environ.get(key)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _require_uc_catalog_for_eval() -> str:
    cat = _eval_uc_catalog()
    if not cat:
        raise ValueError(
            "SMA eval loads only from Unity Catalog volumes. Set one of "
            "EDVISE_EVAL_UC_CATALOG, GENAI_HITL_CATALOG, or DB_workspace."
        )
    return cat


def _silver_genai_root_path(institution_id: str) -> Path:
    """``…/{institution_id}_silver/silver_volume/genai_mapping`` on the UC volume."""
    return Path(
        genai_cfg.silver_genai_mapping_root(
            institution_id, catalog=_require_uc_catalog_for_eval()
        )
    )


def _eval_run_root(target_id: str) -> Path:
    """Writable eval workspace: ``<silver_genai_root(target)>/eval``."""
    return _silver_genai_root_path(target_id) / "eval"


def _resolve_gold_mapping_manifest_for_eval(target_id: str) -> Path:
    """
    Ground-truth manifest for Step 2a scoring.

    * ``EDVISE_EVAL_GOLD_MANIFEST_PATH`` — absolute file on a volume (or any path).
    * Else ``<silver_genai_root(target)>/eval/gold/manifest_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_GOLD_MANIFEST_PATH")
    if override and str(override).strip():
        p = Path(str(override).strip()).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"EDVISE_EVAL_GOLD_MANIFEST_PATH not found: {p}")
        return p.resolve()
    p = _eval_run_root(target_id) / "gold" / "manifest_map.json"
    if not p.is_file():
        raise FileNotFoundError(
            "Gold mapping manifest not found. Place ground-truth at "
            f"{p} (same shape as pipeline ``manifest_map.json``) or set "
            "EDVISE_EVAL_GOLD_MANIFEST_PATH."
        )
    return p


def _resolve_gold_transformation_map_for_eval(target_id: str) -> Path:
    """
    Ground-truth transformation map for Step 2b scoring.

    * ``EDVISE_EVAL_GOLD_TRANSFORMATION_MAP_PATH``
    * Else ``<silver_genai_root(target)>/eval/gold/transformation_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_GOLD_TRANSFORMATION_MAP_PATH")
    if override and str(override).strip():
        p = Path(str(override).strip()).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_GOLD_TRANSFORMATION_MAP_PATH not found: {p}"
            )
        return p.resolve()
    p = _eval_run_root(target_id) / "gold" / "transformation_map.json"
    if not p.is_file():
        raise FileNotFoundError(
            "Gold transformation map not found. Place ground-truth at "
            f"{p} or set EDVISE_EVAL_GOLD_TRANSFORMATION_MAP_PATH."
        )
    return p


def _resolve_target_mapping_manifest_for_eval(target_id: str) -> Path:
    """
    Target mapping manifest for Step 2b prompts (field plans).

    * ``EDVISE_EVAL_TARGET_MAPPING_MANIFEST_PATH``
    * Else ``<silver_genai_root(target)>/active/manifest_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_TARGET_MAPPING_MANIFEST_PATH")
    if override and str(override).strip():
        p = Path(str(override).strip()).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_TARGET_MAPPING_MANIFEST_PATH not found: {p}"
            )
        return p.resolve()
    p = _silver_genai_root_path(target_id) / "active" / "manifest_map.json"
    if not p.is_file():
        raise FileNotFoundError(
            "Target mapping manifest for 2b not found at pipeline active path "
            f"{p}. Promote SMA artifacts or set EDVISE_EVAL_TARGET_MAPPING_MANIFEST_PATH."
        )
    return p


def _resolve_reference_mapping_manifest(reference_id: str) -> Path:
    """
    Few-shot reference manifest for Step 2a.

    * ``EDVISE_EVAL_REFERENCE_MANIFEST_PATH`` — explicit file.
    * Else ``…/genai_mapping/active/manifest_map.json`` for the reference institution.
    """
    override = os.environ.get("EDVISE_EVAL_REFERENCE_MANIFEST_PATH")
    if override and str(override).strip():
        p = Path(str(override).strip()).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_REFERENCE_MANIFEST_PATH not found: {p}"
            )
        return p.resolve()
    p = _silver_genai_root_path(reference_id) / "active" / "manifest_map.json"
    if not p.is_file():
        raise FileNotFoundError(
            "Reference manifest not found at pipeline active path "
            f"{p}. Promote execute artifacts or set EDVISE_EVAL_REFERENCE_MANIFEST_PATH."
        )
    return p


def _resolve_reference_transformation_map_for_eval(reference_id: str) -> Path:
    """
    Few-shot reference transformation map for Step 2b.

    * ``EDVISE_EVAL_REFERENCE_TRANSFORMATION_MAP_PATH``
    * Else ``…/genai_mapping/active/transformation_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_REFERENCE_TRANSFORMATION_MAP_PATH")
    if override and str(override).strip():
        p = Path(str(override).strip()).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_REFERENCE_TRANSFORMATION_MAP_PATH not found: {p}"
            )
        return p.resolve()
    p = _silver_genai_root_path(reference_id) / "active" / "transformation_map.json"
    if not p.is_file():
        raise FileNotFoundError(
            "Reference transformation map not found at pipeline active path "
            f"{p}. Set EDVISE_EVAL_REFERENCE_TRANSFORMATION_MAP_PATH or promote artifacts."
        )
    return p


def _resolve_target_schema_contract_for_eval(target_id: str) -> Path:
    """
    Target enriched schema contract for Step 2a / 2b.

    * ``EDVISE_EVAL_TARGET_SCHEMA_CONTRACT_PATH``
    * Else ``…/genai_mapping/active/enriched_schema_contract.json``.
    """
    override = os.environ.get("EDVISE_EVAL_TARGET_SCHEMA_CONTRACT_PATH")
    if override and str(override).strip():
        p = Path(str(override).strip()).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_TARGET_SCHEMA_CONTRACT_PATH not found: {p}"
            )
        return p.resolve()
    p = _silver_genai_root_path(target_id) / "active" / "enriched_schema_contract.json"
    if not p.is_file():
        raise FileNotFoundError(
            "Target schema contract not found at pipeline active path "
            f"{p}. Run IA / promote ``enriched_schema_contract.json`` or set "
            "EDVISE_EVAL_TARGET_SCHEMA_CONTRACT_PATH."
        )
    return p


# ── main execution ───────────────────────────────────────────────────────────
def run():
    """Run evaluation on all models."""
    # ── paths ─────────────────────────────────────────────────────────────
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_id = TARGET_INSTITUTION["id"]
    reference_id = REFERENCE_INSTITUTION["id"]
    catalog = _require_uc_catalog_for_eval()

    logger.info("=" * 80)
    logger.info("EVALUATION CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Target institution_id: {target_id}")
    logger.info(f"Reference institution_id: {reference_id}")
    logger.info(f"UC catalog: {catalog}")
    logger.info("=" * 80)

    EVAL_BASE = _eval_run_root(target_id)
    EVAL_BASE.mkdir(parents=True, exist_ok=True)
    (EVAL_BASE / "scratch").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = EVAL_BASE / "eval_outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── gold manifest ─────────────────────────────────────────────────────
    GOLD_MANIFEST_PATH = _resolve_gold_mapping_manifest_for_eval(target_id)
    GOLD_MANIFEST = load_json(str(GOLD_MANIFEST_PATH))
    logger.info(f"Loaded gold manifest from {GOLD_MANIFEST_PATH}")

    # ── prompt (via shared builder) ───────────────────────────────────────
    target_contract_path = _resolve_target_schema_contract_for_eval(target_id)
    target_contract = load_json(str(target_contract_path))
    logger.info(f"Loaded target schema contract from {target_contract_path}")
    schema_contract_sma = parse_enriched_schema_contract_for_sma(target_contract)

    REFERENCE_MANIFEST_PATH = _resolve_reference_mapping_manifest(reference_id)
    reference_manifest = load_json(str(REFERENCE_MANIFEST_PATH))
    reference_manifest_path = str(REFERENCE_MANIFEST_PATH)

    # Validate that the reference manifest has the correct institution_id
    manifest_institution_id = reference_manifest.get("institution_id")
    if manifest_institution_id != reference_id:
        raise ValueError(
            f"Reference manifest institution_id mismatch! "
            f"Expected '{reference_id}' but manifest contains '{manifest_institution_id}'. "
            f"Loaded from: {reference_manifest_path}"
        )

    logger.info(f"Loaded reference manifest from {reference_manifest_path}")
    logger.info(f"  Reference institution_id: {reference_id}")
    logger.info(f"  Manifest institution_id: {manifest_institution_id} ✓")

    output_path = str(EVAL_BASE / "scratch" / f"{target_id}_mapping_manifest.json")
    if STEP2A_TWO_PASS:
        PROMPT_COHORT = build_step2a_prompt_cohort_pass(
            institution_id=target_id,
            output_path=output_path,
            institution_schema_contract=target_contract,
            reference_manifests=[reference_manifest],
            reference_institution_ids=[reference_id],
            cohort_schema_class=RawEdviseStudentDataSchema,
        )
        PROMPT_COURSE = build_step2a_prompt_course_pass(
            institution_id=target_id,
            output_path=output_path,
            institution_schema_contract=target_contract,
            reference_manifests=[reference_manifest],
            reference_institution_ids=[reference_id],
            course_schema_class=RawEdviseCourseDataSchema,
        )
    else:
        PROMPT_SINGLE = build_step2a_prompt(
            institution_id=target_id,
            output_path=output_path,
            institution_schema_contract=target_contract,
            reference_manifests=[reference_manifest],
            reference_institution_ids=[reference_id],
            cohort_schema_class=RawEdviseStudentDataSchema,
            course_schema_class=RawEdviseCourseDataSchema,
        )

    # ── OpenAI client → Databricks MLflow AI Gateway (SDK bearer; PATs disabled orgs) ──
    client = create_openai_client_for_databricks_gateway()

    # ── run loop ──────────────────────────────────────────────────────────
    rows = []
    total = len(MODELS)
    i = 1

    for model in MODELS:
        logger.info(f"[{i}/{total}] Running model={model}")
        if STEP2A_TWO_PASS:
            result = run_step2a_two_pass(
                model,
                client,
                PROMPT_COHORT,
                PROMPT_COURSE,
                institution_id=target_id,
            )
        else:
            result = run_step2a_single_pass(
                model,
                client,
                PROMPT_SINGLE,
                institution_id=target_id,
            )
        if result["success"]:
            logger.info(
                f"→ done | success={result['success']} | latency={result['latency_s']}s"
            )
        else:
            logger.error(
                f"→ done | success={result['success']} | latency={result['latency_s']}s | error={result['error']}"
            )

        # write manifest JSON under eval workspace on the silver volume
        slug = folder_slug_2a(model)
        manifest_dir = EVAL_BASE / slug
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{target_id}_mapping_manifest.json"

        if result["success"]:
            try:
                # pretty-print valid JSON; fall back to raw string if parse fails
                parsed = json.loads(result["response"])
                envelope = MappingManifestEnvelope.model_validate(parsed)
                manifest_path.write_text(
                    envelope.model_dump_json(indent=2, exclude_none=True)
                )
                logger.info(f"→ manifest saved → {manifest_path}")

                structural: dict[str, list[dict[str, Any]]] = {}
                for entity_key, entity_manifest in envelope.manifests.items():
                    ek = (
                        entity_key.value
                        if hasattr(entity_key, "value")
                        else str(entity_key)
                    )
                    errs = validate_manifest(entity_manifest, schema_contract_sma)
                    structural[ek] = [e.model_dump(mode="json") for e in errs]
                ve_path = manifest_dir / f"{target_id}_validation_errors.json"
                ve_path.write_text(json.dumps(structural, indent=2))
                total_ve = sum(len(v) for v in structural.values())
                if total_ve:
                    logger.warning(
                        f"→ structural validation: {total_ve} issue(s) → {ve_path}"
                    )
                else:
                    logger.info(f"→ structural validation: 0 issues → {ve_path}")
            except json.JSONDecodeError:
                manifest_path.write_text(result["response"])
                logger.warning(
                    f"→ manifest saved (raw, invalid JSON) → {manifest_path}"
                )
            except ValidationError as ve:
                manifest_path.write_text(json.dumps(parsed, indent=2))
                logger.warning(
                    f"→ manifest saved (JSON only; envelope validation failed) → {manifest_path}: {ve}"
                )
        else:
            logger.warning("→ manifest not saved (inference error)")

        # score (Step 2a output vs gold)
        scores = score_result(result, GOLD_MANIFEST, schema_contract=target_contract)
        if scores:
            validation_status = "✓" if scores.get("validation_passed") else "✗"
            logger.info(
                f"→ scores | "
                f"validation={validation_status} | "
                f"map_decision={scores['overall']['map_decision_accuracy']} | "
                f"map_prec_strict={scores['overall']['mappable_precision_strict']} | "
                f"map_rec_strict={scores['overall']['mappable_recall_strict']} | "
                f"source_exact={scores['overall']['source_exact_accuracy_gold_mappable']} | "
                f"row_sel={scores['overall']['row_selection_accuracy']} | "
                f"row_sel_non_anyrow={scores['overall']['row_selection_accuracy_non_anyrow']} | "
                f"join_f1={scores['overall']['join_f1']} | "
                f"degree_filter={scores['overall']['degree_filter_accuracy']} | "
                f"exec_ready={scores['overall']['execution_ready_rate']}"
            )
            if not scores.get("validation_passed"):
                logger.warning(
                    f"  Validation error: {scores.get('validation_error', 'Unknown error')}"
                )
            result["scores"] = scores
        else:
            logger.warning("→ scoring skipped (parse failure or inference error)")
            result["scores"] = None

        result["scores_after_refinement"] = None
        result["refinement_error"] = None
        result["refinement_latency_s"] = None
        result["latency_s_total"] = None
        result["hitl_items_cohort"] = None
        result["hitl_items_course"] = None
        result["structural_errors_after_refinement_total"] = None

        if (
            RUN_REFINEMENT_AFTER_2A
            and result["success"]
            and isinstance(result.get("response"), str)
            and result["response"].strip()
        ):
            try:
                env_for_ref = MappingManifestEnvelope.model_validate(
                    json.loads(result["response"])
                )
                refined_env, ref_meta = run_refine_on_envelope(
                    env_for_ref,
                    institution_id=target_id,
                    schema_contract_sma=schema_contract_sma,
                    model=model,
                    client=client,
                )
                refined_text = refined_env.model_dump_json(indent=2, exclude_none=True)
                refined_manifest_path = (
                    manifest_dir / f"{target_id}_mapping_manifest_refined.json"
                )
                refined_manifest_path.write_text(refined_text)
                logger.info("→ refined manifest saved → %s", refined_manifest_path)

                if _eval_write_hitl_artifacts_to_workspace():
                    hitl_map = ref_meta.get("hitl_by_entity") or {}
                    if isinstance(hitl_map, dict) and hitl_map:
                        _write_eval_hitl_files_for_refinement(
                            manifest_dir,
                            institution_id=target_id,
                            hitl_by_entity=hitl_map,
                        )

                structural_refined: dict[str, list[dict[str, Any]]] = {}
                for entity_key, entity_manifest in refined_env.manifests.items():
                    ek = (
                        entity_key.value
                        if hasattr(entity_key, "value")
                        else str(entity_key)
                    )
                    eerrs = validate_manifest(entity_manifest, schema_contract_sma)
                    structural_refined[ek] = [e.model_dump(mode="json") for e in eerrs]
                ve_refined_path = (
                    manifest_dir
                    / f"{target_id}_validation_errors_after_refinement.json"
                )
                ve_refined_path.write_text(json.dumps(structural_refined, indent=2))
                total_ver = sum(len(v) for v in structural_refined.values())
                logger.info(
                    "→ structural validation after refinement: %d issue(s) → %s",
                    total_ver,
                    ve_refined_path,
                )

                refined_result = {
                    "model": model,
                    "success": True,
                    "response": refined_text,
                }
                scores_r = score_result(
                    refined_result,
                    GOLD_MANIFEST,
                    schema_contract=target_contract,
                )
                result["scores_after_refinement"] = scores_r
                result["refinement_latency_s"] = ref_meta["refinement_latency_s"]
                result["latency_s_total"] = round(
                    float(result["latency_s"] or 0)
                    + float(ref_meta["refinement_latency_s"]),
                    3,
                )
                hmap = ref_meta["hitl_items_by_entity"]
                result["hitl_items_cohort"] = hmap.get("cohort")
                result["hitl_items_course"] = hmap.get("course")
                result["structural_errors_after_refinement_total"] = total_ver
                if scores_r:
                    rs = "✓" if scores_r.get("validation_passed") else "✗"
                    logger.info(
                        "→ scores (after refinement) | validation=%s | "
                        "map_decision=%s | map_prec_strict=%s | map_rec_strict=%s | "
                        "source_exact=%s | exec_ready=%s | hitl_items=(cohort=%s, course=%s)",
                        rs,
                        scores_r["overall"]["map_decision_accuracy"],
                        scores_r["overall"]["mappable_precision_strict"],
                        scores_r["overall"]["mappable_recall_strict"],
                        scores_r["overall"]["source_exact_accuracy_gold_mappable"],
                        scores_r["overall"]["execution_ready_rate"],
                        result["hitl_items_cohort"],
                        result["hitl_items_course"],
                    )
            except Exception as exc:
                result["refinement_error"] = str(exc)
                logger.exception("[eval] Refinement failed for model=%s", model)

        rows.append(result)
        i += 1

    # ── execution summary ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 80)
    for r in rows:
        status = "✓ SUCCESS" if r["success"] else "✗ FAILED"
        lat_note = ""
        if r.get("latency_s_total") is not None:
            lat_note = (
                f" | total incl. refinement: {r['latency_s_total']}s "
                f"(refine {r.get('refinement_latency_s')}s)"
            )
        logger.info(f"{status}: {r['model']} (2a latency: {r['latency_s']}s){lat_note}")
        if not r["success"]:
            error_preview = (
                r["error"][:200] + "..."
                if len(r.get("error", "")) > 200
                else r.get("error", "")
            )
            logger.info(f"  Error: {error_preview}")
    logger.info("=" * 80 + "\n")

    # ── outputs ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # raw results (one row per model)
    raw_path = OUTPUT_DIR / f"raw_{RUN_ID}.csv"
    df.drop(
        columns=["scores", "field_scores", "scores_after_refinement"],
        errors="ignore",
    ).to_csv(raw_path, index=False)
    logger.info(f"Raw results saved → {raw_path}")

    # summary metrics (one row per model)
    summary_rows = []
    for r in rows:
        row_data = {
            "model": r["model"],
            "latency_s": r["latency_s"],
            "response_chars": r["response_chars"],
            "success": r["success"],
            "error": r.get("error"),
        }
        if r.get("step2a_parse_retry_enabled"):
            row_data["step2a_llm_calls"] = r.get("step2a_llm_calls")
            row_data["step2a_envelope_attempts_used"] = r.get(
                "step2a_envelope_attempts_used"
            )
            row_data["step2a_parse_retry_enabled"] = True
        if r["scores"] is not None:
            row_data.update(
                {
                    "validation_passed": r["scores"].get("validation_passed"),
                    "validation_error": r["scores"].get("validation_error"),
                    **{f"overall_{k}": v for k, v in r["scores"]["overall"].items()},
                    **{
                        f"cohort_{k}": v
                        for k, v in r["scores"]["cohort"].items()
                        if k != "field_scores"
                    },
                    **{
                        f"course_{k}": v
                        for k, v in r["scores"]["course"].items()
                        if k != "field_scores"
                    },
                }
            )
        if r.get("refinement_latency_s") is not None:
            row_data["refinement_latency_s"] = r["refinement_latency_s"]
        if r.get("latency_s_total") is not None:
            row_data["latency_s_total"] = r["latency_s_total"]
        if r.get("refinement_error") is not None:
            row_data["refinement_error"] = r["refinement_error"]
        if r.get("hitl_items_cohort") is not None:
            row_data["hitl_items_cohort"] = r["hitl_items_cohort"]
        if r.get("hitl_items_course") is not None:
            row_data["hitl_items_course"] = r["hitl_items_course"]
        if r.get("structural_errors_after_refinement_total") is not None:
            row_data["structural_errors_after_refinement_total"] = r[
                "structural_errors_after_refinement_total"
            ]
        sr = r.get("scores_after_refinement")
        if sr:
            row_data["refined_validation_passed"] = sr.get("validation_passed")
            row_data["refined_validation_error"] = sr.get("validation_error")
            row_data.update(
                {f"refined_overall_{k}": v for k, v in sr["overall"].items()}
            )
            row_data.update(
                {
                    f"refined_cohort_{k}": v
                    for k, v in sr["cohort"].items()
                    if k != "field_scores"
                }
            )
            row_data.update(
                {
                    f"refined_course_{k}": v
                    for k, v in sr["course"].items()
                    if k != "field_scores"
                }
            )
        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / f"summary_{RUN_ID}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved → {summary_path}")

    # field-level detail (one row per model x field)
    field_rows = []
    for r in rows:
        if r["scores"] is not None:
            for entity in ("cohort", "course"):
                for fs in r["scores"][entity]["field_scores"]:
                    field_rows.append({"model": r["model"], "stage": "2a", **fs})
        sr = r.get("scores_after_refinement")
        if sr:
            for entity in ("cohort", "course"):
                for fs in sr[entity]["field_scores"]:
                    field_rows.append(
                        {
                            "model": r["model"],
                            "stage": "after_refinement",
                            **fs,
                        }
                    )

    field_df = pd.DataFrame(field_rows)
    field_path = OUTPUT_DIR / f"field_detail_{RUN_ID}.csv"
    field_df.to_csv(field_path, index=False)
    logger.info(f"Field detail saved → {field_path}")

    # print summary table
    if not summary_df.empty:
        # Build column list, including validation_passed if it exists
        cols = ["model", "latency_s"]
        if "refinement_latency_s" in summary_df.columns:
            cols.append("refinement_latency_s")
        if "latency_s_total" in summary_df.columns:
            cols.append("latency_s_total")
        if "validation_passed" in summary_df.columns:
            cols.append("validation_passed")
        cols.extend(
            [
                "overall_map_decision_accuracy",
                "overall_mappable_precision_strict",
                "overall_mappable_recall_strict",
                "overall_source_exact_accuracy_gold_mappable",
                "overall_row_selection_accuracy",
                "overall_row_selection_accuracy_non_anyrow",
                "overall_join_f1",
                "overall_join_table_accuracy",
                "overall_join_key_accuracy",
                "overall_degree_filter_accuracy",
                "overall_exact_field_match_rate",
                "overall_execution_ready_rate",
            ]
        )
        cols.extend(
            [
                c
                for c in (
                    "refined_validation_passed",
                    "refined_overall_map_decision_accuracy",
                    "refined_overall_mappable_precision_strict",
                    "refined_overall_mappable_recall_strict",
                    "refined_overall_source_exact_accuracy_gold_mappable",
                    "refined_overall_execution_ready_rate",
                    "hitl_items_cohort",
                    "hitl_items_course",
                )
                if c in summary_df.columns
            ]
        )
        # Only include columns that exist in the dataframe
        cols = [c for c in cols if c in summary_df.columns]
        logger.info("\n" + summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    run()
