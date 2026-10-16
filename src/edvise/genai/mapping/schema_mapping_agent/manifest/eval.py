"""Shared Step 2a manifest scoring, gateway calls, and eval volume paths.

The Step 2a model comparison itself runs through
``edvise.genai.mapping.eval``. This module keeps the field comparison,
``run_once``, and Unity Catalog path resolution that production and the
Step 2b evaluator still import.
"""

import json
import logging
import os
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import ValidationError

from edvise.configs import genai as genai_cfg
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    MappingManifestEnvelope,
)
from edvise.genai.mapping.shared.reference_pin import resolve_sma_few_shot_pin
from edvise.genai.mapping.shared.utilities import strip_json_fences

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_EVAL_MODELS: tuple[str, ...] = (
    "claude-opus-test-genai-ai-data-cleaning",
    "claude-sonnet-test-genai-ai-data-cleaning",
    "claude-haiku-test-genai-data-cleaning",
)

# Short label per gateway model → output folders like {base}_2a_{SHOT_TAG}, {base}_2b_{SHOT_TAG}
MODEL_BASE_SLUG = {
    "claude-opus-test-genai-ai-data-cleaning": "opus",
    "claude-sonnet-test-genai-ai-data-cleaning": "sonnet",
    "claude-haiku-test-genai-data-cleaning": "haiku",
}

# Mutable list so notebooks can do ``MODELS[:] = [...]`` (same object as ``transformation.eval`` imports).
MODELS: list[str] = list(_DEFAULT_EVAL_MODELS)

# Folder tag for Step 2b artifact directories, for example ``sonnet_2b_1shot``.
SHOT_TAG = "1shot"


def folder_slug_2b(model: str) -> str:
    """Per-model artifact folder name under ``…/genai_mapping/eval/`` for Step 2b."""
    base = MODEL_BASE_SLUG.get(model, model.replace("-", "_"))
    return f"{base}_2b_{SHOT_TAG}"


def apply_eval_models_from_env() -> None:
    """If ``EDVISE_EVAL_MODELS`` is set, replace ``MODELS`` in place.

    Comma-separated entries; each entry is a short slug (``opus``, ``sonnet``, ``haiku``)
    or a full gateway model id from ``_DEFAULT_EVAL_MODELS``. Order is preserved;
    duplicates are dropped.

    Called automatically at the start of ``run()`` in manifest and transformation eval.
    """
    raw = os.environ.get("EDVISE_EVAL_MODELS", "").strip()
    if not raw:
        return
    slug_to_model = {MODEL_BASE_SLUG[m]: m for m in _DEFAULT_EVAL_MODELS}
    id_lower_to_model = {m.lower(): m for m in _DEFAULT_EVAL_MODELS}
    allowed = set(_DEFAULT_EVAL_MODELS)
    resolved: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        key = token.lower()
        if key in slug_to_model:
            mid = slug_to_model[key]
        elif token in allowed:
            mid = token
        elif key in id_lower_to_model:
            mid = id_lower_to_model[key]
        else:
            slugs = ", ".join(sorted(slug_to_model))
            raise ValueError(
                f"EDVISE_EVAL_MODELS: unknown entry {token!r}. "
                f"Use comma-separated slugs ({slugs}) or a full gateway model id from the eval defaults."
            )
        if mid not in seen:
            seen.add(mid)
            resolved.append(mid)
    if not resolved:
        raise ValueError("EDVISE_EVAL_MODELS produced an empty model list.")
    MODELS[:] = resolved
    logger.info("EDVISE_EVAL_MODELS active → %s", MODELS)


# Models that support assistant message prefill (for JSON output formatting)
MODELS_WITH_PREFILL = {
    "claude-sonnet-test-genai-ai-data-cleaning",
    "claude-haiku-test-genai-data-cleaning",
}

# Sonnet 5 streams a reasoning block ahead of the manifest. It is not JSON.
_SKIPPED_STREAM_BLOCK_TYPES = frozenset({"reasoning", "thinking", "redacted_thinking"})


def _split_stream_content(
    content: Any,
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Split one streamed ``delta.content`` into manifest text and reasoning.

    Sonnet 4.5 sends a string. Sonnet 5 and 5.5 send a list of blocks and
    lead with reasoning. Reasoning, thinking, and redacted thinking are kept
    as their own records so they are not parsed as the manifest. A refusal
    fails the call.
    """
    if content is None or content == "":
        return "", [], None
    if isinstance(content, str):
        return content, [], None
    blocks = content if isinstance(content, list) else [content]
    text_parts: list[str] = []
    reasoning: list[dict[str, Any]] = []
    for block in blocks:
        text, record, refusal = _split_stream_block(block)
        if refusal:
            return "".join(text_parts), reasoning, refusal
        if record is not None:
            reasoning.append(record)
        elif text:
            text_parts.append(text)
    return "".join(text_parts), reasoning, None


def _split_stream_block(
    block: Any,
) -> tuple[str, dict[str, Any] | None, str | None]:
    if isinstance(block, str):
        return block, None, None
    block_type, text, refusal = _stream_block_fields(block)
    if block_type == "refusal" or refusal:
        message = refusal or (text if isinstance(text, str) and text else "")
        return "", None, message or "The model refused the request."
    if block_type in _SKIPPED_STREAM_BLOCK_TYPES:
        return "", _json_block(block, block_type), None
    if isinstance(text, str):
        return text, None, None
    if block_type not in {None, "text", "output_text"}:
        logger.debug("Skipping streamed content block type %s", block_type)
    return "", None, None


def _json_block(block: Any, block_type: str | None) -> dict[str, Any]:
    """JSON-safe copy of a reasoning, thinking, or redacted-thinking block."""
    if isinstance(block, dict):
        record = {str(key): _json_value(value) for key, value in block.items()}
    elif hasattr(block, "model_dump"):
        dumped = block.model_dump()
        record = (
            {str(key): _json_value(value) for key, value in dumped.items()}
            if isinstance(dumped, dict)
            else {"type": block_type, "value": _json_value(dumped)}
        )
    else:
        record = {
            "type": block_type,
            "text": _json_value(getattr(block, "text", None)),
            "thinking": _json_value(getattr(block, "thinking", None)),
            "summary": _json_value(getattr(block, "summary", None)),
            "data": _json_value(getattr(block, "data", None)),
            "signature": _json_value(getattr(block, "signature", None)),
        }
        record = {key: value for key, value in record.items() if value is not None}
    if block_type and "type" not in record:
        record["type"] = block_type
    return record


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump())
    return str(value)


def _stream_block_fields(block: Any) -> tuple[str | None, Any, str | None]:
    if isinstance(block, dict):
        block_type = block.get("type")
        text = block.get("text")
        refusal = block.get("refusal")
    else:
        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        refusal = getattr(block, "refusal", None)
    type_name = str(block_type) if block_type is not None else None
    refusal_text = refusal if isinstance(refusal, str) and refusal.strip() else None
    return type_name, text, refusal_text


# ── inference ─────────────────────────────────────────────────────────────────
def run_once(
    model: str,
    prompt: str | list[dict[str, Any]],
    client: OpenAI,
    *,
    log_cache_usage: bool = False,
) -> dict[str, Any]:
    """
    ``prompt`` is normally the full ``role="user"`` message text, but callers that want
    Anthropic/Databricks prompt caching (see
    :func:`~edvise.genai.mapping.shared.databricks_ai_gateway.build_gateway_message_content`)
    may instead pass a pre-built list of content blocks (with ``cache_control`` on the
    static block); it is forwarded to the gateway unchanged either way.

    ``log_cache_usage`` requests a final usage-only chunk from the stream (via
    ``stream_options={"include_usage": True}``) and logs any cache token fields on it (see
    :func:`~edvise.genai.mapping.shared.databricks_ai_gateway.log_gateway_cache_usage_if_present`).
    Only set this for calls that opted into caching via ``prompt``'s ``cache_control``.
    """
    from edvise.genai.mapping.shared.databricks_ai_gateway import (
        log_gateway_cache_usage_if_present,
    )

    start = time.perf_counter()
    reasoning: list[dict[str, Any]] = []
    try:
        full_text = ""
        final_usage: Any = None

        # Build messages - only add assistant prefill for models that support it
        messages_list: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
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
        if log_cache_usage:
            create_kwargs["stream_options"] = {"include_usage": True}
        resp = client.chat.completions.create(**create_kwargs)
        for chunk in resp:
            # Safely extract content from chunk - handle cases where choices might be empty
            try:
                if not isinstance(chunk, ChatCompletionChunk):
                    continue
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    final_usage = chunk_usage
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                text, blocks, refusal = _split_stream_content(
                    chunk.choices[0].delta.content
                )
                reasoning.extend(blocks)
                if refusal:
                    raise ValueError(refusal)
                if text:
                    full_text += text
            except (IndexError, AttributeError) as e:
                # Skip chunks with malformed structure - this can happen with some models
                logger.debug(f"Skipping malformed chunk: {e}")
                continue

        if log_cache_usage:
            log_gateway_cache_usage_if_present(final_usage, log=logger, model=model)

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
            "reasoning": reasoning,
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
                except Exception:
                    pass
            elif hasattr(e.response, "json"):
                try:
                    response_body = str(e.response.json())
                except Exception:
                    pass

        if not response_body and hasattr(e, "body"):
            try:
                response_body = str(e.body)
            except Exception:
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
            "reasoning": reasoning,
            "error": error_details,
        }


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
    * Else pinned ``…/genai_mapping/references/<id>/current/manifest_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_REFERENCE_MANIFEST_PATH")
    if override and str(override).strip():
        p = Path(str(override).strip()).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_REFERENCE_MANIFEST_PATH not found: {p}"
            )
        return p.resolve()
    few = resolve_sma_few_shot_pin(reference_id, catalog=_require_uc_catalog_for_eval())
    return few.manifest_map


def _resolve_reference_transformation_map_for_eval(reference_id: str) -> Path:
    """
    Few-shot reference transformation map for Step 2b.

    * ``EDVISE_EVAL_REFERENCE_TRANSFORMATION_MAP_PATH``
    * Else pinned ``…/genai_mapping/references/<id>/current/transformation_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_REFERENCE_TRANSFORMATION_MAP_PATH")
    if override and str(override).strip():
        p = Path(str(override).strip()).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_REFERENCE_TRANSFORMATION_MAP_PATH not found: {p}"
            )
        return p.resolve()
    few = resolve_sma_few_shot_pin(reference_id, catalog=_require_uc_catalog_for_eval())
    return few.transformation_map


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
