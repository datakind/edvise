import time
import json
import logging
import os
from collections import Counter
from copy import deepcopy
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError

from edvise.genai.schema_mapping_agent.manifest.prompt_builder import (
    build_step2a_prompt,
    build_step2a_prompt_cohort_pass,
    build_step2a_prompt_course_pass,
    load_json,
    merge_step2a_entity_manifests,
    strip_json_fences,
)

from edvise.genai.schema_mapping_agent.manifest.schemas import MappingManifestEnvelope
from edvise.data_audit.schemas.raw_edvise_student import (
    RawEdviseStudentDataSchema,
)
from edvise.data_audit.schemas.raw_edvise_course import (
    RawEdviseCourseDataSchema,
)

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
    "id": "lee_col",
    "name": "Lee College",
}

REFERENCE_INSTITUTION = {
    "id": "uni_of_central_florida",
    "name": "University of Central Florida",
}

# To swap roles (e.g., evaluate UCF using LC as reference), swap the above:
# TARGET_INSTITUTION = {
#     "id": "uni_of_central_florida",
#     "name": "University of Central Florida",
# }
# REFERENCE_INSTITUTION = {
#     "id": "lee_col",
#     "name": "Lee College",
# }

MODELS = [
    "claude-opus-test-genai-ai-data-cleaning",
    "claude-sonnet-test-genai-ai-data-cleaning",
    "claude-haiku-test-genai-data-cleaning",
    "gpt52-test-genai-ai-datacleaning",
    "gemini-test-genai-ai-datacleaning",
]

# Short label per gateway model → output folders like {base}_2a_{SHOT_TAG}, {base}_2b_{SHOT_TAG}
MODEL_BASE_SLUG = {
    "claude-opus-test-genai-ai-data-cleaning": "opus",
    "claude-sonnet-test-genai-ai-data-cleaning": "sonnet",
    "claude-haiku-test-genai-data-cleaning": "haiku",
    "gpt52-test-genai-ai-datacleaning": "gpt52",
    "gemini-test-genai-ai-datacleaning": "gemini",
}

# Set to "2shot" (or another tag) before run() when evaluating a multi-turn / 2-shot prompt variant
SHOT_TAG = "1shot"

# Two LLM calls (cohort then course), merged to the same full-manifest JSON shape as single-pass.
# Set False to use build_step2a_prompt + one run_once per model (legacy).
STEP2A_TWO_PASS = True


def folder_slug_2a(model: str) -> str:
    """Artifact directory under historical_examples/{institution}/ for Step 2a manifests."""
    base = MODEL_BASE_SLUG.get(model, model.replace("-", "_"))
    return f"{base}_2a_{SHOT_TAG}"


def folder_slug_2b(model: str) -> str:
    """Artifact directory under historical_examples/{institution}/ for Step 2b transformation maps."""
    base = MODEL_BASE_SLUG.get(model, model.replace("-", "_"))
    return f"{base}_2b_{SHOT_TAG}"


# Models that support assistant message prefill (for JSON output formatting)
MODELS_WITH_PREFILL = {
    "claude-sonnet-test-genai-ai-data-cleaning",
    "claude-haiku-test-genai-data-cleaning",
    "gpt52-test-genai-ai-datacleaning",
    "gemini-test-genai-ai-datacleaning",
}


# ── inference ─────────────────────────────────────────────────────────────────
def run_once(model: str, prompt: str, client: OpenAI) -> dict:
    start = time.perf_counter()
    try:
        full_text = ""

        # Build messages - only add assistant prefill for models that support it
        messages = [{"role": "user", "content": prompt}]
        if model in MODELS_WITH_PREFILL:
            messages.append({"role": "assistant", "content": "{"})  # prefill for JSON

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=16000,
            stream=True,
        )
        for chunk in resp:
            # Safely extract content from chunk - handle cases where choices might be empty
            try:
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


def run_step2a_two_pass(
    model: str,
    client: OpenAI,
    prompt_cohort: str,
    prompt_course: str,
    *,
    institution_id: str | None = None,
) -> dict:
    """
    Step 2a only: two gateway calls (cohort JSON, then course JSON), then merge into one manifest.

    ``prompt_cohort`` and ``prompt_course`` are independent (build with
    ``build_step2a_prompt_cohort_pass`` / ``build_step2a_prompt_course_pass``); the course
    pass does not include cohort output. Merge uses ``merge_step2a_entity_manifests``.
    Pass ``institution_id`` when model output is partial (no envelope fields); if omitted,
    merge falls back to ``institution_id`` on cohort/course JSON when present (legacy passes).

    Return shape matches ``run_once`` (same keys) for scoring and CSV export; ``latency_s``
    is the sum of both calls.
    """
    sep = "\n\n=== STEP 2a PASS 2 (course) ===\n\n"
    r1 = run_once(model, prompt_cohort, client)
    prompt_combined = f"=== STEP 2a PASS 1 (cohort) ===\n\n{prompt_cohort}"
    if not r1["success"]:
        return {**r1, "prompt": prompt_combined}

    try:
        cohort_parsed = json.loads(r1["response"])
    except json.JSONDecodeError as e:
        return {
            "model": model,
            "prompt": prompt_combined,
            "success": False,
            "latency_s": r1["latency_s"],
            "response_chars": None,
            "response": None,
            "error": f"cohort pass invalid JSON: {e}",
        }

    prompt_combined = f"{prompt_combined}{sep}{prompt_course}"

    r2 = run_once(model, prompt_course, client)
    lat = round((r1["latency_s"] or 0) + (r2["latency_s"] or 0), 3)

    if not r2["success"]:
        return {
            "model": model,
            "prompt": prompt_combined,
            "success": False,
            "latency_s": lat,
            "response_chars": None,
            "response": None,
            "error": r2.get("error") or "course pass failed",
        }

    try:
        course_parsed = json.loads(r2["response"])
        merged = merge_step2a_entity_manifests(
            cohort_parsed, course_parsed, institution_id=institution_id
        )
        merged_str = json.dumps(merged)
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "model": model,
            "prompt": prompt_combined,
            "success": False,
            "latency_s": lat,
            "response_chars": None,
            "response": None,
            "error": f"course pass or merge failed: {e}",
        }

    return {
        "model": model,
        "prompt": prompt_combined,
        "success": True,
        "latency_s": lat,
        "response_chars": len(merged_str),
        "response": merged_str,
        "error": None,
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
    return alias_map.get((table, column), column)


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

    counts = Counter()
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

    is_valid, validation_error = validate_manifest(pred)
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


def validate_manifest(manifest_dict: dict) -> tuple[bool, str | None]:
    """
    Attempt to parse a raw manifest dict against MappingManifestEnvelope.
    Returns (is_valid, error_message).
    """
    try:
        MappingManifestEnvelope.model_validate(manifest_dict)
        return True, None
    except ValidationError as e:
        return False, str(e)


def _find_eval_project_root() -> Path:
    """
    Resolve the repo root that contains pipelines/gen_ai_cleaning/.

    When edvise is installed as a package, __file__ lives under site-packages, so
    walking upward from there never reaches the data repo. We therefore search from
    cwd (and optional EDVISE_EVAL_PROJECT_ROOT).
    """
    env = os.environ.get("EDVISE_EVAL_PROJECT_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        marker = p / "pipelines" / "gen_ai_cleaning"
        if not marker.is_dir():
            raise FileNotFoundError(
                f"EDVISE_EVAL_PROJECT_ROOT={env!r} does not contain pipelines/gen_ai_cleaning"
            )
        return p

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pipelines" / "gen_ai_cleaning").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find project root: no directory from cwd upward contains "
        "pipelines/gen_ai_cleaning. cd to your repo root (e.g. student-success-intervention) "
        "before calling eval.run(), or set EDVISE_EVAL_PROJECT_ROOT to that root."
    )


def _resolve_mapping_manifest(institution_id: str) -> Path:
    """Prefer final_hitl gold manifest, then v1."""
    base = Path(f"pipelines/gen_ai_cleaning/historical_examples/{institution_id}")
    candidates = [
        base / "final_hitl" / f"{institution_id}_mapping_manifest.json",
        base / "v1" / f"{institution_id}_mapping_manifest.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Mapping manifest not found for institution "
        f"{institution_id!r}. Tried:\n  " + "\n  ".join(str(p) for p in candidates)
    )


# ── main execution ───────────────────────────────────────────────────────────
def run():
    """Run evaluation on all models."""
    project_root = _find_eval_project_root()

    # Change to project root for consistent path resolution
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        # ── paths ─────────────────────────────────────────────────────────────
        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_id = TARGET_INSTITUTION["id"]
        target_name = TARGET_INSTITUTION["name"]
        reference_id = REFERENCE_INSTITUTION["id"]
        reference_name = REFERENCE_INSTITUTION["name"]

        logger.info("=" * 80)
        logger.info("EVALUATION CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Target Institution: {target_name} ({target_id})")
        logger.info(f"Reference Institution: {reference_name} ({reference_id})")
        logger.info(f"Project root (eval cwd): {project_root}")
        logger.info("=" * 80)

        HISTORICAL_BASE = Path(
            f"pipelines/gen_ai_cleaning/historical_examples/{target_id}"
        )
        OUTPUT_DIR = HISTORICAL_BASE / "eval_outputs"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # ── gold manifest ─────────────────────────────────────────────────────
        GOLD_MANIFEST_PATH = _resolve_mapping_manifest(target_id)
        GOLD_MANIFEST = load_json(str(GOLD_MANIFEST_PATH))
        logger.info(f"Loaded gold manifest from {GOLD_MANIFEST_PATH}")

        # ── prompt (via shared builder) ───────────────────────────────────────
        target_contract_path = f"pipelines/gen_ai_cleaning/historical_examples/{target_id}/{target_id}_schema_contract.json"
        target_contract = load_json(target_contract_path)
        logger.info(f"Loaded target schema contract from {target_contract_path}")

        REFERENCE_MANIFEST_PATH = _resolve_mapping_manifest(reference_id)
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
        logger.info(f"  Reference institution: {reference_name} (id: {reference_id})")
        logger.info(f"  Manifest institution_id: {manifest_institution_id} ✓")

        output_path = (
            f"pipelines/gen_ai_cleaning/historical_examples/{target_id}/v0/"
            f"{target_id}_mapping_manifest.json"
        )
        if STEP2A_TWO_PASS:
            PROMPT_COHORT = build_step2a_prompt_cohort_pass(
                institution_id=target_id,
                institution_name=target_name,
                output_path=output_path,
                institution_schema_contract=target_contract,
                reference_manifests=[reference_manifest],
                reference_institution_names=[reference_name],
                cohort_schema_class=RawEdviseStudentDataSchema,
            )
            PROMPT_COURSE = build_step2a_prompt_course_pass(
                institution_id=target_id,
                institution_name=target_name,
                output_path=output_path,
                institution_schema_contract=target_contract,
                reference_manifests=[reference_manifest],
                reference_institution_names=[reference_name],
                course_schema_class=RawEdviseCourseDataSchema,
            )
        else:
            PROMPT_SINGLE = build_step2a_prompt(
                institution_id=target_id,
                institution_name=target_name,
                output_path=output_path,
                institution_schema_contract=target_contract,
                reference_manifests=[reference_manifest],
                reference_institution_names=[reference_name],
                cohort_schema_class=RawEdviseStudentDataSchema,
                course_schema_class=RawEdviseCourseDataSchema,
            )

        # ── OpenAI client ─────────────────────────────────────────────────────
        TOKEN = os.environ.get("DATABRICKS_TOKEN")
        if not TOKEN:
            raise ValueError("DATABRICKS_TOKEN environment variable not set")
        BASE_URL = "https://4437281602191762.ai-gateway.gcp.databricks.com/mlflow/v1"
        client = OpenAI(
            api_key=TOKEN,
            base_url=BASE_URL,
        )

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
                result = run_once(model, PROMPT_SINGLE, client)
            if result["success"]:
                logger.info(
                    f"→ done | success={result['success']} | latency={result['latency_s']}s"
                )
            else:
                logger.error(
                    f"→ done | success={result['success']} | latency={result['latency_s']}s | error={result['error']}"
                )

            # write manifest JSON to historical examples path
            slug = folder_slug_2a(model)
            manifest_dir = HISTORICAL_BASE / slug
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
                logger.warning(f"→ manifest not saved (inference error)")

            # score
            scores = score_result(
                result, GOLD_MANIFEST, schema_contract=target_contract
            )
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
                logger.warning(f"→ scoring skipped (parse failure or inference error)")
                result["scores"] = None

            rows.append(result)
            i += 1

        # ── execution summary ─────────────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 80)
        for r in rows:
            status = "✓ SUCCESS" if r["success"] else "✗ FAILED"
            logger.info(f"{status}: {r['model']} (latency: {r['latency_s']}s)")
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
        df.drop(columns=["scores", "field_scores"], errors="ignore").to_csv(
            raw_path, index=False
        )
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
            if r["scores"] is not None:
                row_data.update(
                    {
                        "validation_passed": r["scores"].get("validation_passed"),
                        "validation_error": r["scores"].get("validation_error"),
                        **{
                            f"overall_{k}": v for k, v in r["scores"]["overall"].items()
                        },
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
            summary_rows.append(row_data)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUTPUT_DIR / f"summary_{RUN_ID}.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved → {summary_path}")

        # field-level detail (one row per model x field)
        field_rows = []
        for r in rows:
            if r["scores"] is None:
                continue
            for entity in ("cohort", "course"):
                for fs in r["scores"][entity]["field_scores"]:
                    field_rows.append({"model": r["model"], **fs})

        field_df = pd.DataFrame(field_rows)
        field_path = OUTPUT_DIR / f"field_detail_{RUN_ID}.csv"
        field_df.to_csv(field_path, index=False)
        logger.info(f"Field detail saved → {field_path}")

        # print summary table
        if not summary_df.empty:
            # Build column list, including validation_passed if it exists
            cols = ["model", "latency_s"]
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
            # Only include columns that exist in the dataframe
            cols = [c for c in cols if c in summary_df.columns]
            logger.info("\n" + summary_df[cols].to_string(index=False))
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    run()
