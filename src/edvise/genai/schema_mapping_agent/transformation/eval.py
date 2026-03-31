"""
Evaluate Step 2b (transformation map) prompts against HITL gold transformation maps.

Mirrors prompt_2a_eval for models, gateway client, and CSV outputs. Scoring compares
per-field plans with DRY metrics: function-chain precision/recall/F1, critical
step-argument accuracy (column, mapping, default, value, extra_columns),
map_values / fill_constant / dependency checks, and execution_safe_rate vs
execution_exact_rate.

Output folders are `{base}_2b_{SHOT_TAG}` (e.g. opus_2b_1shot). SHOT_TAG lives in
manifest.eval — set ``manifest.eval.SHOT_TAG = "2shot"`` before run() for 2-shot runs.
"""

import json
import logging
import os
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from .schemas import TransformationMap
from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema
from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema

from ..manifest.eval import MODELS, _find_eval_project_root, folder_slug_2b, run_once
from .prompt_builder import build_step2b_prompt, load_json

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Declared ``output_dtype`` strings are not used by the field executor. When True, scoring
# ignores dtype for exact plan match, execution_safe, and aggregate / per-field dtype metrics.
# Set False to restore legacy dtype comparisons in eval CSVs and logs.
IGNORE_OUTPUT_DTYPE_IN_T2B_SCORING = True

# Institution configuration — keep in sync with manifest.eval defaults unless testing 2b only
TARGET_INSTITUTION = {
    "id": "lee_col",
    "name": "Lee College",
}

REFERENCE_INSTITUTION = {
    "id": "uni_of_central_florida",
    "name": "University of Central Florida",
}


def _safe_div(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 3)


def _get_plans_by_field(tm: dict, entity: str) -> dict[str, dict]:
    """target_field -> plan dict."""
    section = tm.get("transformation_maps", {}).get(entity) or {}
    out = {}
    for p in section.get("plans") or []:
        tf = p.get("target_field")
        if tf:
            out[tf] = p
    return out


def _sort_mapping(m: dict | None) -> dict | None:
    if not m:
        return m
    return dict(sorted(m.items(), key=lambda kv: (str(kv[0]), str(kv[1]))))


def _normalize_obj(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def _normalize_step(step: dict | None) -> dict:
    """Drop rationale/review-only fields; stable ordering for nested dicts."""
    s = {k: v for k, v in (step or {}).items() if k not in {"rationale", "review_status", "reviewer_notes", "validation_notes"}}
    if "mapping" in s and isinstance(s["mapping"], dict):
        s["mapping"] = _sort_mapping(s["mapping"])
    if "extra_columns" in s and isinstance(s["extra_columns"], dict):
        s["extra_columns"] = _sort_mapping(s["extra_columns"])
    return _normalize_obj(s)


def _normalize_plan_for_compare(plan: dict | None) -> dict:
    plan = deepcopy(plan or {})
    return {
        "target_field": plan.get("target_field"),
        "output_dtype": plan.get("output_dtype"),
        "steps": [_normalize_step(s) for s in (plan.get("steps") or [])],
    }


def _step_signature(step: dict) -> tuple:
    s = _normalize_step(step)
    return (
        s.get("function_name"),
        s.get("column"),
        json.dumps(s.get("extra_columns"), sort_keys=True, default=str),
        json.dumps(s.get("mapping"), sort_keys=True, default=str),
        s.get("default"),
        s.get("value"),
    )


def _step_arg_payload(step: dict) -> dict:
    s = _normalize_step(step)
    return {
        k: v
        for k, v in s.items()
        if k not in {"function_name"}
    }


def _critical_args_for_step(step: dict) -> dict:
    s = _normalize_step(step)
    fn = s.get("function_name")
    out = {}
    if "column" in s:
        out["column"] = s.get("column")
    if fn == "map_values":
        out["mapping"] = s.get("mapping")
        if "default" in s:
            out["default"] = s.get("default")
    if fn == "fill_constant":
        out["value"] = s.get("value")
    if "extra_columns" in s:
        out["extra_columns"] = s.get("extra_columns")
    return out


def _step_chain_names(plan: dict) -> list[str]:
    return [
        _normalize_step(s).get("function_name")
        for s in (plan.get("steps") or [])
    ]


def _multiset_overlap(a: list[str], b: list[str]) -> int:
    ca = Counter(a)
    cb = Counter(b)
    return sum(min(ca[k], cb[k]) for k in set(ca) | set(cb))


def _steps_aligned_by_index_and_name(g_steps: list[dict], p_steps: list[dict]) -> list[tuple[dict, dict]]:
    aligned = []
    for g, p in zip(g_steps, p_steps):
        if _normalize_step(g).get("function_name") == _normalize_step(p).get("function_name"):
            aligned.append((g, p))
    return aligned


def _mapping_overlap_metrics(gold_mapping: dict | None, pred_mapping: dict | None) -> tuple[int | None, int | None]:
    if gold_mapping is None:
        return None, None
    gold_mapping = gold_mapping or {}
    pred_mapping = pred_mapping or {}
    if not gold_mapping:
        return 1, 1 if not pred_mapping else 0
    overlap_keys = set(gold_mapping).intersection(set(pred_mapping))
    key_recall = int(len(overlap_keys) == len(gold_mapping))
    value_correct = int(all(pred_mapping.get(k) == gold_mapping.get(k) for k in overlap_keys) and len(overlap_keys) == len(gold_mapping))
    return key_recall, value_correct


def _prepare_plan(plan: dict | None) -> dict:
    plan = deepcopy(plan or {})
    steps = [_normalize_step(s) for s in (plan.get("steps") or [])]
    return {
        "raw": plan,
        "target_field": plan.get("target_field"),
        "output_dtype": plan.get("output_dtype"),
        "steps": steps,
        "empty": len(steps) == 0,
        "normalized": {
            "target_field": plan.get("target_field"),
            "output_dtype": plan.get("output_dtype"),
            "steps": steps,
        },
        "function_names": [s.get("function_name") for s in steps],
    }


def _compare_transform_plan(field: str, gold_plan: dict, pred_plan: dict) -> dict:
    g = _prepare_plan(gold_plan)
    p = _prepare_plan(pred_plan)

    plan_decision_correct = int(g["empty"] == p["empty"])
    output_dtype_correct = int(g["output_dtype"] == p["output_dtype"])
    if IGNORE_OUTPUT_DTYPE_IN_T2B_SCORING:
        g_ex = {k: v for k, v in g["normalized"].items() if k != "output_dtype"}
        p_ex = {k: v for k, v in p["normalized"].items() if k != "output_dtype"}
        exact_plan_match = int(g_ex == p_ex)
    else:
        exact_plan_match = int(g["normalized"] == p["normalized"])

    overlap = _multiset_overlap(g["function_names"], p["function_names"])
    function_chain_precision = _safe_div(overlap, len(p["function_names"])) if p["function_names"] else None
    function_chain_recall = _safe_div(overlap, len(g["function_names"])) if g["function_names"] else None
    function_chain_f1 = _f1(function_chain_precision, function_chain_recall)
    function_chain_exact = int(g["function_names"] == p["function_names"]) if not g["empty"] else None

    aligned_pairs = _steps_aligned_by_index_and_name(g["steps"], p["steps"])
    aligned_count = len(aligned_pairs)

    step_arg_exact_matches = 0
    step_arg_critical_matches = 0
    step_arg_critical_total = 0

    map_values_steps_gold = 0
    map_values_steps_correct = 0
    map_values_key_recall_total = 0
    map_values_value_correct_total = 0

    fill_constant_steps_gold = 0
    fill_constant_correct = 0

    dependency_steps_gold = 0
    dependency_correct = 0

    for gs, ps in aligned_pairs:
        if _step_arg_payload(gs) == _step_arg_payload(ps):
            step_arg_exact_matches += 1

        g_crit = _critical_args_for_step(gs)
        p_crit = _critical_args_for_step(ps)
        for k, gv in g_crit.items():
            step_arg_critical_total += 1
            if p_crit.get(k) == gv:
                step_arg_critical_matches += 1

        fn = gs.get("function_name")
        if fn == "map_values":
            map_values_steps_gold += 1
            g_map = gs.get("mapping") or {}
            p_map = ps.get("mapping") or {}
            key_recall_exact, value_correct_exact = _mapping_overlap_metrics(g_map, p_map)
            map_values_key_recall_total += int(key_recall_exact == 1)
            map_values_value_correct_total += int(value_correct_exact == 1 and ps.get("default") == gs.get("default"))
            if p_map == g_map and ps.get("default") == gs.get("default"):
                map_values_steps_correct += 1

        if fn == "fill_constant":
            fill_constant_steps_gold += 1
            if ps.get("value") == gs.get("value") and ps.get("column") == gs.get("column"):
                fill_constant_correct += 1

        if "extra_columns" in gs:
            dependency_steps_gold += 1
            if ps.get("extra_columns") == gs.get("extra_columns"):
                dependency_correct += 1

    required_step_recall = function_chain_recall
    extra_step_rate = _safe_div(max(len(p["function_names"]) - overlap, 0), len(g["function_names"]) or 1)

    execution_safe = False
    if g["empty"] and p["empty"]:
        if IGNORE_OUTPUT_DTYPE_IN_T2B_SCORING:
            execution_safe = True
        else:
            execution_safe = output_dtype_correct == 1
    elif (not g["empty"]) and (not p["empty"]):
        chain_ok = function_chain_recall == 1
        critical_args_ok = (step_arg_critical_matches == step_arg_critical_total) if step_arg_critical_total > 0 else True
        if IGNORE_OUTPUT_DTYPE_IN_T2B_SCORING:
            execution_safe = chain_ok and critical_args_ok
        else:
            dtype_ok = output_dtype_correct == 1
            execution_safe = dtype_ok and chain_ok and critical_args_ok

    execution_exact = bool(exact_plan_match)

    row = {
        "field": field,
        "gold_empty": g["empty"],
        "pred_empty": p["empty"],
        "plan_decision_correct": plan_decision_correct,
        "exact_plan_match": exact_plan_match,
        "function_chain_exact": function_chain_exact,
        "function_chain_precision": function_chain_precision,
        "function_chain_recall": function_chain_recall,
        "function_chain_f1": function_chain_f1,
        "required_step_recall": required_step_recall,
        "extra_step_rate": extra_step_rate,
        "aligned_step_count": aligned_count,
        "step_arg_exact_count": step_arg_exact_matches,
        "step_arg_critical_match_count": step_arg_critical_matches,
        "step_arg_critical_total": step_arg_critical_total,
        "map_values_steps_gold": map_values_steps_gold,
        "map_values_steps_correct": map_values_steps_correct,
        "map_values_key_recall_count": map_values_key_recall_total,
        "map_values_value_correct_count": map_values_value_correct_total,
        "fill_constant_steps_gold": fill_constant_steps_gold,
        "fill_constant_correct": fill_constant_correct,
        "dependency_steps_gold": dependency_steps_gold,
        "dependency_correct": dependency_correct,
        "execution_safe": int(execution_safe),
        "execution_exact": int(execution_exact),
    }
    if not IGNORE_OUTPUT_DTYPE_IN_T2B_SCORING:
        row["output_dtype_correct"] = output_dtype_correct
    return row


def _update_counts_t2b(counts: Counter, cmp: dict) -> None:
    counts["n_fields"] += 1
    counts["plan_decision_correct"] += int(cmp["plan_decision_correct"])
    if not IGNORE_OUTPUT_DTYPE_IN_T2B_SCORING:
        counts["output_dtype_correct"] += int(cmp["output_dtype_correct"])
    counts["exact_plan_match"] += int(cmp["exact_plan_match"])
    counts["execution_safe"] += int(cmp["execution_safe"])
    counts["execution_exact"] += int(cmp["execution_exact"])

    if cmp["function_chain_exact"] is not None:
        counts["function_chain_gold_nonempty"] += 1
        counts["function_chain_exact_correct"] += int(cmp["function_chain_exact"])

    if cmp["function_chain_precision"] is not None:
        counts["function_chain_precision_sum"] += cmp["function_chain_precision"]
        counts["function_chain_precision_n"] += 1
    if cmp["function_chain_recall"] is not None:
        counts["function_chain_recall_sum"] += cmp["function_chain_recall"]
        counts["function_chain_recall_n"] += 1
    if cmp["function_chain_f1"] is not None:
        counts["function_chain_f1_sum"] += cmp["function_chain_f1"]
        counts["function_chain_f1_n"] += 1

    counts["aligned_step_count"] += cmp["aligned_step_count"]
    counts["step_arg_exact_count"] += cmp["step_arg_exact_count"]
    counts["step_arg_critical_match_count"] += cmp["step_arg_critical_match_count"]
    counts["step_arg_critical_total"] += cmp["step_arg_critical_total"]

    counts["map_values_steps_gold"] += cmp["map_values_steps_gold"]
    counts["map_values_steps_correct"] += cmp["map_values_steps_correct"]
    counts["map_values_key_recall_count"] += cmp["map_values_key_recall_count"]
    counts["map_values_value_correct_count"] += cmp["map_values_value_correct_count"]

    counts["fill_constant_steps_gold"] += cmp["fill_constant_steps_gold"]
    counts["fill_constant_correct"] += cmp["fill_constant_correct"]

    counts["dependency_steps_gold"] += cmp["dependency_steps_gold"]
    counts["dependency_correct"] += cmp["dependency_correct"]

    if cmp["required_step_recall"] is not None:
        counts["required_step_recall_sum"] += cmp["required_step_recall"]
        counts["required_step_recall_n"] += 1
    if cmp["extra_step_rate"] is not None:
        counts["extra_step_rate_sum"] += cmp["extra_step_rate"]
        counts["extra_step_rate_n"] += 1


def _mean_from_counts(counts: Counter, sum_key: str, n_key: str) -> float | None:
    n = counts[n_key]
    return round(counts[sum_key] / n, 3) if n else None


def _finalize_counts_t2b(entity: str, counts: Counter, field_scores: list[dict]) -> dict:
    out = {
        "entity": entity,
        "n_fields": counts["n_fields"],
        "plan_decision_accuracy": _safe_div(counts["plan_decision_correct"], counts["n_fields"]),
        "exact_plan_match_rate": _safe_div(counts["exact_plan_match"], counts["n_fields"]),
        "function_chain_accuracy": _safe_div(
            counts["function_chain_exact_correct"],
            counts["function_chain_gold_nonempty"],
        ),
        "function_chain_precision": _mean_from_counts(counts, "function_chain_precision_sum", "function_chain_precision_n"),
        "function_chain_recall": _mean_from_counts(counts, "function_chain_recall_sum", "function_chain_recall_n"),
        "function_chain_f1": _mean_from_counts(counts, "function_chain_f1_sum", "function_chain_f1_n"),
        "required_step_recall": _mean_from_counts(counts, "required_step_recall_sum", "required_step_recall_n"),
        "extra_step_rate": _mean_from_counts(counts, "extra_step_rate_sum", "extra_step_rate_n"),
        "step_arg_exact_accuracy": _safe_div(counts["step_arg_exact_count"], counts["aligned_step_count"]),
        "critical_arg_accuracy": _safe_div(counts["step_arg_critical_match_count"], counts["step_arg_critical_total"]),
        "map_values_step_accuracy": _safe_div(counts["map_values_steps_correct"], counts["map_values_steps_gold"]),
        "map_values_key_recall": _safe_div(counts["map_values_key_recall_count"], counts["map_values_steps_gold"]),
        "map_values_value_accuracy": _safe_div(counts["map_values_value_correct_count"], counts["map_values_steps_gold"]),
        "fill_constant_accuracy": _safe_div(counts["fill_constant_correct"], counts["fill_constant_steps_gold"]),
        "dependency_accuracy": _safe_div(counts["dependency_correct"], counts["dependency_steps_gold"]),
        "execution_safe_rate": _safe_div(counts["execution_safe"], counts["n_fields"]),
        "execution_exact_rate": _safe_div(counts["execution_exact"], counts["n_fields"]),
        "field_scores": field_scores,
    }
    if not IGNORE_OUTPUT_DTYPE_IN_T2B_SCORING:
        out["output_dtype_accuracy"] = _safe_div(counts["output_dtype_correct"], counts["n_fields"])
    return out


def score_transformation_map(pred: dict, gold: dict, entity: str) -> dict:
    pred_plans = _get_plans_by_field(pred, entity)
    gold_plans = _get_plans_by_field(gold, entity)

    counts = Counter()
    field_scores = []

    for field, gold_plan in gold_plans.items():
        pred_plan = pred_plans.get(field, {})
        cmp = _compare_transform_plan(field, gold_plan, pred_plan)
        cmp["entity"] = entity
        field_scores.append(cmp)
        _update_counts_t2b(counts, cmp)

    return _finalize_counts_t2b(entity, counts, field_scores)


def validate_transformation_wrapper(m: dict) -> tuple[bool, str | None]:
    """Validate full JSON with cohort + course sections."""
    try:
        inst = m.get("institution_id")
        sv = m.get("schema_version", "0.1.0")
        for entity in ("cohort", "course"):
            sec = m.get("transformation_maps", {}).get(entity)
            if not sec:
                return False, f"Missing transformation_maps.{entity}"
            TransformationMap.model_validate(
                {
                    "schema_version": sv,
                    "institution_id": inst,
                    "entity_type": sec["entity_type"],
                    "target_schema": sec["target_schema"],
                    "plans": sec["plans"],
                }
            )
        return True, None
    except (ValidationError, KeyError, TypeError) as e:
        return False, str(e)


def score_t2b_result(result: dict, gold_tm: dict) -> dict | None:
    if not result["success"]:
        return None
    try:
        pred = json.loads(result["response"])
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed for {result['model']}: {e}")
        return None

    is_valid, validation_error = validate_transformation_wrapper(pred)

    cohort_scores = score_transformation_map(pred, gold_tm, "cohort")
    course_scores = score_transformation_map(pred, gold_tm, "course")

    def _avg(*vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    overall: dict = {
        "plan_decision_accuracy": _avg(
            cohort_scores["plan_decision_accuracy"],
            course_scores["plan_decision_accuracy"],
        ),
        "exact_plan_match_rate": _avg(
            cohort_scores["exact_plan_match_rate"],
            course_scores["exact_plan_match_rate"],
        ),
        "function_chain_accuracy": _avg(
            cohort_scores["function_chain_accuracy"],
            course_scores["function_chain_accuracy"],
        ),
        "function_chain_precision": _avg(
            cohort_scores["function_chain_precision"],
            course_scores["function_chain_precision"],
        ),
        "function_chain_recall": _avg(
            cohort_scores["function_chain_recall"],
            course_scores["function_chain_recall"],
        ),
        "function_chain_f1": _avg(
            cohort_scores["function_chain_f1"],
            course_scores["function_chain_f1"],
        ),
        "required_step_recall": _avg(
            cohort_scores["required_step_recall"],
            course_scores["required_step_recall"],
        ),
        "critical_arg_accuracy": _avg(
            cohort_scores["critical_arg_accuracy"],
            course_scores["critical_arg_accuracy"],
        ),
        "map_values_step_accuracy": _avg(
            cohort_scores["map_values_step_accuracy"],
            course_scores["map_values_step_accuracy"],
        ),
        "dependency_accuracy": _avg(
            cohort_scores["dependency_accuracy"],
            course_scores["dependency_accuracy"],
        ),
        "fill_constant_accuracy": _avg(
            cohort_scores["fill_constant_accuracy"],
            course_scores["fill_constant_accuracy"],
        ),
        "execution_safe_rate": _avg(
            cohort_scores["execution_safe_rate"],
            course_scores["execution_safe_rate"],
        ),
        "execution_exact_rate": _avg(
            cohort_scores["execution_exact_rate"],
            course_scores["execution_exact_rate"],
        ),
    }
    if not IGNORE_OUTPUT_DTYPE_IN_T2B_SCORING:
        od = _avg(
            cohort_scores["output_dtype_accuracy"],
            course_scores["output_dtype_accuracy"],
        )
        overall = {
            "plan_decision_accuracy": overall["plan_decision_accuracy"],
            "output_dtype_accuracy": od,
            **{k: v for k, v in overall.items() if k != "plan_decision_accuracy"},
        }

    return {
        "validation_passed": is_valid,
        "validation_error": validation_error,
        "cohort": cohort_scores,
        "course": course_scores,
        "overall": overall,
    }


def run():
    """Run 2b evaluation on all models."""
    project_root = _find_eval_project_root()
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_id = TARGET_INSTITUTION["id"]
        target_name = TARGET_INSTITUTION["name"]
        reference_id = REFERENCE_INSTITUTION["id"]
        reference_name = REFERENCE_INSTITUTION["name"]

        logger.info("=" * 80)
        logger.info("2b TRANSFORMATION MAP EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Target: {target_name} ({target_id})")
        logger.info(f"Reference: {reference_name} ({reference_id})")
        logger.info(f"Project root (eval cwd): {project_root}")

        HISTORICAL_BASE = Path(f"pipelines/gen_ai_cleaning/historical_examples/{target_id}")
        OUTPUT_DIR = HISTORICAL_BASE / "eval_outputs_2b"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        gold_path = HISTORICAL_BASE / "final_hitl" / f"{target_id}_transformation_map.json"
        if not gold_path.exists():
            gold_path = HISTORICAL_BASE / "v1" / f"{target_id}_transformation_map.json"
        GOLD_TM = load_json(str(gold_path))
        logger.info(f"Loaded gold transformation map from {gold_path}")

        manifest_path = HISTORICAL_BASE / "final_hitl" / f"{target_id}_mapping_manifest.json"
        if not manifest_path.exists():
            manifest_path = HISTORICAL_BASE / "v1" / f"{target_id}_mapping_manifest.json"
        institution_mapping_manifest = load_json(str(manifest_path))
        logger.info(f"Loaded target mapping manifest from {manifest_path}")

        ref_tm_path = (
            Path("pipelines/gen_ai_cleaning/historical_examples")
            / reference_id
            / "final_hitl"
            / f"{reference_id}_transformation_map.json"
        )
        reference_transformation_map = load_json(str(ref_tm_path))
        if reference_transformation_map.get("institution_id") != reference_id:
            raise ValueError(
                f"Reference transformation map institution_id mismatch: expected {reference_id}"
            )
        logger.info(f"Loaded reference transformation map from {ref_tm_path}")

        target_contract_path = (
            HISTORICAL_BASE / f"{target_id}_schema_contract.json"
        )
        target_contract = load_json(str(target_contract_path))
        logger.info(f"Loaded target schema contract from {target_contract_path}")

        PROMPT = build_step2b_prompt(
            institution_id=target_id,
            institution_name=target_name,
            output_path=f"pipelines/gen_ai_cleaning/historical_examples/{target_id}/v0/{target_id}_transformation_map.json",
            institution_mapping_manifest=institution_mapping_manifest,
            institution_schema_contract=target_contract,
            cohort_schema_class=RawEdviseStudentDataSchema,
            course_schema_class=RawEdviseCourseDataSchema,
            reference_transformation_maps=[reference_transformation_map],
            reference_institution_names=[reference_name],
        )

        TOKEN = os.environ.get("DATABRICKS_TOKEN")
        if not TOKEN:
            raise ValueError("DATABRICKS_TOKEN environment variable not set")
        client = OpenAI(
            api_key=TOKEN,
            base_url="https://4437281602191762.ai-gateway.gcp.databricks.com/mlflow/v1",
        )

        rows = []
        for i, model in enumerate(MODELS, start=1):
            logger.info(f"[{i}/{len(MODELS)}] Running model={model}")
            result = run_once(model, PROMPT, client)
            if result["success"]:
                logger.info(
                    f"→ done | success={result['success']} | latency={result['latency_s']}s"
                )
            else:
                logger.error(
                    f"→ done | success={result['success']} | latency={result['latency_s']}s | error={result['error']}"
                )

            slug = folder_slug_2b(model)
            out_dir = HISTORICAL_BASE / slug
            out_dir.mkdir(parents=True, exist_ok=True)
            tm_path = out_dir / f"{target_id}_transformation_map.json"

            if result["success"]:
                try:
                    parsed = json.loads(result["response"])
                    tm_path.write_text(json.dumps(parsed, indent=2))
                    logger.info(f"→ transformation map saved → {tm_path}")
                except json.JSONDecodeError:
                    tm_path.write_text(result["response"])
                    logger.warning(f"→ saved raw (invalid JSON) → {tm_path}")
            else:
                logger.warning("→ transformation map not saved (inference error)")

            scores = score_t2b_result(result, GOLD_TM)
            if scores:
                v_ok = "✓" if scores.get("validation_passed") else "✗"
                ov = scores["overall"]
                _parts = [
                    f"→ scores | validation={v_ok}",
                    f"plan_decision={ov['plan_decision_accuracy']}",
                ]
                if not IGNORE_OUTPUT_DTYPE_IN_T2B_SCORING:
                    _parts.append(f"dtype={ov['output_dtype_accuracy']}")
                _parts.extend(
                    [
                        f"exact_plan={ov['exact_plan_match_rate']}",
                        f"chain_f1={ov['function_chain_f1']}",
                        f"critical_args={ov['critical_arg_accuracy']}",
                        f"exec_safe={ov['execution_safe_rate']}",
                        f"exec_exact={ov['execution_exact_rate']}",
                    ]
                )
                logger.info(" | ".join(_parts))
                if not scores.get("validation_passed"):
                    logger.warning(f"  Validation error: {scores.get('validation_error')}")
                result["scores"] = scores
            else:
                logger.warning("→ scoring skipped")
                result["scores"] = None

            rows.append(result)

        df = pd.DataFrame(rows)
        raw_path = OUTPUT_DIR / f"raw_2b_{RUN_ID}.csv"
        df.drop(columns=["scores"], errors="ignore").to_csv(raw_path, index=False)
        logger.info(f"Raw results saved → {raw_path}")

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
            summary_rows.append(row_data)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = OUTPUT_DIR / f"summary_2b_{RUN_ID}.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved → {summary_path}")

        field_rows = []
        for r in rows:
            if r["scores"] is None:
                continue
            for entity in ("cohort", "course"):
                for fs in r["scores"][entity]["field_scores"]:
                    field_rows.append({"model": r["model"], **fs})

        field_df = pd.DataFrame(field_rows)
        field_path = OUTPUT_DIR / f"field_detail_2b_{RUN_ID}.csv"
        field_df.to_csv(field_path, index=False)
        logger.info(f"Field detail saved → {field_path}")

        if not summary_df.empty:
            cols = ["model", "latency_s"]
            if "validation_passed" in summary_df.columns:
                cols.append("validation_passed")
            _metrics = [
                "plan_decision_accuracy",
                "exact_plan_match_rate",
                "function_chain_accuracy",
                "function_chain_f1",
                "critical_arg_accuracy",
                "map_values_step_accuracy",
                "dependency_accuracy",
                "fill_constant_accuracy",
                "execution_safe_rate",
                "execution_exact_rate",
            ]
            if not IGNORE_OUTPUT_DTYPE_IN_T2B_SCORING:
                _metrics.insert(1, "output_dtype_accuracy")
            for metric in _metrics:
                c = f"overall_{metric}"
                if c in summary_df.columns:
                    cols.append(c)
            cols = [c for c in cols if c in summary_df.columns]
            logger.info("\n" + summary_df[cols].to_string(index=False))
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    run()
