"""Step 2a manifest field scoring."""

import json
import logging
from collections import Counter
from copy import deepcopy
from typing import cast

from dotenv import load_dotenv
from pydantic import ValidationError

from edvise.genai.mapping.eval.rates import f1 as _f1
from edvise.genai.mapping.eval.rates import safe_div as _safe_div
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    MappingManifestEnvelope,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


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
