"""
Rules-based selection of a pinned GenAI few-shot reference for SMA onboard.

Scores active pins against the query institution's post-HITL
``enriched_schema_contract.json``, then materializes a run-local snapshot under
``schema_mapping_agent/few_shot/`` so Step 2a and gate 2 share one immutable copy.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from edvise.genai.mapping.shared.reference_pin import (
    GENAI_REFERENCE_PIN_BASENAME,
    REFERENCE_PIN_REQUIRED_ARTIFACTS,
    SmaFewShotPin,
    resolve_sma_few_shot_pin,
    verify_reference_pin_hash,
)
from edvise.genai.mapping.shared.volume_paths import genai_reference_current_root

LOGGER = logging.getLogger(__name__)

# Weights: datasets / columns / grain co-equal; term is a light tie-breaker.
WEIGHT_DATASETS: float = 0.30
WEIGHT_COLUMNS: float = 0.30
WEIGHT_GRAIN: float = 0.30
WEIGHT_TERM: float = 0.10

FEW_SHOT_DIRNAME: str = "few_shot"
SELECTION_AUDIT_BASENAME: str = "selection.json"
ENRICHED_CONTRACT_BASENAME: str = "enriched_schema_contract.json"


def jaccard(a: set[str], b: set[str]) -> float:
    """``|A∩B| / |A∪B|``; empty–empty is ``1.0``; one empty side is ``0.0``."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _dataset_map(contract: Mapping[str, Any]) -> dict[str, Any]:
    raw = contract.get("datasets")
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, Mapping)}


def _cleaned_column_names(dataset: Mapping[str, Any]) -> set[str]:
    """Cleaned names = values of ``normalized_columns`` (original → cleaned)."""
    nc = dataset.get("normalized_columns")
    if not isinstance(nc, Mapping):
        return set()
    return {str(v).strip() for v in nc.values() if str(v).strip()}


def _unique_key_set(dataset: Mapping[str, Any]) -> set[str]:
    keys = dataset.get("unique_keys")
    if not isinstance(keys, (list, tuple)):
        return set()
    return {str(k).strip() for k in keys if str(k).strip()}


def _term_normalization_block(dataset: Mapping[str, Any]) -> dict[str, Any] | None:
    training = dataset.get("training")
    if not isinstance(training, Mapping):
        return None
    tn = training.get("term_normalization")
    if not isinstance(tn, Mapping):
        return None
    return dict(tn)


@dataclass(frozen=True)
class ReferenceScore:
    """Per-candidate score breakdown (all component scores in ``[0, 1]``)."""

    reference_id: str
    total: float
    score_datasets: float
    score_columns: float
    score_grain: float
    score_term: float
    shared_datasets: tuple[str, ...] = ()

    def as_audit_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "total": self.total,
            "score_datasets": self.score_datasets,
            "score_columns": self.score_columns,
            "score_grain": self.score_grain,
            "score_term": self.score_term,
            "shared_datasets": list(self.shared_datasets),
        }


def score_reference_contracts(
    query_contract: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
    *,
    reference_id: str = "",
) -> ReferenceScore:
    """
    Score one candidate enriched schema contract against the query contract.

    See plan: dataset-key Jaccard, per-shared-dataset column / unique_keys Jaccard,
    term mode+extraction agreement on comparable shared datasets.
    """
    q_ds = _dataset_map(query_contract)
    c_ds = _dataset_map(candidate_contract)
    q_keys = set(q_ds)
    c_keys = set(c_ds)
    shared = sorted(q_keys & c_keys)

    score_datasets = jaccard(q_keys, c_keys)

    if not shared:
        return ReferenceScore(
            reference_id=str(reference_id or "").strip(),
            total=(
                WEIGHT_DATASETS * score_datasets
                + WEIGHT_COLUMNS * 0.0
                + WEIGHT_GRAIN * 0.0
                + WEIGHT_TERM * 0.0
            ),
            score_datasets=score_datasets,
            score_columns=0.0,
            score_grain=0.0,
            score_term=0.0,
            shared_datasets=(),
        )

    col_scores: list[float] = []
    grain_scores: list[float] = []
    term_scores: list[float] = []
    for d in shared:
        col_scores.append(
            jaccard(_cleaned_column_names(q_ds[d]), _cleaned_column_names(c_ds[d]))
        )
        grain_scores.append(jaccard(_unique_key_set(q_ds[d]), _unique_key_set(c_ds[d])))
        q_tn = _term_normalization_block(q_ds[d])
        c_tn = _term_normalization_block(c_ds[d])
        if q_tn is not None and c_tn is not None:
            agree = (
                str(q_tn.get("mode") or "").strip()
                == str(c_tn.get("mode") or "").strip()
                and str(q_tn.get("term_extraction") or "").strip()
                == str(c_tn.get("term_extraction") or "").strip()
            )
            term_scores.append(1.0 if agree else 0.0)

    score_columns = _mean(col_scores)
    score_grain = _mean(grain_scores)
    score_term = _mean(term_scores)  # empty comparable → 0.0

    total = (
        WEIGHT_DATASETS * score_datasets
        + WEIGHT_COLUMNS * score_columns
        + WEIGHT_GRAIN * score_grain
        + WEIGHT_TERM * score_term
    )
    return ReferenceScore(
        reference_id=str(reference_id or "").strip(),
        total=total,
        score_datasets=score_datasets,
        score_columns=score_columns,
        score_grain=score_grain,
        score_term=score_term,
        shared_datasets=tuple(shared),
    )


def list_active_reference_pins(
    catalog: str,
    *,
    spark: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Return all ``status='active'`` rows from ``{catalog}.genai_mapping.reference_pins``.

    When Spark is unavailable, returns ``[]`` (callers must fail or supply candidates).
    """
    from edvise.genai.mapping.state._sql import (
        REFERENCE_PINS,
        get_spark_session,
        qualified_table,
    )

    c = str(catalog).strip()
    if not c:
        raise ValueError("catalog must be non-empty")

    spark = spark if spark is not None else get_spark_session()
    if spark is None:
        LOGGER.warning(
            "No Spark session; cannot list active reference_pins (catalog=%r)", c
        )
        return []

    t = qualified_table(c, REFERENCE_PINS)
    rows = spark.sql(
        f"""
        SELECT
          reference_id,
          archetype,
          pipeline_version,
          content_hash,
          pinned_at,
          pinned_by,
          source_onboard_run_id,
          source_institution_id,
          status,
          uc_catalog,
          artifacts,
          pin_path
        FROM {t}
        WHERE status = 'active'
        ORDER BY reference_id ASC
        """
    ).collect()

    out: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "asDict"):
            out.append(dict(row.asDict()))  # type: ignore[attr-defined]
        else:
            out.append(
                {
                    "reference_id": row["reference_id"],
                    "archetype": row["archetype"],
                    "pipeline_version": row["pipeline_version"],
                    "content_hash": row["content_hash"],
                    "pinned_at": row["pinned_at"],
                    "pinned_by": row["pinned_by"],
                    "source_onboard_run_id": row["source_onboard_run_id"],
                    "source_institution_id": row["source_institution_id"],
                    "status": row["status"],
                    "uc_catalog": row["uc_catalog"],
                    "artifacts": row["artifacts"],
                    "pin_path": row["pin_path"],
                }
            )
    return out


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _pin_has_required_few_shot_and_contract(
    reference_id: str, *, catalog: str
) -> tuple[bool, str | None]:
    """
    Return ``(ok, skip_reason)``.

    Requires verifyable pin + required few-shot files + enriched schema contract.
    """
    ref = str(reference_id).strip()
    current = Path(genai_reference_current_root(ref, catalog=catalog))
    if not current.is_dir():
        return False, f"missing current/ at {current}"
    try:
        verify_reference_pin_hash(current)
    except (FileNotFoundError, ValueError, TypeError) as e:
        return False, f"pin verify failed: {e}"
    for name in REFERENCE_PIN_REQUIRED_ARTIFACTS:
        if not (current / name).is_file():
            return False, f"missing required artifact {name!r}"
    if not (current / ENRICHED_CONTRACT_BASENAME).is_file():
        return False, f"missing {ENRICHED_CONTRACT_BASENAME} (required for selection)"
    return True, None


@dataclass
class ReferenceSelectionResult:
    """Outcome of rules-based (or override) reference selection."""

    reference_id: str
    content_hash: str
    selection_mode: str  # "auto" | "override"
    scores: list[ReferenceScore] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def as_audit_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "content_hash": self.content_hash,
            "selection_mode": self.selection_mode,
            "candidates": [s.as_audit_dict() for s in self.scores],
            "skipped": list(self.skipped),
        }


def select_reference_id(
    *,
    catalog: str,
    institution_id: str,
    query_contract: Mapping[str, Any],
    spark: Any | None = None,
    active_pins: Sequence[Mapping[str, Any]] | None = None,
) -> ReferenceSelectionResult:
    """
    Choose a pinned reference for SMA onboard.

    Scores all active pins that pass hard filters and picks the top total
    (tie-break: lexicographic ``reference_id``). Excludes ``reference_id == institution_id``.
    """
    inst = str(institution_id).strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")
    c = str(catalog).strip()
    if not c:
        raise ValueError("catalog must be non-empty")

    pins = (
        list(active_pins)
        if active_pins is not None
        else list_active_reference_pins(c, spark=spark)
    )
    scores: list[ReferenceScore] = []
    skipped: list[dict[str, str]] = []

    for row in pins:
        ref = str(row.get("reference_id") or "").strip()
        if not ref:
            skipped.append({"reference_id": "", "reason": "empty reference_id"})
            continue
        if ref == inst:
            skipped.append({"reference_id": ref, "reason": "self-exclude"})
            continue
        ok, reason = _pin_has_required_few_shot_and_contract(ref, catalog=c)
        if not ok:
            skipped.append({"reference_id": ref, "reason": reason or "filtered"})
            LOGGER.info("Skipping reference_id=%r for selection: %s", ref, reason)
            continue
        contract_path = (
            Path(genai_reference_current_root(ref, catalog=c))
            / ENRICHED_CONTRACT_BASENAME
        )
        try:
            cand_contract = _load_json_object(contract_path)
        except (OSError, json.JSONDecodeError, TypeError) as e:
            skipped.append({"reference_id": ref, "reason": f"unreadable contract: {e}"})
            continue
        scores.append(
            score_reference_contracts(query_contract, cand_contract, reference_id=ref)
        )

    if not scores:
        raise ValueError(
            f"No selectable active references for institution_id={inst!r} "
            f"catalog={c!r}. Skipped={skipped!r}. Pin gold schools with "
            f"{ENRICHED_CONTRACT_BASENAME} and required few-shot artifacts."
        )

    # Max total; tie-break lexicographic reference_id.
    scores.sort(key=lambda s: (-s.total, s.reference_id))
    winner = scores[0]
    few = resolve_sma_few_shot_pin(winner.reference_id, catalog=c)
    LOGGER.info(
        "Auto-selected reference_id=%r total=%.4f "
        "(datasets=%.4f columns=%.4f grain=%.4f term=%.4f) from %d candidates",
        winner.reference_id,
        winner.total,
        winner.score_datasets,
        winner.score_columns,
        winner.score_grain,
        winner.score_term,
        len(scores),
    )
    return ReferenceSelectionResult(
        reference_id=few.reference_id,
        content_hash=few.content_hash,
        selection_mode="auto",
        scores=scores,
        skipped=skipped,
    )


@dataclass(frozen=True)
class RunFewShotSnapshot:
    """Run-local few-shot copy under ``schema_mapping_agent/few_shot/``."""

    reference_id: str
    content_hash: str
    root: Path
    manifest_map: Path
    transformation_map: Path
    selection_audit: Path | None


def few_shot_snapshot_root(sma_run_root: str | Path) -> Path:
    """``…/schema_mapping_agent/few_shot``."""
    return Path(sma_run_root) / FEW_SHOT_DIRNAME


def resolve_run_few_shot_snapshot(
    sma_run_root: str | Path,
) -> RunFewShotSnapshot | None:
    """
    Load an existing run-local few-shot snapshot, or ``None`` if not materialized.

    Does **not** re-read ``references/*/current/``.
    """
    root = few_shot_snapshot_root(sma_run_root)
    pin_path = root / GENAI_REFERENCE_PIN_BASENAME
    manifest = root / "manifest_map.json"
    tm = root / "transformation_map.json"
    if not pin_path.is_file() or not manifest.is_file() or not tm.is_file():
        return None
    pin = _load_json_object(pin_path)
    ref = str(pin.get("reference_id") or "").strip()
    content_hash = str(pin.get("content_hash") or "").strip()
    if not ref or not content_hash:
        raise ValueError(
            f"Run few-shot snapshot at {root} missing reference_id/content_hash "
            f"in {GENAI_REFERENCE_PIN_BASENAME}"
        )
    audit = root / SELECTION_AUDIT_BASENAME
    return RunFewShotSnapshot(
        reference_id=ref,
        content_hash=content_hash,
        root=root,
        manifest_map=manifest,
        transformation_map=tm,
        selection_audit=audit if audit.is_file() else None,
    )


def materialize_run_few_shot_snapshot(
    sma_run_root: str | Path,
    *,
    catalog: str,
    reference_id: str,
    selection: ReferenceSelectionResult | None = None,
) -> RunFewShotSnapshot:
    """
    Copy verified library ``current/`` few-shot artifacts into the SMA run tree.

    If a complete snapshot already exists, reuse it (resume / gate_2 lock).
    """
    existing = resolve_run_few_shot_snapshot(sma_run_root)
    if existing is not None:
        LOGGER.info(
            "Reusing run few-shot snapshot reference_id=%r hash=%s at %s",
            existing.reference_id,
            existing.content_hash,
            existing.root,
        )
        return existing

    few = resolve_sma_few_shot_pin(reference_id, catalog=catalog)
    root = few_shot_snapshot_root(sma_run_root)
    root.mkdir(parents=True, exist_ok=True)

    # Copy required few-shot + pin sidecar (+ optional contract when present).
    src_root = few.current_root
    for name in (
        *REFERENCE_PIN_REQUIRED_ARTIFACTS,
        GENAI_REFERENCE_PIN_BASENAME,
        ENRICHED_CONTRACT_BASENAME,
    ):
        src = src_root / name
        if src.is_file():
            shutil.copy2(src, root / name)

    if selection is not None:
        audit_path = root / SELECTION_AUDIT_BASENAME
        payload = selection.as_audit_dict()
        # Ensure audit hash matches the verified pin we copied.
        payload["reference_id"] = few.reference_id
        payload["content_hash"] = few.content_hash
        text = json.dumps(payload, indent=2) + "\n"
        tmp = audit_path.with_name(f".{audit_path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(audit_path)

    snap = resolve_run_few_shot_snapshot(sma_run_root)
    if snap is None:
        raise RuntimeError(f"Failed to materialize few-shot snapshot under {root}")
    if snap.content_hash != few.content_hash or snap.reference_id != few.reference_id:
        raise ValueError(
            f"Snapshot mismatch after copy: expected {few.reference_id!r}/{few.content_hash!r}, "
            f"got {snap.reference_id!r}/{snap.content_hash!r}"
        )
    LOGGER.info(
        "Materialized run few-shot snapshot reference_id=%r hash=%s -> %s",
        snap.reference_id,
        snap.content_hash,
        snap.root,
    )
    return snap


def ensure_run_few_shot(
    sma_run_root: str | Path,
    *,
    catalog: str,
    institution_id: str,
    query_contract: Mapping[str, Any],
    spark: Any | None = None,
    active_pins: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[RunFewShotSnapshot, ReferenceSelectionResult | None]:
    """
    Resolve-or-materialize the run few-shot snapshot used by SMA 2a/2b.

    Resume rule: if ``few_shot/`` already exists, reuse it and skip re-selection.
    """
    existing = resolve_run_few_shot_snapshot(sma_run_root)
    if existing is not None:
        return existing, None

    selection = select_reference_id(
        catalog=catalog,
        institution_id=institution_id,
        query_contract=query_contract,
        spark=spark,
        active_pins=active_pins,
    )
    snap = materialize_run_few_shot_snapshot(
        sma_run_root,
        catalog=catalog,
        reference_id=selection.reference_id,
        selection=selection,
    )
    return snap, selection


def snapshot_as_sma_few_shot_pin(snap: RunFewShotSnapshot) -> SmaFewShotPin:
    """Adapt a run snapshot to the ``SmaFewShotPin`` shape used by SMA prompt loaders."""
    return SmaFewShotPin(
        reference_id=snap.reference_id,
        current_root=snap.root,
        manifest_map=snap.manifest_map,
        transformation_map=snap.transformation_map,
        content_hash=snap.content_hash,
    )


__all__ = [
    "ENRICHED_CONTRACT_BASENAME",
    "FEW_SHOT_DIRNAME",
    "ReferenceScore",
    "ReferenceSelectionResult",
    "RunFewShotSnapshot",
    "SELECTION_AUDIT_BASENAME",
    "WEIGHT_COLUMNS",
    "WEIGHT_DATASETS",
    "WEIGHT_GRAIN",
    "WEIGHT_TERM",
    "ensure_run_few_shot",
    "few_shot_snapshot_root",
    "jaccard",
    "list_active_reference_pins",
    "materialize_run_few_shot_snapshot",
    "resolve_run_few_shot_snapshot",
    "score_reference_contracts",
    "select_reference_id",
    "snapshot_as_sma_few_shot_pin",
]
