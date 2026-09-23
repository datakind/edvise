"""Apply promoted GenAI mapping so eligible-terms can use ES columns."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from edvise.data_audit.custom_cleaning import normalize_columns
from edvise.dataio.batch_frame_matching import bind_filenames_to_datasets
from edvise.genai.mapping.identity_agent.term_normalization.schemas import TermContract
from edvise.genai.mapping.identity_agent.term_normalization.term_order import (
    apply_term_order_from_config,
    edvise_term_column_set,
)
from edvise.student_selection.eligible_inference_terms import (
    EligibleInferenceTermsResult,
    invalid_eligible_terms,
    resolve_standardized_eligible_inference_terms,
)

LOGGER = logging.getLogger(__name__)

_COURSE_DATASET_HINTS = ("course",)
_STUDENT_DATASET_HINTS = ("student", "cohort", "learner")
_TERM_TARGETS = ("academic_term", "academic_year")
_ENTRY_TARGETS = ("entry_term", "entry_year")
_ID_TARGETS = ("learner_id", "student_id", "study_id")


def _load_json(payload: dict[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    path = Path(payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return loaded


def _normalize_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    normalized, _mapping = normalize_columns(work.columns)
    work.columns = normalized
    return work


def _dataset_kind(dataset_key: str) -> str | None:
    key = dataset_key.strip().lower()
    if any(hint in key for hint in _COURSE_DATASET_HINTS):
        return "course"
    if any(hint in key for hint in _STUDENT_DATASET_HINTS):
        return "student"
    return None


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) == 1:
        return frames[0]
    return pd.concat(list(frames), ignore_index=True)


def _term_contracts_from_output(
    term_output: dict[str, Any],
    *,
    institution_id: str | None,
) -> dict[str, TermContract]:
    inst = term_output.get("institution_id")
    if institution_id is not None and inst not in (None, institution_id):
        raise ValueError(
            f"term_output institution_id {inst!r} != expected {institution_id!r}"
        )
    datasets = term_output.get("datasets") or {}
    if not isinstance(datasets, dict):
        return {}
    out: dict[str, TermContract] = {}
    for name, payload in datasets.items():
        out[str(name)] = TermContract.model_validate(payload)
    return out


def _apply_term_order_to_datasets(
    frames_by_dataset: dict[str, pd.DataFrame],
    term_contracts: dict[str, TermContract],
    *,
    hook_modules_root: Path | None,
) -> dict[str, pd.DataFrame]:
    applied: dict[str, pd.DataFrame] = {}
    for dataset_key, frame in frames_by_dataset.items():
        contract = term_contracts.get(dataset_key)
        term_config = contract.term_config if contract is not None else None
        if term_config is None:
            applied[dataset_key] = frame
            continue
        if term_config.term_extraction == "hook_required" and hook_modules_root is None:
            LOGGER.warning(
                "Skipping IdentityAgent term order for dataset %s; hook_required "
                "without hook_modules_root.",
                dataset_key,
            )
            applied[dataset_key] = frame
            continue
        try:
            applied[dataset_key] = apply_term_order_from_config(
                frame,
                term_config,
                hook_modules_root=hook_modules_root,
            )
        except Exception:
            LOGGER.exception("Failed applying term order for dataset %s", dataset_key)
            applied[dataset_key] = frame
    return applied


def _manifest_records(manifest_map: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not manifest_map:
        return []
    manifests = manifest_map.get("manifests") or {}
    records: list[dict[str, Any]] = []
    if isinstance(manifests, dict):
        for entity_manifest in manifests.values():
            if not isinstance(entity_manifest, dict):
                continue
            mappings = entity_manifest.get("mappings") or []
            if isinstance(mappings, list):
                records.extend(item for item in mappings if isinstance(item, dict))
    return records


def _project_manifest_columns(
    df: pd.DataFrame,
    records: Sequence[dict[str, Any]],
    targets: Sequence[str],
) -> pd.DataFrame:
    work = df.copy()
    wanted = set(targets)
    for record in records:
        target = record.get("target_field")
        source = record.get("source_column")
        if target not in wanted or not source:
            continue
        if source not in work.columns:
            continue
        if target not in work.columns:
            work[target] = work[source]
    return work


def _alias_identity_term_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    cols = edvise_term_column_set(None)
    if cols.edvise_season in work.columns and "academic_term" not in work.columns:
        work["academic_term"] = work[cols.edvise_season]
    if (
        cols.edvise_academic_year in work.columns
        and "academic_year" not in work.columns
    ):
        work["academic_year"] = work[cols.edvise_academic_year]
    return work


def apply_genai_mapping_for_eligible_terms(
    frames_by_filename: Mapping[str, pd.DataFrame],
    *,
    term_output: dict[str, Any] | Path | str,
    dataset_files: Mapping[str, Sequence[str]],
    manifest_map: dict[str, Any] | Path | str | None = None,
    hook_modules_root: Path | None = None,
    institution_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bind upload files to GenAI datasets, apply IdentityAgent term order, and
    project SMA manifest columns needed for eligible-term discovery.
    """
    if not frames_by_filename:
        raise ValueError("No upload frames were provided.")
    binding = bind_filenames_to_datasets(list(frames_by_filename.keys()), dataset_files)
    if not binding:
        raise ValueError(
            "Could not match upload filenames to GenAI dataset files from inputs.toml."
        )

    frames_by_dataset = {
        dataset_key: _normalize_frame_columns(frames_by_filename[filename])
        for dataset_key, filename in binding.items()
    }
    term_contracts = _term_contracts_from_output(
        _load_json(term_output), institution_id=institution_id
    )
    frames_by_dataset = _apply_term_order_to_datasets(
        frames_by_dataset,
        term_contracts,
        hook_modules_root=hook_modules_root,
    )
    records = _manifest_records(
        _load_json(manifest_map) if manifest_map is not None else None
    )

    student_frames: list[pd.DataFrame] = []
    course_frames: list[pd.DataFrame] = []
    for dataset_key, frame in frames_by_dataset.items():
        kind = _dataset_kind(dataset_key)
        projected = _project_manifest_columns(
            frame, records, _TERM_TARGETS + _ENTRY_TARGETS + _ID_TARGETS
        )
        if kind == "course":
            course_frames.append(_alias_identity_term_columns(projected))
        elif kind == "student":
            student_frames.append(projected)
        elif {"academic_term", "academic_year"} <= set(projected.columns):
            course_frames.append(_alias_identity_term_columns(projected))
        else:
            student_frames.append(projected)

    if not student_frames or not course_frames:
        raise ValueError(
            "GenAI mapping did not produce both student and course frames."
        )
    return _concat_frames(student_frames), _concat_frames(course_frames)


def resolve_genai_eligible_inference_terms(
    frames_by_filename: Mapping[str, pd.DataFrame],
    config: dict[str, Any],
    *,
    term_output: dict[str, Any] | Path | str,
    dataset_files: Mapping[str, Sequence[str]],
    manifest_map: dict[str, Any] | Path | str | None = None,
    hook_modules_root: Path | None = None,
    institution_id: str | None = None,
    batch_name: str | None = None,
) -> EligibleInferenceTermsResult:
    """Discover eligible inference terms from a GenAI upload via promoted mapping."""
    try:
        students, courses = apply_genai_mapping_for_eligible_terms(
            frames_by_filename,
            term_output=term_output,
            dataset_files=dataset_files,
            manifest_map=manifest_map,
            hook_modules_root=hook_modules_root,
            institution_id=institution_id,
        )
    except ValueError as exc:
        return invalid_eligible_terms(str(exc), batch_name)
    return resolve_standardized_eligible_inference_terms(
        students, courses, config, batch_name
    )
