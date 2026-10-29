"""Project a raw GenAI batch onto the columns eligible-term listing expects.

Legacy projection will live beside this module. The shared listing function
stays in edvise-api.

The listing step runs before schema-mapping execute, so only the onboarded identity
term contract, grain contract, ``inputs.toml`` filenames, and a promoted term-hook
module are available. This projects course rows to ``academic_term`` and
``academic_year``, and student rows to the shared id plus ``entry_term`` and
``entry_year`` when the student contract maps them. The caller then passes both
frames to ``resolve_eligible_inference_terms``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from pydantic import ValidationError

from edvise.data_audit.custom_cleaning import normalize_columns
from edvise.dataio.filename_matching import filename_match_score
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    InstitutionTermContract,
    TermOrderConfig,
)
from edvise.genai.mapping.identity_agent.term_normalization.term_order import (
    apply_term_order_from_config,
    edvise_term_column_set,
)
from edvise.utils.data_cleaning import convert_to_snake_case

_STANDARD_ID_COLUMNS = ("learner_id", "student_id", "study_id")


class GenaiTermProjectionError(Exception):
    """The batch cannot be projected from the identity term contract."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class BatchInputFrame(NamedTuple):
    """One uploaded file and its parsed table."""

    name: str
    frame: pd.DataFrame


def project_genai_eligible_terms(
    files: Sequence[BatchInputFrame],
    term_contract: Mapping[str, Any],
    *,
    grain_contract: Mapping[str, Any] | None = None,
    dataset_files: Mapping[str, Sequence[str]] | None = None,
    student_id_col: str | None = None,
    hook_modules_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(students, courses)`` with the columns term listing reads.

    Course rows always receive ``academic_term`` and ``academic_year``. Student
    rows receive the shared id, and ``entry_term`` / ``entry_year`` when that
    dataset's term config is present. ``hook_required`` loads
    ``year_extractor*`` and ``season_extractor*`` from ``hook_modules_root``.
    A missing or unimportable hook fails; this does not invent a parser.

    ``row_selection_required`` chooses the course-grain dataset. Files for a
    dataset are resolved by that dataset's name, the same key in the grain
    contract and ``inputs.toml``, using :func:`filename_match_score`.
    """
    term = _parse_term_contract(term_contract)
    grain_datasets = _grain_datasets(grain_contract)
    course_name, student_name = _select_datasets(term, grain_datasets)
    course_config = term.datasets[course_name].term_config
    student_config = term.datasets[student_name].term_config
    if course_config is None:
        raise GenaiTermProjectionError(
            "Identity term contract does not identify a course-grain dataset."
        )

    normalized_files = [
        BatchInputFrame(item.name, _snake_case_frame(item.frame)) for item in files
    ]
    claimed: set[int] = set()
    course_grain = grain_datasets.get(course_name, {})
    student_grain = grain_datasets.get(student_name, {})
    course_frames = _frames_for_dataset(
        normalized_files,
        course_name,
        dataset_files=dataset_files,
        term_columns=_term_source_columns(course_config),
        grain=course_grain,
        claimed=claimed,
    )
    student_term_columns = (
        _term_source_columns(student_config) if student_config is not None else []
    )
    student_frames = _frames_for_dataset(
        normalized_files,
        student_name,
        dataset_files=dataset_files,
        term_columns=student_term_columns,
        grain=student_grain,
        claimed=claimed,
    )
    courses = _concat_frames(course_frames)
    students = _concat_frames(student_frames)
    id_column = _output_student_id_column(student_id_col)

    courses = _project_term_labels(
        courses,
        course_config,
        term_column="academic_term",
        year_column="academic_year",
        dataset_label=f"Course dataset {course_name!r}",
        hook_modules_root=hook_modules_root,
    )
    course_id = _require_learner_column(courses, course_grain, course_name)
    courses = _with_student_id(courses, course_id, id_column)

    if student_config is not None:
        students = _project_term_labels(
            students,
            student_config,
            term_column="entry_term",
            year_column="entry_year",
            dataset_label=f"Student dataset {student_name!r}",
            hook_modules_root=hook_modules_root,
        )
    student_id = _require_learner_column(students, student_grain, student_name)
    students = _with_student_id(students, student_id, id_column)
    return students, courses


def _parse_term_contract(term_contract: Mapping[str, Any]) -> InstitutionTermContract:
    try:
        return InstitutionTermContract.model_validate(term_contract)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        raise GenaiTermProjectionError(
            f"Identity term contract is not valid: {message}"
        ) from exc


def _grain_datasets(
    grain_contract: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(grain_contract, Mapping):
        return {}
    datasets = grain_contract.get("datasets")
    if not isinstance(datasets, Mapping):
        return {}
    return {
        str(name): dict(entry)
        for name, entry in datasets.items()
        if isinstance(entry, Mapping)
    }


def _select_datasets(
    term: InstitutionTermContract,
    grain_datasets: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    """Pick the course-grain dataset name and the student dataset name."""
    names = list(term.datasets)
    course_pool = [
        name
        for name in names
        if _row_selection_required(grain_datasets.get(name, {})) is True
        and term.datasets[name].term_config is not None
    ]
    if not course_pool:
        course_pool = [
            name for name in names if term.datasets[name].term_config is not None
        ]
    course_name = _one_dataset(
        course_pool,
        missing="Identity term contract does not identify a course-grain dataset.",
        ambiguous="Identity term contract has more than one course-grain dataset.",
    )
    student_pool = [
        name
        for name in names
        if name != course_name
        and _row_selection_required(grain_datasets.get(name, {})) is False
    ]
    if not student_pool:
        student_pool = [name for name in names if name != course_name]
    student_name = _one_dataset(
        student_pool,
        missing="Identity term contract does not identify a student dataset.",
        ambiguous="Identity term contract has more than one student dataset.",
    )
    return course_name, student_name


def _one_dataset(pool: Sequence[str], *, missing: str, ambiguous: str) -> str:
    if len(pool) == 1:
        return pool[0]
    if not pool:
        raise GenaiTermProjectionError(missing)
    raise GenaiTermProjectionError(ambiguous)


def _frames_for_dataset(
    files: Sequence[BatchInputFrame],
    dataset_name: str,
    *,
    dataset_files: Mapping[str, Sequence[str]] | None,
    term_columns: Sequence[str],
    grain: Mapping[str, Any],
    claimed: set[int],
) -> list[pd.DataFrame]:
    """Resolve uploads for one dataset name and record them as claimed."""
    frames: list[pd.DataFrame] = []
    for needle in _configured_names(dataset_name, dataset_files):
        index = _match_dataset_file(files, dataset_name, needle, claimed)
        item = files[index]
        _require_dataset_file(item, dataset_name, term_columns, grain)
        frames.append(item.frame)
        claimed.add(index)
    return frames


def _configured_names(
    dataset_name: str, dataset_files: Mapping[str, Sequence[str]] | None
) -> list[str]:
    if dataset_files:
        for key, paths in dataset_files.items():
            if key.casefold() == dataset_name.casefold() and paths:
                return [str(path) for path in paths]
    return [dataset_name]


def _match_dataset_file(
    files: Sequence[BatchInputFrame],
    dataset_name: str,
    needle: str,
    claimed: set[int],
) -> int:
    scored = [
        (score, index)
        for index, item in enumerate(files)
        if index not in claimed
        and (score := filename_match_score(needle, item.name, dataset_key=dataset_name))
        is not None
    ]
    if not scored:
        raise GenaiTermProjectionError(
            f"No batch file matches dataset {dataset_name!r} ({Path(needle).name})."
        )
    best = max(score for score, _index in scored)
    winners = [index for score, index in scored if score == best]
    if len(winners) != 1:
        names = ", ".join(files[index].name for index in winners)
        raise GenaiTermProjectionError(
            f"Multiple batch files match dataset {dataset_name!r}: {names}."
        )
    return winners[0]


def _require_dataset_file(
    item: BatchInputFrame,
    dataset_name: str,
    term_columns: Sequence[str],
    grain: Mapping[str, Any],
) -> None:
    if term_columns:
        _require_columns(
            item,
            term_columns,
            f"Dataset {dataset_name!r} file {item.name!r} is missing term columns",
        )
    if _learner_column(item.frame.columns, grain) is None:
        raise GenaiTermProjectionError(
            f"Dataset {dataset_name!r} file {item.name!r} is missing the learner id column."
        )


def _project_term_labels(
    frame: pd.DataFrame,
    config: TermOrderConfig,
    *,
    term_column: str,
    year_column: str,
    dataset_label: str,
    hook_modules_root: str | Path | None,
) -> pd.DataFrame:
    if config.term_extraction == "hook_required" and hook_modules_root is None:
        raise GenaiTermProjectionError(
            f"{dataset_label} term extraction is hook_required and the promoted "
            "term hook is not available."
        )
    try:
        labeled = apply_term_order_from_config(
            frame, config, hook_modules_root=hook_modules_root
        )
    except (KeyError, ValueError, OSError, ImportError, SyntaxError) as exc:
        raise GenaiTermProjectionError(
            f"Could not project terms for {dataset_label}: {exc}"
        ) from exc
    columns = edvise_term_column_set(None)
    labeled = labeled.copy()
    labeled[term_column] = labeled[columns.edvise_season]
    labeled[year_column] = labeled[columns.edvise_academic_year]
    drop = [
        name
        for name in (
            columns.year,
            columns.season,
            columns.term_order,
            columns.term_grain,
            columns.edvise_season,
            columns.edvise_academic_year,
        )
        if name in labeled.columns
    ]
    return labeled.drop(columns=drop).reset_index(drop=True)


def _output_student_id_column(student_id_col: str | None) -> str:
    if student_id_col and student_id_col.strip():
        return student_id_col.strip()
    return "learner_id"


def _with_student_id(
    frame: pd.DataFrame, source_column: str, id_column: str
) -> pd.DataFrame:
    out = frame.copy()
    if source_column != id_column:
        out[id_column] = out[source_column]
    return out.reset_index(drop=True)


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return pd.concat(list(frames), ignore_index=True)


def _require_columns(
    item: BatchInputFrame, columns: Sequence[str], prefix: str
) -> None:
    missing = [column for column in columns if column not in item.frame.columns]
    if missing:
        raise GenaiTermProjectionError(f"{prefix}: {', '.join(missing)}.")


def _require_learner_column(
    frame: pd.DataFrame,
    grain: Mapping[str, Any],
    dataset_name: str,
) -> str:
    column = _learner_column(frame.columns, grain)
    if column is not None:
        return column
    alias = grain.get("learner_id_alias")
    expected = (
        str(convert_to_snake_case(str(alias)))
        if isinstance(alias, str) and alias.strip()
        else "learner_id, student_id, or study_id"
    )
    raise GenaiTermProjectionError(
        f"Dataset {dataset_name!r} is missing learner id column {expected}."
    )


def _learner_column(
    columns: pd.Index | set[str], grain: Mapping[str, Any]
) -> str | None:
    alias = grain.get("learner_id_alias")
    if isinstance(alias, str) and alias.strip():
        normalized = str(convert_to_snake_case(alias))
        if normalized in columns:
            return normalized
        return None
    for candidate in _STANDARD_ID_COLUMNS:
        if candidate in columns:
            return candidate
    return None


def _term_source_columns(config: TermOrderConfig) -> list[str]:
    if config.term_col is not None:
        return [str(convert_to_snake_case(config.term_col))]
    if config.year_col is not None and config.season_col is not None:
        return [
            str(convert_to_snake_case(config.year_col)),
            str(convert_to_snake_case(config.season_col)),
        ]
    raise GenaiTermProjectionError("Identity term contract is missing a term column.")


def _row_selection_required(grain: Mapping[str, Any]) -> bool | None:
    value = grain.get("row_selection_required")
    if isinstance(value, bool):
        return value
    return None


def _snake_case_frame(frame: pd.DataFrame) -> pd.DataFrame:
    originals = [str(column) for column in frame.columns]
    normalized, mapping = normalize_columns(originals)
    collisions = [name for name, sources in mapping.items() if len(sources) > 1]
    if collisions:
        raise GenaiTermProjectionError(
            "Batch file column names normalize to the same header: "
            + ", ".join(sorted(collisions))
            + "."
        )
    out = frame.copy()
    out.columns = normalized
    return out
