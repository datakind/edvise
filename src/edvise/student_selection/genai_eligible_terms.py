"""Project a raw GenAI batch onto the columns eligible-term listing expects.

The listing step runs before schema-mapping execute, so only the onboarded identity
term contract, grain contract, ``inputs.toml`` filenames, and a promoted term-hook
module are available. This projects course rows to ``academic_term`` and
``academic_year``, and student rows to the shared id plus ``entry_term`` and
``entry_year`` when the student contract maps them. The caller then passes both
frames to ``resolve_eligible_inference_terms``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
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

_COURSE_NAME_TOKENS = frozenset({"course", "courses"})
_STUDENT_NAME_TOKENS = frozenset({"student", "students", "cohort", "cohorts"})
_STANDARD_ID_COLUMNS = ("learner_id", "student_id", "study_id")


class GenaiTermProjectionError(Exception):
    """The batch cannot be projected from the identity term contract."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class BatchInputFrame(NamedTuple):
    """One uploaded file, its schema tags, and the parsed table."""

    name: str
    schemas: tuple[str, ...]
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

    File roles come from the term and grain contracts. ``inputs.toml`` basenames
    are tried first, then column signatures, so files tagged ``UNKNOWN`` still
    bind. Dataset names are only a tie-break after ``row_selection_required``.
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
        BatchInputFrame(item.name, tuple(item.schemas), _snake_case_frame(item.frame))
        for item in files
    ]
    course_frames, student_frames = _assign_role_frames(
        normalized_files,
        course_name=course_name,
        student_name=student_name,
        course_config=course_config,
        student_config=student_config,
        course_grain=grain_datasets.get(course_name, {}),
        student_grain=grain_datasets.get(student_name, {}),
        dataset_files=dataset_files,
    )
    courses = _concat_frames(course_frames)
    students = _concat_frames(student_frames)
    course_grain = grain_datasets.get(course_name, {})
    student_grain = grain_datasets.get(student_name, {})
    id_column = _output_student_id_column(student_id_col)

    courses = _project_term_labels(
        courses,
        course_config,
        term_column="academic_term",
        year_column="academic_year",
        dataset_label=f"Course dataset {course_name!r}",
        hook_modules_root=hook_modules_root,
    )
    course_id = _require_learner_column(courses, course_grain, course_name, "Course")
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
    student_id = _require_learner_column(
        students, student_grain, student_name, "Student"
    )
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
    """Pick the course-grain dataset and the student dataset."""
    names = list(term.datasets)
    with_term = [name for name in names if term.datasets[name].term_config is not None]
    course_pool = [
        name
        for name in with_term
        if _row_selection_required(grain_datasets.get(name, {})) is True
    ] or with_term
    course_name = _pick_named_dataset(
        course_pool,
        _COURSE_NAME_TOKENS,
        grain_datasets,
        missing="Identity term contract does not identify a course-grain dataset.",
        ambiguous="Identity term contract has more than one course-grain dataset.",
        prefer_longer_key=True,
    )
    student_pool = [name for name in names if name != course_name]
    person_grain = [
        name
        for name in student_pool
        if _row_selection_required(grain_datasets.get(name, {})) is False
    ]
    if person_grain:
        student_pool = person_grain
    student_name = _pick_named_dataset(
        student_pool,
        _STUDENT_NAME_TOKENS,
        grain_datasets,
        missing="Identity term contract does not identify a student dataset.",
        ambiguous="Identity term contract has more than one student dataset.",
        prefer_longer_key=False,
    )
    return course_name, student_name


def _pick_named_dataset(
    pool: Sequence[str],
    tokens: frozenset[str],
    grain_datasets: Mapping[str, Mapping[str, Any]],
    *,
    missing: str,
    ambiguous: str,
    prefer_longer_key: bool,
) -> str:
    named = [name for name in pool if _name_tokens(name) & tokens]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        exact = [name for name in named if name.casefold() in tokens]
        if len(exact) == 1:
            return exact[0]
        raise GenaiTermProjectionError(ambiguous)
    if len(pool) == 1:
        return pool[0]
    if prefer_longer_key and len(pool) > 1:
        ranked = sorted(
            pool,
            key=lambda name: len(_primary_key(grain_datasets.get(name, {}))),
            reverse=True,
        )
        longest = len(_primary_key(grain_datasets.get(ranked[0], {})))
        second = len(_primary_key(grain_datasets.get(ranked[1], {})))
        if longest > second:
            return ranked[0]
    if not pool:
        raise GenaiTermProjectionError(missing)
    raise GenaiTermProjectionError(ambiguous if len(pool) > 1 else missing)


def _assign_role_frames(
    files: Sequence[BatchInputFrame],
    *,
    course_name: str,
    student_name: str,
    course_config: TermOrderConfig,
    student_config: TermOrderConfig | None,
    course_grain: Mapping[str, Any],
    student_grain: Mapping[str, Any],
    dataset_files: Mapping[str, Sequence[str]] | None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    course_terms = _term_source_columns(course_config)
    student_terms = (
        _term_source_columns(student_config) if student_config is not None else []
    )
    course_only = [
        column
        for column in _primary_key(course_grain)
        if column not in _primary_key(student_grain) and column not in course_terms
    ]
    course_frames: list[pd.DataFrame] = []
    student_frames: list[pd.DataFrame] = []
    claimed: set[int] = set()
    if dataset_files:
        for index, item in enumerate(files):
            dataset = _dataset_for_filename(item.name, dataset_files)
            if dataset is None:
                continue
            folded = dataset.casefold()
            if folded == course_name.casefold():
                _require_columns(
                    item,
                    course_terms,
                    f"Course file {item.name!r} is missing term columns",
                )
                if _learner_column(item.frame.columns, course_grain) is None:
                    raise GenaiTermProjectionError(
                        f"Course file {item.name!r} is missing the learner id column."
                    )
                course_frames.append(item.frame)
                claimed.add(index)
            elif folded == student_name.casefold():
                if _learner_column(item.frame.columns, student_grain) is None:
                    raise GenaiTermProjectionError(
                        f"Student file {item.name!r} is missing the learner id column."
                    )
                student_frames.append(item.frame)
                claimed.add(index)
            else:
                claimed.add(index)

    course_tokens = _name_tokens(course_name) | _COURSE_NAME_TOKENS
    student_tokens = _name_tokens(student_name) | _STUDENT_NAME_TOKENS
    if not course_frames:
        chosen = _choose_file(
            files,
            claimed,
            role_schema="COURSE",
            name_tokens=course_tokens,
            eligible=lambda item: _has_course_signature(
                item, course_terms, course_grain
            ),
            bonus=course_only
            + [column for column in course_terms if column not in student_terms],
            penalty=student_terms,
        )
        if chosen is None:
            raise GenaiTermProjectionError(
                "Batch is missing a course file with term columns: "
                + ", ".join(course_terms)
                + "."
            )
        course_frames.append(files[chosen].frame)
        claimed.add(chosen)
    if not student_frames:
        chosen = _choose_file(
            files,
            claimed,
            role_schema="STUDENT",
            name_tokens=student_tokens,
            eligible=lambda item: (
                _learner_column(item.frame.columns, student_grain) is not None
            ),
            bonus=[column for column in student_terms if column not in course_terms],
            penalty=course_only
            + [column for column in course_terms if column not in student_terms],
        )
        if chosen is None:
            raise GenaiTermProjectionError(
                "Batch is missing a student file with a learner id column."
            )
        student_frames.append(files[chosen].frame)
        claimed.add(chosen)
    return course_frames, student_frames


def _choose_file(
    files: Sequence[BatchInputFrame],
    claimed: set[int],
    *,
    role_schema: str,
    name_tokens: frozenset[str],
    eligible: Callable[[BatchInputFrame], bool],
    bonus: Sequence[str],
    penalty: Sequence[str],
) -> int | None:
    scored: list[tuple[int, int]] = []
    for index, item in enumerate(files):
        if index in claimed or not eligible(item):
            continue
        scored.append(
            (
                _role_score(
                    item,
                    role_schema=role_schema,
                    name_tokens=name_tokens,
                    bonus=bonus,
                    penalty=penalty,
                ),
                index,
            )
        )
    if not scored:
        return None
    best = max(score for score, _index in scored)
    winners = [index for score, index in scored if score == best]
    if len(winners) != 1:
        role = "course" if role_schema == "COURSE" else "student"
        names = ", ".join(files[index].name for index in winners)
        raise GenaiTermProjectionError(
            f"Multiple batch files match the {role} dataset in the identity "
            f"term contract: {names}."
        )
    return winners[0]


def _role_score(
    item: BatchInputFrame,
    *,
    role_schema: str,
    name_tokens: frozenset[str],
    bonus: Sequence[str],
    penalty: Sequence[str],
) -> int:
    score = 100
    columns = set(item.frame.columns)
    score += sum(2 for column in bonus if column in columns)
    score -= sum(20 for column in penalty if column in columns)
    schemas = {str(schema).upper() for schema in item.schemas}
    if role_schema in schemas:
        score += 10
    opposite = "STUDENT" if role_schema == "COURSE" else "COURSE"
    if opposite in schemas and role_schema not in schemas:
        score -= 5
    stem = os.path.splitext(os.path.basename(item.name))[0]
    if _name_tokens(stem) & name_tokens:
        score += 5
    return score


def _has_course_signature(
    item: BatchInputFrame,
    term_columns: Sequence[str],
    grain: Mapping[str, Any],
) -> bool:
    columns = item.frame.columns
    return all(column in columns for column in term_columns) and (
        _learner_column(columns, grain) is not None
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
    role: str,
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
        f"{role} file for dataset {dataset_name!r} is missing learner id column "
        f"{expected}."
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


def _primary_key(grain: Mapping[str, Any]) -> list[str]:
    raw = grain.get("post_clean_primary_key") or []
    if not isinstance(raw, list):
        return []
    return [str(convert_to_snake_case(str(column))) for column in raw]


def _row_selection_required(grain: Mapping[str, Any]) -> bool | None:
    value = grain.get("row_selection_required")
    if isinstance(value, bool):
        return value
    return None


def _name_tokens(name: str) -> frozenset[str]:
    return frozenset(
        part for part in name.casefold().replace("-", "_").split("_") if part
    )


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


def _dataset_for_filename(
    filename: str, dataset_files: Mapping[str, Sequence[str]]
) -> str | None:
    scored: list[tuple[int, str]] = []
    for dataset, paths in dataset_files.items():
        best: int | None = None
        for path in paths:
            score = filename_match_score(str(path), filename, dataset_key=str(dataset))
            if score is None:
                continue
            if best is None or score > best:
                best = score
        if best is not None:
            scored.append((best, str(dataset)))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]
