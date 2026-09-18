"""Resolve dataset files under batch bronze landing dirs (``gcs_uploads/{batch_id}/``)."""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

from edvise.dataio.filename_matching import filename_match_score
from edvise.utils.databricks import local_fs_path

LOGGER = logging.getLogger(__name__)

_DATA_FILE_EXTENSIONS = (".csv", ".parquet")


def _is_data_file(path: pathlib.Path) -> bool:
    return path.is_file() and path.suffix.lower() in _DATA_FILE_EXTENSIONS


def resolve_dataset_file_in_batch_dir(
    batch_dir: str,
    dataset_name: str,
    *,
    dataset_key: str | None = None,
) -> Optional[str]:
    """
    Resolve one dataset file under ``batch_dir``.

    Tries exact basename match first, then ranks candidates using normalized
    substrings, stable filename tokens, and dataset semantics. When several files
    have the best score, the newest one wins.
    """
    dir_s = (batch_dir or "").strip()
    name = (dataset_name or "").strip()
    if not dir_s or not name:
        return None

    base = pathlib.Path(local_fs_path(dir_s))
    if not base.is_dir():
        return None

    target = pathlib.Path(name).name
    exact = base / target
    if _is_data_file(exact):
        return str(exact)

    scored = [
        (score, path)
        for path in base.iterdir()
        if _is_data_file(path)
        and (
            score := filename_match_score(
                name,
                path.name,
                dataset_key=dataset_key,
            )
        )
        is not None
    ]
    if not scored:
        return None

    best_score = max(score for score, _ in scored)
    matches = [path for score, path in scored if score == best_score]
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if len(matches) > 1:
        LOGGER.info(
            "Multiple files matched dataset_name=%r with score=%s under %s; "
            "using newest: %s",
            dataset_name,
            best_score,
            dir_s,
            matches[0],
        )
    else:
        LOGGER.info(
            "Resolved dataset_name=%r with score=%s under %s: %s",
            dataset_name,
            best_score,
            dir_s,
            matches[0],
        )
    return str(matches[0])
