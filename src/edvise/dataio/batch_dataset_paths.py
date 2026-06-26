"""Resolve dataset files under batch bronze landing dirs (``gcs_uploads/{batch_id}/``)."""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

from edvise.utils.databricks import local_fs_path

LOGGER = logging.getLogger(__name__)

_DATA_FILE_EXTENSIONS = (".csv", ".parquet")


def _is_data_file(path: pathlib.Path) -> bool:
    return path.is_file() and path.suffix.lower() in _DATA_FILE_EXTENSIONS


def resolve_dataset_file_in_batch_dir(
    batch_dir: str, dataset_name: str
) -> Optional[str]:
    """
    Resolve one dataset file under ``batch_dir``.

    Tries exact basename match first, then substring match (case-insensitive).
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

    needle = target.lower()
    matches = [
        p for p in base.iterdir() if _is_data_file(p) and needle in p.name.lower()
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(matches) > 1:
        LOGGER.info(
            "Multiple files matched dataset_name=%r under %s; using newest: %s",
            dataset_name,
            dir_s,
            matches[0],
        )
    return str(matches[0])
