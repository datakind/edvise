"""Resolve dataset files under batch bronze landing dirs (``gcs_uploads/{batch_id}/``)."""

from __future__ import annotations

import logging
import pathlib
import re
from typing import Optional

from edvise.utils.databricks import local_fs_path

LOGGER = logging.getLogger(__name__)

_DATA_FILE_EXTENSIONS = (".csv", ".parquet")
_FILE_KIND_SUFFIX_RE = re.compile(
    r"\b(student|course|semester|degree)\s+file\b",
    re.IGNORECASE,
)


def _is_data_file(path: pathlib.Path) -> bool:
    return path.is_file() and path.suffix.lower() in _DATA_FILE_EXTENSIONS


def _extract_file_kind_suffix(filename: str) -> str | None:
    """Return a normalized ``'{kind} file'`` suffix from a configured filename, if present."""
    stem = pathlib.Path(filename).stem
    match = _FILE_KIND_SUFFIX_RE.search(stem)
    if match is None:
        return None
    return match.group(0).lower()


def _dataset_match_needles(
    configured_name: str,
    *,
    dataset_key: str | None = None,
) -> list[str]:
    """
    Ordered substring needles for batch file discovery.

    Later needles are broader fallbacks when institution/date prefixes differ
    between ``inputs.toml`` and the landed GCS batch (e.g. ``CCC`` vs ``Edvise``).
    """
    needles: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        key = value.lower()
        if key and key not in seen:
            seen.add(key)
            needles.append(value)

    target = pathlib.Path(configured_name.strip()).name
    if target:
        _add(target)

    kind_suffix = _extract_file_kind_suffix(target)
    if kind_suffix:
        _add(kind_suffix)

    ds_key = (dataset_key or "").strip().lower()
    if ds_key in {"student", "course", "semester", "degree"}:
        _add(f"{ds_key} file")

    return needles


def resolve_dataset_file_in_batch_dir(
    batch_dir: str,
    dataset_name: str,
    *,
    dataset_key: str | None = None,
) -> Optional[str]:
    """
    Resolve one dataset file under ``batch_dir``.

    Tries exact basename match first, then case-insensitive substring match using
    progressively broader needles: configured basename, extracted ``'{kind} file'``
    suffix, and ``'{dataset_key} file'`` when provided.
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

    data_files = [p for p in base.iterdir() if _is_data_file(p)]
    for needle in _dataset_match_needles(name, dataset_key=dataset_key):
        needle_l = needle.lower()
        matches = [p for p in data_files if needle_l in p.name.lower()]
        if not matches:
            continue
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if len(matches) > 1:
            LOGGER.info(
                "Multiple files matched dataset_name=%r via needle=%r under %s; "
                "using newest: %s",
                dataset_name,
                needle,
                dir_s,
                matches[0],
            )
        elif needle != target:
            LOGGER.info(
                "Resolved dataset_name=%r via fallback needle=%r under %s: %s",
                dataset_name,
                needle,
                dir_s,
                matches[0],
            )
        return str(matches[0])
    return None
