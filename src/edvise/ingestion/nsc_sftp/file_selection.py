"""
Select cohort/course SFTP file pairs for NSC PDP ingestion.

Files are expected to end with a shared 14-digit stamp ``_YYYYMMDDHHMMSS`` and
to contain ``cohort`` or ``course`` in the basename (case-insensitive).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Optional

FILE_STAMP_RE = re.compile(r"_(\d{14})(?:\.[^.]+)?$", re.IGNORECASE)

FileSelectionMode = Literal["manual", "latest", "uningested"]

# Manifest statuses that mean "do not auto-pick this file again".
_INGESTED_STATUSES = frozenset({"BRONZE_WRITTEN"})


@dataclass(frozen=True)
class FilePair:
    stamp: str
    cohort_file_name: str
    course_file_name: str
    cohort_row: dict[str, Any]
    course_row: dict[str, Any]


def extract_file_stamp(file_name: str) -> str:
    """Return the 14-digit trailing stamp from a file name."""
    base = os.path.basename(file_name)
    m = FILE_STAMP_RE.search(base)
    if not m:
        raise ValueError(
            "Expected file name to end with a 14-digit file stamp, e.g. "
            f"'..._YYYYMMDDHHMMSS.csv'. Got: {file_name}"
        )
    return m.group(1)


def try_extract_file_stamp(file_name: str) -> Optional[str]:
    """Like extract_file_stamp but returns None when the stamp is missing."""
    try:
        return extract_file_stamp(file_name)
    except ValueError:
        return None


def classify_pdp_file_role(file_name: str) -> Optional[Literal["cohort", "course"]]:
    """
    Classify an SFTP file as cohort or course from its basename.

    Returns None when the name is ambiguous or does not match either role.
    """
    base = os.path.basename(file_name).lower()
    has_cohort = "cohort" in base
    has_course = "course" in base
    if has_cohort and not has_course:
        return "cohort"
    if has_course and not has_cohort:
        return "course"
    return None


def discover_file_pairs(file_rows: Iterable[Mapping[str, Any]]) -> list[FilePair]:
    """
    Group SFTP listing rows into complete cohort+course pairs by stamp.

    Incomplete pairs (missing cohort or course) are omitted.
    """
    by_stamp: dict[str, dict[str, dict[str, Any]]] = {}
    for row in file_rows:
        name = str(row.get("file_name") or "")
        stamp = try_extract_file_stamp(name)
        if not stamp:
            continue
        role = classify_pdp_file_role(name)
        if role is None:
            continue
        by_stamp.setdefault(stamp, {})[role] = dict(row)

    pairs: list[FilePair] = []
    for stamp, roles in sorted(by_stamp.items(), key=lambda item: item[0]):
        cohort_row = roles.get("cohort")
        course_row = roles.get("course")
        if not cohort_row or not course_row:
            continue
        pairs.append(
            FilePair(
                stamp=stamp,
                cohort_file_name=str(cohort_row["file_name"]),
                course_file_name=str(course_row["file_name"]),
                cohort_row=cohort_row,
                course_row=course_row,
            )
        )
    return pairs


def _pair_is_fully_ingested(
    pair: FilePair,
    fingerprint_by_name: Mapping[str, str],
    status_by_fingerprint: Mapping[str, str],
) -> bool:
    fps = [
        fingerprint_by_name.get(pair.cohort_file_name),
        fingerprint_by_name.get(pair.course_file_name),
    ]
    if not all(fps):
        return False
    statuses = [status_by_fingerprint.get(fp, "") for fp in fps if fp]
    return bool(statuses) and all(s in _INGESTED_STATUSES for s in statuses)


def select_file_pair(
    file_rows: list[Mapping[str, Any]],
    *,
    mode: str,
    cohort_file_name: str = "",
    course_file_name: str = "",
    fingerprint_by_name: Optional[Mapping[str, str]] = None,
    status_by_fingerprint: Optional[Mapping[str, str]] = None,
) -> tuple[str, str, str]:
    """
    Resolve cohort/course file names for an ingestion run.

    Returns:
        (cohort_file_name, course_file_name, selection_mode_used)

    Raises:
        ValueError / FileNotFoundError when selection cannot be resolved.
    """
    cohort_file_name = (cohort_file_name or "").strip()
    course_file_name = (course_file_name or "").strip()
    mode_norm = (mode or "uningested").strip().lower()

    if cohort_file_name and course_file_name:
        cohort_stamp = extract_file_stamp(cohort_file_name)
        course_stamp = extract_file_stamp(course_file_name)
        if cohort_stamp != course_stamp:
            raise ValueError(
                "cohort_file_name and course_file_name must end with the same file stamp. "
                f"Got cohort stamp={cohort_stamp}, course stamp={course_stamp}."
            )
        return cohort_file_name, course_file_name, "manual"

    if mode_norm == "manual":
        raise ValueError(
            "file_selection_mode=manual requires both cohort_file_name and "
            "course_file_name job parameters."
        )

    if mode_norm not in {"latest", "uningested"}:
        raise ValueError(
            f"Unsupported file_selection_mode={mode!r}. "
            "Use 'manual', 'latest', or 'uningested'."
        )

    pairs = discover_file_pairs(file_rows)
    if not pairs:
        available = sorted(
            {str(r.get("file_name")) for r in file_rows if r.get("file_name")}
        )
        raise FileNotFoundError(
            "No complete cohort/course pairs found on SFTP (need both roles sharing a "
            f"14-digit stamp). Available file count={len(available)}; "
            f"first 25={available[:25]}"
        )

    if mode_norm == "uningested":
        fingerprint_by_name = fingerprint_by_name or {}
        status_by_fingerprint = status_by_fingerprint or {}
        eligible = [
            p
            for p in pairs
            if not _pair_is_fully_ingested(
                p, fingerprint_by_name, status_by_fingerprint
            )
        ]
        if not eligible:
            stamps = [p.stamp for p in pairs]
            raise FileNotFoundError(
                "All discovered cohort/course pairs are already BRONZE_WRITTEN in "
                f"ingestion_manifest. stamps={stamps}"
            )
        chosen = max(eligible, key=lambda p: p.stamp)
    else:
        chosen = max(pairs, key=lambda p: p.stamp)

    return chosen.cohort_file_name, chosen.course_file_name, mode_norm
