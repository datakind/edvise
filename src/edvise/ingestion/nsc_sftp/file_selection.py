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


@dataclass(frozen=True)
class FilePair:
    stamp: str
    cohort_file_name: str
    course_file_name: str


def extract_file_stamp(file_name: str) -> str:
    """Return the 14-digit trailing stamp from a file name."""
    m = FILE_STAMP_RE.search(os.path.basename(file_name))
    if not m:
        raise ValueError(
            "Expected file name to end with a 14-digit file stamp, e.g. "
            f"'..._YYYYMMDDHHMMSS.csv'. Got: {file_name}"
        )
    return m.group(1)


def try_extract_file_stamp(file_name: str) -> Optional[str]:
    try:
        return extract_file_stamp(file_name)
    except ValueError:
        return None


def classify_pdp_file_role(file_name: str) -> Optional[Literal["cohort", "course"]]:
    base = os.path.basename(file_name).lower()
    has_cohort = "cohort" in base
    has_course = "course" in base
    if has_cohort and not has_course:
        return "cohort"
    if has_course and not has_cohort:
        return "course"
    return None


def discover_file_pairs(file_rows: Iterable[Mapping[str, Any]]) -> list[FilePair]:
    """Group SFTP listing rows into complete cohort+course pairs by stamp."""
    by_stamp: dict[str, dict[str, str]] = {}
    for row in file_rows:
        name = str(row.get("file_name") or "")
        stamp = try_extract_file_stamp(name)
        if not stamp:
            continue
        role = classify_pdp_file_role(name)
        if role is None:
            continue
        by_stamp.setdefault(stamp, {})[role] = name

    return [
        FilePair(
            stamp=stamp,
            cohort_file_name=roles["cohort"],
            course_file_name=roles["course"],
        )
        for stamp, roles in sorted(by_stamp.items())
        if "cohort" in roles and "course" in roles
    ]


def select_file_pair(
    file_rows: list[Mapping[str, Any]],
    *,
    mode: str,
    cohort_file_name: str = "",
    course_file_name: str = "",
    ingested_file_names: Optional[Iterable[str]] = None,
) -> tuple[str, str, str]:
    """
    Resolve cohort/course file names for an ingestion run.

    ``uningested`` skips pairs whose cohort and course names are both present in
    ``ingested_file_names`` (typically BRONZE_WRITTEN file_name values). Stamp-based
    NSC names make file_name a stable version key without Spark fingerprinting.
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

    if mode_norm == "latest":
        chosen = max(pairs, key=lambda p: p.stamp)
        return chosen.cohort_file_name, chosen.course_file_name, mode_norm

    done = set(ingested_file_names or ())
    eligible = [
        p for p in pairs if not ({p.cohort_file_name, p.course_file_name} <= done)
    ]
    if not eligible:
        raise FileNotFoundError(
            "All discovered cohort/course pairs are already BRONZE_WRITTEN in "
            f"ingestion_manifest. stamps={[p.stamp for p in pairs]}"
        )
    chosen = max(eligible, key=lambda p: p.stamp)
    return chosen.cohort_file_name, chosen.course_file_name, mode_norm
