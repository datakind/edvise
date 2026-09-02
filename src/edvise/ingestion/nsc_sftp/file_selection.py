"""
Select cohort/course SFTP file pairs for NSC PDP ingestion.

Files share a trailing 14-digit stamp ``_YYYYMMDDHHMMSS``. Roles match basename
markers (course checked first — it contains the cohort marker as a suffix):

- course: ``COURSE_LEVEL_AR_DEIDENTIFIED_STUDYID``
- cohort: ``AR_DEIDENTIFIED_STUDYID``

Auto modes select every complete pair that shares the same calendar date
(``YYYYMMDD`` prefix of the stamp), not only the newest timestamp.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

FILE_STAMP_RE = re.compile(r"_(\d{14})(?:\.[^.]+)?$", re.IGNORECASE)
COURSE_MARKER = "COURSE_LEVEL_AR_DEIDENTIFIED_STUDYID"
COHORT_MARKER = "AR_DEIDENTIFIED_STUDYID"
FileSelectionMode = Literal["manual", "latest", "skip_ingested"]


@dataclass(frozen=True)
class FilePair:
    stamp: str
    cohort_file_name: str
    course_file_name: str

    @property
    def date(self) -> str:
        """Calendar date ``YYYYMMDD`` from the 14-digit stamp."""
        return self.stamp[:8]


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
    base = os.path.basename(file_name).upper()
    if COURSE_MARKER in base:
        return "course"
    if COHORT_MARKER in base:
        return "cohort"
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


def _pairs_on_date(pairs: Sequence[FilePair], date: str) -> list[FilePair]:
    return [p for p in pairs if p.date == date]


def select_file_pairs(
    file_rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    cohort_file_name: str = "",
    course_file_name: str = "",
    ingested_file_names: Optional[Iterable[str]] = None,
) -> tuple[list[FilePair], str]:
    """
    Resolve cohort/course pairs for an ingestion run.

    ``latest`` / ``skip_ingested`` return **all** complete pairs that share the
    target calendar date (``YYYYMMDD``), because NSC may drop multiple
    timestamped deliveries on the same day.

    ``skip_ingested`` uses the newest date that still has at least one pair not
    fully present in ``ingested_file_names`` (typically BRONZE_WRITTEN names).
    """
    cohort_file_name = (cohort_file_name or "").strip()
    course_file_name = (course_file_name or "").strip()
    mode_norm = (mode or "skip_ingested").strip().lower()

    if cohort_file_name and course_file_name:
        cohort_stamp = extract_file_stamp(cohort_file_name)
        course_stamp = extract_file_stamp(course_file_name)
        if cohort_stamp != course_stamp:
            raise ValueError(
                "cohort_file_name and course_file_name must end with the same file stamp. "
                f"Got cohort stamp={cohort_stamp}, course stamp={course_stamp}."
            )
        return (
            [
                FilePair(
                    stamp=cohort_stamp,
                    cohort_file_name=cohort_file_name,
                    course_file_name=course_file_name,
                )
            ],
            "manual",
        )

    if mode_norm == "manual":
        raise ValueError(
            "file_selection_mode=manual requires both cohort_file_name and "
            "course_file_name job parameters."
        )
    if mode_norm not in {"latest", "skip_ingested"}:
        raise ValueError(
            f"Unsupported file_selection_mode={mode!r}. "
            "Use 'manual', 'latest', or 'skip_ingested'."
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
        target_date = max(p.date for p in pairs)
        return _pairs_on_date(pairs, target_date), mode_norm

    done = set(ingested_file_names or ())
    eligible = [
        p for p in pairs if not ({p.cohort_file_name, p.course_file_name} <= done)
    ]
    if not eligible:
        raise FileNotFoundError(
            "All discovered cohort/course pairs are already BRONZE_WRITTEN in "
            f"ingestion_manifest. stamps={[p.stamp for p in pairs]}"
        )
    target_date = max(p.date for p in eligible)
    # Same-day deliveries: queue every still-pending pair for that date.
    chosen = _pairs_on_date(eligible, target_date)
    return chosen, mode_norm
