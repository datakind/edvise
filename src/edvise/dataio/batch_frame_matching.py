"""Bind in-memory upload filenames to configured dataset keys."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from edvise.dataio.filename_matching import filename_match_score


def bind_filenames_to_datasets(
    available_names: Sequence[str],
    dataset_files: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """
    Map each dataset key to one available filename.

    Scoring uses :func:`filename_match_score` against every configured path for
    that dataset. Each available name is assigned at most once (highest score
    wins, then dataset key order as a tie-break).
    """
    remaining = list(available_names)
    assigned: dict[str, str] = {}
    pending: list[tuple[int, str, str]] = []
    for dataset_key, configured_paths in dataset_files.items():
        for available in remaining:
            best: int | None = None
            for configured in configured_paths:
                score = filename_match_score(
                    Path(configured).name,
                    available,
                    dataset_key=dataset_key,
                )
                if score is None:
                    continue
                if best is None or score > best:
                    best = score
            if best is not None:
                pending.append((best, dataset_key, available))

    pending.sort(key=lambda item: (-item[0], item[1], item[2]))
    used: set[str] = set()
    used_datasets: set[str] = set()
    for _score, dataset_key, available in pending:
        if dataset_key in used_datasets or available in used:
            continue
        assigned[dataset_key] = available
        used.add(available)
        used_datasets.add(dataset_key)
    return assigned
