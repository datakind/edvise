"""Deterministic filename matching for recurring institutional data extracts."""

from __future__ import annotations

import pathlib

from edvise.dataio.path_management import normalize_predict_file_match_text

_GENERIC_TOKENS = {
    "data",
    "dataset",
    "extract",
    "export",
    "file",
    "report",
}
_DATASET_TOKEN_ALIASES = {
    "cohort": "student",
    "learner": "student",
    "learners": "student",
    "students": "student",
    "courses": "course",
    "semesters": "semester",
    "term": "semester",
    "terms": "semester",
    "award": "degree",
    "awards": "degree",
    "degrees": "degree",
}


def filename_match_tokens(raw: str) -> frozenset[str]:
    """
    Return stable semantic tokens from a filename or configured keyword.

    Extensions, generic extract words, and numeric date/time/version tokens are
    omitted. Known dataset synonyms are canonicalized so, for example, ``learner``
    and ``student`` compare as the same token.
    """
    stem = pathlib.Path(str(raw).strip()).stem
    tokens: set[str] = set()
    for token in normalize_predict_file_match_text(stem).split("_"):
        if not token or token.isdigit() or token in _GENERIC_TOKENS:
            continue
        tokens.add(_DATASET_TOKEN_ALIASES.get(token, token))
    return frozenset(tokens)


def filename_match_score(
    configured_name: str,
    candidate_name: str,
    *,
    dataset_key: str | None = None,
) -> int | None:
    """
    Score a candidate filename, returning ``None`` when it is not a safe match.

    Full normalized substring matches rank first, followed by complete stable-token
    matches. When a dataset key is supplied, a final fallback of 100 applies only if
    some token from that key appears in **both** names, so a bronze slot named
    ``raw_student`` can still bind ``CCC Student File`` to ``Edvise Learner Report``
    without attaching an unrelated configured name such as ``financial aid.csv``.
    """
    configured = normalize_predict_file_match_text(pathlib.Path(configured_name).name)
    candidate = normalize_predict_file_match_text(pathlib.Path(candidate_name).name)
    if configured and configured in candidate:
        return 300

    configured_tokens = filename_match_tokens(configured_name)
    candidate_tokens = filename_match_tokens(candidate_name)
    if configured_tokens and configured_tokens <= candidate_tokens:
        return 200 + len(configured_tokens)

    key_tokens = filename_match_tokens(dataset_key or "")
    if key_tokens & configured_tokens & candidate_tokens:
        return 100

    return None
