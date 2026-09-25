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
_MIN_FUZZY_TOKEN_LEN = 5
_EXACT_TOKEN_SCORE = 200
_FUZZY_TOKEN_SCORE = 150


def _canonical_token(token: str) -> str:
    return _DATASET_TOKEN_ALIASES.get(token, token)


def _filename_tokens(raw: str, *, canonicalize: bool) -> frozenset[str]:
    stem = pathlib.Path(str(raw).strip()).stem
    tokens: set[str] = set()
    for token in normalize_predict_file_match_text(stem).split("_"):
        if not token or token.isdigit() or token in _GENERIC_TOKENS:
            continue
        if canonicalize:
            token = _canonical_token(token)
        tokens.add(token)
    return frozenset(tokens)


def filename_match_tokens(raw: str) -> frozenset[str]:
    """
    Return stable semantic tokens from a filename or configured keyword.

    Extensions, generic extract words, and numeric date/time/version tokens are
    omitted. Known dataset synonyms are canonicalized so, for example, ``learner``
    and ``student`` compare as the same token.
    """
    return _filename_tokens(raw, canonicalize=True)


def _within_one_edit(left: str, right: str) -> bool:
    """True when both tokens are long enough and a single edit apart."""
    if len(left) < _MIN_FUZZY_TOKEN_LEN or len(right) < _MIN_FUZZY_TOKEN_LEN:
        return False
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left

    index = 0
    while index < len(left) and left[index] == right[index]:
        index += 1
    if index == len(left):
        return True
    if len(left) == len(right):
        if left[index + 1 :] == right[index + 1 :]:
            return True
        return (
            left[index] == right[index + 1]
            and left[index + 1] == right[index]
            and left[index + 2 :] == right[index + 2 :]
        )
    return left[index:] == right[index + 1 :]


def _pair_kind(configured: str, candidate: str) -> str | None:
    if _canonical_token(configured) == _canonical_token(candidate):
        return "exact"
    if _within_one_edit(configured, candidate):
        return "fuzzy"
    return None


def _claim_token(token: str, remaining: set[str], *, kind: str) -> bool:
    if kind == "exact" and token in remaining:
        remaining.remove(token)
        return True
    for other in remaining:
        if _pair_kind(token, other) == kind:
            remaining.remove(other)
            return True
    return False


def _cover_tokens(
    configured: frozenset[str], candidate: frozenset[str]
) -> tuple[int, int] | None:
    """
    Cover every configured token with a distinct candidate token.

    Returns ``(fuzzy_pairs, extra_candidate_tokens)``. Exact and alias pairs are
    claimed before one-edit pairs.
    """
    remaining = set(candidate)
    unmatched = [
        token
        for token in configured
        if not _claim_token(token, remaining, kind="exact")
    ]
    fuzzy = 0
    for token in unmatched:
        if not _claim_token(token, remaining, kind="fuzzy"):
            return None
        fuzzy += 1
    return fuzzy, len(remaining)


def filename_match_score(
    configured_name: str,
    candidate_name: str,
    *,
    dataset_key: str | None = None,
) -> int | None:
    """
    Score a candidate filename, returning ``None`` when it is not a safe match.

    Full normalized substring matches rank first, followed by complete stable-token
    matches. A token of at least 5 characters may match with one edit (a missing,
    extra, or swapped character); that scores below an exact token match. Among
    token matches, closer names (fewer extra candidate tokens) score higher so a
    newer file that only shares vendor/dataset words cannot beat a tighter title
    match. When a dataset key is supplied, a final fallback of 100 applies only if
    some token from that key appears in **both** names, so a bronze slot named
    ``raw_student`` can still bind ``CCC Student File`` to ``Edvise Learner Report``
    without attaching an unrelated configured name such as ``financial aid.csv``.
    """
    configured = normalize_predict_file_match_text(pathlib.Path(configured_name).name)
    candidate = normalize_predict_file_match_text(pathlib.Path(candidate_name).name)
    if configured and configured in candidate:
        return 300

    configured_tokens = _filename_tokens(configured_name, canonicalize=False)
    candidate_tokens = _filename_tokens(candidate_name, canonicalize=False)
    covered = (
        _cover_tokens(configured_tokens, candidate_tokens)
        if configured_tokens
        else None
    )
    if covered is not None:
        fuzzy_pairs, extra = covered
        base = _FUZZY_TOKEN_SCORE if fuzzy_pairs else _EXACT_TOKEN_SCORE
        return base + len(configured_tokens) - extra

    configured_keys = filename_match_tokens(configured_name)
    candidate_keys = filename_match_tokens(candidate_name)
    key_tokens = filename_match_tokens(dataset_key or "")
    if key_tokens & configured_keys & candidate_keys:
        return 100

    return None
