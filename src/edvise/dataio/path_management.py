import logging
import pathlib
import re
import typing as t

from edvise.utils.databricks import in_databricks, local_fs_path

LOGGER = logging.getLogger(__name__)

_BRONZE_PREDICT_FILE_EXTENSIONS = (".csv", ".parquet")
_LEGACY_BRONZE_GCS_UPLOADS_SUBDIR = "gcs_uploads"
_MATCH_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


def normalize_predict_file_match_text(raw: str) -> str:
    """
    Casefold and collapse every non-alphanumeric run to ``_`` for keyword matching.

    Institutions respell the same extract across drops (``DE-ID Transfer File`` vs
    ``de_id_transfer_file``), so separators and case must not decide whether a
    ``predict_file_keyword`` matches.
    """
    return _MATCH_SEPARATOR_RE.sub("_", str(raw).lower()).strip("_")


def predict_file_keywords(ds: t.Mapping[str, t.Any]) -> list[str]:
    """
    Normalized ``predict_file_keyword`` needles for one ``datasets.bronze`` entry.

    Accepts a single string or a list of alternates, so a renamed extract is a config
    change rather than a failed run. Blank entries are dropped.
    """
    raw = ds.get("predict_file_keyword")
    if raw is None:
        return []
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    needles: list[str] = []
    for value in values:
        needle = normalize_predict_file_match_text(value)
        if needle and needle not in needles:
            needles.append(needle)
    return needles


def path_exists(p: str) -> bool:
    if not p:
        return False
    if p.startswith("dbfs:/"):
        try:
            from databricks.sdk.runtime import dbutils  # lazy import

            dbutils.fs.ls(p)  # will raise if not found
            return True
        except Exception:
            return False
    return pathlib.Path(p).exists()


def pick_existing_path(
    prefer_path: t.Optional[str],
    fallback_path: str,
    label: str,
    use_fallback_on_dbx: bool = True,
) -> str:
    """
    prefer_path: inference-provided path (may be None or empty)
    fallback_path: config path
    label: 'course' or 'cohort'
    use_fallback_on_dbx: only fallback to config automatically when on Databricks
    """
    prefer = (prefer_path or "").strip()
    if prefer and path_exists(prefer):
        LOGGER.info("%s: using inference-provided path: %s", label, prefer)
        return prefer

    if prefer and not path_exists(prefer):
        LOGGER.warning("%s: inference-provided path not found: %s", label, prefer)

    if use_fallback_on_dbx and in_databricks() and path_exists(fallback_path):
        LOGGER.info(
            "%s: utilizing training-config path on Databricks: %s",
            label,
            fallback_path,
        )
        return fallback_path

    tried = [p for p in [prefer, fallback_path] if p]
    raise FileNotFoundError(
        f"{label}: none of the candidate paths exist. Tried: {tried}. "
        f"Environment: {'Databricks' if in_databricks() else 'non-Databricks'}"
    )


def legacy_bronze_gcs_uploads_dir(db_workspace: str, institution_id: str) -> str:
    """Default bronze landing dir for validated GCS sync (``bronze_subdir=gcs_uploads``)."""
    ws = (db_workspace or "").strip()
    inst = (institution_id or "").strip()
    if not ws or not inst:
        raise ValueError(
            "db_workspace and institution_id are required for bronze predict discovery."
        )
    return (
        f"/Volumes/{ws}/{inst}_bronze/bronze_volume/{_LEGACY_BRONZE_GCS_UPLOADS_SUBDIR}"
    )


def legacy_bronze_predict_search_dirs(
    db_workspace: str,
    institution_id: str,
    ds: dict[str, t.Any],
    *,
    bronze_batch_dir: str | None = None,
) -> list[str]:
    """
    Directories searched for ``predict_file_keyword`` at legacy inference time.

    Order: batch-scoped ``gcs_uploads/{batch_id}/`` dir (when the batch GCS ingest task
    ran and produced one), then institution top-level ``gcs_uploads``, then parent of
    ``train_file_path`` when set. The batch-scoped dir takes priority since it reflects
    exactly the files validated for this run; top-level ``gcs_uploads`` remains as a
    fallback for runs without a ``batch_id`` (e.g. older or manually-triggered runs).
    """
    dirs: list[str] = []
    batch_dir = (bronze_batch_dir or "").strip()
    if batch_dir:
        dirs.append(batch_dir)
    if (db_workspace or "").strip() and (institution_id or "").strip():
        top_level = legacy_bronze_gcs_uploads_dir(db_workspace, institution_id)
        if top_level not in dirs:
            dirs.append(top_level)
    train = (ds.get("train_file_path") or ds.get("file_path") or "").strip()
    if train:
        parent = str(pathlib.Path(local_fs_path(train)).parent)
        if parent not in dirs:
            dirs.append(parent)
    return dirs


def _is_bronze_predict_candidate(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    return path.suffix.lower() in _BRONZE_PREDICT_FILE_EXTENSIONS


def _keyword_matches_in_directory(
    directory: str, needles: t.Sequence[str]
) -> list[pathlib.Path]:
    """Candidate files in ``directory`` whose normalized name contains any needle."""
    base = pathlib.Path(local_fs_path(directory))
    if not base.is_dir():
        return []
    matches: list[pathlib.Path] = []
    for entry in base.iterdir():
        if not _is_bronze_predict_candidate(entry):
            continue
        haystack = normalize_predict_file_match_text(entry.name)
        if any(needle in haystack for needle in needles):
            matches.append(entry)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches


def find_predict_file_in_directory(
    directory: str,
    *,
    keyword: str | t.Sequence[str],
    label: str = "dataset",
) -> str:
    """
    Resolve a bronze inference file under ``directory`` by filename keyword.

    ``keyword`` may be a single needle or a list of alternates. Matching ignores case
    and separator spelling. When multiple files match, returns the newest by
    modification time.
    """
    dir_s = (directory or "").strip()
    needles = predict_file_keywords({"predict_file_keyword": keyword})
    if not dir_s:
        raise ValueError(f"{label}: search directory must be non-empty.")
    if not needles:
        raise ValueError(f"{label}: predict_file_keyword must be non-empty.")

    base = pathlib.Path(local_fs_path(dir_s))
    if not base.is_dir():
        raise FileNotFoundError(f"{label}: search directory not found: {dir_s}")

    matches = _keyword_matches_in_directory(dir_s, needles)
    if not matches:
        raise FileNotFoundError(
            f"{label}: no matching file under {dir_s} (keyword={keyword!r}). "
            f"Expected extensions: {', '.join(_BRONZE_PREDICT_FILE_EXTENSIONS)}."
        )

    chosen = matches[0]
    if len(matches) > 1:
        LOGGER.info(
            "%s: %d files matched under %s; using newest: %s",
            label,
            len(matches),
            dir_s,
            chosen,
        )
    else:
        LOGGER.info("%s: resolved predict file under %s: %s", label, dir_s, chosen)
    return str(chosen)


def resolve_legacy_bronze_predict_file(
    ds: dict[str, t.Any],
    *,
    dataset_key: str,
    db_workspace: str,
    institution_id: str,
    bronze_batch_dir: str | None = None,
) -> str | None:
    """
    Resolve ``predict_file_path`` for one ``datasets.bronze`` entry at inference time.

    Priority:
      1. Existing ``predict_file_path`` (or legacy ``file_path``) when present on disk
      2. ``predict_file_keyword`` in the **first** search directory that has a match:
         the batch-scoped ``gcs_uploads/{batch_id}/`` dir (when the batch GCS ingest
         task ran), then top-level ``gcs_uploads``, then ``train_file_path`` parent
      3. Otherwise leave unset (school preprocessing may fall back to ``train_file_path``)

    Directory precedence is absolute: a match in the batch dir wins even when a newer
    file matches in a fallback dir, since the batch dir holds exactly the files
    validated for this run. Newest-by-mtime only breaks ties **within** one directory.
    """
    explicit = (ds.get("predict_file_path") or ds.get("file_path") or "").strip()
    needles = predict_file_keywords(ds)

    if needles:
        search_dirs = legacy_bronze_predict_search_dirs(
            db_workspace, institution_id, ds, bronze_batch_dir=bronze_batch_dir
        )
        for dir_ in search_dirs:
            if not pathlib.Path(local_fs_path(dir_)).is_dir():
                LOGGER.warning(
                    "%s: search directory not found, skipping: %s", dataset_key, dir_
                )
                continue
            matches = _keyword_matches_in_directory(dir_, needles)
            if not matches:
                continue
            chosen = matches[0]
            if len(matches) > 1:
                LOGGER.info(
                    "%s: %d files matched keyword=%r under %s; using newest: %s",
                    dataset_key,
                    len(matches),
                    needles,
                    dir_,
                    chosen,
                )
            else:
                LOGGER.info(
                    "%s: resolved predict file via keyword=%r under %s: %s",
                    dataset_key,
                    needles,
                    dir_,
                    chosen,
                )
            return str(chosen)

        if explicit and path_exists(explicit):
            LOGGER.warning(
                "%s: no file matched keyword=%r; falling back to configured path: %s",
                dataset_key,
                needles,
                explicit,
            )
            return explicit
        raise FileNotFoundError(
            f"{dataset_key}: no file matching predict_file_keyword={needles!r} "
            f"under {search_dirs}."
        )

    if explicit and path_exists(explicit):
        LOGGER.info("%s: using configured predict_file_path: %s", dataset_key, explicit)
        return explicit

    if explicit:
        LOGGER.warning(
            "%s: configured predict_file_path not found (%s) and no "
            "predict_file_keyword set",
            dataset_key,
            explicit,
        )
    return None
