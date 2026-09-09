import os
import sys
import logging
import json
import shutil
import argparse
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from edvise.configs.es import ESProjectConfig
    from edvise.configs.legacy import LegacyProjectConfig
    from edvise.configs.pdp import PDPProjectConfig

LOGGER = logging.getLogger(__name__)


class _FlushTolerantStreamHandler(logging.StreamHandler):
    """
    Console handler that ignores flush failures.

    Databricks notebook / ipykernel stdout can raise ``OSError`` (often errno 95,
    *Operation not supported*) on ``flush()``, which otherwise surfaces as
    ``--- Logging error ---`` and noisy tracebacks while the job continues.
    """

    def flush(self) -> None:
        try:
            super().flush()
        except OSError:
            pass


class _FlushTolerantFileHandler(logging.FileHandler):
    """
    File handler that ignores flush failures on unsupported streams.

    Unity Catalog volume / FUSE-backed log paths sometimes raise ``OSError``
    (e.g. errno 95 *Operation not supported*) on ``flush()`` even when
    ``write()`` succeeds; the stdlib would then emit ``--- Logging error ---``
    for every log line.
    """

    def flush(self) -> None:
        try:
            super().flush()
        except OSError:
            pass


class SimpleLogger:
    """
    A JSONL logger that temporarily moves the institution log file to /tmp,
    appends the new entry, and moves it back to the source directory.
    """

    def __init__(self, log_path: str, institution_id: Optional[str] = None):
        self._institution_id = institution_id
        self._final_log_path = log_path

        tmp_dir = "/tmp/logs"
        os.makedirs(tmp_dir, exist_ok=True)

        # Use a temp file per institution to avoid collision
        self._tmp_log_path = os.path.join(
            tmp_dir, f"{institution_id}_validation.tmp.log"
        )

        # If final log exists, move it into tmp before writing
        if os.path.exists(self._final_log_path):
            shutil.copy2(self._final_log_path, self._tmp_log_path)
        else:
            # Create empty temp log file
            open(self._tmp_log_path, "w", encoding="utf-8").close()

        # Open for appending
        self._fh = open(self._tmp_log_path, "a", encoding="utf-8")

    def _write(self, entry: Dict[str, Any]) -> None:
        entry.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
        line = json.dumps(entry, default=str, indent=4)
        self._fh.write(line + "\n")
        self._fh.flush()
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def error(
        self,
        message: Optional[str] = None,
        *,
        extra_columns: Optional[List[str]] = None,
        missing_required: Optional[List[str]] = None,
        schema_errors: Any = None,
        failure_cases: Any = None,
    ) -> None:
        entry: Dict[str, Any] = {"validation_status": "hard_error"}
        if message:
            entry["message"] = message
        if extra_columns is not None:
            entry["extra_columns"] = extra_columns
        if missing_required is not None:
            entry["missing_required"] = missing_required
        if schema_errors is not None:
            entry["schema_errors"] = schema_errors
        if failure_cases is not None:
            entry["failure_cases"] = failure_cases
        self._write(entry)

    def info(self, *, missing_optional: Optional[List[str]] = None) -> None:
        status = "passed_with_soft_errors" if missing_optional else "passed"
        entry = {
            "validation_status": status,
            "missing_optional": missing_optional or [],
        }
        self._write(entry)

    def exception(self, message: str = "Unexpected exception occurred") -> None:
        exc_type, exc_val, _ = sys.exc_info()
        entry = {
            "validation_status": "exception",
            "message": message,
            "error_type": exc_type.__name__ if exc_type else None,
            "error": str(exc_val),
        }
        self._write(entry)

    def close(self) -> None:
        try:
            self._fh.close()
            final_dir = os.path.dirname(self._final_log_path)
            os.makedirs(final_dir, exist_ok=True)
            shutil.move(self._tmp_log_path, self._final_log_path)
        except Exception as e:
            sys.stderr.write(f"[LOGGER] Failed to move final log file: {e}\n")


def setup_logger(
    institution_id: Optional[str] = None, log_file: str = "validation.log"
) -> SimpleLogger:
    if not institution_id:
        raise ValueError("institution_id is required for institution-specific logging")

    log_dir = f"/Volumes/staging_sst_01/{institution_id}_bronze/bronze_volume/logs"
    log_path = os.path.join(log_dir, log_file)

    return SimpleLogger(log_path=log_path, institution_id=institution_id)


def local_fs_path(p: str) -> str:
    return p.replace("dbfs:/", "/dbfs/") if p and p.startswith("dbfs:/") else p


def resolve_genai_segment_log_path(
    run_root: str | os.PathLike[str],
    *,
    mode: str,
    resume_from: str = "start",
) -> str:
    """
    Per-task log file under ``<run_root>/logs/`` (identity_agent / schema_mapping_agent runs).

    Each Databricks task process should use ``append=False`` with this path so gates
    (e.g. ``onboard_gate_1``) do not share one file with ``onboard_start``.

    - Execute: ``.../logs/execute.log``
    - Onboard: ``.../logs/onboard_<resume_from>.log`` (e.g. ``onboard_start.log``)
    """
    base = os.fspath(run_root)
    logs_dir = os.path.join(base, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    mode_l = (mode or "").strip().lower()
    if mode_l == "execute":
        return os.path.join(logs_dir, "execute.log")
    if mode_l == "onboard":
        rf = (resume_from or "start").strip() or "start"
        return os.path.join(logs_dir, f"onboard_{rf}.log")
    raise ValueError(
        f"Invalid mode={mode!r} for segment log (expected 'onboard' or 'execute')."
    )


_INFERENCE_RUN_ID_FILE = "run_id"


def _inference_dir(silver_volume_path: str, model_run_id: str, *parts: str) -> str:
    return os.path.join(silver_volume_path, model_run_id, "inference", *parts)


def _unique_archive_dest(archive_root: str, name: str) -> str:
    dest = os.path.join(archive_root, name)
    if not os.path.exists(dest):
        return dest
    suffix = 2
    while os.path.exists(os.path.join(archive_root, f"{name}_{suffix}")):
        suffix += 1
    return os.path.join(archive_root, f"{name}_{suffix}")


def _read_inference_run_id(inference_root: str) -> Optional[str]:
    path = os.path.join(inference_root, _INFERENCE_RUN_ID_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            value = fh.read().strip()
    except OSError:
        return None
    return value or None


def _write_inference_run_id(inference_root: str, run_id: str) -> None:
    os.makedirs(inference_root, exist_ok=True)
    with open(
        os.path.join(inference_root, _INFERENCE_RUN_ID_FILE), "w", encoding="utf-8"
    ) as fh:
        fh.write(run_id)


def _archive_previous_inference_runs(
    silver_volume_path: str, model_run_id: str, keep_run_id: str
) -> None:
    """Move prior inference artifacts into ``inference/archive``; new run stays in ``inference/``."""
    inference_root = local_fs_path(_inference_dir(silver_volume_path, model_run_id))
    os.makedirs(inference_root, exist_ok=True)
    if _read_inference_run_id(inference_root) == keep_run_id:
        return

    archive_root = local_fs_path(
        _inference_dir(silver_volume_path, model_run_id, "archive")
    )
    previous_run_id = _read_inference_run_id(inference_root) or "legacy"
    leftovers = [name for name in os.listdir(inference_root) if name != "archive"]
    if leftovers:
        dest = _unique_archive_dest(archive_root, previous_run_id)
        os.makedirs(dest, exist_ok=True)
        for name in leftovers:
            shutil.move(os.path.join(inference_root, name), os.path.join(dest, name))
        LOGGER.info("Archived existing inference files -> %s", dest)

    _write_inference_run_id(inference_root, keep_run_id)


def resolve_run_path(
    args: argparse.Namespace,
    cfg: Union["PDPProjectConfig", "ESProjectConfig", "LegacyProjectConfig"],
    silver_volume_path: str,
) -> str:
    """
    Canonical silver folder for a training or inference job.

    * training: ``{silver}/{db_run_id}/training`` (unchanged)
    * inference: ``{silver}/{model_id}/inference`` (same folder as today)

      Files already in ``inference/`` are moved to
      ``inference/archive/<old_run_id>`` first. Later tasks in the same job
      leave that folder in place.
    """
    if args.job_type == "training":
        if not args.db_run_id:
            raise ValueError("db_run_id must be provided for training runs.")
        return os.path.join(silver_volume_path, args.db_run_id, "training")

    if args.job_type == "inference":
        model_run_id: Optional[str] = getattr(
            getattr(cfg, "model", None), "run_id", None
        )
        if not model_run_id:
            raise ValueError("cfg.model.run_id must be set for inference runs.")
        inference_run_id = getattr(args, "db_run_id", None)
        if not inference_run_id:
            return os.path.join(silver_volume_path, model_run_id, "inference")
        _archive_previous_inference_runs(
            silver_volume_path, model_run_id, inference_run_id
        )
        return _inference_dir(silver_volume_path, model_run_id)

    raise ValueError(f"Unsupported job_type: {args.job_type}")


def _sync_file_log_handlers(root: logging.Logger) -> None:
    """Best-effort flush + fsync so UC / FUSE-backed paths show up in volume UIs promptly."""
    for h in root.handlers:
        if not isinstance(h, logging.FileHandler):
            continue
        stream = getattr(h, "stream", None)
        if stream is None:
            continue
        try:
            stream.flush()
        except OSError:
            pass
        try:
            fd = stream.fileno()
            if fd >= 0:
                os.fsync(fd)
        except OSError:
            pass


def init_file_logging_at_path(
    log_file_path: str | os.PathLike[str],
    logger_name: str = __name__,
    *,
    append: bool = False,
) -> str:
    """
    Configure root logging with console + file handlers at a fixed path (Databricks-safe).

    Use when the run directory is known without PDPProjectConfig (e.g. genai mapping jobs).

    On Unity Catalog volumes, ``FileHandler(delay=True)`` can defer creating the file and
    buffered writes may not appear in the catalog file browser until flush/fsync; this
    function opens the log file immediately (``delay=False``) and syncs after bootstrap lines.

    Args:
        append: If True, new log lines are appended to the file (e.g. resume gate_1 after
            start with the same onboard_run_id). If False, the file is truncated on open.

    Returns:
        str: local filesystem path to the log file.
    """
    log_file_path = os.fspath(log_file_path)
    local_path = local_fs_path(log_file_path)
    log_dir = os.path.dirname(local_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    prior_size = (
        os.path.getsize(local_path) if append and os.path.isfile(local_path) else 0
    )

    configure_console_logging()

    root = logging.getLogger()
    file_mode = "a" if append else "w"
    fh = _FlushTolerantFileHandler(
        local_path, mode=file_mode, encoding="utf-8", delay=False
    )
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root.addHandler(fh)

    log = logging.getLogger(logger_name)
    log.info(
        "File logging initialized → %s (mode=%s)",
        local_path,
        "append" if append else "overwrite",
    )
    if append and prior_size > 0:
        log.info(
            "---------- log continues below (prior file size was %d bytes) ----------",
            prior_size,
        )

    _sync_file_log_handlers(root)

    return local_path


def configure_console_logging(level: int = logging.INFO) -> None:
    """
    Databricks-safe root console logging: INFO+ on ``sys.__stdout__``.

    Replaces existing root handlers (clusters often pre-install WARNING-only
    handlers that make a plain ``basicConfig`` a no-op).
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = _FlushTolerantStreamHandler(stream=sys.__stdout__)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root.addHandler(console)
    logging.getLogger("py4j").setLevel(logging.WARNING)


def init_file_logging(
    args: argparse.Namespace,
    cfg: Any,
    logger_name: str = __name__,
    log_file_name: str | None = None,
) -> str:
    """
    Generic, Databricks-safe logger initializer.

    Creates a per-run log file in the correct run directory and
    attaches it to the root logger. Keeps console output (safe for Databricks).

    Args:
        args: argparse.Namespace containing at least silver_volume_path and job_type.
        cfg:  loaded project config (for resolve_run_path).
        logger_name: optional module logger name.
        log_file_name: optional filename override; defaults to "<job_type>.log".

    Returns:
        str: local filesystem path to the log file.
    """
    current_run_path = resolve_run_path(args, cfg, args.silver_volume_path)
    local_run_path = local_fs_path(current_run_path)
    os.makedirs(local_run_path, exist_ok=True)

    job_type = getattr(args, "job_type", None) or "generic"
    log_file_name = log_file_name or f"{job_type}.log"
    log_file_path = os.path.join(local_run_path, log_file_name)

    return init_file_logging_at_path(log_file_path, logger_name)
