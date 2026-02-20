import pathlib
import logging
import sys
import typing as t

from edvise.utils.databricks import in_databricks
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


def path_exists(p: str) -> bool:
    if not p:
        return False
    # DBFS scheme
    if p.startswith("dbfs:/"):
        try:
            from databricks.sdk.runtime import dbutils  # lazy import

            dbutils.fs.ls(p)  # will raise if not found
            return True
        except Exception:
            return False
    # Local Posix path (Vols are mounted here)
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

    if (
        use_fallback_on_dbx
        and in_databricks()
        and path_exists(fallback_path)
    ):
        LOGGER.info(
            "%s: utilizing training-config path on Databricks: %s",
            label,
            fallback_path,
        )
        return fallback_path

    # If we get here, we couldn't find a usable path
    tried = [p for p in [prefer, fallback_path] if p]
    raise FileNotFoundError(
        f"{label}: none of the candidate paths exist. Tried: {tried}. "
        f"Environment: {'Databricks' if in_databricks() else 'non-Databricks'}"
    )