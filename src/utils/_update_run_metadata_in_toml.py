import logging
import typing as t
import pathlib
import tomlkit
from tomlkit.items import Table


LOGGER = logging.getLogger(__name__)

def update_run_metadata_in_toml(
    config_path: str, run_id: str, experiment_id: str
) -> None:
    """
    Update the 'model.run_id' and 'model.experiment_id' fields in a TOML config file,
    preserving the original formatting and structure using tomlkit.

    Args:
        config_path: Path to the TOML config file.
        run_id: The run ID to set.
        experiment_id: The experiment ID to set.
    """
    path = pathlib.Path(config_path).resolve()  # Absolute path

    try:
        doc = tomlkit.parse(path.read_text())

        if "model" not in doc:
            doc["model"] = tomlkit.table()

        model_table = t.cast(Table, doc["model"])
        model_table["run_id"] = run_id
        model_table["experiment_id"] = experiment_id

        path.write_text(tomlkit.dumps(doc))

        LOGGER.info(
            "Updated TOML config at %s with run_id=%s and experiment_id=%s",
            path,
            run_id,
            experiment_id,
        )

        # Re-read to confirm
        confirmed = tomlkit.parse(path.read_text())
        confirmed_model = t.cast(Table, confirmed["model"])
        LOGGER.info("Confirmed run_id = %s", confirmed_model.get("run_id"))
        LOGGER.info(
            "Confirmed experiment_id = %s", confirmed_model.get("experiment_id")
        )

    except Exception as e:
        LOGGER.error("Failed to update TOML config at %s: %s", path, e)
        raise
