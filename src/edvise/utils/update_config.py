import pathlib
import typing as t
from tomlkit import parse, dumps, table, TOMLDocument
from tomlkit.items import Table
import logging

LOGGER = logging.getLogger(__name__)


class TomlConfigEditor:
    def __init__(self, config_path: str):
        self.path = pathlib.Path(config_path).resolve()
        self._doc: TOMLDocument = parse(self.path.read_text())

    def update_field(self, key_path: list[str], value: t.Any) -> None:
        """
        Update a nested field in the TOML config given a key path.

        Args:
            key_path: A list of keys representing the path to the field.
                      e.g. ['model', 'run_id'] or ['preprocessing', 'target', 'type']
            value: The value to set at that key path.
        """
        current = self._doc
        for key in key_path[:-1]:
            if key not in current:
                current[key] = table()
            current = current[key]
        current[key_path[-1]] = value
        LOGGER.debug("Set %s to %s", ".".join(key_path), value)

    def save(self) -> None:
        self.path.write_text(dumps(self._doc))
        LOGGER.info("Saved updated TOML to %s", self.path)

    def get(self, key_path: list[str], default: t.Any = None) -> t.Any:
        current = self._doc
        for key in key_path:
            if key not in current:
                return default
            current = current[key]
        return current

    def confirm_field(self, key_path: list[str]) -> None:
        value = self.get(key_path, default=None)
        LOGGER.info("Confirmed %s = %s", ".".join(key_path), value)


def update_run_metadata(config_path: str, run_id: str, experiment_id: str):
    editor = TomlConfigEditor(config_path)
    editor.update_field(["model", "run_id"], run_id)
    editor.update_field(["model", "experiment_id"], experiment_id)
    editor.save()
    editor.confirm_field(["model", "run_id"])
    editor.confirm_field(["model", "experiment_id"])
