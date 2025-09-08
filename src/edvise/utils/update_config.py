import pathlib
import typing as t
from typing import cast
from tomlkit import parse, dumps, table, TOMLDocument
from tomlkit.container import Container
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
        """
        # Traverse as a generic Container; TOMLDocument is a Container at runtime.
        current: Container = cast(Container, self._doc)

        # Walk down to the parent of the leaf, creating tables as needed.
        for key in key_path[:-1]:
            node = current.get(key)
            if isinstance(node, Container):
                current = node
            else:
                # Create a table if missing or not a container.
                new_tbl: Table = table()  # returns a Table, which is a Container
                current[key] = new_tbl
                current = new_tbl

        # Set the leaf value (tomlkit will wrap primitives as Items).
        current[key_path[-1]] = value
        LOGGER.debug("Set %s to %s", ".".join(key_path), value)

    def save(self) -> None:
        self.path.write_text(dumps(self._doc))
        LOGGER.info("Saved updated TOML to %s", self.path)

    def get(self, key_path: list[str], default: t.Any = None) -> t.Any:
        # Use a broad type for traversal; narrow only when needed.
        obj: t.Any = self._doc
        for key in key_path:
            if not isinstance(obj, Container) or key not in obj:
                return default
            obj = obj[key]
        return obj

    def confirm_field(self, key_path: list[str]) -> None:
        value = self.get(key_path, default=None)
        LOGGER.info("Confirmed %s = %s", ".".join(key_path), value)


def update_run_metadata(config_path: str, run_id: str, experiment_id: str) -> None:
    editor = TomlConfigEditor(config_path)
    editor.update_field(["model", "run_id"], run_id)
    editor.update_field(["model", "experiment_id"], experiment_id)
    editor.save()
    editor.confirm_field(["model", "run_id"])
    editor.confirm_field(["model", "experiment_id"])
