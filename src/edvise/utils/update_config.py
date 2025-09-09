import pathlib
from typing import cast, MutableMapping, Any
from tomlkit import parse, dumps, table, TOMLDocument
from tomlkit.items import Table
import logging

LOGGER = logging.getLogger(__name__)


class TomlConfigEditor:
    def __init__(self, config_path: str):
        self.path = pathlib.Path(config_path).resolve()
        self._doc: TOMLDocument = parse(self.path.read_text())

    def update_field(self, key_path: list[str], value: Any) -> None:
        """
        Update a nested field in the TOML config given a key path.
        """
        # Work with dict-like interface; TOMLDocument/Table implement MutableMapping.
        current: MutableMapping[str, Any] = cast(MutableMapping[str, Any], self._doc)

        # Walk down to the parent of the leaf, creating tables as needed.
        for key in key_path[:-1]:
            node = current.get(key)
            if isinstance(node, MutableMapping):
                current = cast(MutableMapping[str, Any], node)
            else:
                # Create a table if missing or not a mapping.
                new_tbl: Table = table()
                current[key] = new_tbl  # type: ignore[assignment]  # (value type Any accepts Table)
                current = cast(MutableMapping[str, Any], new_tbl)

        # Set the leaf value (tomlkit will wrap primitives as Items).
        current[key_path[-1]] = value
        LOGGER.debug("Set %s to %s", ".".join(key_path), value)

    def save(self) -> None:
        self.path.write_text(dumps(self._doc))
        LOGGER.info("Saved updated TOML to %s", self.path)

    def get(self, key_path: list[str], default: Any = None) -> Any:
        obj: Any = self._doc
        for key in key_path:
            if not isinstance(obj, MutableMapping) or key not in obj:
                return default
            obj = obj[key]
        return obj

    def confirm_field(self, key_path: list[str]) -> None:
        value = self.get(key_path, default=None)
        LOGGER.info("Confirmed %s = %s", ".".join(key_path), value)
    
    def update_key_course_ids(self, ids: list[str]) -> None:
        self.update_field(["preprocessing", "features", "key_course_ids"], ids)
        self.confirm_field(["preprocessing", "features", "key_course_ids"])

    def update_key_course_subject_areas(self, cips: list[str]) -> None:
        self.update_field(["preprocessing", "features", "key_course_subject_areas"], cips)
        self.confirm_field(["preprocessing", "features", "key_course_subject_areas"])


def update_run_metadata(config_path: str, run_id: str, experiment_id: str) -> None:
    editor = TomlConfigEditor(config_path)
    editor.update_field(["model", "run_id"], run_id)
    editor.update_field(["model", "experiment_id"], experiment_id)
    editor.save()
    editor.confirm_field(["model", "run_id"])
    editor.confirm_field(["model", "experiment_id"])


def update_key_courses_and_cips(
    config_path: str,
    key_course_ids: list[str],
    key_course_subject_areas: list[str],
) -> None:
    """
    Update the TOML config with key course IDs and cip codes under [preprocessing.features].
    """
    editor = TomlConfigEditor(config_path)
    editor.update_key_course_ids(key_course_ids)
    editor.update_key_course_subject_areas(key_course_subject_areas)
    editor.save()