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

    # def save(self) -> None:
    #     self.path.write_text(dumps(self._doc))
    #     LOGGER.info("Saved updated TOML to %s", self.path)

    def save(self, output_path: str | None = None) -> None:
        path_to_write = self.path if output_path is None else pathlib.Path(output_path)
        path_to_write.write_text(dumps(self._doc))
        LOGGER.info("Saved updated TOML to %s", path_to_write)

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

    def _merge_list_field(self, key_path: list[str], new_values: list[str]) -> None:
        """
        Merge new list values into an existing list at the given key path, avoiding duplicates.
        If no values need to be added, no update is performed.
        """
        current_values = self.get(key_path, default=[])
        if not isinstance(current_values, list):
            current_values = []
        merged_values = list(
            dict.fromkeys(current_values + new_values)
        )  # preserves order, avoids duplicates

        if set(merged_values) != set(current_values):
            self.update_field(key_path, merged_values)
            self.confirm_field(key_path)
        else:
            LOGGER.info(
                "No update needed for %s; values already present", ".".join(key_path)
            )

    def update_key_course_ids(self, ids: list[str]) -> None:
        self._merge_list_field(
            key_path=["preprocessing", "features", "key_course_ids"], new_values=ids
        )

    def update_key_course_subject_areas(self, cips: list[str]) -> None:
        self._merge_list_field(
            key_path=["preprocessing", "features", "key_course_subject_areas"],
            new_values=cips,
        )


def update_run_metadata(
    config_path: str,
    run_id: str,
    experiment_id: str,
    extra_save_paths: list[str] | None = None,
) -> None:
    editor = TomlConfigEditor(config_path)
    editor.update_field(["model", "run_id"], run_id)
    editor.update_field(["model", "experiment_id"], experiment_id)
    editor.save()
    # Save to any additional paths provided, e.g. the model run folder
    if extra_save_paths:
        for path in extra_save_paths:
            editor.save(output_path=path)
    editor.confirm_field(["model", "run_id"])
    editor.confirm_field(["model", "experiment_id"])


def update_key_courses_and_cips(
    config_path: str,
    key_course_ids: list[str],
    key_course_subject_areas: list[str],
) -> None:
    """
    Update the TOML config with key course IDs and cip codes under [preprocessing.features],
    and optionally save the updated config to one or more additional paths.
    """
    editor = TomlConfigEditor(config_path)
    editor.update_key_course_ids(key_course_ids)
    editor.update_key_course_subject_areas(key_course_subject_areas)

    # Save to the original config path
    editor.save()


def update_pipeline_version(
    config_path: str,
    pipeline_version: str,
    extra_save_paths: list[str] | None = None,
) -> None:
    """
    Update the TOML config at the end of the training pipeline such that we can save the
    version as metadata from training. This is critical so that we know schools' pipeline version
    without having to manually set it in each config which does not scale well.
    """
    editor = TomlConfigEditor(config_path)
    editor.update_field(["pipeline_version"], pipeline_version)
    editor.save()

    if extra_save_paths:
        for path in extra_save_paths:
            editor.save(output_path=path)

    editor.confirm_field(["pipeline_version"])


def update_training_cohorts(
    config_path: str,
    training_cohorts: list[str],
    extra_save_paths: list[str] | None = None,
) -> None:
    editor = TomlConfigEditor(config_path)
    editor._merge_list_field(key_path=["modeling", "training", "cohort"], new_values=training_cohorts)
    editor.save()
    # Save to any additional paths provided, e.g. the model run folder
    if extra_save_paths:
        for path in extra_save_paths:
            editor.save(output_path=path)
    editor.confirm_field(key_path=["modeling", "training", "cohort"])
