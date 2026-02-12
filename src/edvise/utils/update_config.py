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

    def merge_list_field(self, key_path: list[str], new_values: list[str]) -> None:
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


def update_save_confirm_fields(
    config_path: str,
    updates: dict[tuple[str, ...], Any],
    extra_save_paths: list[str] | None = None,
) -> None:
    """
    Update multiple fields in a TOML config, save, and confirm all updates.
    
    Args:
        config_path: Path to the TOML config file
        updates: Dictionary mapping key paths (as tuples) to new values
                 e.g., {("model", "run_id"): "123", ("model", "experiment_id"): "exp1"}
        extra_save_paths: Optional list of additional paths to save the config to
    """
    editor = TomlConfigEditor(config_path)
    
    # Update all fields
    for key_path, new_value in updates.items():
        editor.update_field(list(key_path), new_value)
    
    # Save once after all updates
    editor.save()
    
    # Save to any additional paths
    if extra_save_paths:
        for path in extra_save_paths:
            editor.save(output_path=path)
    
    # Confirm all fields
    for key_path in updates.keys():
        editor.confirm_field(list(key_path))


def update_run_metadata(
    config_path: str,
    run_id: str,
    experiment_id: str,
    extra_save_paths: list[str] | None = None,
) -> None:
    """Update run metadata (run_id and experiment_id) in the config."""
    update_save_confirm_fields(
        config_path=config_path,
        updates={
            ("model", "run_id"): run_id,
            ("model", "experiment_id"): experiment_id,
        },
        extra_save_paths=extra_save_paths,
    )


def update_pipeline_version(
    config_path: str,
    pipeline_version: Any,
    extra_save_paths: list[str] | None = None,
) -> None:
    """
    Update the TOML config at the end of the training pipeline such that we can save the
    version as metadata from training. This is critical so that we know schools' pipeline version
    without having to manually set it in each config which does not scale well.
    """
    update_save_confirm_fields(
        config_path=config_path,
        updates={
            ("pipeline_version",): pipeline_version,
        },
        extra_save_paths=extra_save_paths,
    )


def update_training_cohorts(
    config_path: str,
    training_cohorts: list[str],
    extra_save_paths: list[str] | None = None,
) -> None:
    """Update training cohorts in the config."""
    update_save_confirm_fields(
        config_path=config_path,
        updates={
            ("modeling", "training", "cohort"): training_cohorts,
        },
        extra_save_paths=extra_save_paths,
    )

def update_key_courses_and_cips(
    config_path: str,
    key_course_ids: list[str],
    key_course_subject_areas: list[str],
    extra_save_paths: list[str] | None = None,
) -> None:
    """
    Update the TOML config with key course IDs and CIP codes under [preprocessing.features],
    merging with existing values to avoid duplicates.
    """
    editor = TomlConfigEditor(config_path)
    editor.merge_list_field(
        key_path=["preprocessing", "features", "key_course_ids"],
        new_values=key_course_ids
    )
    editor.merge_list_field(
        key_path=["preprocessing", "features", "key_course_subject_areas"],
        new_values=key_course_subject_areas
    )
    editor.save()
    
    if extra_save_paths:
        for path in extra_save_paths:
            editor.save(output_path=path)
