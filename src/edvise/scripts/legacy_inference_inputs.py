import argparse
import json
import logging
import os
import tempfile
import sys

# Ensure repo src/ is on sys.path so `import edvise.*` works in Databricks Jobs.
script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

from edvise.utils.databricks import (
    local_fs_path,
    get_dbutils_or_none,
    get_latest_uc_model_run_id,
    find_file_in_run_folder as find_file_in_run_folder_shared,
)
from edvise.utils.gcs import (
    get_storage_client,
    download_gcs_uri_to_dir,
    resolve_input_source_to_local_path,
)


class LegacyInferenceInputs:
    """
    Retrieves inference inputs by looking up the trained model and finding paths
    to config and features_table that were used during training. This ensures
    inference uses the exact same configuration as the trained model.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dbutils = get_dbutils_or_none()
        self.storage_client = get_storage_client()

    def get_latest_model_run_id(
        self,
        model_name: str,
        workspace: str,
        institution: str,
        registry_uri: str = "databricks-uc",
    ) -> str:
        """
        Returns the run ID of the latest version of a model registered in Unity Catalog.

        Args:
            model_name: Short name of the model (without UC path).
            workspace: Unity Catalog workspace (e.g. 'edvise').
            institution: Institution name used in the UC model path.
            registry_uri: Registry URI; defaults to 'databricks-uc'.

        Returns:
            run_id: The run ID associated with the latest version of the model.
        """
        latest_version = get_latest_uc_model_run_id(
            model_name=model_name,
            workspace=workspace,
            institution=institution,
            registry_uri=registry_uri,
        )
        logging.info(
            "Found latest run_id for model '%s.%s_gold.%s': %s",
            workspace,
            institution,
            model_name,
            latest_version,
        )
        return latest_version

    def _download_from_gcs(self, gcs_uri: str, dest_dir: str) -> str:
        os.makedirs(local_fs_path(dest_dir), exist_ok=True)
        dest_path = download_gcs_uri_to_dir(
            gcs_uri,
            local_fs_path(dest_dir),
            storage_client=self.storage_client,
        )
        logging.info("Downloaded %s to %s", gcs_uri, dest_path)
        return dest_path

    def _download_from_default_bucket(self, object_name: str, dest_dir: str) -> str:
        return resolve_input_source_to_local_path(
            object_name,
            local_fs_path(dest_dir),
            gcp_bucket_name=self.args.gcp_bucket_name,
            gcp_blob_prefix=self.args.gcp_blob_prefix,
            storage_client=self.storage_client,
        )

    def _resolve_bronze_input_path(self, value: str, dest_dir: str) -> str:
        resolved = resolve_input_source_to_local_path(
            value,
            local_fs_path(dest_dir),
            gcp_bucket_name=self.args.gcp_bucket_name,
            gcp_blob_prefix=self.args.gcp_blob_prefix,
            storage_client=self.storage_client,
        )
        return local_fs_path(resolved)

    def _load_json_mapping_from_path(self, raw_path: str) -> dict[str, str]:
        if raw_path.startswith("gs://"):
            tmp_dir = tempfile.mkdtemp(prefix="edvise_legacy_manifest_")
            path_to_read = self._download_from_gcs(raw_path, tmp_dir)
        else:
            path_to_read = local_fs_path(raw_path)

        with open(path_to_read, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Bronze overrides manifest must be a JSON object.")
        return {str(k): str(v) for k, v in payload.items()}

    def _bronze_file_overrides(self) -> dict[str, str]:
        from_inline = (self.args.bronze_file_overrides_json or "").strip()
        from_manifest = (self.args.bronze_file_overrides_manifest_path or "").strip()
        if from_inline and from_manifest:
            raise ValueError(
                "Provide only one of --bronze_file_overrides_json or "
                "--bronze_file_overrides_manifest_path."
            )

        if from_inline:
            parsed = json.loads(from_inline)
            if not isinstance(parsed, dict):
                raise ValueError("--bronze_file_overrides_json must be a JSON object.")
            return {str(k): str(v) for k, v in parsed.items()}

        if from_manifest:
            return self._load_json_mapping_from_path(from_manifest)

        return {}

    def _materialize_config_with_overrides(
        self,
        config_file_path: str,
        bronze_overrides: dict[str, str],
        landing_dir: str,
    ) -> str:
        if not bronze_overrides:
            return config_file_path
        try:
            import tomlkit  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "tomlkit is required to materialize legacy config overrides."
            ) from exc

        cfg_local_path = local_fs_path(config_file_path)
        with open(cfg_local_path, encoding="utf-8") as f:
            cfg_text = f.read()
        cfg_doc = tomlkit.parse(cfg_text)

        datasets = cfg_doc.get("datasets")
        if datasets is None or "bronze" not in datasets:
            raise ValueError(
                f"Config missing datasets.bronze section: {config_file_path}"
            )
        bronze = datasets["bronze"]

        for dataset_key, source in bronze_overrides.items():
            if dataset_key not in bronze:
                raise ValueError(
                    f"Unknown bronze dataset key {dataset_key!r} in overrides. "
                    "Expected keys present in datasets.bronze."
                )
            resolved = self._resolve_bronze_input_path(source, landing_dir)
            bronze[dataset_key]["file_path"] = resolved
            logging.info(
                "Configured datasets.bronze.%s.file_path=%s", dataset_key, resolved
            )

        fd, tmp_path = tempfile.mkstemp(prefix="edvise_legacy_inf_cfg_", suffix=".toml")
        try:
            os.write(fd, tomlkit.dumps(cfg_doc).encode("utf-8"))
        finally:
            os.close(fd)
        logging.info("Using overridden legacy config at: %s", tmp_path)
        return tmp_path

    def find_file_in_run_folder(
        self,
        run_root: str,
        file_name: str | None,
        description: str,
        keyword: str | None = None,
    ) -> str:
        """
        Finds a file with the given name under the run root.
        Searches in: run_root, run_root/training/, run_root/inference/
        
        If exact match fails, falls back to searching for .toml files containing
        the specified keyword in the filename (e.g., "config" or "features_table").

        Args:
            run_root: Root directory of the training run
            file_name: Optional exact filename to search for (e.g. "config.toml").
                If None or not found, falls back to searching for files containing keyword.
            description: Human-readable description for error messages
            keyword: Keyword to search for in filename when doing fallback search
                (e.g., "config" or "features_table"). If None, searches for any .toml file.

        Returns:
            Path to the matching file

        Raises:
            FileNotFoundError: If no matching file is found
        """
        try:
            found_path = find_file_in_run_folder_shared(
                run_root,
                file_name=file_name,
                keyword=keyword,
            )
            logging.info("Found %s at: %s", description, found_path)
            return found_path
        except FileNotFoundError as exc:
            error_msg = (
                f"No {description} found under: {run_root} "
                f"(searched run root, training/, inference/)"
            )
            if file_name:
                error_msg += f" with name '{file_name}'"
            if keyword:
                error_msg += f" or containing '{keyword}' in filename"
            raise FileNotFoundError(error_msg) from exc

    def run(self):
        """
        Main execution:
        1. Look up the model and get its run_id
        2. Find the config file from the training run
        3. Find the features_table file from the training run
        4. Return both paths via dbutils.jobs.taskValues (if available)
        """
        # Get the model's run_id
        model_run_id = self.get_latest_model_run_id(
            self.args.model_name,
            self.args.DB_workspace,
            self.args.databricks_institution_name,
        )

        # Construct the run root path in silver volume
        silver_run_root = (
            f"/Volumes/{self.args.DB_workspace}/"
            f"{self.args.databricks_institution_name}_silver/silver_volume/{model_run_id}"
        )
        logging.info("Looking for training artifacts in: %s", silver_run_root)

        # Find config file (try exact name first, fallback to files containing "config")
        config_file_name = getattr(self.args, "config_file_name", None)
        config_file_path = self.find_file_in_run_folder(
            silver_run_root,
            config_file_name,
            "config file",
            keyword="config",
        )

        bronze_root = (
            f"/Volumes/{self.args.DB_workspace}/"
            f"{self.args.databricks_institution_name}_bronze/bronze_volume"
        )
        landing_dir = os.path.join(
            bronze_root,
            self.args.bronze_landing_subdir,
            self.args.db_run_id,
        )
        os.makedirs(local_fs_path(landing_dir), exist_ok=True)
        bronze_overrides = self._bronze_file_overrides()
        config_file_path = self._materialize_config_with_overrides(
            config_file_path,
            bronze_overrides,
            landing_dir,
        )

        # Find features_table file (try exact name first, fallback to files containing "features_table")
        features_table_name = getattr(self.args, "features_table_name", None)
        features_table_path = self.find_file_in_run_folder(
            silver_run_root,
            features_table_name,
            "features table",
            keyword="features_table",
        )

        logging.info("Using config file: %s", config_file_path)
        logging.info("Using features table: %s", features_table_path)

        # Set task values for downstream tasks (if dbutils available)
        if self.dbutils:
            self.dbutils.jobs.taskValues.set(
                key="config_file_path", value=str(config_file_path)
            )
            self.dbutils.jobs.taskValues.set(
                key="features_table_path", value=str(features_table_path)
            )
            logging.info("Task values set for downstream tasks")
        else:
            logging.warning(
                "dbutils not available - task values not set. "
                "This is expected in local/testing environments."
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve inference inputs (config and features table paths) from trained model."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the registered model (without UC path prefix)",
    )
    parser.add_argument(
        "--DB_workspace", required=True, help="Databricks workspace name"
    )
    parser.add_argument(
        "--databricks_institution_name", required=True, help="Institution identifier"
    )
    parser.add_argument("--db_run_id", required=True, help="Databricks job run id")
    parser.add_argument(
        "--gcp_bucket_name",
        required=False,
        default="",
        help=(
            "Default GCS bucket used when bronze overrides are object names instead of gs:// URIs."
        ),
    )
    parser.add_argument(
        "--gcp_blob_prefix",
        required=False,
        default="validated",
        help="Prefix prepended to object-name bronze overrides.",
    )
    parser.add_argument(
        "--config_file_name",
        required=False,
        default=None,
        help="Optional name of the config file to search for (e.g., 'config.toml'). "
             "If provided, searches for exact match first, then falls back to any .toml file "
             "excluding features_table.toml.",
    )
    parser.add_argument(
        "--features_table_name",
        required=False,
        default=None,
        help="Optional name of the features table file to search for (e.g., 'features_table.toml'). "
             "If provided, searches for exact match first, then falls back to any .toml file.",
    )
    parser.add_argument(
        "--bronze_file_overrides_json",
        required=False,
        default="",
        help=(
            "JSON object mapping datasets.bronze keys to input sources. "
            "Values can be absolute paths, dbfs:/ paths, gs:// URIs, or object names "
            "(resolved using gcp_bucket_name + gcp_blob_prefix)."
        ),
    )
    parser.add_argument(
        "--bronze_file_overrides_manifest_path",
        required=False,
        default="",
        help=(
            "Path (local/dbfs/gs://) to JSON object mapping datasets.bronze keys to input sources. "
            "Use this for batch-style payloads."
        ),
    )
    parser.add_argument(
        "--bronze_landing_subdir",
        required=False,
        default="inference_inputs",
        help="Subdirectory under bronze volume where downloaded legacy inputs are staged.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()

    task = LegacyInferenceInputs(args)
    task.run()
