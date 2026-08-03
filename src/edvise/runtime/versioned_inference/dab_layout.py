"""Map ``--schema_type`` to archived DAB bundle paths and inference job keys."""

from __future__ import annotations

from dataclasses import dataclass

_BUNDLE_SNAPSHOT_ROOT = "databricks_bundle_snapshot"
_DAB_YML_SNAPSHOT = f"{_BUNDLE_SNAPSHOT_ROOT}/databricks.yml"


def normalize_schema_type(raw: str) -> str:
    """Mirror :func:`edvise.configs.schema_type.normalize_schema_type` without heavy imports."""
    return raw.strip().lower()
_DAB_YML_SNAPSHOT = f"{_BUNDLE_SNAPSHOT_ROOT}/databricks.yml"


@dataclass(frozen=True)
class DabBundleLayout:
    """GitHub repo paths and UC snapshot layout for a schema's inference bundle."""

    schema_type: str
    pipeline_dir: str
    inference_job_key: str
    inference_yml_filename: str
    inference_schema_type: str

    @property
    def dab_repo_path(self) -> str:
        return f"pipelines/{self.pipeline_dir}/databricks.yml"

    @property
    def inference_yml_repo_path(self) -> str:
        return f"pipelines/{self.pipeline_dir}/resources/{self.inference_yml_filename}"

    @property
    def inference_yml_snapshot_rel(self) -> str:
        return f"{_BUNDLE_SNAPSHOT_ROOT}/resources/{self.inference_yml_filename}"

    def github_snapshot_sources(self) -> dict[str, str]:
        """Repo path → relative path under ``release_dir`` for materialize."""
        return {
            self.dab_repo_path: _DAB_YML_SNAPSHOT,
            self.inference_yml_repo_path: self.inference_yml_snapshot_rel,
        }


_LAYOUTS: dict[str, DabBundleLayout] = {
    "pdp": DabBundleLayout(
        schema_type="pdp",
        pipeline_dir="pdp",
        inference_job_key="edvise_github_sourced_pdp_inference_pipeline",
        inference_yml_filename="github_pdp_inference.yml",
        inference_schema_type="pdp",
    ),
    "edvise": DabBundleLayout(
        schema_type="edvise",
        pipeline_dir="es",
        inference_job_key="github_sourced_genai_es_inference_pipeline",
        inference_yml_filename="github_es_inference.yml",
        inference_schema_type="edvise",
    ),
    "legacy": DabBundleLayout(
        schema_type="legacy",
        pipeline_dir="legacy",
        inference_job_key="edvise_github_sourced_legacy_inference_pipeline",
        inference_yml_filename="github_legacy_inference.yml",
        inference_schema_type="legacy",
    ),
}


def resolve_dab_bundle_layout(schema_type: str) -> DabBundleLayout:
    """
    Resolve DAB paths for ``--schema_type`` (``pdp``, ``edvise``/``es``, ``legacy``).

    ``edvise`` and ``es`` share the ES pipeline bundle under ``pipelines/es/``.
    """
    key = normalize_schema_type(schema_type)
    if key == "es":
        key = "edvise"
    try:
        return _LAYOUTS[key]
    except KeyError as exc:
        msg = (
            f"Unknown schema_type {schema_type!r}; "
            "expected 'pdp', 'edvise', 'es', or 'legacy'."
        )
        raise ValueError(msg) from exc


def default_dab_bundle_layout() -> DabBundleLayout:
    """Default layout for PDP (current production launcher consumer)."""
    return _LAYOUTS["pdp"]
