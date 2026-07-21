# Edvise

Edvise is a school-agnostic Python library for implementing Student Success
Tool workflows. It provides shared components for preparing institutional
data, training and running predictive models, and producing reports for
student-success interventions.

Edvise is designed primarily for workflows orchestrated on Databricks. Using
the complete pipelines requires institution-specific configuration, data, and
cloud infrastructure.

## What Edvise provides

- Data ingestion from Google Cloud Storage into Databricks and Unity Catalog
- Schema validation and data-quality auditing
- Feature generation, student selection, and outcome target creation
- H2O model training, calibration, evaluation, registration, and inference
- Model monitoring and drift detection
- MLflow-backed model cards and reporting
- GenAI-assisted identity and schema mapping with human review
- Databricks Asset Bundles for deploying data, modeling, and reporting jobs

## Requirements

- Python 3.10, 3.11, or 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management

Running production pipelines may also require:

- A Databricks workspace with Unity Catalog
- Databricks CLI authentication and appropriate workspace permissions
- Google Cloud credentials and access to the required GCS buckets

## Local setup

Clone the repository and install the locked development environment:

```bash
git clone https://github.com/datakind/edvise.git
cd edvise
uv sync --frozen --dev
```

Run the test suite:

```bash
uv run python -m pytest
```

Run the local quality checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m mypy src
```

Some tests and runtime paths depend on external services and may require
additional credentials or configuration.

## Using Edvise

Edvise is organized around configurable workflows rather than a single
command-line interface. A typical implementation:

1. Maps institution data to the expected schema.
2. Selects and adapts a configuration template from [`configs/`](configs/).
3. Configures a Databricks Asset Bundle from [`pipelines/`](pipelines/).
4. Deploys and runs the relevant ingestion, training, inference, and reporting
   jobs in Databricks.

Useful starting points include:

- [PDP H2O configuration templates](configs/pdp_h2o/)
- [Legacy H2O configuration template](configs/legacy_h2o/config-TEMPLATE.toml)
- [GenAI mapping configuration](configs/genai_mapping/inputs-TEMPLATE.toml)
- [H2O training entry point](src/edvise/scripts/training_h2o.py)
- [H2O inference entry point](src/edvise/scripts/inference_h2o.py)

Configuration and infrastructure vary by institution. Treat the templates as
starting points rather than complete deployment configurations.

## Repository structure

```text
.
├── configs/       # Workflow and institution configuration templates
├── notebooks/     # Legacy workflow notebook templates
├── pipelines/     # Databricks Asset Bundles
├── src/edvise/    # Python package and job entry points
└── tests/         # Automated test suite
```

The package is divided into modules for ingestion, data auditing, feature
generation, modeling, reporting, student selection, synthetic data, and GenAI
mapping.

## Contributing

Before opening a pull request:

1. Add or update tests for behavioral changes.
2. Run the test suite, Ruff checks, and mypy.
3. Update [`CHANGELOG.md`](CHANGELOG.md) when the change is relevant to a
   release.

Use [GitHub issues](https://github.com/datakind/edvise/issues) to report bugs
or propose features.

## License

Edvise is available under the [Apache License 2.0](LICENSE).