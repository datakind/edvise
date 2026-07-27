# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`edvise` is a school-agnostic Python library implementing Student Success Tool (SST) workflows for
Datakind: ingesting institution data, auditing/standardizing it, generating features, training/running
H2O AutoML models, and producing model card reports. It runs on Databricks — most orchestration lives
in Databricks Asset Bundle (DAB) job YAMLs, not in a Python `main()`.

## Commands

Dependency management is via `uv` (see `uv.lock`); dev tools (`pytest`, `ruff`, `mypy`) are in the `dev`
dependency group and installed via `uv sync --frozen --dev`.

```bash
# Run the full test suite
uv run python -m pytest

# Run a single test file / test
uv run python -m pytest tests/genai/mapping/schema_mapping_agent/hitl/test_apply_manifest_mapping_override.py
uv run python -m pytest tests/targets/test_graduation.py::test_specific_case -v

# Lint (CI runs this only against changed files, with --diff)
uv tool run ruff check <files>
uv tool run ruff format --diff <files>

# Type-check (CI runs this only against changed src/**/*.py files)
uv run python -m mypy --install-types --non-interactive <files>
```

CI runs `test.yml` across Python 3.10/3.11/3.12 on every PR; `style.yml` and `type-check.yml` only lint/type-check
files changed in the PR diff, so running ruff/mypy on the full `src/` tree locally may surface pre-existing issues
outside your diff — don't feel obligated to fix unrelated ones.

PR titles are enforced as Conventional Commits (`feat|fix|chore|docs|refactor|test|ci|perf|build|revert|style`) by
the `semantic-pr` check in `style.yml`.

## Architecture

There is no single Python orchestrator — pipeline order is defined by Databricks Asset Bundle job graphs in
`pipelines/*/resources/*.yml`, which chain scripts from `src/edvise/scripts/` as dependent tasks
(`task_key`/`depends_on`). Reading those YAMLs is the fastest way to see real call order. The canonical PDP
training DAG is:

```
data_audit → feature_generation → {checkpoints, student_selection} → targets → model_prep → training_h2o
```

and inference swaps the tail for `inf_prep → inference_h2o → inference_output_publish`. This maps directly onto
`src/edvise` packages: `ingestion` → `data_audit` → `feature_generation` → `checkpoints`/`student_selection` →
`targets` → `model_prep` → `modeling.h2o_ml` → `reporting`.

`pipelines/` has one DAB bundle per pipeline family, each with its own `databricks.yml` and `dev`/`prod` targets
pointing at specific workspaces (`dev_sst_02` / `staging_sst_01`):
- `pdp/` — the standardized PDP (Postsecondary Data Partnership / NSC) schema pipeline.
- `es/` — "Edvise Schema", the standardized path for non-PDP institutions.
- `legacy/` — older, fully custom per-institution pipelines predating the ES schema (one config maps to one
  model; see `notebooks/legacy_templates/README.md` for the manual notebook execution order:
  `00-data-assessment → 01-preprocess-data → 02-train-h2o-model → 03-make-h2o-predictions →
  04-register-h2o-model-create-card → 05-inference-validation`).
- `genai_mapping/` — the GenAI onboarding pipeline (below); runs upstream of `pdp`/`es`, producing the
  standardized silver data those pipelines consume.
- `ingestion/shared/` — shared ingestion bundle.

DAB jobs pull `src/edvise` from a **git ref** at run time (`--var git_commit=...` for dev, `--var git_tag=...`
for prod deploys — see the `variables:` block in each `databricks.yml`), not a built wheel. The two Streamlit
apps (`streamlit-genai-hitl-app/`, `shared/dashboard_metadata/streamlit-data-app/`) are the exception: CI builds
an `edvise` wheel and pins it into the app's `requirements.txt` before deploying. To exercise a bundle manually:

```bash
databricks bundle deploy --target dev --var "git_commit=$(git rev-parse HEAD)" --var "databricks_institution_name=<inst>" ...
databricks bundle run <job_name> --target dev --params config_file_name=config.toml,job_type=training,...
```

`.github/workflows/release-integration.yml` shows real end-to-end examples of the vars/params each bundle needs
(institution name, schema_type, service account, run id, etc).

Root-level `configs/` holds per-institution TOML templates (`configs/pdp_h2o/`, `configs/legacy_h2o/`,
`configs/genai_mapping/`) that get deployed per institution and passed to scripts via `--config_file_path`.
These are distinct from `src/edvise/configs/*.py`, the Pydantic schema classes (`PDPProjectConfig`,
`LegacyProjectConfig`, etc.) that parse and validate them; `src/edvise/configs/schema_type.py` is the single
dispatch point mapping a `--schema_type` (`pdp`/`edvise`/`legacy`) flag to the right config class. That flag is
the main axis pipeline scripts and `reporting/sections/{pdp,es,custom,legacy}` branch on.

### GenAI mapping (`src/edvise/genai/mapping/`)

Every institution ships data in a different raw schema. Instead of hand-written per-institution ingestion
mappings, this subsystem uses two LLM-driven agents plus mandatory human-in-the-loop (HITL) review to infer
and validate them before anything is trusted downstream:

- **IdentityAgent** (`identity_agent/`, entry point `scripts/edvise_genai_ia.py`, `--mode onboard|execute`) —
  profiles raw files, infers grain/primary key, normalizes term/semester conventions, classifies column roles,
  and can generate custom-transform "hooks".
- **SchemaMappingAgent / SMA** (`schema_mapping_agent/`, entry point `scripts/edvise_genai_sma.py`, staged as
  `gate_1`/`gate_2`) — produces the field-mapping **manifest** (source column → target schema field), resolves
  ambiguous grains, and builds/executes transformations, validating output against the same Pandera base schema
  used by `data_audit/schemas`.
- Both agents call LLMs (Claude via Databricks) through `shared/databricks_ai_gateway.py`, which uses an
  OpenAI-compatible client against the MLflow AI Gateway, authenticates via `databricks-sdk`'s workspace bearer
  token (not a PAT), and deliberately disables MLflow's OpenAI autologging/tracing since these are gateway
  utility calls, not model-training runs.
- **HITL gates**: any ambiguous LLM decision is written as a JSON gate file; `check_sma_hitl_gate`
  (`schema_mapping_agent/manifest/hitl/resolver.py`) blocks the pipeline until a reviewer supplies a `choice`.
  `resolve_sma_items` applies those choices back into the manifest. `manifest/hitl/override.py` is a
  post-gate, batch/CLI path (`--overrides_json_path` on `edvise_genai_sma.py`) for corrections after the
  interactive gate has already passed, built on `apply_manifest_mapping_override`, which preserves original
  `confidence`/`rationale` unless explicitly overridden and appends an audit `MappingOverrideEvent`.
- **streamlit-genai-hitl-app/** is the actual reviewer UI (a Databricks App), reading/writing the UC table
  `<catalog>.genai_mapping.hitl_reviews`: `1_HITL_Review.py` (approve/reject/edit pending items),
  `2_Maps_and_Outputs.py` (view manifests/outputs), `3_Manifest_Explorer.py` (browse mappings + HITL status).
- `state/` tracks onboarding job/pipeline state in a `{catalog}.genai_mapping` UC schema so long-running onboard
  jobs can pause at a gate and resume later.

### Other conventions worth knowing

- **Pandera schemas** (`data_audit/schemas/`): a base org-wide schema plus per-institution extension schemas;
  `validation.py` distinguishes hard errors (missing required / extra columns, failed checks) from soft errors
  (missing optional columns). The GenAI SMA execution path reuses this same validation on LLM-produced output.
- **Model cards** (`reporting/model_card/`, see `reporting/README.md`): built from a `config.toml` + MLflow run
  artifacts into a shared Markdown template (`reporting/template/`), editable locally, then rendered to
  HTML/PDF. `sections/` holds pluggable, independently-testable report sections.
- `modeling/automl/` contains only stale `__pycache__` — it's dead code from a prior Databricks-native AutoML
  approach, fully superseded by `modeling/h2o_ml/`. Don't treat it as live.

## Release process

Git-flow, driven end-to-end by chained GitHub Actions — there's no manual tagging:

1. **Start Release** (`start-release.yml`, manual `workflow_dispatch` with a `version` input) branches
   `release/<version>` off `develop`, runs `python -m edvise.scripts.update_version` to bump `pyproject.toml`
   and generate `CHANGELOG.md` entries from merged PRs since the last tag, pushes the branch, then dispatches
   `release-branch-ci-dev`.
2. **release-branch-ci-dev** (`release-integration.yml`, on push to `release/*`) classifies the semver bump
   (`major`/`minor`/`patch`/`initial`, via `edvise.utils.automate_releases`). Minor/major/initial bumps get a
   full smoke test: deploy the `pdp` and `es` bundles to `dev_sst_02` and run train+inference against synthetic
   institutions (`synthetic_integration`, `synthetic_integration_es`); patch bumps skip this. On success it
   dispatches **Finish Release**.
3. **Finish Release** (`finish-release.yml`) opens the `release/<version> -> main` PR (blocked by `pre-release.yml`
   unless `CHANGELOG.md` was touched). Merging that PR triggers the same workflow's tag job: tags `v<version>`,
   dispatches `release-deploy.yml` for that tag, and opens a `main -> develop` back-merge PR. Merging the
   back-merge PR deletes the `release/<version>` branch.
4. **release-deploy.yml** (on the `v*` tag push) fans out in parallel: core DAB bundles (`deploy.yml`, gated to
   staging_sst_01 only on a `v*` tag ref), the GenAI mapping bundle, the GenAI HITL Streamlit app, and the
   metadata dashboard app — each redeployed to both `dev` and `staging`.

Two scheduled jobs guard branch health outside the release flow, independent of the above: `weekly-develop-integration.yml`
(Mondays) runs the same PDP/ES train+infer smoke test directly against `develop`'s HEAD, and `weekly-cleanup.yml`
(Fridays) deletes synthetic-integration models/experiments/runs older than 30 days from dev.
