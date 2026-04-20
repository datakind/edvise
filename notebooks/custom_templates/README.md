## SST custom templates

- Define the _run_type_ parameter in a Databricks workflow as `"train"` or `"predict"` for notebooks that branch on that flag.
- One `config.toml` maps to one model. Multiple models need multiple configs (same pattern as PDP).

### Suggested notebook order

| Step | Template | Role |
|------|----------|------|
| 00 | `00-data-audit-TEMPLATE.py` | Table-only institutional audit (keys, credits, merges). |
| 01 | `01-data-assessment-TEMPLATE.py` | Charts / deeper EDA on bronze. |
| 02 | `02-preprocess-data-TEMPLATE.py` | Silver build + schema contract via `custom_cleaning`. |
| 03–06 | training, registration, predictions, validation | Downstream modeling. |

Numbers on files (e.g. `06` before `03`) reflect historical naming; follow the pipeline role in the table above.

### `edvise.data_audit` imports

- **`eda`** — exploratory only: logging helpers, `analyze_merge`, `value_counts_*`, summaries. Does not own column inference or cleaning transforms.
- **`custom_data_audit`** — structured audit: `find_dupes`, `validate_credit_consistency`, pass/fail vs grade checks, cohort credit checks, etc. (no automatic column guessing—set names in the notebook or config).
- **`custom_cleaning`** — transforms: `order_terms`, `normalize_student_id_column`, `clean_bronze_datasets`, schema contract helpers, etc.

`eda` re-exports selected `custom_data_audit` / `custom_cleaning` names for older notebooks; new code should import from the submodule directly (as in templates 00 and 01).
