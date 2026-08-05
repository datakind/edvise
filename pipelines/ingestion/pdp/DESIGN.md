# Design: Convert PDP ingestion notebooks into a DAB

**Goal:** Automate `PIPELINE_pdp_to_databricks.py` (SFTP → filter by institution → bronze volume) as a Databricks Asset Bundle job that can be run on demand or scheduled — same pattern as other Edvise DABs under `pipelines/`.

**Status:** Design draft. Implementation can reuse / evolve work from `feature/nsc-sftp-scripts-and-dab` and `notebooks/nsc_sftp_automated_data_ingestion/`.

---

## Current state

### Interactive notebook (`PIPELINE_pdp_to_databricks.py`)

One-shot, human-driven flow:

1. Load `gcp_config.yaml` (secret scope, institution ID map, catalog prefixes).
2. Connect to SFTP `./receive`.
3. **Widgets:** pick a remote file + pick a PDP institution.
4. Download (atomic/resumable), normalize columns, filter rows for that institution.
5. Print cohort / cohort_term checks.
6. Write filtered CSV to `/Volumes/{catalog}/{inst}_bronze/bronze_volume/`, skip if file already exists.

### Existing automation direction

`notebooks/nsc_sftp_automated_data_ingestion/` and branch `feature/nsc-sftp-scripts-and-dab` already split this into:

| Stage | Responsibility | State |
|-------|----------------|--------|
| 01 scan + stage | SFTP list/download, upsert `ingestion_manifest`, queue staged paths | Delta + UC `tmp` volume |
| 02 expand | Distinct PDP institution IDs per staged file → `institution_ingest_plan` | Delta |
| 03 bronze ingest | SST API resolve PDP ID → bronze schema/volume; filter + write; update manifest | Bronze volumes |

DAB sketch lives at `pipelines/ingestion/pdp` on that branch (`nsc_sftp_automated_ingestion` job, git-sourced `spark_python_task` chain). Notebooks are also acceptable as DAB `notebook_task`s — conversion to scripts is optional.

**Gap for scheduling:** stage 01 still requires explicit `cohort_file_name` / `course_file_name` job params. School registry still conceptually tied to `gcp_config` in the old notebook; automated path prefers API + bronze discovery.

---

## Proposed pipeline shape

Do **not** wrap the widget notebook as a single scheduled task. Keep the 3-stage DAG:

```text
sftp_receive_scan → file_institution_expand → per_institution_bronze_ingest
```

Shared state:

- `ingestion_manifest` — file fingerprint, status (`NEW` / `BRONZE_WRITTEN` / `FAILED`), errors, run id
- `pending_ingest_queue` — staged local UC volume path (downstream does not re-hit SFTP)
- `institution_ingest_plan` — `(file × institution_id)` work items

`max_concurrent_runs: 1` so scheduled runs do not race the same SFTP files / queue rows.

---

## Design decisions

### 1. File selection (filename unknown until SFTP list)

NSC cohort/course files share a 14-digit stamp: `..._YYYYMMDDHHMMSS.csv`.

| Mode | Job params | Behavior |
|------|------------|----------|
| **Manual** | `cohort_file_name` + `course_file_name` | Current behavior; fail if missing on SFTP |
| **Latest pair** | empty / `mode=latest` | List `./receive`, pair by stamp, take newest stamp |
| **Uningested** (default for schedule) | empty / `mode=uningested` | Same pairing; skip fingerprints already `BRONZE_WRITTEN` (or queued); take newest remaining |

Rules:

- Pair cohort + course by identical stamp; log / fail on partial pairs.
- Idempotency key = `file_fingerprint` in `ingestion_manifest` (not “path exists in bronze”).
- Log: available stamps, chosen stamp, skip reasons.
- Optional guards: `min_file_date`, `max_age_days`, `dry_run=true` (list only).

### 2. New schools / `gcp_config.yaml`

Old notebook uses `gcp_config` for:

1. SFTP secret scope/keys  
2. Allowed PDP institution IDs (dropdown)  
3. Institution → `{prefix}_bronze` catalog mapping  

Automated path should **not** require a yaml edit per new school:

| Concern | Approach |
|---------|----------|
| Secrets | Fixed secret scope (e.g. `nsc-sftp-asset`); not per-institution |
| Which schools | IDs discovered from file; resolve via SST `GET /institutions/pdp-id/{pdp_id}` |
| Where to write | `databricksify_inst_name` + find `{inst}_bronze` schema / bronze volume |

**Onboarding checklist (outside the ingest job):**

1. Institution exists in SST/API with correct PDP ID  
2. UC schema `{inst}_bronze` + bronze volume exist  
3. SFTP secrets already configured in the shared scope  

Optional allowlist (Delta table or job param JSON) if we must not auto-ingest every ID in the dump. Prefer: ingest IDs that resolve **and** have bronze provisioned; log/skip `UNKNOWN_OR_UNPROVISIONED`.

Treat workspace-local `gcp_config.yaml` school lists as tech debt for this DAB.

### 3. School checks / logging (replace widget validation)

Structured logs (and optional summary table) in stages 02/03:

**After expand (02):**

- Distinct PDP IDs per file + row counts  
- API resolve success vs failure  
- Resolve OK but missing bronze schema/volume  
- Intersection with optional allowlist  

**Before/after write (03):**

- Per `(file, institution)`: filtered row count, latest cohort / cohort_term (parity with old notebook prints), destination path, skip-vs-wrote  
- End rollup: `ingested`, `skipped_already_present`, `unresolved`, `no_bronze`  
- Fail job (or fail-soft + Slack) on policies such as `unresolved > 0` or `ingested == 0` when files were `NEW`

### 4. Operating modes

1. **Scheduled** — `mode=uningested`; Slack on failure / zero-ingest  
2. **Manual** — pass filenames or stamp for one-off / reprocess  
3. **Force reprocess** — `force=true` even if manifest says done (define overwrite policy)  
4. **Optional filter** — `institution_ids=...` for targeted runs without dropdown UX  

### 5. Bronze overwrite policy (open)

Old notebook: skip if destination file exists. Scheduled runs may need:

- versioned paths under bronze (e.g. stamp subdirectory), or  
- overwrite with audit fields on the manifest  

Decide explicitly before enabling cron.

---

## Implementation priorities (vs existing branch)

1. Auto file discovery (`latest` / `uningested`) when filename params are empty  
2. School validation report (API + bronze provisioning)  
3. Overwrite / versioning policy for bronze writes  
4. Decouple school registry from `gcp_config` (secrets only)  
5. Bundle under `pipelines/ingestion/pdp` aligned with other DABs (`git_commit` / `git_tag`, permissions, webhooks)  
6. Notebook vs script tasks — either is fine; prefer shared helpers in `src/edvise/ingestion/`

---

## Out of scope (for later)

- Chaining this DAB into PDP training/inference automatically  
- Replacing GCS validated → bronze sync (`pipelines/ingestion/shared`)  
- GenAI mapping onboarding  

---

## References

- Behavior source: `PIPELINE_pdp_to_databricks.py` (interactive)  
- Automation notebooks: `notebooks/nsc_sftp_automated_data_ingestion/`  
- Prior WIP: `origin/feature/nsc-sftp-scripts-and-dab`  
- Helpers on `develop`: `src/edvise/ingestion/nsc_sftp_helpers.py`, `src/edvise/utils/sftp.py`, `src/edvise/ingestion/constants.py`  
- Sibling DAB pattern: `pipelines/ingestion/shared/`
