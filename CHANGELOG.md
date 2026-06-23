## 1.2.3 (2026-06-23)
- fix: target functionality logic for schools with 4 terms (#137)
- fix(genai): validate term hook groups at HITL emit time (#166)
- feat(genai): semantic grain profiling to improve grain inference reliability & duplicate transparency (#169)
- fix(es): ensure edvise raw features have feature table mapping (#172)

## 1.2.2 (2026-06-18)
- fix(legacy): setting up staging SA in preprocessing & postprocessing scripts (#168)
- fix: systemic optional-column guards for ES feature generation (#170)

## 1.2.1 (2026-06-16)
- fix: calendar year codes inferred as datetime during cleaning (#165)

## 1.2.0 (2026-06-15)
- feat: adding training & inference jobs for legacy schools for webapp compatibility (#96)
- refactor: simplify and refine SMA transformation HITL  (#155)
- fix: load ES project config in training_h2o when schema_type is edvise (#156)
- fix: wire optional UC dataio converters into ES data audit (#157)
- fix: genai mapping staging deploy git source (#158)
- fix: allowing SMA to have less than 3 options for HITL from prompt guidance (#161)
- refactor: refining genai config and including template (#162)
- fix: resolve package-relative features table path in training validation (#164)

## 1.1.0 (2026-05-29)
- feat: genai mapping hitl dashboard, production scripts/DABs, prompt improvements & grain resolution  (#150)
- feat: config toggle for GenAI parquets vs bronze raw CSV in data_audit (#154)

## 1.0.0 (2026-05-27)
- feat: genai edvise core mapping logic; identity agent, schema mapping agent, and hitl defined, implemented, and tested (#128)
- feat: copy validated/ GCS objects to institution bronze (Databricks job) (#145)
- feat: add edvise schema training pipeline and refactor pdp pipeline to be more generalizable (#147)
- docs: remove local community health files to inherit from org-wide .github (#149)
- feat: es inference pipeline .yml setup (#152)
- feat: create edvise inference tasks (#153)

## 0.2.1 (2026-05-07)
- fix: expanding feature name for hybrid courses feature (#131)
- fix: metadata dashboard deployment (#132)
- fix: classify gateway course level by first digit, not 200 threshold (#133)
- fix: model naming convention (#135)
- fix: bump weasyprint and pydyf due to datadog callout (#136)
- fix: EdaSummary: normalize Pell recipient counts (Y/YES vs No) in `pell_recipient_status` (#138)
- feat: change shape of EdaSummary's enrollment_type_by_intensity to in… (#139)

## 0.2.0 (2026-04-02)
- feat: custom data assessment template (#87)
- feat: add Pandera schemas for ES raw cohort and course (#110)
- feat: enrich metadata tables (#118)
- fix: CVE-2024-52338 bump pyarrow to >=17.0.0 in deps and PDP training pipeline (#123)
- feat: add missing flag for no numeric grades submitted, remove unused checkpoints, use term program of study instead of cohort file program of study year 1/term 1 data, add more loggers to audit (#124)
- feat: change deduping functionality (#125)
- fix: databricks directory issues (#126)
- feat: add program of study to model card glossary (#127)
- feat: metadata dashboard (#129)

## 0.1.12 (2026-03-06)
- feat: adding run_id subfolder in output path (#78)
- fix: select inference terms for students meeting checkpoint in desired terms instead of cohorts (#105)
- feat: added automated ingestion workflow (#113)
- fix: tighten EDA pell status null handling (#114)
- feat: custom classification threshold functionality for training, inference, and model cards  (#115)
- feat: model card glossary & methodology section updates (#117)
- fix: EDA course enrollments by academic year/term, simplify tests, skip unknown values (#119)
- fix: move term filter into inference prep script (#121)
- fix: include UNKNOWN degree types (#122)

## 0.1.11 (2026-02-20)
- fix: modifying target functions to include an additional eligible term during training  (#98)
- feat: create EDASummary class (#99)
- feat: Add term_filter job param to PDP inference pipeline (#100)
- fix: configure parameters in deploy.yml to dev target and avoid webhook error (#101)
- feat: adding enrollment intensity percents to data audit (#102)
- feat: standardizing PDP model names for better readability in webapp (#103)
- feat: allowing for more flexibility across custom cleaning functions for edvise schema & current custom schools (#104)
- feat: add cohorts selected for training into the config (#106)
- feat: improving handle dupes func (#107)
- feat: improving validate credit consistency function (#108)
- fix: lower case model name in configs for UC consistency (#109)
- fix: eda term and course data (#111)
- fix: PDP model names in integration test parameters (#112)

## 0.1.10 (2026-02-02)
- feat: adding slack notifications for training/inference pipelines failures on staging & dev (#51)
- feat: automate releases (#84)
- feat: adding bias variable check to data audit (#86)
- feat: adding custom school functions to repo (#88)
- feat: Add `get_institution_id_by_name` helper function to retrieve institution ID by name (#90)
- fix: preserve pandas nullable dtypes pre-modeling; fix boolean classification post-modeling (#92)
- fix: remove framework field and dataio with sklearn models; standardize on h2o model loading (#93)
- refactor: refining ModelCard classes (#94)

## 0.1.9 (2025-01-20)
- fix: import error in inference script from validation module 
- fix: fetching latest commit from develop for health check 
- feat: adding PDP pipeline high-level validation and sanity checks 
- feat: update pipeline version during training
- fix: email indents 
- fix: cleanup schedule 
- refactor: remove sklearn modeling 
- feat: model card revision 

## 0.1.8 (2025-12-11)
- fix: adding exclude_frameworks into training script 
- fix: dtype overrides in custom cleaning module
- fix: indent issue in the case of no duplicated cols 
- feat: added printing enrollment type to data audit EDA 

## 0.1.7 (2025-12-02)
- fix: support distribution bug 
- feat: adding 3 functions for custom school data audits 
- feat: additional data loggers and enhancements, reducing gateway course limit from 25 to 10 
- fix: rewording webapp emails
- feat: added DFWI and consistency check functions 
- feat: custom cleaning module
- feat: adding action-semantic-pull-request into style.yml
- feat: setting up integration CICD actions 

## 0.1.6 (2025-10-28)

- Added saving of log files in catalog throughout training & inference pipelines.
- Log files are now separated by job folder for easier tracking.
- Refactored log saving code so it's less redundant.
- Gateway course automation has an extra safety check to see if any upper-level courses were mistakenly referred as gateway.
- Custom processing code was added in appropriate modules for our custom school refactoring effort.

## 0.1.5 (2025-10-15)

- Saving logging files for data audit, ckpt, and model prep from pipeline in run folders under silver vol
- Added model calibration in h2o as a toggle to improve recall when models underpredict
- Dropping nfolds down to 3 if we have a very large amount of data to train for h2o
- Edited the credits_earned target to checkpoint arg and moved the checkpoint step to be ahead of targets

## 0.1.4 (2025-09-29)

- Update file paths to store all run files into a folder named with the model number

## 0.1.3 (2025-09-27)

- Added sklearn inference to pipeline
- Added reading model type from config
- Added version number and inference cohort to configs

## 0.1.2 (2025-09-19)

- Enhanced EDA with logging 
- Added overfit score

## 0.1.1 (2025-09-15)

- Update pyproject

## 0.1.0 (2025-09-15)

- Initial release of edvise repo 
- Includes PDP code broken into components
- Includes scripts for running PDP from ingestion through inference 
- Includes Databricks DAB workflows set up for running the scripts for PDP
- Includes h2o code for training models and running inference 
