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
