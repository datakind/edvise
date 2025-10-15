## 0.1.5 (2025-10-15)

- Saving logging files for data audit, ckpt, and model prep from pipeline in run folders under silver vol
- Added model calibration in h2o as a toggle to improve recall when models underpredict
- Dropping nfolds down to 3 if we have large amount of data to train for h2o
- Edited the credits_earned target to checkpoint arg and moved the checkpoint step ahead of targets

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
