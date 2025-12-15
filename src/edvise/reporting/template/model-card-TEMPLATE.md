
{logo}

## Model Card: {institution_name}

### Contents {{#contents}}
- [<span class="toc-label">Overview</span>](#overview)
- [<span class="toc-label">Methodology</span>](#methodology)
- [<span class="toc-label">Performance</span>](#performance)
- [<span class="toc-label">Quantitative Bias Analysis</span>](#bias)
- [<span class="toc-label">Important Features</span>](#features)
- [<span class="toc-label">Appendix</span>](#appendix)
- [<span class="toc-label">Glossary</span>](#glossary)

### Overview {{#overview}}
- {outcome_section}
- {checkpoint_section}
- {development_note_section}
- Key technical and evaluation terms are defined in the [Glossary](#glossary) section.
- If there are questions or concerns about the model, you can contact **education@datakind.org** or your customer success manager.

### Intended Use {{#intended-use}}

- #### Primary Purpose
    - Identify students who may need support to retain or graduate on time. 
    - Empower academic advisors to provide strategic interventions based on the factors contributing to students' need for support.

- #### Out-of-Scope Uses
    - Outside of the [target population](#target-population)
    - Without intervention strategies carefully designed by academic advisors, student success professionals, and researchers. 

### Methodology {{#methodology}}

- #### Sample Development
    - Our first step was our data audit & validation, which included handling null and duplicate values, checking for inconsistencies between files, and ensuring all student IDs are unique.
    - After validation, we then proceeded with exploratory data analysis (EDA) to develop a deeper understanding of the raw dataset prior to our [feature engineering](#glossary-feature-engineering) & model development, ensuring alignment with stakeholders through an iterative process.

- #### Feature Development
    - We then proceeded with feature engineering, which involved transforming raw data into meaningful representations by applying semantic abstractions, aggregating at varying levels of term, course, or section analysis, and comparing values cumulatively over time.
    - Stakeholder collaboration was also essential to our feature engineering effort, ensuring domain and use-case knowledge shaped the development of insightful features.

- #### Feature Selection
    - Collinearity Threshold
        - Threshold Applied: Removed features with VIF greater than {collinearity_threshold} were removed to reduce multicollinearity and improve model stability.
        - Explanation: [Variance Inflation Factor (VIF)](#glossary-vif) measures multicollinearity between features.
    - Low Variance Threshold
        - Threshold Applied: Removed features with variance less than {low_variance_threshold}.
        - Explanation: Features with very low variance do not vary much across observations, meaning they carry little predictive signal.
    - Missing Data Threshold
        - Threshold Applied: Removed features with {incomplete_threshold}% or more missing values.
        - Explanation: Features with a high percentage of missing values may introduce noise or require extensive imputation.
    - After our feature selection process, **{number_of_features} actionable features** were retained for modeling.

- #### Target Population {{#target-population}}
{target_population_section}
    - This resulted in a dataset of **{training_dataset_size} students** within the target timeframe.

- #### Model Development
{sample_weight_section}
{data_split_table}

- #### Model Evaluation
    - Evaluated top models across standard classification metrics such as [Accuracy](#glossary-accuracy), [AUC](#glossary-auc), [F1-Score](#glossary-f1), [Log Loss](#glossary-log-loss), [Precision](#glossary-precision), [Recall](#glossary-recall).
    - Evaluated [SHAP](#glossary-shap) values to assess the relative importance of key features.
    - Evaluated initial model output for interpretability and actionability.
    - Prioritized model quality with transparent and interpretable model outputs.

{model_comparison_plot}

### Performance {{#performance}}
- #### Model Performance Metric
{primary_metric_section}

- #### Model Performance Plots
{test_confusion_matrix}
{test_calibration_curve}
{test_roc_curve}
{test_histogram}

### Quantitative Bias Analysis {{#bias}}

#### Model Bias Metric
- Our bias evaluation metric is [_False Negative Rate (FNR)_](#glossary-fnr).
- We assess [FNR Parity](#glossary-fnr-parity) to determine whether underprediction occurs at disproportionate rates across student subgroups.

- #### Analyzing Bias Across Student Groups
{bias_groups_section}
    - We evaluated [FNR](#glossary-fnr) across these student groups and tested for statistically significant disparities.

{bias_summary_section}

### Important Features {{#features}}
- #### Analyzing Feature Importance
    - This figure shows how individual features contribute to the model’s predictions for each student-term record using [SHAP](#glossary-shap) values.
        - Guidelines to interpret the plot:
            - Each dot represents a single student-term record.
            - Features are ordered by overall importance, with the most influential at the top.
            - SHAP values (x-axis) indicate whether a feature increases (+) or decreases (–) the predicted likelihood of needing support.
            - **Color** reflects the feature’s value for that student:
                - <span class="dk-red">High</span> values in red
                - <span class="dk-blue">Low</span> values in blue
                - <span class="dk-gray">Categorical features</span> in gray
        - Example: _Students with a lower percentage of grades above the section’s average tend to have SHAP values further to the right, indicating that this feature contributes to the model predicting a higher likelihood of needing support._

{feature_importances_by_shap_plot}

### Appendix {{#appendix}}

{performance_by_splits_section}

{selected_features_ranked_by_shap}

{evaluation_by_group_section}

### Glossary {{#glossary}}

_This section defines technical, statistical, and modeling terms used throughout this model card._

#### Evaluation & Performance Metrics

***Accuracy*** <a id="glossary-accuracy"></a>  
Shows the share of all predictions the model gets right. Typical values range from 0.6–0.8, but accuracy alone can be misleading when groups are imbalanced.

***AUC (Area Under the ROC Curve)*** <a id="glossary-auc"></a>  
Shows how well the model separates students who need support from those who do not. Typical values range from 0.6–0.8; higher is better.

***Calibration Curve*** <a id="glossary-calibration-curve"></a>  
A plot comparing predicted probabilities to observed outcomes.

***Confusion Matrix*** <a id="glossary-confusion-matrix"></a>  
A table summarizing model predictions versus actual outcomes.

***F1 Score*** <a id="glossary-f1"></a>  
A metric that balances precision and recall, particularly useful when classes are imbalanced. Typical values range from 0.6–0.8; higher is better.

***Log Loss*** <a id="glossary-log-loss"></a>  
Shows how accurate the model’s probability estimates are. Lower values mean the model assigns more reliable likelihoods. Typical values for this model fall between 0.4–0.6; lower is better.

***Precision*** <a id="glossary-precision"></a>  
The proportion of students predicted to need support who actually do need support. Typical values range from 0.6–0.8; higher is better.

***Recall*** <a id="glossary-recall"></a>  
Shows how many students who truly need support the model successfully identifies. Typical values range from 0.6–0.8; higher is better.

***ROC Curve (Receiver Operating Characteristic Curve)*** <a id="glossary-roc"></a>  
A plot showing the tradeoff between true positive and false positive rates.

***Threshold*** <a id="glossary-threshold"></a>  
The probability cutoff used to convert model scores into binary predictions.

---

#### Fairness & Bias

***Bias (Model Bias)*** <a id="glossary-bias"></a>  
Systematic differences in model performance across student subgroups.

***False Negative Rate (FNR)*** <a id="glossary-fnr"></a>  
The proportion of students who need support but are predicted as not needing support.

***FNR Parity*** <a id="glossary-fnr-parity"></a>  
A measure assessing whether false negative rates are similar across student groups.

***Subgroup*** <a id="glossary-subgroup"></a>  
A defined subset of students used to evaluate model performance and fairness.

---

#### Features & Modeling

***Actionable Feature*** <a id="glossary-actionable-feature"></a>  
A model input representing outcomes that can plausibly be influenced through intervention.

***Collinearity (Multicollinearity)*** <a id="glossary-collinearity"></a>  
A condition where two or more features contain highly overlapping information.

***Feature Engineering*** <a id="glossary-feature-engineering"></a>  
The process of transforming raw data into meaningful variables.

***Feature Importance*** <a id="glossary-feature-importance"></a>  
A measure of how much each feature contributes to the model’s predictions.

***Feature Selection*** <a id="glossary-feature-selection"></a>  
The process of retaining a subset of features that provide the strongest predictive signal.

***H2O AutoML*** <a id="glossary-h2o-automl"></a>  
An automated machine learning framework that trains, tunes, and evaluates multiple model types (such as generalized linear models, gradient boosting machines, random forests, and stacked ensembles) to identify high-performing models based on specified evaluation metrics.

***Imputation*** <a id="glossary-imputation"></a>  
The process of filling in missing data values.

***Low Variance Feature*** <a id="glossary-low-variance"></a>  
A feature that changes very little across students.

***Sample Weighting*** <a id="glossary-sample-weighting"></a>  
A technique that assigns different importance to observations during model training.

***Variance Inflation Factor (VIF)*** <a id="glossary-vif"></a>  
A statistic used to quantify multicollinearity between features.

---

#### Interpretability

***SHAP (Shapley Additive Explanations)*** <a id="glossary-shap"></a>  
A method used to explain model predictions by quantifying how much each feature contributes to a prediction. SHAP values indicate both the **direction** (whether a feature increases or decreases the predicted likelihood of needing support) and the **magnitude** of that contribution. When aggregated across students, SHAP values provide insight into which features are most influential overall.

---

#### Model & Data Concepts

***Checkpoint*** <a id="glossary-checkpoint"></a>  
A specific point in time at which a prediction is generated for a student.

***Target Population*** <a id="glossary-target-population"></a>  
The group of students for whom the model is designed and validated.

***Training Dataset*** <a id="glossary-training-dataset"></a>  
The subset of data used to fit the model.
