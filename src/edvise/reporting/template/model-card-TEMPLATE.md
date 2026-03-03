
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

Before we train any model, we take several careful steps to make sure your data is accurate, consistent, and set up in a way that helps the model learn.

{funnel_image}

- #### Sample Development

    - _Goal: Create a reliable starting dataset_
    - We begin with a comprehensive data audit and validation process to ensure the dataset is complete, consistent, and trustworthy.
    - This includes:
        - Reviewing and addressing [null values](#glossary-null-values)
        - Removing [duplicate records](#glossary-duplicate-records)
        - Confirming that each student–term (or student–course) record is unique
        - Checking for inconsistencies across files
    - After validation, we conduct exploratory data analysis (EDA) to better understand patterns in the data,  such as enrollment trends, grade distributions, and outcome rates.  
    - If unusual patterns appear, we pause and review them with stakeholders before moving forward. This iterative process ensures alignment prior to [feature engineering](#glossary-feature-engineering) and model development.


- #### Feature Development

    - _Goal: Transform raw data into meaningful student signals_
    - Next, we apply [feature engineering](#glossary-feature-engineering) techniques to convert raw institutional data into variables that reflect real student behavior and progress.
    - Examples include:
        - Aggregating academic performance by term, year, or course level  
        - Converting counts into interpretable rates (e.g., percent of courses passed)  
        - Calculating cumulative metrics (e.g., total credits earned)  
        - Measuring trends over time (e.g., GPA changes across terms)
    - Throughout this process, we collaborate closely with institutional stakeholders to ensure features reflect domain knowledge, policy context, and intervention strategy needs.

- #### Feature Selection

    - _Goal: Retain the most informative and stable predictors_
    - Not all features contribute meaningful predictive value. To improve model stability and interpretability, we apply the following selection criteria:
        - Collinearity Threshold
            - Removed features with [Variance Inflation Factor (VIF)](#glossary-vif) greater than {collinearity_threshold}
            - This reduces [multicollinearity](#glossary-collinearity), where features contain overlapping information.
        - Low Variance Threshold
            - Removed features with very low variance (below the [low variance threshold](#glossary-low-variance-threshold) of {low_variance_threshold})
            - Features that vary very little across students provide limited predictive signal.
        - Missing Data Threshold
            - Removed features with {incomplete_threshold}% or more missing values
            - Features with excessive missingness may introduce noise into our modeling dataset.

    - After completing the [feature selection](#glossary-feature-selection) process, **{number_of_features} actionable features** were retained for modeling.

- #### Target Population {{#target-population}}
    - _Goal: Define who the model is designed to support_
{target_population_section}
    - This resulted in a dataset of **{training_dataset_size} students** within the target timeframe.

- #### Final Modeling Dataset
    - At this stage, we have a clean, streamlined dataset that:
        - Is consistent and validated
        - Reflects meaningful student behavior patterns
        - Includes only the strongest and most reliable predictors  

    - This dataset serves as the foundation for model training.

- #### Model Training
{sample_weight_section}
{classification_threshold_section}
{data_split_table}

- #### Model Evaluation
    - Evaluated top models across standard classification metrics such as [Accuracy](#glossary-accuracy), [AUC](#glossary-auc), [F1-Score](#glossary-f1), [Log Loss](#glossary-log-loss), [Precision](#glossary-precision), [Recall](#glossary-recall).
    - Evaluated [SHAP](#glossary-shap) values to assess the relative importance of key features.
    - Evaluated initial model output for interpretability and actionability.
    - Prioritized model quality with transparent and interpretable model outputs.

- #### Model Selection
    - From the evaluated candidates, we selected the final model using a standard multi-metric approach designed to balance predictive performance, generalization to new data, and fairness.
    - Candidate models were compared based on strong performance across [Recall](#glossary-recall), [AUC](#glossary-auc), [Log Loss](#glossary-log-loss), and [F1 Score](#glossary-f1).
    - Final selection also incorporated fairness considerations, prioritizing models with lower disparities in [False Negative Rate (FNR)](#glossary-fnr) across student subgroups.
    - This approach ensures the selected model is accurate, stable, and equitable in identifying students who may need support.

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

- #### Model Bias Metric
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

---

#### Education & Domain Terms

***Enrollment Intensity*** <a id="glossary-enrollment-intensity"></a><br>
A classification indicating whether a student is enrolled full-time or part-time during a term.

***Peak COVID Term*** <a id="glossary-peak-covid"></a><br>
Academic terms that occurred during the height of the COVID-19 pandemic, when instructional formats and student outcomes may have differed from typical conditions.

***Pell Grant Recipient*** <a id="glossary-pell"></a><br>
A student who receives a U.S. federal Pell Grant, typically awarded to students with demonstrated financial need.

***Program of Study Area*** <a id="glossary-program-of-study"></a><br>  
A student’s primary academic discipline or field of study.

***Term-over-Term Comparison*** <a id="glossary-term-over-term"></a><br>
A comparison of metrics across consecutive academic terms (e.g., fall to spring) to examine changes over time.

#### Evaluation & Performance Metrics

***Accuracy*** <a id="glossary-accuracy"></a><br>
Shows the share of all predictions the model gets right. Typical values range from 0.6–0.8, but accuracy alone can be misleading when groups are imbalanced.

***AUC (Area Under the ROC Curve)*** <a id="glossary-auc"></a><br>  
Shows how well the model separates students who need support from those who do not. Typical values range from 0.6–0.8; higher is better.

***Calibration Curve*** <a id="glossary-calibration-curve"></a><br>
A calibration curve shows Support Scores on the x-axis ("Mean Predicted Probability") compared to the proportion of students who receive that support score that are truly 'In Need of Support' ("Fraction of Positives"). The closer the blue line is to the dotted line, the better the model is calibrated.

***Classification Threshold*** <a id="glossary-classification-threshold"></a><br>
The probability cutoff used to convert model scores into binary predictions.

***Confidence Interval (CI)*** <a id="glossary-confidence-interval"></a><br>
A range of values used to express uncertainty around a metric estimate. For example, a 95% confidence interval indicates the range within which the true value is likely to fall.

***Confusion Matrix*** <a id="glossary-confusion-matrix"></a><br>
A table summarizing model predictions versus actual outcomes.

***F1 Score*** <a id="glossary-f1"></a><br>
A metric that balances precision and recall, particularly useful when classes are imbalanced. Typical values range from 0.6–0.8; higher is better.

***Log Loss*** <a id="glossary-log-loss"></a><br>
Shows how accurate the model's probability estimates are. Lower values mean the model assigns more reliable likelihoods. Typical values for this range between 0.4–0.6; lower is better.

***Precision*** <a id="glossary-precision"></a><br>
The proportion of students predicted to need support who actually do need support. Typical values range from 0.6–0.8; higher is better.

***Recall*** <a id="glossary-recall"></a><br>
Shows how many students who truly need support the model successfully identifies. Typical values range from 0.6–0.8; higher is better.

***ROC Curve (Receiver Operating Characteristic Curve)*** <a id="glossary-roc"></a><br>
A plot showing the tradeoff between true positive and false positive rates.

***Statistical Significance*** <a id="glossary-statistical-significance"></a><br>
An assessment of whether observed differences in results are likely due to real effects rather than random variation.

---

#### Fairness & Bias

***Bias (Model Bias)*** <a id="glossary-bias"></a><br>
Systematic differences in model performance across student subgroups.

***False Negative Rate (FNR)*** <a id="glossary-fnr"></a><br>
The proportion of students who need support but are predicted as not needing support.

***FNR Parity*** <a id="glossary-fnr-parity"></a><br>
A measure assessing whether false negative rates are similar across student groups.

***Subgroup*** <a id="glossary-subgroup"></a><br>
A defined subset of students used to evaluate model performance and fairness.

---

#### Features & Modeling

***Actionable Feature*** <a id="glossary-actionable-feature"></a><br>
A model input representing outcomes that can plausibly be influenced through intervention.

***AutoML (Automated Machine Learning)*** <a id="glossary-automl"></a><br>
Software that automatically trains, tunes, and compares multiple machine learning models to identify high-performing configurations based on predefined metrics.

***Collinearity (Multicollinearity)*** <a id="glossary-collinearity"></a><br>
A condition where two or more features contain highly overlapping information.

***Feature Engineering*** <a id="glossary-feature-engineering"></a><br>
The process of transforming raw data into meaningful variables.

***Feature Importance*** <a id="glossary-feature-importance"></a><br>
A measure of how much each feature contributes to the model's predictions. Feature importance plots visually display which features most strongly influence predictions overall.

***Feature Selection*** <a id="glossary-feature-selection"></a><br>
The process of retaining a subset of features that provide the strongest predictive signal.

***H2O AutoML*** <a id="glossary-h2o-automl"></a><br>
An automated machine learning framework that trains, tunes, and evaluates multiple model types (such as generalized linear models, gradient boosting machines, random forests, and stacked ensembles) to identify high-performing models based on specified evaluation metrics.

***Imputation*** <a id="glossary-imputation"></a><br>
The process of filling in missing data values.

***Low Variance Feature*** <a id="glossary-low-variance"></a><br>
A feature that changes very little across students.

***Model Interpretability*** <a id="glossary-model-interpretability"></a><br>
The degree to which a person can understand how and why a model makes its predictions.

***Model Pipeline (MLOps)*** <a id="glossary-model-pipeline"></a><br>
The structured process and supporting infrastructure used to prepare data, train models, evaluate performance, and deploy predictions in a consistent and reproducible way.

***Sample Weighting*** <a id="glossary-sample-weighting"></a><br>
A technique that assigns different importance to observations during model training.

***SHAP (Shapley Additive Explanations)*** <a id="glossary-shap"></a><br>
A method used to explain model predictions by quantifying how much each feature contributes to a prediction. SHAP values indicate both the **direction** (whether a feature increases or decreases the predicted likelihood of needing support) and the **magnitude** of that contribution. When aggregated across students, SHAP values provide insight into which features are most influential overall.

***Variance Inflation Factor (VIF)*** <a id="glossary-vif"></a><br>
A statistic used to quantify multicollinearity between features.

---

#### Model & Data Concepts

***Checkpoint*** <a id="glossary-checkpoint"></a><br>
A specific point in time at which a prediction is generated for a student.

***Duplicate Records*** <a id="glossary-duplicate-records"></a><br>
Repeated entries representing the same observation. Duplicate records can distort model training and evaluation if not identified and handled appropriately.

***Low Variance Threshold*** <a id="glossary-low-variance-threshold"></a><br>
A rule used to remove features that vary very little across students and therefore contribute limited predictive value.

***Missing Data Threshold*** <a id="glossary-missing-data-threshold"></a><br>
A predefined cutoff used to determine when a feature contains too many missing values to be considered reliable for modeling.

***Null Values*** <a id="glossary-null-values"></a><br>
Empty or missing data entries in a dataset. These may occur when information was not collected, not recorded, or not applicable.

***Target Population*** <a id="glossary-target-population"></a><br>
The group of students for whom the model is designed and validated.

***Training Dataset*** <a id="glossary-training-dataset"></a><br>
The subset of data used to fit the model.

***Variance*** <a id="glossary-variance"></a><br>
A statistical measure of how spread out values are from their average. Higher variance indicates greater variability in a feature.
