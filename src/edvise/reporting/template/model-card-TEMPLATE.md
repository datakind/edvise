
{logo}

## Model Card: {institution_name}

### Contents {{#contents}}
- [Overview](#overview)
- [Intended Use](#intended-use)
- [Methodology](#methodology)
- [Performance](#performance)
- [Quantitative Bias Analysis](#bias)
- [Important Features](#features)
- [Appendix](#appendix)
- [Glossary](#glossary)

### Overview {{#overview}}
- {outcome_section}
- {checkpoint_section}
- {development_note_section}
- Key technical and evaluation terms are defined in the **[Glossary](#glossary)**.
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
    - After validation, we then proceeded with exploratory data analysis (EDA) to develop a deeper understanding of the raw dataset prior to our feature engineering & model development, ensuring alignment with stakeholders through an iterative process.
- #### Feature Development
    - We then proceeded with feature engineering, which involved transforming raw data into meaningful representations by applying semantic abstractions, aggregating at varying levels of term, course, or section analysis, and comparing values cumulatively over time.
    - Stakeholder collaboration was also essential to our feature engineering effort, ensuring domain and use-case knowledge shaped the development of insightful features.
    - Then, our next step was feature selection, applying the following processing below.
        - **Collinearity Threshold**
            - Threshold Applied: Removed features with VIF greater than {collinearity_threshold} were removed to reduce multicollinearity and improve model stability.
            - Explanation: Variance Inflation Factor (VIF) measures multicollinearity between features (see **[Glossary](#glossary)**).
        - **Low Variance Threshold**
            - Threshold Applied: Removed features with variance less than {low_variance_threshold}.
            - Explanation: Features with very low variance do not vary much across observations, meaning they carry little predictive signal.
        - **Missing Data Threshold**
            - Threshold Applied: Removed features with {incomplete_threshold}% or more missing values.
            - Explanation: Features with a high percentage of missing values may introduce noise or require extensive imputation.
    - After our feature selection process, **{number_of_features} actionable features** were retained for modeling.
- #### Target Population {{#target-population}}
{target_population_section}
    - This resulted in a training dataset of **{training_dataset_size} students** within the target timeframe.
- #### Model Development
{sample_weight_section}
{data_split_table}

- #### Model Evaluation
    - Evaluated top 10 models for performance across key metrics: accuracy, precision, AUC, recall, log loss, F-1.
    - Evaluated SHAP values indicating relative importance in the models of key features for top-performing models.
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
- #### Model Bias Metric
    - Our bias evaluation metric is _False Negative Rate (FNR)_ (see **[Glossary](#glossary)**).
    - We assess **FNR Parity** to determine whether underprediction occurs at disproportionate rates across student subgroups.

- #### Analyzing Bias Across Student Groups
{bias_groups_section}
    - We evaluated FNR across these student groups and tested for statistically significant disparities.

{bias_summary_section}

### Important Features {{#features}}
- #### Analyzing Feature Importance
    - This figure shows how individual features contribute to the model’s predictions for each student-term record using SHAP values (see **[Glossary](#glossary)**).
        - Guidelines to interpret the plot:
        - Each dot represents a single student-term record.
        - Features are ordered by overall importance, with the most influential at the top.
        - **SHAP values (x-axis)** indicate whether a feature increases (+) or decreases (–) the predicted likelihood of needing support.
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

**Accuracy**  
The proportion of all predictions that the model classifies correctly, including both students who need support and those who do not.

**AUC (Area Under the ROC Curve)**  
A metric measuring the model’s ability to distinguish between students who need support and those who do not.

**Calibration Curve**  
A plot comparing predicted probabilities to observed outcomes.

**Confusion Matrix**  
A table summarizing model predictions versus actual outcomes.

**F1 Score**  
A metric that balances precision and recall, particularly useful when classes are imbalanced.

**Log Loss**  
A metric that penalizes confident but incorrect probability predictions.

**Precision**  
The proportion of students predicted to need support who actually do need support.

**Recall**  
The proportion of students who truly need support that the model successfully identifies.

**ROC Curve (Receiver Operating Characteristic Curve)**  
A plot showing the tradeoff between true positive and false positive rates.

**Threshold**  
The probability cutoff used to convert model scores into binary predictions (e.g., “needs support” vs. “does not need support”).

---

#### Fairness & Bias

**Bias (Model Bias)**  
Systematic differences in model performance across student subgroups.

**False Negative Rate (FNR)**  
The proportion of students who need support but are predicted as not needing support.

**FNR Parity**  
A measure assessing whether false negative rates are similar across student groups.

**Subgroup**  
A defined subset of students (e.g., by demographic or academic characteristic) used to evaluate model performance and fairness.

---

#### Features & Modeling

**Actionable Feature**  
A model input designed to reflect behaviors or outcomes that can plausibly be influenced through academic advising or institutional interventions.

**Collinearity (Multicollinearity)**  
A condition where two or more features contain highly overlapping information.

**Feature Engineering**  
The process of transforming raw data into meaningful variables by aggregating, normalizing, or deriving new representations.

**Feature Importance**  
A measure of how much each feature contributes to the model’s predictions, assessed using SHAP values.

**Feature Selection**  
The process of identifying and retaining a subset of features that provide the most predictive signal while improving model stability and interpretability.

**Imputation**  
The process of filling in missing data values using statistical or model-based methods.

**Low Variance Feature**  
A feature that changes very little across students and is typically removed during feature selection.

**Sample Weighting**  
A technique that assigns different importance to observations during model training.

**Variance Inflation Factor (VIF)**  
A statistic used to quantify multicollinearity by measuring how strongly a feature is correlated with other features.

---

#### Interpretability

**SHAP (Shapley Additive Explanations)**  
A game-theoretic method used to explain model predictions by quantifying how much each feature contributes to a prediction. SHAP values indicate both the **direction** (whether a feature increases or decreases the predicted likelihood of needing support) and the **magnitude** of that contribution. When aggregated across students, SHAP values provide insight into which features are most influential overall.


---

#### Model & Data Concepts

**Checkpoint**  
A specific point in time (e.g., after a term or credit threshold) at which a prediction is generated for a student.

**Target Population**  
The specific group of students for whom the model is designed and validated. Predictions outside this population are considered out of scope.

**Training Dataset**  
The subset of data used to fit the model, consisting only of students and records that meet the target population and checkpoint criteria.
