
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

### Overview {{#overview}}
- {outcome_section}
- {checkpoint_section}
- {development_note_section}
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
        - Collinearity Threshold
            - Threshold Applied: Removed features with VIF greater than {collinearity_threshold} were removed to reduce multicollinearity and improve model stability.
            - Explanation: Variance Inflation Factor (VIF) measures how much a feature is linearly correlated with other features. A VIF of 1 would imply no multicollinearity, while a VIF of 10 indicates high collinearity, meaning the feature's information is largely redundant.
        - Low Variance Threshold
            - Threshold Applied: Removed features with variance less than {low_variance_threshold}.
            - Explanation: Features with very low variance do not vary much across observations, meaning they carry little predictive signal. For example, features with variance below 0.01 are often considered near-constant.
        - Missing Data Threshold
            - Threshold Applied: Removed features with {incomplete_threshold}% or more missing values.
            - Explanation: Features with a high percentage of missing values may introduce noise or require extensive imputation.
    - After our feature selection process, {number_of_features} actionable features were retained for modeling.
- #### Target Population {{#target-population}}
{target_population_section}
    - This resulted in a training dataset of {training_dataset_size} students within the target timeframe.
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
    - Our bias evaluation metric is _False Negative Rate (FNR)_, which measures the rate at which the model is incorrectly predicting students in need of support as NOT in need of support. 
    - FNR Parity helps us assess whether the model is underpredicting students in need of support at _disproportionate_ rates for any particular subgroup. In other words, it checks whether the model is incorrectly classifying students who need support more often in some demographic groups than in others

- #### Analyzing Bias Across Student Groups
{bias_groups_section}
    - We evaluated FNR across these student groups and tested for statistically significant disparities.

{bias_summary_section}

### Important Features {{#features}}
- #### Analyzing Feature Importance
    - SHAP (Shapley Additive Explanations) is a method based on cooperative game theory that quantifies the contribution of each feature to a model's prediction for an individual instance. 
    - It helps us understand how much a particular feature contributed to predicting whether a student needs more or less support.

- #### Feature Importance Plot
    - This figure below helps explain how individual features contribute to the model’s predictions for each student-term record.
    - Guidelines to interpret the plot below:
        - Each dot represents a single student-term record.
        - Features are ordered from top to bottom by their overall importance to the model — the most influential features appear at the top.
        - **SHAP values (x-axis, left to right)** → indicate how a feature’s value influences the model’s prediction:
            - More left (-) → feature value contributes to the model predicting a lower likelihood of needing support
            - More right (+) → feature value contributes to the model predicting a higher likelihood of needing support
            - **Feature values (y-axis, top to bottom)** → the numeric value of that feature; <span class="dk-red">high</span> or <span class="dk-blue">low</span>
            - For True/False variables:
                - <span class="dk-red">True</span> is represented by a _high_ feature value (1) in <span class="dk-red">red</span>.
                - <span class="dk-blue">False</span> is represented by a _low_ feature value (0) in (<span class="dk-blue">blue</span>).
            - <span class="dk-gray">Categorical features</span>, which are not continuous numeric features (e.g., enrollment type), are represented in <span class="dk-gray">gray</span>.
        - Example: _Students with a lower percentage of grades above the section’s average tend to have SHAP values further to the right, indicating that this feature contributes to the model predicting a higher likelihood of needing support._

{feature_importances_by_shap_plot}

### Appendix {{#appendix}}

{performance_by_splits_section}

{selected_features_ranked_by_shap}

{evaluation_by_group_section}