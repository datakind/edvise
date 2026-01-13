# Databricks notebook source
# MAGIC %md
# MAGIC # Edvise Custom Data Assessment and Preprocessing 
# MAGIC
# MAGIC First step in the process of transforming raw data into actionable, data-driven insights for advisors: load raw data, build a schema contract to enhance data & pipeline reliability, and ensure limited training-inference skew.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks Classification with AutoML](https://docs.databricks.com/en/machine-learning/automl/classification.html)
# MAGIC - [Databricks AutoML Python API reference](https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# WARNING: AutoML/mlflow expect particular packages with version constraints
# that directly conflicts with dependencies in our SST repo. As a temporary fix,
# we need to manually install a certain version of pandas and scikit-learn in order
# for our models to load and run properly.

# %pip install git+https://github.com/datakind/edvise.git@v0.1.8
# %restart_python

# COMMAND ----------

import logging
import seaborn as sns
import pandas as pd
import matplotlib.pyplot  as plt
from databricks.connect import DatabricksSession
from py4j.protocol import Py4JJavaError

from edvise import dataio, configs


## Data Audit Imports

from edvise.data_audit.eda import (
    find_dupes, 
    check_pf_grade_consistency,
    check_earned_vs_attempted,
    order_terms,
    validate_credit_consistency,
)

## Data Cleaning Imports

from edvise.data_audit.custom_cleaning import (
    drop_readmits_and_dedupe_keep_earliest,
    convert_numeric_columns,
    assign_numeric_grade,
)

from edvise.utils.data_cleaning import handling_duplicates 

try:
  # Get the pipeline type from job definition
  run_type = dbutils.widgets.get("run_type") # noqa: F821
except Py4JJavaError:
  # Run script interactively
  run_type = 'train'

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## configuration

# COMMAND ----------

# project configuration stored as a config file in TOML format
cfg = dataio.read.read_config(
    "./config.toml", schema=configs.custom.CustomProjectConfig
)
cfg

# COMMAND ----------

# MAGIC %md
# MAGIC ## read raw datasets

# COMMAND ----------

student_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze['raw_cohort'].file_path,
    spark_session=spark,
)
course_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze['raw_course'].file_path,
    spark_session=spark,
)

# OPTIONAL gen ed course list
# file_path = "{insert file path to gen ed courses csv}"
# gen_eds = dataio.read.from_csv_file(file_path)
# gen_eds

# COMMAND ----------

# MAGIC %md
# MAGIC # Explore unique keys & Null Values
# MAGIC According to this [file](https://docs.google.com/spreadsheets/d/1zOLv2VOIhDpy6f_2KdOJqLOgA9GNhxW8ZUwneMPF-8A/edit?gid=0#gid=0), the  keys should be as follows: 
# MAGIC - student: Student ID
# MAGIC - course: Student ID, Semester, Course Prefix, Course Number
# MAGIC - semester: Student ID, Semester
# MAGIC
# MAGIC Check if these are the case, and if not, explore what is unique

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw Cohort Data

# COMMAND ----------

student_raw_df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploring NA values:**
# MAGIC
# MAGIC Note any columns with high numbers of NA values to point out to the institution (if they are of relevance).

# COMMAND ----------

# check for NAs
student_raw_df.isna().sum().sort_values(ascending=False)

# COMMAND ----------

# check for NAs - percents 
na_percent_cohort = (student_raw_df.isna().mean() * 100).sort_values(ascending=False)
print(na_percent_cohort)

# Plot
na_percent_cohort.head(10).plot(kind="bar")
plt.title("Percentage of Missing Values by Column")
plt.ylabel("Percent of Missing Values (%)")
plt.xlabel("Columns")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# COMMAND ----------

na_percent_by_cohort = (
    student_raw_df
    .groupby("entry_term")
    .apply(lambda df: df.isna().mean() * 100)
    .T  
)

na_percent_by_cohort

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Cohort Stats

# COMMAND ----------

# Enforce term order 
ordered_cohort = order_terms(student_raw_df, "entry_term")
ordered_cohort.head()

# COMMAND ----------

# Plot
ax = sns.histplot(
    ordered_cohort.sort_values("entry_term"),
    y="entry_term",
    hue="entry_type",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(xlabel="Number of Students")

plt.show()

# COMMAND ----------

# what cohorts exist
student_raw_df['entry_term'].value_counts().sort_index()

# what are the common entry types 
# raw counts
print(student_raw_df["entry_type"].value_counts())

# percents
print(student_raw_df["entry_type"].value_counts(normalize=True)*100)

# COMMAND ----------

# MAGIC %md
# MAGIC **Investigating Cohort Credits Earned Data**
# MAGIC
# MAGIC We are seeking to ensure there is _consistency_ between the credits attempted vs credits earned columns: 
# MAGIC
# MAGIC 1. Credits earned is less than or equal to credits attempted. 
# MAGIC 2. Credits earned are zero if credits attempted are zero. 
# MAGIC     - Sometimes school will utilize this logic if a student withdraws from a course.

# COMMAND ----------

results = check_earned_vs_attempted(
    df=student_raw_df, 
    earned_col="inst_tot_credits_earned", 
    attempted_col="inst_tot_credits_attempted"
)

anomalies = results["anomalies"]
summary = results["summary"]

print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring Cohort Duplicates

# COMMAND ----------

# check number of unique student IDs
print(student_raw_df["student_id"].nunique())

# confirm if Student_ID is unique - should be FALSE
student_raw_df['student_id'].duplicated().any()

# COMMAND ----------

cohort_dupes = find_dupes(student_raw_df, primary_keys = cfg.datasets.bronze["raw_cohort"].primary_keys)
cohort_dupes.head()

# COMMAND ----------

# what are the common entry types 
# raw counts
print(cohort_dupes["entry_type"].value_counts())

# COMMAND ----------

# check for duplicates of both stuent ID AND entry term 
cohort_term_dupes = find_dupes(student_raw_df, primary_keys = ['student_id', 'entry_term'])
cohort_term_dupes.head()

# COMMAND ----------

# remove readmits and check for any more dupes 
cleaned_cohort = drop_readmits_and_dedupe_keep_earliest(student_raw_df)

# COMMAND ----------

# check now to ensure all student IDs are unique
print(cleaned_cohort.shape)
print(f"{raw_cohort.shape[0] - cleaned_cohort.shape[0]} records dropped due to re-admit data and duplicates, leaving us with {cleaned_cohort["student_id"].nunique()} unique student IDs.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## cleaned cohort stats

# COMMAND ----------

# Entry Types
# raw counts
print(cleaned_cohort["entry_type"].value_counts())

# percents
print(cleaned_cohort["entry_type"].value_counts(normalize=True)*100)

# Entry Terms
# raw counts
print(cleaned_cohort["entry_term"].value_counts())

# percents
cleaned_cohort["entry_term"].value_counts(normalize=True)*100

# COMMAND ----------

# first gen count
print(cleaned_cohort["first_gen"].value_counts(normalize=True))

# first_gen count
(
    sns.histplot(
        cleaned_cohort,
        y="first_gen",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

# pell count
cleaned_cohort["awarded_pell"].value_counts(normalize=True)

cleaned_cohort["awarded_pell"] = (
    cleaned_cohort["awarded_pell"]
      .astype(str)
      .str.strip()
      .str.upper()
      .map({"Y": "Y", "N": "N"})         
)

# Lock the category order so Seaborn doesn't invent a gray level
cats = ["Y", "N"]
cleaned_cohort["awarded_pell"] = pd.Categorical(cleaned_cohort["awarded_pell"], categories=cats, ordered=True)

palette = {"Y": "#1f77b4", "N": "#ff7f0e"}

ax = sns.histplot(
    cleaned_cohort,
    y="first_gen",
    hue="awarded_pell",
    hue_order=cats,            # ensures consistent color assignment
    multiple="stack",          # avoids overlay artifacts that can look gray
    shrink=0.75,
    edgecolor="white",
    palette=palette,
)
ax.set(xlabel="Number of Students")
plt.show()

# COMMAND ----------

cleaned_cohort.groupby("first_gen")["awarded_pell"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

# degree types
(
    sns.histplot(
        cleaned_cohort,
        y="deg_at_grad",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

cleaned_cohort["deg_at_grad"].value_counts(normalize=True)

# COMMAND ----------

# helper function to change string -> numeric
cleaned_numeric_cohort = convert_numeric_columns(cleaned_cohort, ["college_gpa", "inst_tot_credits_attempted", "inst_tot_credits_earned", "overall_credits_earned"])

# COMMAND ----------

# Plot
jg = sns.jointplot(
    cleaned_numeric_cohort,
    x="inst_tot_credits_attempted",
    y="inst_tot_credits_earned",
    kind="hex",
    joint_kws={"bins": "log"},
    marginal_kws={"edgecolor": "white"},
    ratio=4,
)
jg.refline(y=120.0)  # or whichever num credits earned is a relavent benchmark
jg.set_axis_labels("Number of Credits Attempted", "Number of Credits Earned")

# COMMAND ----------

# average and median number of credits earned by transfers vs first time admits
summary = (
    cleaned_numeric_cohort.groupby("entry_type")[
        ["inst_tot_credits_earned", "inst_tot_credits_attempted", "overall_credits_earned"]
    ]
    .agg(["count", "mean", "median"])
    .reset_index()
)
summary

# COMMAND ----------

# overall credits earned
summary = (
    cleaned_numeric_cohort
    .groupby("entry_type")["overall_credits_earned"]
    .agg(["mean", "median"])
)

summary.plot(kind="bar", figsize=(6,4))
plt.title("Mean and Median Overall Credits Earned by Entry Type")
plt.ylabel("Overall Credits Earned")
plt.xlabel("Entry Type")
plt.xticks(rotation=0, ha="center")
plt.legend(["Mean (Average) Overall Credits Earned", "Median Overall Credits Earned"])
plt.tight_layout()
plt.show()

# COMMAND ----------

# what percent of each group achieves >120 credits?
(
    cleaned_numeric_cohort
    .groupby("entry_type")["overall_credits_earned"]
    .apply(lambda x: (x >= 120).mean() * 100)
)

# COMMAND ----------

# check missing GPAs
(
    cleaned_numeric_cohort
    .groupby("entry_type")["college_gpa"]
    .apply(lambda x: x.isna().mean() * 100)
)

# COMMAND ----------

# GPA by entry term and type
ax = sns.barplot(
    cleaned_numeric_cohort.sort_values(by="entry_term"),
    x="entry_term",
    y="college_gpa",
    estimator="mean",
    hue="entry_type",
    edgecolor="white",
)

# Set the ylabel
ax.set(ylabel="Average College GPA")

# Move the legend to a different location (e.g., upper left)
ax.legend(loc="lower left", title="Entry Type")
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw Course Data

# COMMAND ----------

course_raw_df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploring NA values:**
# MAGIC
# MAGIC Note any columns with high numbers of NA values to point out to the institution (if they are of relevance).

# COMMAND ----------

# check for NAs
course_raw_df.isna().sum().sort_values(ascending=False)

# COMMAND ----------

# check for NAs - percents 
na_percent_course = (course_raw_df.isna().mean() * 100).sort_values(ascending=False)
print(na_percent_course)

# Plot
na_percent_course.head(10).plot(kind="bar")
plt.title("Percentage of Missing Values by Column")
plt.ylabel("Percent of Missing Values (%)")
plt.xlabel("Columns")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Course Stats

# COMMAND ----------

# Enforce term order
ordered_course = order_terms(course_raw_df, "term")

# COMMAND ----------

# Plot
ax = sns.histplot(
    ordered_course.sort_values("term"),
    y="term",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(ylabel="Academic Term and Year", xlabel="Number of Enrollments", title="Course Enrollments by Academic Term and Year")

plt.show()

# COMMAND ----------

# unique course subjects and titles
print(course_raw_df["course_title"].nunique())
course_raw_df["course_subject"].nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleaning Course Data

# COMMAND ----------

# MAGIC %md
# MAGIC Since we aren't interested in student IDs outside our cleaned cohort file, let's filter to just those records first before further investigation.

# COMMAND ----------

# stratifying to just new freshmans and transfers, excluding readmits
student_ids = cleaned_cohort["student_id"].tolist()
filtered_course = course_raw_df[course_raw_df["student_id"].isin(student_ids)]
print(filtered_course.shape)
# how many records were dropped? 
print(f"{course_raw_df.shape[0] - filtered_course.shape[0]} student-course records were filtered out")
filtered_course.head()

# COMMAND ----------

# MAGIC %md
# MAGIC We also want to handle any duplicates, similar to how we handled our cohort dataset.

# COMMAND ----------

# exploring course file duplicates
course_dupes = find_dupes(filtered_course, primary_keys = cfg.datasets.bronze["raw_course"].primary_keys)
course_dupes.head()

# COMMAND ----------

# for deck
course_dupes[["student_id", "term", "course_subject", "course_num", "course_section", "course_title", "course_type", "pass_fail_flag", "grade", "course_credits", "credits_earned"]].head()

# COMMAND ----------

# view course types
print(course_dupes["course_type"].value_counts(dropna=False))
print(course_dupes["course_type"].value_counts(normalize=True, dropna=False)*100)

# view pass fail flag 
print(course_dupes.groupby("course_type")["pass_fail_flag"].value_counts(dropna=False).sort_index())

# view grades
print(course_dupes.groupby("course_type")["grade"].value_counts(dropna=False).sort_index())

# view credits earned
course_dupes.groupby("course_type")[["course_credits_attempted", "course_credits_earned"]].value_counts(dropna=False).sort_index()

# COMMAND ----------

cleaned_course = handling_duplicates(course_raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleaned Course Data: EDA

# COMMAND ----------

# MAGIC %md
# MAGIC **Investigating Course Grades Column**
# MAGIC
# MAGIC We are seeking to ensure 3 main rules: 
# MAGIC 1. Students NEVER earn credits for failing grades.
# MAGIC 2. Students DO always earn credits for passing grades.
# MAGIC 3. Grade and pass_fail_flag are consistent.
# MAGIC 4. Credits Earned <= Credits Attempted 

# COMMAND ----------

# investigating grades column
cleaned_course[["grade", "pass_fail_flag", "course_credits_earned"]]

# COMMAND ----------

print(cleaned_course["pass_fail_flag"].value_counts(dropna=False))
print(cleaned_course["pass_fail_flag"].value_counts(dropna=False, normalize=True)*100)
print(cleaned_course["grade"].value_counts(dropna=False))
cleaned_course["grade"].value_counts(dropna=False, normalize=True)*100

# COMMAND ----------

# ex. inspect what unknown grade values are marked as
# incompletes
# print(cleaned_course[cleaned_course["grade"] == "I"][["grade", "pass_fail_flag", "course_credits_attempted", "course_credits_earned"]])

# withdraws
# print(cleaned_course[cleaned_course["grade"] == "W"][["grade", "pass_fail_flag", "course_credits_attempted", "course_credits_earned"]])

# CH ? 
# print(cleaned_course[cleaned_course["grade"] == "CH"][["grade", "pass_fail_flag", "course_credits_attempted", "coruse_credits_earned"]])

# NR ? Not Repeat 
# cleaned_course[cleaned_course["grade"] == "NR"][["grade", "pass_fail_flag", "course_credits_attempted", "course_credits_earned"]]

# REP - remedial 
# cleaned_course[cleaned_course["grade"] == "REP"][["grade", "pass_fail_flag", "course_credits_attempted", "course_credits_earned"]]

# COMMAND ----------

anomalies, summary = check_pf_grade_consistency(cleaned_course)
summary

# COMMAND ----------

anomalies["grade"].value_counts(dropna=False)

# COMMAND ----------

# ex. inspect anomalies 
# pass grade but earned 0 credits? 
# print(anomalies[anomalies["pass_fail_flag"] == "P"]["course_credits_earned"].value_counts())

# pass grade but earned 0 credits? 
# print(anomalies[anomalies["pass_fail_flag"] == "P"]["course_credits_attempted"].value_counts())

# pass/fail flag disagree - pass grades
# print(anomalies[anomalies["pass_fail_flag"] == "P"]["grade"].value_counts())

# pass/fail flag disagree - fail grades 
# print(anomalies[anomalies["pass_fail_flag"] == "F"]["grade"].value_counts())

# COMMAND ----------

# inspect course delivery values 
print(cleaned_course["course_delivery"].value_counts(dropna=False))


# see nulls by term
(
    cleaned_course
    .groupby("term")["course_delivery"]
    .apply(lambda x: x.isna().sum())
    .reset_index(name="null_count")
    .sort_values("null_count", ascending=False)
)

# COMMAND ----------

# check credits attempted and credits earned fields 
validate_credit_consistency(course_raw_df, student_raw_df)

# COMMAND ----------

# assign numeric grade
cleaned_course = assign_numeric_grade(cleaned_course)

# COMMAND ----------

# Plot
ax = sns.histplot(
    cleaned_course,
    x="course_numeric_grade",
    hue="course_delivery",
    multiple="stack",
    binwidth=1,
    binrange=(0, 5),
    edgecolor="white",
)

ax.set(xlabel="Course grade", ylabel="Number of course enrollments")
plt.title("Course Grades by Delivery Method")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raw Semester Data

# COMMAND ----------

# MAGIC %md
# MAGIC We won't be needing this for future Edvise Schema schools, but for old ones, it's useful!

# COMMAND ----------

# reading raw semester data 
raw_semester = dataio.read.from_csv_file(cfg.datasets.bronze["raw_semester"].file_path)
raw_semester

# COMMAND ----------

raw_semester.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC **Exploring NA values:**
# MAGIC
# MAGIC Note any columns with high numbers of NA values to point out to the institution (if they are of relevance).

# COMMAND ----------

# check for NAs
raw_semester.isna().sum().sort_values(ascending=False)

# COMMAND ----------

# check for NAs - percents 
na_percent_semester = (raw_semester.isna().mean() * 100).sort_values(ascending=False)
print(na_percent_semester)

# Plot
na_percent_semester.head(3).plot(kind="bar")
plt.title("Percentage of Missing Values by Column")
plt.ylabel("Percent of Missing Values (%)")
plt.xlabel("Columns")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# COMMAND ----------

na_percent_by_term = (
    raw_semester
    .groupby("term")
    .apply(lambda df: df.isna().mean() * 100)
    .T  
)

na_percent_by_term

# COMMAND ----------

semester_dupes = find_dupes(raw_semester, primary_keys = cfg.datasets.bronze["raw_semester"].primary_keys)
semester_dupes.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Key Semester Stats

# COMMAND ----------

# enforce term order
ordered_semester = order_terms(raw_semester, "term")

# COMMAND ----------

# Plot
ax = sns.histplot(
    ordered_semester.sort_values("term"),
    y="term",
    hue="ftpt",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set_xlabel("Number of Students")

plt.tight_layout()
plt.show()

# COMMAND ----------

print(raw_semester["term"].nunique())
raw_semester["term"].value_counts(dropna=False).sort_index()

# COMMAND ----------

raw_semester["ftpt"].value_counts(normalize=True, dropna=False)*100

# COMMAND ----------

# NOTE: IF YOU HAVE A SEMESTER FILE, RUN THIS
# validate_credit_consistency(raw_semester, course_raw_df, student_raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleaning Semester Data & EDA

# COMMAND ----------

cleaned_semester = raw_semester[raw_semester["student_id"].isin(student_ids)]
print(cleaned_semester.shape)
# how many records were dropped? 
print(f"{raw_semester.shape[0] - cleaned_semester.shape[0]} student-semester records were filtered out")
cleaned_semester.head() 

# COMMAND ----------

# Filter first
filtered = cleaned_semester[
    cleaned_semester["term"].str.contains("Spring|Summer|Fall", case=False, na=False)
].copy()

# Enforce term order 
filtered = order_terms(filtered, "term")

# COMMAND ----------

# Plot
# GPA by term and enrollment intensity

ax = sns.barplot(
    filtered.sort_values(by="term").astype(
        {"cum_gpa": "Float32"}
    ),
    x="term",
    y="cum_gpa",
    estimator="mean",
    hue="ftpt",
    edgecolor="white",
    errorbar=None,
)

# Set the ylabel
ax.set(ylabel="Cumulative GPA")

# Move the legend to a different location (e.g., upper left)
ax.legend(loc="lower left", title="Enrollment Intensity")

# Show the plot
plt.xticks(rotation=45, ha="right")
plt.title("Cumulative GPA by Term and Enrollment Intensity")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## clean raw datasets

# COMMAND ----------

# 1) Start from raw dfs
raw_map = {
    "student_df": student_raw_df,
    "course_df":  course_raw_df,
}

# 2) Define df pairs from bronze dataset config
DF_MAP = {
    "student_df":   ("raw_cohort",   raw_map["student_df"]),
    "course_df":    ("raw_course",   raw_map["course_df"]),
}

# 3) Deduplication logic per dataset
# CAN ADJUST THIS BASED ON CUSTOM FUNCS - these are default fallbacks!
dedupe_fn_by_dataset = {
    "student_df": drop_readmits_and_dedupe_keep_earliest,
    "course_df": handling_duplicates,
}

# 4) Pull cleaning config
cleaning_cfg = cfg.preprocessing.cleaning
cleaning_cfg

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merging all 3 Datasets: Cohort, Course and Semester

# COMMAND ----------

# MAGIC %md
# MAGIC Begin with validating the merges on just the raw datasets first to investigate any inconsistencies. 

# COMMAND ----------

# cohort x course
student_raw_df.merge(course_raw_df, on="student_id", indicator=True)._merge.value_counts()

# COMMAND ----------

# NOTE: ONLY RUN IF YOU HAVE A SEMESTER FILE
# cohort x semester
# student_raw_df.merge(raw_semester, on=["student_id", "first_reg_date"], indicator=True)._merge.value_counts()

# COMMAND ----------

# NOTE: ONLY RUN IF YOU HAVE A SEMESTER FILE
# semester x course
# raw_semester.merge(course_raw_df, on=["student_id", "term"], indicator=True)._merge.value_counts()
