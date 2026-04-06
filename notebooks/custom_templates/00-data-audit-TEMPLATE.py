# Databricks notebook source
# MAGIC %md
# MAGIC # Edvise Custom Data Audit (tables only)
# MAGIC
# MAGIC Institutional audit of raw bronze files: completeness (missingness), key uniqueness (duplicates),
# MAGIC cross-file linkage (outer merges), grade or pass-fail logic, and credit reconciliation. No charts.
# MAGIC
# MAGIC **Pipeline context:** this notebook is **00**. After remediation, continue with
# MAGIC `01-preprocess-data-TEMPLATE.py`, then `02-train-h2o-model-TEMPLATE.py`, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### NOTE
# MAGIC Column names vary by school. Edit cells to match your schema; confirm names against `config.toml` and sample `head()` output.

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# WARNING: AutoML/mlflow expect particular packages with version constraints
# that directly conflicts with dependencies in our SST repo. As a temporary fix,
# we need to manually install a certain version of pandas and scikit-learn in order
# for our models to load and run properly.

# %pip install git+https://github.com/datakind/edvise.git@v0.1.10
# %restart_python

# COMMAND ----------

import logging

import pandas as pd
from databricks.connect import DatabricksSession
from py4j.protocol import Py4JJavaError

from edvise import dataio, configs

from edvise.data_audit.eda import (
    analyze_merge,
    check_earned_vs_attempted,
    check_pf_grade_consistency,
    find_dupes,
    infer_term_column,
    order_terms,
    validate_credit_consistency,
)

from edvise.utils.data_cleaning import handling_duplicates

try:
    run_type = dbutils.widgets.get("run_type")  # noqa: F821
except Py4JJavaError:
    run_type = "train"

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC **What this proves:** documented pipeline settings and file paths used for the audit.

# COMMAND ----------

cfg = dataio.read.read_config(
    "./config.toml", schema=configs.custom.CustomProjectConfig
)
display(cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data
# MAGIC
# MAGIC **What this proves:** the same bronze inputs the pipeline will use.

# COMMAND ----------

student_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze["raw_student"].file_path,
    spark_session=spark,
)
course_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze["raw_course"].file_path,
    spark_session=spark,
)
semester_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze["raw_semester"].file_path,
    spark_session=spark,
)

# COMMAND ----------

# Infer term-like columns from values (and name hints). Override printed names if wrong for your school.
TERM_NAME_HINTS_COHORT = (
    "entry_term",
    "admit_term",
    "cohort_term",
    "enrollment_term",
    "first_enrollment_term",
    "matriculation_term",
    "start_term",
    "term",
)
TERM_NAME_HINTS_SCHEDULE = (
    "term",
    "semester",
    "academic_term",
    "acad_term",
    "strm",
    "term_id",
    "semester_term",
)

TERM_COL_STUDENT = infer_term_column(
    student_raw_df, name_hints=TERM_NAME_HINTS_COHORT
)
TERM_COL_COURSE = infer_term_column(
    course_raw_df, name_hints=TERM_NAME_HINTS_SCHEDULE
)
TERM_COL_SEMESTER = infer_term_column(
    semester_raw_df, name_hints=TERM_NAME_HINTS_SCHEDULE
)

print(
    "Inferred term columns (inspect and override if needed):",
    {
        "student (cohort / entry)": TERM_COL_STUDENT,
        "course (enrollment term)": TERM_COL_COURSE,
        "semester": TERM_COL_SEMESTER,
    },
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Missing values
# MAGIC
# MAGIC **What this proves:** which fields are incomplete in each raw extract (risk for modeling and reporting).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Student (cohort) file

# COMMAND ----------

student_raw_df.dtypes

# COMMAND ----------

na_counts_student = student_raw_df.isna().sum().sort_values(ascending=False)
na_counts_student

# COMMAND ----------

na_pct_student = (student_raw_df.isna().mean() * 100).sort_values(ascending=False)
na_pct_student

# COMMAND ----------

# Term roster counts in chronological order (table only; uses inferred TERM_COL_STUDENT)
if TERM_COL_STUDENT and TERM_COL_STUDENT in student_raw_df.columns:
    _oc = order_terms(student_raw_df, TERM_COL_STUDENT)
    display(
        _oc[TERM_COL_STUDENT]
        .value_counts()
        .sort_index()
        .rename("student_rows")
        .to_frame()
    )
else:
    print("No inferred term column for student file; skip term value counts.")

# COMMAND ----------

# Optional: NA % by cohort / entry term (uses inferred TERM_COL_STUDENT)
if TERM_COL_STUDENT and TERM_COL_STUDENT in student_raw_df.columns:
    na_by_term_student = (
        student_raw_df.groupby(TERM_COL_STUDENT)
        .apply(lambda df: df.isna().mean() * 100)
        .T
    )
    display(na_by_term_student)
else:
    print("No inferred term column for student file; skip NA% by term.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Course file

# COMMAND ----------

course_raw_df.dtypes

# COMMAND ----------

na_counts_course = course_raw_df.isna().sum().sort_values(ascending=False)
na_counts_course

# COMMAND ----------

na_pct_course = (course_raw_df.isna().mean() * 100).sort_values(ascending=False)
na_pct_course

# COMMAND ----------

if TERM_COL_COURSE and TERM_COL_COURSE in course_raw_df.columns:
    _o_course = order_terms(course_raw_df, TERM_COL_COURSE)
    display(
        _o_course[TERM_COL_COURSE]
        .value_counts()
        .sort_index()
        .rename("course_rows")
        .to_frame()
    )
else:
    print("No inferred term column for course file; skip term value counts.")

# COMMAND ----------

# Optional: NA % by academic term (uses inferred TERM_COL_COURSE)
if TERM_COL_COURSE and TERM_COL_COURSE in course_raw_df.columns:
    na_by_term_course = (
        course_raw_df.groupby(TERM_COL_COURSE)
        .apply(lambda df: df.isna().mean() * 100)
        .T
    )
    display(na_by_term_course)
else:
    print("No inferred term column for course file; skip NA% by term.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Semester file

# COMMAND ----------

semester_raw_df.dtypes

# COMMAND ----------

na_counts_semester = semester_raw_df.isna().sum().sort_values(ascending=False)
na_counts_semester

# COMMAND ----------

na_pct_semester = (semester_raw_df.isna().mean() * 100).sort_values(ascending=False)
na_pct_semester

# COMMAND ----------

if TERM_COL_SEMESTER and TERM_COL_SEMESTER in semester_raw_df.columns:
    _o_sem = order_terms(semester_raw_df, TERM_COL_SEMESTER)
    display(
        _o_sem[TERM_COL_SEMESTER]
        .value_counts()
        .sort_index()
        .rename("semester_rows")
        .to_frame()
    )
else:
    print("No inferred term column for semester file; skip term value counts.")

# COMMAND ----------

if TERM_COL_SEMESTER and TERM_COL_SEMESTER in semester_raw_df.columns:
    na_by_term_semester = (
        semester_raw_df.groupby(TERM_COL_SEMESTER)
        .apply(lambda df: df.isna().mean() * 100)
        .T
    )
    display(na_by_term_semester)
else:
    print("No inferred term column for semester file; skip NA% by term.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cohort: earned vs attempted credits (row-level)
# MAGIC
# MAGIC **What this proves:** institutional totals do not violate earned ≤ attempted (and zero earned when zero attempted).

# COMMAND ----------

# Edit column names to match config.toml / schema for your school
results_cohort_credits = check_earned_vs_attempted(
    df=student_raw_df,
    earned_col="inst_tot_credits_earned",
    attempted_col="inst_tot_credits_attempted",
)

display(results_cohort_credits["summary"])
display(results_cohort_credits["anomalies"].head(50))

# COMMAND ----------

# MAGIC %md
# MAGIC # Duplicates
# MAGIC
# MAGIC **What this proves:** whether claimed primary keys are unique. When duplicates exist, printed tables show
# MAGIC which columns disagree within duplicate key groups (`pct_conflicting_groups`). Review `head()` below for examples.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cohort (student) file

# COMMAND ----------

cohort_dupes = find_dupes(
    student_raw_df,
    primary_keys=cfg.datasets.bronze["raw_student"].primary_keys,
)
if not cohort_dupes.empty:
    display(cohort_dupes.head(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Course file

# COMMAND ----------

course_dupes = find_dupes(
    course_raw_df,
    primary_keys=cfg.datasets.bronze["raw_course"].primary_keys,
)
if not course_dupes.empty:
    display(course_dupes.head(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Semester file

# COMMAND ----------

semester_dupes = find_dupes(
    semester_raw_df,
    primary_keys=cfg.datasets.bronze["raw_semester"].primary_keys,
)
if not semester_dupes.empty:
    display(semester_dupes.head(50))

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross-file merge checks
# MAGIC
# MAGIC **What this proves:** coverage between roster, course, and semester extracts (same students and terms where expected).
# MAGIC
# MAGIC **`_merge` categories (pandas `indicator=True`, outer join):**
# MAGIC - **`both`:** key appears in left and right tables.
# MAGIC - **`left_only`:** key only in the left table (right file missing those rows).
# MAGIC - **`right_only`:** key only in the right table (left file missing those rows).
# MAGIC
# MAGIC `analyze_merge` prints row and distinct-student counts; pass `student_df=student_raw_df` so percentages use the roster denominator.

# COMMAND ----------

_ = analyze_merge(
    student_raw_df,
    course_raw_df,
    "student",
    "course",
    student_df=student_raw_df,
    merge_on="student_id",
    id_col="student_id",
)

# COMMAND ----------

# student × semester: use inferred term columns; align names with a common join key when they differ
_id = "student_id"
if (
    TERM_COL_STUDENT
    and TERM_COL_SEMESTER
    and _id in student_raw_df.columns
    and _id in semester_raw_df.columns
):
    if TERM_COL_STUDENT == TERM_COL_SEMESTER:
        _stu_sm = student_raw_df
        _sem_sm = semester_raw_df
        _keys_sm = [_id, TERM_COL_STUDENT]
    else:
        _join = "_audit_term_join"
        _stu_sm = student_raw_df.rename(columns={TERM_COL_STUDENT: _join})
        _sem_sm = semester_raw_df.rename(columns={TERM_COL_SEMESTER: _join})
        _keys_sm = [_id, _join]
        print(
            f"student×semester join: aligned {TERM_COL_STUDENT!r} (student) with "
            f"{TERM_COL_SEMESTER!r} (semester) as {_join!r}"
        )
    _ = analyze_merge(
        _stu_sm,
        _sem_sm,
        "student",
        "semester",
        student_df=student_raw_df,
        merge_on=_keys_sm,
        id_col=_id,
    )
else:
    print(
        "Skip student × semester merge: need student_id plus inferred term on both files, "
        "or use first_reg_date-style keys manually (rename columns and edit this cell)."
    )

# COMMAND ----------

# semester × course: inferred term columns; align names when they differ
if (
    TERM_COL_SEMESTER
    and TERM_COL_COURSE
    and _id in semester_raw_df.columns
    and _id in course_raw_df.columns
):
    if TERM_COL_SEMESTER == TERM_COL_COURSE:
        _sem_sc = semester_raw_df
        _crs_sc = course_raw_df
        _keys_sc = [_id, TERM_COL_SEMESTER]
    else:
        _join = "_audit_term_join"
        _sem_sc = semester_raw_df.rename(columns={TERM_COL_SEMESTER: _join})
        _crs_sc = course_raw_df.rename(columns={TERM_COL_COURSE: _join})
        _keys_sc = [_id, _join]
        print(
            f"semester×course join: aligned {TERM_COL_SEMESTER!r} (semester) with "
            f"{TERM_COL_COURSE!r} (course) as {_join!r}"
        )
    _ = analyze_merge(
        _sem_sc,
        _crs_sc,
        "semester",
        "course",
        student_df=student_raw_df,
        merge_on=_keys_sc,
        id_col=_id,
    )
else:
    print(
        "Skip semester × course merge: need student_id plus inferred term on both files."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Grades and pass-fail consistency
# MAGIC
# MAGIC **What this proves:** pass-fail flags, letter grades, and credit earned fields follow consistent business rules
# MAGIC (e.g., no credits earned on failing rows when the flag says fail).

# COMMAND ----------

# De-duplicate course rows the same way as downstream pipelines before grading rules
cleaned_course = handling_duplicates(course_raw_df)

# Edit column names to match schema: grade, pass_fail_flag, credits earned on the course row
GRADE_COL = "grade"
PF_COL = "pass_fail_flag"
CREDITS_COL = "course_credits_earned"

anomalies_pf, summary_pf = check_pf_grade_consistency(
    cleaned_course,
    grade_col=GRADE_COL,
    pf_col=PF_COL,
    credits_col=CREDITS_COL,
)

display(summary_pf)
display(anomalies_pf.head(50))

# COMMAND ----------

# MAGIC %md
# MAGIC # Credit consistency across files
# MAGIC
# MAGIC **What this proves:** course-level earned vs attempted, reconciliation of course sums to semester totals where columns exist,
# MAGIC and cohort-level earned vs attempted (embedded in this helper). Align all names with `config.toml`.

# COMMAND ----------

# Reconciliation needs the same sem_col on course and semester; align with renames when inferred names differ
if TERM_COL_COURSE and TERM_COL_SEMESTER and TERM_COL_SEMESTER != TERM_COL_COURSE:
    semester_for_credit = semester_raw_df.rename(
        columns={TERM_COL_SEMESTER: TERM_COL_COURSE}
    )
    SEM_COL_FOR_CREDIT = TERM_COL_COURSE
    print(
        f"Credit validation: reconciling on course term column {TERM_COL_FOR_CREDIT!r}; "
        f"renamed semester column from {TERM_COL_SEMESTER!r}."
    )
elif TERM_COL_COURSE and (
    TERM_COL_COURSE == TERM_COL_SEMESTER or TERM_COL_COURSE in semester_raw_df.columns
):
    semester_for_credit = semester_raw_df
    SEM_COL_FOR_CREDIT = TERM_COL_COURSE
elif TERM_COL_SEMESTER and TERM_COL_SEMESTER in course_raw_df.columns:
    semester_for_credit = semester_raw_df
    SEM_COL_FOR_CREDIT = TERM_COL_SEMESTER
else:
    semester_for_credit = semester_raw_df
    SEM_COL_FOR_CREDIT = "term"
    print(
        "Warning: set SEM_COL_FOR_CREDIT manually if reconciliation skips; "
        "term columns could not be aligned across course and semester."
    )

credit_audit = validate_credit_consistency(
    course_df=course_raw_df,
    semester_df=semester_for_credit,
    cohort_df=student_raw_df,
    id_col="student_id",
    sem_col=SEM_COL_FOR_CREDIT,
    # Placeholders — replace with your bronze column names from config.toml
    course_credits_attempted_col="course_credits",
    course_credits_earned_col="number_of_credits_earned",
    semester_credits_attempted_col="number_of_credits_attempted",
    semester_credits_earned_col="number_of_credits_earned",
    semester_courses_count_col="number_of_courses_enrolled",
    # Must match cohort bronze columns (same names as check_earned_vs_attempted above unless your school uses totals under other fields)
    cohort_credits_attempted_col="inst_tot_credits_attempted",
    cohort_credits_earned_col="inst_tot_credits_earned",
)

display(credit_audit["course_anomalies_summary"])
display(credit_audit["reconciliation_summary"])
display(credit_audit["cohort_anomalies_summary"])

ca = credit_audit["course_anomalies"]
if ca is not None and not ca.empty:
    display(ca.head(50))

rm = credit_audit["reconciliation_mismatches"]
if rm is not None and not rm.empty:
    display(rm.head(50))

coh = credit_audit["cohort_anomalies"]
if isinstance(coh, pd.DataFrame) and not coh.empty:
    display(coh.head(50))

# Detailed merge frame for investigation (optional)
rd = credit_audit["reconciliation_merged_detail"]
if rd is not None and not rd.empty:
    display(rd.head(30))
