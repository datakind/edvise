# Databricks notebook source
# MAGIC %md
# MAGIC # Edvise Custom Data Audit (tables only)
# MAGIC
# MAGIC Institutional audit of raw bronze files: completeness (missingness), key uniqueness (duplicates),
# MAGIC cross-file linkage (outer merges), grade or pass-fail logic, and credit reconciliation. No charts.
# MAGIC
# MAGIC **Pipeline context:** this notebook is **00**. After remediation, continue with
# MAGIC `01-data-assessment-TEMPLATE.py`, then `02-preprocess-data-TEMPLATE.py`, then training / registration notebooks (e.g. `06-train-h2o-model-TEMPLATE.py`).
# MAGIC
# MAGIC **Code layout (``edvise.data_audit``):**
# MAGIC - ``eda`` — exploratory helpers only (logging, crosstabs, ``analyze_merge``, ``value_counts_*``).
# MAGIC - ``custom_data_audit`` — structured checks (``find_dupes``, ``validate_credit_consistency``, ``check_pf_grade_consistency``, …).
# MAGIC - ``custom_cleaning`` — DataFrame transforms (``order_terms``, ``normalize_student_id_column``, schema cleaning, …).

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

# %pip install git+https://github.com/datakind/edvise.git@v0.2.0
# %restart_python

# COMMAND ----------

import logging

import pandas as pd
from databricks.connect import DatabricksSession
from py4j.protocol import Py4JJavaError

from edvise import dataio, configs

from edvise.data_audit.custom_cleaning import normalize_student_id_column, order_terms
from edvise.data_audit.custom_data_audit import (
    bias_variable_codebook_line,
    check_earned_vs_attempted,
    check_pf_grade_consistency,
    find_dupes,
    validate_credit_consistency,
)
from edvise.data_audit.eda import (
    analyze_merge,
    value_counts_percent_df,
    value_counts_sorted_count_df,
)

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
    cfg.datasets.bronze["raw_student"].train_file_path,
    spark_session=spark,
)
course_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze["raw_course"].train_file_path,
    spark_session=spark,
)
semester_raw_df = dataio.read.from_csv_file(
    cfg.datasets.bronze["raw_semester"].train_file_path,
    spark_session=spark,
)

# COMMAND ----------

# Column names for this school (edit after inspecting head() below). ``None`` = not used / not present.
# Pass ``source_col`` to ``normalize_student_id_column`` when the roster id is not already named ``student_id``.

STUDENT_ID_SOURCE_STUDENT = None
STUDENT_ID_SOURCE_COURSE = None
STUDENT_ID_SOURCE_SEMESTER = None

INST_TOT_CREDITS_ATTEMPTED_COL = None
INST_TOT_CREDITS_EARNED_COL = None

TERM_COL_STUDENT = None
TERM_COL_COURSE = None
TERM_COL_SEMESTER = None

COURSE_CRED_ATTEMPTED_COL = None
COURSE_CRED_EARNED_COL = None

SEM_CRED_ATTEMPTED_COL = None
SEM_CRED_EARNED_COL = None
SEM_COURSE_COUNT_COL = None

ENROLLMENT_INTENSITY_COL_SEMESTER = None

GRADE_COL = None
PF_COL = None
CREDITS_COL = None

STUDENT_TYPE_COL_STUDENT = None
FIRST_GEN_COL_STUDENT = None
RACE_COL_STUDENT = None
ETHNICITY_COL_STUDENT = None
GENDER_COL_STUDENT = None
AGE_COL_STUDENT = None
PELL_COL_STUDENT = None
INCARCERATION_COL_STUDENT = None
MILITARY_COL_STUDENT = None
EMPLOYMENT_COL_STUDENT = None
DISABILITY_COL_STUDENT = None

# COMMAND ----------

# Normalize primary student identifier to ``student_id`` on all bronze tables.
student_raw_df, STUDENT_ID_RESOLVED_STUDENT = normalize_student_id_column(
    student_raw_df, source_col=STUDENT_ID_SOURCE_STUDENT
)
course_raw_df, STUDENT_ID_RESOLVED_COURSE = normalize_student_id_column(
    course_raw_df, source_col=STUDENT_ID_SOURCE_COURSE
)
semester_raw_df, STUDENT_ID_RESOLVED_SEMESTER = normalize_student_id_column(
    semester_raw_df, source_col=STUDENT_ID_SOURCE_SEMESTER
)
print(
    "student_id column (after normalize_student_id_column):",
    {
        "student_file": STUDENT_ID_RESOLVED_STUDENT,
        "course_file": STUDENT_ID_RESOLVED_COURSE,
        "semester_file": STUDENT_ID_RESOLVED_SEMESTER,
    },
)

print(
    "Institutional credit total columns (cohort file):",
    {"attempted": INST_TOT_CREDITS_ATTEMPTED_COL, "earned": INST_TOT_CREDITS_EARNED_COL},
)
print(
    "Term columns:",
    {
        "student (cohort / entry)": TERM_COL_STUDENT,
        "course (enrollment term)": TERM_COL_COURSE,
        "semester": TERM_COL_SEMESTER,
    },
)
print(
    "Course / semester validation columns:",
    {
        "course_credits_attempted": COURSE_CRED_ATTEMPTED_COL,
        "course_credits_earned": COURSE_CRED_EARNED_COL,
        "semester_credits_attempted": SEM_CRED_ATTEMPTED_COL,
        "semester_credits_earned": SEM_CRED_EARNED_COL,
        "semester_course_count": SEM_COURSE_COUNT_COL,
        "semester_enrollment_intensity": ENROLLMENT_INTENSITY_COL_SEMESTER,
        "grade": GRADE_COL,
        "pass_fail_or_completion": PF_COL,
        "credits_for_pf_check": CREDITS_COL,
    },
)
print(
    "Student-type & bias columns:",
    {
        "STUDENT_TYPE_COL_STUDENT": STUDENT_TYPE_COL_STUDENT,
        "FIRST_GEN_COL_STUDENT": FIRST_GEN_COL_STUDENT,
        "RACE_COL_STUDENT": RACE_COL_STUDENT,
        "ETHNICITY_COL_STUDENT": ETHNICITY_COL_STUDENT,
        "GENDER_COL_STUDENT": GENDER_COL_STUDENT,
        "AGE_COL_STUDENT": AGE_COL_STUDENT,
        "PELL_COL_STUDENT": PELL_COL_STUDENT,
        "INCARCERATION_COL_STUDENT": INCARCERATION_COL_STUDENT,
        "MILITARY_COL_STUDENT": MILITARY_COL_STUDENT,
        "EMPLOYMENT_COL_STUDENT": EMPLOYMENT_COL_STUDENT,
        "DISABILITY_COL_STUDENT": DISABILITY_COL_STUDENT,
    },
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Validate your column choices by looking at head() on each DF.**

# COMMAND ----------

student_raw_df.head()

# COMMAND ----------

course_raw_df.head()

# COMMAND ----------

semester_raw_df.head()

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
    _tv = value_counts_sorted_count_df(_oc[TERM_COL_STUDENT], count_col="student_rows")
    display(_tv)
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
# MAGIC ## Student file: entry term & student type (percent distributions)
# MAGIC
# MAGIC **What this proves:** roster mix by inferred entry term and inferred student type (e.g. transfer, FTIC, re-admit).

# COMMAND ----------

# Entry term: percent of rows per term (includes NaN as its own bucket if present)
if TERM_COL_STUDENT and TERM_COL_STUDENT in student_raw_df.columns:
    display(value_counts_percent_df(student_raw_df[TERM_COL_STUDENT]))
else:
    print("Skip entry-term distribution: no inferred term column.")

# COMMAND ----------

# Student type: percent of rows per category (inferred column)
if STUDENT_TYPE_COL_STUDENT and STUDENT_TYPE_COL_STUDENT in student_raw_df.columns:
    display(value_counts_percent_df(student_raw_df[STUDENT_TYPE_COL_STUDENT]))
else:
    print("Skip student-type distribution: no inferred student-type column.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Student file: bias & equity variables (populatedness + distributions)
# MAGIC
# MAGIC **What this proves:** `pct_populated` is the percent of cohort rows with a non-null value for each inferred column (entry term, student type, first-gen, race, ethnicity, gender/sex, age, pell, incarceration, military, employment, disability). Distribution tables below use `pct_of_non_null_rows` (sums to 100% among non-null rows only). Codebook lines under several variables reflect typical registrar encodings.

# COMMAND ----------

_POPULATEDNESS_SPECS = (
    ("entry term", TERM_COL_STUDENT),
    ("student type", STUDENT_TYPE_COL_STUDENT),
    ("first generation", FIRST_GEN_COL_STUDENT),
    ("race", RACE_COL_STUDENT),
    ("ethnicity", ETHNICITY_COL_STUDENT),
    ("gender / sex", GENDER_COL_STUDENT),
    ("age", AGE_COL_STUDENT),
    ("pell status", PELL_COL_STUDENT),
    ("incarceration status", INCARCERATION_COL_STUDENT),
    ("military status", MILITARY_COL_STUDENT),
    ("employment status", EMPLOYMENT_COL_STUDENT),
    ("disability status", DISABILITY_COL_STUDENT),
)

_pop_rows = []
for _label, _col in _POPULATEDNESS_SPECS:
    if _col and _col in student_raw_df.columns:
        _pct_pop = round(student_raw_df[_col].notna().mean() * 100, 2)
        _pop_rows.append(
            {"variable": _label, "column": _col, "pct_populated": _pct_pop}
        )
    else:
        _pop_rows.append({"variable": _label, "column": None, "pct_populated": None})

_populatedness_df = pd.DataFrame(_pop_rows)
display(_populatedness_df)

# COMMAND ----------

for _label, _col, _codebook_role in (
    ("first generation", FIRST_GEN_COL_STUDENT, "first_gen"),
    ("race", RACE_COL_STUDENT, None),
    ("ethnicity", ETHNICITY_COL_STUDENT, None),
    ("gender / sex", GENDER_COL_STUDENT, None),
    ("age", AGE_COL_STUDENT, None),
    ("pell status", PELL_COL_STUDENT, "pell"),
    ("incarceration status", INCARCERATION_COL_STUDENT, "incarceration"),
    ("military status", MILITARY_COL_STUDENT, "military"),
    ("employment status", EMPLOYMENT_COL_STUDENT, "employment"),
    ("disability status", DISABILITY_COL_STUDENT, "disability"),
):
    if not _col or _col not in student_raw_df.columns:
        print(f"Skip distribution for {_label!r}: column not inferred or missing.")
        continue
    _s = (
        student_raw_df.loc[student_raw_df[_col].notna(), _col]
        .astype("string")
        .str.strip()
    )
    _s.name = _col
    _dist = value_counts_percent_df(_s, pct_col="pct_of_non_null_rows")
    print(f"\n--- {_label} ({_col}) — among non-null rows only ---")
    _cb = bias_variable_codebook_line(_codebook_role) if _codebook_role else None
    if _cb:
        print(_cb)
    display(_dist)

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
        value_counts_sorted_count_df(
            _o_course[TERM_COL_COURSE], count_col="course_rows"
        )
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

# Course: percent of rows per term (includes NaN as its own bucket if present)
if TERM_COL_COURSE and TERM_COL_COURSE in course_raw_df.columns:
    display(value_counts_percent_df(course_raw_df[TERM_COL_COURSE]))
else:
    print("Skip course term percent distribution: no inferred term column.")

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
        value_counts_sorted_count_df(
            _o_sem[TERM_COL_SEMESTER], count_col="semester_rows"
        )
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

# Semester: percent of rows per term (includes NaN as its own bucket if present)
if TERM_COL_SEMESTER and TERM_COL_SEMESTER in semester_raw_df.columns:
    display(value_counts_percent_df(semester_raw_df[TERM_COL_SEMESTER]))
else:
    print("Skip semester term percent distribution: no inferred term column.")

# COMMAND ----------

# Semester file: full-time vs part-time (enrollment intensity), inferred by column name
if (
    ENROLLMENT_INTENSITY_COL_SEMESTER
    and ENROLLMENT_INTENSITY_COL_SEMESTER in semester_raw_df.columns
):
    print(
        "Inferred enrollment intensity column (semester file):",
        repr(ENROLLMENT_INTENSITY_COL_SEMESTER),
    )
    display(value_counts_percent_df(semester_raw_df[ENROLLMENT_INTENSITY_COL_SEMESTER]))
else:
    print(
        "Skip enrollment intensity distribution: column not inferred or not present "
        "(expected names like ftpt, enrollment_intensity, student_term_enrollment_intensity)."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cohort: earned vs attempted credits (row-level)
# MAGIC
# MAGIC **What this proves:** institutional totals do not violate earned ≤ attempted (and zero earned when zero attempted).

# COMMAND ----------

if INST_TOT_CREDITS_ATTEMPTED_COL and INST_TOT_CREDITS_EARNED_COL:
    results_cohort_credits = check_earned_vs_attempted(
        df=student_raw_df,
        earned_col=INST_TOT_CREDITS_EARNED_COL,
        attempted_col=INST_TOT_CREDITS_ATTEMPTED_COL,
    )
    display(results_cohort_credits["summary"])
    display(results_cohort_credits["anomalies"].head(50))
else:
    print(
        "Skip cohort earned vs attempted: could not infer institutional credit total columns."
    )

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
# MAGIC # Grades and pass-fail consistency
# MAGIC
# MAGIC **What this proves:** pass-fail flags, letter grades, and credit earned fields follow consistent business rules
# MAGIC (e.g., no credits earned on failing rows when the flag says fail).

# COMMAND ----------

# Grade column distribution (percent of rows) — run immediately before PF/grade rules
if GRADE_COL and GRADE_COL in course_raw_df.columns:
    display(value_counts_percent_df(course_raw_df[GRADE_COL]))
else:
    print("Skip grade distribution: GRADE_COL not set or not in frame.")

# COMMAND ----------

if GRADE_COL and PF_COL and CREDITS_COL:
    anomalies_pf, summary_pf = check_pf_grade_consistency(
        course_raw_df,
        grade_col=GRADE_COL,
        pf_col=PF_COL,
        credits_col=CREDITS_COL,
    )
    display(summary_pf)
else:
    print(
        "Skip PF/grade check: set GRADE_COL, PF_COL, and CREDITS_COL above (or leave None if absent)."
    )
    anomalies_pf = None
    summary_pf = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## PF/grade anomalies — example rows by category

# COMMAND ----------

# uncomment and run based on above results
# anomalies_pf[anomalies_pf["earned_with_failing_grade"] == True]
# anomalies_pf[anomalies_pf["no_credits_with_passing_grade"] == True]
# anomalies_pf[anomalies_pf["grade_pf_disagree"] == True]

# COMMAND ----------

# MAGIC %md
# MAGIC # Credit consistency across files
# MAGIC
# MAGIC **What this proves:** course-level earned vs attempted, reconciliation of course sums to semester totals where columns exist,
# MAGIC and cohort-level earned vs attempted (embedded in this helper). Align all names with `config.toml`.
# MAGIC
# MAGIC Read **`institution_report`** first (plain-language summary and suggested next steps); use the tables below for row-level investigation.

# COMMAND ----------

# Reconciliation needs the same sem_col on course and semester; align with renames when inferred names differ
if TERM_COL_COURSE and TERM_COL_SEMESTER and TERM_COL_SEMESTER != TERM_COL_COURSE:
    semester_for_credit = semester_raw_df.rename(
        columns={TERM_COL_SEMESTER: TERM_COL_COURSE}
    )
    SEM_COL_FOR_CREDIT = TERM_COL_COURSE
    print(
        f"Credit validation: reconciling on course term column {SEM_COL_FOR_CREDIT!r}; "
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
    strict_columns=True,
    course_credits_attempted_col=COURSE_CRED_ATTEMPTED_COL,
    course_credits_earned_col=COURSE_CRED_EARNED_COL,
    semester_credits_attempted_col=SEM_CRED_ATTEMPTED_COL,
    semester_credits_earned_col=SEM_CRED_EARNED_COL,
    semester_courses_count_col=SEM_COURSE_COUNT_COL,
    cohort_credits_attempted_col=INST_TOT_CREDITS_ATTEMPTED_COL,
    cohort_credits_earned_col=INST_TOT_CREDITS_EARNED_COL,
)

print(credit_audit["institution_report"])

# COMMAND ----------

# Detail tables (for analysts — sample anomalous / mismatched rows)
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

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross-file merge checks (last)
# MAGIC
# MAGIC **What this proves:** coverage between roster, course, and semester extracts (same students and terms where expected).
# MAGIC Runs after table QA so empty or header-only extracts still load (pandas fallback in `from_csv_file`) before these joins.
# MAGIC
# MAGIC **`_merge` categories (pandas `indicator=True`, outer join):**
# MAGIC - **`both`:** key appears in left and right tables.
# MAGIC - **`left_only`:** key only in the left table (right file missing those rows).
# MAGIC - **`right_only`:** key only in the right table (left file missing those rows).
# MAGIC
# MAGIC `analyze_merge` prints row and distinct-student counts; pass `student_df=student_raw_df` so percentages use the roster denominator.

# COMMAND ----------

_id = "student_id"
if (
    len(student_raw_df) > 0
    and len(course_raw_df) > 0
    and _id in student_raw_df.columns
    and _id in course_raw_df.columns
):
    _ = analyze_merge(
        student_raw_df,
        course_raw_df,
        "student",
        "course",
        student_df=student_raw_df,
        merge_on="student_id",
        id_col="student_id",
    )
else:
    print(
        "Skip student × course merge: need non-empty tables and student_id on both sides."
    )

# COMMAND ----------

# student × semester: use inferred term columns; align names with a common join key when they differ
if (
    TERM_COL_STUDENT
    and TERM_COL_SEMESTER
    and _id in student_raw_df.columns
    and _id in semester_raw_df.columns
    and len(student_raw_df) > 0
    and len(semester_raw_df) > 0
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
        "Skip student × semester merge: need non-empty tables, student_id, and inferred term on both files, "
        "or use first_reg_date-style keys manually (rename columns and edit this cell)."
    )

# COMMAND ----------

# semester × course: inferred term columns; align names when they differ
if (
    TERM_COL_SEMESTER
    and TERM_COL_COURSE
    and _id in semester_raw_df.columns
    and _id in course_raw_df.columns
    and len(semester_raw_df) > 0
    and len(course_raw_df) > 0
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
        "Skip semester × course merge: need non-empty tables, student_id, and inferred term on both files."
    )
