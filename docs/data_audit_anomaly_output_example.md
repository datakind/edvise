# Data Audit Anomaly Output - Example

This document shows example output when the data audit logs anomalies with the percent of overall data each anomaly impacts.

---

## 1. `check_earned_vs_attempted` (credits earned vs attempted)

**Input**: DataFrame with 1000 rows, 15 rows have earned > attempted, 3 rows have credits earned when attempted = 0.

**Summary DataFrame** (returned in `result["summary"]`):

| earned_gt_attempted | earned_gt_attempted_pct | earned_when_no_attempt | earned_when_no_attempt_pct | total_anomalous_rows | total_anomalous_rows_pct |
|---------------------|-------------------------|------------------------|----------------------------|----------------------|--------------------------|
| 15                  | 1.5                     | 3                      | 0.3                        | 18                   | 1.8                      |

**Note**: Rows can have both anomaly types, so total_anomalous_rows may be less than the sum of individual counts.

---

## 2. `check_pf_grade_consistency` (pass/fail and grade alignment)

**Input**: DataFrame with 5000 course records, 42 rows have PF/grade inconsistencies.

**Summary DataFrame** (returned as second element of tuple):

| earned_with_failing_grade | earned_with_failing_grade_pct | no_credits_with_passing_grade | no_credits_with_passing_grade_pct | grade_pf_disagree | grade_pf_disagree_pct | total_anomalous_rows | total_anomalous_rows_pct |
|---------------------------|-------------------------------|-------------------------------|-----------------------------------|-------------------|-----------------------|----------------------|--------------------------|
| 12                        | 0.24                          | 25                            | 0.5                               | 5                 | 0.1                    | 42                   | 0.84                     |

**Logger output**:
```
WARNING - Detected 42 PF/grade consistency anomalies (0.84% of data)
```

---

## 3. `validate_credit_consistency` (course, semester, cohort checks)

### Course-level anomalies
**Input**: 2000 course rows, 8 have earned > attempted or negative credits.

**course_anomalies_summary** (in returned dict):
```python
{
    "rows_checked": 2000,
    "rows_with_anomalies": 8,
    "pct_of_data": 0.4
}
```

**Logger output**:
```
WARNING - Detected 8 course-level anomalies (0.40% of course data)
```

### Cohort-level anomalies
**Input**: 500 cohort rows, 2 have earned > attempted.

**cohort_anomalies_summary** (DataFrame from check_earned_vs_attempted):
- Includes `total_anomalous_rows_pct` = 0.4

**Logger output**:
```
WARNING - Detected 2 cohort-level anomalies (0.40% of cohort data)
```

### Semester reconciliation
**reconciliation_summary** (in returned dict when semester_df provided):
```python
{
    "total_semester_rows": 1500,
    "mismatched_rows": 22,
    "pct_of_data": 1.47
}
```

---

## 4. No anomalies case

When no anomalies are found, percent columns are 0:

**check_earned_vs_attempted** summary:
| earned_gt_attempted | earned_gt_attempted_pct | earned_when_no_attempt | earned_when_no_attempt_pct | total_anomalous_rows | total_anomalous_rows_pct |
|---------------------|-------------------------|------------------------|----------------------------|----------------------|--------------------------|
| 0                   | 0                       | 0                      | 0                          | 0                    | 0                        |

**Logger output**:
```
INFO - No course-level credit anomalies detected
INFO - No PF/grade consistency anomalies detected
```
