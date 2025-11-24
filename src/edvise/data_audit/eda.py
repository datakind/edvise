import itertools
import logging
import typing as t

import numpy as np
import pandas as pd
import scipy.stats as ss
from edvise import utils as edvise_utils

LOGGER = logging.getLogger(__name__)


def assess_unique_values(data: pd.DataFrame, cols: str | list[str]) -> dict[str, int]:
    """
    Assess unique values in ``data`` given by the combination of columns in ``cols`` ,
    including counts of nunique, duplicates, and nulls.

    Args:
        data
        cols
    """
    unique_data = data.loc[:, edvise_utils.types.to_list(cols)]
    is_duplicated = unique_data.duplicated()
    return {
        "num_uniques": is_duplicated.eq(False).sum(),
        "num_dupes": is_duplicated.sum(),
        "num_with_null_values": unique_data.isna().sum(axis="columns").gt(0).sum(),
    }


def compute_summary_stats(
    data: pd.DataFrame,
    *,
    include: t.Optional[str | list[str]] = None,
    exclude: t.Optional[str | list[str]] = None,
    percentiles: t.Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Compute summary stats for columns in ``data`` matching one or multiple dtypes
    using standard :meth:`pd.DataFrame.describe()` , supplemented with null count/pct.

    Args:
        data
        include: One or multiple dtypes whose columns will be included in result.
        exclude: One or multiple dtypes whose columns will be excluded from result.
        percentiles: Percentiles to include in result, given as floats between 0 and 1.

    References:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
    """
    num_rows = data.shape[0]
    include = edvise_utils.types.to_list(include) if include is not None else None
    exclude = edvise_utils.types.to_list(exclude) if exclude is not None else None
    data_selected = data.select_dtypes(include=include, exclude=exclude)  # type: ignore
    data_described = data_selected.describe(percentiles=percentiles).T.assign(
        null_count=data_selected.isna().sum(),
        null_pct=lambda df: (100 * df["null_count"] / num_rows).round(decimals=1),
    )
    return data_described


def compute_group_counts_pcts(
    data: pd.DataFrame,
    cols: str | list[str],
    *,
    sort: bool = True,
    ascending: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute value counts and percent totals in ``data`` for groups defined by ``cols`` .

    Args:
        data
        cols
        sort
        ascending
        dropna

    References:
        - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html
    """
    return (
        # data.value_counts(cols, dropna = False) for some reason still drops NaNs. We use
        # data[cols].value_counts() to avoid this error
        data[cols]
        .value_counts(sort=sort, ascending=ascending, dropna=dropna)
        .to_frame(name="count")
        .assign(
            pct=lambda df: (100 * df["count"] / df["count"].sum()).round(decimals=1)
        )
    )


def compute_crosstabs(
    data: pd.DataFrame,
    index_cols: str | list[str],
    column_cols: str | list[str],
    value_col: t.Optional[str] = None,
    aggfunc: t.Optional[t.Callable] = None,
    margins: bool = True,
    normalize: bool | t.Literal["all", "index", "columns"] = False,
) -> pd.DataFrame:
    """
    Args:
        data
        index_cols
        column_cols
        value_col
        aggfunc
        margins
        normalize

    References:
        - https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html
    """
    index = (
        data[index_cols]
        if isinstance(index_cols, str)
        else [data[col] for col in index_cols]
    )
    columns = (
        data[column_cols]
        if isinstance(column_cols, str)
        else [data[col] for col in column_cols]
    )
    values = None if value_col is None else data[value_col]
    ct = pd.crosstab(
        index,
        columns,
        values=values,  # type: ignore
        aggfunc=aggfunc,  # type: ignore
        margins=margins,
        normalize=normalize,
    )
    assert isinstance(ct, pd.DataFrame)  # type guard
    if normalize is not False:
        ct = ct.round(decimals=3)
    return ct


def compute_pairwise_associations(
    df: pd.DataFrame,
    *,
    ref_col: t.Optional[str] = None,
    exclude_cols: t.Optional[str | list[str]] = None,
) -> pd.DataFrame:
    """
    Compute pairwise associations between all columns and each other or, instead,
    all columns and a specified reference column.

    Per-pair association metrics depend on the data types of each:

        - nominal-nominal => Cramer's V
        - numeric-numeric => Spearman rank correlation
        - nominal-numeric => Correlation ratio

    Args:
        df
        ref_col: Reference column against which associations are to be computed.
            If None, all pairwise associations are computed.
        exclude_cols: One or multiple columns to exclude from computing associations;
            for example, if values are unique identifiers or all a single value,
            making their associations irrelevant.

    References:
        - https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
        - https://en.wikipedia.org/wiki/Correlation_ratio
    """
    # cast datetime columns to numeric, boolean to string
    df = df.assign(
        **{
            col: pd.to_numeric(df[col])
            for col in df.select_dtypes(include="datetime").columns
        }
        | {
            col: df[col].astype("string")
            for col in df.select_dtypes(include="boolean").columns
        }
    )
    # identify and organize columns in df
    if exclude_cols:
        df = df.drop(columns=exclude_cols)
    cols = df.columns.tolist()
    nominal_cols = set(
        df.select_dtypes(include=["category", "string", "boolean"]).columns.tolist()
    )
    numeric_cols = set(df.select_dtypes(include="number").columns.tolist())
    single_value_cols = _get_single_value_columns(df)
    # store col-col association values
    ref_cols = cols if ref_col is None else [ref_col]
    df_assoc = pd.DataFrame(index=cols, columns=ref_cols, dtype="Float32")
    for col1, col2 in itertools.product(cols, ref_cols):
        if not pd.isna(df_assoc.at[col1, col2]):
            continue

        is_symmetric = False
        if col1 == col2:  # self-association
            assoc = 1.0
        elif col1 in single_value_cols or col2 in single_value_cols:  # n/a
            assoc = None
        elif col1 in nominal_cols and col2 in nominal_cols:  # nom-nom
            assoc = _cramers_v(df[col1], df[col2])
            is_symmetric = True
        elif (col1 in nominal_cols and col2 in numeric_cols) or (
            col1 in numeric_cols and col2 in nominal_cols
        ):  # nom-num
            assoc = _correlation_ratio(df[col1], df[col2])
        elif col1 in numeric_cols and col2 in numeric_cols:  # num-num
            assoc = df[col1].corr(df[col2], method="spearman")
            is_symmetric = True
        else:
            LOGGER.warning(
                " ⚠️ '%s' and/or '%s' columns' dtypes (%s and/or %s) aren't supported "
                "for association computation; skipping ...",
                col1,
                col2,
                df[col1].dtype,
                df[col2].dtype,
            )
            assoc = None

        df_assoc.loc[col1, col2] = assoc
        if is_symmetric and len(ref_cols) > 1:
            df_assoc.loc[col2, col1] = assoc
        if assoc is not None:
            LOGGER.debug("%s – %s association = %s", col1, col2, assoc)
    return df_assoc


def _get_single_value_columns(df: pd.DataFrame) -> set[str]:
    sv_cols = []
    for col in df.columns:
        try:
            nunique = df[col].nunique()
        except TypeError:  # womp
            continue
        if nunique == 1:
            sv_cols.append(col)
    return set(sv_cols)


def _cramers_v(s1: pd.Series, s2: pd.Series) -> float | None:
    """
    Compute Cramer's V statistic for nominal-nominal association,
    which is symmetric -- i.e. V(x, y) == V(y, x).

    References:
        - https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

    See Also:
        - :func:`scipy.stats.contingency.association()`
    """
    if not pd.api.types.is_string_dtype(s1) or not pd.api.types.is_string_dtype(s2):
        raise ValueError()

    s1, s2 = _drop_incomplete_pairs(s1, s2)
    if s1.empty or s2.empty:
        return None

    confusion_matrix = pd.crosstab(s1, s2)
    correction = False if confusion_matrix.shape[0] == 2 else True
    try:
        result = ss.contingency.association(
            confusion_matrix, method="cramer", correction=correction
        )
        assert isinstance(result, float)
        return result
    except ValueError:
        return None


def _correlation_ratio(s1: pd.Series, s2: pd.Series) -> float | None:
    """
    Compute the Correlation Ratio for nominal-numeric association.

    References:
        - https://en.wikipedia.org/wiki/Correlation_ratio

    Note:
        ``s1`` and ``s2`` are automatically detected as being categorical or numeric,
        and handled correspondingly in the calculations.
    """
    if pd.api.types.is_string_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
        categories = s1
        measurements = s2
    elif pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_string_dtype(s2):
        categories = s2
        measurements = s1
    else:
        raise ValueError()

    categories, measurements = _drop_incomplete_pairs(categories, measurements)
    if categories.empty or measurements.empty:
        return None

    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def _drop_incomplete_pairs(s1: pd.Series, s2: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.DataFrame({"s1": s1, "s2": s2})
    df = df.dropna(axis="index", how="any", ignore_index=True)  # type: ignore
    return (df["s1"], df["s2"])


def log_high_null_columns(df: pd.DataFrame, threshold: float = 0.2) -> None:
    null_ratios = df.isna().mean(axis="index").sort_values(ascending=False)
    high_nulls = null_ratios[null_ratios > threshold]

    if high_nulls.empty:
        LOGGER.info(" No columns with more than %.0f%% null values.", threshold * 100)
    else:
        LOGGER.info(
            " Printing columns with >20% missing values to later be dropped during feature selection:"
        )
        for col, ratio in high_nulls.items():
            LOGGER.warning(
                ' Column "%s" has %.1f%% null values. ',
                col,
                ratio * 100,
            )


def compute_gateway_course_ids_and_cips(
    df_course: pd.DataFrame,
) -> tuple[list[str], list[str], bool, list[str], list[str]]:
    """
    Returns: (ids, cips, has_upper_level_gateway, lower_ids, lower_cips)
      - ids: all gateway course IDs (M/E)
      - cips: CIP 2-digit codes from LOWER-LEVEL rows only (same as lower_cips)
      - has_upper_level_gateway: True if any gateway course has level >=200
      - lower_ids: gateway IDs with level <200
      - lower_cips: CIP 2-digit codes for lower_ids
    """

    # ---- helpers ----
    def _s(x: pd.Series) -> pd.Series:
        """Normalize to string, strip, and remove literal 'nan' (categorical-safe)."""
        s = x.astype("string")  # cast before fillna to avoid Categorical fill errors
        s = s.fillna("")
        s = s.str.strip().replace("^nan$", "", regex=True)
        return s

    def _cip_series(x: pd.Series) -> list[str]:
        """
        Accept canonical CIP codes like '24', '24.02', '24.0201' and return unique 2-digit series (e.g., '24').
        Ignores placeholders and malformed values.
        """
        s = x.astype("string").str.strip().replace("^nan$", pd.NA, regex=True)
        # Strictly match valid CIP shapes and capture the 2-digit series as group 1
        series = (
            s.str.extract(r"^\s*(\d{2})(?:\.(\d{2})(?:\d{2})?)?\s*$", expand=True)[0]
            .dropna()
            .astype("string")
            .loc[lambda z: z.ne("")]
            .drop_duplicates()
            .tolist()
        )
        return list(series)

    def _last_level(num: pd.Series) -> pd.Series:
        """Parse last numeric token, then last up-to-3 digits as integer level."""
        tok = _s(num).str.extract(r"(\d+)(?!.*\d)", expand=True)[0]
        return pd.to_numeric(tok.str[-3:], errors="coerce")

    def _starts_with_any(arr: list[str], prefixes: list[str]) -> bool:
        arr = list(arr)  # handles numpy arrays / pandas .unique()
        return len(arr) > 0 and all(
            any(str(p).upper().startswith(ch) for ch in prefixes) for p in arr
        )

    # ---- column checks ----
    required = {"math_or_english_gateway", "course_prefix", "course_number"}
    if not required.issubset(df_course.columns):
        LOGGER.warning(" ⚠️ Cannot compute key_course_ids: required columns missing.")
        return ([], [], False, [], [])

    # ---- full-length masks ----
    gate = _s(df_course["math_or_english_gateway"])
    is_gateway = gate.isin({"M", "E"})  # full-length
    if not is_gateway.any():
        LOGGER.info(" No Math/English gateway courses found.")
        return ([], [], False, [], [])

    level = _last_level(df_course["course_number"])  # full-length
    upper_mask = is_gateway & level.ge(200).fillna(False)
    lower_mask = is_gateway & level.lt(200).fillna(False)
    has_upper_level_gateway = bool(upper_mask.any())

    # ---- IDs ----
    ids_series = (
        _s(df_course.loc[is_gateway, "course_prefix"])
        + _s(df_course.loc[is_gateway, "course_number"])
    ).str.strip()
    ids = ids_series[ids_series.ne("")].drop_duplicates().tolist()
    LOGGER.info(" Identified %d unique gateway course IDs: %s", len(ids), ids)

    lower_ids_series = (
        _s(df_course.loc[lower_mask, "course_prefix"])
        + _s(df_course.loc[lower_mask, "course_number"])
    ).str.strip()
    lower_ids = lower_ids_series[lower_ids_series.ne("")].drop_duplicates().tolist()
    LOGGER.info(" Identified %d lower-level (<200) gateway IDs.", len(lower_ids))

    # ---- CIP extraction from LOWER rows only ----
    if "course_cip" in df_course.columns:
        lower_cips = _cip_series(df_course.loc[lower_mask, "course_cip"])
        cips = lower_cips.copy()
        if not lower_cips:
            LOGGER.warning(
                " ⚠️ 'course_cip' present but yielded no lower-level CIP codes."
            )
        else:
            LOGGER.info(" CIPs restricted to lower-level (<200) rows: %s", cips)
    else:
        cips, lower_cips = [], []
        LOGGER.info(" No 'course_cip' column; skipping CIP extraction.")

    # ---- log upper-level anomalies (if any) ----
    if has_upper_level_gateway:
        upper_ids_series = (
            _s(df_course.loc[upper_mask, "course_prefix"])
            + _s(df_course.loc[upper_mask, "course_number"])
        ).str.strip()
        upper_ids = upper_ids_series[upper_ids_series.ne("")].drop_duplicates().tolist()
        LOGGER.warning(
            " ⚠️ Warning: courses with level >=200 flagged as gateway (%d found). Course IDs: %s. "
            "This is unusual; contact the school for more information.",
            len(upper_ids),
            upper_ids,
        )
        LOGGER.info(
            " ✅ Lower-level IDs found: %d; lower-level CIP codes found: %d",
            len(lower_ids),
            len(lower_cips),
        )
    else:
        LOGGER.info(" No gateway courses with level >=200 were detected.")

    # ---- prefix sanity check (compact) ----
    pref_e = (
        _s(df_course.loc[gate.eq("E"), "course_prefix"])
        .replace("", pd.NA)
        .dropna()
        .unique()
    )
    pref_m = (
        _s(df_course.loc[gate.eq("M"), "course_prefix"])
        .replace("", pd.NA)
        .dropna()
        .unique()
    )

    e_ok, m_ok = (
        _starts_with_any(pref_e, ["E", "W"]),
        _starts_with_any(pref_m, ["M", "S"]),
    )
    if e_ok and m_ok:
        LOGGER.info(" Prefix starts look correct (E/W for English, M/S for Math).")
    elif not e_ok and not m_ok:
        LOGGER.warning(
            " ⚠️ Prefixes MAY be swapped. Consider swapping E <-> M. E=%s, M=%s",
            list(pref_e),
            list(pref_m),
        )
    else:
        LOGGER.warning(
            " ⚠️ Prefixes MAY be incorrect; one group inconsistent. English OK=%s, Math OK=%s",
            e_ok,
            m_ok,
        )

    return ids, cips, has_upper_level_gateway, lower_ids, lower_cips


def log_record_drops(
    df_cohort_before: pd.DataFrame,
    df_cohort_after: pd.DataFrame,
    df_course_before: pd.DataFrame,
    df_course_after: pd.DataFrame,
) -> None:
    """
    Logs row counts before and after processing for cohort and course data.
    Also logs the number of dropped students and dropped course records.
    """
    cohort_before = len(df_cohort_before)
    cohort_after = len(df_cohort_after)
    cohort_dropped = cohort_before - cohort_after

    course_before = len(df_course_before)
    course_after = len(df_course_after)
    course_dropped = course_before - course_after

    LOGGER.info(
        " Cohort file: %d → %d rows (%d total students dropped) after preprocessing",
        cohort_before,
        cohort_after,
        cohort_dropped,
    )
    LOGGER.info(
        " Course file: %d → %d rows (%d total course records dropped) after preprocessing",
        course_before,
        course_after,
        course_dropped,
    )


def log_terms(df_course: pd.DataFrame, df_cohort: pd.DataFrame) -> None:
    """
    Logs ALL cohort year/term pairs and ALL academic year/term pairs,
    each sorted by year then term, including value counts.
    """

    # --- Cohort year/term pairs ---
    if {"cohort", "cohort_term"}.issubset(df_cohort.columns):
        cohort_terms_counts = (
            df_cohort[["cohort", "cohort_term"]]
            .dropna()
            .value_counts()
            .reset_index(name="count")
            .sort_values(by=["cohort", "cohort_term"])
        )
        LOGGER.info(
            "All cohort year/term pairs with counts:\n%s",
            cohort_terms_counts.to_string(index=False),
        )
    else:
        LOGGER.warning(
            " ⚠️ Missing fields: 'cohort' or 'cohort_term' in cohort dataframe."
        )

    # --- Academic year/term pairs ---
    if {"academic_year", "academic_term"}.issubset(df_course.columns):
        academic_terms_counts = (
            df_course[["academic_year", "academic_term"]]
            .dropna()
            .value_counts()
            .reset_index(name="count")
            .sort_values(by=["academic_year", "academic_term"])
        )
        LOGGER.info(
            "All academic year/term pairs with counts:\n%s",
            academic_terms_counts.to_string(index=False),
        )
    else:
        LOGGER.warning(
            " ⚠️ Missing fields: 'academic_year' or 'academic_term' in course dataframe."
        )


def log_misjoined_records(df_cohort: pd.DataFrame, df_course: pd.DataFrame) -> None:
    """
    Merges raw cohort and course data, identifies misjoined student records,
    and logs value counts for mismatches to help identify possible trends.

    Args:
        df_cohort (pd.DataFrame): Cohort-level student data
        df_course (pd.DataFrame): Course-level student data

    Returns:
        pd.DataFrame: Mismatched records with diagnostic columns.
    """
    # Merge with indicator
    df_merged = (
        pd.merge(
            df_cohort,
            df_course,
            on="study_id",
            how="outer",
            suffixes=("_cohort", "_course"),
            indicator=True,
        )
        .rename(
            columns={
                "cohort_cohort": "cohort",
                "cohort_term_cohort": "cohort_term",
                "student_age_cohort": "student_age",
                "race_cohort": "race",
                "ethnicity_cohort": "ethnicity",
                "gender_cohort": "gender",
                "institution_id_cohort": "institution_id",
            }
        )
        .drop(
            columns=[
                "cohort_course",
                "cohort_term_course",
                "student_age_course",
                "race_course",
                "ethnicity_course",
                "gender_course",
                "institution_id_course",
            ],
            errors="ignore",
        )
    )

    # Count merge results
    merge_counts = df_merged["_merge"].value_counts()
    left_only = merge_counts.get("left_only", 0)
    right_only = merge_counts.get("right_only", 0)
    both = merge_counts.get("both", 0)
    total = len(df_merged)
    total_misjoined = left_only + right_only
    pct_misjoined = (total_misjoined / total) * 100 if total else 0

    # Filter misjoined records only
    df_misjoined = df_merged[df_merged["_merge"] != "both"]

    # Log mismatch summary (custom format)
    if pct_misjoined < 0.1:
        pct_str = "<0.1%%"
    else:
        pct_str = f"{pct_misjoined:.1f}%%"

    LOGGER.warning(
        " ⚠️ inspect_misjoined_records: Found %d total misjoined records (%s of data): "
        "%d records in cohort file not found in course file, %d records in course file not found in cohort file.",
        total_misjoined,
        pct_str,
        left_only,
        right_only,
    )

    # Print misjoined ids
    misjoined_ids = df_misjoined["study_id"].dropna().unique().tolist()
    LOGGER.info(f" Misjoined student IDs: {misjoined_ids}")

    # Additional warning if mismatch is significant
    if total_misjoined > 100 or pct_misjoined > 10:
        LOGGER.warning(
            " ⚠️ inspect_misjoined_records: HIGH mismatch detected — %d records (%.1f%% of data). This is uncommon: please contact data team for further investigation.",
            total_misjoined,
            pct_misjoined,
        )

    # Log dropped student impact
    dropped_students = df_misjoined["study_id"].dropna().nunique()
    total_students = df_merged["study_id"].dropna().nunique()
    pct_dropped = (dropped_students / total_students) * 100 if total_students else 0

    # Log value counts of key fields
    for col in ["enrollment_type", "enrollment_intensity_first_term"]:
        if col in df_misjoined.columns:
            value_counts = df_misjoined[col].value_counts(dropna=False)
            LOGGER.info(
                " Value counts for mismatched records in column '%s' to identify potential trends:\n%s",
                col,
                value_counts.to_string(),
            )

    # Log grouped cohort & cohort_term
    if "cohort" in df_misjoined.columns and "cohort_term" in df_misjoined.columns:
        cohort_group_counts = (
            df_misjoined.groupby(["cohort", "cohort_term"], dropna=False, observed=True)
            .size()
            .sort_index()
        )
        LOGGER.info(
            " Grouped counts for mismatched records by cohort and cohort_term to identify potential trends:\n%s",
            cohort_group_counts.to_string(),
        )

    # Log grouped academic_year & academic_term
    if (
        "academic_year" in df_misjoined.columns
        and "academic_term" in df_misjoined.columns
    ):
        academic_group_counts = (
            df_misjoined.groupby(
                ["academic_year", "academic_term"], dropna=False, observed=True
            )
            .size()
            .sort_index()
        )
        LOGGER.info(
            " Grouped counts for mismatched records by academic_year and academic_term to identify potential trends:\n%s",
            academic_group_counts.to_string(),
        )

    if pct_dropped < 0.1:
        LOGGER.warning(
            " ⚠️ inspect_misjoined_records: These mismatches will later result in dropping %d students (<0.1%% of all students).",
            dropped_students,
        )
    else:
        LOGGER.warning(
            " ⚠️ inspect_misjoined_records: These mismatches will later result in dropping %d students (%.1f%% of all students).",
            dropped_students,
            pct_dropped,
        )


def print_credential_types_and_retention(df_cohort: pd.DataFrame) -> None:
    pct_credentials = (
        df_cohort["credential_type_sought_year_1"].value_counts(
            dropna=False, normalize=True
        )
        * 100
    )
    retention = (
        df_cohort[["cohort", "retention"]].value_counts(dropna=False).sort_index()
    )
    LOGGER.warning(
        "  ⚠️ Breakdown for retention by cohort: IF MOST RECENT YEAR'S SPLIT IS DISPROPORTIONATE, exclude from training by changing max_academic_year in the config! \n%s ",
        retention.to_string(),
    )
    LOGGER.info(
        " Percent breakdown for credential types: \n%s ",
        pct_credentials.to_string(),
    )


def validate_ids_terms_consistency(
    student_df: t.Optional[pd.DataFrame],
    semester_df: pd.DataFrame,
    course_df: pd.DataFrame,
    *,
    id_col: str = "student_id",
    sem_col: str = "semester_code",
    student_id_col: t.Optional[str] = None,
) -> t.Dict[str, t.Any]:
    """
    Check key-level and ID-level consistency between Course, Semester and Student.

    Returns:
    {
        "summary": {....},
        "unmatched_course_side": DataFrame[(id, sem)],
        "unmatched_semester_side": DataFrame[(id, sem)],
        "course_ids_not_in_semester": DataFrame[id],
        "course_ids_not_in_student": DataFrame[id],
        "semester_ids_not_in_student": DataFrame[id],
        "course_terms_not_in_semester_terms": DataFrame[sem],
        "null_course_keys": DataFrame[(id, sem)],
        "null_semester_keys": DataFrame[(id, sem)],
    }
    """
    student_id_col = student_id_col or id_col
    c_keys = course_df[[id_col, sem_col]].copy()
    s_keys = semester_df[[id_col, sem_col]].copy()

    st = None
    if student_df is not None:
        st = student_df[[student_id_col]].drop_duplicates().copy()
        if student_id_col != id_col:
            st = st.rename(columns={student_id_col: id_col})

    null_course_keys = course_df.loc[
        course_df[id_col].isna() | course_df[sem_col].isna(),
        [id_col, sem_col],
    ].drop_duplicates()

    null_semester_keys = semester_df.loc[
        semester_df[id_col].isna() | semester_df[sem_col].isna(),
        [id_col, sem_col],
    ].drop_duplicates()

    course_keys = c_keys.drop_duplicates()
    sem_keys = s_keys.drop_duplicates()

    unmatched_course_side = (
        course_keys.merge(sem_keys, on=[id_col, sem_col], how="left", indicator=True)
        .loc[lambda df: df["_merge"].eq("left_only"), [id_col, sem_col]]
        .sort_values([id_col, sem_col])
        .reset_index(drop=True)
    )

    unmatched_semester_side = (
        sem_keys.merge(course_keys, on=[id_col, sem_col], how="left", indicator=True)
        .loc[lambda df: df["_merge"].eq("left_only"), [id_col, sem_col]]
        .sort_values([id_col, sem_col])
        .reset_index(drop=True)
    )

    course_ids = course_df[[id_col]].dropna().drop_duplicates()
    semester_ids = semester_df[[id_col]].dropna().drop_duplicates()

    course_ids_not_in_semester = (
        course_ids.merge(semester_ids, on=id_col, how="left", indicator=True)
        .loc[lambda df: df["_merge"].eq("left_only"), [id_col]]
        .sort_values(id_col)
        .reset_index(drop=True)
    )

    if st is not None:
        course_ids_not_in_student = (
            course_ids.merge(st, on=id_col, how="left", indicator=True)
            .loc[lambda df: df["_merge"].eq("left_only"), [id_col]]
            .sort_values(id_col)
            .reset_index(drop=True)
        )
        semester_ids_not_in_student = (
            semester_ids.merge(st, on=id_col, how="left", indicator=True)
            .loc[lambda df: df["_merge"].eq("left_only"), [id_col]]
            .sort_values(id_col)
            .reset_index(drop=True)
        )
    else:
        course_ids_not_in_student = pd.DataFrame(columns=[id_col])
        semester_ids_not_in_student = pd.DataFrame(columns=[id_col])

    course_terms = set(course_df[sem_col].dropna().unique())
    semester_terms = set(semester_df[sem_col].dropna().unique())
    course_terms_missing = sorted(course_terms - semester_terms)
    course_terms_not_in_semester_terms = pd.DataFrame({sem_col: course_terms_missing})

    summary = {
        "total_semesters_in_semester_file": int(len(semester_df)),
        "unique_student_semesters_in_courses": int(len(course_keys)),
        "unmatched_course_keys": int(len(unmatched_course_side)),
        "unmatched_semester_keys": int(len(unmatched_semester_side)),
        "course_ids_not_in_semester": int(len(course_ids_not_in_semester)),
        "course_ids_not_in_student": int(len(course_ids_not_in_student)),
        "semester_ids_not_in_student": int(len(semester_ids_not_in_student)),
        "course_terms_not_in_semester_terms": int(
            len(course_terms_not_in_semester_terms)
        ),
        "course_rows_with_null_keys": int(len(null_course_keys)),
        "semester_rows_with_null_keys": int(len(null_semester_keys)),
    }

    return {
        "summary": summary,
        "unmatched_course_side": unmatched_course_side,
        "unmatched_semester_side": unmatched_semester_side,
        "course_ids_not_in_semester": course_ids_not_in_semester,
        "course_ids_not_in_student": course_ids_not_in_student,
        "semester_ids_not_in_student": semester_ids_not_in_student,
        "course_terms_not_in_semester_terms": course_terms_not_in_semester_terms,
        "null_course_keys": null_course_keys,
        "null_semester_keys": null_semester_keys,
    }


def validate_credit_consistency(
    semester_df: pd.DataFrame,
    course_df: pd.DataFrame,
    *,
    id_col: str = "student_id",
    sem_col: str = "semester_code",
    course_credits_attempted_col: str = "credits_attempted",
    course_credits_earned_col: str = "credits_earned",
    semester_credits_attempted_col: str = "number_of_semester_credits_attempted",
    semester_credits_earned_col: str = "number_of_semester_credits_earned",
    semester_courses_count_col: str = "number_of_semester_courses_enrolled",
    credit_tol: float = 0.0,
) -> t.Dict[str, t.Any]:
    """
    Reconcile semester-level aggregates against course-level details.
    """

    c = course_df[
        [id_col, sem_col, course_credits_attempted_col, course_credits_earned_col]
    ].copy()
    s = semester_df[
        [
            id_col,
            sem_col,
            semester_credits_attempted_col,
            semester_credits_earned_col,
            semester_courses_count_col,
        ]
    ].copy()

    agg = (
        c.groupby([id_col, sem_col], dropna=False)
        .agg(
            course_sum_attempted=(course_credits_attempted_col, "sum"),
            course_sum_earned=(course_credits_earned_col, "sum"),
            course_count=(course_credits_attempted_col, "size"),
        )
        .reset_index()
    )

    merged = s.merge(agg, on=[id_col, sem_col], how="left", indicator="_merge_agg")
    merged["has_course_rows"] = merged["_merge_agg"].eq("both")
    merged["course_sum_attempted"] = merged["course_sum_attempted"].fillna(0.0)
    merged["course_sum_earned"] = merged["course_sum_earned"].fillna(0.0)
    merged["course_count"] = merged["course_count"].fillna(0.0)

    for col in (
        semester_credits_attempted_col,
        semester_credits_earned_col,
        semester_courses_count_col,
    ):
        if not pd.api.types.is_numeric_dtype(merged[col]):
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["diff_attempted"] = (
        merged["course_sum_attempted"] - merged[semester_credits_attempted_col]
    )
    merged["diff_earned"] = (
        merged["course_sum_earned"] - merged[semester_credits_earned_col]
    )
    merged["diff_courses"] = merged["course_count"] - merged[
        semester_courses_count_col
    ].fillna(0.0)

    merged["match_attempted"] = merged["diff_attempted"].abs() <= credit_tol
    merged["match_earned"] = merged["diff_earned"].abs() <= credit_tol
    merged["match_courses"] = merged["diff_courses"] == 0.0

    mismatches = (
        merged.loc[
            ~(
                merged["match_attempted"]
                & merged["match_earned"]
                & merged["match_courses"]
            ),
            [
                id_col,
                sem_col,
                semester_credits_attempted_col,
                "course_sum_attempted",
                "diff_attempted",
                semester_credits_earned_col,
                "course_sum_earned",
                "diff_earned",
                semester_courses_count_col,
                "course_count",
                "diff_courses",
                "has_course_rows",
            ],
        ]
        .sort_values([id_col, sem_col])
        .reset_index(drop=True)
    )

    summary = {
        "total_semesters_in_semester_file": int(len(s)),
        "unique_student_semesters_in_courses": int(len(agg)),
        "rows_with_mismatches": int(len(mismatches)),
    }

    return {
        "summary": summary,
        "mismatches": mismatches,
        "merged_detail": merged,
    }

def find_dupes(df, primary_keys, sort=None, summarize=False, n=20):
    """
    Quickly find and summarize duplicates by primary key columns for each dataset (cohort, course, and semester).
    """
    if summarize:
        out = df.groupby(primary_keys).size().value_counts().rename_axis('dup_count').reset_index(name='n_groups')
        print(out.head(10))
        return out
    dupes = df[df.duplicated(primary_keys, keep=False)]
    if sort: dupes = dupes.sort_values(sort, ignore_index=True)
    print(f"{len(dupes)} duplicates based on {primary_keys}")
    return dupes

def check_credit_earned_attempted_consistency(df, earned_col="inst_tot_credits_earned", attempted_col="inst_tot_credits_attempted"):
    """
    Checks that:
      1. Credits earned <= credits attempted
      2. Credits earned = 0 when credits attempted = 0
    Returns (anomalies_df, summary_df)
    """
    # Ensure numeric columns
    earned = pd.to_numeric(df[earned_col], errors="coerce")
    attempted = pd.to_numeric(df[attempted_col], errors="coerce")

    # Rules
    earned_gt_attempted = earned > attempted
    earned_when_no_attempt = (attempted == 0) & (earned > 0)

    # Combine anomalies
    anomalies = df[earned_gt_attempted | earned_when_no_attempt].copy()
    anomalies["earned_gt_attempted"] = earned_gt_attempted
    anomalies["earned_when_no_attempt"] = earned_when_no_attempt

    # Summary
    summary = pd.DataFrame({
        "earned_gt_attempted": [earned_gt_attempted.sum()],
        "earned_when_no_attempt": [earned_when_no_attempt.sum()],
        "total_anomalous_rows": [(earned_gt_attempted | earned_when_no_attempt).sum()]
    })

    return anomalies, summary

def check_pf_grade_consistency(
    df,
    grade_col="grade",
    pf_col="pass_fail_flag",
    credits_col="credits_earned",
    *,
    passing_grades=("P","P*","A","A-","B+","B","B-","C+","C","C-","D+","D","D-","REP","S", "^C-","^D-","^D","^D+","ZD-","ZD","^C","CH"),
    failing_grades=("F","E","^E","F*","REF","ZE","NR", "W","W*","WI","WE","WC","WA","WB+","WB","WB-","WD","WD-","WC+","WC-","WA-", "I"),
    pass_flags=("P",),
    fail_flags=("F",),
):
    """
    Checks that:
      1. Students NEVER earn credits for failing grades.
      2. Students DO always earn credits for passing grades.
      3. Grade and pass_fail_flag are consistent.

    Returns (anomalies_df, summary_df)
    """
    out = df.copy()

    # Normalize
    g  = out[grade_col].astype(str).str.strip().str.upper()
    pf = out[pf_col].astype(str).str.strip().str.upper()
    credits = pd.to_numeric(out[credits_col], errors="coerce")  # keep NaNs as NaN

    # Pass/fail from grade (for disagreement only)
    pfg = pd.Series(
        np.where(g.isin(passing_grades), True,
        np.where(g.isin(failing_grades), False, np.nan)),
        index=out.index, dtype="object"
    )

    # Pass/fail from flag (drives credit rules)
    pff = pd.Series(
        np.where(pf.isin(pass_flags), True,
        np.where(pf.isin(fail_flags), False, np.nan)),
        index=out.index, dtype="object"
    )

    # Credit rules (PF-based only)
    earned_with_failing_pf   = (pff == False) & credits.notna() & (credits > 0)
    no_credits_with_passing  = (pff == True)  & credits.notna() & (credits == 0)

    # Grade vs PF disagreement (only where both known)
    grade_pf_disagree = pfg.notna() & pff.notna() & (pfg != pff)

    # Collect anomalies
    mask = earned_with_failing_pf | no_credits_with_passing | grade_pf_disagree
    anomalies = out.loc[mask].copy()
    anomalies["earned_with_failing_grade"] = earned_with_failing_pf.loc[anomalies.index]
    anomalies["no_credits_with_passing_grade"] = no_credits_with_passing.loc[anomalies.index]
    anomalies["grade_pf_disagree"] = grade_pf_disagree.loc[anomalies.index]

    # Summary
    summary = pd.DataFrame({
        "earned_with_failing_grade": [int(earned_with_failing_pf.sum())],
        "no_credits_with_passing_grade": [int(no_credits_with_passing.sum())],
        "grade_pf_disagree": [int(grade_pf_disagree.sum())],
        "total_anomalous_rows": [int(mask.sum())],
    })

    return anomalies, summary