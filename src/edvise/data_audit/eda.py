import itertools
import logging
import typing as t

import numpy as np
import pandas as pd
import scipy.stats as ss
from functools import cached_property, wraps
from edvise import utils as edvise_utils
from edvise.shared.utils import as_percent, percent_of_rows, validate_optional_column

LOGGER = logging.getLogger(__name__)

DEFAULT_BIAS_VARS = ["first_gen", "gender", "race", "ethnicity", "student_age"]


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


def value_counts_sorted_count_df(
    series: pd.Series,
    *,
    count_col: str = "count",
) -> pd.DataFrame:
    """
    ``value_counts`` sorted by index (e.g. term order after ``order_terms``), as a two-column frame.

    The first column is named from ``series.name`` when set, otherwise ``value``.
    """
    vc = series.value_counts(sort=False).sort_index()
    out = vc.reset_index()
    val_name = series.name if series.name is not None else "value"
    out.columns = [val_name, count_col]
    return out


def value_counts_percent_df(
    series: pd.Series,
    *,
    dropna: bool = False,
    sort_index: bool = True,
    pct_col: str = "pct_of_rows",
    pct_round: int = 2,
) -> pd.DataFrame:
    """
    Normalized value counts as percentages (0–100), for audit tables and notebooks.

    The first column is named from ``series.name`` when set, otherwise ``value``.
    """
    vc = series.value_counts(dropna=dropna, normalize=True).mul(100).round(pct_round)
    if sort_index:
        vc = vc.sort_index()
    out = vc.reset_index()
    val_name = series.name if series.name is not None else "value"
    out.columns = [val_name, pct_col]
    return out


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
            " ⚠️ inspect_misjoined_records: These mismatches will later result in dropping %d students (<0.1 percent of all students).",
            dropped_students,
        )
    else:
        LOGGER.warning(
            " ⚠️ inspect_misjoined_records: These mismatches will later result in dropping %d students (%.1f%% of all students).",
            dropped_students,
            pct_dropped,
        )


def pct_breakdown(series: pd.Series) -> pd.Series:
    return series.value_counts(dropna=False, normalize=True).map(as_percent)


def print_credential_and_enrollment_types_and_intensities(
    df_cohort: pd.DataFrame,
) -> None:
    pct_credentials = pct_breakdown(df_cohort["credential_type_sought_year_1"])

    pct_enroll_types = pct_breakdown(df_cohort["enrollment_type"])

    pct_enroll_intensity = pct_breakdown(df_cohort["enrollment_intensity_first_term"])

    LOGGER.info(
        "Percent breakdown for credential types:\n%s",
        pct_credentials.to_string(),
    )
    LOGGER.info(
        "Percent breakdown for enrollment types:\n%s",
        pct_enroll_types.to_string(),
    )
    LOGGER.info(
        "Percent breakdown for enrollment intensities:\n%s",
        pct_enroll_intensity.to_string(),
    )


def print_retention(df_cohort: pd.DataFrame) -> None:
    retention = df_cohort.groupby("cohort")["retention"].apply(pct_breakdown)

    LOGGER.warning(
        "⚠️ Breakdown for retention by cohort: "
        "IF MOST RECENT YEAR'S SPLIT IS DISPROPORTIONATE, "
        "exclude from training by changing max_academic_year in the config!\n%s",
        retention.to_string(),
    )


def log_top_majors(df_cohort: pd.DataFrame) -> None:
    """
    Logs the top majors by program of study for the first term.
    """
    top_majors = (
        df_cohort["program_of_study_term_1"]
        .value_counts(dropna=False)
        .sort_values(ascending=False)
        .head(10)
    )
    LOGGER.info(
        " Top majors: \n%s ",
        top_majors.to_string(),
    )


def log_grade_distribution(df_course: pd.DataFrame, grade_col: str = "grade") -> None:
    """
    Logs value counts of the 'grade' column and flags if 'M' grades exceed 5%.
    Also flags when grades contain only status codes (P, F, I, W, A, M, O) and no numeric grades.

    Args:
        df (pd.DataFrame): The course or student dataset.
        grade_col (str): Name of the grade column to analyze (default is "grade").

    Logs:
        - Value counts for all grades
        - Percentage of 'M' grades
        - Warning if 'M' grades exceed 5% of all non-null grades
        - Warning if no numeric grades exist (only status-only grades like P, F, I, W, A, M, O)
    """
    # Status-only grades: P=Pass, F=Fail, I=Incomplete, W=Withdraw, A=Audit, M=Missing, O=Other.
    status_only_grades = frozenset({"P", "F", "I", "W", "A", "M", "O"})
    gpa_letter_grades = frozenset(
        {"A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-"}
    )

    def grade_is_numeric(val: t.Any) -> bool:
        """True if grade has a numeric GPA equivalent (float or GPA letter)."""
        if pd.isna(val):
            return False
        s = str(val).strip().upper()
        if not s:
            return False
        if s in status_only_grades:
            return False
        coerced = pd.to_numeric(s, errors="coerce")
        if not pd.isna(coerced):
            return True
        if s in gpa_letter_grades:
            return True
        return False

    resolved_grade_col = validate_optional_column(
        df_course, grade_col, "grade", logger=LOGGER
    )
    if resolved_grade_col is None:
        return

    grade_counts = df_course[resolved_grade_col].value_counts(dropna=False).sort_index()
    total_grades = df_course[resolved_grade_col].notna().sum()

    LOGGER.info("Grade value counts:\n%s", grade_counts.to_string())

    if "M" in grade_counts.index and total_grades > 0:
        m_count = grade_counts["M"]
        m_pct = (m_count / total_grades) * 100

        if m_pct > 5:
            LOGGER.warning(
                "High proportion of 'M' grades: %d (%.1f%% of non-null grades). Consider reaching out to schoool for additional information.",
                m_count,
                m_pct,
            )
        else:
            LOGGER.info("'M' grades: %d (%.1f%% of non-null grades).", m_count, m_pct)
    else:
        LOGGER.info("'M' grade not found or no valid grade data available.")

    if total_grades > 0:
        grades_series = (
            df_course[resolved_grade_col].astype("string").str.strip().str.upper()
        )
        any_numeric = grades_series.apply(grade_is_numeric).any()
        if not any_numeric:
            unique_vals = sorted(grades_series.dropna().unique())
            LOGGER.warning(
                "No numeric grades detected. Grades are only status codes (e.g. P=Pass, F=Fail, "
                "I=Incomplete, W=Withdraw, A=Audit, M=Missing, O=Other). Unique values: %s. "
                "Analytics that depend on numeric course grades (e.g. GPA, mean grade) will be unusable.",
                unique_vals,
            )


def check_variable_missingness(
    df: pd.DataFrame,
    var_list: list[str],
    null_threshold_pct: float = 50.0,
) -> None:
    """
    Log missingness diagnostics for variables.

    For each variable in `var_list`, this function:
    - Verifies the column exists in the DataFrame.
    - Logs the percentage distribution of all values, including NaNs.
    - Flags variables whose percentage of missing values meets or exceeds
      the specified null threshold.

    This function is intended for exploratory data validation and bias auditing.
    It does not modify the input DataFrame and does not return any values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing bias-related variables to be checked.
    var_list : list of str, optional
        List of column names to inspect for value distribution and missingness.
    null_threshold_pct : float, optional
        Percentage threshold for missing values at or above which a warning
        is logged. Default is 50.0.

    Returns
    -------
    None
        All results are reported via logging.
    """

    LOGGER.info(" Missing Variable Check: ")

    for var in var_list:
        if var not in df.columns:
            LOGGER.warning(f"\n⚠️  MISSING COLUMN: '{var}' not found in DataFrame")
            continue

        LOGGER.info(f"\n--- {var} ---")

        pct_counts = (
            df[var].value_counts(dropna=False, normalize=True).mul(100).round(2)
        )

        null_pct = 0.0

        for value, pct in pct_counts.items():
            if pd.isna(value):
                label = "NaN"
                null_pct = pct
            else:
                label = value
            LOGGER.info(f"{label}: {pct}%")

        if null_pct >= null_threshold_pct:
            LOGGER.warning(
                f"⚠️  NOTE: >={null_threshold_pct}% missingness in '{var}' "
                f"({null_pct}% nulls; threshold = {null_threshold_pct}%)"
            )


def check_bias_variables(
    df: pd.DataFrame,
    bias_vars: list[str] | None = None,
) -> None:
    if bias_vars is None:
        bias_vars = DEFAULT_BIAS_VARS
    LOGGER.info("Check Bias Variables Missingness")
    check_variable_missingness(df, bias_vars)


def analyze_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_name: str,
    right_name: str,
    *,
    student_df: pd.DataFrame | None = None,
    merge_on: str | list[str] = "student_id",
    id_col: str = "student_id",
) -> pd.DataFrame:
    """
    Outer-merge two tables on ``merge_on`` and print join coverage and dimension breakdowns.

    Pass the full ``student_df`` roster so percentages use a stable denominator and so
    course-only rows can be cross-checked against the student file.

    When ``left_name`` / ``right_name`` is ``\"course\"`` or ``\"semester\"``, reference
    row and ``id_col`` counts are taken from the corresponding merge operand.
    """
    merged = left_df.merge(right_df, on=merge_on, indicator=True, how="outer")
    counts = merged["_merge"].value_counts(dropna=False)

    if id_col not in merged.columns:
        raise KeyError(
            f"analyze_merge: id_col {id_col!r} not in merged columns "
            f"(merge_on={merge_on!r}); set id_col to the student id column."
        )

    both_ids = int(merged.loc[merged["_merge"] == "both", id_col].nunique())
    left_only_ids = int(merged.loc[merged["_merge"] == "left_only", id_col].nunique())
    right_only_ids = int(merged.loc[merged["_merge"] == "right_only", id_col].nunique())

    if student_df is not None and id_col in student_df.columns:
        total_ids = int(student_df[id_col].nunique())
    else:
        total_ids = int(
            pd.Index(left_df[id_col].dropna().unique())
            .union(right_df[id_col].dropna().unique())
            .size
        )

    both_rows = counts.get("both", 0)
    left_only_rows = counts.get("left_only", 0)
    right_only_rows = counts.get("right_only", 0)

    def pct(n: int) -> str:
        return f"{n / total_ids:.1%} of roster" if total_ids else "n/a"

    print(f"{'=' * 50}")
    print(f"  {left_name}  x  {right_name}")
    print(f"{'=' * 50}")
    if "course" in (left_name, right_name):
        ref = left_df if left_name == "course" else right_df
        if id_col in ref.columns:
            print(
                f"  (reference course table: {len(ref):,} rows, "
                f"{ref[id_col].nunique():,} unique {id_col})"
            )
    if "semester" in (left_name, right_name):
        ref = left_df if left_name == "semester" else right_df
        if id_col in ref.columns:
            print(
                f"  (reference semester table: {len(ref):,} rows, "
                f"{ref[id_col].nunique():,} unique {id_col})"
            )
    print(
        f"Shared unique student IDs:        {both_ids:>6} ({pct(both_ids)}) | {both_rows:>6} rows"
    )
    print(
        f"Missing from {right_name:<20} {left_only_ids:>6} ({pct(left_only_ids)}) | {left_only_rows:>6} rows"
    )
    print(
        f"Missing from {left_name:<20} {right_only_ids:>6} ({pct(right_only_ids)}) | {right_only_rows:>6} rows"
    )

    def print_breakdown(title, frame, cols, normalize=True):
        n_ids = frame[id_col].nunique()
        print(f"\n[{title} (n={len(frame)} rows, {n_ids} unique {id_col})]")
        for col in cols:
            if col in frame.columns:
                print(
                    f"\n  {col}:\n{frame[col].value_counts(dropna=False, normalize=normalize).to_string()}"
                )

    # --- course: Class Grades + course dims + roster cross-check (left or right) ---
    if "course" in (left_name, right_name):
        course_side = "right_only" if right_name == "course" else "left_only"
        missing = merged[merged["_merge"] == course_side]
        n_ids = missing[id_col].nunique()
        label = f"{course_side} (rows only in {right_name if course_side == 'right_only' else left_name})"
        print(f"[Class Grade nulls in {label}]")
        if "Class Grade" in missing.columns:
            print(f"  Null Class Grades:  {missing['Class Grade'].isna().sum()}")
            print(f"  Total rows:   {len(missing)} ({n_ids} unique student IDs)")
            print(f"  Pct null:     {missing['Class Grade'].isna().mean():.1%}")
        else:
            print("  (no 'Class Grade' column on this side)")
        print_breakdown(
            f"Course dimensions ({course_side})",
            missing,
            [
                "course_classification",
                "department",
                "course_delivery_method_online_hybrid_in_person",
            ],
        )

        only_ids_list = missing[id_col].unique()
        if student_df is not None and id_col in student_df.columns:
            matched = student_df[student_df[id_col].isin(only_ids_list)]
            in_roster = matched[id_col].nunique()
            not_in_roster = len(set(only_ids_list) - set(matched[id_col]))
            print(f"[{course_side} IDs vs student roster]")
            print(f"  Found in student file:  {in_roster}")
            print(f"  Not in student file:    {not_in_roster}")
            if not matched.empty:
                print_breakdown(
                    f"heh_type_desc (students with {course_side} course rows)",
                    matched,
                    ["heh_type_desc"],
                )
        else:
            print(
                f"[{course_side} IDs vs student roster] (skipped: pass student_df for cross-check)"
            )

    # --- student file: missing side gets heh + first enrollment ---
    if "student" in (left_name, right_name):
        side = "left_only" if left_name == "student" else "right_only"
        print_breakdown(
            f"Student file gaps ({side})",
            merged[merged["_merge"] == side],
            ["heh_type_desc", "first_enrollment_date"],
        )

    # --- semester file: missing side gets semester + enrollment_intensity together ---
    if "semester" in (left_name, right_name):
        side = "left_only" if left_name == "semester" else "right_only"
        print_breakdown(
            f"Semester file gaps ({side})",
            merged[merged["_merge"] == side],
            ["semester", "enrollment_intensity"],
        )

    print()
    return merged


class EdaSummary:
    """
    Provides summary statistics and analysis for student cohort and course data.

    This class encapsulates EDA (Exploratory Data Analysis) calculations that can be
    used across multiple contexts: dashboards, reports, and API endpoints.

    Args:
        df_cohort: DataFrame containing cohort/student-level data
        df_course: Optional DataFrame containing course-level data
    """

    @staticmethod
    def required_columns(
        *,
        cohort: list[str] | None = None,
        course: list[str] | None = None,
    ) -> t.Callable[[t.Callable[..., t.Any]], t.Callable[..., t.Any]]:
        """
        Decorator for EdaSummary methods that require specific columns.

        Logs a warning and returns None if required columns are missing.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                required_map = {
                    "cohort": (cohort or [], "df_cohort"),
                    "course": (course or [], "df_course"),
                }

                for label, (cols, df_attr) in required_map.items():
                    if not cols:
                        continue
                    df = getattr(self, df_attr, None)
                    if df is None:
                        LOGGER.warning(
                            "%s: could not compute because %s is missing",
                            func.__name__,
                            df_attr,
                        )
                        return None
                    missing = [c for c in cols if c not in df.columns]
                    if missing:
                        LOGGER.warning(
                            "%s: could not compute because missing %s columns: %s",
                            func.__name__,
                            label,
                            missing,
                        )
                        return None

                return func(self, *args, **kwargs)

            return wrapper

        return decorator

    def __init__(
        self,
        df_cohort: pd.DataFrame,
        df_course: pd.DataFrame | None = None,
    ):
        """
        Initialize EdaSummary with cohort and course data.

        Args:
            df_cohort: DataFrame containing cohort/student data with columns like
                'study_id', 'enrollment_type', 'gpa_group_year_1', etc.
            df_course: Optional DataFrame containing course data
        """
        self.df_cohort = df_cohort
        self.df_course = df_course

    def cohort_years(self, formatted: bool = True) -> list[str]:
        """
        Get the unique cohort years from the cohort data, sorted.
        If formatted is True, return the years as "YYYY - YYYY" format.
        Otherwise, return the years as "YYYY-YYYY" format.
        """

        years = (
            self.df_cohort["cohort"]
            .dropna()
            .astype(str)
            .sort_values()
            .drop_duplicates()
        )
        if formatted:
            years = years.str.replace("-", " - ", regex=False)

        return t.cast(list[str], years.tolist())

    def _format_series_data(self, df: pd.DataFrame) -> list[dict[str, t.Any]]:
        result = (
            df.reset_index(names="name")
            .assign(
                data=lambda d: [
                    [None if pd.isna(x) else round(float(x), 2) for x in row]
                    for row in d.drop(columns="name").to_numpy()
                ]
            )
            .loc[:, ["name", "data"]]
            .to_dict(orient="records")
        )
        return t.cast(list[dict[str, t.Any]], result)

    @cached_property
    @required_columns(cohort=["student_id"])
    def total_students(self) -> dict[str, t.Any]:
        """
        Total number of cohort records (rows in the cohort DataFrame).
        """
        return {
            "name": "Total Students",
            "value": int(self.df_cohort["student_id"].nunique()),
        }

    @cached_property
    @required_columns(cohort=["enrollment_type"])
    def transfer_students(self) -> dict[str, t.Any] | None:
        """
        Compute the number of transfer students.
        Returns None if there are no transfer students.
        """
        n = int((self.df_cohort["enrollment_type"] == "TRANSFER-IN").sum())
        return {"name": "Transfer Students", "value": n} if n else None

    @cached_property
    @required_columns(cohort=["gpa_group_year_1"])
    def avg_year1_gpa_all_students(self) -> dict[str, t.Any]:
        """
        Compute the average GPA for all students.
        """
        return {
            "name": "Avg. Year 1 GPA - All Students",
            "value": round(
                float(
                    pd.to_numeric(
                        self.df_cohort["gpa_group_year_1"], errors="coerce"
                    ).mean()
                ),
                2,
            ),
        }

    @cached_property
    @required_columns(cohort=["enrollment_type", "gpa_group_year_1"])
    def gpa_by_enrollment_type(self) -> dict[str, list | float | None]:
        """
        Compute GPA by enrollment type across cohort years.

        Returns:
            Dictionary with:
                - cohort_years: List of cohort year strings
                - series: List of dicts with 'name' and 'data' keys
        """

        gpa_df = (
            self.df_cohort.assign(
                gpa=pd.to_numeric(self.df_cohort["gpa_group_year_1"], errors="coerce"),
                enrollment_type=self.df_cohort["enrollment_type"]
                .astype(str)
                .str.strip()
                .str.title(),
            )
            .loc[
                lambda d: (
                    d["enrollment_type"].isin(["First-Time", "Re-Admit", "Transfer-In"])
                    & d["gpa"].notna()
                )
            ][["cohort", "enrollment_type", "gpa"]]
            .groupby(["enrollment_type", "cohort"], observed=True)["gpa"]
            .mean()
            .unstack()
            .reindex(columns=self.cohort_years(formatted=False))
        )

        series_data = self._format_series_data(gpa_df)

        return {
            "cohort_years": self.cohort_years(formatted=True),
            "series": series_data,
            "min_gpa": round(float(gpa_df.replace(0, np.nan).min().min()), 2)
            if pd.notna(gpa_df.replace(0, np.nan).min().min())
            else None,
        }

    @cached_property
    @required_columns(cohort=["enrollment_intensity_first_term", "gpa_group_year_1"])
    def gpa_by_enrollment_intensity(self) -> dict[str, list | float]:
        """
        Compute GPA by enrollment intensity across cohort years.

        Returns:
            Dictionary with:
                - cohort_years: List of cohort year strings
                - series: List of dicts with 'name' and 'data' keys
        """

        gpa_df = (
            self.df_cohort.assign(
                gpa=pd.to_numeric(self.df_cohort["gpa_group_year_1"], errors="coerce"),
                enrollment_intensity_first_term=self.df_cohort[
                    "enrollment_intensity_first_term"
                ]
                .astype(str)
                .str.strip()
                .str.title(),
            )
            .loc[
                lambda d: (
                    d["enrollment_intensity_first_term"].isin(
                        ["Full-Time", "Part-Time"]
                    )
                    & d["gpa"].notna()
                )
            ][["cohort", "enrollment_intensity_first_term", "gpa"]]
            .groupby(["enrollment_intensity_first_term", "cohort"], observed=True)[
                "gpa"
            ]
            .mean()
            .unstack()
            .reindex(columns=self.cohort_years(formatted=False))
        )

        series_data = self._format_series_data(gpa_df)

        return {
            "cohort_years": self.cohort_years(formatted=True),
            "series": series_data,
            "min_gpa": round(float(gpa_df.replace(0, np.nan).min().min()), 2),
        }

    @cached_property
    @required_columns(cohort=["cohort_term"])
    def students_by_cohort_term(self) -> dict[str, t.Any]:
        """
        Student counts by term across cohort years. Only terms with count > 0 anywhere are included.
        """
        df = self.df_cohort
        counts_df = (
            df.assign(cohort_term=df["cohort_term"].astype(str).str.strip().str.title())
            .groupby(["cohort", "cohort_term"], observed=True)
            .size()
            .unstack(level=1, fill_value=0)
            .reindex(index=self.cohort_years(formatted=False), fill_value=0)
            .astype(int)
        )
        ordered_terms = ["Fall", "Winter", "Spring", "Summer"]
        reindexed = counts_df.reindex(columns=ordered_terms, fill_value=0).astype(int)
        terms_with_data = [term for term in ordered_terms if reindexed[term].sum() > 0]
        years = self.cohort_years(formatted=True)
        year_totals = reindexed.sum(axis=1).tolist()
        by_year = []
        for i, year in enumerate(years):
            total = int(year_totals[i])
            terms_for_year = []
            for term_name in terms_with_data:
                count = int(reindexed.iloc[i][term_name])
                terms_for_year.append(
                    {
                        "count": count,
                        "percentage": percent_of_rows(count, total),
                        "name": term_name,
                    }
                )
            by_year.append({"year": year, "total": total, "terms": terms_for_year})
        return {"years": years, "by_year": by_year}

    @cached_property
    @required_columns(course=["academic_year", "academic_term"])
    def course_enrollments(self) -> dict[str, t.Any]:
        """
        Course enrollment counts by academic_term across academic_year (when courses were offered).
        Returns empty dict when df_course is None.
        """
        if self.df_course is None:
            return {}

        df = self.df_course
        years_raw = (
            df["academic_year"].dropna().astype(str).sort_values().drop_duplicates()
        )
        years_formatted = [
            y.replace("-", " - ", 1) if "-" in y else y for y in years_raw
        ]
        years_raw = years_raw.tolist()
        counts_df = (
            df.dropna(subset=["academic_term"])
            .assign(
                academic_term=lambda d: (
                    d["academic_term"].astype(str).str.strip().str.title()
                )
            )
            .groupby(["academic_year", "academic_term"], observed=True)
            .size()
            .unstack(level=1, fill_value=0)
            .reindex(index=years_raw, fill_value=0)
            .astype(int)
        )
        ordered_terms = ["Fall", "Winter", "Spring", "Summer"]
        reindexed = counts_df.reindex(columns=ordered_terms, fill_value=0).astype(int)
        terms_with_data = [term for term in ordered_terms if reindexed[term].sum() > 0]
        year_totals = reindexed.sum(axis=1).tolist()
        by_year = []
        for i, year in enumerate(years_formatted):
            total = int(year_totals[i])
            terms_for_year = []
            for term_name in terms_with_data:
                count = int(reindexed.iloc[i][term_name])
                terms_for_year.append(
                    {
                        "count": count,
                        "percentage": percent_of_rows(count, total),
                        "name": term_name,
                    }
                )
            by_year.append({"year": year, "total": total, "terms": terms_for_year})
        return {"years": years_formatted, "by_year": by_year}

    @cached_property
    @required_columns(cohort=["credential_type_sought_year_1"])
    def degree_types(self) -> dict[str, t.Any]:
        """
        Compute degree type counts and percentages.

        Returns:
            Dict with keys:
                - total: Total number of students with a degree type
                - degrees: List of { count, percentage, name } per degree type
        """
        value_counts = (
            self.df_cohort["credential_type_sought_year_1"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.title()
            .str.replace("'S", "'s", regex=False)
            .value_counts()
        )
        total = int(value_counts.sum())
        if total == 0:
            return {"total": 0, "degrees": []}

        degree_df = value_counts.rename("count").to_frame().reset_index(names="name")
        degree_df["percentage"] = (degree_df["count"] / total * 100).round(2)
        degree_df["count"] = degree_df["count"].astype(int)
        degrees = degree_df[["name", "count", "percentage"]].to_dict(orient="records")

        return {"total": total, "degrees": degrees}

    @cached_property
    @required_columns(cohort=["enrollment_type", "enrollment_intensity_first_term"])
    def enrollment_type_by_intensity(self) -> dict[str, t.Any]:
        """
        Compute enrollment type by intensity.

        Returns:
            Dictionary with keys:
                - categories: Sorted list of enrollment type names
                - series: List of dictionaries with keys:
                    - name: Enrollment intensity value (e.g., "Full-Time", "Part-Time")
                    - data: List of counts per category
        """
        df = self.df_cohort.dropna(
            subset=["enrollment_type", "enrollment_intensity_first_term"]
        )
        counts_df = (
            df.assign(
                enrollment_type=df["enrollment_type"]
                .astype(str)
                .str.strip()
                .str.title(),
                enrollment_intensity_first_term=self.df_cohort[
                    "enrollment_intensity_first_term"
                ]
                .astype(str)
                .str.strip()
                .str.title(),
            )[["enrollment_type", "enrollment_intensity_first_term"]]
            .dropna()
            .groupby(
                ["enrollment_intensity_first_term", "enrollment_type"],
                observed=True,
            )
            .size()
            .unstack(fill_value=0)
        )

        return {
            "categories": counts_df.columns.tolist(),
            "series": self._format_series_data(counts_df),
        }

    @cached_property
    @required_columns(cohort=["first_gen", "pell_status_first_year"])
    def pell_recipient_by_first_gen(self) -> dict[str, t.Any] | None:
        """
        Compute Pell recipient status by first generation status.

        Returns:
            Dictionary with keys:
                - categories: Sorted list of Pell status values
                - series: List of dictionaries with keys:
                    - name: First generation status value
                    - data: List of counts per category
        """
        df = self.df_cohort
        df = df.dropna(subset=["first_gen", "pell_status_first_year"])
        if "first_gen" not in df.columns or "pell_status_first_year" not in df.columns:
            return None
        if (
            df["first_gen"].dropna().empty
            or df["pell_status_first_year"].dropna().empty
        ):
            return None

        pell_df = (
            df.assign(
                pell_status_first_year=df["pell_status_first_year"]
                .astype(str)
                .str.strip()
                .str.title(),
                first_gen=df["first_gen"]
                .fillna("N")
                .astype(str)
                .str.upper()
                .map({"Y": "Yes", "N": "No"}),
            )[["pell_status_first_year", "first_gen"]]
            .dropna(subset=["pell_status_first_year"])
            .value_counts()
            .unstack(fill_value=0)
        )

        return {
            "categories": pell_df.index.tolist(),
            "series": self._format_series_data(pell_df.T),
        }

    @cached_property
    @required_columns(cohort=["pell_status_first_year"])
    def pell_recipient_status(self) -> dict[str, t.Any] | None:
        """
        Compute Pell recipient status without first generation split.

        Returns:
            Dictionary with keys:
                - series: Single series with counts per Pell status
        """
        if "pell_status_first_year" not in self.df_cohort.columns:
            return None
        if self.df_cohort["pell_status_first_year"].dropna().empty:
            return None

        data = (
            self.df_cohort.assign(
                pell_status_first_year=self.df_cohort["pell_status_first_year"]
                .astype(str)
                .str.strip()
                .str.title()
            )
            .dropna(subset=["pell_status_first_year"])
            .groupby("pell_status_first_year", observed=True)
            .size()
            .to_dict()
        )
        return {
            "series": [{"name": "All Students", "data": data}],
        }

    @cached_property
    @required_columns(cohort=["gender", "student_age"])
    def student_age_by_gender(self) -> dict[str, t.Any]:
        """
        Compute student age groups by gender.

        Returns:
            Dictionary with keys:
                - categories: Sorted list of gender values (excluding NaN)
                - series: List of dictionaries with keys:
                    - name: Age group ("20 or younger", "20 - 24", "Older than 24")
                    - data: List of counts per category
        """

        age_group_df = (
            self.df_cohort.assign(
                gender=self.df_cohort["gender"].astype(str).str.strip().str.title(),
                student_age=self.df_cohort["student_age"]
                .astype(str)
                .str.strip()
                .str.title(),
            )[["gender", "student_age"]]
            .loc[lambda d: d["gender"] != "Uk"]
            .dropna()
            .value_counts()
            .unstack(fill_value=0)
        )

        return {
            "categories": age_group_df.index.tolist(),
            "series": self._format_series_data(age_group_df.T),
        }

    @cached_property
    @required_columns(cohort=["race", "pell_status_first_year"])
    def race_by_pell_status(self) -> dict[str, t.Any]:
        """
        Compute race by Pell recipient status.

        Returns:
            Dictionary with keys:
                - categories: Race values ordered by count descending (most common first)
                - series: List of dicts with "name" (Pell status) and "data" (counts per category)
        """

        race_df = (
            self.df_cohort.assign(
                race=self.df_cohort["race"].astype(str).str.strip().str.title(),
                pell_status_first_year=self.df_cohort["pell_status_first_year"]
                .astype(str)
                .str.upper()
                .map({"Y": "Yes", "N": "No"}),
            )[["race", "pell_status_first_year"]]
            .dropna()
            .loc[lambda d: d["pell_status_first_year"].isin(["Yes", "No"])]
        )

        counts_df = (
            race_df.groupby(["pell_status_first_year", "race"], observed=True)
            .size()
            .unstack(fill_value=0)
        )

        return {
            "categories": counts_df.columns.tolist(),
            "series": self._format_series_data(counts_df),
        }


# --- Structured audit & light transforms (re-exported for backward compatibility) ---
# New code should import from ``custom_data_audit`` or ``custom_cleaning`` directly.
from . import custom_cleaning as _cc
from . import custom_data_audit as _cda

age_single_value_plausible = _cda.age_single_value_plausible
audit_demographic_column_name_blocked = _cda.audit_demographic_column_name_blocked
audit_value_substring_match_rate = _cda.audit_value_substring_match_rate
bias_variable_codebook_line = _cda.bias_variable_codebook_line
check_earned_vs_attempted = _cda.check_earned_vs_attempted
check_pf_grade_consistency = _cda.check_pf_grade_consistency
duplicate_conflict_columns = _cda.duplicate_conflict_columns
find_dupes = _cda.find_dupes
format_credit_consistency_institution_report = (
    _cda.format_credit_consistency_institution_report
)
infer_age_column = _cda.infer_age_column
infer_check_pf_grade_list_kwargs = _cda.infer_check_pf_grade_list_kwargs
infer_course_credit_columns = _cda.infer_course_credit_columns
infer_course_grade_pf_columns = _cda.infer_course_grade_pf_columns
infer_inst_tot_credits_columns = _cda.infer_inst_tot_credits_columns
infer_pass_fail_flag_tuples = _cda.infer_pass_fail_flag_tuples
infer_semester_credit_aggregate_columns = _cda.infer_semester_credit_aggregate_columns
infer_semester_enrollment_intensity_column = (
    _cda.infer_semester_enrollment_intensity_column
)
infer_student_audit_columns = _cda.infer_student_audit_columns
infer_student_file_categorical = _cda.infer_student_file_categorical
infer_student_id_column = _cda.infer_student_id_column
infer_term_column = _cda.infer_term_column
semester_enrollment_intensity_column_name_score = (
    _cda.semester_enrollment_intensity_column_name_score
)
string_looks_like_age_bucket = _cda.string_looks_like_age_bucket
term_column_name_hint_score = _cda.term_column_name_hint_score
validate_credit_consistency = _cda.validate_credit_consistency
validate_ids_terms_consistency = _cda.validate_ids_terms_consistency
value_looks_like_term = _cda.value_looks_like_term
convert_numeric_columns = _cc.convert_numeric_columns
normalize_student_id_column = _cc.normalize_student_id_column
order_terms = _cc.order_terms
