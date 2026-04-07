import itertools
import logging
import re
import typing as t

import numpy as np
import pandas as pd
import scipy.stats as ss
from functools import cached_property, wraps
from edvise import utils as edvise_utils
from edvise.shared.utils import as_percent, validate_optional_column

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


def percent_of_rows(count: int, total_rows: int, *, ndigits: int = 2) -> float:
    """``100 * count / total_rows`` rounded, or ``0.0`` when ``total_rows == 0``."""
    if not total_rows:
        return 0.0
    return round(100 * count / total_rows, ndigits)


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


def find_dupes(df: pd.DataFrame, primary_keys: list[str]) -> pd.DataFrame:
    """
    Find duplicate rows by key columns and print a summary of column-level conflicts
    within duplicate groups.

    Returns
    -------
    dupes : pd.DataFrame
        All rows involved in duplicate key groups (sorted by student_id)
    """
    dupes = df[df.duplicated(subset=primary_keys, keep=False)].copy()

    # Always sort by student_id (guard in case column missing)
    if "student_id" in dupes.columns:
        dupes = dupes.sort_values("student_id", ignore_index=True)

    total_rows = len(df)
    dupe_rows = len(dupes)
    pct_dupes = percent_of_rows(dupe_rows, total_rows)

    print(
        f"{dupe_rows} duplicate rows based on {primary_keys} "
        f"({pct_dupes:.2f}% of {total_rows} total rows)"
    )

    if dupes.empty:
        conflicts = pd.DataFrame(columns=["column", "pct_conflicting_groups"])
        print(conflicts)
        return dupes

    grp = dupes.groupby(primary_keys, dropna=False)

    # does each column conflict within each dup group?
    conflict = grp.nunique(dropna=False) > 1

    # keep only groups with at least one conflict
    conflict = conflict[conflict.any(axis=1)]

    if conflict.empty:
        conflicts = pd.DataFrame(columns=["column", "pct_conflicting_groups"])
        print(conflicts)
        return dupes

    conflicts = (
        conflict.mean()
        .mul(100)
        .rename("pct_conflicting_groups")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("pct_conflicting_groups", ascending=False)
        .reset_index(drop=True)
    )

    print(conflicts)
    return dupes


def check_earned_vs_attempted(
    df: pd.DataFrame,
    *,
    earned_col: str,
    attempted_col: str,
) -> t.Dict[str, pd.DataFrame]:
    """
    CUSTOM SCHOOL FUNCTION

    Row-wise checks that:
      1. credits_earned <= credits_attempted
      2. credits_earned = 0 when credits_attempted = 0
    """
    earned = pd.to_numeric(df[earned_col], errors="coerce")
    attempted = pd.to_numeric(df[attempted_col], errors="coerce")

    earned_gt_attempted = earned > attempted
    earned_when_no_attempt = (attempted == 0) & (earned > 0)
    mask = earned_gt_attempted | earned_when_no_attempt

    anomalies = df[mask].copy()
    anomalies["earned_gt_attempted"] = earned_gt_attempted[mask]
    anomalies["earned_when_no_attempt"] = earned_when_no_attempt[mask]

    total_rows = len(df)
    n_earned_gt = int(earned_gt_attempted.sum())
    n_earned_no_attempt = int(earned_when_no_attempt.sum())
    n_total = int(mask.sum())
    summary = pd.DataFrame(
        {
            "earned_gt_attempted": [n_earned_gt],
            "earned_gt_attempted_pct": [percent_of_rows(n_earned_gt, total_rows)],
            "earned_when_no_attempt": [n_earned_no_attempt],
            "earned_when_no_attempt_pct": [
                percent_of_rows(n_earned_no_attempt, total_rows)
            ],
            "total_anomalous_rows": [n_total],
            "total_anomalous_rows_pct": [percent_of_rows(n_total, total_rows)],
        }
    )

    return {"anomalies": anomalies, "summary": summary}


def _credit_reconciliation_mismatch_mask(
    merged: pd.DataFrame,
    *,
    sem_has_attempted: bool,
    sem_has_earned: bool,
    match_attempted_col: str = "match_attempted",
    match_earned_col: str = "match_earned",
) -> pd.Series:
    """True where semester vs course credit reconciliation disagrees (missing match counts as mismatch)."""
    mismatch_mask = pd.Series(False, index=merged.index)
    if sem_has_attempted and match_attempted_col in merged.columns:
        mismatch_mask |= ~merged[match_attempted_col].fillna(True)
    if sem_has_earned and match_earned_col in merged.columns:
        mismatch_mask |= ~merged[match_earned_col].fillna(True)
    return mismatch_mask


def log_semester_reconciliation_summary(
    *,
    logger: logging.Logger,
    merged: pd.DataFrame,
    agg: pd.DataFrame,
    s: pd.DataFrame,
    id_col: str,
    sem_col: str,
    sem_has_attempted: bool,
    sem_has_earned: bool,
    diff_attempted_col: str = "diff_attempted",
    match_attempted_col: str = "match_attempted",
    diff_earned_col: str = "diff_earned",
    match_earned_col: str = "match_earned",
) -> None:
    """
    Log a concise, consistent reconciliation report.

    Assumes:
      - s is the semester slice used for merge (has id_col, sem_col)
      - agg is course aggregates (has id_col, sem_col)
      - merged is s merged with agg and has:
          has_course_rows, diff_* and match_* cols (as available)
    """
    total_sem_rows = int(len(s))
    mismatch_mask = _credit_reconciliation_mismatch_mask(
        merged,
        sem_has_attempted=sem_has_attempted,
        sem_has_earned=sem_has_earned,
        match_attempted_col=match_attempted_col,
        match_earned_col=match_earned_col,
    )

    mismatch_rows = int(mismatch_mask.sum())
    mismatch_pct = percent_of_rows(mismatch_rows, total_sem_rows)

    no_course_rows = (
        int((~merged["has_course_rows"]).sum())
        if "has_course_rows" in merged.columns
        else 0
    )

    logger.warning(
        "Semester reconciliation: rows=%d, mismatches=%d (%.1f%%), semester_rows_without_course_rows=%d",
        total_sem_rows,
        mismatch_rows,
        mismatch_pct,
        no_course_rows,
    )

    LOGGER.warning(
        "Semester reconciliation scope: %d student-semester rows compared; raw key coverage verified prior to aggregation",
        total_sem_rows,
    )

    def _log_credit_diff(label: str, diff_col: str, match_col: str) -> None:
        if diff_col not in merged.columns or match_col not in merged.columns:
            return

        mism = int((~merged[match_col].fillna(True)).sum())
        neg = int((merged[diff_col] < 0).sum())
        pos = int((merged[diff_col] > 0).sum())

        logger.warning(
            " - %s: mismatches=%d (%.1f%%); direction sem>courses=%d, courses>sem=%d; abs_diff median=%.1f, p90=%.1f, max=%.1f",
            label,
            mism,
            percent_of_rows(mism, total_sem_rows),
            neg,
            pos,
            float(merged[diff_col].abs().median()),
            float(merged[diff_col].abs().quantile(0.90)),
            float(merged[diff_col].abs().max()),
        )

    if sem_has_attempted:
        _log_credit_diff("Attempted credits", diff_attempted_col, match_attempted_col)

    if sem_has_earned:
        _log_credit_diff("Earned credits", diff_earned_col, match_earned_col)


def validate_credit_consistency(
    course_df: pd.DataFrame,
    semester_df: t.Optional[pd.DataFrame] = None,
    cohort_df: t.Optional[pd.DataFrame] = None,
    *,
    id_col: str = "student_id",
    sem_col: str = "semester",
    course_credits_attempted_col: t.Optional[str] = "credits_attempted",
    course_credits_earned_col: t.Optional[str] = "credits_earned",
    semester_credits_attempted_col: t.Optional[
        str
    ] = "number_of_semester_credits_attempted",
    semester_credits_earned_col: t.Optional[str] = "number_of_semester_credits_earned",
    semester_courses_count_col: t.Optional[str] = "number_of_semester_courses_enrolled",
    cohort_credits_attempted_col: t.Optional[str] = "inst_tot_credits_attempted",
    cohort_credits_earned_col: t.Optional[str] = "inst_tot_credits_earned",
    credit_tol: float = 0.0,
    strict_columns: bool = False,
) -> t.Dict[str, t.Any]:
    """
    Args:
        strict_columns: If True, each credit column name is used only when it is non-empty
            and present on the frame — no alternate name fallbacks (for audit notebooks that
            pass inferred names only).
    """
    LOGGER.info(
        "Starting credit consistency validation "
        "(course_df=%d rows, semester_df=%s, cohort_df=%s)",
        len(course_df),
        "provided" if semester_df is not None else "None",
        "provided" if cohort_df is not None else "None",
    )

    # -------------------------------------------------------
    # Resolve course credit column names
    # -------------------------------------------------------
    if strict_columns:
        resolved_attempted = (
            course_credits_attempted_col
            if course_credits_attempted_col
            and course_credits_attempted_col in course_df.columns
            else None
        )
        resolved_earned = (
            course_credits_earned_col
            if course_credits_earned_col
            and course_credits_earned_col in course_df.columns
            else None
        )
    else:
        resolved_attempted = (
            course_credits_attempted_col
            if course_credits_attempted_col
            and course_credits_attempted_col in course_df.columns
            else "course_credits_attempted"
            if "course_credits_attempted" in course_df.columns
            else None
        )

        resolved_earned = (
            course_credits_earned_col
            if course_credits_earned_col
            and course_credits_earned_col in course_df.columns
            else "course_credits_earned"
            if "course_credits_earned" in course_df.columns
            else None
        )

    has_course_credit_cols = (
        resolved_attempted is not None and resolved_earned is not None
    )

    # =======================================================
    # A) COURSE-LEVEL CHECKS
    # =======================================================
    course_anomalies = None
    course_anomalies_summary = None

    if has_course_credit_cols:
        LOGGER.info("Running course-level earned <= attempted checks")

        cchk = course_df[
            [c for c in [id_col, sem_col] if c in course_df.columns]
            + [resolved_attempted, resolved_earned]
        ].copy()

        cchk[resolved_attempted] = pd.to_numeric(
            cchk[resolved_attempted], errors="coerce"
        )
        cchk[resolved_earned] = pd.to_numeric(cchk[resolved_earned], errors="coerce")

        cchk["diff"] = cchk[resolved_earned] - cchk[resolved_attempted]
        cchk["earned_exceeds_attempted"] = cchk["diff"] > credit_tol
        cchk["attempted_negative"] = cchk[resolved_attempted] < 0
        cchk["earned_negative"] = cchk[resolved_earned] < 0

        course_anomalies = cchk.loc[
            cchk["earned_exceeds_attempted"]
            | cchk["attempted_negative"]
            | cchk["earned_negative"]
        ]

        total_course_rows = len(cchk)
        n_anomalies = int(len(course_anomalies))
        course_anomalies_summary = {
            "rows_checked": total_course_rows,
            "rows_with_anomalies": n_anomalies,
            "pct_of_data": percent_of_rows(n_anomalies, total_course_rows),
        }

        if len(course_anomalies) > 0:
            LOGGER.warning(
                "Detected %d course-level anomalies (%.2f%% of course data)",
                len(course_anomalies),
                course_anomalies_summary["pct_of_data"],
            )
        else:
            LOGGER.info("No course-level credit anomalies detected")

    # =======================================================
    # B) SEMESTER RECONCILIATION
    # =======================================================
    mismatches = None
    merged = None
    reconciliation_summary = None

    sem_has_attempted = False
    sem_has_earned = False
    sem_has_count = False
    if semester_df is not None:
        sem_has_attempted = (
            bool(semester_credits_attempted_col)
            and semester_credits_attempted_col in semester_df.columns
        )
        sem_has_earned = (
            bool(semester_credits_earned_col)
            and semester_credits_earned_col in semester_df.columns
        )
        sem_has_count = (
            bool(semester_courses_count_col)
            and semester_courses_count_col in semester_df.columns
        )

    if (
        semester_df is not None
        and has_course_credit_cols
        and id_col in course_df.columns
        and sem_col in course_df.columns
        and id_col in semester_df.columns
        and sem_col in semester_df.columns
        and (sem_has_attempted or sem_has_earned)
    ):
        LOGGER.info("Reconciling semester aggregates with course data")

        c = course_df[[id_col, sem_col, resolved_attempted, resolved_earned]].copy()
        c[resolved_attempted] = pd.to_numeric(c[resolved_attempted], errors="coerce")
        c[resolved_earned] = pd.to_numeric(c[resolved_earned], errors="coerce")

        s_cols = [id_col, sem_col]
        if sem_has_attempted:
            assert semester_credits_attempted_col is not None
            s_cols.append(semester_credits_attempted_col)
        if sem_has_earned:
            assert semester_credits_earned_col is not None
            s_cols.append(semester_credits_earned_col)
        if sem_has_count:
            assert semester_courses_count_col is not None
            s_cols.append(semester_courses_count_col)

        s = semester_df[s_cols].copy()

        agg = (
            c.groupby([id_col, sem_col], dropna=False)
            .agg(
                course_sum_attempted=(resolved_attempted, "sum"),
                course_sum_earned=(resolved_earned, "sum"),
                course_count=(resolved_attempted, "size"),
            )
            .reset_index()
        )

        merged = s.merge(agg, on=[id_col, sem_col], how="left", indicator="_merge")
        merged["has_course_rows"] = merged["_merge"] == "both"

        if sem_has_attempted:
            merged["diff_attempted"] = merged["course_sum_attempted"] - pd.to_numeric(
                merged[semester_credits_attempted_col], errors="coerce"
            )
            merged["match_attempted"] = merged["diff_attempted"].abs() <= credit_tol

        if sem_has_earned:
            merged["diff_earned"] = merged["course_sum_earned"] - pd.to_numeric(
                merged[semester_credits_earned_col], errors="coerce"
            )
            merged["match_earned"] = merged["diff_earned"].abs() <= credit_tol

        mismatch_mask = _credit_reconciliation_mismatch_mask(
            merged,
            sem_has_attempted=sem_has_attempted,
            sem_has_earned=sem_has_earned,
        )

        mismatches = merged.loc[mismatch_mask]

        total_sem = int(len(s))
        n_mismatched = int(len(mismatches))
        reconciliation_summary = {
            "total_semester_rows": total_sem,
            "mismatched_rows": n_mismatched,
            "pct_of_data": percent_of_rows(n_mismatched, total_sem),
        }

        # 🔹 Clean summary logging
        log_semester_reconciliation_summary(
            logger=LOGGER,
            merged=merged,
            agg=agg,
            s=s,
            id_col=id_col,
            sem_col=sem_col,
            sem_has_attempted=sem_has_attempted,
            sem_has_earned=sem_has_earned,
        )

    # =======================================================
    # C) COHORT CHECKS
    # =======================================================
    cohort_anomalies = None
    cohort_anomalies_summary = None

    cohort_attempted_ok = (
        bool(cohort_credits_attempted_col)
        and cohort_credits_attempted_col in cohort_df.columns
        if cohort_df is not None
        else False
    )
    cohort_earned_ok = (
        bool(cohort_credits_earned_col)
        and cohort_credits_earned_col in cohort_df.columns
        if cohort_df is not None
        else False
    )

    if cohort_df is not None and cohort_attempted_ok and cohort_earned_ok:
        LOGGER.info("Running cohort-level earned <= attempted checks")

        cohort_checks = check_earned_vs_attempted(
            cohort_df,
            earned_col=t.cast(str, cohort_credits_earned_col),
            attempted_col=t.cast(str, cohort_credits_attempted_col),
        )

        cohort_anomalies = cohort_checks.get("anomalies")
        cohort_anomalies_summary = cohort_checks.get("summary")

        if isinstance(cohort_anomalies, pd.DataFrame) and len(cohort_anomalies) > 0:
            pct = None
            if (
                cohort_anomalies_summary is not None
                and isinstance(cohort_anomalies_summary, pd.DataFrame)
                and "total_anomalous_rows_pct" in cohort_anomalies_summary.columns
            ):
                pct = cohort_anomalies_summary["total_anomalous_rows_pct"].iloc[0]
            if pct is not None:
                LOGGER.warning(
                    "Detected %d cohort-level anomalies (%.2f%% of cohort data)",
                    len(cohort_anomalies),
                    pct,
                )
            else:
                LOGGER.warning(
                    "Detected %d cohort-level anomalies", len(cohort_anomalies)
                )
        else:
            LOGGER.info("No cohort-level credit anomalies detected")

    # =======================================================
    # Final Summary
    # =======================================================
    LOGGER.info(
        "Credit validation summary: course_anomalies=%d, semester_mismatches=%s, cohort_anomalies=%s",
        0 if course_anomalies is None else int(len(course_anomalies)),
        "skipped" if mismatches is None else int(len(mismatches)),
        "skipped" if cohort_anomalies is None else int(len(cohort_anomalies)),
    )

    out: dict[str, t.Any] = {
        "course_anomalies": course_anomalies,
        "course_anomalies_summary": course_anomalies_summary,
        "reconciliation_summary": reconciliation_summary,
        "reconciliation_mismatches": mismatches,
        "reconciliation_merged_detail": merged,
        "cohort_anomalies": cohort_anomalies,
        "cohort_anomalies_summary": cohort_anomalies_summary,
    }
    out["institution_report"] = format_credit_consistency_institution_report(out)
    return out


def _credit_report_df_scalar(df: t.Any, col: str, row: int = 0) -> t.Any:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if col not in df.columns:
        return None
    return df[col].iloc[row]


def format_credit_consistency_institution_report(
    result: t.Mapping[str, t.Any],
) -> str:
    """
    Turn the dict returned by :func:`validate_credit_consistency` into a short narrative
    for institutional readers, including suggested next steps.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("CREDIT CONSISTENCY — INSTITUTION SUMMARY")
    lines.append("=" * 72)

    course_sum = result.get("course_anomalies_summary")
    recon_sum = result.get("reconciliation_summary")
    cohort_sum = result.get("cohort_anomalies_summary")
    course_bad = result.get("course_anomalies")
    recon_bad = result.get("reconciliation_mismatches")
    cohort_bad = result.get("cohort_anomalies")

    n_course = int(len(course_bad)) if isinstance(course_bad, pd.DataFrame) else 0
    n_recon = int(len(recon_bad)) if isinstance(recon_bad, pd.DataFrame) else 0
    n_cohort = int(len(cohort_bad)) if isinstance(cohort_bad, pd.DataFrame) else 0

    # --- Course-level ---
    lines.append("")
    lines.append("1) Course file (per-row earned vs attempted credits)")
    if isinstance(course_sum, dict):
        checked = course_sum.get("rows_checked", 0)
        bad = course_sum.get("rows_with_anomalies", 0)
        pct = course_sum.get("pct_of_data", 0.0)
        lines.append(
            f"   Checked {checked:,} enrollment rows. "
            f"{bad:,} rows ({pct}%) show earned > attempted, negative credits, or similar issues."
        )
        if bad == 0:
            lines.append("   Status: No issues flagged at course row level.")
        else:
            lines.append(
                "   Status: Review recommended — course-level credits contradict basic rules."
            )
    else:
        lines.append(
            "   Not run — the course file is missing usable attempted/earned credit columns "
            "(or names did not resolve). Confirm column names in config / bronze extract."
        )

    # --- Semester reconciliation ---
    lines.append("")
    lines.append("2) Semester file vs summed course credits (same student + term)")
    if isinstance(recon_sum, dict):
        total = recon_sum.get("total_semester_rows", 0)
        mm = recon_sum.get("mismatched_rows", 0)
        pct = recon_sum.get("pct_of_data", 0.0)
        lines.append(
            f"   Compared {total:,} student-term rows on the semester file to aggregates from courses."
        )
        lines.append(
            f"   {mm:,} rows ({pct}%) do not match within tolerance (attempted and/or earned totals)."
        )
        if mm == 0:
            lines.append("   Status: Semester totals align with summed course credits.")
        else:
            lines.append(
                "   Status: Investigate term keys, withdrawal rules, and how semester aggregates are built."
            )
    else:
        lines.append(
            "   Not run — needs semester extract plus matching student_id and term columns on both "
            "course and semester files, and compatible credit columns."
        )

    # --- Cohort ---
    lines.append("")
    lines.append("3) Cohort / student file (institutional attempted vs earned totals)")
    if isinstance(cohort_sum, pd.DataFrame) and not cohort_sum.empty:
        total_anom = int(
            _credit_report_df_scalar(cohort_sum, "total_anomalous_rows") or 0
        )
        total_pct = _credit_report_df_scalar(cohort_sum, "total_anomalous_rows_pct")
        eg = int(_credit_report_df_scalar(cohort_sum, "earned_gt_attempted") or 0)
        ena = int(_credit_report_df_scalar(cohort_sum, "earned_when_no_attempt") or 0)
        lines.append(
            f"   {total_anom:,} student rows ({total_pct}%) break earned <= attempted or "
            f"earned credit with zero attempted (earned>attempted: {eg:,}; "
            f"credit with no attempt: {ena:,})."
        )
        if total_anom == 0:
            lines.append(
                "   Status: Institutional totals look consistent at row level."
            )
        else:
            lines.append(
                "   Status: Fix upstream SIS totals or clarify transfer / test credit treatment."
            )
    else:
        lines.append(
            "   Not run — cohort file missing institutional attempted/earned total columns, "
            "or cohort extract not provided."
        )

    # --- Overall ---
    lines.append("")
    lines.append("-" * 72)
    any_issue = (n_course + n_recon + n_cohort) > 0
    skipped_all = (
        not isinstance(course_sum, dict)
        and recon_sum is None
        and (not isinstance(cohort_sum, pd.DataFrame) or cohort_sum.empty)
    )
    if skipped_all:
        lines.append(
            'Overall: Checks could not run end-to-end — see sections marked "Not run".'
        )
    elif any_issue:
        lines.append(
            "Overall: At least one layer failed checks. Use the detailed tables in the audit "
            "notebook to sample offending rows and trace back to source systems."
        )
    else:
        lines.append(
            "Overall: No credit consistency issues were flagged in the checks that ran. "
            "Keep monitoring after SIS or ETL changes."
        )

    # --- Next steps ---
    lines.append("")
    lines.append("SUGGESTED NEXT STEPS FOR THE INSTITUTION")
    lines.append("-" * 72)
    steps: list[str] = []
    if not isinstance(course_sum, dict):
        steps.append(
            "Map the correct course-level attempted and earned credit fields in `config.toml` "
            "and re-run the audit (or rename columns in the bronze extract)."
        )
    elif n_course > 0:
        steps.append(
            "Course file: Spot-check programs with the highest anomaly rates; verify credit hours "
            "vs enrollment status and repeat/audit courses."
        )
    if recon_sum is None:
        steps.append(
            "Semester reconciliation: Ensure course and semester files share the same student ID "
            "and term identifier; align column names with `validate_credit_consistency` arguments."
        )
    elif n_recon > 0:
        steps.append(
            "Semester file: Reconcile aggregation logic (sum of course credits vs official term "
            "totals); confirm part-term drops and cross-listed sections are handled consistently."
        )
    if not isinstance(cohort_sum, pd.DataFrame) or cohort_sum.empty:
        steps.append(
            "Cohort file: Expose institutional cumulative attempted and earned credits in the "
            "extract if you want this check; confirm field definitions with the registrar."
        )
    elif n_cohort > 0:
        steps.append(
            "Cohort totals: Work with the registrar or data owner to correct lifetime attempted/"
            "earned totals or document known exceptions (e.g. transfer credit timing)."
        )
    if not any_issue and not skipped_all:
        steps.append(
            "Documentation: Archive this run (counts and date) as evidence for internal QA or "
            "accreditation folders."
        )
    if skipped_all:
        steps.append(
            "Prioritize fixing bronze schema and paths so all three layers (course, semester, cohort) "
            "can be validated automatically."
        )
    if not steps:
        steps.append("No specific follow-up beyond routine monitoring.")
    for i, s in enumerate(steps, start=1):
        lines.append(f"  {i}. {s}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


CHECK_PF_DEFAULT_PASSING_GRADES: tuple[str, ...] = (
    "P",
    "P*",
    "A",
    "A-",
    "B+",
    "B",
    "B-",
    "C+",
    "C",
    "C-",
    "D+",
    "D",
    "D-",
)
CHECK_PF_DEFAULT_FAILING_GRADES: tuple[str, ...] = (
    "F",
    "E",
    "^E",
    "F*",
    "REF",
    "NR",
    "W",
    "W*",
    "I",
)
CHECK_PF_DEFAULT_PASS_FLAGS: tuple[str, ...] = ("P",)
CHECK_PF_DEFAULT_FAIL_FLAGS: tuple[str, ...] = ("F",)

_PASS_FAIL_FLAG_VALUE_PAIRS: dict[
    frozenset[str], tuple[tuple[str, ...], tuple[str, ...]]
] = {
    frozenset({"Y", "N"}): (("Y",), ("N",)),
    frozenset({"P", "F"}): (("P",), ("F",)),
    frozenset({"PASS", "FAIL"}): (("PASS",), ("FAIL",)),
    frozenset({"COMPLETE", "INCOMPLETE"}): (("COMPLETE",), ("INCOMPLETE",)),
    frozenset({"1", "0"}): (("1",), ("0",)),
    frozenset({"T", "F"}): (("T",), ("F",)),
}


def _observed_upper_tokens(series: pd.Series) -> set[str]:
    return {
        x
        for x in series.dropna().astype(str).str.strip().str.upper().unique()
        if x and str(x).upper() != "NAN"
    }


def infer_pass_fail_flag_tuples(
    pf_series: pd.Series,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """
    Infer ``pass_flags`` and ``fail_flags`` for :func:`check_pf_grade_consistency` when the
    column has exactly two distinct non-null tokens (e.g. Y/N, P/F).
    """
    u = frozenset(_observed_upper_tokens(pf_series))
    if len(u) != 2:
        return None
    return _PASS_FAIL_FLAG_VALUE_PAIRS.get(u)


_LIKELY_PASSING_GRADE_RE = re.compile(
    r"^([ABCD][+-]?|P\*?|PASS)$",
    re.IGNORECASE,
)
_LIKELY_FAILING_GRADE_RE = re.compile(
    r"^(F[\*]?|E|W[\*]?|I|NR|REF|\^E|INCOMP)$",
    re.IGNORECASE,
)


def infer_check_pf_grade_list_kwargs(
    df: pd.DataFrame,
    grade_col: str,
    pf_col: str,
) -> dict[str, tuple[str, ...]]:
    """
    Build ``passing_grades``, ``failing_grades``, ``pass_flags``, and ``fail_flags`` for
    :func:`check_pf_grade_consistency`: defaults plus pass/fail flags inferred from two-value
    columns, and any observed grade tokens that look like pass/fail but are not in the defaults.
    """
    passing: set[str] = set(CHECK_PF_DEFAULT_PASSING_GRADES)
    failing: set[str] = set(CHECK_PF_DEFAULT_FAILING_GRADES)
    if grade_col in df.columns:
        for g in _observed_upper_tokens(df[grade_col]):
            if g in passing or g in failing:
                continue
            if _LIKELY_PASSING_GRADE_RE.match(g):
                passing.add(g)
            elif (
                _LIKELY_FAILING_GRADE_RE.match(g)
                or len(g) == 1
                and g in {"F", "E", "I", "W"}
            ):
                failing.add(g)

    pf_inf = None
    if pf_col in df.columns:
        pf_inf = infer_pass_fail_flag_tuples(df[pf_col])
    if pf_inf is not None:
        pass_flags, fail_flags = pf_inf
    else:
        pass_flags, fail_flags = (
            CHECK_PF_DEFAULT_PASS_FLAGS,
            CHECK_PF_DEFAULT_FAIL_FLAGS,
        )

    return {
        "passing_grades": tuple(sorted(passing)),
        "failing_grades": tuple(sorted(failing)),
        "pass_flags": pass_flags,
        "fail_flags": fail_flags,
    }


PF_GRADE_ANOMALY_FLAG_COLUMNS: tuple[str, ...] = (
    "earned_with_failing_grade",
    "no_credits_with_passing_grade",
    "grade_pf_disagree",
)


def _pass_fail_label_series(
    normalized_tokens: pd.Series,
    pass_values: tuple[str, ...],
    fail_values: tuple[str, ...],
) -> pd.Series:
    """Map normalized tokens to pass (``True``), fail (``False``), or unknown (missing)."""
    return pd.Series(
        np.where(
            normalized_tokens.isin(pass_values),
            True,
            np.where(normalized_tokens.isin(fail_values), False, np.nan),
        ),
        index=normalized_tokens.index,
        dtype="object",
    )


def check_pf_grade_consistency(
    df: pd.DataFrame,
    grade_col: str = "grade",
    pf_col: str = "pass_fail_flag",
    credits_col: str = "credits_earned",
    *,
    passing_grades: tuple[str, ...] = CHECK_PF_DEFAULT_PASSING_GRADES,
    failing_grades: tuple[str, ...] = CHECK_PF_DEFAULT_FAILING_GRADES,
    pass_flags: tuple[str, ...] = CHECK_PF_DEFAULT_PASS_FLAGS,
    fail_flags: tuple[str, ...] = CHECK_PF_DEFAULT_FAIL_FLAGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    CUSTOM SCHOOL FUNCTION

    Checks that:
      1. Students NEVER earn credits for failing grades.
      2. Students DO always earn credits for passing grades.
      3. Grade and pass_fail_flag are consistent.

    Returns (anomalies_df, summary_df)
    """
    LOGGER.info(
        "Running PF/grade consistency checks "
        "(rows=%d, grade_col=%s, pf_col=%s, credits_col=%s)",
        len(df),
        grade_col,
        pf_col,
        credits_col,
    )

    out = df.copy()

    # Normalize
    g = out[grade_col].astype(str).str.strip().str.upper()
    pf = out[pf_col].astype(str).str.strip().str.upper()
    credits = pd.to_numeric(out[credits_col], errors="coerce")  # keep NaNs as NaN
    LOGGER.debug(
        "Normalized grade/PF/credits (non-null counts: grade=%d, pf=%d, credits=%d)",
        g.notna().sum(),
        pf.notna().sum(),
        credits.notna().sum(),
    )

    pfg = _pass_fail_label_series(g, passing_grades, failing_grades)
    pff = _pass_fail_label_series(pf, pass_flags, fail_flags)

    LOGGER.debug(
        "Derived PF indicators (from grade: pass=%d fail=%d unknown=%d; "
        "from flag: pass=%d fail=%d unknown=%d)",
        int((pfg == True).sum()),
        int((pfg == False).sum()),
        int(pfg.isna().sum()),
        int((pff == True).sum()),
        int((pff == False).sum()),
        int(pff.isna().sum()),
    )

    rules = dict(
        zip(
            PF_GRADE_ANOMALY_FLAG_COLUMNS,
            (
                (pff == False) & credits.notna() & (credits > 0),
                (pff == True) & credits.notna() & (credits == 0),
                pfg.notna() & pff.notna() & (pfg != pff),
            ),
            strict=True,
        )
    )

    LOGGER.debug(
        "Rule violations: %s",
        ", ".join(f"{k}={int(v.sum())}" for k, v in rules.items()),
    )

    mask = pd.Series(False, index=out.index)
    for series in rules.values():
        mask |= series

    anomalies = out.loc[mask].copy()
    for name, series in rules.items():
        anomalies[name] = series.loc[anomalies.index]

    total_rows = len(df)
    summary_payload: dict[str, list[t.Any]] = {}
    for name, series in rules.items():
        n = int(series.sum())
        summary_payload[name] = [n]
        summary_payload[f"{name}_pct"] = [percent_of_rows(n, total_rows)]
    n_total = int(mask.sum())
    summary_payload["total_anomalous_rows"] = [n_total]
    summary_payload["total_anomalous_rows_pct"] = [percent_of_rows(n_total, total_rows)]
    summary = pd.DataFrame(summary_payload)

    if summary["total_anomalous_rows"].iloc[0] > 0:
        LOGGER.warning(
            "Detected %d PF/grade consistency anomalies (%.2f%% of data)",
            summary["total_anomalous_rows"].iloc[0],
            summary["total_anomalous_rows_pct"].iloc[0],
        )
    else:
        LOGGER.info("No PF/grade consistency anomalies detected")

    LOGGER.debug("PF/grade anomaly summary:\n%s", summary)
    return anomalies, summary


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


def order_terms(
    df: pd.DataFrame, term_col: str, season_order: dict[str, int] | None = None
) -> pd.DataFrame:
    """
    CUSTOM SCHOOL FUNCTION

    Make df[term_col] an ordered categorical based on Season + Year.
    Handles both 'Spring 2024' and '2024 Spring' formats.
    Compatible with CleanSpec.term_order_fn(df, term_col).

    Args:
        df: DataFrame containing the term column
        term_col: Name of the column containing term strings
        season_order: Optional dict mapping season names to sort order.
                     Defaults to Spring=1, Summer=2, Fall=3, Winter=4
    """
    if term_col not in df.columns:
        return df

    if season_order is None:
        season_order = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

    out = df.copy()
    out[term_col] = out[term_col].astype("string")

    unique_terms = out[term_col].dropna().unique()
    if len(unique_terms) == 0:
        return out

    sorted_terms = sorted(
        unique_terms, key=lambda term: _parse_term(term, season_order)
    )

    out[term_col] = pd.Categorical(
        out[term_col],
        categories=sorted_terms,
        ordered=True,
    )

    logging.info(
        "term_order_fn: term_col=%s, categories=%s",
        term_col,
        list(out[term_col].cat.categories),
    )
    return out


def _parse_term(term: str, season_order: dict[str, int]) -> tuple[int, int]:
    """
    Parse a term string into a (year, season_rank) tuple for sorting.
    Handles both 'Spring 2024' and '2024 Spring' formats.
    """
    parts = str(term).split()
    if len(parts) != 2:
        return (9999, 99)

    if parts[0].isdigit():
        year, season = int(parts[0]), parts[1]  # '2024 Spring'
    else:
        season, year = parts[0], int(parts[1])  # 'Spring 2024'

    return (year, season_order.get(season, 99))


_ACADEMIC_TERM_SEASON_TOKENS = frozenset(
    {"Spring", "Summer", "Fall", "Winter", "Autumn"}
)


def value_looks_like_term(val: t.Any) -> bool:
    """
    True if *val* looks like a term string such as ``Spring 2024`` or ``2024 Spring``.
    """
    if pd.isna(val):
        return False
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return False
    parts = s.split()
    if len(parts) != 2:
        return False
    if parts[0].isdigit() and len(parts[0]) == 4:
        token = parts[1].strip().title()
    elif parts[1].isdigit() and len(parts[1]) == 4:
        token = parts[0].strip().title()
    else:
        return False
    return token in _ACADEMIC_TERM_SEASON_TOKENS


def term_column_name_hint_score(col: str, name_hints: tuple[str, ...]) -> float:
    """Small bonus when *col* matches typical term column name substrings."""
    c = col.lower()
    if c in {h.lower() for h in name_hints}:
        return 0.15
    for h in name_hints:
        if h.lower() in c:
            return 0.08
    return 0.0


def infer_term_column(
    df: pd.DataFrame,
    *,
    name_hints: tuple[str, ...],
    min_match_rate: float = 0.35,
    max_sample: int = 8000,
) -> str | None:
    """
    Choose the column whose non-null values most often look like academic term strings.

    Scores each candidate as value match rate plus :func:`term_column_name_hint_score`.
    """
    best_col: str | None = None
    best_score = -1.0

    def consider_col(col: str) -> None:
        nonlocal best_col, best_score
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            return
        non_null = s.dropna()
        if len(non_null) == 0:
            return
        sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null
        str_sample = sample.astype("string")
        rates = str_sample.map(value_looks_like_term)
        rate = float(rates.mean()) if len(rates) else 0.0
        hint = term_column_name_hint_score(col, name_hints)
        score = rate + hint
        if rate >= min_match_rate or (hint >= 0.08 and rate >= 0.15):
            if score > best_score:
                best_score = score
                best_col = col

    for col in df.columns:
        consider_col(col)

    if best_col is None:
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                continue
            non_null = s.dropna()
            if len(non_null) == 0:
                continue
            sample = (
                non_null.head(max_sample) if len(non_null) > max_sample else non_null
            )
            str_sample = sample.astype("string")
            rate = float(str_sample.map(value_looks_like_term).mean())
            hint = term_column_name_hint_score(col, name_hints)
            score = rate + hint
            if score > best_score and rate >= 0.2:
                best_score = score
                best_col = col

    return best_col


# --- Column inference for student-level audits (IDs, credits, demographics) ---

_AUDIT_DEMOGRAPHIC_NAME_BLOCKLIST = (
    "ssn",
    "email",
    "phone",
    "address",
    "uuid",
    "hash",
    "password",
    "name",
    "first_name",
    "last_name",
    "middle_name",
    "street",
    "zip",
    "dob",
    "date_of_birth",
    "birth",
)


def audit_demographic_column_name_blocked(col: str) -> bool:
    """
    True if *col* should not be used for demographic / student-type inference
    (PII-ish or free-text name fields). Does not block legitimate ``student_id``.
    """
    c = col.lower()
    return any(f in c for f in _AUDIT_DEMOGRAPHIC_NAME_BLOCKLIST)


def audit_value_substring_match_rate(
    series: pd.Series,
    substrings: tuple[str, ...],
    *,
    max_sample: int = 8000,
) -> float:
    """Fraction of non-null *series* values whose string form contains a substring."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null

    def matches(v: t.Any) -> bool:
        t_ = str(v).strip().lower()
        if not t_ or t_ == "nan":
            return False
        return any(sub in t_ for sub in substrings)

    return float(sample.map(matches).mean())


DEFAULT_STUDENT_TYPE_NAME_HINTS: tuple[str, ...] = (
    "entry_type",
    "student_type",
    "admit_type",
    "admission_type",
    "enrollment_type",
    "stu_type",
    "student_class",
    "class_level",
    "cohort_type",
)
DEFAULT_STUDENT_TYPE_VALUE_SUBSTRINGS: tuple[str, ...] = (
    "transfer",
    "freshman",
    "fresh",
    "ftic",
    "ftf",
    "first time",
    "first-time",
    "first year",
    "readmit",
    "re-admit",
    "readm",
    "re_admit",
    "continuing",
    "returning",
    "non-degree",
    "nondegree",
    "transient",
    "dual",
    "new",
)
DEFAULT_FIRST_GEN_NAME_HINTS: tuple[str, ...] = (
    "first_gen",
    "first_generation",
    "firstgen",
    "fg_status",
    "firstgeneration",
    "fgen",
    "first_time_college",
    "gen1",
    "parent_education",
)
DEFAULT_RACE_NAME_HINTS: tuple[str, ...] = (
    "race",
    "ipeds_race",
    "racial",
    "race_code",
    "race_ethnicity",
    "ethrace",
)
DEFAULT_ETHNICITY_NAME_HINTS: tuple[str, ...] = (
    "ethnicity",
    "ethnic",
    "hispanic",
    "latinx",
    "latino",
    "latina",
    "hl_indicator",
    "hispanic_latino",
    "is_hispanic",
)
DEFAULT_GENDER_NAME_HINTS: tuple[str, ...] = (
    "gender",
    "legal_sex",
    "biological_sex",
    "sex",
    "gender_identity",
)
DEFAULT_AGE_NAME_HINTS: tuple[str, ...] = (
    "age",
    "student_age",
    "age_at_entry",
    "age_as_of",
    "stu_age",
    "current_age",
    "age_years",
)
DEFAULT_PELL_NAME_HINTS: tuple[str, ...] = (
    "pell",
    "awarded_pell",
    "pell_elig",
    "pell_eligible",
    "pell_recipient",
    "pell_flag",
    "pell_status",
)
DEFAULT_INCARCERATION_NAME_HINTS: tuple[str, ...] = (
    "incarceration",
    "incarcerat",
    "correctional",
    "justice_involved",
    "corrections",
)
DEFAULT_MILITARY_NAME_HINTS: tuple[str, ...] = (
    "military",
    "military_status",
    "veteran",
    "vet_status",
    "armed_forces",
    "service_status",
    "ad_t",
    "national_guard",
    "reserve",
)
DEFAULT_EMPLOYMENT_STATUS_NAME_HINTS: tuple[str, ...] = (
    "employment_status",
    "emp_status",
    "work_status",
    "student_employment",
    "employment",
    "job_status",
    "labor_status",
)
DEFAULT_DISABILITY_NAME_HINTS: tuple[str, ...] = (
    "disability",
    "disab_status",
    "ada",
    "accessibility",
    "disabled",
)

DEFAULT_STUDENT_ID_NAME_HINTS: tuple[str, ...] = (
    "student_id",
    "student id",
    "studentid",
    "stu_id",
    "stuid",
    "emplid",
    "empl_id",
    "pid",
    "person_id",
    "banner_id",
    "bannerid",
    "sis_id",
    "school_id",
    "student_number",
    "student_num",
    "id_number",
)


def infer_student_file_categorical(
    df: pd.DataFrame,
    *,
    name_hints: tuple[str, ...],
    value_substrings: tuple[str, ...] | None,
    exclude_cols: set[str],
    max_sample: int = 8000,
    min_nunique: int = 2,
    max_nunique: int = 80,
    min_value_rate: float = 0.12,
    min_name_hint: float = 0.08,
) -> str | None:
    """
    Pick a column using name hints plus optional substring matches in values.
    Skips PII-ish names and near-unique columns (likely IDs).
    """
    n_rows = len(df)
    best_col: str | None = None
    best_score = -1.0

    for col in df.columns:
        if col in exclude_cols or audit_demographic_column_name_blocked(col):
            continue
        s = df[col]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        nunique = int(non_null.astype(str).nunique())
        if nunique < min_nunique or nunique > max_nunique:
            continue
        if n_rows and nunique > max(0.92 * n_rows, 500):
            continue

        hint = term_column_name_hint_score(col, name_hints)
        if value_substrings is not None:
            rate = audit_value_substring_match_rate(
                s, value_substrings, max_sample=max_sample
            )
            if hint < min_name_hint and rate < min_value_rate:
                continue
            score = 2.5 * hint + rate
        else:
            if hint < min_name_hint:
                continue
            score = 3.0 * hint + min(1.0, nunique / 50.0)

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def string_looks_like_age_bucket(s: str) -> bool:
    """
    True if *s* looks like an age band or inequality label (e.g. ``<24``, ``20-24``,
    ``older than 24``), not arbitrary free text.
    """
    t = s.strip().lower()
    if not t or t == "nan":
        return False
    compact = re.sub(r"\s+", " ", t)

    if re.search(r"^\d{1,2}\s*[-–—]\s*\d{1,2}$", compact):
        return True
    if re.search(r"^\d{1,2}\s+to\s+\d{1,2}$", compact):
        return True
    if re.search(r"^\d{1,2}\s+through\s+\d{1,2}$", compact):
        return True
    if re.search(r"^<\s*\d{1,2}$", compact):
        return True
    if re.search(r"^<=\s*\d{1,2}$", compact):
        return True
    if re.search(r"^≤\s*\d{1,2}$", compact):
        return True
    if re.search(r"^>\s*\d{1,2}$", compact):
        return True
    if re.search(r"^>=\s*\d{1,2}$", compact):
        return True
    if re.search(r"^≥\s*\d{1,2}$", compact):
        return True
    if re.search(r"older\s+than\s+\d{1,2}", compact):
        return True
    if re.search(r"over\s+\d{1,2}", compact):
        return True
    if re.search(r"under\s+\d{1,2}", compact):
        return True
    if re.search(r"less\s+than\s+\d{1,2}", compact):
        return True
    if re.search(r"below\s+\d{1,2}", compact):
        return True
    if re.search(r"at\s+least\s+\d{1,2}", compact):
        return True
    if re.search(r"more\s+than\s+\d{1,2}", compact):
        return True
    if re.search(r"^\d{1,2}\s*\+\s*$", compact):
        return True
    if re.search(r"^\d{1,3}\s*\+\s*$", compact):
        return True
    return False


def age_single_value_plausible(val: t.Any) -> bool:
    """True for plausible numeric age (10–100) or bucket string (e.g. ``20-24``, ``<24``)."""
    if pd.isna(val):
        return False
    n = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    if pd.notna(n) and np.isfinite(float(n)) and 10 <= float(n) <= 100:
        return True
    s = str(val).strip().lower()
    if not s or s == "nan":
        return False
    return string_looks_like_age_bucket(s)


def _age_plausibility_rate(series: pd.Series, max_sample: int = 8000) -> float:
    """Fraction of non-null values that are plausible numeric ages or age-band strings."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null
    ok = sample.map(age_single_value_plausible)
    return float(ok.mean()) if len(ok) else 0.0


def infer_age_column(
    df: pd.DataFrame,
    *,
    name_hints: tuple[str, ...] = DEFAULT_AGE_NAME_HINTS,
    exclude_cols: t.AbstractSet[str] | None = None,
    min_name_hint: float = 0.08,
    min_plausible_rate: float = 0.65,
    max_nunique: int = 120,
) -> str | None:
    """
    Infer a student age column: integer ages, or categorical bands such as ``<24``,
    ``20-24``, ``older than 24`` (see :func:`string_looks_like_age_bucket`).
    """
    used = set(exclude_cols or ())
    best_col: str | None = None
    best_score = -1.0
    n_rows = len(df)

    for col in df.columns:
        if col in used or audit_demographic_column_name_blocked(col):
            continue
        s = df[col]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        nunique = int(non_null.nunique(dropna=True))
        if nunique < 2 or nunique > max_nunique:
            continue
        if n_rows and nunique > 0.95 * n_rows:
            continue
        hint = term_column_name_hint_score(col, name_hints)
        rate = _age_plausibility_rate(s)
        if hint < min_name_hint and rate < min_plausible_rate:
            continue
        score = 2.5 * hint + rate
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def infer_student_audit_columns(
    df: pd.DataFrame,
    *,
    term_col: str | None = None,
    student_type_name_hints: tuple[str, ...] = DEFAULT_STUDENT_TYPE_NAME_HINTS,
    student_type_value_substrings: tuple[
        str, ...
    ] = DEFAULT_STUDENT_TYPE_VALUE_SUBSTRINGS,
    first_gen_name_hints: tuple[str, ...] = DEFAULT_FIRST_GEN_NAME_HINTS,
    race_name_hints: tuple[str, ...] = DEFAULT_RACE_NAME_HINTS,
    ethnicity_name_hints: tuple[str, ...] = DEFAULT_ETHNICITY_NAME_HINTS,
    gender_name_hints: tuple[str, ...] = DEFAULT_GENDER_NAME_HINTS,
    age_name_hints: tuple[str, ...] = DEFAULT_AGE_NAME_HINTS,
    pell_name_hints: tuple[str, ...] = DEFAULT_PELL_NAME_HINTS,
    incarceration_name_hints: tuple[str, ...] = DEFAULT_INCARCERATION_NAME_HINTS,
    military_name_hints: tuple[str, ...] = DEFAULT_MILITARY_NAME_HINTS,
    employment_name_hints: tuple[str, ...] = DEFAULT_EMPLOYMENT_STATUS_NAME_HINTS,
    disability_name_hints: tuple[str, ...] = DEFAULT_DISABILITY_NAME_HINTS,
) -> dict[str, str | None]:
    """
    Infer student-type and equity-related columns; each role maps to at most one column.

    Roles: ``student_type``, ``first_gen``, ``race``, ``ethnicity``, ``gender``, ``age``, ``pell``,
    plus ``incarceration``, ``military``, ``employment``, ``disability`` for extended bias audits.
    """
    used: set[str] = set()
    if term_col:
        used.add(term_col)

    out: dict[str, str | None] = {}

    out["student_type"] = infer_student_file_categorical(
        df,
        name_hints=student_type_name_hints,
        value_substrings=student_type_value_substrings,
        exclude_cols=used,
    )
    if out["student_type"]:
        used.add(out["student_type"])

    for key, hints in (
        ("first_gen", first_gen_name_hints),
        ("race", race_name_hints),
        ("ethnicity", ethnicity_name_hints),
        ("gender", gender_name_hints),
    ):
        out[key] = infer_student_file_categorical(
            df,
            name_hints=hints,
            value_substrings=None,
            exclude_cols=used,
            max_nunique=120,
        )
        inferred_col = out[key]
        if inferred_col:
            used.add(inferred_col)

    out["age"] = infer_age_column(df, name_hints=age_name_hints, exclude_cols=used)
    if out["age"]:
        used.add(out["age"])

    out["pell"] = infer_student_file_categorical(
        df,
        name_hints=pell_name_hints,
        value_substrings=None,
        exclude_cols=used,
        max_nunique=120,
    )
    if out["pell"]:
        used.add(out["pell"])

    for key, hints in (
        ("incarceration", incarceration_name_hints),
        ("military", military_name_hints),
        ("employment", employment_name_hints),
        ("disability", disability_name_hints),
    ):
        out[key] = infer_student_file_categorical(
            df,
            name_hints=hints,
            value_substrings=None,
            exclude_cols=used,
            max_nunique=120,
        )
        if out[key]:
            used.add(out[key])

    return out


def bias_variable_codebook_line(role: str) -> str | None:
    """
    Short decoding hint for institutional audit printouts (codes vary by SIS).

    Typical encodings align with common IPEDS-style and registrar exports.
    """
    hints: dict[str, str] = {
        "first_gen": "Typical codes: Y=Yes, N=No (optional at some institutions).",
        "pell": "Typical codes: Y=Yes, N=No (recipient or eligibility; optional).",
        "incarceration": "Typical codes: Y=Yes, N=No (optional field).",
        "military": (
            "Typical codes: 1=Veteran; 2=Active Duty/Reserves/National Guard; "
            "3=Never served (optional field)."
        ),
        "employment": (
            "Typical codes: 1=full-time; 2=less than full-time but at least half-time; "
            "3=less than half-time; 4=not employed (optional field)."
        ),
        "disability": "Typical codes: Y=has a disability; N=does not.",
    }
    return hints.get(role)


def infer_student_id_column(
    df: pd.DataFrame,
    *,
    name_hints: tuple[str, ...] = DEFAULT_STUDENT_ID_NAME_HINTS,
    min_name_hint: float = 0.08,
    max_sample: int = 8000,
) -> str | None:
    """
    Infer the primary student identifier column (high cardinality, stable string/int tokens).

    Strong name matches (e.g. exact ``student_id``) only require two distinct values;
    otherwise requires enough distinct IDs for roster- or transaction-style files.
    """
    n_rows = len(df)
    if n_rows == 0:
        return None

    best_col: str | None = None
    best_score = -1.0

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        nunique = int(non_null.nunique())
        hint = term_column_name_hint_score(col, name_hints)
        if hint < min_name_hint:
            continue
        if hint >= 0.15:
            if nunique < 2:
                continue
        else:
            min_distinct = max(10, min(500, n_rows // 500 or 1))
            if nunique < min_distinct:
                continue
            id_ratio = nunique / n_rows if n_rows else 0.0
            if id_ratio < 0.02:
                continue
        sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null
        str_sample = sample.astype("string").str.strip()
        frac_short = float((str_sample.str.len() <= 32).mean())
        if frac_short < 0.95:
            continue
        id_ratio = nunique / n_rows if n_rows else 0.0
        score = 3.0 * hint + id_ratio + 0.1 * frac_short
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def normalize_student_id_column(
    df: pd.DataFrame,
    *,
    target_name: str = "student_id",
    source_col: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """
    Return a copy of *df* with the inferred (or provided) id column renamed to *target_name*.

    If *target_name* already exists, returns ``(df.copy(), target_name)`` without renaming.
    If no id column can be resolved, returns ``(df.copy(), None)``.
    """
    out = df.copy()
    if target_name in out.columns:
        return out, target_name
    src = source_col if source_col is not None else infer_student_id_column(out)
    if src is None:
        return out, None
    if src == target_name:
        return out, target_name
    out = out.rename(columns={src: target_name})
    return out, target_name


def _numeric_coercion_rate(series: pd.Series, max_sample: int = 8000) -> float:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null
    cleaned = sample.astype("string").str.strip().str.replace(",", "", regex=False)
    n = pd.to_numeric(cleaned, errors="coerce")
    return float(n.notna().mean())


DEFAULT_INST_TOT_CREDITS_ATTEMPTED_NAME_HINTS: tuple[str, ...] = (
    "inst_tot_credits_attempted",
    "institution_credits_attempted",
    "inst_credits_attempted",
    "total_credits_attempted",
    "ug_credits_attempted",
    "undergrad_credits_attempted",
    "cumulative_credits_attempted",
    "cum_credits_attempted",
    "career_credits_attempted",
    "credits_attempted_inst",
    "number_of_credits_attempted",
    "num_credits_attempted",
    "credit_hours",
    "credit_hours_attempted",
)
DEFAULT_INST_TOT_CREDITS_EARNED_NAME_HINTS: tuple[str, ...] = (
    "inst_tot_credits_earned",
    "institution_credits_earned",
    "inst_credits_earned",
    "total_credits_earned",
    "ug_credits_earned",
    "undergrad_credits_earned",
    "cumulative_credits_earned",
    "cum_credits_earned",
    "career_credits_earned",
    "credits_earned_inst",
    "number_of_credits_earned",
    "num_credits_earned",
    "no_of_credits_earned",
    "credit_hours_earned",
)


_CREDIT_NAME_TYPO_NORMALIZATIONS: tuple[tuple[str, str], ...] = (
    ("cumlative", "cumulative"),
    ("cumulitive", "cumulative"),
    ("comulative", "cumulative"),
    ("comulitive", "cumulative"),
)


def _normalize_credit_column_name(col: str) -> str:
    c = col.lower().replace(" ", "_").replace("-", "_")
    for wrong, right in _CREDIT_NAME_TYPO_NORMALIZATIONS:
        c = c.replace(wrong, right)
    return c


def credit_column_name_has_attempt_marker(col: str) -> bool:
    """
    True if *col* name signals **attempted** credits (not cumulative totals).

    Matches ``attempt``, ``attmpt``, or a standalone ``att`` token (e.g. ``sem_att_credits``).
    """
    c = _normalize_credit_column_name(col)
    if "attempt" in c or "attmpt" in c:
        return True
    return re.search(r"(^|_)att(_|$)", c) is not None


def credits_attempted_column_name_score(col: str) -> float:
    """
    **Attempted** institutional / aggregate credits: requires **attempt** markers (or
    enrollment **hours**). Cumulative/total **without** attempt → not scored here (those
    default to **earned** in :func:`credits_earned_column_name_score`).
    """
    c = _normalize_credit_column_name(col)
    if "credit" not in c:
        return 0.0
    if ("earned" in c) and not credit_column_name_has_attempt_marker(col):
        return 0.0
    has_att = credit_column_name_has_attempt_marker(col)
    s = 0.0
    if "hour" in c or "hrs" in c or c.endswith("_hr") or "_hr_" in c:
        if "earned" not in c:
            s += 1.15
    if has_att:
        s += 1.25
        if "cum" in c or "cumulative" in c:
            s += 0.5
        if "total" in c or "tot_" in c or c.startswith("inst") or "_inst_" in c:
            s += 0.3
        if "number_of" in c or "num_" in c or "nbr_" in c:
            s += 0.15
    return s


def credits_earned_column_name_score(col: str) -> float:
    """
    **Earned** credits: explicit ``earned``, or cumulative/total credit columns **without**
    attempt markers (e.g. ``total_cumlative_credits`` → earned total).
    """
    c = _normalize_credit_column_name(col)
    if "credit" not in c:
        return 0.0
    has_att = credit_column_name_has_attempt_marker(col)
    if has_att and "earned" not in c:
        return 0.0
    s = 0.0
    if "earned" in c:
        s += 1.25
    if not has_att:
        if "cum" in c or "cumulative" in c:
            s += 0.85
        if "total" in c or "tot_" in c or c.startswith("inst") or "_inst_" in c:
            s += 0.55
        if "number_of" in c or "num_" in c or "nbr_" in c:
            s += 0.15
        if s < 0.9 and ("cum" in c or "cumulative" in c):
            s = max(s, 0.95)
    return s


def _pick_distinct_credit_columns(
    att_ranked: list[tuple[float, str]],
    ern_ranked: list[tuple[float, str]],
) -> tuple[str | None, str | None]:
    attempted_col = att_ranked[0][1] if att_ranked else None
    earned_col = ern_ranked[0][1] if ern_ranked else None
    if attempted_col and earned_col and attempted_col == earned_col:
        alt_e = next((c for _, c in ern_ranked if c != attempted_col), None)
        alt_a = next((c for _, c in att_ranked if c != earned_col), None)
        if alt_e is not None:
            earned_col = alt_e
        elif alt_a is not None:
            attempted_col = alt_a
        else:
            earned_col = None
    return attempted_col, earned_col


def _rank_columns_by_name_score(
    df: pd.DataFrame,
    *,
    column_score_fn: t.Callable[[str], float],
    name_hints: tuple[str, ...],
    min_base_score: float,
    tiebreak_fn: t.Callable[[pd.Series], float],
    tiebreak_weight: float,
    exclude_cols: t.AbstractSet[str] | None = None,
) -> list[tuple[float, str]]:
    """
    Rank dataframe columns by ``column_score_fn`` + hint bonus, with an optional tie-break
    from *tiebreak_fn(series)* (e.g. numeric coercion rate or string populated rate).
    """
    ranked: list[tuple[float, str]] = []
    skip = set(exclude_cols or ())
    for col in df.columns:
        if col in skip or pd.api.types.is_bool_dtype(df[col]):
            continue
        base = float(column_score_fn(col))
        base += 2.0 * term_column_name_hint_score(col, name_hints)
        if base < min_base_score:
            continue
        tb = tiebreak_fn(df[col])
        ranked.append((base + tiebreak_weight * tb, col))
    ranked.sort(key=lambda x: -x[0])
    return ranked


def _infer_two_credit_columns_by_name(
    df: pd.DataFrame,
    *,
    attempted_score_fn: t.Callable[[str], float],
    earned_score_fn: t.Callable[[str], float],
    attempted_name_hints: tuple[str, ...],
    earned_name_hints: tuple[str, ...],
    min_name_score: float = 0.45,
    numeric_tiebreak_weight: float = 0.12,
    max_sample: int = 8000,
) -> tuple[str | None, str | None]:
    """Shared ranking for attempted vs earned credit columns (numeric tie-break only)."""
    if len(df.columns) == 0:
        return None, None

    def _num_tb(ser: pd.Series) -> float:
        return _numeric_coercion_rate(ser, max_sample=max_sample)

    return _pick_distinct_credit_columns(
        _rank_columns_by_name_score(
            df,
            column_score_fn=attempted_score_fn,
            name_hints=attempted_name_hints,
            min_base_score=min_name_score,
            tiebreak_fn=_num_tb,
            tiebreak_weight=numeric_tiebreak_weight,
        ),
        _rank_columns_by_name_score(
            df,
            column_score_fn=earned_score_fn,
            name_hints=earned_name_hints,
            min_base_score=min_name_score,
            tiebreak_fn=_num_tb,
            tiebreak_weight=numeric_tiebreak_weight,
        ),
    )


def infer_inst_tot_credits_columns(
    df: pd.DataFrame,
    *,
    attempted_name_hints: tuple[
        str, ...
    ] = DEFAULT_INST_TOT_CREDITS_ATTEMPTED_NAME_HINTS,
    earned_name_hints: tuple[str, ...] = DEFAULT_INST_TOT_CREDITS_EARNED_NAME_HINTS,
    min_name_score: float = 0.45,
    numeric_tiebreak_weight: float = 0.12,
    max_sample: int = 8000,
) -> tuple[str | None, str | None]:
    """
    Infer credits **attempted** and **earned** by **column name** (cohort or semester).

    Cumulative/total credit fields **without** ``attempt`` / ``att`` tokens are treated as
    **earned** (e.g. misspelled ``total_cumlative_credits``). Attempted requires attempt
    markers or **credit hours**-style enrollment columns.
    """
    if df is None or len(df.columns) == 0:
        return None, None
    return _infer_two_credit_columns_by_name(
        df,
        attempted_score_fn=credits_attempted_column_name_score,
        earned_score_fn=credits_earned_column_name_score,
        attempted_name_hints=attempted_name_hints,
        earned_name_hints=earned_name_hints,
        min_name_score=min_name_score,
        numeric_tiebreak_weight=numeric_tiebreak_weight,
        max_sample=max_sample,
    )


# --- Course-row and semester aggregate columns (validate_credit_consistency / check_pf) ---

DEFAULT_COURSE_ROW_CREDITS_ATTEMPTED_NAME_HINTS: tuple[str, ...] = (
    "course_credits_attempted",
    "credits_attempted",
    "credit_hours",
    "credithours",
    "credit_hour",
    "class_credit_hours",
    "ug_credits_attempted",
    "attempted_credits",
    "enrollment_credits",
    "registered_credits",
)
DEFAULT_COURSE_ROW_CREDITS_EARNED_NAME_HINTS: tuple[str, ...] = (
    "course_credits_earned",
    "credits_earned",
    "number_of_credits_earned",
    "num_credits_earned",
    "no_of_credits_earned",
    "no._of_credits_earned",
    "credit_earned",
    "credits_earned_course",
)

DEFAULT_COURSE_GRADE_NAME_HINTS: tuple[str, ...] = (
    "grade",
    "class_grade",
    "course_grade",
    "letter_grade",
    "final_grade",
    "official_grade",
)
DEFAULT_COURSE_PF_NAME_HINTS: tuple[str, ...] = (
    "pass_fail",
    "passfail",
    "completion",
    "complete",
    "completion_status",
    "course_status",
    "enrollment_status",
)

DEFAULT_SEMESTER_COURSE_COUNT_NAME_HINTS: tuple[str, ...] = (
    "number_of_courses_enrolled",
    "number_of_courses",
    "courses_enrolled",
    "no_of_classes",
    "num_classes",
    "class_count",
    "n_courses",
    "course_count",
    "courses_taken",
)


def course_row_credits_attempted_name_score(col: str) -> float:
    """
    Course-enrollment **attempted** credits: e.g. *Credit Hours*, *credits attempted*, *units*.
    """
    c = _normalize_credit_column_name(col)
    if "earned" in c:
        if "hour" in c and "earned" in c:
            return 0.0
        if "attempt" not in c and "hour" not in c and "unit" not in c:
            return 0.0
    s = float(credits_attempted_column_name_score(col))
    if "hour" in c or "hrs" in c or c.endswith("_hr") or "_hr_" in c:
        s += 1.2
    if "unit" in c:
        s += 0.95
    if "credit" in c and "hour" in c and "earned" not in c:
        s = max(s, 1.4)
    if (
        "credit" in c
        and s < 0.55
        and not credit_column_name_has_attempt_marker(col)
        and "hour" not in c
        and "unit" not in c
    ):
        s = max(s, 0.52)
    return s


def course_row_credits_earned_name_score(col: str) -> float:
    """Course-row **earned** credits: e.g. *No. of Credits Earned*."""
    s = float(credits_earned_column_name_score(col))
    c = _normalize_credit_column_name(col)
    if "no_" in c or "number" in c or "nbr_" in c:
        s += 0.22
    return s


def infer_course_credit_columns(
    df: pd.DataFrame,
    *,
    attempted_name_hints: tuple[
        str, ...
    ] = DEFAULT_COURSE_ROW_CREDITS_ATTEMPTED_NAME_HINTS,
    earned_name_hints: tuple[str, ...] = DEFAULT_COURSE_ROW_CREDITS_EARNED_NAME_HINTS,
    min_name_score: float = 0.45,
    numeric_tiebreak_weight: float = 0.12,
    max_sample: int = 8000,
) -> tuple[str | None, str | None]:
    """
    Infer per-enrollment credits **attempted** and **earned** on a **course** file
    (e.g. *Credit Hours* vs *No. of Credits Earned*).
    """
    if len(df.columns) == 0:
        return None, None
    return _infer_two_credit_columns_by_name(
        df,
        attempted_score_fn=course_row_credits_attempted_name_score,
        earned_score_fn=course_row_credits_earned_name_score,
        attempted_name_hints=attempted_name_hints,
        earned_name_hints=earned_name_hints,
        min_name_score=min_name_score,
        numeric_tiebreak_weight=numeric_tiebreak_weight,
        max_sample=max_sample,
    )


def semester_course_count_column_name_score(col: str) -> float:
    """Semester-level count of classes/courses (not credit hours)."""
    c = _normalize_credit_column_name(col)
    if "credit" in c and "hour" in c:
        return 0.0
    if "credit" in c and "attempt" in c:
        return 0.0
    s = 0.0
    if "class" in c:
        s += 0.9
    if "classes" in c:
        s += 0.95
    if "course" in c and (
        "count" in c or "number" in c or "num" in c or "nbr" in c or "no_" in c
    ):
        s += 1.2
    if "enroll" in c and ("course" in c or "class" in c):
        s += 0.8
    if "sections" in c:
        s += 0.55
    return s


def _string_populated_rate(series: pd.Series, max_sample: int = 8000) -> float:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null
    t = sample.astype("string").str.strip()
    return float((t.str.len() > 0).mean())


def infer_semester_credit_aggregate_columns(
    df: pd.DataFrame,
    *,
    min_name_score: float = 0.45,
    numeric_tiebreak_weight: float = 0.12,
    count_min_name_score: float = 0.42,
    count_tiebreak_weight: float = 0.06,
    max_sample: int = 8000,
    attempted_name_hints: tuple[
        str, ...
    ] = DEFAULT_INST_TOT_CREDITS_ATTEMPTED_NAME_HINTS,
    earned_name_hints: tuple[str, ...] = DEFAULT_INST_TOT_CREDITS_EARNED_NAME_HINTS,
    course_count_name_hints: tuple[str, ...] = DEFAULT_SEMESTER_COURSE_COUNT_NAME_HINTS,
) -> tuple[str | None, str | None, str | None]:
    """
    Infer semester file columns: credits **attempted**, **earned**, and **course count**
    (e.g. *credit_hours*, *no_of_credits_earned*, *no_of_classes*).
    """
    if len(df.columns) == 0:
        return None, None, None

    att, earn = infer_inst_tot_credits_columns(
        df,
        attempted_name_hints=attempted_name_hints,
        earned_name_hints=earned_name_hints,
        min_name_score=min_name_score,
        numeric_tiebreak_weight=numeric_tiebreak_weight,
        max_sample=max_sample,
    )
    used = {c for c in (att, earn) if c}
    ranked = _rank_columns_by_name_score(
        df,
        column_score_fn=semester_course_count_column_name_score,
        name_hints=course_count_name_hints,
        min_base_score=count_min_name_score,
        tiebreak_fn=lambda ser: _string_populated_rate(ser, max_sample=max_sample),
        tiebreak_weight=count_tiebreak_weight,
        exclude_cols=used,
    )
    count_col = ranked[0][1] if ranked else None
    return att, earn, count_col


DEFAULT_SEMESTER_ENROLLMENT_INTENSITY_NAME_HINTS: tuple[str, ...] = (
    "student_term_enrollment_intensity",
    "enrollment_intensity",
    "ftpt",
    "full_part_time",
    "full_part",
    "full_time",
    "part_time",
    "time_status",
    "academic_load",
    "credit_load",
)


def semester_enrollment_intensity_column_name_score(col: str) -> float:
    """
    Semester-level **student** full-time vs part-time (enrollment intensity), not instructor FT/PT.
    """
    c = _normalize_credit_column_name(col)
    if "instructor" in c:
        return 0.0
    if "frac_" in c or "cumfrac" in c or "cum_frac" in c:
        return 0.0
    s = 0.0
    if c == "ftpt" or "_ftpt" in c or c.startswith("ftpt_"):
        s = max(s, 1.42)
    if "student_term_enrollment_intensity" in c:
        s = max(s, 1.38)
    if "enrollment_intensity" in c or "enroll_intensity" in c:
        s = max(s, 1.32)
    if "enroll" in c and "intensity" in c:
        s = max(s, 1.28)
    if "full" in c and "part" in c and "time" in c:
        s = max(s, 1.18)
    if ("full_time" in c or "part_time" in c) and "instructor" not in c:
        s = max(s, 1.08)
    if "time_status" in c or "timestatus" in c:
        s = max(s, 1.02)
    if "term" in c and "intensity" in c and "instructor" not in c:
        s = max(s, 1.0)
    if "load" in c and ("acad" in c or "enroll" in c):
        s = max(s, 0.55)
    return s


def infer_semester_enrollment_intensity_column(
    df: pd.DataFrame,
    *,
    exclude_cols: t.AbstractSet[str] | None = None,
    min_name_score: float = 0.40,
    tiebreak_weight: float = 0.06,
    max_sample: int = 8000,
    name_hints: tuple[str, ...] = DEFAULT_SEMESTER_ENROLLMENT_INTENSITY_NAME_HINTS,
) -> str | None:
    """
    Infer **full-time vs part-time** (enrollment intensity) on a **semester** / student-term file.

    Typical column names: *ftpt*, *student_term_enrollment_intensity*, *enrollment_intensity*.
    Excludes credit-hour and term-key columns when passed via ``exclude_cols``.
    """
    if len(df.columns) == 0:
        return None
    ranked = _rank_columns_by_name_score(
        df,
        column_score_fn=semester_enrollment_intensity_column_name_score,
        name_hints=name_hints,
        min_base_score=min_name_score,
        tiebreak_fn=lambda ser: _string_populated_rate(ser, max_sample=max_sample),
        tiebreak_weight=tiebreak_weight,
        exclude_cols=exclude_cols,
    )
    return ranked[0][1] if ranked else None


def course_grade_column_name_score(col: str) -> float:
    c = _normalize_credit_column_name(col)
    s = 0.0
    if "grade" in c:
        s += 1.2
    if "letter" in c:
        s += 0.45
    if "final" in c and "grade" in c:
        s += 0.35
    if "gpa" in c and "course" not in c and "class" not in c:
        s += 0.2
    if "mark" in c and "grade" not in c:
        s += 0.4
    if "midterm" in c or "mid_term" in c:
        s *= 0.35
    if "point" in c and "grade" not in c:
        return 0.0
    return s


def course_pass_fail_column_name_score(col: str) -> float:
    c = _normalize_credit_column_name(col)
    s = 0.0
    if "pass" in c and "fail" in c:
        s += 1.3
    if "pass_fail" in c or "passfail" in c:
        s += 1.25
    if "completion" in c or "complete" in c:
        s += 1.05
    if "status" in c:
        s += 0.5
    if "success" in c:
        s += 0.45
    return s


def infer_course_grade_pf_columns(
    df: pd.DataFrame,
    *,
    exclude_cols: t.AbstractSet[str] | None = None,
    min_name_score: float = 0.42,
    tiebreak_weight: float = 0.06,
    max_sample: int = 8000,
    grade_name_hints: tuple[str, ...] = DEFAULT_COURSE_GRADE_NAME_HINTS,
    pf_name_hints: tuple[str, ...] = DEFAULT_COURSE_PF_NAME_HINTS,
) -> tuple[str | None, str | None]:
    """
    Infer **grade** and **pass/fail or completion status** columns on a course file.

    Typical examples: *Class Class Grade*, *Class Completion Status*.
    """
    if len(df.columns) == 0:
        return None, None

    used: set[str] = set(exclude_cols or ())

    def pick(score_fn: t.Callable[[str], float], hints: tuple[str, ...]) -> str | None:
        ranked = _rank_columns_by_name_score(
            df,
            column_score_fn=score_fn,
            name_hints=hints,
            min_base_score=min_name_score,
            tiebreak_fn=lambda ser: _string_populated_rate(ser, max_sample=max_sample),
            tiebreak_weight=tiebreak_weight,
            exclude_cols=used,
        )
        return ranked[0][1] if ranked else None

    grade_col = pick(course_grade_column_name_score, grade_name_hints)
    if grade_col:
        used.add(grade_col)
    pf_col = pick(course_pass_fail_column_name_score, pf_name_hints)
    return grade_col, pf_col


def convert_numeric_columns(df, columns):
    """
    CUSTOM SCHOOL FUNCTION

    Converts string-based numeric columns to a numeric dtype for plotting purposes ONLY
    (we want to maintain dtypes in our modeling dataframe, so ONLY use this for help with EDA).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the columns to clean.
    columns : list of str
        List of column names to clean and convert.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns cleaned and converted to numeric.
    """
    df = df.copy()
    for col in columns:
        s = df[col].astype(str).str.strip()
        s = s.str.replace(",", "", regex=False)  # remove commas
        s = s.str.replace(
            r"[^\d.\-]", "", regex=True
        )  # keep only digits, dot (decimals), dash (negative sign)
        df[col] = pd.to_numeric(s, errors="coerce")
    return df


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


def duplicate_conflict_columns(
    df: pd.DataFrame, primary_keys: list[str]
) -> pd.DataFrame:
    dup = df[df.duplicated(subset=primary_keys, keep=False)]

    if dup.empty:
        return pd.DataFrame(columns=["column", "pct_conflicting_groups"])

    grp = dup.groupby(primary_keys, dropna=False)

    # For each group + column: does this column conflict?
    conflict = grp.nunique(dropna=False) > 1

    # Keep only groups that have *any* conflict
    conflict = conflict[conflict.any(axis=1)]

    if conflict.empty:
        return pd.DataFrame(columns=["column", "pct_conflicting_groups"])

    # Percent of conflicting groups where each column conflicts
    pct = conflict.mean().mul(100)

    return (
        pct.rename("pct_conflicting_groups")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("pct_conflicting_groups", ascending=False)
        .reset_index(drop=True)
    )


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
