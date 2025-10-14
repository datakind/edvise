"""Shared functionality for data processing, cleaning, aggregating, and feature engineering
"""

import datetime
import re

import numpy as np
import pandas as pd


term_course_sum_prefix = "term_n_courses_"
term_n_courses_enrolled_colname = term_course_sum_prefix + "enrolled"
sum_name = "total"
sum_prefix = sum_name + "_"
term_course_sum_hist_prefix = sum_prefix + "n_courses_"
term_flag_sum_hist_prefix = "n_terms_"

term_course_pct_prefix = "term_pct_courses_"
mean_name = "avg"

term_flag_prefix = "term_flag_"
term_number_col = "term_n"

min_prefix = "min_"
max_prefix = "max_"

hist_suffix = "_to_date"


def clean_column_name(col_name):
    """
    Clean and standardize a column name by replacing any characters that are not alphanumeric or underscores
    with a single underscore, converting the name to uppercase, and ensuring there are no sequences of two or
    more consecutive underscores.

    Args:
        col_name (str): The original column name to be cleaned.

    Returns:
        str: The cleaned and standardized column name, which is uppercase and contains only alphanumeric characters
             and underscores, with no sequences of multiple underscores.

    Example:
        >>> clean_column_name('Student Name (2020)')
        'student_name_2020'
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", col_name).lower()
    cleaned = re.sub(
        r"_{2,}", "_", cleaned
    )  # Ensure there are no 3 or more consecutive underscores
    return cleaned


def cumulative_list_aggregation(list_values_over_time):
    """Aggregate column values over time into
    cumulative lists. Example use case: we have a student term dataset
    containing the term numbers a student is enrolled in. Applying this
    function to that dataset using
    df.sort_values(term_rank_col).groupby(student_id_col).transform(cumulative_list_aggregation)
    would give us, at each term, a list of the terms a student has been enrolled in
    to date.

    Args:
        list_values_over_time (list): list of values to accumulate over time

    Returns:
        list of lists
    """
    out = [[]]
    for x in list_values_over_time:
        out.append(out[-1] + [x])
    return out[1:]


def calculate_pct_terms_unenrolled(
    student_term_df,
    possible_terms_list,
    new_col_prefix,
    term_rank_col="term_rank",
    student_id_col="Student.ID",
):
    """Calculate percent of a student's terms unenrolled to date.

    Args:
        student_term_df (pd.DataFrame): data at the student + term level, containing columns term_rank_col and student_id_col
        possible_terms_list (list): list of possible terms to consider. This can be all possible terms or a subset of terms,
            for example - only Fall/Spring terms
        new_col_prefix (str): prefix of new column indicating percent of a student's terms unenrolled to date, according to the scope of possible_terms_list
        term_rank_col (str, optional): column name of the column containing the student's term rank. Defaults to 'term_rank'.
        student_id_col (str, optional): column name of the column containing student IDs. Defaults to 'Student.ID'.

    Returns:
        pd.DataFrame: student_term_df with a new column, new_col_prefix + hist_suffix
    """
    student_term_df["term_ranks_enrolled_to_date"] = (
        student_term_df.sort_values(term_rank_col)
        .groupby(student_id_col)[term_rank_col]
        .transform(cumulative_list_aggregation)
    )
    student_term_df["first_enrolled_term_rank"] = [
        min(enrolled_ranks)
        for enrolled_ranks in student_term_df["term_ranks_enrolled_to_date"]
    ]
    student_term_df["possible_term_ranks_to_date"] = [
        [
            rank
            for rank in possible_terms_list
            if (row["first_enrolled_term_rank"] <= rank <= row[term_rank_col])
        ]
        for _, row in student_term_df.iterrows()
    ]
    student_term_df["skipped_term_ranks_to_date"] = [
        [
            rank
            for rank in row["possible_term_ranks_to_date"]
            if (rank not in row["term_ranks_enrolled_to_date"])
        ]
        for _, row in student_term_df.iterrows()
    ]

    possible_n_terms_col = term_flag_sum_hist_prefix + "possible" + hist_suffix
    unenrolled_n_terms_col = term_flag_sum_hist_prefix + "unenrolled" + hist_suffix
    student_term_df[possible_n_terms_col] = [
        len(possible_term_ranks)
        for possible_term_ranks in student_term_df["possible_term_ranks_to_date"]
    ]
    student_term_df[unenrolled_n_terms_col] = [
        len(skipped_term_ranks)
        for skipped_term_ranks in student_term_df["skipped_term_ranks_to_date"]
    ]
    student_term_df[new_col_prefix + hist_suffix] = (
        student_term_df[unenrolled_n_terms_col] / student_term_df[possible_n_terms_col]
    )
    student_term_df = student_term_df.drop(
        columns=[
            "term_ranks_enrolled_to_date",
            "first_enrolled_term_rank",
            "possible_term_ranks_to_date",
            "skipped_term_ranks_to_date",
            possible_n_terms_col,
            unenrolled_n_terms_col,
        ]
    )
    return student_term_df


def extract_course_level_from_course_number(num):
    """Across Texas, it is standard that the first digit of a course number indicates
    the course level (i.e. 100-level course, 200-level, etc.). This function extracts
    the course level from the course number.

    Args:
        num (str): course number

    Raises:
        Exception: Course number not expected, when the format of the course number is not handled in the function

    Returns:
        int: course level of the course number
    """

    stripped_num = num.strip()

    if re.match(r"[A-Z]", stripped_num[-1]):
        stripped_num = stripped_num[:-1]

    len_num = len(stripped_num)

    if len_num == 4:
        return int(stripped_num[0])
    elif len_num == 3:
        return 0
    else:
        raise Exception("Course number not expected.")


def course_data_to_student_term_level(
    df, groupby_cols, count_unique_cols, sum_cols, mean_cols, dummy_cols
):
    """Convert student-course data to student-term data by aggregating the following:

    Args:
        df (pd.DataFrame): data unique by student and course
        groupby_cols (list[str]): names of groupby columns for aggregation i.e. term columns,
            student IDs and characteristics, any semester-level variables that
            do not change across courses within a semester
        count_unique_cols (list[str]): names of columns to count(distinct) values from at the
            student-term level. Consider categorical columns with many possible values.
        sum_cols (list[str]): names of numeric columns to sum to the student-term level,
            for example - course credit columns (attempted, earned, etc.)
        mean_cols (list[str]): names of numeric columns to calculate the mean at the student-term level,
            for example - total class size
        dummy_cols (list[str]): names of categorical columns to convert into dummy variables and
            then sum up to the student-term level. Consider categorical columns with few possible
            levels (<5-10) or whose values (although many) may have a significant impact on the outcome
            variable.

    Note that any modification to these list arguments within the function will change
    the objects permanently, if used outside of the function.

    This was developed with NSC data in mind. From our call with NSC,
    we learned that data at the course level is less common, so we made
    intentional choices to make less detailed calculations at the course level,
    because we are thinking about how to generalize this module to other schools
    and data outside of NSC. We could consider adding more defined course-level
    features and aggregations later. For example, the "largest" course in terms
    of credits attempted over time, but right now this adds a lot of complexity
    to engineering historical features in a later processing step.

    Returns:
        pd.DataFrame: data unique by student and term
    """

    n_courses = (
        df.groupby(groupby_cols)
        .size()
        .reset_index()
        .rename(columns={0: term_n_courses_enrolled_colname})
    )

    dummies_df = (
        pd.get_dummies(df[groupby_cols + dummy_cols], dummy_na=True, columns=dummy_cols)
        .groupby(groupby_cols)
        .agg(["sum", "mean"])
    )
    dummies_df.columns = [
        (
            (term_course_sum_prefix + category)
            if fn == "sum"
            else (term_course_pct_prefix + category)
        )
        for category, fn in dummies_df.columns
    ]

    derived_cols_dict = {
        col: [] for col in set(count_unique_cols + sum_cols + mean_cols)
    }
    for count_unique_col in count_unique_cols:
        derived_cols_dict[count_unique_col].append(("nunique", pd.Series.nunique))
    for sum_col in sum_cols:
        derived_cols_dict[sum_col].append((sum_name, "sum"))
    for mean_col in mean_cols:
        derived_cols_dict[mean_col].append((mean_name, "mean"))
    derived_df = df.groupby(groupby_cols).agg(derived_cols_dict).reset_index()
    if derived_df.columns.nlevels == 2:
        derived_df.columns = [
            f"term_{col2}_{col1}" if col2 != "" else col1
            for col1, col2 in derived_df.columns
        ]

    student_term_df = n_courses.merge(dummies_df.reset_index(), on=groupby_cols).merge(
        derived_df, on=groupby_cols
    )

    return student_term_df


def extract_year_season(term_data):
    """Extract calendar year and season from term.

    Args:
        term_data (pd.Series): column of term data in the format YYYYTT, where
           YYYY is the calendar year, and TT denotes the term. For example, FA
           is Fall, SP is Spring, S1 and S2 are summer terms.

    Returns:
        pd.DataFrame: containing two columns - year and season
    """
    year_season_cols = term_data.str.extract(
        "^([0-9]{4})([a-zA-Z0-9]{2})$", expand=True
    )
    null_bool_index = year_season_cols.isna().any(axis=1)
    if (null_bool_index.sum() > 0) and (term_data[null_bool_index].notnull().sum() > 0):
        raise Exception(
            f"Term format not expected: {term_data[null_bool_index].unique()} Please revise the function and try again!"
        )
    year_season_cols[0] = pd.to_numeric(year_season_cols[0])
    return year_season_cols


def convert_number_of_courses_cols_to_term_flag_cols(df, col_prefix, orig_col):
    """For some columns specifying how many courses a student took with that
     characteristic that term, it may be less noisy and more predictive to instead
     convert these columns to a 1/0 flag to indicate if a student took a course
     with that characteristic this term.

     Subject has a lot of possible values, so the
     numeric variables for number of courses taken in each subject are very sparse
     and not that varied. Intuitively, it may be more predictive of completion (or not)
     if a student took,
     for example, a math class this semester, versus how many math classes they took
     (likely only 1 anyways) or the percent of classes that they took in math. Subject
     data is a good candidate for this function. We also can make this choice
     to drop the number of courses taken in a Subject from any later aggregations - we can
     get at this through Major and Major department (v2).

     For other columns, like grade, it may be more important for completion, for example,
     how many F's a student received, versus if they received an F at all. Grade data can
     be used in this function, but it's likely that you'll want to keep the raw number or
     percent of grades in each category, as well as the flag.

    Args:
        df (pd.DataFrame): data containing columns prefixed with col_prefix + orig_col
        col_prefix (str): prefix of columns to use to convert to flag columns
        orig_col (str): name of original column aggregated to columns starting with col_prefix.
            For example, "Subject" uses the columns prefixed with f"{col_prefix}Subject" columns to convert to
            to flag columns

    Returns:
        pd.DataFrame: original data containing the flag columns and excluding
            the numeric columns
    """
    n_courses_cols = [
        col for col in df.columns if col.startswith(col_prefix + orig_col)
    ]
    term_flag_df = (
        df[n_courses_cols]
        .mask(df[n_courses_cols] > 0, 1)
        .rename(
            mapper=lambda x: x.replace(col_prefix, "taking_course_") + "_this_term",
            axis="columns",
        )
    )
    full_df = pd.concat([df, term_flag_df], axis=1).drop(columns=n_courses_cols)
    return full_df


# TODO: test
def assign_month_to_season(season):
    """Assign a season to a month, for creating a datetime object.

    Args:
        season (str): season indicator, with possible values: Fall, FA,
            Winter, Spring, SP, S1, Summer, S2

    Raises:
        Exception: Season indicator not expected.

    Returns:
        int: month number of season
    """
    if season in ["Fall", "FA"]:
        return 12
    if season in ["Winter"]:
        return 2
    if season in ["Spring", "SP"]:
        return 6
    if season in ["S1"]:
        return 7
    if season in ["Summer", "S2"]:
        return 8
    else:
        raise Exception(f"Season {season} not expected. Try again!")


def create_terms_lkp(min_year, max_year, possible_seasons):
    """Create a dataframe of all possible terms.

    Args:
        min_year (str or int): earliest possible academic year
        max_year (str or int): latest possible academic year
        possible_seasons (pd.DataFrame): contains columns "season" and "order",
            where "season" indicates fall, spring, etc. in the format the school
            uses, and "order" is used to sort the seasons

    Returns:
        pd.DataFrame: all possible terms across the time frame, along with the rank order of each term,
            term_rank, the academic and calendar year, and the season
    """
    years = list(range(int(min_year), int(max_year) + 1))
    years = pd.DataFrame({"academic_year": [str(year) for year in years]})

    # doing this cross-join because one of our custom schools dropped the S2 term, but
    # for our definition of the outcome variable, we need each year to have the same number of terms
    terms_lkp = years.merge(possible_seasons, how="cross")
    terms_lkp["term_order"] = (
        terms_lkp["academic_year"] + terms_lkp["order"].astype(str)
    ).astype(int)
    # For one of our custom schools, term_order indicates the year of the fall of that academic year.
    # We define the calendar year as the next year for any season other than the Fall.
    terms_lkp["calendar_year"] = np.where(
        terms_lkp["season"] != "FA",
        terms_lkp["academic_year"].astype(int) + 1,
        terms_lkp["academic_year"],
    ).astype(int)
    terms_lkp["term"] = terms_lkp["calendar_year"].astype(str) + terms_lkp["season"]

    # The date created here itself is somewhat arbitrary
    # but can be used for windowing functions are better to look back 365 days rather
    # than x rows, etc.
    terms_lkp["term_end_date"] = terms_lkp[["season", "calendar_year"]].apply(
        lambda x: pd.to_datetime(
            str(assign_month_to_season(x.season)) + "-01-" + str(x.calendar_year)
        ),
        axis=1,
    )

    terms_lkp["term_rank"] = terms_lkp["term_order"].rank(method="dense")
    return terms_lkp.sort_values("term_rank")


def create_term_end_date(academic_year, season):
    """Create term end date from an academic year and season

    Args:
        academic_year (str): school year in the format YYYY-YY
        season (str): Fall, Winter, Spring, or Summer

    Raises:
        Exception: if season is not one of the standard NSC seasons

    Returns:
        datetime
    """
    if season == "Fall":
        year = academic_year.split("-")[0]
    elif season in ["Winter", "Spring", "Summer"]:
        year = "20" + academic_year.split("-")[1]
    else:
        raise Exception(f"Invalid season {season}")

    month = assign_month_to_season(season)

    return pd.to_datetime(f"{month}-01-{year}")


def add_historical_features_student_term_data(
    df, student_id_col, sort_cols, gpa_cols, num_cols
):
    """Append cumulative and rolling historical features to student-term level data

    Args:
        df (pd.DataFrame): data unique at the student-term level
        student_id_col (str): name of column containing student IDs
        sort_cols (list[str]): time or date-based columns to sort by. If the dataset contains
            a date for the observation, enter this column only in the list.
            If the dataset does not contain a date, include the year column as the first element of
            the list, and the season column as the second element of the list. Year should be in
            the format YYYY-YY of the academic year, and season should be one of Fall, Winter, Spring,
            Summer
        gpa_cols (list[str]): names of columns containing GPA data
        num_cols (list[str]): names of other numeric columns. Note that columns starting with
            'nunique_' and {term_course_sum_prefix} are already included in the aggregations.

    Note that any modification to these list arguments within the function will change
    the objects permanently, if used outside of the function.

    Returns:
        pd.DataFrame: student-term level data, with appended cumulative and rolling historical features
    """

    if len(sort_cols) == 2:
        year_col = sort_cols[0]
        season_col = sort_cols[1]
        df["Term End Date"] = df.apply(
            lambda x: create_term_end_date(x[year_col], x[season_col]), axis=1
        )
        sort_cols += ["Term End Date"]

    sorted_df = df.sort_values([student_id_col] + sort_cols, ascending=True)

    # term number - note that this is the student's term number and they do not necessarily indicate consecutive terms enrolled
    sorted_df[term_number_col] = sorted_df.groupby([student_id_col]).cumcount() + 1

    num_cols.append(term_n_courses_enrolled_colname)
    num_cols = list(set(num_cols))

    # Note: The old way of implementing has been buggy:
    # groupby().expanding().agg({column1: [fn1, fn2, fn3],
    #                            column2: [fn1, fn3, fn4]})
    # Since DataFrame.groupby().transform() does not accept a dictionary like DataFrame.transform() does,
    # implementing this more verbose way for accuracy, even though there may be a more concise way of doing this
    cumul_min_df = sorted_df.groupby(student_id_col)[num_cols + gpa_cols].transform(
        "cummin"
    )  # lowest X so far
    cumul_min_df.columns = [
        min_prefix + orig_col + hist_suffix for orig_col in cumul_min_df.columns.values
    ]

    cumul_max_df = sorted_df.groupby(student_id_col)[num_cols + gpa_cols].transform(
        "cummax"
    )  # highest X so far
    cumul_max_df.columns = [
        max_prefix + orig_col + hist_suffix for orig_col in cumul_max_df.columns.values
    ]

    # adding this only for numeric columns separate from GPA columns because it doesn't make sense to get a cumulative sum of GPA
    dummy_sum_cols = [
        dummy_sum_col
        for dummy_sum_col in df.columns.values
        if (
            dummy_sum_col.startswith((term_course_sum_prefix, term_flag_prefix))
            and (dummy_sum_col not in num_cols + gpa_cols)
        )
    ]
    cumul_sum_df = sorted_df.groupby(student_id_col)[
        list(set(num_cols + dummy_sum_cols))
    ].transform(
        "cumsum"
    )  # total X so far
    cumul_sum_df.columns = [
        (
            orig_col.replace(term_course_sum_prefix, term_course_sum_hist_prefix)
            + hist_suffix
            if orig_col.startswith(term_course_sum_prefix)
            else (
                orig_col.replace(term_flag_prefix, term_flag_sum_hist_prefix)
                + hist_suffix
                if orig_col.startswith(term_flag_prefix)
                else sum_prefix + orig_col + hist_suffix
            )
        )
        for orig_col in cumul_sum_df.columns.values
    ]

    mean_cols = [
        col
        for col in sorted_df.columns.values
        if col.startswith(("term_nunique_", f"term_{mean_name}"))
    ]
    cumul_avg_df = (
        sorted_df.groupby(student_id_col)[num_cols + gpa_cols + mean_cols]
        .expanding()
        .mean()
        .reset_index()
    )
    cumul_avg_df = cumul_avg_df.drop(columns=[student_id_col, "level_1"])
    cumul_avg_df.columns = [
        mean_name + "_" + orig_col + hist_suffix
        for orig_col in cumul_avg_df.columns.values
    ]

    # TODO: rolling std for num_cols + gpa_cols
    # def std(x):  # ensure that we are using population standard deviation
    #     return np.std(x, ddof=0)

    student_term_hist_df = pd.concat(
        [sorted_df, cumul_min_df, cumul_max_df, cumul_sum_df, cumul_avg_df], axis=1
    )

    # make sure no rows got dropped
    assert (
        cumul_max_df.shape[0]
        == cumul_min_df.shape[0]
        == cumul_sum_df.shape[0]
        == cumul_avg_df.shape[0]
        == sorted_df.shape[0]
        == student_term_hist_df.shape[0]
    )

    return student_term_hist_df


def get_sum_hist_terms_or_courses_prefix(level):
    """Get prefix used to identify columns created using addition over time
    for either course- or term-level features.

    Args:
        level (str): course or term

    Raises:
        Exception: if level is not one of "course" or "term"

    Returns:
        str
    """
    if level == "course":
        return term_course_sum_hist_prefix
    if level == "term":
        return term_flag_sum_hist_prefix
    else:
        raise Exception(f"Level {level} not expected. Try again!")


def get_n_terms_or_courses_col(level):
    """Get column name of a student's total courses over time or total terms over time

    Args:
        level (str): course or term

    Raises:
        Exception: if level is not one of "course" or "term"

    Returns:
        str
    """
    if level == "course":
        return term_course_sum_hist_prefix + "enrolled" + hist_suffix
    if level == "term":
        return term_number_col
    else:
        raise Exception(f"Level {level} not expected. Try again!")


def get_sum_hist_terms_or_courses_cols(df, level):
    """Get column names from a dataframe created using addition over time
    for either course- or term-level features.

    Args:
        df (pd.DataFrame): contains column names prefixed by get_sum_hist_terms_or_courses_prefix()
            and the column get_n_terms_or_courses_col()
        level (str): course or term

    Returns:
        list[str]
    """
    orig_prefix = get_sum_hist_terms_or_courses_prefix(level)
    denominator_col = get_n_terms_or_courses_col(level)
    return [
        col
        for col in df.columns
        if col.startswith(orig_prefix) and col != denominator_col
    ]


def calculate_pct_terms_or_courses_hist(df, level):
    """Calculate percent of terms or courses with a particular characteristic
    across a student's history so far

    Args:
        df (pd.DataFrame): contains column names prefixed by get_sum_hist_terms_or_courses_prefix()
            and the column get_n_terms_or_courses_col()
        level (str): course or term

    Returns:
        pd.DataFrame: df with new percent of terms or courses columns
    """
    orig_prefix = get_sum_hist_terms_or_courses_prefix(level)
    denominator_col = get_n_terms_or_courses_col(level)
    numerator_cols = get_sum_hist_terms_or_courses_cols(df, level)
    
    # removes duplicates, keeps order
    numerator_cols = list(dict.fromkeys(numerator_cols))
    
    print(f"Calculating percent of {level}s to date for {len(numerator_cols)} columns")
    print(f"Sample of columns: {numerator_cols[:5]}")
    print(f"Denominator column: {denominator_col}")
    new_colnames = [
        col.replace(orig_prefix, f"pct_{level}s_") for col in numerator_cols
    ]
    df[new_colnames] = df.loc[:, numerator_cols].div(df[denominator_col], axis=0)
    return df


def add_cumulative_nunique_col(df, sort_cols, groupby_cols, colname):
    """Calculate number of unique values within a group over time

    Args:
        df (pd.DataFrame): historical student data containing sort_cols, groupby_cols, and colname to count unique values
        sort_cols (list[str]): list of columns to sort by. For example, term or date.
        groupby_cols (list[str]): list of columns to group within. For example, Student ID.
        colname (str): column name to count unique values of over time

    Returns:
        pd.DataFrame: original data frame with new nunique_ column calculated over time.
    """
    sorted_df = df.sort_values(groupby_cols + sort_cols)
    new_colname = f"nunique_{colname}{hist_suffix}"
    sorted_df[new_colname] = (
        sorted_df.drop_duplicates(groupby_cols + [colname])
        .groupby(groupby_cols)
        .cumcount()
        + 1
    )
    sorted_df[new_colname] = sorted_df[new_colname].ffill()
    return sorted_df


def calculate_avg_credits_rolling(df, date_col, student_id_col, credit_col, n_days):
    """Calculate average credits per term enrolled within a time period.

    Args:
        df (pd.DataFrame): term dataset containing date_col, student_id_col, and credit_col
        date_col (str): column name of column containing dates of terms to use in windowing function
        student_id_col (str): column name of column containing student ID, for grouping the windowing function
        credit_col (str): column name of column containing term-level credits to aggregate
        n_days (int): number of days to calculate average credits across

    Returns:
        pd.DataFrame: df with new columns rolling_credit_col, rolling_n_terms_col, and rolling_avg_col
    """
    window = f"{n_days}D"
    rolling_credit_col = sum_prefix + credit_col + "_" + window + hist_suffix
    rolling_n_terms_col = "n_terms_enrolled_" + window + hist_suffix
    rolling_avg_col = f"{mean_name}_{credit_col}_" + window + hist_suffix

    # When rolling() is specified by an integer window, min_periods defaults to this integer.
    # That means if, for example, we wanted to calculate something across a window of 4 terms,
    # the rolling window is null early in the student's history when they don't yet have 4 terms.
    # We use a timedelta window instead to avoid this.
    grouped_rolling_df = (
        df.sort_values(date_col)
        .groupby(student_id_col)
        .rolling(window=window, on=date_col)
    )

    attempted_credits_last_365 = (
        grouped_rolling_df[credit_col]
        .sum()
        .reset_index()
        .rename(columns={credit_col: rolling_credit_col})
    )
    terms_enrolled_last_365 = grouped_rolling_df[date_col].count()
    terms_enrolled_last_365.name = rolling_n_terms_col

    merged_df = (
        df.merge(attempted_credits_last_365, on=[student_id_col, date_col])
        .merge(terms_enrolled_last_365.reset_index(), on=[student_id_col, date_col])
        .sort_values([student_id_col, date_col])
    )
    merged_df[rolling_avg_col] = (
        merged_df[rolling_credit_col] / merged_df[rolling_n_terms_col]
    )

    return merged_df
