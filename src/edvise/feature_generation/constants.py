DEFAULT_MIN_PASSING_GRADE = 1.0
DEFAULT_MIN_NUM_CREDITS_FULL_TIME = 12.0
DEFAULT_COURSE_CREDIT_CHECK = 12
DEFAULT_FIRST_TERM_OF_YEAR = "FALL"
DEFAULT_CORE_TERMS = {"FALL", "SPRING"}
DEFAULT_COURSE_LEVEL_PATTERN = r"^(?P<course_level>\d)\d{2}(?:[A-Z]{,2})?$"
DEFAULT_PEAK_COVID_TERMS = {
    ("2019-20", "SPRING"),  # Spring 2020
    ("2019-20", "SUMMER"),  # Summer 2020
    ("2020-21", "FALL"),  # Fall 2020
    ("2020-21", "WINTER"),  # Winter 2020/2021
    ("2020-21", "SPRING"),  # Spring 2021
    ("2020-21", "SUMMER"),  # Summer 2021
}
DEFAULT_SEASON_ORDER_MAP = {
    "spring": 1,
    "summer": 2,
    "fall": 3,
    "winter": 4,
}

NUM_COURSE_FEATURE_COL_PREFIX = "num_courses"
FRAC_COURSE_FEATURE_COL_PREFIX = "frac_courses"
DUMMY_COURSE_FEATURE_COL_PREFIX = "took"
# Cumulative counterpart of ``num_courses_*`` dummy expansions (see cumulative.py).
CUMFRAC_NUM_COURSE_FEATURE_COL_PREFIX = "cumfrac_num_courses"

# CourseInputColumns attrs whose physical values are get_dummies'd into
# num_courses_* / frac_courses_* / cumfrac_num_courses_* features.
# Keep in sync with features_table.toml regex alternations.
COURSE_DUMMY_AGG_INPUT_ATTRS: tuple[str, ...] = (
    "course_type",
    "delivery_method",
    "math_or_english_gateway",
    "co_requisite_course",
    "course_instructor_employment_status",
    "course_instructor_rank",
)

# Always-named derived columns also dummy-expanded (not CourseInputColumns attrs).
COURSE_DUMMY_AGG_FIXED_COLUMNS: tuple[str, ...] = (
    "course_level",
    "course_grade",
)

# Prefixes produced for each dummy token × value (exact / regex lookup vocabulary).
COURSE_DUMMY_EXPANSION_PREFIXES: tuple[str, ...] = (
    NUM_COURSE_FEATURE_COL_PREFIX,
    FRAC_COURSE_FEATURE_COL_PREFIX,
    CUMFRAC_NUM_COURSE_FEATURE_COL_PREFIX,
)

# CONSTANTS FROM CUSTOM SCHOOL PROCESSING
TERM_COURSE_SUM_PREFIX = "term_n_courses_"
TERM_N_COURSES_ENROLLED_COLNAME = TERM_COURSE_SUM_PREFIX + "enrolled"
SUM_NAME = "total"
SUM_PREFIX = SUM_NAME + "_"
TERM_COURSE_SUM_HIST_PREFIX = SUM_PREFIX + "n_courses_"
TERM_FLAG_SUM_HIST_PREFIX = "n_terms_"

TERM_COURSE_PROP_PREFIX = "term_prop_courses_"
MEAN_NAME = "avg"

TERM_FLAG_PREFIX = "term_flag_"
TERM_NUMBER_COL = "term_n"

MIN_PREFIX = "min_"
MAX_PREFIX = "max_"

HIST_SUFFIX = "_to_date"
