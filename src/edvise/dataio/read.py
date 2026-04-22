import logging
import mlflow
import pandas as pd
import typing as t
import pydantic as pyd
import pyspark.sql
import pathlib
import edvise.utils as utils

try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib  # noqa
try:
    import pandera as pda
    from pandera.errors import SchemaErrors
except ModuleNotFoundError:
    utils.databricks.mock_pandera()
    import pandera as pda
    from pandera.errors import SchemaErrors


LOGGER = logging.getLogger(__name__)

S = t.TypeVar("S", bound=pyd.BaseModel)

# Disable mlflow autologging (due to Databricks issues during feature selection)
mlflow.autolog(disable=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Ignore Databricks logger


def read_config(file_path: str, *, schema: type[S]) -> S:
    """
    Read config from ``file_path`` and validate it using ``schema`` ,
    returning an instance with parameters accessible by attribute.
    """
    try:
        cfg = from_toml_file(file_path)
        return schema.model_validate(cfg)
    except FileNotFoundError:
        LOGGER.error("Configuration file not found at %s", file_path)
        raise
    except Exception as e:
        LOGGER.error("Error reading configuration file: %s", e)
        raise


def from_csv_file(
    file_path: str,
    spark_session: t.Optional[pyspark.sql.SparkSession] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Read data from a CSV file at ``file_path`` and return it as a DataFrame.

    Args:
        file_path: Path to file on disk from which data will be read.
        spark_session: If given, data is loaded via ``pyspark.sql.DataFrameReader.csv`` ;
            otherwise, data is loaded via :func:`pandas.read_csv()` .
        **kwargs: Additional arguments passed as-is into underlying read func.

    See Also:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        - https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.csv.html
    """
    if spark_session is None:
        df = pd.read_csv(file_path, dtype="string", header="infer", **kwargs)  # type: ignore
    else:
        df = spark_session.read.csv(
            file_path,
            inferSchema=False,
            header=True,
            **kwargs,  # type: ignore
        ).toPandas()
    assert isinstance(df, pd.DataFrame)  # type guard
    LOGGER.info("loaded rows x cols = %s from '%s'", df.shape, file_path)
    return df


def from_delta_table(
    table_path: str, spark_session: pyspark.sql.SparkSession
) -> pd.DataFrame:
    """
    Read data from a table in Databricks Unity Catalog and return it as a DataFrame.

    Args:
        table_path: Path in Unity Catalog from which data will be read,
            including the full three-level namespace: ``catalog.schema.table`` .
        spark_session: Entry point to using spark dataframes and the databricks integration.

    See Also:
        - https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.html#pyspark.sql.DataFrameReader
        - https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/spark_session.html
    """
    df = spark_session.read.format("delta").table(table_path).toPandas()
    assert isinstance(df, pd.DataFrame)  # type guard
    LOGGER.info("loaded rows x cols = %s of data from '%s'", df.shape, table_path)
    return df


def from_toml_file(file_path: str) -> dict[str, object]:
    """
    Read data from ``file_path`` and return it as a dict.

    Args:
        file_path: Path to file on disk from which data will be read.
    """
    fpath = pathlib.Path(file_path).resolve()
    with fpath.open(mode="rb") as f:
        data = tomllib.load(f)
    LOGGER.info("loaded config from '%s'", fpath)
    assert isinstance(data, dict)  # type guard
    return data


def read_features_table(file_path: str) -> dict[str, dict[str, str]]:
    """
    Read a features table mapping columns to readable names and (optionally) descriptions
    from a TOML file located at ``fpath``, which can either refer to a relative path in this
    package or an absolute path loaded from local disk.

    Args:
        file_path: Path to features table TOML file relative to package root or absolute;
            for example: "assets/pdp/features_table.toml" or "/path/to/features_table.toml".
    """
    pkg_root_dir = next(
        p for p in pathlib.Path(__file__).parents if p.parts[-1] == "edvise"
    )
    fpath = (
        pathlib.Path(file_path)
        if pathlib.Path(file_path).is_absolute()
        else pkg_root_dir / file_path
    )
    features_table = from_toml_file(str(fpath))
    LOGGER.info("loaded features table from '%s'", fpath)
    return features_table  # type: ignore


def read_parquet(
    path: str, dtype: t.Optional[dict] = None, verbose: bool = False
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if dtype:
        df = df.astype(dtype)
    if verbose:
        print(f"Read {df.shape[0]} rows from {path}")
    return df


def _read_and_prepare_pdp_data(
    *,
    file_path: t.Optional[str],
    table_path: t.Optional[str],
    spark_session: t.Optional[pyspark.sql.SparkSession],
    schema: t.Optional[type[pda.DataFrameModel]],
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]],
    dttm_format: t.Optional[str] = None,
    string_cols_to_uppercase: t.Sequence[str] = (),
    datetime_cols: t.Sequence[str] = (),
    null_replacements: dict[str, str | list[str]] = {},
    bool_cols: t.Sequence[str] = (),
    **kwargs: object,
) -> pd.DataFrame:
    if not bool(table_path) ^ bool(file_path):
        raise ValueError("exactly one of table_path or file_path must be specified")
    elif table_path and not spark_session:
        raise ValueError("spark session must be given when reading data from table")

    df = (
        from_csv_file(file_path, spark_session, **kwargs)  # type: ignore
        if file_path
        else from_delta_table(table_path, spark_session)  # type: ignore
    )

    df = df.rename(columns=utils.data_cleaning.convert_to_snake_case)

    transformations: dict[str, pd.Series] = {}

    # String -> uppercase
    for col in string_cols_to_uppercase:
        transformations[col] = utils.data_cleaning.uppercase_string_values(df, col=col)

    # Parse datetime
    for col in datetime_cols:
        transformations[col] = utils.data_cleaning.parse_dttm_values(
            df, col=col, fmt=dttm_format or "%Y%m%d"
        )

    # Replace null values
    for col, to_replace in null_replacements.items():
        transformations[col] = utils.data_cleaning.replace_values_with_null(
            df, col=col, to_replace=to_replace
        )

    # Convert to bool
    for col in bool_cols:
        transformations[col] = utils.data_cleaning.cast_to_bool_via_int(df, col=col)

    if transformations:
        df = df.assign(**transformations)

    return _maybe_convert_maybe_validate_data(df, converter_func, schema)


def read_raw_pdp_course_data(
    *,
    table_path: t.Optional[str] = None,
    file_path: t.Optional[str] = None,
    schema: t.Optional[type[pda.DataFrameModel]] = None,
    dttm_format: str = "%Y%m%d",
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    spark_session: t.Optional[pyspark.sql.SparkSession] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Read raw PDP course data from table (in Unity Catalog) or file (in CSV format),
    and parse+validate it via ``schema`` .

    Args:
        table_path
        file_path
        schema: "DataFrameModel", such as those specified in :mod:`schemas` ,
            used to parse and validate the raw data. If None, parsing/validation
            is skipped, and the raw data is returned as-is.
        dttm_format: Datetime format for "Course Begin/End Date" columns.
        converter_func: If the raw data is incompatible with ``schema`` ,
            provide a function that takes the raw dataframe as its sole input,
            performs whatever (minimal) transformations necessary to bring the data
            into line with ``schema`` , and then returns it. This converted dataset
            will then be passed into ``schema`` , if specified.
            NOTE: Allowances for minor differences in the data should be implemented
            on the school-specific schema class directly. This function is intended
            to handle bigger problems, such as duplicate ids or borked columns.
        spark_session: Required if reading data from ``table_path`` , and optional
            if reading data from ``file_path`` .
        **kwargs: Additional arguments passed as-is into underlying read func.
            Note that raw data is always read in as "string" dtype, then coerced
            into the correct dtypes using ``schema`` .

    See Also:
        - :func:`read_data_from_csv_file()`
        - :func:`read_data_from_delta_table()`

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/course-level-analysis-ready-file-data-dictionary
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        - https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.dataframe.model.DataFrameModel.html#pandera.api.dataframe.model.DataFrameModel.validate
    """
    return _read_and_prepare_pdp_data(
        table_path=table_path,
        file_path=file_path,
        schema=schema,
        spark_session=spark_session,
        converter_func=converter_func,
        dttm_format=dttm_format,
        string_cols_to_uppercase=("academic_term",),
        datetime_cols=("course_begin_date", "course_end_date"),
        **kwargs,  # type: ignore
    )


def read_raw_pdp_cohort_data(
    *,
    table_path: t.Optional[str] = None,
    file_path: t.Optional[str] = None,
    schema: t.Optional[type[pda.DataFrameModel]] = None,
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    spark_session: t.Optional[pyspark.sql.SparkSession] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Read raw PDP cohort data from table (in Unity Catalog) or file (in CSV format),
    and parse+validate it via ``schema`` .

    Args:
        table_path
        file_path
        schema: "DataFrameModel", such as those specified in :mod:`schemas` ,
            used to parse and validate the raw data. If None, parsing/validation
            is skipped, and the raw data is returned as-is.
        converter_func: If the raw data is incompatible with ``schema`` ,
            provide a function that takes the raw dataframe as its sole input,
            performs whatever (minimal) transformations necessary to bring the data
            into line with ``schema`` , and then returns it. This converted dataset
            will then be passed into ``schema`` , if specified.
            NOTE: Allowances for minor differences in the data should be implemented
            on the school-specific schema class directly. This function is intended
            to handle bigger problems, such as duplicate ids or borked columns.
        spark_session: Required if reading data from ``table_path`` , and optional
            if reading data from ``file_path`` .
        **kwargs: Additional arguments passed as-is into underlying read func.
            Note that raw data is always read in as "string" dtype, then coerced
            into the correct dtypes using ``schema`` .

    See Also:
        - :func:`read_data_from_csv_file()`
        - :func:`read_data_from_delta_table()`

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/cohort-level-analysis-ready-file-data-dictionary
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        - https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.dataframe.model.DataFrameModel.html#pandera.api.dataframe.model.DataFrameModel.validate
    """
    return _read_and_prepare_pdp_data(
        table_path=table_path,
        file_path=file_path,
        schema=schema,
        spark_session=spark_session,
        converter_func=converter_func,
        string_cols_to_uppercase=("cohort_term",),
        null_replacements={
            "gpa_group_term_1": "UK",
            "gpa_group_year_1": "UK",
        },
        bool_cols=("retention", "persistence"),
        **kwargs,  # type: ignore
    )


def _maybe_convert_maybe_validate_data(
    df: pd.DataFrame,
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    schema: t.Optional[type[pda.DataFrameModel]] = None,
) -> pd.DataFrame:
    if converter_func is not None:
        LOGGER.info("applying %s converter to raw data", converter_func)
        df = converter_func(df)
    if schema is None:
        return df
    else:
        try:
            df_validated = schema.validate(df, lazy=True)
            assert isinstance(df_validated, pd.DataFrame)
            return df_validated
        except SchemaErrors:
            LOGGER.error("unable to parse/validate raw data")
            raise


def _prepare_edvise_cohort_after_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cohort prep after Edvise schema validation.

    Keeps native Edvise column names (including ``learner_id``). Adds ``study_id``
    when missing and ``entry_type`` (for readmit logic). ``institution_id`` and
    joins use :attr:`student_id_col` from the project config (see
    :class:`edvise.configs.es.ESProjectConfig`). No synthetic ``student_id``
    column is created.
    """
    out = df.copy()
    out["learner_id"] = out["learner_id"].astype("string")
    if "study_id" not in out.columns:
        out["study_id"] = out["learner_id"]
    if "enrollment_type" in out.columns:
        out["entry_type"] = (
            out["enrollment_type"].astype("string").str.strip().str.lower()
        )
    return out


def _load_cohort_lookup_next_to_course(course_file_path: str) -> pd.DataFrame:
    """Load a student CSV beside the course file (for cohort / cohort_term on course rows)."""
    parent = pathlib.Path(course_file_path).resolve().parent
    for name in ("edvise_students_full.csv", "edvise_students.csv"):
        p = parent / name
        if p.is_file():
            raw = from_csv_file(str(p), None)
            return raw.rename(columns=utils.data_cleaning.convert_to_snake_case)
    raise FileNotFoundError(
        f"No edvise_students_full.csv or edvise_students.csv next to {course_file_path!r}"
    )


def _prepare_edvise_course_after_validation(
    df: pd.DataFrame, *, cohort_lookup: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge cohort entry keys onto Edvise course rows for pre-cohort filtering.

    Adds ``cohort`` / ``cohort_term`` (Clearinghouse-style names) because
    :func:`~edvise.utils.data_cleaning.remove_pre_cohort_courses` expects them on
    the course frame. Native Edvise course field names are otherwise preserved;
    PDP-only feature columns are added in ``ESCourseStandardizer``.
    """
    sub = cohort_lookup[["learner_id", "entry_year", "entry_term"]].copy()
    sub["learner_id"] = sub["learner_id"].astype("string")
    sub["entry_year"] = sub["entry_year"].astype("string")
    if "entry_term" in sub.columns:
        sub["entry_term"] = utils.data_cleaning.uppercase_string_values(
            sub, col="entry_term"
        )
    sub = sub.rename(columns={"entry_year": "cohort", "entry_term": "cohort_term"})
    out = df.copy()
    out["learner_id"] = out["learner_id"].astype("string")
    out = out.merge(sub, on="learner_id", how="left", validate="m:1")
    out["cohort_term"] = pd.Categorical(
        out["cohort_term"].astype("string"),
        categories=["FALL", "WINTER", "SPRING", "SUMMER"],
        ordered=True,
    )
    if "study_id" not in out.columns:
        out["study_id"] = out["learner_id"]
    if "course_credits_attempted" not in out.columns:
        out["course_credits_attempted"] = out["course_credits_earned"]
    else:
        missing_att = out["course_credits_attempted"].isna()
        if bool(missing_att.all()) or (
            out["course_credits_attempted"].notna().sum() == 0
        ):
            out["course_credits_attempted"] = out["course_credits_earned"]
        else:
            out.loc[missing_att, "course_credits_attempted"] = out.loc[
                missing_att, "course_credits_earned"
            ]
    return out


def _read_and_prepare_es_cohort(
    *,
    file_path: t.Optional[str],
    table_path: t.Optional[str],
    spark_session: t.Optional[pyspark.sql.SparkSession],
    schema: t.Optional[type[pda.DataFrameModel]],
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]],
    **kwargs: object,
) -> pd.DataFrame:
    if not bool(table_path) ^ bool(file_path):
        raise ValueError("exactly one of table_path or file_path must be specified")
    if table_path and not spark_session:
        raise ValueError("spark session must be given when reading data from table")

    df = (
        from_csv_file(file_path, spark_session, **kwargs)  # type: ignore
        if file_path
        else from_delta_table(table_path, spark_session)  # type: ignore
    )
    df = df.rename(columns=utils.data_cleaning.convert_to_snake_case)

    transformations: dict[str, pd.Series] = {}
    if "entry_term" in df.columns:
        transformations["entry_term"] = utils.data_cleaning.uppercase_string_values(
            df, col="entry_term"
        )
    for col in (
        "matriculation_date",
        "bachelors_degree_conferral_date",
        "associates_degree_conferral_date",
        "certificate1_date",
        "certificate2_date",
        "certificate3_date",
    ):
        if col in df.columns:
            transformations[col] = pd.to_datetime(df[col], errors="coerce")
    if transformations:
        df = df.assign(**transformations)

    if schema is None:
        if "learner_id" in df.columns:
            df = df.assign(study_id=df["learner_id"].astype("string"))
        return df

    df = _maybe_convert_maybe_validate_data(df, converter_func, schema)
    assert isinstance(df, pd.DataFrame)
    return _prepare_edvise_cohort_after_validation(df)


def read_raw_es_cohort_data(
    *,
    table_path: t.Optional[str] = None,
    file_path: t.Optional[str] = None,
    schema: t.Optional[type[pda.DataFrameModel]] = None,
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    spark_session: t.Optional[pyspark.sql.SparkSession] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Read Edvise-format cohort (student) CSV or Delta; validate then map for ES audit."""
    return _read_and_prepare_es_cohort(
        file_path=file_path,
        table_path=table_path,
        spark_session=spark_session,
        schema=schema,
        converter_func=converter_func,
        **kwargs,
    )


def _read_and_prepare_es_course(
    *,
    file_path: t.Optional[str],
    table_path: t.Optional[str],
    spark_session: t.Optional[pyspark.sql.SparkSession],
    schema: t.Optional[type[pda.DataFrameModel]],
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]],
    dttm_format: t.Optional[str] = None,
    **kwargs: object,
) -> pd.DataFrame:
    del dttm_format  # always flexible-parse below (avoids ValueError on empty/ISO cells)
    if not bool(table_path) ^ bool(file_path):
        raise ValueError("exactly one of table_path or file_path must be specified")
    if table_path and not spark_session:
        raise ValueError("spark session must be given when reading data from table")

    df = (
        from_csv_file(file_path, spark_session, **kwargs)  # type: ignore
        if file_path
        else from_delta_table(table_path, spark_session)  # type: ignore
    )
    df = df.rename(columns=utils.data_cleaning.convert_to_snake_case)

    transformations: dict[str, pd.Series] = {}
    if "academic_term" in df.columns:
        transformations["academic_term"] = utils.data_cleaning.uppercase_string_values(
            df, col="academic_term"
        )
    for col in ("course_begin_date", "course_end_date"):
        if col in df.columns:
            transformations[col] = pd.to_datetime(df[col], errors="coerce")
    if transformations:
        df = df.assign(**transformations)

    if schema is None:
        if "learner_id" in df.columns:
            df = df.assign(study_id=df["learner_id"].astype("string"))
        return df

    assert file_path is not None
    cohort_lookup = _load_cohort_lookup_next_to_course(file_path)
    df = _maybe_convert_maybe_validate_data(df, converter_func, schema)
    assert isinstance(df, pd.DataFrame)
    return _prepare_edvise_course_after_validation(df, cohort_lookup=cohort_lookup)


def read_raw_es_course_data(
    *,
    table_path: t.Optional[str] = None,
    file_path: t.Optional[str] = None,
    schema: t.Optional[type[pda.DataFrameModel]] = None,
    dttm_format: str = "%Y%m%d",
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    spark_session: t.Optional[pyspark.sql.SparkSession] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Read Edvise-format course CSV or Delta; validate, merge cohort keys, map for ES audit."""
    return _read_and_prepare_es_course(
        file_path=file_path,
        table_path=table_path,
        spark_session=spark_session,
        schema=schema,
        converter_func=converter_func,
        dttm_format=dttm_format,
        **kwargs,
    )
