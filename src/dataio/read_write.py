import logging
import mlflow
import pandas as pd
import typing as t
import pydantic as pyd
import pandera as pda
import pyspark.sql

import utils, dataio

LOGGER = logging.getLogger(__name__)

S = t.TypeVar("S", bound=pyd.BaseModel)

# Disable mlflow autologging (due to Databricks issues during feature selection)
mlflow.autolog(disable=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Ignore Databricks logger

def read_PDP_config(self, toml_file_path: str):
        """Reads the institution's model's configuration file."""
        try:
            cfg = dataio.read_config(toml_file_path, schema=PDPProjectConfig)
            return cfg
        except FileNotFoundError:
            logging.error("Configuration file not found at %s", toml_file_path)
            raise
        except Exception as e:
            logging.error("Error reading configuration file: %e", e)
            raise

def read_data_from_delta(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
        """
        Reads course and cohort data from Delta Lake tables.

        Returns:
            A tuple containing the course and cohort DataFrames, or (None, None) if the
            Spark session is not available.
        """
        if self.spark_session:
            try:
                df_course = schemas.raw_course.RawPDPCourseDataSchema(
                    dataio.from_delta_table(
                        self.args.course_dataset_validated_path,
                        spark_session=self.spark_session,
                    )
                )

                df_cohort = schemas.raw_cohort.RawPDPCohortDataSchema(
                    dataio.from_delta_table(
                        self.args.cohort_dataset_validated_path,
                        spark_session=self.spark_session,
                    )
                )
                return df_course, df_cohort
            except Exception as e:
                logging.error("Error reading data from Delta Lake: %s", e)
                raise
        else:
            logging.error("Spark session not initialized. Cannot read delta tables.")
            raise

def write_data_to_delta(self, df_processed: pd.DataFrame):
        """
        Saves the processed dataset to a Delta Lake table.

        Args:
            df_processed: The processed DataFrame.
        """
        if not self.spark_session:
            logging.error(
                "Spark session not initialized. Cannot write processed dataset."
            )
            return

        write_schema = f"{self.args.databricks_institution_name}_silver"
        write_table_path = f"{self.args.DB_workspace}.{write_schema}.{self.args.db_run_id}_processed_dataset"

        try:
            dataio.to_delta_table(
                df_processed, write_table_path, spark_session=self.spark_session
            )
            logging.info("Processed dataset written to table: %s", write_table_path)
            return write_table_path

        except Exception as e:
            logging.error("Error writing processed data to Delta Lake: %s", e)
            raise

def read_parquet(path: str, dtype: dict = None, verbose: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if dtype:
        df = df.astype(dtype)
    if verbose:
        print(f"Read {df.shape[0]} rows from {path}")
    return df


def write_parquet(
    df: pd.DataFrame,
    file_path: str,
    index: bool = False,
    overwrite: bool = True,
    verbose: bool = True,
) -> str:
    """
    Writes a Pandas DataFrame to a Parquet file.

    Args:
        df (pd.DataFrame): The DataFrame to write.
        file_path (str): Destination file path.
        index (bool): Whether to write the index to file. Default is False.
        overwrite (bool): If False, raises error if file exists. Default is True.
        verbose (bool): Whether to log info. Default is True.

    Returns:
        str: Absolute path of the written Parquet file.
    """
    path = pathlib.Path(file_path).resolve()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite=False: {path}")

    df.to_parquet(path, index=index)

    if verbose:
        logger.info(f"Wrote {df.shape[0]} rows × {df.shape[1]} columns to {path}")

    return str(path)

def read_raw_PDP_course_data(
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
    if not bool(table_path) ^ bool(file_path):
        raise ValueError("exactly one of table_path or file_path must be specified")
    elif table_path is not None and spark_session is None:
        raise ValueError("spark session must be given when reading data from table")

    df = (
        read.from_csv_file(file_path, spark_session, **kwargs)  # type: ignore
        if file_path
        else read.from_delta_table(table_path, spark_session)  # type: ignore
    )
    # apply to the data what pandera calls "parsers" before validation
    # ideally, all these operations would be dataframe parsers on the schema itself
    # but pandera applies core before custom parsers under the hood :/
    df = (
        # standardize column names
        df.rename(columns=utils.misc.convert_to_snake_case)
        # standardize certain column values
        .assign(
            # uppercase string values for some cols to avoid case inconsistency later on
            **{
                col: ft.partial(utils._data_cleaning._uppercase_string_values, col=col)
                for col in ("academic_term",)
            }
            # help pandas to parse non-standard datetimes... read_csv() struggles
            | {
                col: ft.partial(utils._data_cleaning._parse_dttm_values, col=col, fmt=dttm_format)
                for col in ("course_begin_date", "course_end_date")
            }
        )
    )
    return _maybe_convert_maybe_validate_data(df, converter_func, schema)


def read_raw_PDP_cohort_data(
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
    if not bool(table_path) ^ bool(file_path):
        raise ValueError("exactly one of table_path or file_path must be specified")
    elif table_path is not None and spark_session is None:
        raise ValueError("spark session must be given when reading data from table")

    df = (
        read.from_csv_file(file_path, spark_session, **kwargs)  # type: ignore
        if file_path
        else read.from_delta_table(table_path, spark_session)  # type: ignore
    )
    # apply to the data what pandera calls "parsers" before validation
    # ideally, all these operations would be dataframe parsers on the schema itself
    # but pandera applies core before custom parsers under the hood :/
    df = (
        # standardize column names
        df.rename(columns=utils.misc.convert_to_snake_case)
        # standardize column values
        .assign(
            # uppercase string values for some cols to avoid case inconsistency later on
            # for practical reasons, this is the only place where it's easy to do so
            **{
                col: ft.partial(_uppercase_string_values, col=col)
                for col in ("cohort_term",)
            }
            # replace "UK" with null in GPA cols, so we can coerce to float via schema
            | {
                col: ft.partial(_replace_values_with_null, col=col, to_replace="UK")
                for col in ("gpa_group_term_1", "gpa_group_year_1")
            }
            # help pandas to coerce string "1"/"0" values into True/False
            | {
                col: ft.partial(_cast_to_bool_via_int, col=col)
                for col in ("retention", "persistence")
            }
        )
    )
    return _maybe_convert_maybe_validate_data(df, converter_func, schema)


def _maybe_convert_maybe_validate_data(
    df: pd.DataFrame,
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    schema: t.Optional[type[pda.DataFrameModel]] = None,
) -> pd.DataFrame:
    # HACK: we're hiding this pandera import here so databricks doesn't know about it
    # pandera v0.23+ pulls in pandas v2.1+ while databricks runtimes are stuck in v1.5
    # resulting in super dumb dependency errors when loading automl trained models
    import pandera.errors

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
        except pandera.errors.SchemaErrors:
            LOGGER.error("unable to parse/validate raw data")
            raise


