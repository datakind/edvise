import logging
import mlflow
import pandas as pd
import typing as t
import pydantic as pyd

from dataio import read

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
