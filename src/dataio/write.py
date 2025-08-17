import logging
import time

import pandas as pd
import pyspark.sql
import pathlib

from .. import utils

LOGGER = logging.getLogger(__name__)


def to_delta_table(
    df: pd.DataFrame, table_path: str, spark_session: pyspark.sql.SparkSession
) -> None:
    """
    Write pandas DataFrame to Databricks Unity Catalog.

    Args:
        df
        table_path: Path in Unity Catalog to which ``df`` will be written,
            including the full three-level namespace: ``catalog.schema.table`` .
        spark_session: Entry point to using spark dataframes and the databricks integration.

    See Also:
        - https://docs.databricks.com/en/delta/drop-table.html#when-to-replace-a-table
        - https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/spark_session.html
    """
    start_time = time.time()
    LOGGER.info("saving data to %s ...", table_path)
    df_spark = spark_session.createDataFrame(
        df.rename(columns=utils.data_cleaning.convert_to_snake_case)
    )
    (
        pyspark.sql.DataFrameWriterV2(df_spark, table_path)
        .options(format="delta")
        # this *should* do what databricks recomends -- and retains table history!
        .createOrReplace()
    )
    run_time = time.time() - start_time

    table_rows = spark_session.sql(f"SELECT COUNT(*) FROM {table_path}").collect()
    if table_rows[0][0] != len(df):
        raise IOError(
            f"{table_rows[0][0]} written to delta table, "
            f"but {len(df)} rows in original dataframe"
        )

    history = spark_session.sql(f"DESCRIBE history {table_path} LIMIT 1").collect()
    verno = int(history[0][0])
    LOGGER.info("data saved to %s (v%s) in %s seconds", table_path, verno, run_time)



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
        LOGGER.info(f"Wrote {df.shape[0]} rows × {df.shape[1]} columns to {path}")

    return str(path)