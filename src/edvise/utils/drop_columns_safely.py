import pandas as pd
import logging


LOGGER = logging.getLogger(__name__)


def drop_columns_safely(df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
    """
    Drop ``cols`` from ``df`` *safely*: If any are missing, log a clear warning,
    then drop the non-missing columns from the DataFrame without crashing.

    Args:
        df
        cols
    """
    df_cols = set(df.columns)
    drop_cols = set(cols_to_drop) & df_cols
    missing_cols = set(cols_to_drop) - df_cols

    if missing_cols:
        LOGGER.warning("Missing columns not found in df: %s", missing_cols)

    df_trf = df.drop(columns=list(drop_cols))
    LOGGER.info("Dropped %s columns not needed safely", len(drop_cols))
    LOGGER.info("Columns Dropped: %s", drop_cols)
    return df_trf
