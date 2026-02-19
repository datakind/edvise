import logging
import mlflow
import typing as t
import pydantic as pyd
import os

LOGGER = logging.getLogger(__name__)

S = t.TypeVar("S", bound=pyd.BaseModel)

from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession

# Disable mlflow autologging (due to Databricks issues during feature selection)
mlflow.autolog(disable=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Ignore Databricks logger


def get_spark_session() -> SparkSession:
    """
    Attempts to create a Spark session.
    Returns:
        SparkSession: A Spark session if successful, None otherwise.
    """
    try:
        spark_session = DatabricksSession.builder.getOrCreate()
        logging.info("Spark session created successfully.")
        return spark_session
    except Exception:
        logging.error("Unable to create Spark session.")
        raise


def get_db_widget_param(name: str, *, default: t.Optional[object] = None) -> object:
    """
    Get a Databricks widget parameter by ``name``,
    returning a ``default`` value if not found.

    References:
        - https://docs.databricks.com/en/dev-tools/databricks-utils.html#dbutils-widgets-get
    """
    # these only work in a databricks env, so hide them here
    from databricks.sdk.runtime import dbutils
    from py4j.protocol import Py4JJavaError

    try:
        return dbutils.widgets.get(name)
    except Py4JJavaError:
        LOGGER.warning(
            "no db widget found with name=%s; returning default=%s", name, default
        )
        return default


def mock_pandera():
    """
    Databricks doesn't include ``pandera`` in its runtimes, and it's also very picky
    about which packages are installed when training and/or loading models with AutoML
    and mlflow. However, we need ``pandera`` to be available in order for this package
    to import, since it's used at the module-level for data schema validation.

    So, here we mock out functionality used in our data schemas in such a way that
    this package can import without error, even if ``pandera`` isn't actually installed,
    as we're forced to do in certain Databricks notebooks. Yes, this sucks!
    """
    import sys
    import types

    m1 = types.ModuleType("pandera")
    m2 = types.ModuleType("pandera.typing")

    GenericDtype = t.TypeVar("GenericDtype")

    class DataFrameModel: ...

    def Field(**kwargs): ...

    def dataframe_parser(_fn=None, **parser_kwargs):
        def _wrapper(fn): ...

        return _wrapper(_fn)

    def parser(*fields, **parser_kwargs):
        def _wrapper(fn): ...

        return _wrapper

    def dataframe_check(_fn=None, **check_kwargs):
        def _wrapper(fn): ...

        if _fn:
            return _wrapper(_fn)
        return _wrapper

    def check(*fields, regex=False, **check_kwargs):
        def _wrapper(fn): ...

        return _wrapper

    class Series(t.Generic[GenericDtype]): ...

    m1.DataFrameModel = DataFrameModel  # type: ignore
    m1.Field = Field  # type: ignore
    m1.dataframe_parser = dataframe_parser  # type: ignore
    m1.parser = parser  # type: ignore
    m1.dataframe_check = dataframe_check  # type: ignore
    m1.check = check  # type: ignore
    m2.Series = Series  # type: ignore

    sys.modules[m1.__name__] = m1
    sys.modules[m2.__name__] = m2


def in_databricks() -> bool:
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION") or os.getenv("DB_IS_DRIVER"))
