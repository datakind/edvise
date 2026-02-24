import logging
import mlflow
import typing as t
from typing import Any
import pydantic as pyd

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


import logging
import typing as t
from typing import Any

LOGGER = logging.getLogger(__name__)


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


# Schema and volume caches for Databricks catalog operations
_schema_cache: dict[str, set[str]] = {}
_bronze_volume_cache: dict[str, str] = {}  # key: f"{catalog}.{schema}" -> volume_name


def list_schemas_in_catalog(spark: SparkSession, catalog: str) -> set[str]:
    """
    List all schemas in a catalog (with caching).

    Args:
        spark: Spark session
        catalog: Catalog name

    Returns:
        Set of schema names
    """
    if catalog in _schema_cache:
        return _schema_cache[catalog]

    rows = spark.sql(f"SHOW SCHEMAS IN {catalog}").collect()

    schema_names: set[str] = set()
    for row in rows:
        d = row.asDict()
        for k in ["databaseName", "database_name", "schemaName", "schema_name", "name"]:
            v = d.get(k)
            if v:
                schema_names.add(v)
                break
        else:
            schema_names.add(list(d.values())[0])

    _schema_cache[catalog] = schema_names
    return schema_names


def find_bronze_schema(spark: SparkSession, catalog: str, inst_prefix: str) -> str:
    """
    Find bronze schema for institution prefix.

    Args:
        spark: Spark session
        catalog: Catalog name
        inst_prefix: Institution prefix (e.g., "motlow_state_cc")

    Returns:
        Bronze schema name (e.g., "motlow_state_cc_bronze")

    Raises:
        ValueError: If bronze schema not found
    """
    target = f"{inst_prefix}_bronze"
    schemas = list_schemas_in_catalog(spark, catalog)
    if target not in schemas:
        raise ValueError(f"Bronze schema not found: {catalog}.{target}")
    return target


def find_bronze_volume_name(spark: SparkSession, catalog: str, schema: str) -> str:
    """
    Find bronze volume name in schema (with caching).

    Args:
        spark: Spark session
        catalog: Catalog name
        schema: Schema name

    Returns:
        Volume name containing "bronze"

    Raises:
        ValueError: If no bronze volume found
    """
    key = f"{catalog}.{schema}"
    if key in _bronze_volume_cache:
        return _bronze_volume_cache[key]

    vols = spark.sql(f"SHOW VOLUMES IN {catalog}.{schema}").collect()
    if not vols:
        raise ValueError(f"No volumes found in {catalog}.{schema}")

    # Usually "volume_name", but be defensive
    def _get_vol_name(row: Any) -> str:
        d = row.asDict()
        for k in ["volume_name", "volumeName", "name"]:
            if k in d:
                return str(d[k])
        return str(list(d.values())[0])

    vol_names = [_get_vol_name(v) for v in vols]
    bronze_like = [v for v in vol_names if "bronze" in str(v).lower()]
    if bronze_like:
        result = bronze_like[0]
        _bronze_volume_cache[key] = result
        return result

    raise ValueError(
        f"No volume containing 'bronze' found in {catalog}.{schema}. Volumes={vol_names}"
    )
