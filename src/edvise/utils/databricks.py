import logging
import mlflow
import typing as t
from typing import Any
import pydantic as pyd
import re

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


# Compiled regex patterns for reverse transformation (performance optimization)
_REVERSE_REPLACEMENTS = {
    "ctc": "community technical college",
    "cc": "community college",
    "st": "of science and technology",
    "uni": "university",
    "col": "college",
}

# Pre-compile regex patterns for word boundary matching
_COMPILED_REVERSE_PATTERNS = {
    abbrev: re.compile(r"\b" + re.escape(abbrev) + r"\b")
    for abbrev in _REVERSE_REPLACEMENTS.keys()
}


def _validate_databricks_name_format(databricks_name: str) -> None:
    """
    Validate that databricks name matches expected format.

    Args:
        databricks_name: Name to validate

    Raises:
        ValueError: If name is empty or contains invalid characters
    """
    if not isinstance(databricks_name, str) or not databricks_name.strip():
        raise ValueError("databricks_name must be a non-empty string")

    pattern = "^[a-z0-9_]*$"
    if not re.match(pattern, databricks_name):
        raise ValueError(
            f"Invalid databricks name format '{databricks_name}'. "
            "Must contain only lowercase letters, numbers, and underscores."
        )


def _reverse_abbreviation_replacements(name: str) -> str:
    """
    Reverse abbreviation replacements in the name.

    Handles the ambiguous "st" abbreviation:
    - If "st" appears as the first word, it's kept as "st" (abbreviation for Saint)
      and will be capitalized to "St" by title() case
    - Otherwise, "st" is treated as "of science and technology"

    Args:
        name: Name with underscores replaced by spaces

    Returns:
        Name with abbreviations expanded to full forms
    """
    # Split into words to handle "st" at the beginning specially
    words = name.split()

    # Keep "st" at the beginning as-is (will be capitalized to "St" by title() case)
    # Don't expand it to "saint" - preserve the abbreviation

    # Replace "st" in remaining positions with "of science and technology"
    for i in range(len(words)):
        if words[i] == "st" and i > 0:  # Only replace if not the first word
            words[i] = "of science and technology"

    # Rejoin and apply other abbreviation replacements
    name = " ".join(words)

    # Apply other abbreviation replacements (excluding "st" which we handled above)
    for abbrev, full_form in _REVERSE_REPLACEMENTS.items():
        if abbrev != "st":  # Skip "st" as we handled it above
            pattern = _COMPILED_REVERSE_PATTERNS[abbrev]
            name = pattern.sub(full_form, name)

    return name


def databricksify_inst_name(inst_name: str) -> str:
    """
    Transform institution name to Databricks-compatible format.

    Follows DK standardized rules for naming conventions used in Databricks:
    - Lowercases the name
    - Replaces common phrases with abbreviations (e.g., "community college" → "cc")
    - Replaces special characters and spaces with underscores
    - Validates final format contains only lowercase letters, numbers, and underscores

    Args:
        inst_name: Original institution name (e.g., "Motlow State Community College")

    Returns:
        Databricks-compatible name (e.g., "motlow_state_cc")

    Raises:
        ValueError: If the resulting name contains invalid characters

    Example:
        >>> databricksify_inst_name("Motlow State Community College")
        'motlow_state_cc'
        >>> databricksify_inst_name("University of Science & Technology")
        'uni_of_st_technology'
    """
    name = inst_name.lower()

    # Apply abbreviation replacements (most specific first)
    dk_replacements = {
        "community technical college": "ctc",
        "community college": "cc",
        "of science and technology": "st",
        "university": "uni",
        "college": "col",
    }

    for old, new in dk_replacements.items():
        name = name.replace(old, new)

    # Replace special characters
    special_char_replacements = {" & ": " ", "&": " ", "-": " "}
    for old, new in special_char_replacements.items():
        name = name.replace(old, new)

    # Replace spaces with underscores
    final_name = name.replace(" ", "_")

    # Validate format
    pattern = "^[a-z0-9_]*$"
    if not re.match(pattern, final_name):
        raise ValueError(
            f"Unexpected character found in Databricks compatible name: '{final_name}'"
        )

    return final_name


def reverse_databricksify_inst_name(databricks_name: str) -> str:
    """
    Reverse the databricksify transformation to get back the original institution name.

    This function attempts to reverse the transformation done by databricksify_inst_name.
    Since the transformation is lossy (multiple original names can map to the same
    databricks name), this function produces the most likely original name.

    Args:
        databricks_name: The databricks-transformed institution name (e.g., "motlow_state_cc")
            Case inconsistencies are normalized (input is lowercased before processing).

    Returns:
        The reversed institution name with proper capitalization (e.g., "Motlow State Community College")

    Raises:
        ValueError: If the databricks name contains invalid characters
    """
    # Normalize to lowercase to handle case inconsistencies
    # (databricksify_inst_name always produces lowercase output)
    databricks_name = databricks_name.lower()
    _validate_databricks_name_format(databricks_name)

    # Step 1: Replace underscores with spaces
    name = databricks_name.replace("_", " ")

    # Step 2: Reverse the abbreviation replacements
    # The original replacements were done in this order (most specific first):
    # 1. "community technical college" → "ctc"
    # 2. "community college" → "cc"
    # 3. "of science and technology" → "st"
    # 4. "university" → "uni"
    # 5. "college" → "col"
    name = _reverse_abbreviation_replacements(name)

    # Step 3: Capitalize appropriately (title case)
    return name.title()
