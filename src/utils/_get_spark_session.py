import logging
import mlflow
import typing as t
import pydantic as pyd

LOGGER = logging.getLogger(__name__)

S = t.TypeVar("S", bound=pyd.BaseModel)

from databricks.connect import DatabricksSession

# Disable mlflow autologging (due to Databricks issues during feature selection)
mlflow.autolog(disable=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Ignore Databricks logger

def get_spark_session() -> DatabricksSession | None:
        """
        Attempts to create a Spark session.
        Returns:
            DatabricksSession | None: A Spark session if successful, None otherwise.
        """
        try:
            spark_session = DatabricksSession.builder.getOrCreate()
            logging.info("Spark session created successfully.")
            return spark_session
        except Exception:
            logging.error("Unable to create Spark session.")
            raise
