# Set before any import that might load pyspark/pyarrow (e.g. via databricks-connect).
import os

os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")
