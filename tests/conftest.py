# Silence pyspark/pandas warning about PYARROW_IGNORE_TIMEZONE
import os

os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")
