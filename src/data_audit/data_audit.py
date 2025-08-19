import pandas as pd
import logging
import functools as ft
import typing as t
import re
from standardizer import PDPCohortStandardizer, PDPCourseStandardizer
from src import dataio

LOGGER = logging.getLogger(__name__)

class RunDataAuditTask:
    """Performs the data audit and data cleaning tasks for the SST pipeline."""

    def __init__(self, args):
        self.args = args
        self.cohort_std = PDPCohortStandardizer()
        self.course_std = PDPCourseStandardizer()

    def run(self, df_cohort_raw: pd.DataFrame, df_course_raw: pd.DataFrame):
        # Standardize cohort data
        df_cohort_validated = self.cohort_std.standardize(df_cohort_raw)
        
        # Check it passes schema
        df_cohort_validated = dataio.pdp.read_raw_cohort_data(
            df_cohort_validated,
            schema=dataio.schemas.pdp.RawPDPCohortDataSchema,
        )

        df_cohort_validated.to_parquet(
            f"{self.args.cohort_dataset_validated_path}/df_cohort_validated.parquet",
            index=False,
        )

        # Standardize course data
        df_course_validated = self.course_std.standardize(df_course_raw)

        # Check it passes schema
        df_course_validated = dataio.pdp.read_raw_cohort_data(
            df_course_validated,
            schema=dataio.schemas.pdp.RawPDPCourseDataSchema,
            dttm_format="%Y%m%d.0",
        )

        df_course_validated.to_parquet(
            f"{self.args.course_dataset_validated_path}/df_course_validated.parquet",
            index=False,
        )

        return df_cohort_validated, df_course_validated