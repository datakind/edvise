import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)
from .. import utils


def infer_first_term_of_year(s: pd.Series) -> utils.types.TermType:
    """
    Infer the first term of the (academic) year by the ordering of its categorical values.

    See Also:
        - :class:`schemas.base.TermField()`
    """
    if isinstance(s.dtype, pd.CategoricalDtype) and s.cat.ordered is True:
        first_term_of_year = s.cat.categories[0]
        LOGGER.info("'%s' inferred as the first term of the year", first_term_of_year)
        assert isinstance(first_term_of_year, str)  # type guard
        return first_term_of_year  # type: ignore
    else:
        raise ValueError(
            f"'{s.name}' series is not an ordered categorical: {s.dtype=} ..."
            "so the first term of the academic year can't be inferred. "
            "Update the raw course data schema to properly order its categories!"
        )


def infer_num_terms_in_year(s: pd.Series) -> int:
    """
    Infer the number of terms in the (academic) year by the number of its categorical values.

    See Also:
        - :class:`schemas.base.TermField()`
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        num_terms_in_year = len(s.cat.categories)
        LOGGER.info("%s inferred as the number of term in the year", num_terms_in_year)
        return num_terms_in_year
    else:
        raise ValueError(
            f"'{s.name}' series is not a categorical: {s.dtype=} ..."
            "so the number of term in the academic year can't be inferred. "
            "Update the raw course data schema to properly set its categories!"
        )
