import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)


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