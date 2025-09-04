import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)


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
