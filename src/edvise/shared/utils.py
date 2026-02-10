"""Shared utility functions used across multiple modules."""


def as_percent(val: float | int) -> str:
    """
    Convert a decimal value to a percentage string.

    Args:
        val: A decimal value (e.g., 0.75 for 75%)

    Returns:
        A string representation of the percentage (e.g., "75" or "75.5")

    Examples:
        >>> as_percent(0.75)
        '75'
        >>> as_percent(0.755)
        '75.5'
        >>> as_percent(1.0)
        '100'
    """
    val = float(val) * 100
    return str(int(val) if val.is_integer() else round(val, 2))
