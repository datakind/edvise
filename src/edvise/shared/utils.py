import re
import typing as t

Num = t.Union[int, float]

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


def normalize_degree(text: str) -> str:
    """
    Normalize degree text by removing the word 'degree' and standardizing capitalization.

    Removes trailing 'degree' (case-insensitive) and converts text to title case
    (lowercase with first letter capitalized).

    Args:
        text: Degree text to normalize (e.g., "ASSOCIATE'S DEGREE", "Bachelor's degree")

    Returns:
        Normalized degree text (e.g., "Associate's", "Bachelor's")

    Examples:
        normalize_degree("ASSOCIATE'S DEGREE")
        "Associates"
    """
    # remove the word "degree" (case-insensitive) at the end
    text = text.strip()

    # remove the word "degree" (case-insensitive) at the end
    text = re.sub(r"\s*degree\s*$", "", text, flags=re.IGNORECASE)

    # normalize possessive degrees: bachelor's/master's/associate's -> bachelors/masters/associates
    # handles straight and curly apostrophes
    text = re.sub(r"[’']s\b", "s", text, flags=re.IGNORECASE)

    # normalize capitalization
    return text.lower().capitalize()


def format_intensity_time_limit(
    self,
    duration: t.Tuple[Num, str],
    *,
    style: t.Literal["long", "compact"] = "long",
    intensity: str | None = None,
) -> str:
    """
    Backward compatible:
      - default returns the same as before: "3 years"
    New options:
      - style="compact" returns "3Y" or "3Y FT" if intensity is provided
    """
    num, unit = duration

    # Normalize number
    if isinstance(num, float):
        if num.is_integer():
            num = int(num)
        else:
            num = round(num, 2)

    if style == "long":
        unit_out = unit if num == 1 else unit + "s"
        return f"{num} {unit_out}"

    # compact style
    num_str = str(num)
    unit_abbrev = unit[0].upper()  # e.g. year -> Y, month -> M

    if intensity is None:
        return f"{num_str}{unit_abbrev}"

    intensity_abbrev = "".join(word[0] for word in intensity.split("-"))
    return f"{num_str}{unit_abbrev} {intensity_abbrev}"


def extract_time_limits(
    self,
    intensity_time_limits: dict[str, t.Tuple[Num, str]],
) -> str:
    """
    Helper that uses the above `format_intensity_time_limit` to create a similar time limit format for the model name which is human-readable and shorter.
    """
    order = ["FULL-TIME", "PART-TIME"]
    parts = []

    for intensity in order:
        if intensity not in intensity_time_limits:
            continue
        parts.append(
            self.format_intensity_time_limit(
                intensity_time_limits[intensity],
                style="compact",
                intensity=intensity,
            )
        )

    return ", ".join(parts)
