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


def format_enrollment_intensity_time_limits(
    *,
    intensity_time_limits: dict[str, t.Tuple[Num, str]] | None = None,
    duration: t.Tuple[Num, str] | None = None,
    style: t.Literal["underscore", "compact", "long"] = "underscore",
    intensity: str | None = None,
    order: tuple[str, ...] = ("FULL-TIME", "PART-TIME"),
) -> str:
    """
    Format enrollment time limits into a string.

    Provide exactly one of these inputs:

    1) Map input (FT/PT map)
       - intensity_time_limits: mapping like {"FULL-TIME": (3, "year"), "PART-TIME": (6, "year")}
       Supported styles:
         - style="underscore" -> meant for unity catalog naming:
             "3y_ft_6y_pt"
           (lowercase, joined with underscores, ordered by `order`)

         - style="compact" -> meant for front end model name:
             "3Y FT, 6Y PT"
           (comma-separated, ordered by `order`)

         - style="long" -> meant for model card:
             "3 years (FT), 6 years (PT)"
           (comma-separated, ordered by `order`)

    2) Single duration input
       - duration: tuple like (3, "year") or (6, "month")
       Supported styles:
         - style="long" -> "3 years" (pluralized)
         - style="compact" -> "3Y"
         - style="compact" with `intensity="FULL-TIME"` -> "3Y FT"

    Normalization rules:
      - numeric values like 3.0 are rendered as 3
      - non-integer floats are rounded to 2 decimal places
      - unit abbreviation uses the first letter of the unit ("year"->Y/y, "month"->M/m)
      - intensity abbreviation uses first letters of hyphen-separated words ("FULL-TIME"->FT)

    Parameters
    ----------
    intensity_time_limits:
        FT/PT (or similar) mapping from intensity -> (number, unit).
    duration:
        Single (number, unit) tuple.
    style:
        Output format: "underscore", "compact", or "long".
    intensity:
        Only used with `duration` + style="compact" to append "FT"/"PT".
    order:
        Output ordering for the map input.

    Returns
    -------
    str
        The formatted string.

    Raises
    ------
    ValueError
        If neither or both of `intensity_time_limits` and `duration` are provided,
        or if style="underscore" is used with `duration`.
    """

    def _normalize_num(num: Num) -> Num:
        if isinstance(num, float):
            if num.is_integer():
                return int(num)
            return round(num, 2)
        return num

    def _intensity_abbrev(intensity_str: str) -> str:
        return "".join(word[0] for word in intensity_str.split("-"))

    def _fmt_single(
        dur: t.Tuple[Num, str], *, style: str, intensity: str | None
    ) -> str:
        num, unit = dur
        num = _normalize_num(num)

        if style == "long":
            unit_out = unit if num == 1 else unit + "s"
            return f"{num} {unit_out}"

        # compact
        num_str = str(num)
        unit_abbrev = unit[0].upper()
        if intensity is None:
            return f"{num_str}{unit_abbrev}"
        return f"{num_str}{unit_abbrev} {_intensity_abbrev(intensity)}"

    # Validate input mode
    if (intensity_time_limits is None) == (duration is None):
        raise ValueError(
            "Provide exactly one of `intensity_time_limits` or `duration`."
        )

    # Single duration mode
    if duration is not None:
        if style == "underscore":
            raise ValueError(
                'style="underscore" requires `intensity_time_limits`, not `duration`.'
            )
        return _fmt_single(duration, style=style, intensity=intensity)

    # Map mode
    assert intensity_time_limits is not None

    if style == "underscore":
        parts: list[str] = []
        for k in order:
            if k not in intensity_time_limits:
                continue

            num, unit = intensity_time_limits[k]
            num = _normalize_num(num)

            # keep "3" instead of "3.0" when integer-valued
            if isinstance(num, float) and num.is_integer():
                duration_str = str(int(num))
            else:
                duration_str = str(num)

            unit_abbrev = unit[0].lower()
            intensity_abbrev = _intensity_abbrev(k).lower()
            parts.append(f"{duration_str}{unit_abbrev}_{intensity_abbrev}")

        return "_".join(parts)

    if style == "compact":
        parts: list[str] = []
        for k in order:
            if k not in intensity_time_limits:
                continue
            parts.append(
                _fmt_single(intensity_time_limits[k], style="compact", intensity=k)
            )
        return ", ".join(parts)

    if style == "long":
        parts: list[str] = []
        for k in order:
            if k not in intensity_time_limits:
                continue
            parts.append(
                f"{_fmt_single(intensity_time_limits[k], style='long', intensity=None)} ({_intensity_abbrev(k)})"
            )
        return ", ".join(parts)

    raise ValueError(f"Unknown style: {style}")
