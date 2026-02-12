import re


class Formatting:
    def __init__(self, base_spaces: int = 4):
        """
        Initialize the formatter with a base indentation size.

        Args:
            base_spaces: The number of spaces for each indent level. The default
            needs to be 4, since for markdown parsers and PDF export, this would
            create a reliable interpretation of nested lists.
        """
        self.base_spaces = base_spaces

    def indent_level(self, depth: int) -> str:
        """
        Generate a string of spaces for indentation.
        """
        return " " * (self.base_spaces * depth)

    def header_level(self, depth: int) -> str:
        """
        Generate Markdown header prefix based on depth.
        """
        return "#" * depth

    def bold(self, text: str) -> str:
        """
        Apply Markdown bold formatting to a given text.
        """
        return f"**{text}**"

    def italic(self, text: str) -> str:
        """
        Apply Markdown italic formatting to a given text.
        """
        return f"_{text}_"

    def ordinal(self, n: int) -> str:
        """
        Converts an integer to its ordinal form (e.g. 1 -> 1st).
        """
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def friendly_case(self, text: str, capitalize: bool = True) -> str:
        """
        Converts strings like 'bachelor's degree' or 'full-time' into human-friendly forms,
        preserving hyphens and apostrophes, with optional capitalization.

        Also handles model name formatting with special rules:
        - Adds colon after "Year 2"
        - Adds commas between time limit components (e.g., "3Y FT, 6Y PT")
        - Wraps checkpoint info in parentheses with colon

        Args:
            text: Text to be converted
            capitalize: Whether to title-case each word. If False, keeps original casing.

        Returns:
            Human-friendly string.
        """
        if isinstance(text, (int, float)):
            return str(text)

        # If the string is numeric-like (int or float), return as-is
        try:
            float_val = float(text)
            if text.strip().replace(".", "", 1).isdigit() or text.strip().isdigit():
                return text
        except ValueError:
            pass  # Not a float-like string; continue formatting

        text = text.replace("_", " ")

        def smart_cap(word: str) -> str:
            # Handles hyphenated subwords like "full-time"
            return "-".join(
                part[0].upper() + part[1:].lower() if part else ""
                for part in word.split("-")
            )

        if not capitalize:
            return text

        # Regex preserves apostrophes and hyphens
        tokens = re.findall(r"[\w'-]+", text)
        result = " ".join(smart_cap(tok) for tok in tokens)

        # Apply model name formatting rules
        # Add colon after "Year 2" (for retention models)
        result = re.sub(r"\bYear 2\b(?= \w)", r"Year 2:", result)

        # Add comma between time limit components (e.g., "3Y Ft 6Y Pt" → "3Y FT, 6Y PT")
        result = re.sub(
            r"(\d+[YyTt]) ([Ff][Tt])(\s+\d+[YyTt]) ([Pp][Tt])",
            lambda m: (
                f"{m.group(1).upper()} {m.group(2).upper()}, {m.group(3).upper()} {m.group(4).upper()}"
            ),
            result,
        )

        # Wrap checkpoint in parentheses with colon (e.g., "Checkpoint X" → "(Checkpoint: X)")
        result = re.sub(
            r"\bCheckpoint\b\s+(\d+|First|[^_]+?)(?=\s*(?:Core|Total|Terms|Credits|$))",
            r"(Checkpoint: \1",
            result,
        )
        # Add closing parenthesis at the end of checkpoint section
        if "(Checkpoint:" in result:
            # Find where to close the parenthesis - before underscore suffix or at end
            result = re.sub(r"(Checkpoint:[^)]+?)\s*(?=$|_)", r"\1)", result)

        return result
