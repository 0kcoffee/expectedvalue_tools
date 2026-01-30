"""Text utility functions."""

import re


def strip_ansi(text: str) -> str:
    """
    Remove ANSI escape sequences from a string to get visible length.

    Args:
        text: String that may contain ANSI codes

    Returns:
        String with ANSI codes removed
    """
    # Remove ANSI escape sequences (CSI sequences)
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def visible_length(text: str) -> int:
    """
    Get the visible length of a string, excluding ANSI escape sequences.

    Args:
        text: String that may contain ANSI codes

    Returns:
        Visible length of the string
    """
    return len(strip_ansi(text))
