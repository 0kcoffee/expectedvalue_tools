"""ANSI color codes for terminal output."""


class Colors:
    """ANSI color codes for terminal output."""

    # Text styles
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Bright colors (used in the code)
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Custom accent color #3D15E5 (RGB: 61, 21, 229)
    ACCENT = "\033[38;2;61;21;229m"
