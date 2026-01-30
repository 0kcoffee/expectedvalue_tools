"""Text formatting utilities for output."""

from typing import Optional
import numpy as np
from ..utils.colors import Colors
from ..utils.text import visible_length


def print_ascii_distribution(
    data: np.ndarray,
    title: str = "Distribution",
    width: int = 60,
    is_percentage: bool = False,
) -> None:
    """
    Print an ASCII histogram with colors showing the distribution of data.

    Args:
        data: Array of values to visualize
        title: Title for the histogram
        width: Width of the histogram in characters
        is_percentage: If True, format values as percentages instead of dollars
    """
    if len(data) == 0:
        print(f"{title}: No data")
        return

    # Calculate bins
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)

    # Create bins (20 bins)
    n_bins = 20
    bins = np.linspace(min_val, max_val, n_bins + 1)
    hist, bin_edges = np.histogram(data, bins=bins)

    # Normalize histogram to fit width
    max_count = np.max(hist)
    if max_count == 0:
        return

    # Use color constants
    GREEN = Colors.BRIGHT_GREEN
    RED = Colors.BRIGHT_RED
    YELLOW = Colors.BRIGHT_YELLOW
    RESET = Colors.RESET
    BOLD = Colors.BOLD
    FRAME_COLOR = Colors.BRIGHT_WHITE

    # Calculate frame width: label width (8) + separator (3) + bar width + count (7) = width + 18
    frame_width = width + 18
    title_spaces = frame_width - visible_length(title) - 4  # 4 for borders and padding

    print(f"\n{FRAME_COLOR}{BOLD}{'─' * frame_width}{RESET}")
    print(
        f"{FRAME_COLOR}{BOLD}│{RESET} {BOLD}{Colors.BRIGHT_WHITE}{title}{Colors.RESET} {' ' * title_spaces} {FRAME_COLOR}{BOLD}│{RESET}"
    )
    print(f"{FRAME_COLOR}{BOLD}{'─' * frame_width}{RESET}")

    # Print histogram bars
    for i in range(n_bins):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        bin_center = (bin_start + bin_end) / 2
        count = hist[i]

        # Determine color based on bin center
        if bin_center > 0:
            color = GREEN
            bar_char = "█"
        elif bin_center < 0:
            color = RED
            bar_char = "█"
        else:
            color = YELLOW
            bar_char = "█"

        # Scale bar length
        bar_length = int((count / max_count) * width) if max_count > 0 else 0

        # Format bin label
        if is_percentage:
            if abs(bin_center) < 0.1:
                bin_label = f"{bin_center:7.3f}%"
            elif abs(bin_center) < 10:
                bin_label = f"{bin_center:7.2f}%"
            else:
                bin_label = f"{bin_center:7.1f}%"
        else:
            if abs(bin_center) < 1:
                bin_label = f"${bin_center:7.2f}"
            elif abs(bin_center) < 100:
                bin_label = f"${bin_center:7.1f}"
            else:
                bin_label = f"${bin_center:7.0f}"

        # Create bar
        bar = bar_char * bar_length

        # Print line
        print(f"{bin_label} │{color}{bar}{RESET} ({count:4d})")

    # Print statistics with better formatting
    print(f"{FRAME_COLOR}{BOLD}{'─' * frame_width}{RESET}")
    if is_percentage:
        print(
            f"{Colors.DIM}Statistics:{Colors.RESET} "
            f"{Colors.BRIGHT_CYAN}Mean:{Colors.RESET} {Colors.BOLD}{mean_val:8.2f}%{Colors.RESET}  "
            f"{Colors.BRIGHT_CYAN}Min:{Colors.RESET} {Colors.BOLD}{min_val:8.2f}%{Colors.RESET}  "
            f"{Colors.BRIGHT_CYAN}Max:{Colors.RESET} {Colors.BOLD}{max_val:8.2f}%{Colors.RESET}  "
            f"{Colors.BRIGHT_CYAN}N:{Colors.RESET} {Colors.BOLD}{len(data)}{Colors.RESET}"
        )
    else:
        print(
            f"{Colors.DIM}Statistics:{Colors.RESET} "
            f"{Colors.BRIGHT_CYAN}Mean:{Colors.RESET} {Colors.BOLD}${mean_val:8,.2f}{Colors.RESET}  "
            f"{Colors.BRIGHT_CYAN}Min:{Colors.RESET} {Colors.BOLD}${min_val:8,.2f}{Colors.RESET}  "
            f"{Colors.BRIGHT_CYAN}Max:{Colors.RESET} {Colors.BOLD}${max_val:8,.2f}{Colors.RESET}  "
            f"{Colors.BRIGHT_CYAN}N:{Colors.RESET} {Colors.BOLD}{len(data)}{Colors.RESET}"
        )

    # Add zero line indicator
    if min_val < 0 < max_val:
        # Find which bin contains zero
        zero_bin_idx = None
        for i in range(n_bins):
            if bin_edges[i] <= 0 <= bin_edges[i + 1]:
                zero_bin_idx = i
                break

        if zero_bin_idx is not None:
            # Calculate position within the bin
            bin_start = bin_edges[zero_bin_idx]
            bin_end = bin_edges[zero_bin_idx + 1]
            zero_pos_in_bin = (
                (0 - bin_start) / (bin_end - bin_start) if bin_end != bin_start else 0.5
            )

            # Calculate position in the histogram
            zero_pos = int((zero_bin_idx + zero_pos_in_bin) / n_bins * width)

            # Find the count for the zero bin to position the marker
            zero_bin_count = hist[zero_bin_idx]
            zero_bar_length = (
                int((zero_bin_count / max_count) * width) if max_count > 0 else 0
            )

            # Position marker after the bar or at the position
            marker_pos = max(zero_bar_length, zero_pos)

            if is_percentage:
                label_width = len(f"{min_val:7.2f}%")
            else:
                label_width = len(f"${min_val:7.2f}")
            zero_line = (
                " " * (label_width + 2)
                + "│"
                + " " * marker_pos
                + f"{YELLOW}●{RESET} (zero)"
            )
            print(zero_line)


def print_ascii_timeseries(
    values: list,
    title: str,
    width: int = 70,
    height: int = 12,
    is_percentage: bool = False,
    line_color: str = Colors.BRIGHT_WHITE,
) -> None:
    """
    Print a compact ASCII graph for a time series with line drawing.

    This is intended for quick CLI visualization (equity curves, drawdowns, etc.).

    Args:
        values: Sequence of numeric values (list-like)
        title: Title for the graph
        width: Number of columns for the graph
        height: Number of rows for the graph
        is_percentage: If True, format min/max as percentages
        line_color: Color code for the line (default: white)
    """
    if values is None or len(values) == 0:
        print(f"{title}: No data")
        return

    arr = np.asarray(values, dtype=float)
    arr = arr[~(np.isnan(arr) | np.isinf(arr))]
    if len(arr) == 0:
        print(f"{title}: No valid data")
        return

    # Downsample to width (keep first/last, evenly spaced in between)
    if len(arr) > width:
        idx = np.linspace(0, len(arr) - 1, width).astype(int)
        series = arr[idx]
    else:
        series = arr
        width = len(series)

    min_val = float(np.min(series))
    max_val = float(np.max(series))
    mid_val = (min_val + max_val) / 2.0
    span = max_val - min_val
    if span == 0:
        span = 1.0

    # Map values to rows (0..height-1), top row is max
    # Use float precision for smoother line drawing
    y_float = (series - min_val) / span * (height - 1)
    y = y_float.clip(0, height - 1)

    FRAME = Colors.BRIGHT_WHITE + Colors.BOLD
    RESET = Colors.RESET
    DIM = Colors.DIM
    CYAN = Colors.BRIGHT_CYAN
    BOLD = Colors.BOLD

    def _fmt(v: float) -> str:
        if is_percentage:
            return f"{v:,.2f}%"
        return f"${v:,.2f}"

    # Left axis label width (max of min/mid/max)
    axis_labels = [_fmt(max_val), _fmt(mid_val), _fmt(min_val)]
    axis_w = max(6, max(visible_length(s) for s in axis_labels))

    frame_width = axis_w + 3 + width + 3  # axis + ' │ ' + plot + borders/padding
    title_spaces = frame_width - visible_length(title) - 4  # borders + padding
    print(f"\n{FRAME}{'─' * frame_width}{RESET}")
    print(
        f"{FRAME}│{RESET} {BOLD}{Colors.BRIGHT_WHITE}{title}{RESET}"
        f"{' ' * max(0, title_spaces)} {FRAME}│{RESET}"
    )
    print(f"{FRAME}{'─' * frame_width}{RESET}")

    # Create a 2D grid for plotting (using float precision)
    grid = [[" "] * width for _ in range(height)]

    # Draw line between points - mark each point and fill in between
    for col in range(width):
        y_val = y[col]
        row_idx = int(round(y_val))
        row_idx = max(0, min(height - 1, row_idx))
        grid[height - 1 - row_idx][col] = "●"
        
        # Connect to next point if exists
        if col < width - 1:
            y_next = y[col + 1]
            row_next = int(round(y_next))
            row_next = max(0, min(height - 1, row_next))
            
            # Fill in intermediate rows between points
            if abs(row_next - row_idx) > 1:
                start_row = min(row_idx, row_next)
                end_row = max(row_idx, row_next)
                for r in range(start_row + 1, end_row):
                    # Interpolate to find column position for this row
                    if row_next != row_idx:
                        t = (r - row_idx) / (row_next - row_idx)
                        col_interp = col + t
                        col_idx = int(round(col_interp))
                        if 0 <= col_idx < width and grid[height - 1 - r][col_idx] == " ":
                            grid[height - 1 - r][col_idx] = "·"

    # Tick rows for y-axis labels (top/middle/bottom)
    # row 0 is top (max), row height-1 is bottom (min)
    top_row = 0
    mid_row = (height - 1) // 2
    bot_row = height - 1

    # Build grid (top -> bottom)
    for row in range(height):
        # axis label on 3 tick rows
        if row == top_row:
            axis = f"{CYAN}{axis_labels[0]:>{axis_w}}{RESET}"  # max at top
        elif row == mid_row:
            axis = f"{CYAN}{axis_labels[1]:>{axis_w}}{RESET}"  # mid
        elif row == bot_row:
            axis = f"{CYAN}{axis_labels[2]:>{axis_w}}{RESET}"  # min at bottom
        else:
            axis = " " * axis_w

        # Get row content with color
        line_chars = []
        for col in range(width):
            char = grid[row][col]
            if char != " ":
                line_chars.append(f"{line_color}{char}{RESET}")
            elif row == height - 1 - mid_row:
                # subtle horizontal guide at mid row
                line_chars.append(f"{DIM}·{RESET}")
            else:
                line_chars.append(" ")

        print(f"{FRAME}│{RESET} {axis}{DIM} │{RESET} {''.join(line_chars)} {FRAME}│{RESET}")

    start_val = float(series[0])
    end_val = float(series[-1])

    footer = (
        f"{DIM}start:{RESET} {BOLD}{_fmt(start_val)}{RESET}  "
        f"{DIM}end:{RESET} {BOLD}{_fmt(end_val)}{RESET}  "
        f"{DIM}min:{RESET} {BOLD}{_fmt(min_val)}{RESET}  "
        f"{DIM}max:{RESET} {BOLD}{_fmt(max_val)}{RESET}"
    )
    footer_spaces = max(0, frame_width - visible_length(footer) - 4)
    print(f"{FRAME}{'─' * frame_width}{RESET}")
    print(f"{FRAME}│{RESET} {footer}{' ' * footer_spaces} {FRAME}│{RESET}")
    print(f"{FRAME}{'─' * frame_width}{RESET}")


def print_box(
    width: int,
    header: str,
    lines: list,
    border_color: str = Colors.BRIGHT_WHITE,
) -> None:
    """
    Print a formatted box with header and lines.

    Args:
        width: Width of the box
        header: Header text
        lines: List of (label, value, value_color) tuples or plain strings
        border_color: Color for the border
    """
    header_spaces = width - visible_length(header) - 4
    print(f"\n{border_color}{Colors.BOLD}{'═' * width}{Colors.RESET}")
    print(
        f"{border_color}{Colors.BOLD}║{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{header}{Colors.RESET} {' ' * header_spaces} {border_color}{Colors.BOLD}║{Colors.RESET}"
    )
    print(f"{border_color}{Colors.BOLD}{'═' * width}{Colors.RESET}")

    for line in lines:
        if isinstance(line, tuple):
            label, value, value_color = line
            line_text = f"{Colors.DIM}{label}{Colors.RESET} {Colors.BOLD}{value_color}{value}{Colors.RESET}"
        else:
            line_text = line

        line_spaces = width - visible_length(line_text) - 4
        print(
            f"{border_color}{Colors.BOLD}║{Colors.RESET} {line_text} {' ' * line_spaces} {border_color}{Colors.BOLD}║{Colors.RESET}"
        )

    print(f"{border_color}{Colors.BOLD}{'═' * width}{Colors.RESET}\n")


def print_section_box(
    width: int,
    title: str,
    lines: list,
    border_color: str = Colors.ACCENT,
) -> None:
    """
    Print a formatted section box with title and lines.

    Args:
        width: Width of the box
        title: Title text
        lines: List of strings to print
        border_color: Color for the border
    """
    title_spaces = width - visible_length(title) - 4
    print(f"\n{border_color}{Colors.BOLD}{'─' * width}{Colors.RESET}")
    print(
        f"{border_color}{Colors.BOLD}│{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{title}{Colors.RESET} {' ' * title_spaces} {border_color}{Colors.BOLD}│{Colors.RESET}"
    )

    for line in lines:
        line_spaces = width - visible_length(line) - 4
        print(
            f"{border_color}{Colors.BOLD}│{Colors.RESET} {Colors.DIM}{line}{Colors.RESET} {' ' * line_spaces} {border_color}{Colors.BOLD}│{Colors.RESET}"
        )

    print(f"{border_color}{Colors.BOLD}{'─' * width}{Colors.RESET}\n")


def print_progress_bar(
    current: int,
    target: int,
    width: int = 70,
    bar_width: int = 50,
    border_color: str = Colors.ACCENT,
) -> None:
    """
    Print a progress bar.

    Args:
        current: Current value
        target: Target value
        width: Width of the box
        bar_width: Width of the progress bar
        border_color: Color for the border
    """
    remaining = target - current
    progress_pct = min(100, (current / target) * 100)

    # Create progress bar
    filled = int(bar_width * progress_pct / 100)
    if progress_pct > 0 and filled == 0:
        filled = 1
    empty = bar_width - filled
    bar = f"{Colors.BRIGHT_GREEN}{'█' * filled}{Colors.RESET}{Colors.DIM}{'░' * empty}{Colors.RESET}"

    header_text = "Progress to Target Power"
    header_spaces = width - visible_length(header_text) - 4

    progress_line = f"{current:,} / {target:,} trades ({progress_pct:.1f}%)"
    progress_spaces = width - visible_length(progress_line) - 4

    print(f"{border_color}{Colors.BOLD}{'─' * width}{Colors.RESET}")
    print(
        f"{border_color}{Colors.BOLD}│{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{header_text}{Colors.RESET} {' ' * header_spaces} {border_color}{Colors.BOLD}│{Colors.RESET}"
    )
    print(f"{border_color}{Colors.BOLD}{'─' * width}{Colors.RESET}")
    print(
        f"{border_color}{Colors.BOLD}│{Colors.RESET} {Colors.DIM}{progress_line}{Colors.RESET} {' ' * progress_spaces} {border_color}{Colors.BOLD}│{Colors.RESET}"
    )
    # Progress bar line
    bar_padding = (width - bar_width - 4) // 2
    bar_right_padding = width - bar_width - bar_padding - 4
    print(
        f"{border_color}{Colors.BOLD}│{Colors.RESET} {' ' * bar_padding}{bar}{' ' * bar_right_padding} {border_color}{Colors.BOLD}│{Colors.RESET}"
    )
    print(f"{border_color}{Colors.BOLD}{'─' * width}{Colors.RESET}\n")


def print_comparison_summary(results: dict) -> None:
    """
    Print overall comparison summary.

    Args:
        results: Dictionary with comparison results
    """
    box_width = 80
    lines = [
        (
            "Overall P/L Difference (Live - Backtest):",
            f"${results['overall_pl_diff']:,.2f}",
            Colors.BRIGHT_GREEN if results["overall_pl_diff"] >= 0 else Colors.BRIGHT_RED,
        ),
        (
            "Overall Premium Difference (Live - Backtest, per contract):",
            f"${results['overall_premium_diff']:,.2f}",
            Colors.BRIGHT_GREEN if results["overall_premium_diff"] >= 0 else Colors.BRIGHT_RED,
        ),
        (
            "Total Backtest P/L:",
            f"${results['total_backtest_pl']:,.2f}",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Total Live P/L:",
            f"${results['total_live_pl']:,.2f}",
            Colors.BRIGHT_CYAN,
        ),
    ]
    print_box(box_width, "COMPARISON SUMMARY", lines)


def print_match_statistics(
    results: dict, backtest_df, live_df
) -> None:
    """
    Print matching statistics.

    Args:
        results: Dictionary with comparison results
        backtest_df: Backtest DataFrame
        live_df: Live DataFrame
    """
    box_width = 80
    match_rate_bt = (
        (results["num_matches"] / results["num_backtest_trades"] * 100)
        if results["num_backtest_trades"] > 0
        else 0
    )
    match_rate_live = (
        (results["num_matches"] / results["num_live_trades"] * 100)
        if results["num_live_trades"] > 0
        else 0
    )

    lines = [
        (
            "Backtest Trades:",
            f"{results['num_backtest_trades']}",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Live Trades:",
            f"{results['num_live_trades']}",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Matched Trades:",
            f"{results['num_matches']}",
            Colors.BRIGHT_GREEN,
        ),
        (
            "Fully Matched Trades:",
            f"{results['num_full_matches']}",
            Colors.BRIGHT_GREEN,
        ),
        (
            "Match Rate (Backtest):",
            f"{match_rate_bt:.1f}%",
            Colors.BRIGHT_YELLOW,
        ),
        (
            "Match Rate (Live):",
            f"{match_rate_live:.1f}%",
            Colors.BRIGHT_YELLOW,
        ),
        (
            "Time Window:",
            f"±{results['window_minutes']} minutes",
            Colors.DIM,
        ),
    ]
    print_box(box_width, "MATCHING STATISTICS", lines)


def print_slippage_analysis(results: dict) -> None:
    """
    Print slippage analysis for fully matched trades.

    Args:
        results: Dictionary with comparison results
    """
    stats = results["full_match_stats"]

    if stats["count"] == 0:
        print_section_box(
            80,
            "SLIPPAGE ANALYSIS (FULLY MATCHED TRADES)",
            ["No fully matched trades found for slippage analysis."],
        )
        return

    box_width = 80
    lines = [
        (
            "Fully Matched Trades:",
            f"{stats['count']}",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Mean P/L Difference (Live - Backtest, per contract):",
            f"${stats['mean_pl_diff']:,.2f}",
            Colors.BRIGHT_GREEN if stats["mean_pl_diff"] >= 0 else Colors.BRIGHT_RED,
        ),
        (
            "Median P/L Difference (per contract):",
            f"${stats['median_pl_diff']:,.2f}",
            Colors.BRIGHT_GREEN if stats["median_pl_diff"] >= 0 else Colors.BRIGHT_RED,
        ),
        (
            "Std Dev P/L Difference:",
            f"${stats['std_pl_diff']:,.2f}",
            Colors.BRIGHT_YELLOW,
        ),
        (
            "Mean Entry Difference (Premium Diff):",
            f"${stats['mean_entry_diff']:,.2f}",
            Colors.BRIGHT_GREEN if stats["mean_entry_diff"] >= 0 else Colors.BRIGHT_RED,
        ),
        (
            "Median Entry Difference (Premium Diff):",
            f"${stats['median_entry_diff']:,.2f}",
            Colors.BRIGHT_GREEN if stats["median_entry_diff"] >= 0 else Colors.BRIGHT_RED,
        ),
        (
            "Std Dev Entry Difference:",
            f"${stats['std_entry_diff']:,.2f}",
            Colors.BRIGHT_YELLOW,
        ),
        (
            "Mean Exit Difference (Close Cost Diff):",
            f"${stats['mean_exit_diff']:,.2f}",
            Colors.BRIGHT_GREEN if stats["mean_exit_diff"] >= 0 else Colors.BRIGHT_RED,
        ),
        (
            "Median Exit Difference (Close Cost Diff):",
            f"${stats['median_exit_diff']:,.2f}",
            Colors.BRIGHT_GREEN if stats["median_exit_diff"] >= 0 else Colors.BRIGHT_RED,
        ),
        (
            "Std Dev Exit Difference:",
            f"${stats['std_exit_diff']:,.2f}",
            Colors.BRIGHT_YELLOW,
        ),
        (
            "Backtest Win Rate:",
            f"{stats['backtest_win_rate']*100:.1f}%",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Live Win Rate:",
            f"{stats['live_win_rate']*100:.1f}%",
            Colors.BRIGHT_CYAN,
        ),
    ]
    print_box(box_width, "SLIPPAGE ANALYSIS (FULLY MATCHED TRADES)", lines)

    # Print distribution of entry differences (premium differences)
    if len(stats["entry_diffs"]) > 0:
        print_ascii_distribution(
            stats["entry_diffs"],
            f"Entry Difference Distribution (Premium Diff, Live - Backtest) - Fully Matched Trades Only (N={stats['count']})",
        )
        # Add note about binning (histogram width is 60, frame width is 78)
        note = "Note: Histogram shows binned data (values are bin centers representing ranges). See table above for exact per-trade values."
        note_spaces = 78 - visible_length(note) - 4
        print(f"{Colors.DIM}{Colors.BRIGHT_WHITE}{Colors.BOLD}│{Colors.RESET} {Colors.DIM}{note}{Colors.RESET} {' ' * note_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}│{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'─' * 78}{Colors.RESET}")

    # Print distribution of exit differences (closing cost differences)
    if len(stats["exit_diffs"]) > 0:
        print_ascii_distribution(
            stats["exit_diffs"],
            f"Exit Difference Distribution (Close Cost Diff, Live - Backtest) - Fully Matched Trades Only (N={stats['count']})",
        )
        # Add note about binning (histogram width is 60, frame width is 78)
        note = "Note: Histogram shows binned data (values are bin centers representing ranges). See table above for exact per-trade values."
        note_spaces = 78 - visible_length(note) - 4
        print(f"{Colors.DIM}{Colors.BRIGHT_WHITE}{Colors.BOLD}│{Colors.RESET} {Colors.DIM}{note}{Colors.RESET} {' ' * note_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}│{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'─' * 78}{Colors.RESET}")

    # Print distribution of P/L differences (for fully matched trades only)
    if len(stats["pl_diffs"]) > 0:
        print_ascii_distribution(
            stats["pl_diffs"],
            f"P/L Difference Distribution (Live - Backtest, per contract) - Fully Matched Trades Only (N={stats['count']})",
        )
        # Add note about binning (histogram width is 60, frame width is 78)
        note = "Note: Histogram shows binned data (values are bin centers representing ranges). See table above for exact per-trade values."
        note_spaces = 78 - visible_length(note) - 4
        print(f"{Colors.DIM}{Colors.BRIGHT_WHITE}{Colors.BOLD}│{Colors.RESET} {Colors.DIM}{note}{Colors.RESET} {' ' * note_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}│{Colors.RESET}")
        print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'─' * 78}{Colors.RESET}")


def print_missed_trades(results: dict) -> None:
    """
    Print missed trades (in backtest but not in live).

    Args:
        results: Dictionary with comparison results
    """
    missed = results["missed_trades"]
    num_missed = len(missed)

    if num_missed == 0:
        print_section_box(
            80,
            "MISSED TRADES",
            ["No missed trades - all backtest trades were matched."],
        )
        return

    box_width = 80
    lines = [
        (
            "Number of Missed Trades:",
            f"{num_missed}",
            Colors.BRIGHT_YELLOW,
        ),
        (
            "Total P/L of Missed Trades:",
            f"${missed['P/L'].sum():,.2f}",
            Colors.BRIGHT_CYAN,
        ),
    ]
    print_box(box_width, "MISSED TRADES", lines)

    # Print details of missed trades
    if num_missed > 0 and num_missed <= 20:  # Only show details if not too many
        print_section_box(
            80,
            "Missed Trade Details",
            [
                f"Date: {row.get('Date Opened', 'N/A')}, "
                f"Time: {row.get('Time Opened', 'N/A')}, "
                f"P/L: ${row.get('P/L', 0):,.2f}, "
                f"Strategy: {row.get('Strategy', 'N/A')}"
                for _, row in missed.head(20).iterrows()
            ],
        )


def print_over_trades(results: dict) -> None:
    """
    Print over trades (in live but not in backtest).

    Args:
        results: Dictionary with comparison results
    """
    over = results["over_trades"]
    num_over = len(over)

    if num_over == 0:
        print_section_box(
            80,
            "OVER TRADES",
            ["No over trades - all live trades were matched."],
        )
        return

    box_width = 80
    lines = [
        (
            "Number of Over Trades:",
            f"{num_over}",
            Colors.BRIGHT_YELLOW,
        ),
        (
            "Total P/L of Over Trades:",
            f"${over['P/L'].sum():,.2f}",
            Colors.BRIGHT_CYAN,
        ),
    ]
    print_box(box_width, "OVER TRADES", lines)

    # Print details of over trades
    if num_over > 0 and num_over <= 20:  # Only show details if not too many
        print_section_box(
            80,
            "Over Trade Details",
            [
                f"Date: {row.get('Date Opened', 'N/A')}, "
                f"Time: {row.get('Time Opened', 'N/A')}, "
                f"P/L: ${row.get('P/L', 0):,.2f}, "
                f"Strategy: {row.get('Strategy', 'N/A')}"
                for _, row in over.head(20).iterrows()
            ],
        )


def print_allocation_analysis(results: dict) -> None:
    """
    Print allocation consistency analysis.

    Args:
        results: Dictionary with comparison results
    """
    alloc = results.get("allocation_analysis", {})
    
    if not alloc or alloc.get("mean_backtest_allocation", 0) == 0:
        print_section_box(
            80,
            "ALLOCATION CONSISTENCY ANALYSIS",
            ["Allocation analysis not available (missing Funds at Close or Margin Req. columns)."],
        )
        return

    box_width = 80
    lines = [
        (
            "Mean Backtest Allocation:",
            f"{alloc['mean_backtest_allocation']:.2f}%",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Std Dev Backtest Allocation:",
            f"{alloc['std_backtest_allocation']:.2f}%",
            Colors.BRIGHT_YELLOW,
        ),
        (
            "Std Dev Live Allocation:",
            f"{alloc.get('std_live_allocation', 0):.2f}%",
            Colors.BRIGHT_YELLOW,
        ),
        (
            "Mean Live Allocation:",
            f"{alloc['mean_live_allocation']:.2f}%",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Starting Portfolio Size:",
            f"${alloc['starting_portfolio_size']:,.2f}",
            Colors.DIM,
        ),
        (
            "Deviant Trades (>2 std dev from respective mean):",
            f"{alloc['num_deviant_trades']}",
            Colors.BRIGHT_RED if alloc['num_deviant_trades'] > 0 else Colors.BRIGHT_GREEN,
        ),
    ]
    print_box(box_width, "ALLOCATION CONSISTENCY ANALYSIS", lines)

    # Print deviant trades if any
    deviant_trades = alloc.get("deviant_trades", [])
    if deviant_trades:
        print_section_box(
            80,
            "Trades with Significant Allocation Deviations",
            [
                f"Date: {d['date']}, Time: {d['time']}, "
                f"Allocation: {d['allocation_pct']:.2f}% "
                f"(Mean: {d.get('mean_live_allocation', d.get('mean_backtest_allocation', 0)):.2f}%, "
                f"Deviation: {d['deviation']:.2f}%)"
                for d in deviant_trades[:20]  # Limit to 20
            ],
        )


def print_matched_trades_table(results: dict) -> None:
    """
    Print a table of matched trades for comparison.

    Args:
        results: Dictionary with comparison results
    """
    table_data = results.get("matched_trades_table", [])
    
    if not table_data:
        print_section_box(
            80,
            "MATCHED TRADES COMPARISON",
            ["No matched trades to display."],
        )
        return

    box_width = 220
    header_text = "MATCHED TRADES COMPARISON"
    header_spaces = box_width - visible_length(header_text) - 4

    print(f"\n{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}")
    print(
        f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{header_text}{Colors.RESET} {' ' * header_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
    )
    print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}")

    # Print table header with consistent column widths and proper spacing
    header_line = (
        f"{'Date':<12} {'Time':<12} {'Premium BT':>13} {'Premium Live':>14} {'Premium Diff':>14} "
        f"{'Close Cost BT':>15} {'Close Cost Live':>17} {'Close Cost Diff':>16} "
        f"{'P/L BT':>10} {'P/L Live':>11} {'P/L Diff':>10} {'P/L Diff/C':>11} "
        f"{'Cont BT':>9} {'Cont Live':>10} {'Margin/Cont':>12} {'Alloc BT':>9} {'Alloc Live':>9} {'Match':>6}"
    )
    header_spaces = box_width - visible_length(header_line) - 4
    print(
        f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_CYAN}{header_line}{Colors.RESET} {' ' * header_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
    )
    print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'─' * box_width}{Colors.RESET}")

    # Print table rows (limit to 50 for readability)
    for row in table_data[:50]:
        date = str(row.get("date", ""))[:10]
        time_bt = str(row.get("time_bt", ""))[:8]
        premium_bt = row.get("premium_bt", 0)
        premium_live = row.get("premium_live", 0)
        premium_diff = row.get("premium_diff", 0)
        closing_cost_bt = row.get("closing_cost_bt", 0)
        closing_cost_live = row.get("closing_cost_live", 0)
        closing_cost_diff = row.get("closing_cost_diff", 0)
        pl_bt = row.get("pl_bt", 0)
        pl_live = row.get("pl_live", 0)
        pl_diff = row.get("pl_diff", 0)
        pl_diff_per_contract = row.get("pl_diff_per_contract", 0)
        contracts_bt = row.get("contracts_bt", 0)
        contracts_live = row.get("contracts_live", 0)
        margin_per_contract = row.get("margin_per_contract", 0)
        alloc_bt = row.get("alloc_bt", 0)
        alloc_live = row.get("alloc_live", 0)
        is_full_match = row.get("legs_match", False) and row.get("reason_match", False)
        
        match_indicator = "✓" if is_full_match else "~"
        match_color = Colors.BRIGHT_GREEN if is_full_match else Colors.BRIGHT_YELLOW
        
        pl_diff_color = Colors.BRIGHT_GREEN if pl_diff >= 0 else Colors.BRIGHT_RED
        pl_diff_per_contract_color = Colors.BRIGHT_GREEN if pl_diff_per_contract >= 0 else Colors.BRIGHT_RED
        premium_diff_color = Colors.BRIGHT_GREEN if premium_diff >= 0 else Colors.BRIGHT_RED
        closing_cost_diff_color = Colors.BRIGHT_GREEN if closing_cost_diff >= 0 else Colors.BRIGHT_RED
        
        # Format margin per contract (show as 0 if not available)
        if margin_per_contract > 0:
            margin_str = f"${margin_per_contract:>10,.2f}"
        else:
            margin_str = f"{'N/A':>12}"
        
        # Format allocation percentages (right-aligned for consistency)
        if alloc_bt > 0:
            alloc_bt_str = f"{alloc_bt:>8.2f}%"
        else:
            alloc_bt_str = f"{'N/A':>9}"
        
        if alloc_live > 0:
            alloc_live_str = f"{alloc_live:>8.2f}%"
        else:
            alloc_live_str = f"{'N/A':>9}"
        
        data_line = (
            f"{date:<12} {time_bt:<12} ${premium_bt:>11,.2f} ${premium_live:>12,.2f} {premium_diff_color}${premium_diff:>12,.2f}{Colors.RESET} "
            f"${closing_cost_bt:>13,.2f} ${closing_cost_live:>15,.2f} {closing_cost_diff_color}${closing_cost_diff:>14,.2f}{Colors.RESET} "
            f"${pl_bt:>8,.2f} ${pl_live:>9,.2f} {pl_diff_color}${pl_diff:>8,.2f}{Colors.RESET} "
            f"{pl_diff_per_contract_color}${pl_diff_per_contract:>9,.2f}{Colors.RESET} "
            f"{int(contracts_bt):>9} {int(contracts_live):>10} {margin_str:>12} {alloc_bt_str:>9} {alloc_live_str:>9} {match_color}{match_indicator:>5}{Colors.RESET}"
        )
        data_spaces = box_width - visible_length(data_line) - 4
        print(
            f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {data_line} {' ' * data_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
        )

    if len(table_data) > 50:
        note = f"... and {len(table_data) - 50} more trades (showing first 50)"
        note_spaces = box_width - visible_length(note) - 4
        print(
            f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.DIM}{note}{Colors.RESET} {' ' * note_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
        )

    print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}\n")


def print_pl_breakdown(results: dict) -> None:
    """
    Print P/L difference breakdown by category.

    Args:
        results: Dictionary with comparison results
    """
    breakdown = results.get("pl_breakdown", {})
    
    if not breakdown:
        return
    
    box_width = 80
    lines = []
    
    # Over trading
    over_trading = breakdown.get("over_trading", {})
    over_trading_val = over_trading.get("value", 0)
    over_trading_pct = over_trading.get("percentage", 0)
    if abs(over_trading_val) > 0.01:
        color = Colors.BRIGHT_RED if over_trading_val < 0 else Colors.BRIGHT_GREEN
        lines.append((
            f"${over_trading_val:,.2f} ({over_trading_pct:.1f}%) diff was due to over trading",
            "",
            color,
        ))
    
    # Missed trades
    missed_trades = breakdown.get("missed_trades", {})
    missed_trades_val = missed_trades.get("value", 0)
    missed_trades_pct = missed_trades.get("percentage", 0)
    if abs(missed_trades_val) > 0.01:
        color = Colors.BRIGHT_RED if missed_trades_val < 0 else Colors.BRIGHT_GREEN
        lines.append((
            f"${missed_trades_val:,.2f} ({missed_trades_pct:.1f}%) diff was due to missed trades",
            "",
            color,
        ))
    
    # Entry slippage
    entry_slippage = breakdown.get("entry_slippage", {})
    entry_slippage_val = entry_slippage.get("value", 0)
    entry_slippage_pct = entry_slippage.get("percentage", 0)
    if abs(entry_slippage_val) > 0.01:
        color = Colors.BRIGHT_RED if entry_slippage_val < 0 else Colors.BRIGHT_GREEN
        lines.append((
            f"${entry_slippage_val:,.2f} ({entry_slippage_pct:.1f}%) diff was due to entry slippage",
            "",
            color,
        ))
    
    # Exit slippage
    exit_slippage = breakdown.get("exit_slippage", {})
    exit_slippage_val = exit_slippage.get("value", 0)
    exit_slippage_pct = exit_slippage.get("percentage", 0)
    if abs(exit_slippage_val) > 0.01:
        color = Colors.BRIGHT_RED if exit_slippage_val < 0 else Colors.BRIGHT_GREEN
        lines.append((
            f"${exit_slippage_val:,.2f} ({exit_slippage_pct:.1f}%) diff was due to exit slippage",
            "",
            color,
        ))
    
    # Under-allocation
    under_allocation = breakdown.get("under_allocation", {})
    under_allocation_val = under_allocation.get("value", 0)
    under_allocation_pct = under_allocation.get("percentage", 0)
    if abs(under_allocation_val) > 0.01:
        color = Colors.BRIGHT_RED if under_allocation_val < 0 else Colors.BRIGHT_GREEN
        lines.append((
            f"${under_allocation_val:,.2f} ({under_allocation_pct:.1f}%) diff was due to under-allocation",
            "",
            color,
        ))
    
    # Over-allocation
    over_allocation = breakdown.get("over_allocation", {})
    over_allocation_val = over_allocation.get("value", 0)
    over_allocation_pct = over_allocation.get("percentage", 0)
    if abs(over_allocation_val) > 0.01:
        color = Colors.BRIGHT_RED if over_allocation_val < 0 else Colors.BRIGHT_GREEN
        lines.append((
            f"${over_allocation_val:,.2f} ({over_allocation_pct:.1f}%) diff was due to over-allocation",
            "",
            color,
        ))
    
    if not lines:
        lines.append((
            "No significant P/L differences to attribute",
            "",
            Colors.DIM,
        ))
    
    print_box(box_width, "P/L DIFFERENCE BREAKDOWN", lines)


def print_margin_check(
    margin_check: dict, portfolio_size: float, desired_allocation_pct: float
) -> None:
    """
    Print margin requirement check analysis.

    Args:
        margin_check: Dictionary with margin check results
        portfolio_size: Initial portfolio size
        desired_allocation_pct: Desired allocation percentage
    """
    if not margin_check.get("has_margin_data", False):
        print_section_box(
            80,
            "CAN YOU RUN IT - MARGIN CHECK",
            ["Margin requirement data not available (backtest data only)."],
        )
        return

    box_width = 80
    allocations = margin_check.get("allocations", [])
    mean_alloc = margin_check.get("mean_allocation_pct", 0.0)
    max_alloc = margin_check.get("max_allocation_pct", 0.0)
    mean_margin = margin_check.get("mean_margin_per_contract", 0.0)
    max_margin = margin_check.get("max_margin_per_contract", 0.0)

    lines = [
        (
            "Portfolio Size:",
            f"${portfolio_size:,.2f}",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Desired Allocation:",
            f"{desired_allocation_pct:.2f}%",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Mean Margin Allocation (1-lot):",
            f"{mean_alloc:.2f}% (${mean_margin:,.2f})",
            Colors.BRIGHT_YELLOW,
        ),
        (
            "Max Margin Allocation (1-lot, conservative):",
            f"{max_alloc:.2f}% (${max_margin:,.2f})",
            Colors.BRIGHT_RED if max_alloc > desired_allocation_pct else Colors.BRIGHT_GREEN,
        ),
    ]

    if max_alloc > desired_allocation_pct:
        warning = (
            f"⚠ WARNING: Max margin ({max_alloc:.2f}%) exceeds desired allocation "
            f"({desired_allocation_pct:.2f}%). You may need to expect {max_alloc/desired_allocation_pct:.1f}x "
            f"the drawdown."
        )
        lines.append((warning, "", Colors.BRIGHT_YELLOW))
    else:
        success = (
            f"✓ Portfolio size is sufficient for desired allocation "
            f"(max margin {max_alloc:.2f}% <= desired {desired_allocation_pct:.2f}%)"
        )
        lines.append((success, "", Colors.BRIGHT_GREEN))

    print_box(box_width, "CAN YOU RUN IT - MARGIN CHECK", lines)

    # Print histogram of allocations
    if len(allocations) > 0:
        print_ascii_distribution(
            np.array(allocations),
            "Margin Allocation Distribution (1-lot, % of portfolio)",
            is_percentage=True,
        )


def print_biggest_loss(margin_check: dict, portfolio_size: float) -> None:
    """
    Print biggest loss per contract information.

    Args:
        margin_check: Dictionary with margin check results
        portfolio_size: Initial portfolio size
    """
    biggest_loss = margin_check.get("biggest_loss_per_contract", 0.0)
    biggest_loss_pct = margin_check.get("biggest_loss_pct_of_portfolio", 0.0)

    if biggest_loss == 0.0:
        return

    box_width = 80
    lines = [
        (
            "Biggest Loss per Contract:",
            f"${biggest_loss:,.2f}",
            Colors.BRIGHT_RED,
        ),
        (
            "As % of Portfolio:",
            f"{biggest_loss_pct:.2f}%",
            Colors.BRIGHT_RED,
        ),
    ]
    print_box(box_width, "BIGGEST LOSS", lines)


def print_drawdown_metrics(drawdown_results: dict) -> None:
    """
    Print drawdown metrics.

    Args:
        drawdown_results: Dictionary with drawdown metrics
    """
    box_width = 80
    max_dd_dollars = drawdown_results.get("max_drawdown_dollars", 0.0)
    max_dd_pct = drawdown_results.get("max_drawdown_pct", 0.0)
    longest_dd = drawdown_results.get("longest_drawdown")
    shortest_dd = drawdown_results.get("shortest_drawdown")
    num_drawdowns = drawdown_results.get("num_drawdowns", 0)
    avg_length = drawdown_results.get("average_drawdown_length", 0.0)
    percent_time = drawdown_results.get("percent_time_in_drawdown", 0.0)

    lines = [
        (
            "Max Drawdown:",
            f"${max_dd_dollars:,.2f}",
            Colors.BRIGHT_RED,
        ),
        (
            "Max Drawdown (%):",
            f"{max_dd_pct:.2f}%",
            Colors.BRIGHT_RED,
        ),
    ]

    if longest_dd:
        start_date = longest_dd["start_date"]
        end_date = longest_dd["end_date"]
        length_days = longest_dd["length_days"]
        lines.append((
            "Longest Drawdown:",
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({length_days} days)",
            Colors.BRIGHT_YELLOW,
        ))

    if shortest_dd:
        start_date = shortest_dd["start_date"]
        end_date = shortest_dd["end_date"]
        length_days = shortest_dd["length_days"]
        lines.append((
            "Shortest Drawdown:",
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({length_days} days)",
            Colors.BRIGHT_CYAN,
        ))

    lines.extend([
        (
            "Number of Drawdowns:",
            f"{num_drawdowns}",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Average Drawdown Length:",
            f"{avg_length:.1f} days",
            Colors.BRIGHT_CYAN,
        ),
        (
            "Percent Time in Drawdown:",
            f"{percent_time:.2f}%",
            Colors.BRIGHT_YELLOW,
        ),
    ])

    # Add over-allocation info if available
    over_allocated_trades = drawdown_results.get("over_allocated_trades", 0)
    over_allocation_details = drawdown_results.get("over_allocation_details", [])
    if over_allocated_trades > 0 and over_allocation_details:
        # Calculate average over-allocation
        avg_over_allocation = np.mean([d.get("over_allocation_pct", 0.0) for d in over_allocation_details])
        max_over_allocation = max([d.get("over_allocation_pct", 0.0) for d in over_allocation_details])
        
        lines.append((
            "Over-Allocated Trades (forced 1-lot):",
            f"{over_allocated_trades} trades",
            Colors.BRIGHT_YELLOW,
        ))
        lines.append((
            "Average Over-Allocation:",
            f"{avg_over_allocation:.2f}%",
            Colors.BRIGHT_YELLOW,
        ))
        lines.append((
            "Max Over-Allocation:",
            f"{max_over_allocation:.2f}%",
            Colors.BRIGHT_RED,
        ))

    print_box(box_width, "DRAWDOWN METRICS", lines)


def print_drawdown_calendar(calendar_results: dict) -> None:
    """
    Print ASCII calendar visualizations for weekly and monthly profitability.

    Args:
        calendar_results: Dictionary with weekly and monthly calendar data
    """
    weekly = calendar_results.get("weekly", {})
    monthly = calendar_results.get("monthly", {})

    if not weekly and not monthly:
        print_section_box(
            80,
            "CALENDAR VISUALIZATION",
            ["No calendar data available."],
        )
        return

    # Print weekly calendar
    if weekly:
        print_section_box(
            80,
            "WEEKLY PROFITABILITY CALENDAR",
            ["Green = profitable week, Red = losing week"],
        )
        _print_calendar_grid(weekly, "week")

    # Print monthly calendar
    if monthly:
        print_section_box(
            80,
            "MONTHLY PROFITABILITY CALENDAR",
            ["Green = profitable month, Red = losing month"],
        )
        _print_calendar_grid(monthly, "month")


def _print_calendar_grid(period_data: dict, period_type: str) -> None:
    """
    Print a grid calendar showing profitable/losing periods.

    Args:
        period_data: Dictionary mapping period strings to {"pl": float, "profitable": bool}
        period_type: "week" or "month"
    """
    if not period_data:
        return

    # Sort periods chronologically
    sorted_periods = sorted(period_data.items())

    # Group into rows (7 columns for weeks, 4 columns for months)
    cols_per_row = 7 if period_type == "week" else 4
    rows = []
    for i in range(0, len(sorted_periods), cols_per_row):
        rows.append(sorted_periods[i:i + cols_per_row])

    # Print header
    box_width = 80
    header_text = f"{period_type.upper()} CALENDAR"
    header_spaces = box_width - visible_length(header_text) - 4
    print(f"\n{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}")
    print(
        f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{header_text}{Colors.RESET} {' ' * header_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
    )
    print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'─' * box_width}{Colors.RESET}")

    # Print calendar grid
    for row in rows:
        row_text = ""
        for period_str, data in row:
            # Format period string
            # Pandas Period objects convert to strings like "2025-W01" for weeks or "2025-01" for months
            if period_type == "week":
                # Period format: "2025-W01" -> "2025-W01"
                if "W" in period_str:
                    period_display = period_str[:8]  # "2025-W01"
                else:
                    # Fallback: try to extract date
                    period_display = period_str[:7] if len(period_str) >= 7 else period_str
            else:
                # Period format: "2025-01" -> "2025-01"
                period_display = period_str[:7] if len(period_str) >= 7 else period_str

            # Color based on profitability
            color = Colors.BRIGHT_GREEN if data["profitable"] else Colors.BRIGHT_RED
            symbol = "█"
            
            # Format: symbol + period (max 10 chars for period)
            period_display = period_display[:10]
            row_text += f"{color}{symbol}{Colors.RESET} {period_display:<10}"

        row_spaces = box_width - visible_length(row_text) - 4
        print(
            f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {row_text} {' ' * row_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
        )

    print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}\n")
