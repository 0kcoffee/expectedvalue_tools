"""Main CLI entry point using Typer."""

import sys
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import typer
from ..parsers import BaseParser, OptionOmegaParser
from ..normalizers import BaseNormalizer, TradeDataNormalizer
from ..enrichers import BaseEnricher, TradeEnricher
from ..tests import BaseTest, PowerAnalysisTest, LiveBacktestComparisonTest, DrawdownAnalysisTest, TailOverfittingTest, TrackTest, PortfolioStressTest, PortfolioCorrelationTest
from ..output.visualizers import create_histograms
from ..output.formatters import print_box
from ..output.html_reporter import HTMLReporter
from ..utils.colors import Colors
from ..utils.text import visible_length

app = typer.Typer(
    name="evtools",
    help="Expected Value Tools - Statistical analysis for trading strategies",
    add_completion=False,
)

# Registry for parsers (auto-detect format)
PARSERS = [OptionOmegaParser()]

# Registry for tests (auto-discover)
TESTS = [PowerAnalysisTest(), LiveBacktestComparisonTest(), DrawdownAnalysisTest(), TailOverfittingTest(), TrackTest(), PortfolioStressTest(), PortfolioCorrelationTest()]

# Default normalizer and enricher
NORMALIZER = TradeDataNormalizer()
ENRICHER = TradeEnricher()


def _print_header():
    """Print the ASCII art header."""
    ascii_art = [
        "                             ",
        "     ++++      +++++++++++++++",
        "   +++ ++++  +++++          ++",
        " +++     +++++++++          ++",
        "++         +++++++++++++    ++",
        "+++          +++++++++++    ++",
        "  +++          +++++++++    ++",
        "    +++          +++++++++++++",
        "      +++          +++++++++++",
        "      +++            +++++++  ",
        "    +++                +++    ",
        "  ++++                  ++++  ",
        "++++         ++++         ++++",
        "++         +++++++         +++",
        " +++     ++++    +++      +++ ",
        "   +++  +++        +++  +++   ",
        "    +++++            ++++     ",
    ]

    terminal_width = 80
    for line in ascii_art:
        spaces = (terminal_width - len(line)) // 2
        print(f"{' ' * spaces}{Colors.ACCENT}{Colors.BOLD}{line}{Colors.RESET}")

    title = "the  E[X]PECTED  VALUE"
    title_colored = title.replace(
        "X", f"{Colors.BOLD}X{Colors.RESET}{Colors.ACCENT}{Colors.BOLD}"
    )
    title_spaces = (terminal_width - visible_length(title)) // 2
    print(
        f"{' ' * title_spaces}{Colors.ACCENT}{Colors.BOLD}{title_colored}{Colors.RESET}"
    )

    url = "https://expectedvalue.trade"
    url_spaces = (terminal_width - visible_length(url)) // 2
    print(f"{' ' * url_spaces}{Colors.ACCENT}{url}{Colors.RESET}")
    print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'─' * 80}{Colors.RESET}\n")


def _find_parser(file_path: str) -> BaseParser:
    """
    Find a parser that can handle the given file.

    Args:
        file_path: Path to the file

    Returns:
        Parser instance

    Raises:
        ValueError: If no parser can handle the file
    """
    for parser in PARSERS:
        if parser.detect_format(file_path):
            return parser
    raise ValueError(f"No parser found for file: {file_path}")


def _find_test(test_name: str) -> BaseTest:
    """
    Find a test by name.

    Args:
        test_name: Name of the test

    Returns:
        Test instance

    Raises:
        ValueError: If test not found
    """
    for test in TESTS:
        if test.get_name() == test_name:
            return test
    raise ValueError(f"Test not found: {test_name}. Available tests: {[t.get_name() for t in TESTS]}")


def _process_file(
    file_path: str,
    test: BaseTest,
    test_kwargs: dict,
    output_dir: Optional[str],
    html_report: bool = False,
) -> None:
    """
    Process a file through the pipeline: parse -> normalize -> enrich -> test.

    Args:
        file_path: Path to the file
        test: Test instance to run
        test_kwargs: Additional arguments for the test
        output_dir: Directory to save visualizations
    """
    # Parse
    parser = _find_parser(file_path)
    df = parser.parse(file_path)
    metadata = parser.get_metadata(df)

    # Normalize
    df = NORMALIZER.normalize(df)

    # Enrich
    df = ENRICHER.enrich(df)

    # Check if portfolio
    is_portfolio = metadata.get("is_portfolio", False)

    if is_portfolio:
        # Process each strategy separately
        strategies = metadata.get("strategies", [])
        box_width = 80
        lines = [
            (
                "Detected portfolio with",
                f"{len(strategies)}",
                Colors.BRIGHT_CYAN,
            ),
        ]
        print_box(box_width, "PORTFOLIO ANALYSIS", lines)

        for strategy in strategies:
            strategy_data = df[df["Strategy"] == strategy]
            _process_strategy(
                strategy_data,
                str(strategy),
                test,
                test_kwargs,
                output_dir,
                html_report,
            )
    else:
        # Single strategy
        strategy_name = "Single Strategy"
        unique_strategies = metadata.get("strategies", [])
        if len(unique_strategies) == 1:
            strategy_name = str(unique_strategies[0])

        _process_strategy(df, strategy_name, test, test_kwargs, output_dir, html_report)


def _generate_html_report(
    html_reporter: HTMLReporter,
    test: BaseTest,
    results: dict,
    data: pd.DataFrame,
    histogram_figure=None,
) -> None:
    """
    Generate HTML report from test results.

    Args:
        html_reporter: HTMLReporter instance
        test: Test instance
        results: Test results dictionary
        data: Original data DataFrame
        histogram_figure: Optional matplotlib figure for histograms
    """
    test_name = test.get_name()

    # Add histogram if available
    if histogram_figure is not None:
        html_reporter.add_chart(histogram_figure, "P/L Distribution Histograms")

    # Test-specific content
    if test_name == "power":
        _add_power_html(html_reporter, results, data)
    elif test_name == "compare":
        _add_compare_html(html_reporter, results)
    elif test_name == "drawdown":
        _add_drawdown_html(html_reporter, results)
    elif test_name == "tail_overfitting":
        _add_tail_overfitting_html(html_reporter, results)
    elif test_name == "track":
        _add_track_html(html_reporter, results)
    elif test_name == "portfolio_stress":
        _add_portfolio_stress_html(html_reporter, results)
    elif test_name == "portfolio_correlation":
        _add_portfolio_correlation_html(html_reporter, results)


def _add_power_html(html_reporter: HTMLReporter, results: dict, data: pd.DataFrame) -> None:
    """Add power analysis test content to HTML report."""
    # Performance metrics
    stats = [
        ("Observed Edge (Mean Return per Contract)", f"${results.get('observed_mean', 0):,.2f}", "value-positive" if results.get('observed_mean', 0) >= 0 else "value-negative"),
        ("Observed Volatility", f"${results.get('observed_std', 0):,.2f}", "value-default"),
        ("Current Sample Size", f"{results.get('current_n', 0):,}", "value-info"),
        ("Win Rate", f"{results.get('win_rate', 0)*100:.2f}%", "value-positive" if results.get('win_rate', 0) > 0.5 else "value-default"),
    ]
    html_reporter.add_statistics_box("Strategy Performance Metrics", stats)

    # Power assessment
    current_power = results.get("current_power")
    if current_power is not None:
        power_stats = [
            ("Statistical Power", f"{current_power*100:.2f}%", "value-positive" if current_power >= 0.95 else "value-warning"),
            ("Current Sample Size", f"{results.get('current_n', 0):,}", "value-info"),
        ]
        html_reporter.add_statistics_box(f"Power Assessment (Current Sample: N={results.get('current_n', 0):,})", power_stats)

    # Recommendation
    recommended_n = results.get("recommended_n")
    if recommended_n is not None:
        target_power = results.get("target_power", 0.95)
        rec_stats = [
            ("Minimum Required Sample Size", f"{recommended_n:,} trades", "value-info"),
            ("Target Confidence", f"{target_power*100:.0f}%", "value-positive"),
        ]
        html_reporter.add_statistics_box("Recommendation", rec_stats)


def _add_compare_html(html_reporter: HTMLReporter, results: dict) -> None:
    """Add live/backtest comparison test content to HTML report."""
    # Comparison summary
    stats = [
        ("Overall P/L Difference (Live - Backtest)", f"${results.get('overall_pl_diff', 0):,.2f}", "value-positive" if results.get('overall_pl_diff', 0) >= 0 else "value-negative"),
        ("Total Backtest P/L", f"${results.get('total_backtest_pl', 0):,.2f}", "value-info"),
        ("Total Live P/L", f"${results.get('total_live_pl', 0):,.2f}", "value-info"),
    ]
    html_reporter.add_statistics_box("Comparison Summary", stats)

    # Matching statistics
    num_matches = results.get("num_matches", 0)
    num_full_matches = results.get("num_full_matches", 0)
    num_backtest = results.get("num_backtest_trades", 0)
    num_live = results.get("num_live_trades", 0)
    match_rate_bt = (num_matches / num_backtest * 100) if num_backtest > 0 else 0
    match_rate_live = (num_matches / num_live * 100) if num_live > 0 else 0

    match_stats = [
        ("Backtest Trades", f"{num_backtest}", "value-info"),
        ("Live Trades", f"{num_live}", "value-info"),
        ("Matched Trades", f"{num_matches}", "value-positive"),
        ("Fully Matched Trades", f"{num_full_matches}", "value-positive"),
        ("Match Rate (Backtest)", f"{match_rate_bt:.1f}%", "value-positive" if match_rate_bt > 80 else "value-warning"),
        ("Match Rate (Live)", f"{match_rate_live:.1f}%", "value-positive" if match_rate_live > 80 else "value-warning"),
        ("Time Window", f"±{results.get('window_minutes', 0)} minutes", "value-default"),
    ]
    html_reporter.add_statistics_box("Matching Statistics", match_stats)

    # Slippage analysis
    full_match_stats = results.get("full_match_stats", {})
    if full_match_stats.get("count", 0) > 0:
        slippage_stats = [
            ("Fully Matched Trades", f"{full_match_stats.get('count', 0)}", "value-info"),
            ("Mean P/L Difference (Live - Backtest, per contract)", f"${full_match_stats.get('mean_pl_diff', 0):,.2f}", "value-positive" if full_match_stats.get('mean_pl_diff', 0) >= 0 else "value-negative"),
            ("Median P/L Difference (per contract)", f"${full_match_stats.get('median_pl_diff', 0):,.2f}", "value-positive" if full_match_stats.get('median_pl_diff', 0) >= 0 else "value-negative"),
            ("Std Dev P/L Difference", f"${full_match_stats.get('std_pl_diff', 0):,.2f}", "value-warning"),
            ("Mean Entry Difference (Premium Diff)", f"${full_match_stats.get('mean_entry_diff', 0):,.2f}", "value-positive" if full_match_stats.get('mean_entry_diff', 0) >= 0 else "value-negative"),
            ("Median Entry Difference (Premium Diff)", f"${full_match_stats.get('median_entry_diff', 0):,.2f}", "value-positive" if full_match_stats.get('median_entry_diff', 0) >= 0 else "value-negative"),
            ("Std Dev Entry Difference", f"${full_match_stats.get('std_entry_diff', 0):,.2f}", "value-warning"),
            ("Mean Exit Difference (Close Cost Diff)", f"${full_match_stats.get('mean_exit_diff', 0):,.2f}", "value-positive" if full_match_stats.get('mean_exit_diff', 0) >= 0 else "value-negative"),
            ("Median Exit Difference (Close Cost Diff)", f"${full_match_stats.get('median_exit_diff', 0):,.2f}", "value-positive" if full_match_stats.get('median_exit_diff', 0) >= 0 else "value-negative"),
            ("Std Dev Exit Difference", f"${full_match_stats.get('std_exit_diff', 0):,.2f}", "value-warning"),
            ("Backtest Win Rate", f"{full_match_stats.get('backtest_win_rate', 0)*100:.1f}%", "value-info"),
            ("Live Win Rate", f"{full_match_stats.get('live_win_rate', 0)*100:.1f}%", "value-info"),
        ]
        html_reporter.add_statistics_box("Slippage Analysis (Fully Matched Trades)", slippage_stats)

        # Add distribution histograms for entry, exit, and P/L differences
        entry_diffs = full_match_stats.get("entry_diffs")
        if entry_diffs is not None and len(entry_diffs) > 0:
            # Convert to numpy array if not already
            if not isinstance(entry_diffs, np.ndarray):
                entry_diffs = np.array(entry_diffs)
            html_reporter.add_distribution_chart(
                entry_diffs,
                "Entry Difference Distribution (Premium Diff, Live - Backtest) - Fully Matched Trades Only",
                is_percentage=False,
            )
        
        exit_diffs = full_match_stats.get("exit_diffs")
        if exit_diffs is not None and len(exit_diffs) > 0:
            # Convert to numpy array if not already
            if not isinstance(exit_diffs, np.ndarray):
                exit_diffs = np.array(exit_diffs)
            html_reporter.add_distribution_chart(
                exit_diffs,
                "Exit Difference Distribution (Close Cost Diff, Live - Backtest) - Fully Matched Trades Only",
                is_percentage=False,
            )
        
        pl_diffs = full_match_stats.get("pl_diffs")
        if pl_diffs is not None and len(pl_diffs) > 0:
            # Convert to numpy array if not already
            if not isinstance(pl_diffs, np.ndarray):
                pl_diffs = np.array(pl_diffs)
            html_reporter.add_distribution_chart(
                pl_diffs,
                "P/L Difference Distribution (Live - Backtest, per contract) - Fully Matched Trades Only",
                is_percentage=False,
            )

    # Allocation analysis
    allocation = results.get("allocation_analysis", {})
    if allocation and allocation.get("mean_backtest_allocation", 0) > 0:
        starting_portfolio = allocation.get("starting_portfolio_size", 0)
        mean_backtest_pct = allocation.get("mean_backtest_allocation", 0)
        mean_live_pct = allocation.get("mean_live_allocation", 0)
        mean_backtest_dollar = starting_portfolio * (mean_backtest_pct / 100)
        mean_live_dollar = starting_portfolio * (mean_live_pct / 100)
        
        alloc_stats = [
            ("Mean Backtest Allocation", f"{mean_backtest_pct:.2f}% (${mean_backtest_dollar:,.2f})", "value-info"),
            ("Std Dev Backtest Allocation", f"{allocation.get('std_backtest_allocation', 0):.2f}%", "value-warning"),
            ("Mean Live Allocation", f"{mean_live_pct:.2f}% (${mean_live_dollar:,.2f})", "value-info"),
            ("Std Dev Live Allocation", f"{allocation.get('std_live_allocation', 0):.2f}%", "value-warning"),
            ("Starting Portfolio Size", f"${starting_portfolio:,.2f}", "value-default"),
            ("Deviant Trades (>2 std dev from respective mean)", f"{allocation.get('num_deviant_trades', 0)}", "value-negative" if allocation.get('num_deviant_trades', 0) > 0 else "value-positive"),
        ]
        html_reporter.add_statistics_box("Allocation Consistency Analysis", alloc_stats)

    # Matched trades table with special color coding
    matched_table = results.get("matched_trades_table", [])
    if matched_table:
        df = pd.DataFrame(matched_table)
        
        # Filter to only columns shown in stdout and rename them
        stdout_columns = {
            "date": "Date",
            "time_bt": "Time",
            "premium_bt": "Premium BT",
            "premium_live": "Premium Live",
            "premium_diff": "Premium Diff",
            "closing_cost_bt": "Close Cost BT",
            "closing_cost_live": "Close Cost Live",
            "closing_cost_diff": "Close Cost Diff",
            "pl_bt": "P/L BT",
            "pl_live": "P/L Live",
            "pl_diff": "P/L Diff",
            "pl_diff_per_contract": "P/L Diff/C",
            "contracts_bt": "Cont BT",
            "contracts_live": "Cont Live",
            "margin_per_contract": "Margin/Cont",
            "alloc_bt": "Alloc BT",
            "alloc_live": "Alloc Live",
        }
        
        # Select only columns that exist and are in stdout_columns
        available_columns = {k: v for k, v in stdout_columns.items() if k in df.columns}
        df = df[list(available_columns.keys())].copy()
        df = df.rename(columns=available_columns)
        
        # Create a copy for styling and convert numeric columns to object type for HTML strings
        styled_df = df.copy()
        
        # Convert columns that will contain HTML to object type
        html_columns = ["Premium Diff", "Close Cost Diff", "P/L Diff", "P/L Diff/C", "P/L BT", "P/L Live"]
        for col in html_columns:
            if col in styled_df.columns:
                styled_df[col] = styled_df[col].astype('object')
        
        # Get original column names for lookup
        reverse_mapping = {v: k for k, v in available_columns.items()}
        
        # Apply trade-type-specific color logic
        for idx in df.index:
            # Get original row data for trade type lookup
            original_idx = matched_table[idx] if isinstance(matched_table, list) else matched_table.iloc[idx]
            trade_type = original_idx.get("trade_type", "credit") if isinstance(original_idx, dict) else "credit"
            
            # Get values using display column names
            premium_diff = df.at[idx, "Premium Diff"] if "Premium Diff" in df.columns else 0
            closing_cost_diff = df.at[idx, "Close Cost Diff"] if "Close Cost Diff" in df.columns else 0
            pl_diff = df.at[idx, "P/L Diff"] if "P/L Diff" in df.columns else 0
            pl_diff_per_contract = df.at[idx, "P/L Diff/C"] if "P/L Diff/C" in df.columns else 0
            pl_bt = df.at[idx, "P/L BT"] if "P/L BT" in df.columns else 0
            pl_live = df.at[idx, "P/L Live"] if "P/L Live" in df.columns else 0
            
            # Get match status from original data
            original_row = matched_table[idx] if isinstance(matched_table, list) else matched_table.iloc[idx]
            is_full_match = False
            if isinstance(original_row, dict):
                is_full_match = original_row.get("legs_match", False) and original_row.get("reason_match", False)
            
            # Premium diff: For credit trades, positive is good. For debit trades, negative is good.
            if trade_type == "credit":
                premium_color = "value-positive" if premium_diff >= 0 else "value-negative"
            else:  # debit
                premium_color = "value-positive" if premium_diff <= 0 else "value-negative"
            
            # Closing cost diff: For credit trades, negative is good. For debit trades, positive is good.
            if trade_type == "credit":
                closing_color = "value-positive" if closing_cost_diff <= 0 else "value-negative"
            else:  # debit
                closing_color = "value-positive" if closing_cost_diff >= 0 else "value-negative"
            
            # Apply colors to styled dataframe
            if pd.notna(premium_diff) and "Premium Diff" in styled_df.columns:
                styled_df.at[idx, "Premium Diff"] = f'<span class="{premium_color}">${premium_diff:,.2f}</span>'
            if pd.notna(closing_cost_diff) and "Close Cost Diff" in styled_df.columns:
                styled_df.at[idx, "Close Cost Diff"] = f'<span class="{closing_color}">${closing_cost_diff:,.2f}</span>'
            if pd.notna(pl_diff) and "P/L Diff" in styled_df.columns:
                pl_color = "value-positive" if pl_diff >= 0 else "value-negative"
                styled_df.at[idx, "P/L Diff"] = f'<span class="{pl_color}">${pl_diff:,.2f}</span>'
            if pd.notna(pl_diff_per_contract) and "P/L Diff/C" in styled_df.columns:
                pl_pc_color = "value-positive" if pl_diff_per_contract >= 0 else "value-negative"
                styled_df.at[idx, "P/L Diff/C"] = f'<span class="{pl_pc_color}">${pl_diff_per_contract:,.2f}</span>'
            if pd.notna(pl_bt) and "P/L BT" in styled_df.columns:
                pl_bt_color = "value-positive" if pl_bt >= 0 else "value-negative"
                styled_df.at[idx, "P/L BT"] = f'<span class="{pl_bt_color}">${pl_bt:,.2f}</span>'
            if pd.notna(pl_live) and "P/L Live" in styled_df.columns:
                pl_live_color = "value-positive" if pl_live >= 0 else "value-negative"
                styled_df.at[idx, "P/L Live"] = f'<span class="{pl_live_color}">${pl_live:,.2f}</span>'
            
            # Match indicator - create if doesn't exist
            if "Match" not in styled_df.columns:
                styled_df["Match"] = ""
            match_indicator = "✓" if is_full_match else "~"
            match_color = "value-positive" if is_full_match else "value-warning"
            styled_df.at[idx, "Match"] = f'<span class="{match_color}">{match_indicator}</span>'
        
        # Format other numeric columns that weren't styled
        exclude_cols = ["Premium Diff", "Close Cost Diff", "P/L Diff", "P/L Diff/C", "P/L BT", "P/L Live", "Match"]
        for col in df.columns:
                if col not in exclude_cols:
                    if df[col].dtype in [np.float64, np.float32]:
                        # Convert to object type to allow string formatting
                        if styled_df[col].dtype != 'object':
                            styled_df[col] = styled_df[col].astype('object')
                        for idx in df.index:
                            val = df.at[idx, col]
                            if pd.notna(val):
                                # Check if already formatted as HTML
                                current_val = styled_df.at[idx, col]
                                if not isinstance(current_val, str) or '<span' not in str(current_val):
                                    styled_df.at[idx, col] = f"${val:,.2f}"
                    elif df[col].dtype in [np.int64, np.int32]:
                        # Convert to object type to allow string formatting
                        if styled_df[col].dtype != 'object':
                            styled_df[col] = styled_df[col].astype('object')
                        for idx in df.index:
                            val = df.at[idx, col]
                            if pd.notna(val):
                                current_val = styled_df.at[idx, col]
                                if not isinstance(current_val, str) or '<span' not in str(current_val):
                                    styled_df.at[idx, col] = f"{val:,}"
        
        html_reporter.add_table(styled_df, "Matched Trades Comparison")

    # Missed trades
    missed_trades = results.get("missed_trades")
    if missed_trades is not None and len(missed_trades) > 0:
        missed_stats = [
            ("Number of Missed Trades", f"{len(missed_trades)}", "value-warning"),
            ("Total P/L of Missed Trades", f"${missed_trades['P/L'].sum():,.2f}", "value-info"),
        ]
        html_reporter.add_statistics_box("Missed Trades", missed_stats)
        
        if len(missed_trades) <= 20:
            # Filter to only show Date, Time, P/L, Strategy columns (as shown in stdout)
            missed_df = pd.DataFrame(missed_trades)
            stdout_cols = ["Date", "Time", "P/L", "Strategy"]
            available_cols = [col for col in stdout_cols if col in missed_df.columns]
            if available_cols:
                missed_df = missed_df[available_cols]
            color_cols = {"P/L": "positive_negative"} if "P/L" in missed_df.columns else None
            html_reporter.add_table(missed_df, "Missed Trade Details", color_cols)

    # Over trades
    over_trades = results.get("over_trades")
    if over_trades is not None and len(over_trades) > 0:
        over_stats = [
            ("Number of Over Trades", f"{len(over_trades)}", "value-warning"),
            ("Total P/L of Over Trades", f"${over_trades['P/L'].sum():,.2f}", "value-info"),
        ]
        html_reporter.add_statistics_box("Over Trades", over_stats)
        
        if len(over_trades) <= 20:
            # Filter to only show Date, Time, P/L, Strategy columns (as shown in stdout)
            over_df = pd.DataFrame(over_trades)
            stdout_cols = ["Date", "Time", "P/L", "Strategy"]
            available_cols = [col for col in stdout_cols if col in over_df.columns]
            if available_cols:
                over_df = over_df[available_cols]
            color_cols = {"P/L": "positive_negative"} if "P/L" in over_df.columns else None
            html_reporter.add_table(over_df, "Over Trade Details", color_cols)

    # P/L Breakdown
    pl_breakdown = results.get("pl_breakdown", {})
    if pl_breakdown:
        breakdown_lines = []
        
        def add_breakdown_line(category: str, key: str, description: str):
            data = pl_breakdown.get(key, {})
            val = data.get("value", 0)
            pct = data.get("percentage", 0)
            count = data.get("count", 0)
            avg = data.get("average", 0)
            if abs(val) > 0.01:
                breakdown_lines.append((
                    f"${val:,.2f} ({pct:.1f}%) diff was due to {description}",
                    f"N={count}, avg=${avg:,.2f}" if count > 0 else "",
                    "value-positive" if val >= 0 else "value-negative"
                ))
        
        add_breakdown_line("over_trading", "over_trading", "over trading")
        add_breakdown_line("missed_trades", "missed_trades", "missed trades")
        add_breakdown_line("entry_slippage", "entry_slippage", "entry slippage")
        add_breakdown_line("exit_slippage", "exit_slippage", "exit slippage")
        add_breakdown_line("different_outcome", "different_outcome", "different outcome")
        add_breakdown_line("under_allocation", "under_allocation", "under-allocation")
        add_breakdown_line("over_allocation", "over_allocation", "over-allocation")
        
        if breakdown_lines:
            breakdown_stats = [(line[0] + (f" {line[1]}" if line[1] else ""), "", line[2]) for line in breakdown_lines]
            html_reporter.add_statistics_box("P/L Difference Breakdown", breakdown_stats)


def _add_drawdown_html(html_reporter: HTMLReporter, results: dict) -> None:
    """Add drawdown analysis test content to HTML report."""
    drawdown_results = results.get("drawdowns", {})
    margin_check = results.get("margin_check", {})

    # Drawdown metrics
    if drawdown_results:
        dd_stats = [
            ("Max Drawdown", f"${drawdown_results.get('max_drawdown_dollars', 0):,.2f}", "value-negative"),
            ("Max Drawdown (%)", f"{drawdown_results.get('max_drawdown_pct', 0):.2f}%", "value-negative"),
            ("Number of Drawdowns", f"{drawdown_results.get('num_drawdowns', 0)}", "value-info"),
            ("Average Drawdown Length", f"{drawdown_results.get('average_drawdown_length', 0):.1f} days", "value-default"),
            ("Percent Time in Drawdown", f"{drawdown_results.get('percent_time_in_drawdown', 0):.2f}%", "value-warning"),
        ]
        html_reporter.add_statistics_box("Drawdown Metrics", dd_stats)

    # Margin check
    if margin_check.get("has_margin_data", False):
        margin_stats = [
            ("Mean Margin Allocation (1-lot)", f"{margin_check.get('mean_allocation_pct', 0):.2f}%", "value-info"),
            ("Max Margin Allocation (1-lot)", f"{margin_check.get('max_allocation_pct', 0):.2f}%", "value-warning" if margin_check.get('max_allocation_pct', 0) > 2 else "value-positive"),
        ]
        html_reporter.add_statistics_box("Margin Check", margin_stats)

    # Biggest loss
    if margin_check.get("biggest_loss_per_contract", 0) != 0:
        loss_stats = [
            ("Biggest Loss per Contract", f"${margin_check.get('biggest_loss_per_contract', 0):,.2f}", "value-negative"),
            ("As % of Portfolio", f"{margin_check.get('biggest_loss_pct_of_portfolio', 0):.2f}%", "value-negative"),
        ]
        html_reporter.add_statistics_box("Biggest Loss", loss_stats)
    
    # Calendar visualizations
    calendars = results.get("calendars", {})
    if calendars:
        _add_calendar_html(html_reporter, calendars)


def _add_tail_overfitting_html(html_reporter: HTMLReporter, results: dict) -> None:
    """Add tail overfitting test content to HTML report."""
    overfitting_score = results.get("overfitting_score")
    if overfitting_score is not None:
        max_score = results.get("max_score", 12.0)
        stats = [
            ("Overfitting Score", f"{overfitting_score:.2f}", "value-negative" if overfitting_score > max_score else "value-positive"),
            ("Maximum Acceptable Score", f"{max_score:.2f}", "value-default"),
            ("Tail Percentage", f"{results.get('tail_percentage', 0):.2f}%", "value-info"),
            ("Tail Count", f"{results.get('tail_count', 0)}", "value-info"),
        ]
        html_reporter.add_statistics_box("Overfitting Analysis", stats)

        # Baseline metrics
        baseline = results.get("baseline_metrics", {})
        if baseline:
            baseline_stats = [
                ("Baseline Mean P/L", f"${baseline.get('mean_pl', 0):,.2f}", "value-positive" if baseline.get('mean_pl', 0) >= 0 else "value-negative"),
                ("Baseline Win Rate", f"{baseline.get('win_rate', 0)*100:.2f}%", "value-default"),
            ]
            html_reporter.add_statistics_box("Baseline Performance (Without Tail Events)", baseline_stats)


def _add_track_html(html_reporter: HTMLReporter, results: dict) -> None:
    """Add track test content to HTML report."""
    portfolio_metrics = results.get("portfolio_metrics", {})
    drawdown_analysis = results.get("drawdown_analysis", {})

    # Portfolio metrics - include ALL metrics that are printed
    if portfolio_metrics:
        stats = []
        
        # Net P/L
        net_pl = portfolio_metrics.get('net_pl', 0)
        stats.append(("Net P/L", f"${net_pl:,.2f}", "value-positive" if net_pl >= 0 else "value-negative"))
        
        # Extra Fees (if applicable)
        extra_fees_monthly = portfolio_metrics.get("extra_fees_monthly", 0.0)
        total_fees = portfolio_metrics.get("total_fees", 0.0)
        if extra_fees_monthly > 0:
            stats.append(("Extra Fees (Monthly)", f"${extra_fees_monthly:,.2f}", "value-warning"))
            stats.append(("Total Fees", f"${total_fees:,.2f}", "value-warning"))
            net_pl_after_fees = portfolio_metrics.get("net_pl_after_fees", 0.0)
            stats.append(("Net P/L After Fees", f"${net_pl_after_fees:,.2f}", "value-positive" if net_pl_after_fees >= 0 else "value-negative"))
        
        # CAGR
        cagr = portfolio_metrics.get('cagr', 0)
        stats.append(("CAGR", f"{cagr:.2f}%", "value-positive" if cagr >= 0 else "value-negative"))
        
        # Max Drawdown (from drawdown_analysis)
        max_dd_pct = drawdown_analysis.get("max_drawdown_pct", 0.0)
        stats.append(("Max Drawdown", f"{max_dd_pct:.2f}%", "value-negative"))
        
        # MAR (CAGR / Max Drawdown)
        mar = 0.0
        if max_dd_pct > 0:
            mar = cagr / max_dd_pct
        stats.append(("MAR (CAGR / Max DD)", f"{mar:.2f}", "value-positive" if mar >= 0 else "value-negative"))
        
        # Sharpe Ratio
        sharpe = portfolio_metrics.get('sharpe', 0)
        stats.append(("Sharpe Ratio", f"{sharpe:.2f}", "value-positive" if sharpe > 1 else "value-default"))
        
        # Sortino Ratio
        sortino = portfolio_metrics.get('sortino', 0)
        stats.append(("Sortino Ratio", f"{sortino:.2f}", "value-positive" if sortino > 1 else "value-default"))
        
        # Total Premium
        total_premium = portfolio_metrics.get('total_premium', 0)
        stats.append(("Total Premium", f"${total_premium:,.2f}", "value-default"))
        
        # PCR (Premium Capture Rate)
        pcr = portfolio_metrics.get('pcr', 0)
        stats.append(("PCR (Premium Capture Rate)", f"{pcr:.2f}%", "value-positive" if pcr >= 0 else "value-negative"))
        
        html_reporter.add_statistics_box("Portfolio Metrics", stats)

    # Drawdown analysis - include ALL fields that are printed
    if drawdown_analysis:
        dd_stats = []
        
        # Max Drawdown (with dollars and percentage)
        max_dd_dollars = drawdown_analysis.get("max_drawdown_dollars", 0.0)
        max_dd_pct = drawdown_analysis.get("max_drawdown_pct", 0.0)
        dd_stats.append(("Max Drawdown", f"${max_dd_dollars:,.2f} ({max_dd_pct:.2f}%)", "value-negative"))
        
        # Longest Drawdown
        longest_dd = drawdown_analysis.get("longest_drawdown")
        if longest_dd:
            length_days = longest_dd.get("length_days", 0)
            depth_pct = longest_dd.get("depth_pct", 0.0)
            start_date = longest_dd.get("start_date")
            end_date = longest_dd.get("end_date")
            if start_date and end_date:
                date_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            else:
                date_str = "N/A"
            dd_stats.append(("Longest Drawdown", f"{length_days} days, {depth_pct:.2f}% depth ({date_str})", "value-warning"))
        
        # Shortest Drawdown
        shortest_dd = drawdown_analysis.get("shortest_drawdown")
        if shortest_dd:
            length_days = shortest_dd.get("length_days", 0)
            depth_pct = shortest_dd.get("depth_pct", 0.0)
            start_date = shortest_dd.get("start_date")
            end_date = shortest_dd.get("end_date")
            if start_date and end_date:
                date_str = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            else:
                date_str = "N/A"
            dd_stats.append(("Shortest Drawdown", f"{length_days} days, {depth_pct:.2f}% depth ({date_str})", "value-warning"))
        
        # Average Drawdown Length
        avg_length = drawdown_analysis.get("average_drawdown_length", 0.0)
        dd_stats.append(("Average Drawdown Length", f"{avg_length:.1f} days", "value-default"))
        
        # Average Drawdown Depth
        avg_depth = drawdown_analysis.get("average_drawdown_depth", 0.0)
        dd_stats.append(("Average Drawdown Depth", f"{avg_depth:.2f}%", "value-default"))
        
        # Number of Drawdowns
        num_drawdowns = drawdown_analysis.get("num_drawdowns", 0)
        dd_stats.append(("Number of Drawdowns", f"{num_drawdowns}", "value-info"))
        
        # Current Drawdown
        current_dd = drawdown_analysis.get("current_drawdown")
        if current_dd:
            length_days = current_dd.get("length_days", 0)
            depth_pct = current_dd.get("depth_pct", 0.0)
            start_date = current_dd.get("start_date")
            if start_date:
                date_str = f"since {start_date.strftime('%Y-%m-%d')}"
            else:
                date_str = ""
            dd_stats.append(("Current Drawdown", f"{length_days} days, {depth_pct:.2f}% depth {date_str}", "value-negative"))
        else:
            dd_stats.append(("Current Drawdown", "None (at peak)", "value-positive"))
        
        html_reporter.add_statistics_box("Drawdown Analysis", dd_stats)

    # Strategy statistics table
    strategy_stats = results.get("strategy_stats", [])
    if strategy_stats:
        df = pd.DataFrame(strategy_stats)
        html_reporter.add_table(df, "Strategy Statistics")
    
    # Calendar visualizations
    calendars = results.get("calendars", {})
    if calendars:
        _add_calendar_html(html_reporter, calendars)


def _add_portfolio_stress_html(html_reporter: HTMLReporter, results: dict) -> None:
    """Add portfolio stress test content to HTML report."""
    aggregated = results.get("aggregated_metrics", {})
    
    if not aggregated:
        return
    
    # Add percentile curves chart if available
    percentile_figure = results.get("percentile_figure")
    if percentile_figure is not None:
        html_reporter.add_chart(percentile_figure, "Percentile Equity and Drawdown Curves")
    
    # Worst-case scenario
    worst_stats = []
    worst_stats.append(("Worst Max Drawdown", f"${aggregated.get('worst_max_drawdown_dollars', 0):,.2f}", "value-negative"))
    worst_stats.append(("Worst Max Drawdown (%)", f"{aggregated.get('worst_max_drawdown_pct', 0):.2f}%", "value-negative"))
    worst_stats.append(("Worst Final Portfolio", f"${aggregated.get('worst_final_portfolio', 0):,.2f}", "value-negative"))
    worst_stats.append(("Worst Total Return", f"{aggregated.get('worst_total_return', 0):.2f}%", "value-negative"))
    worst_stats.append(("Worst CAGR", f"{aggregated.get('worst_cagr', 0):.2f}%", "value-negative"))
    worst_stats.append(("Worst Sharpe Ratio", f"{aggregated.get('worst_sharpe', 0):.2f}", "value-negative"))
    worst_stats.append(("Worst Longest Drawdown", f"{aggregated.get('worst_longest_drawdown_days', 0)} days", "value-negative"))
    html_reporter.add_statistics_box("Worst-Case Scenario", worst_stats)
    
    # Distribution statistics
    dist_stats = []
    dist_stats.append(("Mean Max Drawdown", f"${aggregated.get('mean_max_drawdown_dollars', 0):,.2f}", "value-default"))
    dist_stats.append(("Mean Max Drawdown (%)", f"{aggregated.get('mean_max_drawdown_pct', 0):.2f}%", "value-default"))
    dist_stats.append(("Mean Final Portfolio", f"${aggregated.get('mean_final_portfolio', 0):,.2f}", "value-positive"))
    dist_stats.append(("Mean Total Return", f"{aggregated.get('mean_total_return', 0):.2f}%", "value-positive"))
    dist_stats.append(("Mean CAGR", f"{aggregated.get('mean_cagr', 0):.2f}%", "value-positive"))
    dist_stats.append(("Std Dev Max Drawdown", f"${aggregated.get('std_max_drawdown_dollars', 0):,.2f}", "value-info"))
    dist_stats.append(("Std Dev Final Portfolio", f"${aggregated.get('std_final_portfolio', 0):,.2f}", "value-info"))
    html_reporter.add_statistics_box("Distribution Statistics", dist_stats)
    
    # Allocation and skipped trades statistics
    alloc_stats = []
    total_trades_all = aggregated.get("total_trades_all_simulations", 0)
    mean_trades_per_sim = aggregated.get("mean_trades_per_sim", 0.0)
    alloc_stats.append(("Total Trades (All Simulations)", f"{total_trades_all:,}", "value-info"))
    alloc_stats.append(("Mean Trades per Simulation", f"{mean_trades_per_sim:.2f}", "value-info"))
    
    total_skipped = aggregated.get("total_skipped_trades", 0)
    mean_skipped = aggregated.get("mean_skipped_trades_per_sim", 0.0)
    max_skipped = aggregated.get("max_skipped_trades_per_sim", 0)
    
    if total_skipped > 0:
        alloc_stats.append(("Total Skipped Trades", f"{total_skipped:,}", "value-warning"))
        alloc_stats.append(("Mean Skipped per Simulation", f"{mean_skipped:.2f}", "value-warning"))
        alloc_stats.append(("Max Skipped in Single Simulation", f"{max_skipped}", "value-negative"))
        if total_trades_all > 0:
            skip_rate = (total_skipped / total_trades_all) * 100
            alloc_stats.append(("Skip Rate", f"{skip_rate:.2f}%", "value-warning"))
    else:
        alloc_stats.append(("Skipped Trades", "None (all trades executed)", "value-positive"))
    
    mean_actual = aggregated.get("mean_actual_allocation", 0.0)
    mean_target = aggregated.get("mean_target_allocation", 0.0)
    std_actual = aggregated.get("std_actual_allocation", 0.0)
    
    if mean_actual > 0:
        alloc_stats.append(("Mean Target Allocation", f"{mean_target:.2f}%", "value-info"))
        alloc_stats.append(("Mean Actual Allocation", f"{mean_actual:.2f}%", "value-positive"))
        if mean_target > 0:
            allocation_diff = mean_actual - mean_target
            diff_class = "value-warning" if abs(allocation_diff) > 0.1 else "value-positive"
            alloc_stats.append(("Allocation Difference", f"{allocation_diff:+.2f}%", diff_class))
        alloc_stats.append(("Std Dev Actual Allocation", f"{std_actual:.2f}%", "value-info"))
    
    html_reporter.add_statistics_box("Allocation & Skipped Trades Statistics", alloc_stats)
    
    # Max Drawdown Percentiles
    dd_pct_percentiles = aggregated.get("max_drawdown_pct_percentiles", {})
    if dd_pct_percentiles:
        dd_percentile_stats = []
        dd_percentile_stats.append(("5th Percentile", f"{dd_pct_percentiles.get('p5', 0):.2f}%", "value-info"))
        dd_percentile_stats.append(("25th Percentile", f"{dd_pct_percentiles.get('p25', 0):.2f}%", "value-default"))
        dd_percentile_stats.append(("50th Percentile (Median)", f"{dd_pct_percentiles.get('p50', 0):.2f}%", "value-default"))
        dd_percentile_stats.append(("75th Percentile", f"{dd_pct_percentiles.get('p75', 0):.2f}%", "value-warning"))
        dd_percentile_stats.append(("95th Percentile", f"{dd_pct_percentiles.get('p95', 0):.2f}%", "value-negative"))
        html_reporter.add_statistics_box("Max Drawdown Percentiles (%)", dd_percentile_stats)
    
    # Final Portfolio Percentiles
    fp_percentiles = aggregated.get("final_portfolio_percentiles", {})
    if fp_percentiles:
        fp_percentile_stats = []
        fp_percentile_stats.append(("5th Percentile", f"${fp_percentiles.get('p5', 0):,.2f}", "value-negative"))
        fp_percentile_stats.append(("25th Percentile", f"${fp_percentiles.get('p25', 0):,.2f}", "value-warning"))
        fp_percentile_stats.append(("50th Percentile (Median)", f"${fp_percentiles.get('p50', 0):,.2f}", "value-default"))
        fp_percentile_stats.append(("75th Percentile", f"${fp_percentiles.get('p75', 0):,.2f}", "value-positive"))
        fp_percentile_stats.append(("95th Percentile", f"${fp_percentiles.get('p95', 0):,.2f}", "value-positive"))
        html_reporter.add_statistics_box("Final Portfolio Value Percentiles", fp_percentile_stats)
    
    # Total Return Percentiles
    tr_percentiles = aggregated.get("total_return_percentiles", {})
    if tr_percentiles:
        tr_percentile_stats = []
        tr_percentile_stats.append(("5th Percentile", f"{tr_percentiles.get('p5', 0):.2f}%", "value-negative"))
        tr_percentile_stats.append(("25th Percentile", f"{tr_percentiles.get('p25', 0):.2f}%", "value-warning"))
        tr_percentile_stats.append(("50th Percentile (Median)", f"{tr_percentiles.get('p50', 0):.2f}%", "value-default"))
        tr_percentile_stats.append(("75th Percentile", f"{tr_percentiles.get('p75', 0):.2f}%", "value-positive"))
        tr_percentile_stats.append(("95th Percentile", f"{tr_percentiles.get('p95', 0):.2f}%", "value-positive"))
        html_reporter.add_statistics_box("Total Return Percentiles (%)", tr_percentile_stats)


def _add_portfolio_correlation_html(html_reporter: HTMLReporter, results: dict) -> None:
    """Add portfolio correlation test content to HTML report."""
    correlation_matrices = results.get("correlation_matrices", {})
    statistical_metrics = results.get("statistical_metrics", {})
    figures = results.get("figures", {})
    
    # Add correlation heatmaps
    heatmaps = figures.get("heatmaps", {})
    for matrix_key, figure in heatmaps.items():
        if figure is not None:
            title = matrix_key.replace("_", " ").title()
            html_reporter.add_chart(figure, f"Correlation Heatmap - {title}")
    
    # Add scatter plot matrix
    scatter_fig = figures.get("scatter_matrix")
    if scatter_fig is not None:
        html_reporter.add_chart(scatter_fig, "Strategy Returns Scatter Plot Matrix")
    
    # Add time series overlay
    time_series_fig = figures.get("time_series")
    if time_series_fig is not None:
        html_reporter.add_chart(time_series_fig, "Strategy Equity Curves Overlay")
    
    # Add rolling correlation plots
    rolling_fig = figures.get("rolling_correlations")
    if rolling_fig is not None:
        html_reporter.add_chart(rolling_fig, "Rolling Correlations Over Time")
    
    # Add correlation matrices as tables
    for matrix_key, title in [
        ("pearson_returns", "Pearson Correlation - Returns"),
        ("spearman_returns", "Spearman Correlation - Returns"),
        ("pearson_cumulative", "Pearson Correlation - Cumulative Returns"),
        ("spearman_cumulative", "Spearman Correlation - Cumulative Returns"),
    ]:
        if matrix_key in correlation_matrices:
            corr_matrix = correlation_matrices[matrix_key]
            html_reporter.add_table(corr_matrix, title)
            
            # Add p-values if available
            pvalue_key = f"{matrix_key}_pvalues"
            if pvalue_key in correlation_matrices:
                pvalues = correlation_matrices[pvalue_key]
                html_reporter.add_table(pvalues, f"{title} - P-Values")
    
    # Add statistical summary
    for matrix_name in [
        "pearson_returns",
        "spearman_returns",
        "pearson_cumulative",
        "spearman_cumulative",
    ]:
        mean_key = f"{matrix_name}_mean"
        if mean_key not in statistical_metrics:
            continue
        
        stats = []
        stats.append(("Mean Correlation", f"{statistical_metrics[mean_key]:.4f}", "value-info"))
        std_key = f"{matrix_name}_std"
        if std_key in statistical_metrics:
            stats.append(("Std Deviation", f"{statistical_metrics[std_key]:.4f}", "value-info"))
        min_key = f"{matrix_name}_min"
        if min_key in statistical_metrics:
            stats.append(("Min Correlation", f"{statistical_metrics[min_key]:.4f}", "value-warning"))
        max_key = f"{matrix_name}_max"
        if max_key in statistical_metrics:
            stats.append(("Max Correlation", f"{statistical_metrics[max_key]:.4f}", "value-positive"))
        median_key = f"{matrix_name}_median"
        if median_key in statistical_metrics:
            stats.append(("Median Correlation", f"{statistical_metrics[median_key]:.4f}", "value-info"))
        
        max_pair = statistical_metrics.get(f"{matrix_name}_max_pair")
        max_value = statistical_metrics.get(f"{matrix_name}_max_value")
        if max_pair and max_value is not None:
            stats.append(("Highest Correlation Pair", f"{max_pair[0]} vs {max_pair[1]} ({max_value:.4f})", "value-positive"))
        
        min_pair = statistical_metrics.get(f"{matrix_name}_min_pair")
        min_value = statistical_metrics.get(f"{matrix_name}_min_value")
        if min_pair and min_value is not None:
            stats.append(("Lowest Correlation Pair", f"{min_pair[0]} vs {min_pair[1]} ({min_value:.4f})", "value-warning"))
        
        title = matrix_name.replace("_", " ").title() + " - Summary"
        html_reporter.add_statistics_box(title, stats)


def _add_calendar_html(html_reporter: HTMLReporter, calendars: dict) -> None:
    """Add calendar visualizations to HTML report."""
    weekly = calendars.get("weekly", {})
    monthly = calendars.get("monthly", {})
    
    if not weekly and not monthly:
        return
    
    # Weekly calendar
    if weekly:
        weekly_html = _generate_calendar_html(weekly, "week")
        html_reporter.add_section("Weekly Profitability Calendar", weekly_html, "box")
    
    # Monthly calendar
    if monthly:
        monthly_html = _generate_calendar_html(monthly, "month")
        html_reporter.add_section("Monthly Profitability Calendar", monthly_html, "box")


def _generate_calendar_html(period_data: dict, period_type: str) -> str:
    """
    Generate HTML for calendar grid visualization.
    
    Args:
        period_data: Dictionary mapping period strings to {"pl": float, "profitable": bool}
        period_type: "week" or "month"
    
    Returns:
        HTML string for calendar grid
    """
    if not period_data:
        return "<p>No calendar data available.</p>"
    
    # Sort periods chronologically
    sorted_periods = sorted(period_data.items())
    
    # Group into rows (7 columns for weeks, 4 columns for months)
    cols_per_row = 7 if period_type == "week" else 4
    rows = []
    for i in range(0, len(sorted_periods), cols_per_row):
        rows.append(sorted_periods[i:i + cols_per_row])
    
    # Generate HTML
    html_parts = []
    html_parts.append('<div class="calendar-grid">')
    html_parts.append(f'<p style="margin-bottom: 15px; color: #6b7280;">{"Green = profitable " + period_type + ", Red = losing " + period_type}</p>')
    
    for row in rows:
        html_parts.append('<div class="calendar-row">')
        for period_str, data in row:
            # Format period string
            if period_type == "week":
                if "W" in period_str and len(period_str) >= 8:
                    period_display = period_str[:8]
                else:
                    period_display = period_str[:10] if len(period_str) >= 10 else period_str
            else:
                period_display = period_str[:7] if len(period_str) >= 7 else period_str
            
            # Color based on profitability
            color_class = "value-positive" if data["profitable"] else "value-negative"
            pl_value = data.get("pl", 0)
            pl_str = f"${pl_value:,.2f}" if abs(pl_value) >= 1 else f"${pl_value:,.2f}"
            
            html_parts.append(
                f'<div class="calendar-cell {color_class}" title="{period_display}: {pl_str}">'
                f'<div class="calendar-symbol">█</div>'
                f'<div class="calendar-label">{period_display}</div>'
                f'<div class="calendar-pl">{pl_str}</div>'
                f'</div>'
            )
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)


def _process_strategy(
    data,
    strategy_name: str,
    test: BaseTest,
    test_kwargs: dict,
    output_dir: Optional[str],
    html_report: bool = False,
) -> None:
    """
    Process a single strategy's data.

    Args:
        data: DataFrame with strategy data
        strategy_name: Name of the strategy
        test: Test instance to run
        test_kwargs: Additional arguments for the test
        output_dir: Directory to save visualizations
    """
    # Print strategy header
    box_width = 80
    header_text = "STRATEGY ANALYSIS"
    header_spaces = box_width - visible_length(header_text) - 4
    strategy_spaces = box_width - visible_length(strategy_name) - 4

    print(f"\n{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}")
    print(
        f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{header_text}{Colors.RESET} {' ' * header_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
    )
    print(
        f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{strategy_name}{Colors.RESET} {' ' * strategy_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
    )
    print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}")

    # Initialize HTML reporter if requested
    html_reporter = None
    if html_report:
        # Find logo path - try multiple locations
        logo_path = None
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "images", "logo.png"),
            "images/logo.png",
            os.path.join(os.getcwd(), "images", "logo.png"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logo_path = path
                break
        
        if logo_path:
            html_reporter = HTMLReporter(logo_path)
            html_reporter.start_report(test.get_name(), strategy_name, test_kwargs)
        else:
            print(f"{Colors.BRIGHT_YELLOW}Warning: Logo not found, HTML report will be generated without logo{Colors.RESET}")
            html_reporter = HTMLReporter("")
            html_reporter.start_report(test.get_name(), strategy_name, test_kwargs)

    # Run test
    results = test.run(data, verbose=True, **test_kwargs)

    # Create histograms if matplotlib available
    histogram_figure = None
    if "P/L per Contract" in data.columns:
        pnl_per_contract = data["P/L per Contract"].values
        pnl_per_contract = pnl_per_contract[
            ~(np.isnan(pnl_per_contract) | np.isinf(pnl_per_contract))
        ]

        if len(pnl_per_contract) > 0:
            histogram_figure = create_histograms(
                pnl_per_contract,
                results.get("mc_means"),
                strategy_name,
                output_dir,
                {
                    "observed_mean": results.get("observed_mean", 0),
                    "observed_std": results.get("observed_std", 0),
                    "current_n": results.get("current_n", 0),
                    "win_rate": results.get("win_rate", 0),
                    "current_power": results.get("current_power", 0.0),
                    "recommended_n": results.get("recommended_n"),
                },
                return_figure=html_report,
            )

    # Generate HTML report if requested
    if html_reporter:
        # Capture drawdown chart if this is a drawdown test
        chart_figure = histogram_figure
        if test.get_name() == "drawdown":
            drawdown_results = results.get("drawdowns", {})
            if drawdown_results.get("equity_curve") and drawdown_results.get("drawdown_curve") and drawdown_results.get("dates"):
                from ..output.visualizers import create_drawdown_chart
                chart_figure = create_drawdown_chart(
                    drawdown_results["equity_curve"],
                    drawdown_results["drawdown_curve"],
                    drawdown_results["dates"],
                    strategy_name,
                    output_dir,
                    return_figure=True,
                )
        
        _generate_html_report(html_reporter, test, results, data, chart_figure)
        
        report_path = html_reporter.save_report()
        print(f"\n{Colors.BRIGHT_GREEN}HTML report saved to: {report_path}{Colors.RESET}")


def _process_compare_test(
    backtest_file: str,
    live_file: str,
    test: BaseTest,
    window_minutes: int,
    starting_portfolio_size: float,
    source_of_truth: str,
    output_dir: Optional[str],
    html_report: bool = False,
) -> None:
    """
    Process comparison test with two files.

    Args:
        backtest_file: Path to backtest file
        live_file: Path to live file
        test: Test instance (should be LiveBacktestComparisonTest)
        window_minutes: Time window for matching
        starting_portfolio_size: Starting portfolio size
        source_of_truth: Source of truth for date range ("live" or "backtest")
        output_dir: Directory to save visualizations
    """
    # Parse both files
    backtest_parser = _find_parser(backtest_file)
    live_parser = _find_parser(live_file)

    backtest_df = backtest_parser.parse(backtest_file)
    live_df = live_parser.parse(live_file)

    # Normalize both with source information
    backtest_df = NORMALIZER.normalize(backtest_df, source="backtest")
    live_df = NORMALIZER.normalize(live_df, source="live")

    # Enrich both
    backtest_df = ENRICHER.enrich(backtest_df)
    live_df = ENRICHER.enrich(live_df)

    # Print header
    box_width = 80
    header_text = "LIVE VS BACKTEST COMPARISON"
    header_spaces = box_width - visible_length(header_text) - 4

    print(f"\n{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}")
    print(
        f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{header_text}{Colors.RESET} {' ' * header_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
    )
    print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}")

    # Initialize HTML reporter if requested
    html_reporter = None
    if html_report:
        # Find logo path - try multiple locations
        logo_path = None
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "images", "logo.png"),
            "images/logo.png",
            os.path.join(os.getcwd(), "images", "logo.png"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logo_path = path
                break
        
        if logo_path:
            html_reporter = HTMLReporter(logo_path)
        else:
            html_reporter = HTMLReporter("")
        html_reporter.start_report(
            test.get_name(),
            "Live vs Backtest Comparison",
            {
                "window_minutes": window_minutes,
                "starting_portfolio_size": starting_portfolio_size,
                "source_of_truth": source_of_truth,
            }
        )

    # Run test
    results = test.run(
        backtest_data=backtest_df,
        live_data=live_df,
        window_minutes=window_minutes,
        starting_portfolio_size=starting_portfolio_size,
        source_of_truth=source_of_truth,
        verbose=True,
    )

    # Generate HTML report if requested
    if html_reporter:
        _generate_html_report(html_reporter, test, results, None)
        
        report_path = html_reporter.save_report()
        print(f"\n{Colors.BRIGHT_GREEN}HTML report saved to: {report_path}{Colors.RESET}")


def _print_disclaimer():
    """Print disclaimer at the end."""
    print(f"\n{Colors.DIM}{'─' * 80}{Colors.RESET}")
    print(
        f"{Colors.DIM}DISCLAIMER: This tool is provided for educational purposes only.{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}The analysis and results should not be considered as financial advice.{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}Always perform your own due diligence and consult with qualified{Colors.RESET}"
    )
    print(
        f"{Colors.DIM}professionals before making any trading or investment decisions.{Colors.RESET}"
    )
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}\n")


# Create a test group for subcommands
test_app = typer.Typer(name="test", help="Run statistical tests on trading data")

@test_app.command("power")
def test_power(
    file_path: str = typer.Argument(..., help="Path to data file"),
    target_power: float = typer.Option(
        0.95, "--target-power", "-p", help="Target power level (0-1, default: 0.95)"
    ),
    simulations: int = typer.Option(
        10000, "--simulations", "-s", help="Number of Monte Carlo simulations"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save histogram images"
    ),
    html_report: bool = typer.Option(
        False, "--html-report", help="Generate HTML report"
    ),
):
    """
    Estimate the sample size needed for an observed edge to be statistically stable.
    
    Uses bootstrap sampling with replacement to test statistical power.
    Statistical power = probability that mean return > 0 in a random sample.
    
    Examples:
        evtools test power data.csv --target-power 0.95
        evtools test power data.csv --target-power 0.90 --simulations 50000 --output-dir ./output
    """
    # Validate target_power
    if not 0 < target_power <= 1:
        typer.echo("Error: target-power must be between 0 and 1", err=True)
        raise typer.Exit(1)

    # Validate file exists
    if not Path(file_path).exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    # Print header
    _print_header()

    try:
        # Find test
        test = _find_test("power")

        # Prepare test kwargs
        test_kwargs = {
            "target_power": target_power,
            "simulations": simulations,
        }

        # Process file
        _process_file(file_path, test, test_kwargs, output_dir, html_report)

        # Print disclaimer
        _print_disclaimer()

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@test_app.command("compare")
def test_compare(
    backtest_file: str = typer.Argument(..., help="Path to backtest data file"),
    live_file: str = typer.Option(
        ..., "--live-file", "-l", help="Path to live data file (required)"
    ),
    window_minutes: int = typer.Option(
        10, "--window-minutes", "-w", help="Time window in minutes for matching trades (default: 10)"
    ),
    starting_portfolio_size: float = typer.Option(
        100000.0, "--starting-portfolio", "-sp", help="Starting portfolio size for live trading (default: 100000)"
    ),
    source_of_truth: str = typer.Option(
        "live", "--source-of-truth", "-sot", help="Source of truth for date range: 'live' or 'backtest' (default: 'live')"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save histogram images"
    ),
    html_report: bool = typer.Option(
        False, "--html-report", help="Generate HTML report"
    ),
):
    """
    Compare real live trading execution against backtest data for the same period.
    
    Matches trades by time window and strategy, calculates P/L differences, premium differences,
    and identifies missed/over trades. Provides slippage analysis for fully matched trades.
    
    Examples:
        evtools test compare backtest.csv --live-file live.csv --window-minutes 10
        evtools test compare backtest.csv --live-file live.csv --source-of-truth backtest
    """
    # Validate file exists
    if not Path(backtest_file).exists():
        typer.echo(f"Error: Backtest file not found: {backtest_file}", err=True)
        raise typer.Exit(1)
    if not Path(live_file).exists():
        typer.echo(f"Error: Live file not found: {live_file}", err=True)
        raise typer.Exit(1)
    if source_of_truth not in ["live", "backtest"]:
        typer.echo("Error: --source-of-truth must be 'live' or 'backtest'", err=True)
        raise typer.Exit(1)

    # Print header
    _print_header()

    try:
        # Find test
        test = _find_test("compare")

        # Process compare test
        _process_compare_test(backtest_file, live_file, test, window_minutes, starting_portfolio_size, source_of_truth, output_dir, html_report)

        # Print disclaimer
        _print_disclaimer()

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@test_app.command("drawdown")
def test_drawdown(
    file_path: str = typer.Argument(..., help="Path to data file"),
    portfolio_size: float = typer.Option(
        100000.0, "--portfolio-size", "-p", help="Initial portfolio size (default: 100000)"
    ),
    allocation: float = typer.Option(
        1.0, "--allocation", "-a", help="Desired allocation percentage (default: 1.0)"
    ),
    force_one_lot: bool = typer.Option(
        False, "--force-one-lot", "-f", help="Force at least 1 contract even when allocation is insufficient"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save outputs"
    ),
    html_report: bool = typer.Option(
        False, "--html-report", help="Generate HTML report"
    ),
):
    """
    Analyze drawdowns for trading strategies.
    
    Performs margin requirement validation, calculates drawdown metrics,
    and generates calendar visualizations of profitable/losing periods.
    
    Examples:
        evtools test drawdown data.csv --portfolio-size 100000 --allocation 1.5
        evtools test drawdown data.csv -p 50000 -a 2.0
    """
    # Validate file exists
    if not Path(file_path).exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    # Validate allocation
    if allocation <= 0:
        typer.echo("Error: allocation must be greater than 0", err=True)
        raise typer.Exit(1)

    # Print header
    _print_header()

    try:
        # Find test
        test = _find_test("drawdown")

        # Parse file to get metadata
        parser = _find_parser(file_path)
        df = parser.parse(file_path)
        metadata = parser.get_metadata(df)

        # Check if portfolio (multiple strategies)
        is_portfolio = metadata.get("is_portfolio", False)
        strategies = metadata.get("strategies", [])

        strategy_allocations = {}
        if is_portfolio and len(strategies) > 1:
            # Prompt for each strategy allocation
            print(f"\n{Colors.BRIGHT_CYAN}Portfolio detected with {len(strategies)} strategies.{Colors.RESET}")
            print(f"{Colors.DIM}Enter desired allocation % for each strategy (default: {allocation}%):{Colors.RESET}\n")
            
            for strategy in strategies:
                prompt = f"Allocation % for '{strategy}' [{allocation}]: "
                try:
                    user_input = input(prompt).strip()
                    if user_input:
                        strategy_allocations[strategy] = float(user_input)
                    else:
                        strategy_allocations[strategy] = allocation
                except (ValueError, KeyboardInterrupt):
                    typer.echo(f"\nUsing default allocation {allocation}% for '{strategy}'")
                    strategy_allocations[strategy] = allocation
        else:
            # Single strategy, use default allocation
            strategy_allocations = None

        # Prepare test kwargs
        test_kwargs = {
            "portfolio_size": portfolio_size,
            "desired_allocation_pct": allocation,
            "strategy_allocations": strategy_allocations if strategy_allocations else None,
            "force_one_lot": force_one_lot,
            "output_dir": output_dir,
        }

        # Process file
        _process_file(file_path, test, test_kwargs, output_dir)

        # Print disclaimer
        _print_disclaimer()

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@test_app.command("portfolio_stress")
def test_portfolio_stress(
    file_path: str = typer.Argument(..., help="Path to portfolio data file"),
    portfolio_size: float = typer.Option(
        100000.0, "--portfolio-size", "-p", help="Initial portfolio size (default: 100000)"
    ),
    allocation: float = typer.Option(
        1.0, "--allocation", "-a", help="Default allocation percentage (default: 1.0)"
    ),
    simulations: int = typer.Option(
        10000, "--simulations", "-s", help="Number of Monte Carlo simulations (default: 10000)"
    ),
    force_one_lot: bool = typer.Option(
        False, "--force-one-lot", "-f", help="Force at least 1 contract even when allocation is insufficient"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save outputs"
    ),
    html_report: bool = typer.Option(
        False, "--html-report", help="Generate HTML report"
    ),
):
    """
    Perform Monte Carlo stress testing on portfolio backtests.
    
    Generates thousands of simulations by sampling with replacement from each strategy's
    trade distribution, combines them into portfolio equity curves using dynamic allocation,
    and analyzes worst-case scenarios including max drawdown and comprehensive risk metrics.
    
    Examples:
        evtools test portfolio_stress portfolio.csv --portfolio-size 100000 --allocation 1.5 --simulations 10000
        evtools test portfolio_stress portfolio.csv -p 50000 -a 2.0 -s 5000
    """
    # Validate file exists
    if not Path(file_path).exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    # Validate allocation
    if allocation <= 0:
        typer.echo("Error: allocation must be greater than 0", err=True)
        raise typer.Exit(1)

    # Validate simulations
    if simulations <= 0:
        typer.echo("Error: simulations must be greater than 0", err=True)
        raise typer.Exit(1)

    # Print header
    _print_header()

    try:
        # Find test
        test = _find_test("portfolio_stress")

        # Parse file to get metadata
        parser = _find_parser(file_path)
        df = parser.parse(file_path)
        metadata = parser.get_metadata(df)

        # Normalize
        df = NORMALIZER.normalize(df)

        # Enrich
        df = ENRICHER.enrich(df)

        # Check if portfolio (multiple strategies)
        is_portfolio = metadata.get("is_portfolio", False)
        strategies = metadata.get("strategies", [])

        # Calculate average allocations from enriched data if available
        calculated_allocations = {}
        if "Used Allocation" in df.columns:
            for strategy in strategies:
                strategy_df = df[df["Strategy"] == strategy]
                if len(strategy_df) > 0:
                    avg_allocation = strategy_df["Used Allocation"].mean()
                    if not (np.isnan(avg_allocation) or np.isinf(avg_allocation) or avg_allocation <= 0):
                        calculated_allocations[strategy] = float(avg_allocation)

        strategy_allocations = {}
        if is_portfolio and len(strategies) > 1:
            # Prompt for each strategy allocation with calculated defaults
            print(f"\n{Colors.BRIGHT_CYAN}Portfolio detected with {len(strategies)} strategies.{Colors.RESET}")
            if calculated_allocations:
                print(f"{Colors.DIM}Calculated average allocations from historical data. Enter desired allocation % for each strategy:{Colors.RESET}\n")
            else:
                print(f"{Colors.DIM}Enter desired allocation % for each strategy (default: {allocation}%):{Colors.RESET}\n")
            
            for strategy in strategies:
                # Use calculated allocation as default if available, otherwise use provided default
                default_alloc = calculated_allocations.get(strategy, allocation)
                if calculated_allocations.get(strategy):
                    prompt = f"Allocation % for '{strategy}' [calculated: {default_alloc:.2f}%]: "
                else:
                    prompt = f"Allocation % for '{strategy}' [{default_alloc}%]: "
                try:
                    user_input = input(prompt).strip()
                    if user_input:
                        strategy_allocations[strategy] = float(user_input)
                    else:
                        strategy_allocations[strategy] = default_alloc
                except (ValueError, KeyboardInterrupt):
                    typer.echo(f"\nUsing default allocation {default_alloc:.2f}% for '{strategy}'")
                    strategy_allocations[strategy] = default_alloc
        else:
            # Single strategy, use default allocation
            strategy_allocations = None

        # Initialize HTML reporter if requested
        html_reporter = None
        if html_report:
            # Find logo path - try multiple locations
            logo_path = None
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "images", "logo.png"),
                "images/logo.png",
                os.path.join(os.getcwd(), "images", "logo.png"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    logo_path = path
                    break
            
            if logo_path:
                html_reporter = HTMLReporter(logo_path)
            else:
                html_reporter = HTMLReporter("")
            html_reporter.start_report(
                test.get_name(),
                "Portfolio Stress Test",
                {
                    "portfolio_size": portfolio_size,
                    "allocation": allocation,
                    "simulations": simulations,
                    "force_one_lot": force_one_lot,
                }
            )

        # Prepare test kwargs
        test_kwargs = {
            "portfolio_size": portfolio_size,
            "allocation_pct": allocation,
            "strategy_allocations": strategy_allocations if strategy_allocations else None,
            "simulations": simulations,
            "force_one_lot": force_one_lot,
            "output_dir": output_dir,
        }

        # Run test on entire portfolio (not split by strategy)
        results = test.run(df, verbose=True, **test_kwargs)

        # Generate HTML report if requested
        if html_reporter:
            # Get percentile figure from results
            percentile_figure = results.get("percentile_figure")
            _generate_html_report(html_reporter, test, results, df, percentile_figure)
            report_path = html_reporter.save_report()
            print(f"\n{Colors.BRIGHT_GREEN}HTML report saved to: {report_path}{Colors.RESET}")

        # Print disclaimer
        _print_disclaimer()

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@test_app.command("portfolio_correlation")
def test_portfolio_correlation(
    file_path: str = typer.Argument(..., help="Path to portfolio data file"),
    starting_capital: float = typer.Option(
        100000.0, "--starting-capital", "-c", help="Starting capital for equity curve calculation (default: 100000)"
    ),
    rolling_window: int = typer.Option(
        30, "--rolling-window", "-w", help="Rolling window size for correlation (default: 30 trades)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save outputs"
    ),
    html_report: bool = typer.Option(
        False, "--html-report", help="Generate HTML report"
    ),
):
    """
    Analyze correlation between all portfolio strategies.
    
    Calculates Pearson and Spearman correlation coefficients on both returns and cumulative returns,
    includes rolling correlation analysis, and generates comprehensive visualizations including
    heatmaps, scatter plots, time series overlays, and rolling correlation plots.
    
    Examples:
        evtools test portfolio_correlation portfolio.csv --starting-capital 100000 --rolling-window 30
        evtools test portfolio_correlation portfolio.csv -c 50000 -w 50 --html-report
    """
    # Validate file exists
    if not Path(file_path).exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    # Validate rolling window
    if rolling_window <= 0:
        typer.echo("Error: rolling_window must be greater than 0", err=True)
        raise typer.Exit(1)

    # Print header
    _print_header()

    try:
        # Find test
        test = _find_test("portfolio_correlation")

        # Parse file to get metadata
        parser = _find_parser(file_path)
        df = parser.parse(file_path)
        metadata = parser.get_metadata(df)

        # Normalize
        df = NORMALIZER.normalize(df)

        # Enrich
        df = ENRICHER.enrich(df)

        # Check if portfolio (multiple strategies)
        is_portfolio = metadata.get("is_portfolio", False)
        strategies = metadata.get("strategies", [])

        if not is_portfolio or len(strategies) < 2:
            typer.echo("Error: Portfolio correlation test requires at least 2 strategies", err=True)
            raise typer.Exit(1)

        # Initialize HTML reporter if requested
        html_reporter = None
        if html_report:
            # Find logo path - try multiple locations
            logo_path = None
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "images", "logo.png"),
                "images/logo.png",
                os.path.join(os.getcwd(), "images", "logo.png"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    logo_path = path
                    break
            
            if logo_path:
                html_reporter = HTMLReporter(logo_path)
            else:
                html_reporter = HTMLReporter("")
            html_reporter.start_report(
                test.get_name(),
                "Portfolio Correlation Analysis",
                {
                    "starting_capital": starting_capital,
                    "rolling_window": rolling_window,
                }
            )

        # Prepare test kwargs
        test_kwargs = {
            "starting_capital": starting_capital,
            "rolling_window": rolling_window,
            "output_dir": output_dir,
        }

        # Run test on entire portfolio (not split by strategy)
        results = test.run(df, verbose=True, **test_kwargs)

        # Generate HTML report if requested
        if html_reporter:
            _generate_html_report(html_reporter, test, results, df)
            report_path = html_reporter.save_report()
            print(f"\n{Colors.BRIGHT_GREEN}HTML report saved to: {report_path}{Colors.RESET}")

        # Print disclaimer
        _print_disclaimer()

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@test_app.command("tail_overfitting")
def test_tail_overfitting(
    file_path: str = typer.Argument(..., help="Path to data file"),
    tail_percentage: float = typer.Option(
        1.0, "--tail-percentage", "-t", help="Percentage of trades to consider as tail events (default: 1.0)"
    ),
    max_score: float = typer.Option(
        12.0, "--max-score", "-m", help="Maximum acceptable overfitting score (default: 12.0)"
    ),
    tail_direction: str = typer.Option(
        "all", "--tail-direction", "-d", help="Which tail events to analyze: 'all', 'positive', or 'negative' (default: 'all')"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save outputs"
    ),
    html_report: bool = typer.Option(
        False, "--html-report", help="Generate HTML report"
    ),
):
    """
    Detect potential overfitting to extreme tail events in backtest results.
    
    Identifies tail events (top X% by absolute P/L), calculates overfitting scores,
    and shows baseline performance metrics without tail events to assess strategy robustness.
    
    Examples:
        evtools test tail_overfitting data.csv --tail-percentage 1.0 --max-score 12.0
        evtools test tail_overfitting data.csv --tail-direction positive --tail-percentage 2.0
    """
    # Validate file exists
    if not Path(file_path).exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    # Validate tail_percentage
    if not 0.1 <= tail_percentage <= 50.0:
        typer.echo("Error: tail-percentage must be between 0.1 and 50.0", err=True)
        raise typer.Exit(1)

    # Validate max_score
    if max_score < 1.0:
        typer.echo("Error: max-score must be at least 1.0", err=True)
        raise typer.Exit(1)

    # Validate tail_direction
    if tail_direction not in ["all", "positive", "negative"]:
        typer.echo("Error: tail-direction must be 'all', 'positive', or 'negative'", err=True)
        raise typer.Exit(1)

    # Print header
    _print_header()

    try:
        # Find test
        test = _find_test("tail_overfitting")

        # Prepare test kwargs
        test_kwargs = {
            "tail_percentage": tail_percentage,
            "max_score": max_score,
            "tail_direction": tail_direction,
        }

        # Process file
        _process_file(file_path, test, test_kwargs, output_dir)

        # Print disclaimer
        _print_disclaimer()

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@test_app.command("track")
def test_track(
    file_path: str = typer.Argument(..., help="Path to live trading data file"),
    starting_capital: float = typer.Option(
        100000.0, "--starting-capital", "-c", help="Starting capital for equity curve calculation (default: 100000)"
    ),
    extra_fees: float = typer.Option(
        0.0, "--extra-fees", "-f", help="Monthly extra fees (e.g., automation costs) (default: 0.0)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Directory to save outputs"
    ),
    html_report: bool = typer.Option(
        False, "--html-report", help="Generate HTML report"
    ),
):
    """
    Analyze live trading portfolio performance.
    
    Calculates equity curve, CAGR, Sharpe/Sortino ratios, drawdown analysis,
    and provides strategy-level statistics with calendar visualizations.
    
    Examples:
        evtools test track portfolio-track.csv --starting-capital 100000
        evtools test track portfolio-track.csv -c 50000 -f 150 -o ./output
    """
    # Validate file exists
    if not Path(file_path).exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    # Validate starting_capital
    if starting_capital <= 0:
        typer.echo("Error: starting-capital must be greater than 0", err=True)
        raise typer.Exit(1)

    # Print header
    _print_header()

    try:
        # Find test
        test = _find_test("track")

        # Parse, normalize, and enrich the entire file (don't split by strategy)
        parser = _find_parser(file_path)
        df = parser.parse(file_path)
        
        # Normalize
        df = NORMALIZER.normalize(df)
        
        # Enrich
        df = ENRICHER.enrich(df)

        # Initialize HTML reporter if requested
        html_reporter = None
        if html_report:
            # Find logo path - try multiple locations
            logo_path = None
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "images", "logo.png"),
                "images/logo.png",
                os.path.join(os.getcwd(), "images", "logo.png"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    logo_path = path
                    break
            
            if logo_path:
                html_reporter = HTMLReporter(logo_path)
            else:
                html_reporter = HTMLReporter("")
            html_reporter.start_report(
                test.get_name(),
                "Portfolio Track Analysis",
                {
                    "starting_capital": starting_capital,
                    "extra_fees": extra_fees,
                }
            )

        # Prepare test kwargs
        test_kwargs = {
            "starting_capital": starting_capital,
            "extra_fees": extra_fees,
            "output_dir": output_dir,
        }

        # Run test on entire portfolio (not split by strategy)
        results = test.run(df, verbose=True, **test_kwargs)

        # Generate HTML report if requested
        if html_reporter:
            # Capture track chart if available
            track_figure = None
            equity_curve_data = results.get("equity_curve", {})
            drawdown_analysis = results.get("drawdown_analysis", {})
            if equity_curve_data.get("equity_curve") and equity_curve_data.get("dates"):
                from ..output.visualizers import create_track_chart
                track_figure = create_track_chart(
                    equity_curve_data["equity_curve"],
                    drawdown_analysis.get("drawdown_curve", []),
                    equity_curve_data["dates"],
                    "Portfolio Track Analysis",
                    output_dir,
                    return_figure=True,
                )
            _generate_html_report(html_reporter, test, results, df, track_figure)
            
            report_path = html_reporter.save_report()
            print(f"\n{Colors.BRIGHT_GREEN}HTML report saved to: {report_path}{Colors.RESET}")

        # Print disclaimer
        _print_disclaimer()

    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


# Register the test subcommands
app.add_typer(test_app)


@app.command()
def list_tests():
    """List all available tests."""
    _print_header()
    print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}Available Tests:{Colors.RESET}\n")

    for test in TESTS:
        print(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}{test.get_name()}{Colors.RESET}")
        print(f"  {Colors.DIM}{test.get_description()}{Colors.RESET}\n")

    _print_disclaimer()


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
