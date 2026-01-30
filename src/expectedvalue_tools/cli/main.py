"""Main CLI entry point using Typer."""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import typer
from ..parsers import BaseParser, OptionOmegaParser
from ..normalizers import BaseNormalizer, TradeDataNormalizer
from ..enrichers import BaseEnricher, TradeEnricher
from ..tests import BaseTest, PowerAnalysisTest, LiveBacktestComparisonTest, DrawdownAnalysisTest
from ..output.visualizers import create_histograms
from ..output.formatters import print_box
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
TESTS = [PowerAnalysisTest(), LiveBacktestComparisonTest(), DrawdownAnalysisTest()]

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
            )
    else:
        # Single strategy
        strategy_name = "Single Strategy"
        unique_strategies = metadata.get("strategies", [])
        if len(unique_strategies) == 1:
            strategy_name = str(unique_strategies[0])

        _process_strategy(df, strategy_name, test, test_kwargs, output_dir)


def _process_strategy(
    data,
    strategy_name: str,
    test: BaseTest,
    test_kwargs: dict,
    output_dir: Optional[str],
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

    # Run test
    results = test.run(data, verbose=True, **test_kwargs)

    # Create histograms if matplotlib available
    if "P/L per Contract" in data.columns:
        pnl_per_contract = data["P/L per Contract"].values
        pnl_per_contract = pnl_per_contract[
            ~(np.isnan(pnl_per_contract) | np.isinf(pnl_per_contract))
        ]

        if len(pnl_per_contract) > 0:
            create_histograms(
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
            )


def _process_compare_test(
    backtest_file: str,
    live_file: str,
    test: BaseTest,
    window_minutes: int,
    starting_portfolio_size: float,
    source_of_truth: str,
    output_dir: Optional[str],
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

    # Run test
    results = test.run(
        backtest_data=backtest_df,
        live_data=live_df,
        window_minutes=window_minutes,
        starting_portfolio_size=starting_portfolio_size,
        source_of_truth=source_of_truth,
        verbose=True,
    )


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
        _process_file(file_path, test, test_kwargs, output_dir)

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
        _process_compare_test(backtest_file, live_file, test, window_minutes, starting_portfolio_size, source_of_truth, output_dir)

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
