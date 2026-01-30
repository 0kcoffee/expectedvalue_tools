"""Power analysis test for statistical power estimation."""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from .base import BaseTest
from ..output.formatters import (
    print_ascii_distribution,
    print_box,
    print_section_box,
    print_progress_bar,
)
from ..utils.colors import Colors
from ..utils.text import visible_length


def calculate_power_for_n(
    trade_returns: np.ndarray,
    n: int,
    simulations: int = 10000,
) -> Tuple[float, np.ndarray]:
    """
    Calculate statistical power for a given sample size n.

    Args:
        trade_returns: Raw P&L per contract from backtest
        n: Sample size to test
        simulations: Number of Monte Carlo simulations

    Returns:
        Tuple of (power, path_means) where power is the probability that mean > 0
    """
    simulated_paths = np.random.choice(
        trade_returns, size=(simulations, n), replace=True
    )
    path_means = np.mean(simulated_paths, axis=1)
    empirical_power = np.mean(path_means > 0)
    return empirical_power, path_means


def find_minimum_sample_size(
    trade_returns: np.ndarray,
    target_power: float,
    lower_bound: int,
    upper_bound: int,
    simulations: int = 10000,
    tolerance: int = 1,
) -> Tuple[int, float, np.ndarray]:
    """
    Use binary search to find the minimum sample size that achieves target power.

    Args:
        trade_returns: Raw P&L per contract from backtest
        target_power: Target power level (0-1)
        lower_bound: Lower bound for search (power < target)
        upper_bound: Upper bound for search (power >= target)
        simulations: Number of Monte Carlo simulations per test
        tolerance: Minimum difference in N to consider (default: 1 for precision)

    Returns:
        Tuple of (minimum_n, power_at_minimum_n, path_means)
    """
    # Binary search to find minimum N
    low = lower_bound
    high = upper_bound
    best_n = upper_bound
    best_power = None
    best_path_means = None

    # Cache results to avoid recalculating
    power_cache = {}

    while high - low > tolerance:
        mid = (low + high) // 2

        # Check cache first
        if mid in power_cache:
            power, path_means = power_cache[mid]
        else:
            power, path_means = calculate_power_for_n(trade_returns, mid, simulations)
            power_cache[mid] = (power, path_means)

        if power >= target_power:
            # This N works, try smaller
            best_n = mid
            best_power = power
            best_path_means = path_means
            high = mid
        else:
            # Need larger N
            low = mid + 1

    # Final check: verify the found N and check if we can go one lower
    if best_n > lower_bound:
        # Check if N-1 also works (if tolerance allows)
        check_n = best_n - tolerance
        if check_n >= lower_bound:
            if check_n in power_cache:
                check_power, check_path_means = power_cache[check_n]
            else:
                check_power, check_path_means = calculate_power_for_n(
                    trade_returns, check_n, simulations
                )

            if check_power >= target_power:
                best_n = check_n
                best_power = check_power
                best_path_means = check_path_means

    # Ensure we have valid results
    if best_power is None or best_path_means is None:
        # Fallback: recalculate for best_n
        best_power, best_path_means = calculate_power_for_n(
            trade_returns, best_n, simulations
        )

    return best_n, best_power, best_path_means


class PowerAnalysisTest(BaseTest):
    """Statistical power analysis test for trading strategies."""

    def get_name(self) -> str:
        """Get the name of the test."""
        return "power"

    def get_description(self) -> str:
        """Get a description of what the test does."""
        return (
            "Estimates the sample size needed for an observed edge to be statistically stable. "
            "Uses bootstrap sampling with replacement to test statistical power. "
            "Statistical power = probability that mean return > 0 in a random sample."
        )

    def run(
        self,
        data: pd.DataFrame,
        target_power: float = 0.95,
        simulations: int = 10000,
        verbose: bool = True,
    ) -> Dict:
        """
        Run power analysis on the provided data.

        Args:
            data: DataFrame with "P/L per Contract" column
            target_power: Target power level (0-1, default: 0.95)
            simulations: Number of Monte Carlo simulations (default: 10000)
            verbose: If True, print formatted output (default: True)

        Returns:
            Dictionary with analysis results
        """
        # Validate required column
        self.validate_data(data, ["P/L per Contract"])

        # Extract P/L per contract
        trade_returns = data["P/L per Contract"].values

        # Remove any remaining NaN or inf values
        trade_returns = trade_returns[~np.isnan(trade_returns)]
        trade_returns = trade_returns[~np.isinf(trade_returns)]

        if len(trade_returns) == 0:
            raise ValueError("No valid data points for power analysis")

        # Calculate statistics
        observed_mean = np.mean(trade_returns)
        observed_std = np.std(trade_returns)
        current_n = len(trade_returns)
        win_rate = np.mean(trade_returns > 0) if current_n > 0 else 0.0

        # Range of sample sizes to test
        candidate_ns = [25, 50, 100, 250, 500, 1000, 2000]

        candidate_results = []
        recommended_n = None
        mc_means = None

        if verbose:
            self._print_performance_metrics(observed_mean, observed_std, current_n, win_rate)
            print_ascii_distribution(trade_returns, "P/L per Contract Distribution")
            self._print_testing_section(observed_mean)

        # First pass: test candidate sizes to find rough range
        lower_bound_n = None
        upper_bound_n = None
        first_candidate = candidate_ns[0] if candidate_ns else 1

        for n in candidate_ns:
            power, path_means = calculate_power_for_n(trade_returns, n, simulations)
            empirical_power = power

            candidate_results.append(
                {
                    "n": n,
                    "power": empirical_power,
                    "path_means": path_means,
                }
            )

            # Track bounds for binary search
            if empirical_power < target_power:
                lower_bound_n = n
            elif empirical_power >= target_power and upper_bound_n is None:
                upper_bound_n = n
                if recommended_n is None:
                    recommended_n = n
                    mc_means = path_means

        # If first candidate already achieves target, search below it
        if upper_bound_n == first_candidate and first_candidate > 1:
            test_n = max(1, min(current_n, first_candidate - 1))
            if test_n < first_candidate:
                test_power, test_path_means = calculate_power_for_n(
                    trade_returns, test_n, simulations
                )
                candidate_results.append(
                    {
                        "n": test_n,
                        "power": test_power,
                        "path_means": test_path_means,
                    }
                )

                if test_power >= target_power:
                    upper_bound_n = test_n
                    lower_bound_n = max(1, test_n - 10)
                    recommended_n = test_n
                    mc_means = test_path_means
                else:
                    lower_bound_n = test_n

        # Refine recommendation using binary search if we found bounds
        if (
            lower_bound_n is not None
            and upper_bound_n is not None
            and upper_bound_n > lower_bound_n + 1
        ):
            if verbose:
                print(
                    f"{Colors.DIM}Refining sample size estimate using binary search...{Colors.RESET}"
                )
            min_n, min_power, min_path_means = find_minimum_sample_size(
                trade_returns,
                target_power,
                lower_bound_n,
                upper_bound_n,
                simulations,
                tolerance=1,
            )
            recommended_n = min_n
            mc_means = min_path_means
            candidate_results.append(
                {
                    "n": min_n,
                    "power": min_power,
                    "path_means": min_path_means,
                }
            )

        # If target not reached, estimate recommended_n
        if recommended_n is None:
            if verbose:
                print(f"\n{Colors.BRIGHT_YELLOW}{Colors.BOLD}⚠  WARNING{Colors.RESET}")
                print(
                    f"{Colors.BRIGHT_YELLOW}Target confidence ({target_power*100:.0f}%) not reached within the test range.{Colors.RESET}"
                )
            if candidate_results:
                max_tested = candidate_results[-1]["n"]
                max_power = candidate_results[-1]["power"]
                if max_power > 0 and max_power < target_power:
                    power_gap = target_power - max_power
                    estimated_multiplier = 1 + (power_gap * 3)
                    recommended_n = int(max_tested * estimated_multiplier)
                    recommended_n = ((recommended_n + 49) // 50) * 50
                else:
                    recommended_n = max_tested * 2
                mc_means = candidate_results[-1]["path_means"]
            else:
                recommended_n = current_n * 10

        # Calculate power for CURRENT sample size
        current_power = None
        current_mc_means = None

        if current_n > 0:
            current_simulated_paths = np.random.choice(
                trade_returns, size=(simulations, current_n), replace=True
            )
            current_mc_means = np.mean(current_simulated_paths, axis=1)
            current_power = np.mean(current_mc_means > 0)

            if verbose:
                negative_count = np.sum(current_mc_means <= 0)
                if negative_count > 0:
                    print(f"\n{Colors.BRIGHT_YELLOW}{Colors.BOLD}⚠  WARNING{Colors.RESET}")
                    print(
                        f"{Colors.BRIGHT_YELLOW}{negative_count} out of {len(current_mc_means):,} simulations had negative mean returns{Colors.RESET}\n"
                    )

                print()
                print_ascii_distribution(
                    current_mc_means,
                    f"Monte Carlo Mean Returns Distribution (N={current_n:,}, {simulations:,} simulations)",
                )

                self._print_power_assessment(current_power, current_n, observed_mean)

        # Check if current power already meets target
        target_already_reached = current_power is not None and current_power >= target_power

        if target_already_reached and recommended_n is not None and recommended_n > current_n:
            recommended_n = current_n

        if verbose:
            self._print_recommendation(
                recommended_n,
                target_power,
                observed_mean,
                candidate_results,
                target_already_reached,
                current_n,
                current_power,
            )

        return {
            "observed_mean": observed_mean,
            "observed_std": observed_std,
            "current_n": current_n,
            "win_rate": win_rate,
            "current_power": current_power,
            "recommended_n": recommended_n,
            "mc_means": current_mc_means,
            "candidate_results": candidate_results,
        }

    def _print_performance_metrics(
        self, observed_mean: float, observed_std: float, current_n: int, win_rate: float
    ) -> None:
        """Print performance metrics box."""
        box_width = 70
        lines = [
            (
                "Observed Edge (Mean Return per Contract):",
                f"${observed_mean:>10,.2f}",
                Colors.BRIGHT_GREEN,
            ),
            ("Observed Volatility:", f"${observed_std:>10,.2f}", Colors.BRIGHT_YELLOW),
            (f"Current Sample Size:", f"{current_n:>10,}", Colors.BRIGHT_CYAN),
            ("Win Rate (Profitable Trades):", f"{win_rate*100:>6.2f}%", Colors.BRIGHT_GREEN),
        ]
        print_box(box_width, "Strategy Performance Metrics", lines)

    def _print_testing_section(self, observed_mean: float) -> None:
        """Print testing section box."""
        box_width = 70
        lines = [
            "Power = probability that mean return > 0 in a random sample",
            "Using bootstrap sampling with replacement",
            f"Note: Power analysis is specific to the observed effect (${observed_mean:.2f}/contract)",
        ]
        print_section_box(
            box_width, "Testing Sample Sizes for Strategy Stability", lines
        )

    def _print_power_assessment(
        self, current_power: float, current_n: int, observed_mean: float
    ) -> None:
        """Print power assessment box."""
        if current_power is None:
            return

        box_width = 80
        header_text = f"Power Assessment (Current Sample: N={current_n:,})"
        header_spaces = box_width - visible_length(header_text) - 4

        line1_text = "Statistical Power = probability that mean return > 0 in a random sample"
        line1_spaces = box_width - visible_length(line1_text) - 4

        line2_text = f"Power is specific to the observed effect (${observed_mean:.2f}/contract)"
        line2_spaces = box_width - visible_length(line2_text) - 4

        power_label = "Power (for observed effect):"
        power_value = f"{current_power*100:>10.2f}%"
        power_line = f"{power_label} {power_value}"
        power_spaces = box_width - visible_length(power_line) - 4

        print(f"\n{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}")
        print(
            f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{header_text}{Colors.RESET} {' ' * header_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
        )
        print(
            f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.DIM}{line1_text}{Colors.RESET} {' ' * line1_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
        )
        print(
            f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.DIM}{line2_text}{Colors.RESET} {' ' * line2_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
        )
        print(
            f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.DIM}{power_label}{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_GREEN}{power_value}{Colors.RESET} {' ' * power_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
        )
        print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}\n")

    def _print_recommendation(
        self,
        recommended_n: Optional[int],
        target_power: float,
        observed_mean: float,
        candidate_results: list,
        target_already_reached: bool,
        current_n: int,
        current_power: Optional[float],
    ) -> None:
        """Print recommendation box."""
        if recommended_n is None:
            return

        box_width = 70
        is_estimate = recommended_n is not None and (
            not candidate_results or candidate_results[-1]["power"] < target_power
        )

        if is_estimate:
            header_text = "⚠ ESTIMATED RECOMMENDATION"
        else:
            header_text = "✓ RECOMMENDATION"
        header_spaces = box_width - visible_length(header_text) - 4

        label = "Minimum Required Sample Size:"
        n_str = f"{recommended_n:,}"
        trades_text = "trades for"
        confidence_str = f"{target_power*100:.0f}%"
        confidence_text = "confidence"
        effect_note = f"for the observed effect (${observed_mean:.2f}/contract)"

        if is_estimate:
            estimate_note = "Note: Estimated (target not reached in test range)"
        else:
            estimate_note = None

        line_content = f"{label} {n_str} {trades_text} {confidence_str} {confidence_text}"
        line_spaces = box_width - visible_length(line_content) - 4
        note_spaces = box_width - visible_length(effect_note) - 4
        if estimate_note:
            estimate_spaces = box_width - visible_length(estimate_note) - 4

        print(f"\n{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}")
        print(
            f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{header_text}{Colors.RESET} {' ' * header_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
        )
        print(
            f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.DIM}{label}{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_CYAN}{n_str}{Colors.RESET} {Colors.DIM}{trades_text}{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_GREEN}{confidence_str}{Colors.RESET} {Colors.DIM}{confidence_text}{Colors.RESET} {' ' * max(0, line_spaces)} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
        )
        print(
            f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.DIM}{effect_note}{Colors.RESET} {' ' * note_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
        )
        if estimate_note:
            print(
                f"{Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET} {Colors.BRIGHT_YELLOW}{estimate_note}{Colors.RESET} {' ' * estimate_spaces} {Colors.BRIGHT_WHITE}{Colors.BOLD}║{Colors.RESET}"
            )
        print(f"{Colors.BRIGHT_WHITE}{Colors.BOLD}{'═' * box_width}{Colors.RESET}\n")

        # Show success message if target has been reached
        if target_already_reached:
            box_width = 70
            header_text = "✓ TARGET POWER ACHIEVED"
            header_spaces = box_width - visible_length(header_text) - 4

            success_line = f"Current sample size ({current_n:,} trades) already achieves {target_power*100:.0f}% power"
            success_spaces = box_width - visible_length(success_line) - 4

            power_line = f"Current power: {current_power*100:.2f}% (target: {target_power*100:.0f}%)"
            power_spaces = box_width - visible_length(power_line) - 4

            print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'─' * box_width}{Colors.RESET}")
            print(
                f"{Colors.BRIGHT_GREEN}{Colors.BOLD}│{Colors.RESET} {Colors.BOLD}{Colors.BRIGHT_WHITE}{header_text}{Colors.RESET} {' ' * header_spaces} {Colors.BRIGHT_GREEN}{Colors.BOLD}│{Colors.RESET}"
            )
            print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'─' * box_width}{Colors.RESET}")
            print(
                f"{Colors.BRIGHT_GREEN}{Colors.BOLD}│{Colors.RESET} {Colors.DIM}{success_line}{Colors.RESET} {' ' * success_spaces} {Colors.BRIGHT_GREEN}{Colors.BOLD}│{Colors.RESET}"
            )
            print(
                f"{Colors.BRIGHT_GREEN}{Colors.BOLD}│{Colors.RESET} {Colors.DIM}{power_line}{Colors.RESET} {' ' * power_spaces} {Colors.BRIGHT_GREEN}{Colors.BOLD}│{Colors.RESET}"
            )
            print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'─' * box_width}{Colors.RESET}\n")
        elif recommended_n > current_n:
            print_progress_bar(current_n, recommended_n)
