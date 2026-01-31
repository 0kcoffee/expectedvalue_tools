"""Tail events overfitting test for trading strategies."""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from .base import BaseTest
from ..output.formatters import (
    print_box,
    print_section_box,
    print_ascii_distribution,
)
from ..utils.colors import Colors
from ..utils.text import visible_length


class TailOverfittingTest(BaseTest):
    """Test that detects potential overfitting to extreme tail events in backtest results."""

    def get_name(self) -> str:
        """Get the name of the test."""
        return "tail_overfitting"

    def get_description(self) -> str:
        """Get a description of what the test does."""
        return (
            "Detects whether a strategy's performance is disproportionately driven by a small number "
            "of extreme tail events. Identifies tail events, calculates overfitting scores, and "
            "shows baseline performance metrics without tail events."
        )

    def run(
        self,
        data: pd.DataFrame,
        tail_percentage: float = 1.0,
        max_score: float = 12.0,
        tail_direction: str = "all",
        verbose: bool = True,
    ) -> Dict:
        """
        Run tail overfitting analysis on the provided data.

        Args:
            data: DataFrame with normalized and enriched trade data
            tail_percentage: Percentage of trades to consider as tail events (default: 1.0)
            max_score: Maximum acceptable overfitting score (default: 12.0)
            tail_direction: Which tail events to analyze - "all", "positive", or "negative" (default: "all")
            verbose: If True, print formatted output (default: True)

        Returns:
            Dictionary with analysis results
        """
        # Validate required columns
        self.validate_data(data, ["P/L per Contract"])

        # Extract P/L per contract values
        pnl_values = data["P/L per Contract"].values.copy()

        # Filter by tail direction
        if tail_direction == "positive":
            mask = pnl_values > 0
            pnl_values = pnl_values[mask]
            direction_label = "Positive (Profits)"
        elif tail_direction == "negative":
            mask = pnl_values < 0
            pnl_values = pnl_values[mask]
            direction_label = "Negative (Losses)"
        else:
            direction_label = "All"

        if len(pnl_values) < 10:
            return {
                "total_trades": len(pnl_values),
                "tail_count": 0,
                "avg_abs_pnl": None,
                "avg_tail_abs_pnl": None,
                "overfitting_score": None,
                "tail_contribution": None,
                "tail_percentage": tail_percentage,
                "max_score": max_score,
                "tail_direction": tail_direction,
                "passed": False,
                "error": f"Insufficient trades (minimum 10 required, found {len(pnl_values)} {direction_label.lower()} trades)",
                "baseline_metrics": None,
            }

        # Calculate absolute P/L values
        abs_pnl = np.abs(pnl_values)

        # Sort by absolute value (descending) to find tail events
        sorted_indices = np.argsort(abs_pnl)[::-1]

        # Determine number of tail events
        tail_count = max(1, int(np.ceil(len(pnl_values) * tail_percentage / 100)))
        tail_indices = sorted_indices[:tail_count]

        # Calculate metrics
        avg_abs_pnl = float(np.mean(abs_pnl))
        avg_tail_abs_pnl = float(np.mean(abs_pnl[tail_indices]))

        # Calculate overfitting score
        overfitting_score = avg_tail_abs_pnl / avg_abs_pnl if avg_abs_pnl > 0 else 0

        # Calculate tail contribution (what % of total absolute P/L comes from tail events)
        total_abs_pnl = float(np.sum(abs_pnl))
        tail_abs_pnl = float(np.sum(abs_pnl[tail_indices]))
        tail_contribution = (tail_abs_pnl / total_abs_pnl * 100) if total_abs_pnl > 0 else 0

        # Determine pass/fail
        passed = overfitting_score < max_score

        # BASELINE PERFORMANCE ANALYSIS: Remove tail events and calculate metrics
        baseline_indices = np.setdiff1d(np.arange(len(pnl_values)), tail_indices)
        baseline_pnl = pnl_values[baseline_indices]

        baseline_total_pl = float(np.sum(baseline_pnl)) if len(baseline_pnl) > 0 else 0.0
        baseline_mean_pl = float(np.mean(baseline_pnl)) if len(baseline_pnl) > 0 else 0.0
        baseline_win_rate = float(np.mean(baseline_pnl > 0)) if len(baseline_pnl) > 0 else 0.0
        baseline_total_trades = len(baseline_pnl)

        # Comparison metrics
        pl_with_tail = float(np.sum(pnl_values))
        pl_difference = pl_with_tail - baseline_total_pl
        pl_difference_pct = (pl_difference / abs(pl_with_tail) * 100) if pl_with_tail != 0 else 0.0

        baseline_metrics = {
            "baseline_total_pl": baseline_total_pl,
            "baseline_mean_pl": baseline_mean_pl,
            "baseline_win_rate": baseline_win_rate,
            "baseline_total_trades": baseline_total_trades,
            "pl_with_tail": pl_with_tail,
            "pl_difference": pl_difference,
            "pl_difference_pct": pl_difference_pct,
        }

        if verbose:
            self._print_results(
                total_trades=len(pnl_values),
                tail_count=tail_count,
                avg_abs_pnl=avg_abs_pnl,
                avg_tail_abs_pnl=avg_tail_abs_pnl,
                overfitting_score=overfitting_score,
                tail_contribution=tail_contribution,
                tail_percentage=tail_percentage,
                max_score=max_score,
                tail_direction=tail_direction,
                direction_label=direction_label,
                passed=passed,
                baseline_metrics=baseline_metrics,
                pnl_values=pnl_values,
                tail_indices=tail_indices,
            )

        return {
            "total_trades": int(len(pnl_values)),
            "tail_count": int(tail_count),
            "avg_abs_pnl": round(avg_abs_pnl, 2),
            "avg_tail_abs_pnl": round(avg_tail_abs_pnl, 2),
            "overfitting_score": round(overfitting_score, 2),
            "tail_contribution": round(tail_contribution, 2),
            "tail_percentage": tail_percentage,
            "max_score": max_score,
            "tail_direction": tail_direction,
            "passed": passed,
            "baseline_metrics": baseline_metrics,
        }

    def _print_results(
        self,
        total_trades: int,
        tail_count: int,
        avg_abs_pnl: float,
        avg_tail_abs_pnl: float,
        overfitting_score: float,
        tail_contribution: float,
        tail_percentage: float,
        max_score: float,
        tail_direction: str,
        direction_label: str,
        passed: bool,
        baseline_metrics: Dict,
        pnl_values: np.ndarray,
        tail_indices: np.ndarray,
    ) -> None:
        """Print formatted results."""
        # Print main metrics
        box_width = 70
        lines = [
            (
                "Total Trades Analyzed:",
                f"{total_trades:,}",
                Colors.BRIGHT_CYAN,
            ),
            (
                f"Tail Events ({tail_percentage}%):",
                f"{tail_count:,}",
                Colors.BRIGHT_YELLOW,
            ),
            (
                "Average Absolute P/L (All Trades):",
                f"${avg_abs_pnl:>10,.2f}",
                Colors.BRIGHT_WHITE,
            ),
            (
                "Average Absolute P/L (Tail Events):",
                f"${avg_tail_abs_pnl:>10,.2f}",
                Colors.BRIGHT_YELLOW,
            ),
            (
                "Overfitting Score:",
                f"{overfitting_score:.2f}x",
                Colors.BRIGHT_RED if not passed else Colors.BRIGHT_GREEN,
            ),
            (
                "Tail Contribution:",
                f"{tail_contribution:.2f}%",
                Colors.BRIGHT_YELLOW,
            ),
            (
                "Max Score Threshold:",
                f"{max_score:.2f}x",
                Colors.BRIGHT_WHITE,
            ),
            (
                "Direction Analyzed:",
                direction_label,
                Colors.BRIGHT_CYAN,
            ),
        ]
        status_text = "PASSED" if passed else "FAILED"
        status_color = Colors.BRIGHT_GREEN if passed else Colors.BRIGHT_RED
        print_box(box_width, f"Tail Events Overfitting Test - {status_text}", lines, status_color)

        # Print baseline performance section
        self._print_baseline_section(baseline_metrics, box_width)

        # Print distribution
        if len(pnl_values) > 0:
            print_ascii_distribution(
                pnl_values,
                f"P/L per Contract Distribution (analyzing {direction_label.lower()})",
            )

    def _print_baseline_section(self, baseline_metrics: Dict, box_width: int) -> None:
        """Print baseline performance metrics section."""
        baseline_total_pl = baseline_metrics["baseline_total_pl"]
        baseline_mean_pl = baseline_metrics["baseline_mean_pl"]
        baseline_win_rate = baseline_metrics["baseline_win_rate"]
        baseline_total_trades = baseline_metrics["baseline_total_trades"]
        pl_with_tail = baseline_metrics["pl_with_tail"]
        pl_difference = baseline_metrics["pl_difference"]
        pl_difference_pct = baseline_metrics["pl_difference_pct"]

        # Section header
        section_lines = [
            "This section shows performance metrics WITHOUT tail events removed.",
            "This helps assess strategy robustness - if baseline is still profitable,",
            "the strategy is less dependent on extreme events.",
        ]
        print_section_box(box_width, "Baseline Performance (Without Tail Events)", section_lines)

        # Baseline metrics box
        baseline_color = Colors.BRIGHT_GREEN if baseline_total_pl > 0 else Colors.BRIGHT_RED
        lines = [
            (
                "Baseline Total P/L (without tail events):",
                f"${baseline_total_pl:>10,.2f}",
                baseline_color,
            ),
            (
                "Baseline Mean P/L per Contract:",
                f"${baseline_mean_pl:>10,.2f}",
                Colors.BRIGHT_WHITE,
            ),
            (
                "Baseline Win Rate:",
                f"{baseline_win_rate*100:>6.2f}%",
                Colors.BRIGHT_CYAN,
            ),
            (
                "Baseline Total Trades:",
                f"{baseline_total_trades:,}",
                Colors.BRIGHT_CYAN,
            ),
            (
                "Total P/L (with tail events):",
                f"${pl_with_tail:>10,.2f}",
                Colors.BRIGHT_WHITE,
            ),
            (
                "P/L Difference (tail contribution):",
                f"${pl_difference:>10,.2f}",
                Colors.BRIGHT_YELLOW,
            ),
            (
                "Tail Contribution (% of total P/L):",
                f"{abs(pl_difference_pct):.2f}%",
                Colors.BRIGHT_YELLOW,
            ),
        ]
        print_box(box_width, "Baseline Performance Metrics", lines)
