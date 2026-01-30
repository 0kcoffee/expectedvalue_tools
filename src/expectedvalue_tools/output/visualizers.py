"""Visualization utilities using matplotlib."""

import os
import sys
from typing import Dict, Optional, List
from datetime import datetime
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_histograms(
    original_data: np.ndarray,
    mc_means: Optional[np.ndarray],
    strategy_name: str,
    output_dir: Optional[str],
    stats: Dict,
) -> None:
    """
    Create and save histograms for original data and Monte Carlo results.

    Args:
        original_data: Original P/L per contract data
        mc_means: Monte Carlo mean returns (can be None)
        strategy_name: Name of the strategy (for file naming)
        output_dir: Directory to save images (None = display only)
        stats: Dictionary with statistics for annotations
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping histogram generation (matplotlib not available)")
        print("Install matplotlib to generate histograms: pip install matplotlib")
        return

    # Sanitize strategy name for filename
    safe_name = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_" for c in strategy_name
    )
    safe_name = safe_name.replace(" ", "_")[:50]  # Limit length

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Original Data Histogram
    ax1.hist(original_data, bins=30, edgecolor="black", alpha=0.7, color="#6366f1")
    ax1.axvline(
        stats["observed_mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: ${stats["observed_mean"]:.2f}',
    )
    ax1.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5, label="Zero")
    ax1.set_xlabel("P/L per Contract ($)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Original Data Distribution\n{strategy_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Mean: ${stats["observed_mean"]:.2f}\n'
    stats_text += f'Std: ${stats["observed_std"]:.2f}\n'
    stats_text += f'N: {stats["current_n"]}\n'
    win_rate = stats.get("win_rate", 0) * 100
    stats_text += f"Win Rate: {win_rate:.1f}%\n"
    stats_text += f'Power: {stats["current_power"]*100:.1f}%'
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Monte Carlo Results Histogram
    if mc_means is not None:
        ax2.hist(mc_means, bins=30, edgecolor="black", alpha=0.7, color="#10b981")
        ax2.axvline(
            np.mean(mc_means),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: ${np.mean(mc_means):.2f}",
        )
        ax2.axvline(
            0, color="black", linestyle="-", linewidth=1, alpha=0.5, label="Zero"
        )
        ax2.set_xlabel("Mean Return per Contract ($)")
        ax2.set_ylabel("Frequency")

        mc_power = np.mean(mc_means > 0)
        n_used = stats["current_n"]  # Use current sample size, not recommended
        ax2.set_title(f"Monte Carlo Results (N={n_used})\n{strategy_name}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add statistics text
        mc_stats_text = f"Mean: ${np.mean(mc_means):.2f}\n"
        mc_stats_text += f"Std: ${np.std(mc_means):.2f}\n"
        mc_stats_text += f"Simulations: {len(mc_means)}\n"
        mc_stats_text += f"Power: {mc_power*100:.1f}%"
        ax2.text(
            0.02,
            0.98,
            mc_stats_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "No Monte Carlo results available",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )
        ax2.set_title(f"Monte Carlo Results\n{strategy_name}")

    plt.tight_layout()

    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{safe_name}_histograms.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Histograms saved to: {filepath}")
    else:
        plt.show()

    plt.close()


def create_drawdown_chart(
    equity_curve: List[float],
    drawdown_curve: List[float],
    dates: List[datetime],
    strategy_name: str,
    output_dir: Optional[str] = None,
) -> None:
    """
    Create and display/save a drawdown chart showing equity curve and drawdown over time.

    Args:
        equity_curve: List of portfolio values over time
        drawdown_curve: List of drawdown percentages over time
        dates: List of datetime objects corresponding to values
        strategy_name: Name of the strategy (for title)
        output_dir: Directory to save image (None = display only)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping drawdown chart generation (matplotlib not available)")
        print("Install matplotlib to generate charts: pip install matplotlib")
        return

    if len(equity_curve) == 0 or len(dates) == 0:
        return

    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot equity curve on left axis
    color1 = "#6366f1"
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Portfolio Value ($)", color=color1, fontsize=12)
    ax1.plot(dates, equity_curve, color=color1, linewidth=2, label="Equity Curve")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Drawdown Analysis - {strategy_name}", fontsize=14, fontweight="bold")

    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Create second y-axis for drawdown
    ax2 = ax1.twinx()
    color2 = "#ef4444"
    ax2.set_ylabel("Drawdown (%)", color=color2, fontsize=12)
    ax2.fill_between(
        dates, drawdown_curve, 0, color=color2, alpha=0.3, label="Drawdown"
    )
    ax2.plot(dates, drawdown_curve, color=color2, linewidth=1.5, linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.invert_yaxis()  # Invert so drawdown goes up (negative is better)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Add max drawdown annotation
    max_dd_idx = np.argmax(drawdown_curve)
    max_dd_value = drawdown_curve[max_dd_idx]
    max_dd_date = dates[max_dd_idx]
    max_dd_equity = equity_curve[max_dd_idx]

    ax2.annotate(
        f"Max DD: {max_dd_value:.2f}%",
        xy=(max_dd_date, max_dd_value),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_" for c in strategy_name
        )
        safe_name = safe_name.replace(" ", "_")[:50]
        filename = f"{safe_name}_drawdown_chart.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Drawdown chart saved to: {filepath}")
    else:
        plt.show()

    plt.close()
