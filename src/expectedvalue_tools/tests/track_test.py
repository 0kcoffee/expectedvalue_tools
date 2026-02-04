"""Track test for analyzing live trading portfolio performance."""

from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from .base import BaseTest
from ..output.formatters import (
    print_box,
    print_section_box,
    print_ascii_timeseries,
    print_track_metrics,
    print_track_strategy_table,
    print_track_drawdown_analysis,
    print_drawdown_calendar,
)
from ..output.visualizers import create_track_chart
from ..utils.colors import Colors


class TrackTest(BaseTest):
    """Test that analyzes live trading portfolio performance."""

    def get_name(self) -> str:
        """Get the name of the test."""
        return "track"

    def get_description(self) -> str:
        """Get a description of what the test does."""
        return (
            "Analyzes live trading portfolio performance including equity curve, CAGR, "
            "Sharpe/Sortino ratios, drawdown analysis, and strategy-level statistics."
        )

    def run(
        self,
        data: pd.DataFrame,
        starting_capital: float = 100000.0,
        extra_fees: float = 0.0,
        verbose: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run track analysis on the provided data.

        Args:
            data: DataFrame with normalized and enriched trade data
            starting_capital: Starting capital for equity curve calculation (default: 100000)
            extra_fees: Monthly extra fees (e.g., automation costs) (default: 0.0)
            verbose: If True, print formatted output (default: True)
            output_dir: Directory to save visualizations (default: None)

        Returns:
            Dictionary with analysis results
        """
        # Validate required columns
        self.validate_data(data, ["P/L", "datetime_opened"])

        # Make a copy to avoid modifying original
        df = data.copy()

        # Sort by datetime
        if "datetime_opened" in df.columns:
            df = df.sort_values("datetime_opened").reset_index(drop=True)
        else:
            raise ValueError("Missing datetime information (datetime_opened)")

        results = {}

        # Calculate equity curve
        equity_curve_data = self._calculate_equity_curve(df, starting_capital)
        results["equity_curve"] = equity_curve_data

        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            df, equity_curve_data, starting_capital, extra_fees
        )
        results["portfolio_metrics"] = portfolio_metrics

        # Calculate drawdown analysis
        drawdown_analysis = self._calculate_drawdown_analysis(equity_curve_data)
        results["drawdown_analysis"] = drawdown_analysis

        # Calculate strategy statistics
        strategy_stats = self._calculate_strategy_stats(df)
        results["strategy_stats"] = strategy_stats

        # Generate calendars
        calendar_results = self._generate_calendars(df)
        results["calendars"] = calendar_results

        if verbose:
            self._print_results(results, output_dir)

        return results

    def _calculate_equity_curve(
        self, data: pd.DataFrame, starting_capital: float
    ) -> Dict:
        """
        Calculate equity curve from starting capital and cumulative P/L.

        Args:
            data: DataFrame with trade data sorted by datetime
            starting_capital: Starting capital amount

        Returns:
            Dictionary with equity curve data
        """
        if len(data) == 0:
            return {
                "equity_curve": [],
                "dates": [],
                "starting_capital": starting_capital,
                "ending_value": starting_capital,
            }

        equity_curve = []
        dates = []
        current_value = starting_capital

        for _, row in data.iterrows():
            pl = row.get("P/L", 0)
            current_value += pl
            equity_curve.append(current_value)
            dates.append(row["datetime_opened"])

        return {
            "equity_curve": equity_curve,
            "dates": dates,
            "starting_capital": starting_capital,
            "ending_value": current_value,
        }

    def _calculate_portfolio_metrics(
        self, data: pd.DataFrame, equity_curve_data: Dict, starting_capital: float, extra_fees: float = 0.0
    ) -> Dict:
        """
        Calculate portfolio-level metrics.

        Args:
            data: DataFrame with trade data
            equity_curve_data: Dictionary with equity curve data
            starting_capital: Starting capital amount
            extra_fees: Monthly extra fees (e.g., automation costs)

        Returns:
            Dictionary with portfolio metrics
        """
        # Net P/L
        net_pl = float(data["P/L"].sum())

        # Calculate total fees based on date range
        dates = equity_curve_data["dates"]
        total_fees = 0.0
        if len(dates) > 0 and extra_fees > 0:
            first_date = dates[0]
            last_date = dates[-1]
            # Calculate number of months (including partial months)
            # Count each month that has at least one trade
            if len(dates) > 0:
                # Get unique year-month combinations
                unique_months = set()
                for date in dates:
                    unique_months.add((date.year, date.month))
                num_months = len(unique_months)
                total_fees = extra_fees * num_months

        # Net P/L after fees
        net_pl_after_fees = net_pl - total_fees

        # Total Premium
        if "Premium" in data.columns:
            total_premium = float(data["Premium"].sum())
        elif "Initial Premium" in data.columns:
            total_premium = float(data["Initial Premium"].sum())
        else:
            total_premium = 0.0

        # PCR (Premium Capture Rate)
        pcr = (net_pl / total_premium * 100) if total_premium > 0 else 0.0

        # CAGR
        equity_curve = equity_curve_data["equity_curve"]
        ending_value = equity_curve_data["ending_value"]

        cagr = 0.0
        if len(dates) > 1:
            first_date = dates[0]
            last_date = dates[-1]
            years = (last_date - first_date).days / 365.25
            if years > 0 and starting_capital > 0:
                cagr = ((ending_value / starting_capital) ** (1 / years) - 1) * 100

        # Sharpe and Sortino ratios
        sharpe = 0.0
        sortino = 0.0
        if len(equity_curve) > 1:
            # Calculate daily returns
            returns = []
            for i in range(1, len(equity_curve)):
                if equity_curve[i - 1] > 0:
                    daily_return = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                    returns.append(daily_return)

            if len(returns) > 0:
                returns = np.array(returns)
                mean_return = np.mean(returns)
                std_return = np.std(returns)

                # Annualize (assuming daily data)
                if std_return > 0:
                    sharpe = (mean_return / std_return) * np.sqrt(252) * 100

                # Sortino: only use negative returns for downside deviation
                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_std = np.std(negative_returns)
                    if downside_std > 0:
                        sortino = (mean_return / downside_std) * np.sqrt(252) * 100

        return {
            "net_pl": net_pl,
            "total_fees": total_fees,
            "net_pl_after_fees": net_pl_after_fees,
            "extra_fees_monthly": extra_fees,
            "total_premium": total_premium,
            "pcr": pcr,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "starting_capital": starting_capital,
            "ending_value": ending_value,
        }

    def _calculate_drawdown_analysis(self, equity_curve_data: Dict) -> Dict:
        """
        Calculate drawdown analysis from equity curve.

        Args:
            equity_curve_data: Dictionary with equity curve data

        Returns:
            Dictionary with drawdown analysis results
        """
        equity_curve = equity_curve_data["equity_curve"]
        dates = equity_curve_data["dates"]
        starting_capital = equity_curve_data["starting_capital"]

        if len(equity_curve) == 0:
            return {
                "max_drawdown_dollars": 0.0,
                "max_drawdown_pct": 0.0,
                "longest_drawdown": None,
                "shortest_drawdown": None,
                "num_drawdowns": 0,
                "average_drawdown_length": 0.0,
                "average_drawdown_depth": 0.0,
                "current_drawdown": None,
                "drawdown_periods": [],
                "drawdown_curve": [],
            }

        equity_array = np.array(equity_curve)

        # Calculate drawdown curve
        drawdown_curve = self._calculate_drawdown_curve(equity_array)

        # Find drawdown periods
        drawdown_periods = self._identify_drawdown_periods(equity_array, dates)

        # Calculate max drawdown
        max_dd_dollars = 0.0
        max_dd_pct = 0.0
        peak = starting_capital
        tol = 1e-9

        for value in equity_array:
            if value >= peak - tol:
                peak = max(peak, value)
            drawdown = peak - value
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0.0
            if drawdown > max_dd_dollars:
                max_dd_dollars = drawdown
            if drawdown_pct > max_dd_pct:
                max_dd_pct = drawdown_pct

        # Calculate longest and shortest drawdown
        longest_drawdown = None
        shortest_drawdown = None
        num_drawdowns = len(drawdown_periods)
        
        if drawdown_periods:
            # Calculate depths for each period
            for period in drawdown_periods:
                start_idx = dates.index(period["start_date"])
                end_idx = dates.index(period["end_date"])
                period_values = equity_array[start_idx:end_idx + 1]
                peak_in_period = np.max(equity_array[:end_idx + 1])
                min_in_period = np.min(period_values)
                depth_dollars = peak_in_period - min_in_period
                depth_pct = (depth_dollars / peak_in_period * 100) if peak_in_period > 0 else 0.0
                period["depth_dollars"] = depth_dollars
                period["depth_pct"] = depth_pct

            longest = max(drawdown_periods, key=lambda x: (x["end_date"] - x["start_date"]).days)
            longest_drawdown = {
                "start_date": longest["start_date"],
                "end_date": longest["end_date"],
                "length_days": (longest["end_date"] - longest["start_date"]).days,
                "depth_dollars": longest.get("depth_dollars", 0.0),
                "depth_pct": longest.get("depth_pct", 0.0),
            }

            non_zero_drawdowns = [
                period for period in drawdown_periods
                if (period["end_date"] - period["start_date"]).days > 0
            ]
            if non_zero_drawdowns:
                shortest = min(non_zero_drawdowns, key=lambda x: (x["end_date"] - x["start_date"]).days)
                shortest_drawdown = {
                    "start_date": shortest["start_date"],
                    "end_date": shortest["end_date"],
                    "length_days": (shortest["end_date"] - shortest["start_date"]).days,
                    "depth_dollars": shortest.get("depth_dollars", 0.0),
                    "depth_pct": shortest.get("depth_pct", 0.0),
                }

        # Calculate average drawdown length and depth
        avg_length = 0.0
        avg_depth = 0.0
        if drawdown_periods:
            lengths = [
                (period["end_date"] - period["start_date"]).days
                for period in drawdown_periods
            ]
            depths = [period.get("depth_pct", 0.0) for period in drawdown_periods]
            avg_length = np.mean(lengths)
            avg_depth = np.mean(depths) if depths else 0.0

        # Current drawdown
        current_drawdown = None
        if len(equity_array) > 0:
            current_value = equity_array[-1]
            peak = np.max(equity_array)
            if current_value < peak - tol:
                current_dd_dollars = peak - current_value
                current_dd_pct = (current_dd_dollars / peak * 100) if peak > 0 else 0.0
                # Find when current drawdown started
                current_peak_idx = np.argmax(equity_array)
                if current_peak_idx < len(dates):
                    current_drawdown = {
                        "start_date": dates[current_peak_idx],
                        "current_date": dates[-1],
                        "length_days": (dates[-1] - dates[current_peak_idx]).days,
                        "depth_dollars": current_dd_dollars,
                        "depth_pct": current_dd_pct,
                    }

        return {
            "max_drawdown_dollars": max_dd_dollars,
            "max_drawdown_pct": max_dd_pct,
            "longest_drawdown": longest_drawdown,
            "shortest_drawdown": shortest_drawdown,
            "num_drawdowns": num_drawdowns,
            "average_drawdown_length": avg_length,
            "average_drawdown_depth": avg_depth,
            "current_drawdown": current_drawdown,
            "drawdown_periods": drawdown_periods,
            "drawdown_curve": drawdown_curve.tolist(),
        }

    def _calculate_drawdown_curve(self, equity_curve: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown curve from equity curve.

        Args:
            equity_curve: Array of portfolio values over time

        Returns:
            Array of drawdown values (as percentages)
        """
        if len(equity_curve) == 0:
            return np.array([])

        drawdowns = []
        peak = equity_curve[0]
        tol = 1e-9

        for value in equity_curve:
            if value >= peak - tol:
                peak = max(peak, value)
            drawdown_pct = ((peak - value) / peak * 100) if peak > 0 else 0.0
            drawdowns.append(drawdown_pct)

        return np.array(drawdowns)

    def _identify_drawdown_periods(
        self, equity_curve: np.ndarray, dates: List[datetime]
    ) -> List[Dict]:
        """
        Identify all drawdown periods.

        Args:
            equity_curve: Array of portfolio values over time
            dates: List of dates corresponding to values

        Returns:
            List of drawdown period dictionaries with start_date, end_date
        """
        if len(equity_curve) == 0:
            return []

        periods = []
        peak = equity_curve[0]
        tol = 1e-9
        in_drawdown = False
        drawdown_start = None

        for i, value in enumerate(equity_curve):
            if value >= peak - tol:
                if in_drawdown:
                    periods.append(
                        {
                            "start_date": drawdown_start,
                            "end_date": dates[i],
                        }
                    )
                    in_drawdown = False
                peak = max(peak, value)
            elif value < peak:
                if not in_drawdown:
                    drawdown_start = dates[i]
                    in_drawdown = True

        # Handle case where drawdown continues to end
        if in_drawdown and len(dates) > 0:
            periods.append({
                "start_date": drawdown_start,
                "end_date": dates[-1],
            })

        return periods

    def _calculate_strategy_stats(self, data: pd.DataFrame) -> List[Dict]:
        """
        Calculate statistics for each strategy.

        Args:
            data: DataFrame with trade data

        Returns:
            List of dictionaries with strategy statistics
        """
        if "Strategy" not in data.columns:
            return []

        strategy_stats = []
        today = datetime.now()

        for strategy in data["Strategy"].unique():
            strategy_data = data[data["Strategy"] == strategy].copy()

            # Number of trades
            num_trades = len(strategy_data)

            # Net P/L
            net_pl = float(strategy_data["P/L"].sum())

            # Average P/L per contract
            if "P/L per Contract" in strategy_data.columns:
                avg_pl_per_contract = float(strategy_data["P/L per Contract"].mean())
            else:
                contracts = strategy_data.get("No. of Contracts", 1)
                avg_pl_per_contract = float((strategy_data["P/L"] / contracts).mean())

            # Average P/L per trade
            avg_pl_per_trade = float(strategy_data["P/L"].mean())

            # Average Win per contract (only winning trades)
            winning_trades = strategy_data[strategy_data["P/L"] > 0]
            if len(winning_trades) > 0:
                if "P/L per Contract" in winning_trades.columns:
                    avg_win_per_contract = float(winning_trades["P/L per Contract"].mean())
                else:
                    contracts = winning_trades.get("No. of Contracts", 1)
                    avg_win_per_contract = float((winning_trades["P/L"] / contracts).mean())
            else:
                avg_win_per_contract = 0.0

            # Average Loss per contract (only losing trades)
            losing_trades = strategy_data[strategy_data["P/L"] < 0]
            if len(losing_trades) > 0:
                if "P/L per Contract" in losing_trades.columns:
                    avg_loss_per_contract = float(losing_trades["P/L per Contract"].mean())
                else:
                    contracts = losing_trades.get("No. of Contracts", 1)
                    avg_loss_per_contract = float((losing_trades["P/L"] / contracts).mean())
            else:
                avg_loss_per_contract = 0.0

            # Win rate
            wins = len(strategy_data[strategy_data["P/L"] > 0])
            win_rate = (wins / num_trades * 100) if num_trades > 0 else 0.0

            # PCR (strategy-level)
            if "Premium" in strategy_data.columns:
                strategy_premium = float(strategy_data["Premium"].sum())
            elif "Initial Premium" in strategy_data.columns:
                strategy_premium = float(strategy_data["Initial Premium"].sum())
            else:
                strategy_premium = 0.0

            strategy_pcr = (net_pl / strategy_premium * 100) if strategy_premium > 0 else 0.0

            # Last trade date and days since
            last_trade_date = strategy_data["datetime_opened"].max()
            days_since = (today - last_trade_date).days if pd.notna(last_trade_date) else None

            strategy_stats.append({
                "strategy": str(strategy),
                "num_trades": num_trades,
                "net_pl": net_pl,
                "avg_pl_per_contract": avg_pl_per_contract,
                "avg_pl_per_trade": avg_pl_per_trade,
                "avg_win_per_contract": avg_win_per_contract,
                "avg_loss_per_contract": avg_loss_per_contract,
                "win_rate": win_rate,
                "pcr": strategy_pcr,
                "last_trade_date": last_trade_date,
                "days_since": days_since,
            })

        # Sort by most recent to oldest (by last_trade_date)
        strategy_stats.sort(key=lambda x: x["last_trade_date"] if x["last_trade_date"] is not None else datetime.min, reverse=True)

        return strategy_stats

    def _generate_calendars(self, data: pd.DataFrame) -> Dict:
        """
        Generate calendar data for weekly and monthly profitability.

        Args:
            data: DataFrame with trade data

        Returns:
            Dictionary with weekly and monthly calendar data
        """
        if len(data) == 0:
            return {"weekly": {}, "monthly": {}}

        # Group by week
        data["week"] = data["datetime_opened"].dt.to_period("W")
        weekly_pl = data.groupby("week")["P/L"].sum()

        # Group by month
        data["month"] = data["datetime_opened"].dt.to_period("M")
        monthly_pl = data.groupby("month")["P/L"].sum()

        weekly_calendar = {
            f"{week.start_time.isocalendar()[0]}-W{week.start_time.isocalendar()[1]:02d}": {"pl": pl, "profitable": pl > 0}
            for week, pl in weekly_pl.items()
        }

        monthly_calendar = {
            str(month): {"pl": pl, "profitable": pl > 0}
            for month, pl in monthly_pl.items()
        }

        return {
            "weekly": weekly_calendar,
            "monthly": monthly_calendar,
        }

    def _print_results(
        self,
        results: Dict,
        output_dir: Optional[str] = None,
    ) -> None:
        """Print formatted results."""
        portfolio_metrics = results["portfolio_metrics"]
        drawdown_analysis = results["drawdown_analysis"]
        equity_curve_data = results["equity_curve"]
        strategy_stats = results["strategy_stats"]

        # Print portfolio metrics
        print_track_metrics(portfolio_metrics, drawdown_analysis)

        # Print equity curve chart
        if equity_curve_data.get("equity_curve") and equity_curve_data.get("dates"):
            create_track_chart(
                equity_curve_data["equity_curve"],
                drawdown_analysis["drawdown_curve"],
                equity_curve_data["dates"],
                "Portfolio Track Analysis",
                output_dir,
            )

        # ASCII graphs for CLI visibility
        if equity_curve_data.get("equity_curve"):
            print_ascii_timeseries(
                equity_curve_data["equity_curve"],
                title="EQUITY_CURVE_ASCII",
                width=70,
                height=12,
                is_percentage=False,
                line_color=Colors.ACCENT,
            )
        if drawdown_analysis.get("drawdown_curve"):
            print_ascii_timeseries(
                drawdown_analysis["drawdown_curve"],
                title="DRAWDOWN_ASCII_(%)",
                width=70,
                height=12,
                is_percentage=True,
                line_color=Colors.BRIGHT_RED,
            )

        # Print drawdown analysis
        print_track_drawdown_analysis(drawdown_analysis)

        # Print strategy table
        print_track_strategy_table(strategy_stats)

        # Print calendars
        print_drawdown_calendar(results["calendars"])
