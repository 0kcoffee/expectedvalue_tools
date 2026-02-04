"""Drawdown analysis test for trading strategies."""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from .base import BaseTest
from ..output.formatters import (
    print_box,
    print_section_box,
    print_ascii_distribution,
    print_margin_check,
    print_drawdown_metrics,
    print_drawdown_calendar,
    print_biggest_loss,
    print_ascii_timeseries,
)
from ..output.visualizers import create_drawdown_chart
from ..utils.colors import Colors
from ..utils.dynamic_allocation import DynamicAllocationSimulator


class DrawdownAnalysisTest(BaseTest):
    """Test that analyzes drawdowns for trading strategies."""

    def get_name(self) -> str:
        """Get the name of the test."""
        return "drawdown"

    def get_description(self) -> str:
        """Get a description of what the test does."""
        return (
            "Analyzes drawdowns for trading strategies including margin requirement validation, "
            "maximum drawdown, drawdown periods, and calendar visualizations of profitable/losing periods."
        )

    def run(
        self,
        data: pd.DataFrame,
        portfolio_size: float = 100000.0,
        desired_allocation_pct: float = 1.0,
        strategy_allocations: Optional[Dict[str, float]] = None,
        force_one_lot: bool = False,
        verbose: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run drawdown analysis on the provided data.

        Args:
            data: DataFrame with normalized and enriched trade data
            portfolio_size: Initial portfolio size (default: 100000)
            desired_allocation_pct: Desired allocation percentage (default: 1.0)
            strategy_allocations: Optional dict mapping strategy names to allocation percentages
            verbose: If True, print formatted output (default: True)

        Returns:
            Dictionary with analysis results
        """
        # Validate required columns
        self.validate_data(data, ["P/L", "No. of Contracts", "datetime_opened"])

        # Make a copy to avoid modifying original
        df = data.copy()

        # Sort by datetime
        if "datetime_opened" in df.columns:
            df = df.sort_values("datetime_opened").reset_index(drop=True)
        elif "Date Opened" in df.columns:
            df["datetime_opened"] = pd.to_datetime(df["Date Opened"], errors="coerce")
            df = df.sort_values("datetime_opened").reset_index(drop=True)
        else:
            raise ValueError("Missing datetime information (datetime_opened or Date Opened)")

        # Use strategy-specific allocation if provided, otherwise use default
        allocation_pct = desired_allocation_pct
        if strategy_allocations and len(df) > 0:
            strategy = df.iloc[0].get("Strategy", "")
            if strategy in strategy_allocations:
                allocation_pct = strategy_allocations[strategy]

        results = {}

        # Perform margin check (backtest data only)
        margin_check_results = self._check_margin_requirements(
            df, portfolio_size, allocation_pct
        )
        results["margin_check"] = margin_check_results

        # Calculate drawdowns with proper position sizing
        drawdown_results = self._calculate_drawdowns(
            df, portfolio_size, allocation_pct, margin_check_results, force_one_lot
        )
        results["drawdowns"] = drawdown_results

        # Generate calendars
        calendar_results = self._generate_calendars(df)
        results["calendars"] = calendar_results

        if verbose:
            self._print_results(results, df, portfolio_size, allocation_pct, output_dir)

        return results

    def _check_margin_requirements(
        self, data: pd.DataFrame, portfolio_size: float, desired_allocation_pct: float
    ) -> Dict:
        """
        Check margin requirements against desired allocation.

        Args:
            data: DataFrame with trade data
            portfolio_size: Initial portfolio size
            desired_allocation_pct: Desired allocation percentage

        Returns:
            Dictionary with margin check results
        """
        results = {
            "has_margin_data": False,
            "mean_allocation_pct": 0.0,
            "max_allocation_pct": 0.0,
            "std_allocation_pct": 0.0,
            "mean_margin_per_contract": 0.0,
            "max_margin_per_contract": 0.0,
            "allocations": [],
            "biggest_loss_per_contract": 0.0,
            "biggest_loss_pct_of_portfolio": 0.0,
        }

        # Only check for backtest data (has "Margin Req." column)
        if "Margin Req." not in data.columns:
            return results

        results["has_margin_data"] = True

        # Calculate margin per contract and allocation for each trade
        allocations = []
        margins_per_contract = []
        for _, row in data.iterrows():
            margin_req = row.get("Margin Req.", 0)
            contracts = row.get("No. of Contracts", 1)
            
            if margin_req > 0 and contracts > 0:
                margin_per_contract = margin_req / contracts
                margins_per_contract.append(margin_per_contract)
                allocation_pct = (margin_per_contract * 1) / portfolio_size * 100
                allocations.append(allocation_pct)

        if allocations:
            results["allocations"] = allocations
            results["mean_allocation_pct"] = np.mean(allocations)
            results["max_allocation_pct"] = np.max(allocations)
            results["std_allocation_pct"] = np.std(allocations)
        
        if margins_per_contract:
            results["mean_margin_per_contract"] = np.mean(margins_per_contract)
            results["max_margin_per_contract"] = np.max(margins_per_contract)

        # Find biggest loss per contract
        if "P/L per Contract" in data.columns:
            pl_per_contract = data["P/L per Contract"].values
            losses = pl_per_contract[pl_per_contract < 0]
            if len(losses) > 0:
                biggest_loss = np.min(losses)  # Most negative
                results["biggest_loss_per_contract"] = biggest_loss
                results["biggest_loss_pct_of_portfolio"] = (
                    abs(biggest_loss) / portfolio_size * 100
                )

        return results

    def _calculate_drawdowns(
        self,
        data: pd.DataFrame,
        portfolio_size: float,
        allocation_pct: float,
        margin_check: Dict,
        force_one_lot: bool = False,
    ) -> Dict:
        """
        Calculate drawdown metrics from portfolio value over time with proper position sizing.

        Args:
            data: DataFrame with trade data sorted by datetime
            portfolio_size: Initial portfolio size
            allocation_pct: Desired allocation percentage
            margin_check: Dictionary with margin check results
            force_one_lot: If True, force at least 1 contract even when allocation is insufficient

        Returns:
            Dictionary with drawdown metrics
        """
        if len(data) == 0:
            return {
                "max_drawdown_dollars": 0.0,
                "max_drawdown_pct": 0.0,
                "longest_drawdown": None,
                "shortest_drawdown": None,
                "num_drawdowns": 0,
                "average_drawdown_length": 0.0,
                "percent_time_in_drawdown": 0.0,
                "drawdown_periods": [],
                "equity_curve": [],
                "drawdown_curve": [],
                "dates": [],
            }

        # Calculate portfolio value over time with proper position sizing
        portfolio_values, dates, equity_data = self._calculate_portfolio_value(
            data, portfolio_size, allocation_pct, margin_check, force_one_lot
        )

        # Calculate drawdown curve
        drawdown_curve = self._calculate_drawdown_curve(portfolio_values)

        # Find drawdown periods
        drawdown_periods = self._identify_drawdown_periods(portfolio_values, dates)

        # Calculate metrics
        max_dd_dollars = 0.0
        max_dd_pct = 0.0
        # Use running max peak of the evolving equity curve
        peak = portfolio_size
        # Small tolerance so "touching" the prior peak counts as recovery (float/rounding safe)
        tol = 1e-9

        for i, value in enumerate(portfolio_values):
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
            # Filter out 0-day drawdowns for shortest calculation
            non_zero_drawdowns = [
                period for period in drawdown_periods
                if (period["end_date"] - period["start_date"]).days > 0
            ]
            
            longest = max(drawdown_periods, key=lambda x: (x["end_date"] - x["start_date"]).days)
            longest_drawdown = {
                "start_date": longest["start_date"],
                "end_date": longest["end_date"],
                "length_days": (longest["end_date"] - longest["start_date"]).days,
            }
            
            # Only calculate shortest if there are non-zero drawdowns
            if non_zero_drawdowns:
                shortest = min(non_zero_drawdowns, key=lambda x: (x["end_date"] - x["start_date"]).days)
                shortest_drawdown = {
                    "start_date": shortest["start_date"],
                    "end_date": shortest["end_date"],
                    "length_days": (shortest["end_date"] - shortest["start_date"]).days,
                }

        # Calculate average drawdown length
        avg_length = 0.0
        if drawdown_periods:
            lengths = [
                (period["end_date"] - period["start_date"]).days
                for period in drawdown_periods
            ]
            avg_length = np.mean(lengths)

        # Calculate percent time in drawdown
        total_days = 0
        drawdown_days = 0
        if len(dates) > 1:
            total_days = (dates[-1] - dates[0]).days
            for period in drawdown_periods:
                drawdown_days += (period["end_date"] - period["start_date"]).days

        percent_time = (drawdown_days / total_days * 100) if total_days > 0 else 0.0

        # Get over-allocation info from simulator result
        over_allocated_trades = equity_data.get("over_allocated_trades", 0)
        over_allocation_details = equity_data.get("over_allocation_details", [])

        return {
            "max_drawdown_dollars": max_dd_dollars,
            "max_drawdown_pct": max_dd_pct,
            "longest_drawdown": longest_drawdown,
            "shortest_drawdown": shortest_drawdown,
            "num_drawdowns": num_drawdowns,
            "average_drawdown_length": avg_length,
            "percent_time_in_drawdown": percent_time,
            "drawdown_periods": drawdown_periods,
            "equity_curve": portfolio_values.tolist(),
            "drawdown_curve": drawdown_curve.tolist(),
            "dates": dates,
            "over_allocated_trades": over_allocated_trades,
            "over_allocation_details": over_allocation_details,
        }

    def _calculate_portfolio_value(
        self,
        data: pd.DataFrame,
        portfolio_size: float,
        allocation_pct: float,
        margin_check: Dict,
        force_one_lot: bool = False,
    ) -> Tuple[np.ndarray, List[datetime], Dict]:
        """
        Calculate portfolio value over time with proper position sizing based on margin.
        
        Uses DynamicAllocationSimulator when possible. If "Used Allocation" column exists
        (from enriched backtest data), uses that for dynamic allocation per trade.

        Args:
            data: DataFrame with trade data sorted by datetime
            portfolio_size: Initial portfolio size
            allocation_pct: Desired allocation percentage (used if "Used Allocation" not available)
            margin_check: Dictionary with margin check results
            force_one_lot: If True, force at least 1 contract even when allocation is insufficient

        Returns:
            Tuple of (portfolio_values array, dates list, equity_data dict)
        """
        if len(data) == 0:
            return np.array([]), [], {}

        # Always use DynamicAllocationSimulator with the desired allocation percentage
        # This ensures the equity curve reflects the user's specified allocation
        return self._calculate_portfolio_value_with_simulator(
            data, portfolio_size, allocation_pct, margin_check, force_one_lot
        )

    def _calculate_portfolio_value_with_simulator(
        self,
        data: pd.DataFrame,
        portfolio_size: float,
        allocation_pct: float,
        margin_check: Dict,
        force_one_lot: bool = False,
    ) -> Tuple[np.ndarray, List[datetime], Dict]:
        """
        Calculate portfolio value using DynamicAllocationSimulator.

        Args:
            data: DataFrame with trade data
            portfolio_size: Initial portfolio size
            allocation_pct: Desired allocation percentage
            margin_check: Dictionary with margin check results

        Returns:
            Tuple of (portfolio_values array, dates list, equity_data dict)
        """
        # Check if this is a portfolio (multiple strategies)
        is_portfolio = False
        strategy_allocations = None
        if "Strategy" in data.columns:
            unique_strategies = data["Strategy"].dropna().unique()
            unique_strategies = [s for s in unique_strategies if str(s).strip() != ""]
            is_portfolio = len(unique_strategies) > 1
            
            # If portfolio, use same allocation for all strategies
            if is_portfolio:
                strategy_allocations = {strategy: allocation_pct for strategy in unique_strategies}

        # Use DynamicAllocationSimulator
        simulator = DynamicAllocationSimulator(portfolio_size, allocation_pct, force_one_lot=force_one_lot)
        result = simulator.simulate_equity_curve(data, strategy_allocations)

        portfolio_values = result["portfolio_values"]
        dates = result["dates"]
        equity_data = {
            "portfolio_values": portfolio_values,
            "dates": dates,
        }

        return np.array(portfolio_values), dates, equity_data

    def _calculate_drawdown_curve(self, portfolio_values: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown curve from portfolio values.

        Args:
            portfolio_values: Array of portfolio values over time

        Returns:
            Array of drawdown values (as percentages)
        """
        if len(portfolio_values) == 0:
            return np.array([])

        drawdowns = []
        peak = portfolio_values[0]
        tol = 1e-9

        for value in portfolio_values:
            if value >= peak - tol:
                peak = max(peak, value)
            drawdown_pct = ((peak - value) / peak * 100) if peak > 0 else 0.0
            drawdowns.append(drawdown_pct)

        return np.array(drawdowns)

    def _identify_drawdown_periods(
        self, portfolio_values: np.ndarray, dates: List[datetime]
    ) -> List[Dict]:
        """
        Identify all drawdown periods.

        Args:
            portfolio_values: Array of portfolio values over time
            dates: List of dates corresponding to values

        Returns:
            List of drawdown period dictionaries with start_date, end_date
        """
        if len(portfolio_values) == 0:
            return []

        periods = []
        peak = portfolio_values[0]
        tol = 1e-9
        in_drawdown = False
        drawdown_start = None

        for i, value in enumerate(portfolio_values):
            # Recovery happens when we reach (or slightly exceed, within tol) the prior peak.
            if value >= peak - tol:
                # New peak (or full recovery) reached
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
                # In drawdown
                if not in_drawdown:
                    # Start of drawdown period
                    drawdown_start = dates[i]
                    in_drawdown = True

        # Handle case where drawdown continues to end
        if in_drawdown and len(dates) > 0:
            periods.append({
                "start_date": drawdown_start,
                "end_date": dates[-1],
            })

        return periods

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
        data: pd.DataFrame,
        portfolio_size: float,
        allocation_pct: float,
        output_dir: Optional[str] = None,
    ) -> None:
        """Print formatted results."""
        # Print margin check
        if results["margin_check"]["has_margin_data"]:
            print_margin_check(
                results["margin_check"],
                portfolio_size,
                allocation_pct,
            )
            print_biggest_loss(results["margin_check"], portfolio_size)

        # Print drawdown chart after margin check
        if results["drawdowns"].get("dates"):
            create_drawdown_chart(
                results["drawdowns"]["equity_curve"],
                results["drawdowns"]["drawdown_curve"],
                results["drawdowns"]["dates"],
                "Drawdown Analysis",
                output_dir,
            )

        # ASCII graphs for CLI visibility (always available)
        if results.get("drawdowns", {}).get("equity_curve"):
            print_ascii_timeseries(
                results["drawdowns"]["equity_curve"],
                title="EQUITY_CURVE_ASCII",
                width=70,
                height=12,
                is_percentage=False,
                line_color=Colors.ACCENT,
            )
        if results.get("drawdowns", {}).get("drawdown_curve"):
            print_ascii_timeseries(
                results["drawdowns"]["drawdown_curve"],
                title="DRAWDOWN_ASCII_(%)",
                width=70,
                height=12,
                is_percentage=True,
                line_color=Colors.BRIGHT_RED,
            )

        # Print drawdown metrics
        print_drawdown_metrics(results["drawdowns"])

        # Print calendars
        print_drawdown_calendar(results["calendars"])
