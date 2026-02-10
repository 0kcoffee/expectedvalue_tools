"""Portfolio stress test using Monte Carlo simulations."""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from .base import BaseTest
from ..output.formatters import (
    print_box,
    print_section_box,
    print_ascii_distribution,
    print_ascii_timeseries,
)
from ..utils.colors import Colors
from ..utils.dynamic_allocation import DynamicAllocationSimulator

# Try to import matplotlib for charts
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mdates = None


class PortfolioStressTest(BaseTest):
    """Test that performs Monte Carlo stress testing on portfolio backtests."""

    def get_name(self) -> str:
        """Get the name of the test."""
        return "portfolio_stress"

    def get_description(self) -> str:
        """Get a description of what the test does."""
        return (
            "Performs Monte Carlo stress testing on portfolio backtests. "
            "Generates thousands of simulations by sampling with replacement from each strategy's trade distribution, "
            "combines them into portfolio equity curves using dynamic allocation, and analyzes worst-case scenarios "
            "including max drawdown and comprehensive risk metrics."
        )

    def run(
        self,
        data: pd.DataFrame,
        portfolio_size: float = 100000.0,
        allocation_pct: float = 1.0,
        strategy_allocations: Optional[Dict[str, float]] = None,
        simulations: int = 10000,
        force_one_lot: bool = False,
        verbose: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run portfolio stress test on the provided data.

        Args:
            data: DataFrame with normalized and enriched trade data
            portfolio_size: Initial portfolio size (default: 100000)
            allocation_pct: Default allocation percentage (default: 1.0)
            strategy_allocations: Optional dict mapping strategy names to allocation percentages
            simulations: Number of Monte Carlo simulations (default: 10000)
            force_one_lot: If True, force at least 1 contract even when allocation is insufficient
            verbose: If True, print formatted output (default: True)
            output_dir: Directory to save outputs (default: None)

        Returns:
            Dictionary with stress test results
        """
        # Validate required columns
        self.validate_data(data, ["P/L", "No. of Contracts", "datetime_opened", "Strategy"])

        # Make a copy to avoid modifying original
        df = data.copy()

        # Check if this is a portfolio (multiple strategies)
        if "Strategy" not in df.columns:
            raise ValueError("Portfolio stress test requires 'Strategy' column")

        unique_strategies = df["Strategy"].dropna().unique()
        unique_strategies = [s for s in unique_strategies if str(s).strip() != ""]

        if len(unique_strategies) < 1:
            raise ValueError("No valid strategies found in data")

        # Calculate average allocation per strategy from enriched data if available
        calculated_allocations = self._calculate_average_allocations(df, unique_strategies)
        
        # Determine allocation percentages for each strategy
        if strategy_allocations:
            allocation_map = strategy_allocations
        else:
            # Use calculated allocations if available, otherwise use default
            allocation_map = {}
            for strategy in unique_strategies:
                if strategy in calculated_allocations:
                    allocation_map[strategy] = calculated_allocations[strategy]
                else:
                    allocation_map[strategy] = allocation_pct

        # Split data by strategy
        strategy_data = {}
        for strategy in unique_strategies:
            strategy_df = df[df["Strategy"] == strategy].copy()
            if len(strategy_df) == 0:
                continue
            strategy_data[strategy] = strategy_df

        if len(strategy_data) == 0:
            raise ValueError("No valid strategy data found")

        if verbose:
            self._print_header(unique_strategies, simulations, portfolio_size, allocation_map, force_one_lot)

        # Run Monte Carlo simulations
        simulation_results = self._simulate_portfolio_paths(
            strategy_data,
            portfolio_size,
            allocation_map,
            simulations,
            force_one_lot,
            verbose,
        )

        # Aggregate metrics across all simulations
        aggregated_metrics = self._aggregate_metrics(simulation_results, portfolio_size)

        # Generate percentile curves chart
        percentile_figure = None
        if verbose:
            percentile_figure = self._print_percentile_curves(aggregated_metrics)

        results = {
            "simulation_results": simulation_results,
            "aggregated_metrics": aggregated_metrics,
            "portfolio_size": portfolio_size,
            "num_simulations": simulations,
            "strategies": unique_strategies,
            "percentile_figure": percentile_figure,  # Store figure for HTML report
        }

        if verbose:
            self._print_results(results, output_dir)

        return results

    def _calculate_average_allocations(
        self,
        data: pd.DataFrame,
        strategies: List[str],
    ) -> Dict[str, float]:
        """
        Calculate average allocation percentage per strategy from enriched data.

        Args:
            data: DataFrame with enriched trade data
            strategies: List of strategy names

        Returns:
            Dictionary mapping strategy names to average allocation percentages
        """
        allocations = {}
        
        # Check if we have the required columns for calculation
        if "Used Allocation" in data.columns:
            for strategy in strategies:
                strategy_df = data[data["Strategy"] == strategy]
                if len(strategy_df) > 0:
                    # Calculate mean of Used Allocation for this strategy
                    avg_allocation = strategy_df["Used Allocation"].mean()
                    # Filter out invalid values
                    if not (np.isnan(avg_allocation) or np.isinf(avg_allocation) or avg_allocation <= 0):
                        allocations[strategy] = float(avg_allocation)
        
        return allocations

    def _simulate_portfolio_paths(
        self,
        strategy_data: Dict[str, pd.DataFrame],
        portfolio_size: float,
        allocation_map: Dict[str, float],
        num_simulations: int,
        force_one_lot: bool,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Run Monte Carlo simulations for portfolio paths.

        Args:
            strategy_data: Dictionary mapping strategy names to their DataFrames
            portfolio_size: Initial portfolio size
            allocation_map: Dictionary mapping strategy names to allocation percentages
            num_simulations: Number of simulations to run
            force_one_lot: If True, force at least 1 contract

        Returns:
            List of simulation result dictionaries
        """
        simulation_results = []

        # Prepare strategy trade distributions
        strategy_distributions = {}
        strategy_trade_counts = {}
        for strategy, df in strategy_data.items():
            # Get P/L per Contract values
            pl_per_contract = df["P/L per Contract"].values
            pl_per_contract = pl_per_contract[~(np.isnan(pl_per_contract) | np.isinf(pl_per_contract))]
            
            if len(pl_per_contract) == 0:
                continue

            strategy_distributions[strategy] = pl_per_contract
            strategy_trade_counts[strategy] = len(df)

            # Store other required columns for simulation
            strategy_data[strategy] = {
                "pl_per_contract": pl_per_contract,
                "original_df": df,
                "trade_count": len(df),
            }

        if len(strategy_distributions) == 0:
            return []

        # Run simulations with progress bar
        if verbose:
            self._print_simulation_progress(0, num_simulations)
        
        for sim_idx in range(num_simulations):
            if verbose and (sim_idx + 1) % max(1, num_simulations // 100) == 0 or sim_idx == num_simulations - 1:
                self._print_simulation_progress(sim_idx + 1, num_simulations)
            # Sample trades for each strategy
            combined_trades = []

            for strategy, strategy_info in strategy_data.items():
                pl_dist = strategy_info["pl_per_contract"]
                trade_count = strategy_info["trade_count"]
                original_df = strategy_info["original_df"]

                # Sample with replacement, maintaining original count
                sampled_indices = np.random.choice(len(pl_dist), size=trade_count, replace=True)
                sampled_pl_per_contract = pl_dist[sampled_indices]

                # Create trade rows for this strategy
                for i, pl_per_contract in enumerate(sampled_pl_per_contract):
                    # Sample a random row from original data for other fields
                    original_idx = np.random.choice(len(original_df))
                    original_row = original_df.iloc[original_idx].copy()

                    # Update P/L per Contract
                    original_row["P/L per Contract"] = pl_per_contract
                    
                    # Update P/L based on contracts
                    if "No. of Contracts" in original_row:
                        contracts = original_row.get("No. of Contracts", 1)
                        original_row["P/L"] = pl_per_contract * contracts

                    # Generate random datetime within original date range for interleaving
                    if "datetime_opened" in original_df.columns:
                        min_date = original_df["datetime_opened"].min()
                        max_date = original_df["datetime_opened"].max()
                        time_range = (max_date - min_date).total_seconds()
                        random_offset = np.random.uniform(0, time_range)
                        original_row["datetime_opened"] = min_date + timedelta(seconds=random_offset)
                    else:
                        # Use original datetime if available
                        original_row["datetime_opened"] = original_row.get("datetime_opened", datetime.now())

                    combined_trades.append(original_row)

            # Convert to DataFrame and sort by datetime
            combined_df = pd.DataFrame(combined_trades)
            if "datetime_opened" in combined_df.columns:
                combined_df = combined_df.sort_values("datetime_opened").reset_index(drop=True)

            # Track total trades before any are skipped
            total_trades_in_simulation = len(combined_df)

            # Generate equity curve using DynamicAllocationSimulator
            allocation_pct = allocation_map.get(combined_df.iloc[0].get("Strategy", ""), 1.0)
            simulator = DynamicAllocationSimulator(portfolio_size, allocation_pct, force_one_lot=force_one_lot)
            
            # Use strategy-specific allocations if portfolio
            is_portfolio = len(strategy_data) > 1
            strategy_allocations_for_sim = allocation_map if is_portfolio else None
            
            equity_result = simulator.simulate_equity_curve(combined_df, strategy_allocations_for_sim)

            # Calculate metrics for this simulation
            metrics = self._calculate_simulation_metrics(
                equity_result["equity_curve"],
                equity_result["dates"],
                portfolio_size,
            )

            simulation_results.append({
                "simulation_idx": sim_idx,
                "equity_curve": equity_result["equity_curve"],
                "dates": equity_result["dates"],
                "final_portfolio": equity_result["final_portfolio"],
                "total_return": equity_result["total_return"],
                "metrics": metrics,
                "total_trades": total_trades_in_simulation,
                "skipped_trades": equity_result.get("skipped_trades", 0),
                "actual_allocations": equity_result.get("actual_allocations", []),
                "target_allocations": equity_result.get("target_allocations", []),
            })

        return simulation_results

    def _calculate_simulation_metrics(
        self,
        equity_curve: np.ndarray,
        dates: List[datetime],
        starting_capital: float,
    ) -> Dict:
        """
        Calculate metrics for a single simulation.

        Args:
            equity_curve: Array of portfolio values over time
            dates: List of datetime objects
            starting_capital: Starting portfolio value

        Returns:
            Dictionary with calculated metrics
        """
        if len(equity_curve) == 0:
            return {
                "max_drawdown_dollars": 0.0,
                "max_drawdown_pct": 0.0,
                "final_portfolio": starting_capital,
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "num_drawdowns": 0,
                "longest_drawdown_days": 0,
            }

        equity_array = np.array(equity_curve)

        # Calculate max drawdown
        max_dd_dollars, max_dd_pct = self._calculate_max_drawdown(equity_array, starting_capital)

        # Final portfolio and total return
        final_portfolio = float(equity_array[-1])
        total_return = ((final_portfolio - starting_capital) / starting_capital * 100) if starting_capital > 0 else 0.0

        # Calculate CAGR
        cagr = 0.0
        if len(dates) > 1 and starting_capital > 0:
            first_date = dates[0]
            last_date = dates[-1]
            years = (last_date - first_date).days / 365.25
            if years > 0:
                cagr = ((final_portfolio / starting_capital) ** (1 / years) - 1) * 100

        # Calculate Sharpe and Sortino ratios
        sharpe = 0.0
        sortino = 0.0
        if len(equity_array) > 1:
            returns = []
            for i in range(1, len(equity_array)):
                if equity_array[i - 1] > 0:
                    daily_return = (equity_array[i] - equity_array[i - 1]) / equity_array[i - 1]
                    returns.append(daily_return)

            if len(returns) > 0:
                returns = np.array(returns)
                mean_return = np.mean(returns)
                std_return = np.std(returns)

                if std_return > 0:
                    sharpe = (mean_return / std_return) * np.sqrt(252) * 100

                negative_returns = returns[returns < 0]
                if len(negative_returns) > 0:
                    downside_std = np.std(negative_returns)
                    if downside_std > 0:
                        sortino = (mean_return / downside_std) * np.sqrt(252) * 100

        # Calculate drawdown periods
        drawdown_periods = self._identify_drawdown_periods(equity_array, dates)
        num_drawdowns = len(drawdown_periods)
        longest_drawdown_days = 0
        if drawdown_periods:
            longest = max(drawdown_periods, key=lambda x: (x["end_date"] - x["start_date"]).days)
            longest_drawdown_days = (longest["end_date"] - longest["start_date"]).days

        return {
            "max_drawdown_dollars": max_dd_dollars,
            "max_drawdown_pct": max_dd_pct,
            "final_portfolio": final_portfolio,
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "num_drawdowns": num_drawdowns,
            "longest_drawdown_days": longest_drawdown_days,
        }

    def _calculate_max_drawdown(self, equity_curve: np.ndarray, starting_capital: float) -> Tuple[float, float]:
        """Calculate maximum drawdown in dollars and percentage."""
        if len(equity_curve) == 0:
            return 0.0, 0.0

        max_dd_dollars = 0.0
        max_dd_pct = 0.0
        peak = starting_capital
        tol = 1e-9

        for value in equity_curve:
            if value >= peak - tol:
                peak = max(peak, value)
            drawdown = peak - value
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0.0
            if drawdown > max_dd_dollars:
                max_dd_dollars = drawdown
            if drawdown_pct > max_dd_pct:
                max_dd_pct = drawdown_pct

        return max_dd_dollars, max_dd_pct

    def _identify_drawdown_periods(
        self, equity_curve: np.ndarray, dates: List[datetime]
    ) -> List[Dict]:
        """Identify all drawdown periods."""
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
                    periods.append({
                        "start_date": drawdown_start,
                        "end_date": dates[i],
                    })
                    in_drawdown = False
                peak = max(peak, value)
            elif value < peak:
                if not in_drawdown:
                    drawdown_start = dates[i]
                    in_drawdown = True

        if in_drawdown and len(dates) > 0:
            periods.append({
                "start_date": drawdown_start,
                "end_date": dates[-1],
            })

        return periods

    def _aggregate_metrics(
        self,
        simulation_results: List[Dict],
        starting_capital: float,
    ) -> Dict:
        """
        Aggregate metrics across all simulations.

        Args:
            simulation_results: List of simulation result dictionaries
            starting_capital: Starting portfolio value

        Returns:
            Dictionary with aggregated metrics
        """
        if len(simulation_results) == 0:
            return {}

        # Extract all metrics
        max_dd_dollars = [r["metrics"]["max_drawdown_dollars"] for r in simulation_results]
        max_dd_pct = [r["metrics"]["max_drawdown_pct"] for r in simulation_results]
        final_portfolios = [r["final_portfolio"] for r in simulation_results]
        total_returns = [r["total_return"] for r in simulation_results]
        cagrs = [r["metrics"]["cagr"] for r in simulation_results]
        sharpes = [r["metrics"]["sharpe"] for r in simulation_results]
        sortinos = [r["metrics"]["sortino"] for r in simulation_results]
        num_drawdowns = [r["metrics"]["num_drawdowns"] for r in simulation_results]
        longest_drawdown_days = [r["metrics"]["longest_drawdown_days"] for r in simulation_results]
        total_trades = [r.get("total_trades", 0) for r in simulation_results]
        skipped_trades = [r.get("skipped_trades", 0) for r in simulation_results]
        
        # Collect all actual and target allocations (for overall stats)
        all_actual_allocations = []
        all_target_allocations = []
        # Collect allocations per strategy
        actual_allocations_by_strategy = {}
        for r in simulation_results:
            all_actual_allocations.extend(r.get("actual_allocations", []))
            all_target_allocations.extend(r.get("target_allocations", []))
            # Collect per-strategy allocations
            strategy_allocations = r.get("actual_allocations_by_strategy", {})
            for strategy, allocations in strategy_allocations.items():
                if strategy not in actual_allocations_by_strategy:
                    actual_allocations_by_strategy[strategy] = []
                actual_allocations_by_strategy[strategy].extend(allocations)

        # Convert to numpy arrays
        max_dd_dollars = np.array(max_dd_dollars)
        max_dd_pct = np.array(max_dd_pct)
        final_portfolios = np.array(final_portfolios)
        total_returns = np.array(total_returns)
        cagrs = np.array(cagrs)
        sharpes = np.array(sharpes)
        sortinos = np.array(sortinos)
        num_drawdowns = np.array(num_drawdowns)
        longest_drawdown_days = np.array(longest_drawdown_days)
        total_trades = np.array(total_trades)
        skipped_trades = np.array(skipped_trades)
        all_actual_allocations = np.array(all_actual_allocations) if len(all_actual_allocations) > 0 else np.array([])
        all_target_allocations = np.array(all_target_allocations) if len(all_target_allocations) > 0 else np.array([])

        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]

        def calc_percentiles(arr):
            return {f"p{p}": float(np.percentile(arr, p)) for p in percentiles}

        return {
            "worst_max_drawdown_dollars": float(np.max(max_dd_dollars)),
            "worst_max_drawdown_pct": float(np.max(max_dd_pct)),
            "worst_final_portfolio": float(np.min(final_portfolios)),
            "worst_total_return": float(np.min(total_returns)),
            "worst_cagr": float(np.min(cagrs)),
            "worst_sharpe": float(np.min(sharpes)),
            "worst_sortino": float(np.min(sortinos)),
            "worst_longest_drawdown_days": int(np.max(longest_drawdown_days)),
            "mean_max_drawdown_dollars": float(np.mean(max_dd_dollars)),
            "mean_max_drawdown_pct": float(np.mean(max_dd_pct)),
            "mean_final_portfolio": float(np.mean(final_portfolios)),
            "mean_total_return": float(np.mean(total_returns)),
            "mean_cagr": float(np.mean(cagrs)),
            "mean_sharpe": float(np.mean(sharpes)),
            "mean_sortino": float(np.mean(sortinos)),
            "std_max_drawdown_dollars": float(np.std(max_dd_dollars)),
            "std_max_drawdown_pct": float(np.std(max_dd_pct)),
            "std_final_portfolio": float(np.std(final_portfolios)),
            "std_total_return": float(np.std(total_returns)),
            "std_cagr": float(np.std(cagrs)),
            "max_drawdown_percentiles": calc_percentiles(max_dd_dollars),
            "max_drawdown_pct_percentiles": calc_percentiles(max_dd_pct),
            "final_portfolio_percentiles": calc_percentiles(final_portfolios),
            "total_return_percentiles": calc_percentiles(total_returns),
            "cagr_percentiles": calc_percentiles(cagrs),
            "sharpe_percentiles": calc_percentiles(sharpes),
            "sortino_percentiles": calc_percentiles(sortinos),
            # Total trades statistics
            "total_trades_all_simulations": int(np.sum(total_trades)),
            "mean_trades_per_sim": float(np.mean(total_trades)),
            "min_trades_per_sim": int(np.min(total_trades)),
            "max_trades_per_sim": int(np.max(total_trades)),
            # Skipped trades statistics
            "total_skipped_trades": int(np.sum(skipped_trades)),
            "mean_skipped_trades_per_sim": float(np.mean(skipped_trades)),
            "max_skipped_trades_per_sim": int(np.max(skipped_trades)),
            "skipped_trades_percentiles": calc_percentiles(skipped_trades) if len(skipped_trades) > 0 else {},
            # Actual vs target allocation statistics
            "mean_actual_allocation": float(np.mean(all_actual_allocations)) if len(all_actual_allocations) > 0 else 0.0,
            "mean_target_allocation": float(np.mean(all_target_allocations)) if len(all_target_allocations) > 0 else 0.0,
            "std_actual_allocation": float(np.std(all_actual_allocations)) if len(all_actual_allocations) > 0 else 0.0,
            "std_target_allocation": float(np.std(all_target_allocations)) if len(all_target_allocations) > 0 else 0.0,
            # Store raw arrays for visualization
            "_max_drawdown_dollars": max_dd_dollars,
            "_max_drawdown_pct": max_dd_pct,
            "_final_portfolios": final_portfolios,
            "_total_returns": total_returns,
            "_cagrs": cagrs,
            "_actual_allocations": all_actual_allocations,
            "_target_allocations": all_target_allocations,
            "_actual_allocations_by_strategy": actual_allocations_by_strategy,
            "_simulation_results": simulation_results,  # Store for percentile curve extraction
            "_starting_capital": starting_capital,
        }

    def _print_header(
        self,
        strategies: List[str],
        simulations: int,
        portfolio_size: float,
        allocation_map: Dict[str, float],
        force_one_lot: bool = False,
    ) -> None:
        """Print test header information."""
        box_width = 80
        lines = [
            ("Number of Strategies:", f"{len(strategies)}", Colors.BRIGHT_CYAN),
            ("Strategies:", ", ".join(strategies), Colors.BRIGHT_WHITE),
            ("Monte Carlo Simulations:", f"{simulations:,}", Colors.BRIGHT_CYAN),
            ("Portfolio Size:", f"${portfolio_size:,.2f}", Colors.BRIGHT_GREEN),
        ]
        
        # Add force_one_lot info
        if force_one_lot:
            lines.append(("Force One-Lot:", "Enabled (trades exceeding allocation will be forced to 1-lot)", Colors.BRIGHT_YELLOW))
        else:
            lines.append(("Force One-Lot:", "Disabled (trades exceeding allocation will be skipped)", Colors.BRIGHT_CYAN))
        
        # Add allocation info
        if len(set(allocation_map.values())) == 1:
            # All strategies have same allocation
            lines.append(("Allocation (All Strategies):", f"{list(allocation_map.values())[0]:.2f}%", Colors.BRIGHT_YELLOW))
        else:
            # Different allocations per strategy
            for strategy, alloc in allocation_map.items():
                lines.append((f"Allocation ({strategy}):", f"{alloc:.2f}%", Colors.BRIGHT_YELLOW))

        print_box(box_width, "PORTFOLIO STRESS TEST", lines)

    def _print_results(
        self,
        results: Dict,
        output_dir: Optional[str] = None,
    ) -> None:
        """Print formatted results."""
        aggregated = results["aggregated_metrics"]
        
        # Worst-case scenario box
        self._print_worst_case_scenario(aggregated)
        
        # Distribution statistics
        self._print_distribution_stats(aggregated)
        
        # Allocation and skipped trades statistics
        self._print_allocation_stats(aggregated)
        
        # Visual distributions
        self._print_distributions(aggregated)
        
        # Percentile analysis
        self._print_percentile_analysis(aggregated)
        
        # Percentile equity and drawdown curves
        self._print_percentile_curves(aggregated)

    def _print_worst_case_scenario(self, aggregated: Dict) -> None:
        """Print worst-case scenario metrics."""
        box_width = 80
        lines = [
            ("Worst Max Drawdown:", f"${aggregated.get('worst_max_drawdown_dollars', 0):,.2f}", Colors.BRIGHT_RED),
            ("Worst Max Drawdown (%):", f"{aggregated.get('worst_max_drawdown_pct', 0):.2f}%", Colors.BRIGHT_RED),
            ("Worst Final Portfolio:", f"${aggregated.get('worst_final_portfolio', 0):,.2f}", Colors.BRIGHT_RED),
            ("Worst Total Return:", f"{aggregated.get('worst_total_return', 0):.2f}%", Colors.BRIGHT_RED),
            ("Worst CAGR:", f"{aggregated.get('worst_cagr', 0):.2f}%", Colors.BRIGHT_RED),
            ("Worst Sharpe Ratio:", f"{aggregated.get('worst_sharpe', 0):.2f}", Colors.BRIGHT_RED),
            ("Worst Longest Drawdown:", f"{aggregated.get('worst_longest_drawdown_days', 0)} days", Colors.BRIGHT_RED),
        ]
        print_box(box_width, "WORST-CASE SCENARIO", lines)

    def _print_allocation_stats(self, aggregated: Dict) -> None:
        """Print allocation and skipped trades statistics."""
        box_width = 80
        lines = []
        
        # Total trades statistics
        total_trades_all = aggregated.get("total_trades_all_simulations", 0)
        mean_trades_per_sim = aggregated.get("mean_trades_per_sim", 0.0)
        min_trades_per_sim = aggregated.get("min_trades_per_sim", 0)
        max_trades_per_sim = aggregated.get("max_trades_per_sim", 0)
        
        lines.append(("Total Trades (All Simulations):", f"{total_trades_all:,}", Colors.BRIGHT_CYAN))
        lines.append(("Mean Trades per Simulation:", f"{mean_trades_per_sim:.2f}", Colors.BRIGHT_CYAN))
        if min_trades_per_sim != max_trades_per_sim:
            lines.append(("Trades per Simulation Range:", f"{min_trades_per_sim} - {max_trades_per_sim}", Colors.BRIGHT_CYAN))
        
        # Skipped trades statistics
        total_skipped = aggregated.get("total_skipped_trades", 0)
        mean_skipped = aggregated.get("mean_skipped_trades_per_sim", 0.0)
        max_skipped = aggregated.get("max_skipped_trades_per_sim", 0)
        
        if total_skipped > 0:
            lines.append(("Total Skipped Trades:", f"{total_skipped:,}", Colors.BRIGHT_YELLOW))
            lines.append(("Mean Skipped per Simulation:", f"{mean_skipped:.2f}", Colors.BRIGHT_YELLOW))
            lines.append(("Max Skipped in Single Simulation:", f"{max_skipped}", Colors.BRIGHT_RED))
            # Calculate skip rate
            if total_trades_all > 0:
                skip_rate = (total_skipped / total_trades_all) * 100
                lines.append(("Skip Rate:", f"{skip_rate:.2f}%", Colors.BRIGHT_YELLOW))
        else:
            lines.append(("Skipped Trades:", "None (all trades executed)", Colors.BRIGHT_GREEN))
        
        # Actual vs target allocation
        mean_actual = aggregated.get("mean_actual_allocation", 0.0)
        mean_target = aggregated.get("mean_target_allocation", 0.0)
        std_actual = aggregated.get("std_actual_allocation", 0.0)
        
        if mean_actual > 0:
            lines.append(("Mean Target Allocation:", f"{mean_target:.2f}%", Colors.BRIGHT_CYAN))
            lines.append(("Mean Actual Allocation:", f"{mean_actual:.2f}%", Colors.BRIGHT_GREEN))
            if mean_target > 0:
                allocation_diff = mean_actual - mean_target
                diff_color = Colors.BRIGHT_YELLOW if abs(allocation_diff) > 0.1 else Colors.BRIGHT_GREEN
                lines.append(("Allocation Difference:", f"{allocation_diff:+.2f}%", diff_color))
            lines.append(("Std Dev Actual Allocation:", f"{std_actual:.2f}%", Colors.BRIGHT_CYAN))
        
        if lines:
            print_box(box_width, "ALLOCATION & SKIPPED TRADES STATISTICS", lines)

    def _print_distribution_stats(self, aggregated: Dict) -> None:
        """Print distribution statistics."""
        box_width = 80
        lines = [
            ("Mean Max Drawdown:", f"${aggregated.get('mean_max_drawdown_dollars', 0):,.2f}", Colors.BRIGHT_YELLOW),
            ("Mean Max Drawdown (%):", f"{aggregated.get('mean_max_drawdown_pct', 0):.2f}%", Colors.BRIGHT_YELLOW),
            ("Mean Final Portfolio:", f"${aggregated.get('mean_final_portfolio', 0):,.2f}", Colors.BRIGHT_GREEN),
            ("Mean Total Return:", f"{aggregated.get('mean_total_return', 0):.2f}%", Colors.BRIGHT_GREEN),
            ("Mean CAGR:", f"{aggregated.get('mean_cagr', 0):.2f}%", Colors.BRIGHT_GREEN),
            ("Std Dev Max Drawdown:", f"${aggregated.get('std_max_drawdown_dollars', 0):,.2f}", Colors.BRIGHT_CYAN),
            ("Std Dev Final Portfolio:", f"${aggregated.get('std_final_portfolio', 0):,.2f}", Colors.BRIGHT_CYAN),
        ]
        print_box(box_width, "DISTRIBUTION STATISTICS", lines)

    def _print_distributions(self, aggregated: Dict) -> None:
        """Print visual distributions for key metrics."""
        # Max Drawdown Percentage Distribution
        if "_max_drawdown_pct" in aggregated:
            max_dd_pct = aggregated["_max_drawdown_pct"]
            if len(max_dd_pct) > 0:
                print_ascii_distribution(
                    max_dd_pct,
                    "Max Drawdown Distribution (%)",
                    width=60,
                    is_percentage=True,
                )

        # Final Portfolio Value Distribution
        if "_final_portfolios" in aggregated:
            final_portfolios = aggregated["_final_portfolios"]
            if len(final_portfolios) > 0:
                print_ascii_distribution(
                    final_portfolios,
                    "Final Portfolio Value Distribution",
                    width=60,
                    is_percentage=False,
                )

        # Total Return Distribution
        if "_total_returns" in aggregated:
            total_returns = aggregated["_total_returns"]
            if len(total_returns) > 0:
                print_ascii_distribution(
                    total_returns,
                    "Total Return Distribution (%)",
                    width=60,
                    is_percentage=True,
                )

        # CAGR Distribution
        if "_cagrs" in aggregated:
            cagrs = aggregated["_cagrs"]
            if len(cagrs) > 0:
                print_ascii_distribution(
                    cagrs,
                    "CAGR Distribution (%)",
                    width=60,
                    is_percentage=True,
                )

        # Actual Allocation Distribution per Strategy
        actual_allocations_by_strategy = aggregated.get("_actual_allocations_by_strategy", {})
        if actual_allocations_by_strategy:
            for strategy, allocations in actual_allocations_by_strategy.items():
                if len(allocations) > 0:
                    allocations_array = np.array(allocations)
                    print_ascii_distribution(
                        allocations_array,
                        f"Actual Allocation Distribution - {strategy} (%)",
                        width=60,
                        is_percentage=True,
                    )

    def _print_percentile_analysis(self, aggregated: Dict) -> None:
        """Print percentile analysis."""
        box_width = 80
        
        # Max Drawdown Percentiles
        dd_pct = aggregated.get("max_drawdown_pct_percentiles", {})
        if dd_pct:
            lines = [
                ("5th Percentile:", f"{dd_pct.get('p5', 0):.2f}%", Colors.BRIGHT_CYAN),
                ("25th Percentile:", f"{dd_pct.get('p25', 0):.2f}%", Colors.BRIGHT_YELLOW),
                ("50th Percentile (Median):", f"{dd_pct.get('p50', 0):.2f}%", Colors.BRIGHT_WHITE),
                ("75th Percentile:", f"{dd_pct.get('p75', 0):.2f}%", Colors.BRIGHT_YELLOW),
                ("95th Percentile:", f"{dd_pct.get('p95', 0):.2f}%", Colors.BRIGHT_RED),
            ]
            print_box(box_width, "MAX DRAWDOWN PERCENTILES (%)", lines)

        # Final Portfolio Percentiles
        fp_percentiles = aggregated.get("final_portfolio_percentiles", {})
        if fp_percentiles:
            lines = [
                ("5th Percentile:", f"${fp_percentiles.get('p5', 0):,.2f}", Colors.BRIGHT_RED),
                ("25th Percentile:", f"${fp_percentiles.get('p25', 0):,.2f}", Colors.BRIGHT_YELLOW),
                ("50th Percentile (Median):", f"${fp_percentiles.get('p50', 0):,.2f}", Colors.BRIGHT_WHITE),
                ("75th Percentile:", f"${fp_percentiles.get('p75', 0):,.2f}", Colors.BRIGHT_GREEN),
                ("95th Percentile:", f"${fp_percentiles.get('p95', 0):,.2f}", Colors.BRIGHT_GREEN),
            ]
            print_box(box_width, "FINAL PORTFOLIO VALUE PERCENTILES", lines)

        # Total Return Percentiles
        tr_percentiles = aggregated.get("total_return_percentiles", {})
        if tr_percentiles:
            lines = [
                ("5th Percentile:", f"{tr_percentiles.get('p5', 0):.2f}%", Colors.BRIGHT_RED),
                ("25th Percentile:", f"{tr_percentiles.get('p25', 0):.2f}%", Colors.BRIGHT_YELLOW),
                ("50th Percentile (Median):", f"{tr_percentiles.get('p50', 0):.2f}%", Colors.BRIGHT_WHITE),
                ("75th Percentile:", f"{tr_percentiles.get('p75', 0):.2f}%", Colors.BRIGHT_GREEN),
                ("95th Percentile:", f"{tr_percentiles.get('p95', 0):.2f}%", Colors.BRIGHT_GREEN),
            ]
            print_box(box_width, "TOTAL RETURN PERCENTILES (%)", lines)

    def _print_percentile_curves(self, aggregated: Dict) -> Optional[object]:
        """
        Create and display equity curves and drawdown curves for percentile simulations.
        
        Returns:
            Matplotlib figure if available, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            print("\n" + Colors.BRIGHT_YELLOW + "Note: Percentile curves chart requires matplotlib. Install with: pip install matplotlib" + Colors.RESET)
            return None
        
        simulation_results = aggregated.get("_simulation_results", [])
        starting_capital = aggregated.get("_starting_capital", 0)
        final_portfolio_percentiles = aggregated.get("final_portfolio_percentiles", {})
        
        if not simulation_results or not final_portfolio_percentiles:
            return None
        
        # Find simulations closest to each percentile
        percentile_targets = {
            5: final_portfolio_percentiles.get("p5", 0),
            25: final_portfolio_percentiles.get("p25", 0),
            50: final_portfolio_percentiles.get("p50", 0),
            75: final_portfolio_percentiles.get("p75", 0),
            95: final_portfolio_percentiles.get("p95", 0),
        }
        
        percentile_sims = {}
        for pct, target_value in percentile_targets.items():
            # Find simulation with final portfolio value closest to target
            closest_idx = None
            min_diff = float('inf')
            for idx, result in enumerate(simulation_results):
                diff = abs(result["final_portfolio"] - target_value)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = idx
            if closest_idx is not None:
                percentile_sims[pct] = simulation_results[closest_idx]
        
        if not percentile_sims:
            return None
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Colors for each percentile
        colors = {
            5: "#ef4444",   # Red
            25: "#f59e0b",  # Orange
            50: "#6366f1",  # Blue
            75: "#10b981",  # Green
            95: "#059669",  # Dark Green
        }
        
        # Plot equity curves
        for pct, sim_result in percentile_sims.items():
            equity_curve = sim_result.get("equity_curve", [])
            dates = sim_result.get("dates", [])
            
            if len(equity_curve) > 0 and len(dates) > 0:
                # Filter out None dates
                valid_indices = [i for i, d in enumerate(dates) if d is not None]
                if valid_indices:
                    valid_dates = [dates[i] for i in valid_indices]
                    valid_equity = [equity_curve[i] for i in valid_indices]
                    
                    ax1.plot(
                        valid_dates,
                        valid_equity,
                        color=colors[pct],
                        linewidth=2,
                        label=f"{pct}th Percentile",
                        alpha=0.8,
                    )
        
        ax1.axhline(y=starting_capital, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Starting Capital")
        ax1.set_xlabel("Date", fontsize=12)
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.set_title("Equity Curves - Percentile Simulations", fontsize=14, fontweight="bold")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Calculate and plot drawdown curves
        for pct, sim_result in percentile_sims.items():
            equity_curve = sim_result.get("equity_curve", [])
            dates = sim_result.get("dates", [])
            
            if len(equity_curve) > 0:
                # Calculate drawdown curve
                drawdown_curve = self._calculate_drawdown_curve(equity_curve, starting_capital)
                
                # Filter out None dates
                valid_indices = [i for i, d in enumerate(dates) if d is not None]
                if valid_indices:
                    valid_dates = [dates[i] for i in valid_indices]
                    valid_drawdown = [drawdown_curve[i] for i in valid_indices]
                    
                    ax2.fill_between(
                        valid_dates,
                        valid_drawdown,
                        0,
                        color=colors[pct],
                        alpha=0.3,
                    )
                    ax2.plot(
                        valid_dates,
                        valid_drawdown,
                        color=colors[pct],
                        linewidth=1.5,
                        linestyle="--",
                        label=f"{pct}th Percentile",
                        alpha=0.8,
                    )
        
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Drawdown (%)", fontsize=12)
        ax2.set_title("Drawdown Curves - Percentile Simulations", fontsize=14, fontweight="bold")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Invert so drawdown goes up
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        plt.tight_layout()
        
        # Don't show the figure (it will be added to HTML report)
        # Just return the figure for HTML report inclusion
        print("\n" + Colors.BRIGHT_GREEN + "Percentile equity and drawdown curves chart generated." + Colors.RESET)
        
        return fig
    
    def _calculate_drawdown_curve(self, equity_curve: np.ndarray, starting_capital: float) -> List[float]:
        """Calculate drawdown percentage curve from equity curve."""
        if len(equity_curve) == 0:
            return []
        
        drawdown_curve = []
        peak = starting_capital
        tol = 1e-9
        
        for value in equity_curve:
            if value >= peak - tol:
                peak = value
            drawdown = peak - value
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0.0
            drawdown_curve.append(drawdown_pct)
        
        return drawdown_curve

    def _print_simulation_progress(self, current: int, total: int) -> None:
        """Print progress bar for Monte Carlo simulations."""
        import sys
        
        bar_width = 50
        progress_pct = min(100, (current / total * 100)) if total > 0 else 0

        # Create progress bar
        filled = int(bar_width * progress_pct / 100)
        if progress_pct > 0 and filled == 0:
            filled = 1
        empty = bar_width - filled
        bar = f"{Colors.BRIGHT_GREEN}{'█' * filled}{Colors.RESET}{Colors.DIM}{'░' * empty}{Colors.RESET}"

        # Format progress line
        progress_line = f"Monte Carlo Simulations: {current:,}/{total:,} ({progress_pct:.1f}%) [{bar}]"
        
        # Use carriage return to overwrite the same line
        sys.stdout.write(f"\r{progress_line}")
        sys.stdout.flush()
        
        if current >= total:
            # Newline when complete
            sys.stdout.write("\n")
            sys.stdout.flush()
