"""Live vs Backtest comparison test for execution analysis."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import timedelta
from .base import BaseTest
from ..output.formatters import (
    print_box,
    print_section_box,
    print_ascii_distribution,
    print_comparison_summary,
    print_match_statistics,
    print_slippage_analysis,
    print_missed_trades,
    print_over_trades,
    print_allocation_analysis,
    print_matched_trades_table,
    print_pl_breakdown,
)
from ..utils.colors import Colors
from ..utils.leg_parser import parse_legs_from_dataframe_row, legs_match


class LiveBacktestComparisonTest(BaseTest):
    """Test that compares live trading execution against backtest data."""

    def get_name(self) -> str:
        """Get the name of the test."""
        return "compare"

    def get_description(self) -> str:
        """Get a description of what the test does."""
        return (
            "Compares real live trading execution against backtest data for the same period. "
            "Matches trades by time window and strategy, calculates P/L differences, premium differences, "
            "and identifies missed/over trades. Provides slippage analysis for fully matched trades."
        )

    def run(
        self,
        backtest_data: pd.DataFrame,
        live_data: pd.DataFrame,
        window_minutes: int = 10,
        starting_portfolio_size: float = 100000.0,
        source_of_truth: str = "live",
        verbose: bool = True,
    ) -> Dict:
        """
        Run comparison between backtest and live data.

        Args:
            backtest_data: DataFrame with backtest trade data
            live_data: DataFrame with live trade data
            window_minutes: Time window in minutes for matching trades (default: 10)
            starting_portfolio_size: Starting portfolio size for allocation calculations
            source_of_truth: "live" or "backtest" - determines which dataset defines the time period (default: "live")
            verbose: If True, print formatted output (default: True)

        Returns:
            Dictionary with comparison results
        """
        # Validate required columns
        self._validate_data(backtest_data, live_data)

        # Data should already be normalized and enriched by the normalizer/enricher
        # Make copies to avoid modifying original data
        backtest_df = backtest_data.copy()
        live_df = live_data.copy()

        # Filter dataframes based on source of truth
        backtest_df, live_df = self._filter_by_source_of_truth(
            backtest_df, live_df, source_of_truth
        )

        # Perform matching
        matches = self._match_trades(backtest_df, live_df, window_minutes)
        full_matches = self._find_full_matches(backtest_df, live_df, matches, window_minutes)

        # Calculate metrics
        results = self._calculate_metrics(
            backtest_df, live_df, matches, full_matches, window_minutes, starting_portfolio_size
        )

        if verbose:
            self._print_results(results, backtest_df, live_df, matches, full_matches)

        return results

    def _validate_data(self, backtest_data: pd.DataFrame, live_data: pd.DataFrame) -> None:
        """Validate that both datasets have required columns."""
        required_cols = ["P/L", "No. of Contracts", "datetime_opened"]
        backtest_missing = [col for col in required_cols if col not in backtest_data.columns]
        live_missing = [col for col in required_cols if col not in live_data.columns]

        if backtest_missing:
            raise ValueError(f"Backtest data missing required columns: {backtest_missing}")
        if live_missing:
            raise ValueError(f"Live data missing required columns: {live_missing}")

    def _filter_by_source_of_truth(
        self,
        backtest_df: pd.DataFrame,
        live_df: pd.DataFrame,
        source_of_truth: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter dataframes based on source of truth date range.

        Args:
            backtest_df: Prepared backtest DataFrame
            live_df: Prepared live DataFrame
            source_of_truth: "live" or "backtest" - determines which dataset defines the time period

        Returns:
            Tuple of (filtered_backtest_df, filtered_live_df)
        """
        if source_of_truth not in ["live", "backtest"]:
            raise ValueError(f"source_of_truth must be 'live' or 'backtest', got '{source_of_truth}'")

        if source_of_truth == "live":
            # Use live dates as source of truth - filter backtest to live date range
            if len(live_df) == 0:
                return backtest_df, live_df

            # Extract date only (without time) for comparison
            live_start_date = live_df["datetime_opened"].min().date()
            live_end_date = live_df["datetime_opened"].max().date()

            # Filter backtest to only include trades within live date range (date only)
            backtest_dates = pd.to_datetime(backtest_df["datetime_opened"]).dt.date
            backtest_filtered = backtest_df[
                (backtest_dates >= live_start_date)
                & (backtest_dates <= live_end_date)
            ].copy()

            return backtest_filtered, live_df
        else:
            # Use backtest dates as source of truth - filter live to backtest date range
            if len(backtest_df) == 0:
                return backtest_df, live_df

            # Extract date only (without time) for comparison
            backtest_start_date = backtest_df["datetime_opened"].min().date()
            backtest_end_date = backtest_df["datetime_opened"].max().date()

            # Filter live to only include trades within backtest date range (date only)
            live_dates = pd.to_datetime(live_df["datetime_opened"]).dt.date
            live_filtered = live_df[
                (live_dates >= backtest_start_date)
                & (live_dates <= backtest_end_date)
            ].copy()

            return backtest_df, live_filtered

    def _match_trades(
        self,
        backtest_df: pd.DataFrame,
        live_df: pd.DataFrame,
        window_minutes: int,
    ) -> List[Tuple[int, int]]:
        """
        Match trades between backtest and live data.

        Args:
            backtest_df: Prepared backtest DataFrame
            live_df: Prepared live DataFrame
            window_minutes: Time window in minutes

        Returns:
            List of (backtest_index, live_index) tuples for matched trades
        """
        matches = []
        window_delta = timedelta(minutes=window_minutes)

        # Track which live trades have been matched
        live_matched = set()

        for bt_idx, bt_row in backtest_df.iterrows():
            bt_datetime = bt_row["datetime_opened"]
            bt_strategy = bt_row["Strategy"]

            if pd.isna(bt_datetime):
                continue

            # Find live trades within time window
            time_mask = (
                (live_df["datetime_opened"] >= bt_datetime - window_delta)
                & (live_df["datetime_opened"] <= bt_datetime + window_delta)
            )

            # Filter by strategy
            # If backtest strategy is empty and live has single strategy, allow match
            live_strategies = live_df["Strategy"].unique()
            if bt_strategy == "" and len(live_strategies) == 1:
                strategy_mask = pd.Series([True] * len(live_df), index=live_df.index)
            else:
                strategy_mask = live_df["Strategy"] == bt_strategy

            # Combine masks
            candidate_mask = time_mask & strategy_mask

            # Find best match (closest time) among candidates not already matched
            candidates = live_df[candidate_mask & ~live_df.index.isin(live_matched)]
            if len(candidates) > 0:
                # Find closest time match
                time_diffs = abs(candidates["datetime_opened"] - bt_datetime)
                best_match_idx = time_diffs.idxmin()
                matches.append((bt_idx, best_match_idx))
                live_matched.add(best_match_idx)

        return matches

    def _find_full_matches(
        self,
        backtest_df: pd.DataFrame,
        live_df: pd.DataFrame,
        matches: List[Tuple[int, int]],
        window_minutes: int,
    ) -> List[Tuple[int, int]]:
        """
        Find fully matched trades (time, strategy, legs, reason for close).

        Args:
            backtest_df: Prepared backtest DataFrame
            live_df: Prepared live DataFrame
            matches: List of basic matches
            window_minutes: Time window in minutes

        Returns:
            List of (backtest_index, live_index) tuples for fully matched trades
        """
        full_matches = []

        for bt_idx, live_idx in matches:
            bt_row = backtest_df.loc[bt_idx]
            live_row = live_df.loc[live_idx]

            # Check legs match using normalized leg comparison (ignoring prices)
            bt_legs_parsed = parse_legs_from_dataframe_row(bt_row, "Legs")
            live_legs_parsed = parse_legs_from_dataframe_row(live_row, "Legs")
            
            if not legs_match(bt_legs_parsed, live_legs_parsed):
                continue

            # Check reason for close match
            bt_reason = str(bt_row.get("Reason For Close", ""))
            live_reason = str(live_row.get("Reason For Close", ""))
            if bt_reason != live_reason:
                continue

            full_matches.append((bt_idx, live_idx))

        return full_matches

    def _calculate_metrics(
        self,
        backtest_df: pd.DataFrame,
        live_df: pd.DataFrame,
        matches: List[Tuple[int, int]],
        full_matches: List[Tuple[int, int]],
        window_minutes: int,
        starting_portfolio_size: float,
    ) -> Dict:
        """
        Calculate all comparison metrics.

        Args:
            backtest_df: Prepared backtest DataFrame
            live_df: Prepared live DataFrame
            matches: List of matched trades
            full_matches: List of fully matched trades
            window_minutes: Time window used

        Returns:
            Dictionary with all calculated metrics
        """
        # Overall P/L difference
        total_backtest_pl = backtest_df["P/L"].sum()
        total_live_pl = live_df["P/L"].sum()
        overall_pl_diff = total_live_pl - total_backtest_pl

        # Matched trades P/L difference (normalized to one contract)
        # This is the P/L difference per contract for matched trades only
        # Note: This will equal overall_pl_diff when all trades are matched and all have the same contract count
        matched_backtest_indices = {bt_idx for bt_idx, _ in matches}
        matched_live_indices = {live_idx for _, live_idx in matches}
        
        # Calculate P/L per contract for matched trades
        matched_backtest_pl_per_contract = (
            backtest_df[backtest_df.index.isin(matched_backtest_indices)]["P/L"] / 
            backtest_df[backtest_df.index.isin(matched_backtest_indices)]["No. of Contracts"]
        ).sum()
        matched_live_pl_per_contract = (
            live_df[live_df.index.isin(matched_live_indices)]["P/L"] / 
            live_df[live_df.index.isin(matched_live_indices)]["No. of Contracts"]
        ).sum()
        matched_trades_pl_diff_per_contract = matched_live_pl_per_contract - matched_backtest_pl_per_contract

        # Keep premium sums for reference
        matched_backtest_premium = backtest_df[backtest_df.index.isin(matched_backtest_indices)]["Premium per Contract"].sum()
        matched_live_premium = live_df[live_df.index.isin(matched_live_indices)]["Premium per Contract"].sum()

        # Missed trades (in backtest but not matched)
        matched_backtest_indices = {bt_idx for bt_idx, _ in matches}
        missed_trades = backtest_df[~backtest_df.index.isin(matched_backtest_indices)].copy()

        # Over trades (in live but not matched)
        matched_live_indices = {live_idx for _, live_idx in matches}
        over_trades = live_df[~live_df.index.isin(matched_live_indices)].copy()

        # Fully matched trades statistics
        full_match_stats = self._calculate_full_match_stats(
            backtest_df, live_df, full_matches
        )

        # Allocation consistency analysis
        allocation_analysis = self._calculate_allocation_analysis(
            backtest_df, live_df, matches, starting_portfolio_size
        )

        # Create matched trades comparison table
        matched_trades_table = self._create_matched_trades_table(
            backtest_df, live_df, matches, starting_portfolio_size
        )

        # Calculate P/L difference breakdown
        breakdown = self._calculate_pl_breakdown(
            backtest_df, live_df, matches, missed_trades, over_trades, matched_trades_table, allocation_analysis
        )

        return {
            "overall_pl_diff": overall_pl_diff,
            "matched_trades_pl_diff_per_contract": matched_trades_pl_diff_per_contract,
            "total_backtest_pl": total_backtest_pl,
            "total_live_pl": total_live_pl,
            "total_backtest_premium": matched_backtest_premium,
            "total_live_premium": matched_live_premium,
            "missed_trades": missed_trades,
            "over_trades": over_trades,
            "num_matches": len(matches),
            "num_full_matches": len(full_matches),
            "num_backtest_trades": len(backtest_df),
            "num_live_trades": len(live_df),
            "window_minutes": window_minutes,
            "full_match_stats": full_match_stats,
            "allocation_analysis": allocation_analysis,
            "matched_trades_table": matched_trades_table,
            "pl_breakdown": breakdown,
            "matches": matches,
            "full_matches": full_matches,
        }

    def _calculate_full_match_stats(
        self,
        backtest_df: pd.DataFrame,
        live_df: pd.DataFrame,
        full_matches: List[Tuple[int, int]],
    ) -> Dict:
        """
        Calculate statistics for fully matched trades.

        Args:
            backtest_df: Prepared backtest DataFrame
            live_df: Prepared live DataFrame
            full_matches: List of fully matched trades

        Returns:
            Dictionary with full match statistics
        """
        if len(full_matches) == 0:
            return {
                "count": 0,
                "pl_diffs": [],
                "entry_diffs": [],
                "exit_diffs": [],
                "mean_pl_diff": 0,
                "median_pl_diff": 0,
                "std_pl_diff": 0,
                "mean_entry_diff": 0,
                "median_entry_diff": 0,
                "std_entry_diff": 0,
                "mean_exit_diff": 0,
                "median_exit_diff": 0,
                "std_exit_diff": 0,
                "backtest_win_rate": 0,
                "live_win_rate": 0,
            }

        pl_diffs = []
        entry_diffs = []  # Entry (premium) differences
        exit_diffs = []   # Exit (closing cost) differences

        for bt_idx, live_idx in full_matches:
            bt_row = backtest_df.loc[bt_idx]
            live_row = live_df.loc[live_idx]

            # P/L difference (live - backtest, normalized to one contract)
            pl_per_contract_bt = bt_row["P/L"] / bt_row.get("No. of Contracts", 1)
            pl_per_contract_live = live_row["P/L"] / live_row.get("No. of Contracts", 1)
            pl_diff = pl_per_contract_live - pl_per_contract_bt
            pl_diffs.append(pl_diff)

            # Entry difference (premium difference) - use same calculation as table
            # Table uses: Premium per Contract * No. of Contracts to get total, then difference
            premium_bt = bt_row.get("Premium per Contract", 0) * bt_row.get("No. of Contracts", 1)
            premium_live = live_row.get("Premium per Contract", 0) * live_row.get("No. of Contracts", 1)
            entry_diff = premium_live - premium_bt
            entry_diffs.append(entry_diff)

            # Exit difference (closing cost difference) - use same calculation as table
            # Table uses: Avg. Closing Cost (total) directly, then difference
            bt_closing_cost = bt_row.get("Avg. Closing Cost", 0)
            live_closing_cost = live_row.get("Avg. Closing Cost", 0)
            
            # Handle NaN values
            if pd.isna(bt_closing_cost):
                bt_closing_cost = 0
            if pd.isna(live_closing_cost):
                live_closing_cost = 0
            
            exit_diff = live_closing_cost - bt_closing_cost
            exit_diffs.append(exit_diff)

        pl_diffs = np.array(pl_diffs)
        entry_diffs = np.array(entry_diffs)
        exit_diffs = np.array(exit_diffs)

        # Win rates
        backtest_wins = sum(
            1 for bt_idx, _ in full_matches if backtest_df.loc[bt_idx]["P/L"] > 0
        )
        live_wins = sum(
            1 for _, live_idx in full_matches if live_df.loc[live_idx]["P/L"] > 0
        )

        return {
            "count": len(full_matches),
            "pl_diffs": pl_diffs,
            "entry_diffs": entry_diffs,
            "exit_diffs": exit_diffs,
            "mean_pl_diff": np.mean(pl_diffs),
            "median_pl_diff": np.median(pl_diffs),
            "std_pl_diff": np.std(pl_diffs),
            "mean_entry_diff": np.mean(entry_diffs),
            "median_entry_diff": np.median(entry_diffs),
            "std_entry_diff": np.std(entry_diffs),
            "mean_exit_diff": np.mean(exit_diffs),
            "median_exit_diff": np.median(exit_diffs),
            "std_exit_diff": np.std(exit_diffs),
            "backtest_win_rate": backtest_wins / len(full_matches) if full_matches else 0,
            "live_win_rate": live_wins / len(full_matches) if full_matches else 0,
        }

    def _calculate_allocation_analysis(
        self,
        backtest_df: pd.DataFrame,
        live_df: pd.DataFrame,
        matches: List[Tuple[int, int]],
        starting_portfolio_size: float,
    ) -> Dict:
        """
        Calculate allocation consistency between backtest and live.

        Args:
            backtest_df: Prepared backtest DataFrame
            live_df: Prepared live DataFrame
            matches: List of matched trades
            starting_portfolio_size: Starting portfolio size for live trading

        Returns:
            Dictionary with allocation analysis results
        """
        # Calculate backtest allocations
        backtest_allocations = []
        if "Funds at Close" in backtest_df.columns and "Margin Req." in backtest_df.columns:
            for idx, row in backtest_df.iterrows():
                funds = row.get("Funds at Close", 0)
                margin = row.get("Margin Req.", 0)
                if funds > 0 and margin > 0:
                    allocation_pct = (margin / funds) * 100
                    backtest_allocations.append(allocation_pct)

        mean_backtest_allocation = np.mean(backtest_allocations) if backtest_allocations else 0
        std_backtest_allocation = np.std(backtest_allocations) if backtest_allocations else 0

        # Calculate live allocations using rolling portfolio
        live_allocations = []
        live_allocation_data = []
        current_portfolio = starting_portfolio_size
        matched_live_indices = {live_idx for _, live_idx in matches}
        
        # Create a mapping from live index to backtest margin requirement
        margin_map = {}
        for bt_idx, live_idx in matches:
            if "Margin Req." in backtest_df.columns:
                margin = backtest_df.loc[bt_idx].get("Margin Req.", 0)
                margin_map[live_idx] = margin

        # Sort live trades by datetime
        live_df_sorted = live_df.sort_values("datetime_opened").copy()
        
        # First pass: collect all live allocations
        # Create mapping from live index to contract count and margin per contract
        live_contracts_map = {}
        margin_per_contract_map = {}
        for bt_idx, live_idx in matches:
            if live_idx in matched_live_indices:
                live_contracts_map[live_idx] = live_df.loc[live_idx].get("No. of Contracts", 1)
                bt_margin = margin_map.get(live_idx, 0)
                bt_contracts = backtest_df.loc[bt_idx].get("No. of Contracts", 1)
                margin_per_contract_map[live_idx] = bt_margin / bt_contracts if bt_contracts > 0 else 0
        
        for live_idx, live_row in live_df_sorted.iterrows():
            if live_idx in matched_live_indices:
                # Get margin per contract and scale to live contract count
                margin_per_contract = margin_per_contract_map.get(live_idx, 0)
                live_contracts = live_contracts_map.get(live_idx, 1)
                margin_live = margin_per_contract * live_contracts
                
                # Calculate allocation using current portfolio size (at trade open)
                if margin_live > 0 and current_portfolio > 0:
                    allocation_pct = (margin_live / current_portfolio) * 100
                    live_allocations.append(allocation_pct)
            
            # Update portfolio after trade closes (add P/L)
            # Note: This assumes trades close before the next one opens
            pl = live_row.get("P/L", 0)
            current_portfolio = max(0, current_portfolio + pl)

        # Calculate live allocation statistics
        mean_live_allocation = np.mean(live_allocations) if live_allocations else 0
        std_live_allocation = np.std(live_allocations) if live_allocations else 0
        
        # Second pass: check deviations against respective means
        current_portfolio = starting_portfolio_size
        for live_idx, live_row in live_df_sorted.iterrows():
            if live_idx in matched_live_indices:
                # Get margin per contract and scale to live contract count
                margin_per_contract = margin_per_contract_map.get(live_idx, 0)
                live_contracts = live_contracts_map.get(live_idx, 1)
                margin_live = margin_per_contract * live_contracts
                
                # Calculate allocation using current portfolio size (at trade open)
                if margin_live > 0 and current_portfolio > 0:
                    allocation_pct = (margin_live / current_portfolio) * 100
                    
                    # Check if significantly deviates from live mean (not backtest mean)
                    deviation = abs(allocation_pct - mean_live_allocation)
                    is_deviant = deviation > (2 * std_live_allocation) if std_live_allocation > 0 else False
                    
                    live_allocation_data.append({
                        "live_idx": live_idx,
                        "date": live_row.get("Date Opened", ""),
                        "time": live_row.get("Time Opened", ""),
                        "portfolio_size": current_portfolio,
                        "margin": margin_live,
                        "allocation_pct": allocation_pct,
                        "mean_live_allocation": mean_live_allocation,
                        "deviation": deviation,
                        "is_deviant": is_deviant,
                    })
            
            # Update portfolio after trade closes (add P/L)
            pl = live_row.get("P/L", 0)
            current_portfolio = max(0, current_portfolio + pl)

        deviant_trades = [d for d in live_allocation_data if d["is_deviant"]]

        return {
            "mean_backtest_allocation": mean_backtest_allocation,
            "std_backtest_allocation": std_backtest_allocation,
            "mean_live_allocation": mean_live_allocation,
            "std_live_allocation": std_live_allocation,
            "live_allocations": live_allocations,
            "live_allocation_data": live_allocation_data,
            "deviant_trades": deviant_trades,
            "num_deviant_trades": len(deviant_trades),
            "starting_portfolio_size": starting_portfolio_size,
        }

    def _calculate_pl_breakdown(
        self,
        backtest_df: pd.DataFrame,
        live_df: pd.DataFrame,
        matches: List[Tuple[int, int]],
        missed_trades: pd.DataFrame,
        over_trades: pd.DataFrame,
        matched_trades_table: List[Dict],
        allocation_analysis: Dict,
    ) -> Dict:
        """
        Calculate breakdown of P/L difference by category.

        Args:
            backtest_df: Backtest DataFrame
            live_df: Live DataFrame
            matches: List of matched trades
            missed_trades: DataFrame of missed trades
            over_trades: DataFrame of over trades
            matched_trades_table: Table data with matched trades
            allocation_analysis: Allocation analysis results

        Returns:
            Dictionary with breakdown by category
        """
        overall_pl_diff = live_df["P/L"].sum() - backtest_df["P/L"].sum()
        
        # Over trading: P/L from trades in live but not in backtest
        over_trading_pl = over_trades["P/L"].sum() if len(over_trades) > 0 else 0
        over_trading_count = len(over_trades)
        over_trading_avg = over_trading_pl / over_trading_count if over_trading_count > 0 else 0
        
        # Missed trades: Negative P/L from trades in backtest but not in live (opportunity cost)
        missed_trades_pl = -missed_trades["P/L"].sum() if len(missed_trades) > 0 else 0
        missed_trades_count = len(missed_trades)
        missed_trades_avg = missed_trades_pl / missed_trades_count if missed_trades_count > 0 else 0
        
        # Entry slippage: Sum of premium differences for matched trades
        # Positive premium_diff means we received more premium (good for P/L)
        entry_slippage_rows = [row for row in matched_trades_table if row.get("premium_diff", 0) != 0]
        entry_slippage = sum(row.get("premium_diff", 0) for row in entry_slippage_rows)
        entry_slippage_count = len(entry_slippage_rows)
        entry_slippage_avg = entry_slippage / entry_slippage_count if entry_slippage_count > 0 else 0
        
        # Different outcome: P/L difference from trades that matched but had different outcomes
        # These are trades where the reason for close differs (e.g., Stop Loss vs Expired)
        different_outcome_rows = [row for row in matched_trades_table if not row.get("reason_match", True)]
        different_outcome_pl = sum(row.get("pl_diff", 0) for row in different_outcome_rows)
        different_outcome_count = len(different_outcome_rows)
        different_outcome_avg = different_outcome_pl / different_outcome_count if different_outcome_count > 0 else 0
        
        # Exit slippage: Negate closing cost differences for matched trades WITH SAME OUTCOME
        # Only include trades where reason for close matches (same outcome)
        # Positive closing_cost_diff means we paid more to close (bad for P/L), so negate it
        exit_slippage_rows = [row for row in matched_trades_table if row.get("reason_match", False) and row.get("closing_cost_diff", 0) != 0]
        exit_slippage = -sum(row.get("closing_cost_diff", 0) for row in exit_slippage_rows)
        exit_slippage_count = len(exit_slippage_rows)
        exit_slippage_avg = exit_slippage / exit_slippage_count if exit_slippage_count > 0 else 0
        
        # Allocation impact: Calculate based on allocation deviations
        # Under-allocation: When live allocation < mean, we potentially lost profit on winning trades
        # Over-allocation: When live allocation > mean, we potentially lost more on losing trades
        # Only calculated when contract counts differ between backtest and live
        mean_live_allocation = allocation_analysis.get("mean_live_allocation", 0)
        under_allocation_pl = 0
        over_allocation_pl = 0
        under_allocation_count = 0
        over_allocation_count = 0
        
        for row in matched_trades_table:
            alloc_live = row.get("alloc_live", 0)
            pl_diff = row.get("pl_diff", 0)
            contracts_bt = row.get("contracts_bt", 1)
            contracts_live = row.get("contracts_live", 1)
            
            # Only calculate allocation impact if contract counts differ
            # If both backtest and live used the same number of contracts, there's no allocation impact
            if contracts_bt != contracts_live and mean_live_allocation > 0:
                # Calculate allocation deviation
                alloc_deviation = alloc_live - mean_live_allocation
                
                if alloc_deviation < 0:  # Under-allocated
                    under_allocation_count += 1
                    # If we made profit but were under-allocated, we lost potential profit
                    if pl_diff > 0:
                        # Scale the profit difference by allocation ratio
                        ratio = alloc_live / mean_live_allocation if mean_live_allocation > 0 else 1
                        contribution = pl_diff * (1 - ratio)
                        under_allocation_pl += contribution
                    elif pl_diff < 0:
                        # If we lost money but were under-allocated, we saved some loss
                        ratio = alloc_live / mean_live_allocation if mean_live_allocation > 0 else 1
                        contribution = abs(pl_diff) * (1 - ratio)
                        under_allocation_pl += contribution
                elif alloc_deviation > 0:  # Over-allocated
                    over_allocation_count += 1
                    # If we lost money and were over-allocated, we lost more
                    if pl_diff < 0:
                        ratio = alloc_live / mean_live_allocation if mean_live_allocation > 0 else 1
                        contribution = abs(pl_diff) * (ratio - 1)
                        over_allocation_pl += contribution
                    elif pl_diff > 0:
                        # If we made profit and were over-allocated, we made more
                        ratio = alloc_live / mean_live_allocation if mean_live_allocation > 0 else 1
                        contribution = -pl_diff * (ratio - 1)  # Negative because it's a gain
                        over_allocation_pl += contribution
        
        under_allocation_avg = under_allocation_pl / under_allocation_count if under_allocation_count > 0 else 0
        over_allocation_avg = over_allocation_pl / over_allocation_count if over_allocation_count > 0 else 0
        
        # Calculate percentages
        total_attributed = abs(over_trading_pl) + abs(missed_trades_pl) + abs(entry_slippage) + abs(exit_slippage) + abs(different_outcome_pl) + abs(under_allocation_pl) + abs(over_allocation_pl)
        
        def calc_percentage(value):
            if total_attributed > 0:
                return (abs(value) / total_attributed) * 100
            return 0.0
        
        return {
            "over_trading": {
                "value": over_trading_pl,
                "percentage": calc_percentage(over_trading_pl),
                "count": over_trading_count,
                "average": over_trading_avg,
            },
            "missed_trades": {
                "value": missed_trades_pl,
                "percentage": calc_percentage(missed_trades_pl),
                "count": missed_trades_count,
                "average": missed_trades_avg,
            },
            "entry_slippage": {
                "value": entry_slippage,
                "percentage": calc_percentage(entry_slippage),
                "count": entry_slippage_count,
                "average": entry_slippage_avg,
            },
            "exit_slippage": {
                "value": exit_slippage,
                "percentage": calc_percentage(exit_slippage),
                "count": exit_slippage_count,
                "average": exit_slippage_avg,
            },
            "different_outcome": {
                "value": different_outcome_pl,
                "percentage": calc_percentage(different_outcome_pl),
                "count": different_outcome_count,
                "average": different_outcome_avg,
            },
            "under_allocation": {
                "value": under_allocation_pl,
                "percentage": calc_percentage(under_allocation_pl),
                "count": under_allocation_count,
                "average": under_allocation_avg,
            },
            "over_allocation": {
                "value": over_allocation_pl,
                "percentage": calc_percentage(over_allocation_pl),
                "count": over_allocation_count,
                "average": over_allocation_avg,
            },
            "overall_pl_diff": overall_pl_diff,
        }

    def _create_matched_trades_table(
        self,
        backtest_df: pd.DataFrame,
        live_df: pd.DataFrame,
        matches: List[Tuple[int, int]],
        starting_portfolio_size: float,
    ) -> List[Dict]:
        """
        Create a table of matched trades for comparison.

        Args:
            backtest_df: Prepared backtest DataFrame
            live_df: Prepared live DataFrame
            matches: List of matched trades
            starting_portfolio_size: Starting portfolio size for allocation calculation

        Returns:
            List of dictionaries with matched trade comparison data
        """
        table_data = []
        
        # Track portfolio size for accurate allocation calculation
        current_portfolio = starting_portfolio_size
        live_df_sorted = live_df.sort_values("datetime_opened").copy()
        matched_live_indices = {live_idx for _, live_idx in matches}
        
        # Create margin map from backtest to live
        margin_map = {}
        for bt_idx, live_idx in matches:
            if "Margin Req." in backtest_df.columns:
                margin = backtest_df.loc[bt_idx].get("Margin Req.", 0)
                margin_map[live_idx] = margin
        
        # Create a mapping of live_idx to portfolio size at trade time
        portfolio_at_trade = {}
        for live_idx, live_row in live_df_sorted.iterrows():
            if live_idx in matched_live_indices:
                portfolio_at_trade[live_idx] = current_portfolio
            # Update portfolio after trade closes
            pl = live_row.get("P/L", 0)
            current_portfolio = max(0, current_portfolio + pl)
        
        for bt_idx, live_idx in matches:
            bt_row = backtest_df.loc[bt_idx]
            live_row = live_df.loc[live_idx]
            
            # Use Premium per Contract * contracts for both to ensure consistency
            # This uses the normalized values which are already adjusted for live data
            premium_bt = bt_row.get("Premium per Contract", 0) * bt_row.get("No. of Contracts", 1)
            premium_live = live_row.get("Premium per Contract", 0) * live_row.get("No. of Contracts", 1)
            
            # Calculate premium difference (live - backtest)
            premium_diff = premium_live - premium_bt
            
            # Get margin requirement (prefer from backtest)
            margin_bt = bt_row.get("Margin Req.", 0)
            contracts_bt = bt_row.get("No. of Contracts", 1)
            contracts_live = live_row.get("No. of Contracts", 1)
            
            # Calculate margin per contract
            margin_per_contract = margin_bt / contracts_bt if contracts_bt > 0 else 0
            
            # Calculate allocation percentage for both BT and Live
            # Use margin requirement if available, otherwise use premium
            portfolio_size_at_trade = portfolio_at_trade.get(live_idx, starting_portfolio_size)
            
            # Backtest allocation (use total margin from backtest)
            if margin_bt > 0 and starting_portfolio_size > 0:
                alloc_bt = (margin_bt / starting_portfolio_size) * 100
            elif starting_portfolio_size > 0:
                alloc_bt = (abs(premium_bt) / starting_portfolio_size) * 100
            else:
                alloc_bt = 0
            
            # Live allocation (use margin scaled to live contract count, portfolio at trade time)
            # Calculate margin for live trade: margin_per_contract * live_contracts
            margin_live = margin_per_contract * contracts_live if margin_per_contract > 0 else 0
            if margin_live > 0 and portfolio_size_at_trade > 0:
                alloc_live = (margin_live / portfolio_size_at_trade) * 100
            elif portfolio_size_at_trade > 0:
                # Fall back to premium if margin not available
                alloc_live = (abs(premium_live) / portfolio_size_at_trade) * 100
            else:
                alloc_live = 0
            
            # Calculate P/L per contract for both
            pl_per_contract_bt = bt_row.get("P/L", 0) / contracts_bt if contracts_bt > 0 else 0
            pl_per_contract_live = live_row.get("P/L", 0) / contracts_live if contracts_live > 0 else 0
            pl_diff_per_contract = pl_per_contract_live - pl_per_contract_bt
            
            # Get closing costs (total, not per contract)
            closing_cost_bt = bt_row.get("Avg. Closing Cost", 0)
            closing_cost_live = live_row.get("Avg. Closing Cost", 0)
            # Handle NaN values
            if pd.isna(closing_cost_bt):
                closing_cost_bt = 0
            if pd.isna(closing_cost_live):
                closing_cost_live = 0
            
            # Calculate closing cost difference (live - backtest)
            closing_cost_diff = closing_cost_live - closing_cost_bt
            
            table_data.append({
                "date": bt_row.get("Date Opened", ""),
                "time_bt": bt_row.get("Time Opened", ""),
                "time_live": live_row.get("Time Opened", ""),
                "premium_bt": premium_bt,
                "premium_live": premium_live,
                "premium_diff": premium_diff,
                "closing_cost_bt": closing_cost_bt,
                "closing_cost_live": closing_cost_live,
                "closing_cost_diff": closing_cost_diff,
                "pl_bt": bt_row.get("P/L", 0),
                "pl_live": live_row.get("P/L", 0),
                "pl_diff": live_row.get("P/L", 0) - bt_row.get("P/L", 0),
                "pl_diff_per_contract": pl_diff_per_contract,
                "contracts_bt": contracts_bt,
                "contracts_live": contracts_live,
                "margin_per_contract": margin_per_contract,
                "alloc_bt": alloc_bt,
                "alloc_live": alloc_live,
                "strategy_bt": bt_row.get("Strategy", ""),
                "strategy_live": live_row.get("Strategy", ""),
                "legs_match": legs_match(
                    parse_legs_from_dataframe_row(bt_row, "Legs"),
                    parse_legs_from_dataframe_row(live_row, "Legs")
                ),
                "reason_match": str(bt_row.get("Reason For Close", "")) == str(live_row.get("Reason For Close", "")),
            })
        
        return table_data

    def _print_results(
        self,
        results: Dict,
        backtest_df: pd.DataFrame,
        live_df: pd.DataFrame,
        matches: List[Tuple[int, int]],
        full_matches: List[Tuple[int, int]],
    ) -> None:
        """Print formatted comparison results."""
        print_comparison_summary(results)
        print_match_statistics(results, backtest_df, live_df)
        print_slippage_analysis(results)
        print_allocation_analysis(results)
        print_matched_trades_table(results)
        print_missed_trades(results)
        print_over_trades(results)
        print_pl_breakdown(results)