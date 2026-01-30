"""Dynamic allocation simulator for equity curve generation with Monte Carlo methods."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime


class DynamicAllocationSimulator:
    """
    Simulator for generating equity curves with dynamic allocation.
    
    This class simulates trading with dynamic position sizing based on:
    - Portfolio size (changes as trades execute)
    - Allocation percentage (percentage of portfolio to allocate per trade)
    - Margin per contract (from trade data)
    
    Supports both single strategy and portfolio (multi-strategy) simulations,
    as well as Monte Carlo simulations using bootstrap sampling.
    """

    def __init__(self, portfolio_size: float, allocation_pct: float, force_one_lot: bool = False):
        """
        Initialize the simulator.

        Args:
            portfolio_size: Initial portfolio size in dollars
            allocation_pct: Allocation percentage (e.g., 1.0 for 1%)
            force_one_lot: If True, force at least 1 contract even when allocation is insufficient
        """
        if portfolio_size <= 0:
            raise ValueError("portfolio_size must be greater than 0")
        if allocation_pct <= 0:
            raise ValueError("allocation_pct must be greater than 0")

        self.portfolio_size = portfolio_size
        self.allocation_pct = allocation_pct
        self.force_one_lot = force_one_lot

    def simulate_equity_curve(
        self,
        data: pd.DataFrame,
        strategy_allocations: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Generate a single equity curve with dynamic allocation.

        Args:
            data: DataFrame with trade data. Must have:
                - "P/L per Contract" column
                - "No. of Contracts" column
                - "datetime_opened" column (for sorting)
                - "Strategy" column (if portfolio)
                - "Margin Req." column (optional, for margin-based sizing)
            strategy_allocations: Optional dict mapping strategy names to allocation percentages.
                If None and portfolio detected, uses self.allocation_pct for all strategies.

        Returns:
            Dictionary with:
                - "portfolio_values": List of portfolio values over time
                - "dates": List of datetime objects
                - "equity_curve": Array of portfolio values
                - "final_portfolio": Final portfolio value
                - "total_return": Total return percentage
                - "num_trades": Number of trades executed
                - "over_allocated_trades": Number of trades forced to 1-lot when allocation was insufficient
                - "over_allocation_details": List of dicts with over-allocation info (if force_one_lot=True)
        """
        if len(data) == 0:
            return {
                "portfolio_values": [],
                "dates": [],
                "equity_curve": np.array([]),
                "final_portfolio": self.portfolio_size,
                "total_return": 0.0,
                "num_trades": 0,
                "over_allocated_trades": 0,
                "over_allocation_details": [],
            }

        # Sort by datetime to process trades chronologically
        if "datetime_opened" in data.columns:
            df = data.sort_values("datetime_opened").copy()
        else:
            df = data.copy()

        portfolio_values = []
        dates = []
        current_portfolio = self.portfolio_size
        over_allocated_trades = 0
        over_allocation_details = []

        # Check if this is a portfolio (multiple strategies)
        is_portfolio = False
        if "Strategy" in df.columns:
            unique_strategies = df["Strategy"].dropna().unique()
            unique_strategies = [s for s in unique_strategies if str(s).strip() != ""]
            is_portfolio = len(unique_strategies) > 1

        # Determine allocation percentages for each strategy
        if is_portfolio and strategy_allocations:
            # Use provided strategy allocations
            allocation_map = strategy_allocations
        elif is_portfolio:
            # Use default allocation for all strategies
            unique_strategies = df["Strategy"].dropna().unique()
            unique_strategies = [s for s in unique_strategies if str(s).strip() != ""]
            allocation_map = {strategy: self.allocation_pct for strategy in unique_strategies}
        else:
            # Single strategy
            allocation_map = None

        has_margin_data = "Margin Req." in df.columns

        for idx, (_, row) in enumerate(df.iterrows()):
            # Get allocation percentage for this trade's strategy
            if is_portfolio and allocation_map:
                strategy = str(row.get("Strategy", ""))
                allocation_pct = allocation_map.get(strategy, self.allocation_pct)
            else:
                allocation_pct = self.allocation_pct

            # Calculate contracts to trade
            contracts_result = self._calculate_contracts_to_trade(
                row, current_portfolio, allocation_pct, has_margin_data
            )
            
            # Handle result - can be int or dict with over-allocation info
            if isinstance(contracts_result, dict):
                contracts_to_trade = contracts_result["contracts"]
                if contracts_result.get("over_allocated", False):
                    over_allocated_trades += 1
                    over_allocation_details.append({
                        "trade_index": idx,
                        "date": row.get("datetime_opened") if "datetime_opened" in row else None,
                        "allocation_pct": allocation_pct,
                        "required_allocation_pct": contracts_result.get("required_allocation_pct", 0.0),
                        "over_allocation_pct": contracts_result.get("over_allocation_pct", 0.0),
                        "margin_per_contract": contracts_result.get("margin_per_contract", 0.0),
                    })
            else:
                contracts_to_trade = contracts_result

            # Get P/L per contract
            pl_per_contract = row.get("P/L per Contract", 0)

            # Calculate P/L for this trade
            trade_pl = pl_per_contract * contracts_to_trade

            # Update portfolio
            current_portfolio = max(0, current_portfolio + trade_pl)

            # Store values
            portfolio_values.append(current_portfolio)
            if "datetime_opened" in row:
                dates.append(row["datetime_opened"])
            else:
                dates.append(None)

        equity_curve = np.array(portfolio_values)
        final_portfolio = current_portfolio
        total_return = ((final_portfolio - self.portfolio_size) / self.portfolio_size) * 100

        return {
            "portfolio_values": portfolio_values,
            "dates": dates,
            "equity_curve": equity_curve,
            "final_portfolio": final_portfolio,
            "total_return": total_return,
            "num_trades": len(df),
            "over_allocated_trades": over_allocated_trades,
            "over_allocation_details": over_allocation_details,
        }

    def _calculate_contracts_to_trade(
        self,
        row: pd.Series,
        current_portfolio: float,
        allocation_pct: float,
        has_margin_data: bool,
    ) -> Union[int, dict]:
        """
        Calculate number of contracts to trade based on allocation and margin.

        Args:
            row: DataFrame row with trade data
            current_portfolio: Current portfolio value
            allocation_pct: Allocation percentage for this trade
            has_margin_data: Whether margin data is available

        Returns:
            If force_one_lot=False: Number of contracts to trade (integer, rounded down)
            If force_one_lot=True: Dict with contracts and over-allocation info if forced
        """
        if has_margin_data and "Margin Req." in row.index:
            margin_req = row.get("Margin Req.", 0)
            contracts_in_trade = row.get("No. of Contracts", 1)

            if margin_req > 0 and contracts_in_trade > 0:
                # Calculate margin per contract
                margin_per_contract = margin_req / contracts_in_trade

                # Calculate allocation amount for this trade
                allocation_amount = current_portfolio * (allocation_pct / 100)

                # Calculate how many contracts we can trade (round down)
                if margin_per_contract > 0:
                    contracts_to_trade = int(allocation_amount / margin_per_contract)
                    
                    # Check if we need to force 1-lot
                    if self.force_one_lot and contracts_to_trade == 0:
                        # Force 1 contract and track over-allocation
                        # Calculate what allocation % would be needed for 1 contract
                        required_allocation_pct = (margin_per_contract / current_portfolio * 100) if current_portfolio > 0 else 0.0
                        over_allocation_pct = max(0.0, required_allocation_pct - allocation_pct)
                        return {
                            "contracts": 1,
                            "over_allocated": True,
                            "required_allocation_pct": required_allocation_pct,
                            "over_allocation_pct": over_allocation_pct,
                            "margin_per_contract": margin_per_contract,
                        }
                    elif self.force_one_lot:
                        # force_one_lot is True but we have enough allocation
                        return {"contracts": contracts_to_trade, "over_allocated": False}
                    else:
                        # force_one_lot is False, return int
                        return contracts_to_trade
                else:
                    contracts_to_trade = 0
                    if self.force_one_lot:
                        return {"contracts": 1, "over_allocated": False}
                    return contracts_to_trade
            else:
                # No valid margin data for this trade, use 1 contract
                contracts_to_trade = 1
                if self.force_one_lot:
                    return {"contracts": 1, "over_allocated": False}
                return contracts_to_trade
        else:
            # No margin data available, use 1 contract
            contracts_to_trade = 1
            if self.force_one_lot:
                return {"contracts": 1, "over_allocated": False}
            return contracts_to_trade

    def monte_carlo_simulate(
        self,
        data: pd.DataFrame,
        num_simulations: int = 10000,
        strategy_allocations: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Run Monte Carlo simulations with bootstrap sampling.

        Args:
            data: DataFrame with trade data (same requirements as simulate_equity_curve)
            num_simulations: Number of Monte Carlo simulations to run
            strategy_allocations: Optional dict mapping strategy names to allocation percentages

        Returns:
            Dictionary with:
                - "simulation_results": List of simulation result dicts
                - "final_portfolios": Array of final portfolio values from all simulations
                - "total_returns": Array of total return percentages
                - "mean_final_portfolio": Mean final portfolio value
                - "std_final_portfolio": Standard deviation of final portfolio values
                - "mean_total_return": Mean total return percentage
                - "std_total_return": Standard deviation of total returns
                - "min_final_portfolio": Minimum final portfolio value
                - "max_final_portfolio": Maximum final portfolio value
                - "percentile_5": 5th percentile of final portfolio values
                - "percentile_95": 95th percentile of final portfolio values
        """
        if len(data) == 0:
            return {
                "simulation_results": [],
                "final_portfolios": np.array([]),
                "total_returns": np.array([]),
                "mean_final_portfolio": self.portfolio_size,
                "std_final_portfolio": 0.0,
                "mean_total_return": 0.0,
                "std_total_return": 0.0,
                "min_final_portfolio": self.portfolio_size,
                "max_final_portfolio": self.portfolio_size,
                "percentile_5": self.portfolio_size,
                "percentile_95": self.portfolio_size,
            }

        # Get P/L per Contract values for bootstrap sampling
        if "P/L per Contract" not in data.columns:
            raise ValueError("Data must have 'P/L per Contract' column for Monte Carlo simulation")

        pl_per_contract_values = data["P/L per Contract"].values
        pl_per_contract_values = pl_per_contract_values[
            ~(np.isnan(pl_per_contract_values) | np.isinf(pl_per_contract_values))
        ]

        if len(pl_per_contract_values) == 0:
            raise ValueError("No valid P/L per Contract values found for Monte Carlo simulation")

        # Get other required data for simulation
        num_trades = len(data)
        has_margin_data = "Margin Req." in data.columns

        # Prepare strategy information if portfolio
        is_portfolio = False
        if "Strategy" in data.columns:
            unique_strategies = data["Strategy"].dropna().unique()
            unique_strategies = [s for s in unique_strategies if str(s).strip() != ""]
            is_portfolio = len(unique_strategies) > 1

        if is_portfolio and strategy_allocations:
            allocation_map = strategy_allocations
        elif is_portfolio:
            unique_strategies = data["Strategy"].dropna().unique()
            unique_strategies = [s for s in unique_strategies if str(s).strip() != ""]
            allocation_map = {strategy: self.allocation_pct for strategy in unique_strategies}
        else:
            allocation_map = None

        # Prepare margin data if available
        margin_data = None
        if has_margin_data:
            margin_data = data[["Margin Req.", "No. of Contracts", "Strategy"]].copy()

        simulation_results = []
        final_portfolios = []
        total_returns = []

        for sim_idx in range(num_simulations):
            # Bootstrap sample: randomly sample trades with replacement
            sampled_indices = np.random.choice(len(pl_per_contract_values), size=num_trades, replace=True)
            sampled_pl_per_contract = pl_per_contract_values[sampled_indices]

            # Simulate equity curve
            current_portfolio = self.portfolio_size
            portfolio_values = []

            for i, pl_per_contract in enumerate(sampled_pl_per_contract):
                # Get allocation percentage for this trade
                if is_portfolio and allocation_map and margin_data is not None:
                    # Sample a strategy from original data (weighted by occurrence)
                    strategy = np.random.choice(data["Strategy"].dropna().values)
                    allocation_pct = allocation_map.get(str(strategy), self.allocation_pct)
                else:
                    allocation_pct = self.allocation_pct

                # Calculate contracts to trade
                contracts_to_trade = 1  # Default
                if has_margin_data and margin_data is not None:
                    # Sample margin data from original data
                    margin_row_idx = np.random.choice(len(margin_data))
                    margin_row = margin_data.iloc[margin_row_idx]
                    margin_req = margin_row.get("Margin Req.", 0)
                    contracts_in_trade = margin_row.get("No. of Contracts", 1)

                    if margin_req > 0 and contracts_in_trade > 0:
                        margin_per_contract = margin_req / contracts_in_trade
                        allocation_amount = current_portfolio * (allocation_pct / 100)
                        if margin_per_contract > 0:
                            contracts_to_trade = int(allocation_amount / margin_per_contract)
                        else:
                            contracts_to_trade = 0

                # Calculate trade P/L
                trade_pl = pl_per_contract * contracts_to_trade

                # Update portfolio
                current_portfolio = max(0, current_portfolio + trade_pl)
                portfolio_values.append(current_portfolio)

            final_portfolio = current_portfolio
            total_return = ((final_portfolio - self.portfolio_size) / self.portfolio_size) * 100

            simulation_results.append({
                "simulation_idx": sim_idx,
                "final_portfolio": final_portfolio,
                "total_return": total_return,
                "portfolio_values": portfolio_values,
            })

            final_portfolios.append(final_portfolio)
            total_returns.append(total_return)

        final_portfolios = np.array(final_portfolios)
        total_returns = np.array(total_returns)

        return {
            "simulation_results": simulation_results,
            "final_portfolios": final_portfolios,
            "total_returns": total_returns,
            "mean_final_portfolio": np.mean(final_portfolios),
            "std_final_portfolio": np.std(final_portfolios),
            "mean_total_return": np.mean(total_returns),
            "std_total_return": np.std(total_returns),
            "min_final_portfolio": np.min(final_portfolios),
            "max_final_portfolio": np.max(final_portfolios),
            "percentile_5": np.percentile(final_portfolios, 5),
            "percentile_95": np.percentile(final_portfolios, 95),
        }
