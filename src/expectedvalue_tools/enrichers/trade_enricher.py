"""Trade data enricher for adding calculated metrics."""

import numpy as np
import pandas as pd
from .base import BaseEnricher


class TradeEnricher(BaseEnricher):
    """Enricher that adds calculated metrics to trade data."""

    def enrich(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated metrics to trade data.

        Currently adds:
        - Win rate (if not already present)
        - Funds at Open (calculated from Funds at Close and P/L, for backtest dataframes only)
        - Used Allocation (calculated from Margin Req. and Funds at Open, for backtest dataframes only)
        - Additional statistics can be added here

        Args:
            data: DataFrame with trade data (should have "P/L per Contract")

        Returns:
            Enriched DataFrame
        """
        # Create a copy to avoid modifying original
        df = data.copy()

        # Detect trade type (debit vs credit) from Original Premium Sign
        # This column is added by the normalizer before converting premium to absolute values
        if "Original Premium Sign" in df.columns:
            # Credit trades: Original Premium Sign >= 0 (positive or zero)
            # Debit trades: Original Premium Sign < 0 (negative)
            df["Trade Type"] = df["Original Premium Sign"].apply(
                lambda x: "credit" if x >= 0 else "debit"
            )
        else:
            # Default to credit if Original Premium Sign is missing
            df["Trade Type"] = "credit"

        # Calculate win rate if P/L per Contract exists
        if "P/L per Contract" in df.columns:
            if "Win Rate" not in df.columns:
                df["Win Rate"] = (df["P/L per Contract"] > 0).astype(float)

        # Calculate Funds at Open and Used Allocation for backtest dataframes
        # Formula: Funds at Open = Funds at Close - P/L (since P/L = Funds at Close - Funds at Open)
        # Then we calculate the allocation percentage: (Margin Req. / Funds at Open) * 100
        # This shows the percentage of portfolio used at the time of each trade
        if "Funds at Close" in df.columns and "P/L" in df.columns and "Margin Req." in df.columns:
            funds_at_close = df["Funds at Close"]
            pl = df["P/L"]
            margin_req = df["Margin Req."]
            
            # Calculate Funds at Open: Funds at Open = Funds at Close - P/L
            # When P/L is negative (loss), Funds at Close - P/L = Funds at Open (which is higher)
            # When P/L is positive (gain), Funds at Close - P/L = Funds at Open (which is lower)
            funds_at_open = funds_at_close - pl
            funds_at_open = funds_at_open.replace([np.inf, -np.inf], np.nan)
            
            # Add Funds at Open column
            df["Funds at Open"] = funds_at_open.astype(float)
            
            # Calculate allocation percentage: (Margin Req. / Funds at Open) * 100
            # Handle division by zero and invalid values
            used_allocation = (margin_req / funds_at_open) * 100
            used_allocation = used_allocation.replace([np.inf, -np.inf], np.nan)
            used_allocation = used_allocation.fillna(0.0)
            
            df["Used Allocation"] = used_allocation.astype(float)

        return df
