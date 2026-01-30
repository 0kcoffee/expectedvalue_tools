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
        - Used Allocation (for backtest dataframes only)
        - Additional statistics can be added here

        Args:
            data: DataFrame with trade data (should have "P/L per Contract")

        Returns:
            Enriched DataFrame
        """
        # Create a copy to avoid modifying original
        df = data.copy()

        # Calculate win rate if P/L per Contract exists
        if "P/L per Contract" in df.columns:
            if "Win Rate" not in df.columns:
                df["Win Rate"] = (df["P/L per Contract"] > 0).astype(float)

        # Calculate Used Allocation for backtest dataframes
        # Formula: (Funds at Close + P/L) / Funds at Close gives us the ratio of Funds at Open to Funds at Close
        # Then we calculate the allocation percentage: (Margin Req. / Funds at Open) * 100
        # This shows the percentage of portfolio used at the time of each trade
        if "Funds at Close" in df.columns and "P/L" in df.columns and "Margin Req." in df.columns:
            funds_at_close = df["Funds at Close"]
            pl = df["P/L"]
            margin_req = df["Margin Req."]
            
            # Calculate: (Funds at close + P/L) / Funds at close = Funds at Open / Funds at Close
            # When P/L is negative (loss), Funds at Close + P/L = Funds at Open
            funds_at_open_ratio = (funds_at_close + pl) / funds_at_close
            funds_at_open_ratio = funds_at_open_ratio.replace([np.inf, -np.inf], np.nan)
            
            # Calculate Funds at Open from the ratio
            funds_at_open = funds_at_close * funds_at_open_ratio
            
            # Calculate allocation percentage: (Margin Req. / Funds at Open) * 100
            # Handle division by zero and invalid values
            used_allocation = (margin_req / funds_at_open) * 100
            used_allocation = used_allocation.replace([np.inf, -np.inf], np.nan)
            used_allocation = used_allocation.fillna(0.0)
            
            df["Used Allocation"] = used_allocation.astype(float)

        return df
