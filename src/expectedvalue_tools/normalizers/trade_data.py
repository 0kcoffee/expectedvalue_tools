"""Trade data normalizer for calculating P/L per contract and normalizing data formats."""

from typing import List, Optional
import numpy as np
import pandas as pd
from .base import BaseNormalizer


class TradeDataNormalizer(BaseNormalizer):
    """Normalizer that calculates P/L per contract and normalizes data formats."""

    STANDARD_COLUMNS = [
        "P/L",
        "No. of Contracts",
        "Strategy",
        "P/L per Contract",
        "Premium per Contract",
        "datetime_opened",
    ]

    def normalize(self, data: pd.DataFrame, source: Optional[str] = None) -> pd.DataFrame:
        """
        Normalize trade data by calculating derived columns and normalizing formats.

        Args:
            data: DataFrame with trade data
            source: Optional source identifier ("live" or "backtest"). If None, auto-detects.

        Returns:
            Normalized DataFrame with added columns

        Raises:
            ValueError: If required columns are missing
        """
        # Check for required columns
        required = ["P/L", "No. of Contracts"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns for normalization: {missing}")

        # Create a copy to avoid modifying original
        df = data.copy()

        # Auto-detect source if not provided
        if source is None:
            source = self._detect_source(df)

        # Create datetime column from Date Opened and Time Opened
        if "Date Opened" in df.columns and "Time Opened" in df.columns:
            df["datetime_opened"] = pd.to_datetime(
                df["Date Opened"].astype(str) + " " + df["Time Opened"].astype(str),
                errors="coerce",
            )

        # Normalize premium column
        df = self._normalize_premium(df, source)

        # Normalize closing cost - use absolute values for both live and backtest
        if "Avg. Closing Cost" in df.columns:
            closing_cost_values = df["Avg. Closing Cost"].copy()
            if source == "live":
                # For live data, multiply absolute value by 100
                df["Avg. Closing Cost"] = np.abs(closing_cost_values) * 100
            else:
                # For backtest data, use absolute value (already in correct scale)
                df["Avg. Closing Cost"] = np.abs(closing_cost_values)

        # Calculate Premium per Contract using normalized "Premium" column
        # After _normalize_premium(), "Premium" should always exist for both live and backtest
        if "Premium" in df.columns:
            premium_values = df["Premium"].copy()
            df["Premium per Contract"] = premium_values / df["No. of Contracts"]
            df["Premium per Contract"] = df["Premium per Contract"].replace(
                [np.inf, -np.inf], 0
            )
            df["Premium per Contract"] = df["Premium per Contract"].fillna(0)
        else:
            df["Premium per Contract"] = 0

        # Calculate P/L per contract
        df["P/L per Contract"] = df["P/L"] / df["No. of Contracts"]

        # Handle division by zero (replace inf/nan with 0)
        df["P/L per Contract"] = df["P/L per Contract"].replace([np.inf, -np.inf], 0)
        df["P/L per Contract"] = df["P/L per Contract"].fillna(0)

        # Ensure Strategy column exists (fill empty with empty string)
        if "Strategy" not in df.columns:
            df["Strategy"] = ""
        else:
            df["Strategy"] = df["Strategy"].fillna("").astype(str)

        return df

    def _detect_source(self, df: pd.DataFrame) -> str:
        """
        Auto-detect if data is from live or backtest source.

        Args:
            df: DataFrame to analyze

        Returns:
            "live" or "backtest"
        """
        has_initial_premium = "Initial Premium" in df.columns
        has_premium = "Premium" in df.columns

        # If only Initial Premium exists, likely live
        if has_initial_premium and not has_premium:
            return "live"

        # If only Premium exists, likely backtest
        if has_premium and not has_initial_premium:
            return "backtest"

        # If both exist, check value scales
        if has_initial_premium and has_premium:
            # Check if Initial Premium values are much smaller (live format)
            initial_premium_values = df["Initial Premium"].dropna()
            premium_values = df["Premium"].dropna()

            if len(initial_premium_values) > 0 and len(premium_values) > 0:
                # Live values are typically ~100x smaller
                initial_abs_mean = np.abs(initial_premium_values).mean()
                premium_abs_mean = np.abs(premium_values).mean()

                if initial_abs_mean > 0 and premium_abs_mean > 0:
                    ratio = premium_abs_mean / initial_abs_mean
                    if ratio > 50:  # Premium is much larger, Initial Premium is likely live
                        return "live"
                    elif ratio < 0.02:  # Initial Premium is much larger, Premium is likely live
                        return "backtest"

        # Default to backtest if uncertain
        return "backtest"

    def _normalize_premium(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Normalize premium column based on source.

        Args:
            df: DataFrame to normalize
            source: "live" or "backtest"

        Returns:
            DataFrame with normalized premium columns
        """
        # Handle premium column (backtest uses "Premium", live uses "Initial Premium")
        premium_col = None
        if "Premium" in df.columns:
            premium_col = "Premium"
        elif "Initial Premium" in df.columns:
            premium_col = "Initial Premium"
        else:
            # Try to find similar column
            for col in df.columns:
                if "premium" in col.lower() or "initial" in col.lower():
                    premium_col = col
                    break

        if premium_col:
            premium_values = df[premium_col].copy()

            # Normalize premium using absolute values for both live and backtest
            if source == "live":
                # For live data, multiply absolute value by 100 to match backtest format
                # (live premium is stored as 1/100th of backtest premium)
                premium_values = np.abs(premium_values) * 100
                # Always create/update "Premium" column with normalized values
                df["Premium"] = premium_values
                # Also update "Initial Premium" if it exists
                if "Initial Premium" in df.columns:
                    df["Initial Premium"] = premium_values
            else:
                # For backtest, use absolute value (already in correct scale)
                premium_values = np.abs(premium_values)
                # Standardize column names
                if "Initial Premium" in df.columns and "Premium" not in df.columns:
                    df["Premium"] = premium_values
                elif "Premium" in df.columns and "Initial Premium" not in df.columns:
                    # Update Premium with absolute value
                    df["Premium"] = premium_values
                elif "Initial Premium" in df.columns and "Premium" in df.columns:
                    # Both exist, prefer "Premium" and use absolute value
                    df["Premium"] = premium_values

        return df

    def get_standard_columns(self) -> List[str]:
        """
        Get the list of standard columns this normalizer produces.

        Returns:
            List of column names including "P/L per Contract"
        """
        return self.STANDARD_COLUMNS.copy()
