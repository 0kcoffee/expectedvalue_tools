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

        # Merge duplicate trades for Option Omega live data (before other normalization)
        if source == "live" and "Time Opened" in df.columns:
            df = self._merge_duplicate_trades(df)

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

    def _merge_duplicate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge trades that opened at the exact same time for Option Omega live data.
        
        Groups rows by "Date Opened" and "Time Opened", validates that Strategy values
        match, and merges them into a single trade according to specific rules.
        
        Args:
            df: DataFrame with trade data
            
        Returns:
            DataFrame with merged trades
        """
        # Check for required columns
        required_cols = ["Date Opened", "Time Opened", "Strategy"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Missing required columns, return original DataFrame
            return df
        
        if len(df) == 0:
            return df
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Group by Date Opened and Time Opened
        grouped = result_df.groupby(["Date Opened", "Time Opened"])
        
        merged_rows = []
        
        for (date_opened, time_opened), group in grouped:
            # Skip single-row groups (no merge needed)
            if len(group) == 1:
                merged_rows.append(group.iloc[0])
                continue
            
            # Validate that all Strategy values are identical
            strategy_values = group["Strategy"].dropna().unique()
            if len(strategy_values) > 1:
                # Strategy values don't match, skip merging for this group
                merged_rows.extend([group.iloc[i] for i in range(len(group))])
                continue
            
            # All validations passed, proceed with merge
            first_row = group.iloc[0]
            merged_row = first_row.copy()
            
            # Strategy: Keep first value (guaranteed identical due to validation)
            # Already set from first_row.copy()
            
            # Date Opened & Time Opened: Keep the grouping key values
            # Already set from first_row.copy()
            
            # Opening Price: Keep first value (should be same)
            # Already set from first_row.copy()
            
            # Legs: Concatenate all leg strings with " | " separator
            if "Legs" in group.columns:
                legs_list = group["Legs"].dropna().tolist()
                if legs_list:
                    merged_row["Legs"] = " | ".join(str(leg) for leg in legs_list)
            
            # Initial Premium: Sum all values
            if "Initial Premium" in group.columns:
                merged_row["Initial Premium"] = group["Initial Premium"].sum()
            
            # No. of Contracts: Take minimum (lower of all values)
            if "No. of Contracts" in group.columns:
                merged_row["No. of Contracts"] = group["No. of Contracts"].min()
            
            # P/L: Sum all values
            if "P/L" in group.columns:
                merged_row["P/L"] = group["P/L"].sum()
            
            # Closing Price, Date Closed, Time Closed: Use value from row with latest "Time Closed"
            if "Date Closed" in group.columns and "Time Closed" in group.columns:
                # Parse datetime for comparison
                group_with_datetime = group.copy()
                group_with_datetime["datetime_closed"] = pd.to_datetime(
                    group_with_datetime["Date Closed"].astype(str) + " " + 
                    group_with_datetime["Time Closed"].astype(str),
                    errors="coerce"
                )
                
                # Find row with latest Time Closed (skip NaT values)
                valid_datetimes = group_with_datetime["datetime_closed"].dropna()
                if len(valid_datetimes) > 0:
                    latest_idx = valid_datetimes.idxmax()
                    latest_row = group.loc[latest_idx]
                    
                    if "Closing Price" in group.columns:
                        merged_row["Closing Price"] = latest_row["Closing Price"]
                    merged_row["Date Closed"] = latest_row["Date Closed"]
                    merged_row["Time Closed"] = latest_row["Time Closed"]
                else:
                    # All datetimes are invalid, use first row
                    if "Closing Price" in group.columns:
                        merged_row["Closing Price"] = first_row["Closing Price"]
                    merged_row["Date Closed"] = first_row["Date Closed"]
                    merged_row["Time Closed"] = first_row["Time Closed"]
            
            # Avg. Closing Cost: Sum all values
            if "Avg. Closing Cost" in group.columns:
                merged_row["Avg. Closing Cost"] = group["Avg. Closing Cost"].sum()
            
            # Reason For Close: Priority: 1) All match, 2) "Stop Loss" if present, 3) Non-"OTO", 4) "OTO"
            if "Reason For Close" in group.columns:
                reason_values = group["Reason For Close"].dropna().unique()
                if len(reason_values) == 1:
                    # All match, use that value
                    merged_row["Reason For Close"] = reason_values[0]
                else:
                    # Check for "Stop Loss" first
                    stop_loss_values = [r for r in reason_values if str(r).strip().upper() == "STOP LOSS"]
                    if stop_loss_values:
                        merged_row["Reason For Close"] = stop_loss_values[0]
                    else:
                        # Not all match and no "Stop Loss", use the one that is not "OTO"
                        non_oto_values = [r for r in reason_values if str(r).strip().upper() != "OTO"]
                        if non_oto_values:
                            merged_row["Reason For Close"] = non_oto_values[0]
                        else:
                            # All are "OTO", use "OTO"
                            merged_row["Reason For Close"] = reason_values[0]
            
            merged_rows.append(merged_row)
        
        # Reconstruct DataFrame from merged rows
        if merged_rows:
            merged_df = pd.DataFrame(merged_rows)
            # Reset index to ensure clean sequential indexing
            merged_df = merged_df.reset_index(drop=True)
            # Preserve all original columns
            for col in df.columns:
                if col not in merged_df.columns:
                    merged_df[col] = None
            # Reorder columns to match original
            merged_df = merged_df[df.columns]
            return merged_df
        else:
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

            # Preserve original premium sign before converting to absolute values
            # This is needed to detect trade type (debit vs credit) in the enricher
            # Store sign: 1 for positive/zero (credit), -1 for negative (debit)
            original_sign = np.sign(premium_values)
            # Replace 0 with 1 (treat zero as credit trade)
            original_sign = original_sign.replace(0, 1)
            df["Original Premium Sign"] = original_sign

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
        else:
            # No premium column found, default to credit (positive sign)
            df["Original Premium Sign"] = 1

        return df

    def get_standard_columns(self) -> List[str]:
        """
        Get the list of standard columns this normalizer produces.

        Returns:
            List of column names including "P/L per Contract"
        """
        return self.STANDARD_COLUMNS.copy()
