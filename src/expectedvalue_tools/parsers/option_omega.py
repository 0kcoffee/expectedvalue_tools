"""Option Omega CSV parser."""

import os
from pathlib import Path
from typing import Dict
import pandas as pd
from .base import BaseParser


class OptionOmegaParser(BaseParser):
    """Parser for Option Omega CSV files."""

    REQUIRED_COLUMNS = ["P/L", "No. of Contracts", "Strategy"]

    def parse(self, file_path: str) -> pd.DataFrame:
        """
        Load and parse an Option Omega CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with parsed data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)
        self.validate(df)
        return df

    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has required Option Omega columns.

        Args:
            data: DataFrame to validate

        Returns:
            True if valid

        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = [
            col for col in self.REQUIRED_COLUMNS if col not in data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return True

    def detect_format(self, file_path: str) -> bool:
        """
        Check if file is an Option Omega CSV by checking for required columns.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this appears to be an Option Omega CSV
        """
        try:
            # Read just the header to check columns
            df = pd.read_csv(file_path, nrows=0)
            return all(col in df.columns for col in self.REQUIRED_COLUMNS)
        except Exception:
            return False

    def get_metadata(self, data: pd.DataFrame) -> Dict:
        """
        Extract metadata from Option Omega data.

        Args:
            data: Parsed DataFrame

        Returns:
            Dictionary with metadata including is_portfolio and strategy_count
        """
        strategy_column = data["Strategy"]
        unique_strategies = strategy_column.dropna().unique()
        unique_strategies = [s for s in unique_strategies if str(s).strip() != ""]

        is_portfolio = len(unique_strategies) > 1

        return {
            "is_portfolio": is_portfolio,
            "strategy_count": len(unique_strategies),
            "strategies": unique_strategies if is_portfolio else [unique_strategies[0]] if unique_strategies else [],
        }
