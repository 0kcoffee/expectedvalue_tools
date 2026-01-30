"""Abstract base parser interface."""

from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd


class BaseParser(ABC):
    """Abstract base class for data parsers."""

    @abstractmethod
    def parse(self, file_path: str) -> pd.DataFrame:
        """
        Load and parse a file into a DataFrame.

        Args:
            file_path: Path to the file to parse

        Returns:
            DataFrame with parsed data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        pass

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate that the parsed data has the required structure.

        Args:
            data: DataFrame to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If validation fails with details
        """
        pass

    @abstractmethod
    def detect_format(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given file.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this parser can handle the file, False otherwise
        """
        pass

    def get_metadata(self, data: pd.DataFrame) -> dict:
        """
        Extract metadata from the parsed data.

        Args:
            data: Parsed DataFrame

        Returns:
            Dictionary with metadata (e.g., is_portfolio, strategy_count)
        """
        return {}
