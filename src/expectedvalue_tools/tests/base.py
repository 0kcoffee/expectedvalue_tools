"""Abstract base test interface."""

from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd


class BaseTest(ABC):
    """Abstract base class for statistical tests."""

    @abstractmethod
    def run(self, data: pd.DataFrame, **kwargs) -> Dict:
        """
        Execute the test on the provided data.

        Args:
            data: DataFrame with normalized and enriched data
            **kwargs: Additional test-specific parameters

        Returns:
            Dictionary with test results
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the test.

        Returns:
            Test name (used in CLI)
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of what the test does.

        Returns:
            Test description
        """
        pass

    def validate_data(self, data: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that data has required columns.

        Args:
            data: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if valid

        Raises:
            ValueError: If required columns are missing
        """
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns for test: {missing}")
        return True
