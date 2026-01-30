"""Abstract base normalizer interface."""

from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class BaseNormalizer(ABC):
    """Abstract base class for data normalizers."""

    @abstractmethod
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to a standard format.

        Args:
            data: DataFrame to normalize

        Returns:
            Normalized DataFrame
        """
        pass

    @abstractmethod
    def get_standard_columns(self) -> List[str]:
        """
        Get the list of standard columns this normalizer produces.

        Returns:
            List of column names
        """
        pass
