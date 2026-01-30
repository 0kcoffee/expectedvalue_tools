"""Abstract base enricher interface."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseEnricher(ABC):
    """Abstract base class for data enrichers."""

    @abstractmethod
    def enrich(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated fields to the data.

        Args:
            data: DataFrame to enrich

        Returns:
            Enriched DataFrame with additional calculated columns
        """
        pass
