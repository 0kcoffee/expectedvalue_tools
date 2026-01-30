"""Leg parser for parsing and normalizing option leg strings."""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import pandas as pd


class Leg:
    """Represents a single option leg."""
    
    def __init__(
        self,
        expiry_date: str,
        strike: float,
        option_type: str,
        action: str,
        price: float,
    ):
        """
        Initialize a leg.
        
        Args:
            expiry_date: Expiry date string (e.g., "2022-01-01" or "Jan 1 22")
            strike: Strike price
            option_type: "P" for Put or "C" for Call
            action: "BTO" for Buy To Open or "STO" for Sell To Open
            price: Leg price
        """
        self.expiry_date = expiry_date
        self.strike = strike
        self.option_type = option_type.upper()
        self.action = action.upper()
        self.price = price
    
    def normalize(self) -> Tuple[str, float, str, str]:
        """
        Return normalized leg data (excluding price) for comparison.
        
        Returns:
            Tuple of (expiry_date, strike, option_type, action)
        """
        return (self.expiry_date, self.strike, self.option_type, self.action)
    
    def __eq__(self, other) -> bool:
        """Compare legs based on normalized data (excluding price)."""
        if not isinstance(other, Leg):
            return False
        return self.normalize() == other.normalize()
    
    def __hash__(self) -> int:
        """Hash based on normalized data."""
        return hash(self.normalize())
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Leg({self.expiry_date}, {self.strike}, {self.option_type}, {self.action}, {self.price})"


def parse_leg_string(leg_str: str, trade_year: Optional[int] = None) -> Optional[Leg]:
    """
    Parse a single leg string into a Leg object.
    
    Format: "1 Jan 22 6900 P BTO 6.30" where:
    - 1 = number of contracts (ignored for matching)
    - Jan = month
    - 22 = day of month
    - 6900 = strike
    - P = Put (or C = Call)
    - BTO = Buy To Open (or STO = Sell To Open)
    - 6.30 = price
    
    Args:
        leg_str: Leg string to parse
        trade_year: Year of the trade (used if year not in leg string)
        
    Returns:
        Leg object or None if parsing fails
    """
    if not leg_str or not leg_str.strip():
        return None
    
    # Pattern: contracts month day strike type action price
    # Examples:
    # "1 Jan 22 6900 P BTO 6.30"
    # "2 Jan 15 6950 P BTO 4.50"
    pattern = r'(\d+)\s+([A-Za-z]+)\s+(\d+)\s+(\d+\.?\d*)\s+([PC])\s+(BTO|STO)\s+(\d+\.?\d*)'
    
    match = re.match(pattern, leg_str.strip())
    if not match:
        return None
    
    contracts = int(match.group(1))  # Not used for matching, but parsed for completeness
    month_str = match.group(2)
    day = int(match.group(3))
    strike = float(match.group(4))
    option_type = match.group(5)
    action = match.group(6)
    price = float(match.group(7))
    
    # Convert month string to number
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = month_map.get(month_str.lower(), 1)
    
    # Determine year: use trade_year if provided, otherwise assume same year as trade
    # If day is > 31, it might be a year (2-digit), but we'll treat it as day for now
    if trade_year is not None:
        year = trade_year
    else:
        # Default: assume current year or use a reasonable default
        # For options, expiry is typically in the same year as trade or next year
        year = 2026  # Default fallback
    
    # Format expiry date as YYYY-MM-DD
    try:
        expiry_date = f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, OverflowError):
        # Fallback: use original format
        expiry_date = f"{day} {month_str}"
    
    return Leg(expiry_date, strike, option_type, action, price)


def parse_legs_string(legs_str: str, trade_year: Optional[int] = None) -> List[Leg]:
    """
    Parse a legs string (multiple legs separated by |) into a list of Leg objects.
    
    Format: "1 Jan 22 6900 P BTO 6.30 | 1 Jan 22 6915 C BTO 12.25 | ..."
    
    Args:
        legs_str: Legs string to parse
        trade_year: Year of the trade (used if year not in leg string)
        
    Returns:
        List of Leg objects
    """
    if not legs_str or not legs_str.strip():
        return []
    
    legs = []
    # Split by | and parse each leg
    leg_strings = [s.strip() for s in legs_str.split('|')]
    
    for leg_str in leg_strings:
        if leg_str:
            leg = parse_leg_string(leg_str, trade_year)
            if leg:
                legs.append(leg)
    
    return legs


def normalize_legs(legs: List[Leg]) -> List[Tuple[str, float, str, str]]:
    """
    Normalize a list of legs for comparison (excluding prices).
    
    Args:
        legs: List of Leg objects
        
    Returns:
        List of normalized leg tuples (expiry_date, strike, option_type, action)
    """
    return [leg.normalize() for leg in legs]


def legs_match(legs1: List[Leg], legs2: List[Leg]) -> bool:
    """
    Check if two lists of legs match (ignoring prices).
    
    Args:
        legs1: First list of legs
        legs2: Second list of legs
        
    Returns:
        True if legs match (same expiry, strike, type, action for all legs)
    """
    if len(legs1) != len(legs2):
        return False
    
    # Sort both lists by normalized data for comparison
    normalized1 = sorted(normalize_legs(legs1))
    normalized2 = sorted(normalize_legs(legs2))
    
    return normalized1 == normalized2


def parse_legs_from_dataframe_row(row, legs_column: str = "Legs", date_column: str = "Date Opened") -> List[Leg]:
    """
    Parse legs from a DataFrame row.
    
    Args:
        row: DataFrame row
        legs_column: Name of the column containing legs string
        date_column: Name of the column containing trade date (for year extraction)
        
    Returns:
        List of Leg objects
    """
    legs_str = row.get(legs_column, "")
    if pd.isna(legs_str) if hasattr(pd, 'isna') else (legs_str is None or legs_str == ""):
        return []
    
    # Extract year from trade date if available
    trade_year = None
    if date_column in row.index:
        date_str = row.get(date_column, "")
        if date_str and not (pd.isna(date_str) if hasattr(pd, 'isna') else (date_str is None or date_str == "")):
            try:
                # Try to parse date and extract year
                if isinstance(date_str, str):
                    # Handle formats like "2026-01-22"
                    if '-' in date_str:
                        trade_year = int(date_str.split('-')[0])
                    # Handle other date formats if needed
            except (ValueError, AttributeError):
                pass
    
    return parse_legs_string(str(legs_str), trade_year)
