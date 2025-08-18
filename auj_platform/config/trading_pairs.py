"""
Centralized Trading Pairs Configuration for AUJ Platform

This module provides standardized trading pair definitions and categorization
for consistent use across the entire AUJ Platform ecosystem including:
- Dashboard interface
- Data providers
- Agent analysis
- Risk management
- Performance tracking

All 25 supported trading pairs are organized by category for optimal
portfolio management and risk diversification.
"""

from typing import Dict, List
from enum import Enum


class PairCategory(Enum):
    """Trading pair categories for portfolio organization."""
    MAJOR_PAIRS = "Major Pairs"
    YEN_CROSSES = "Yen Crosses"
    EURO_CROSSES = "Euro Crosses"
    STERLING_CROSSES = "Sterling Crosses"
    COMMODITIES = "Commodities"


# Complete 25-asset portfolio configuration
TRADING_PAIRS = {
    PairCategory.MAJOR_PAIRS: [
        "EUR/USD",  # Euro vs US Dollar
        "GBP/USD",  # British Pound vs US Dollar
        "USD/JPY",  # US Dollar vs Japanese Yen
        "USD/CHF",  # US Dollar vs Swiss Franc
        "AUD/USD",  # Australian Dollar vs US Dollar
        "USD/CAD",  # US Dollar vs Canadian Dollar
        "NZD/USD",  # New Zealand Dollar vs US Dollar
    ],

    PairCategory.YEN_CROSSES: [
        "EUR/JPY",  # Euro vs Japanese Yen
        "GBP/JPY",  # British Pound vs Japanese Yen
        "AUD/JPY",  # Australian Dollar vs Japanese Yen
        "CAD/JPY",  # Canadian Dollar vs Japanese Yen
        "CHF/JPY",  # Swiss Franc vs Japanese Yen
        "NZD/JPY",  # New Zealand Dollar vs Japanese Yen
    ],

    PairCategory.EURO_CROSSES: [
        "EUR/GBP",  # Euro vs British Pound
        "EUR/AUD",  # Euro vs Australian Dollar
        "EUR/CAD",  # Euro vs Canadian Dollar
        "EUR/CHF",  # Euro vs Swiss Franc
        "EUR/NZD",  # Euro vs New Zealand Dollar
    ],

    PairCategory.STERLING_CROSSES: [
        "GBP/AUD",  # British Pound vs Australian Dollar
        "GBP/CAD",  # British Pound vs Canadian Dollar
        "GBP/CHF",  # British Pound vs Swiss Franc
        "GBP/NZD",  # British Pound vs New Zealand Dollar
    ],

    PairCategory.COMMODITIES: [
        "XAU/USD",  # Gold vs US Dollar
        "XAG/USD",  # Silver vs US Dollar
        "WTI/USD",  # West Texas Intermediate Oil vs US Dollar
    ]
}

# Pair metadata for enhanced analysis
PAIR_METADATA = {
    "EUR/USD": {
        "name": "Euro / US Dollar",
        "base_currency": "EUR",
        "quote_currency": "USD",
        "typical_spread": 0.6,
        "volatility": "Medium",
        "session_activity": ["London", "New York"],
        "economic_drivers": ["ECB Policy", "Fed Policy", "EU Economics", "US Economics"]
    },
    "GBP/USD": {
        "name": "British Pound / US Dollar",
        "base_currency": "GBP",
        "quote_currency": "USD",
        "typical_spread": 0.8,
        "volatility": "High",
        "session_activity": ["London", "New York"],
        "economic_drivers": ["BoE Policy", "Fed Policy", "Brexit", "UK Economics"]
    },
    "USD/JPY": {
        "name": "US Dollar / Japanese Yen",
        "base_currency": "USD",
        "quote_currency": "JPY",
        "typical_spread": 0.3,
        "volatility": "Medium",
        "session_activity": ["Tokyo", "London", "New York"],
        "economic_drivers": ["Fed Policy", "BoJ Policy", "US-Japan Trade", "Risk Sentiment"]
    },
    "XAU/USD": {
        "name": "Gold / US Dollar",
        "base_currency": "XAU",
        "quote_currency": "USD",
        "typical_spread": 0.30,
        "volatility": "High",
        "session_activity": ["London", "New York"],
        "economic_drivers": ["Inflation", "Fed Policy", "Risk Sentiment", "Dollar Strength"]
    }
    # Add more metadata as needed
}

# Risk correlation matrix (sample data - would be updated from real market data)
CORRELATION_MATRIX = {
    "EUR/USD": {"GBP/USD": 0.73, "USD/JPY": -0.15, "XAU/USD": 0.42},
    "GBP/USD": {"EUR/USD": 0.73, "USD/JPY": -0.22, "XAU/USD": 0.38},
    "USD/JPY": {"EUR/USD": -0.15, "GBP/USD": -0.22, "XAU/USD": -0.31},
    "XAU/USD": {"EUR/USD": 0.42, "GBP/USD": 0.38, "USD/JPY": -0.31}
}


def get_all_pairs() -> List[str]:
    """
    Get all 25 supported trading pairs.

    Returns:
        List[str]: Complete list of all trading pairs
    """
    all_pairs = []
    for category_pairs in TRADING_PAIRS.values():
        all_pairs.extend(category_pairs)
    return all_pairs


def get_major_pairs() -> List[str]:
    """
    Get the 7 major currency pairs.

    Returns:
        List[str]: Major currency pairs (most liquid)
    """
    return TRADING_PAIRS[PairCategory.MAJOR_PAIRS]


def get_pairs_by_category(category: str) -> List[str]:
    """
    Get trading pairs for a specific category.

    Args:
        category (str): Category name ("Major Pairs", "Yen Crosses", etc.)

    Returns:
        List[str]: Trading pairs in the specified category
    """
    for cat_enum in PairCategory:
        if cat_enum.value == category:
            return TRADING_PAIRS[cat_enum]
    return []


def get_category_for_pair(pair: str) -> str:
    """
    Get the category for a specific trading pair.

    Args:
        pair (str): Trading pair symbol

    Returns:
        str: Category name or "Unknown" if not found
    """
    for category, pairs in TRADING_PAIRS.items():
        if pair in pairs:
            return category.value
    return "Unknown"


def get_pair_metadata(pair: str) -> Dict:
    """
    Get metadata for a specific trading pair.

    Args:
        pair (str): Trading pair symbol

    Returns:
        Dict: Pair metadata or empty dict if not found
    """
    return PAIR_METADATA.get(pair, {})


def get_correlation(pair1: str, pair2: str) -> float:
    """
    Get correlation coefficient between two trading pairs.

    Args:
        pair1 (str): First trading pair
        pair2 (str): Second trading pair

    Returns:
        float: Correlation coefficient (-1 to 1) or 0.0 if not available
    """
    if pair1 in CORRELATION_MATRIX and pair2 in CORRELATION_MATRIX[pair1]:
        return CORRELATION_MATRIX[pair1][pair2]
    elif pair2 in CORRELATION_MATRIX and pair1 in CORRELATION_MATRIX[pair2]:
        return CORRELATION_MATRIX[pair2][pair1]
    else:
        return 0.0


def is_major_pair(pair: str) -> bool:
    """
    Check if a trading pair is a major pair.

    Args:
        pair (str): Trading pair symbol

    Returns:
        bool: True if major pair, False otherwise
    """
    return pair in TRADING_PAIRS[PairCategory.MAJOR_PAIRS]


def is_commodity_pair(pair: str) -> bool:
    """
    Check if a trading pair is a commodity pair.

    Args:
        pair (str): Trading pair symbol

    Returns:
        bool: True if commodity pair, False otherwise
    """
    return pair in TRADING_PAIRS[PairCategory.COMMODITIES]


# Export the category constants for external use
PAIR_CATEGORIES = {
    category.value: pairs for category, pairs in TRADING_PAIRS.items()
}


if __name__ == "__main__":
    """
    Example usage and validation
    """
    print("AUJ Platform Trading Pairs Configuration")
    print("=" * 50)

    print(f"Total pairs supported: {len(get_all_pairs())}")
    print(f"Major pairs: {len(get_major_pairs())}")

    print("\nAll categories:")
    for category in PairCategory:
        pairs = TRADING_PAIRS[category]
        print(f"  {category.value}: {len(pairs)} pairs")
        print(f"    {', '.join(pairs)}")

    print(f"\nSample correlation: EUR/USD vs GBP/USD = {get_correlation('EUR/USD', 'GBP/USD')}")
    print(f"EUR/USD metadata: {get_pair_metadata('EUR/USD')}")
