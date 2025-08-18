"""
Base Indicator Package for AUJ Platform Analytics.

This package provides the base classes and utilities for all indicators
in the AUJ platform analytics system.
"""

from .base_indicator import (
    BaseIndicator,
    NumericIndicator,
    TechnicalIndicator,
    EconomicIndicator
)

__all__ = [
    'BaseIndicator',
    'NumericIndicator',
    'TechnicalIndicator', 
    'EconomicIndicator'
]