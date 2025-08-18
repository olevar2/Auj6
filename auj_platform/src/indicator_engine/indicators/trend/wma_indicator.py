"""
Generic WMA Indicator (Alias for Weighted Moving Average)

This is a wrapper/alias for the main WeightedMovingAverageIndicator
to provide both 'wma' and 'weighted_moving_average' naming conventions.
"""

from .weighted_moving_average_indicator import WeightedMovingAverageIndicator

# Create alias for WMA
class WMAIndicator(WeightedMovingAverageIndicator):
    """Alias for WeightedMovingAverageIndicator"""
    pass