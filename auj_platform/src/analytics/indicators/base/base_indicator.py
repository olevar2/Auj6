"""
Base Indicator Class for AUJ Platform Analytics.

This module provides the abstract base class for all technical indicators
and analytical tools in the AUJ platform.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime


class BaseIndicator(ABC):
    """
    Abstract base class for all indicators in the AUJ platform.
    
    All indicators must implement the calculate method to provide
    standardized indicator computation and result formatting.
    """
    
    def __init__(self, 
                 name: str,
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize base indicator.
        
        Args:
            name: Indicator name
            parameters: Indicator-specific parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.last_calculation_time = None
        self.last_result = None
        
    @abstractmethod
    def calculate(self, 
                  data: Union[pd.DataFrame, Dict[str, Any]], 
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate indicator values.
        
        Args:
            data: Input data (OHLCV DataFrame or other data structure)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing indicator results
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for indicator calculation.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            return False
            
        required_columns = self.get_required_columns()
        return all(col in data.columns for col in required_columns)
    
    def get_required_columns(self) -> List[str]:
        """
        Get list of required data columns.
        
        Returns:
            List of required column names
        """
        return ['close']  # Default minimum requirement
    
    def format_result(self, 
                     raw_result: Any, 
                     confidence: float = 1.0,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format indicator result in standard structure.
        
        Args:
            raw_result: Raw calculation result
            confidence: Confidence level (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            Standardized result dictionary
        """
        result = {
            'indicator_name': self.name,
            'timestamp': datetime.utcnow(),
            'value': raw_result,
            'confidence': max(0.0, min(1.0, confidence)),
            'parameters': self.parameters.copy(),
            'metadata': metadata or {}
        }
        
        self.last_calculation_time = result['timestamp']
        self.last_result = result
        
        return result
    
    def get_display_name(self) -> str:
        """Get formatted display name for the indicator."""
        return self.name.replace('_', ' ').title()
    
    def get_parameter_summary(self) -> str:
        """Get summary of current parameters."""
        if not self.parameters:
            return "Default parameters"
        
        param_strings = []
        for key, value in self.parameters.items():
            param_strings.append(f"{key}={value}")
        
        return f"Parameters: {', '.join(param_strings)}"
    
    def reset(self):
        """Reset indicator state."""
        self.last_calculation_time = None
        self.last_result = None
    
    def __str__(self) -> str:
        """String representation of the indicator."""
        return f"{self.get_display_name()} ({self.get_parameter_summary()})"
    
    def __repr__(self) -> str:
        """Detailed representation of the indicator."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"


class NumericIndicator(BaseIndicator):
    """Base class for indicators that produce numeric values."""
    
    def normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize value to 0-1 range.
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized value between 0 and 1
        """
        if max_val == min_val:
            return 0.5
        
        return (value - min_val) / (max_val - min_val)
    
    def calculate_confidence(self, 
                           current_value: float, 
                           historical_values: List[float],
                           volatility_factor: float = 1.0) -> float:
        """
        Calculate confidence based on historical data stability.
        
        Args:
            current_value: Current indicator value
            historical_values: Historical values
            volatility_factor: Volatility adjustment factor
            
        Returns:
            Confidence level (0.0 to 1.0)
        """
        if len(historical_values) < 2:
            return 0.5
        
        # Calculate standard deviation of historical values
        std_dev = np.std(historical_values)
        mean_val = np.mean(historical_values)
        
        if mean_val == 0:
            return 0.5
        
        # Coefficient of variation
        cv = std_dev / abs(mean_val)
        
        # Higher stability = higher confidence
        confidence = max(0.1, min(1.0, 1.0 - (cv * volatility_factor)))
        
        return confidence


class TechnicalIndicator(NumericIndicator):
    """Base class for technical analysis indicators."""
    
    def get_required_columns(self) -> List[str]:
        """Technical indicators typically need OHLCV data."""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period).mean()
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period).mean()
    
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, 
                                data: pd.Series, 
                                period: int = 20, 
                                std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }


class EconomicIndicator(BaseIndicator):
    """Base class for economic calendar and fundamental analysis indicators."""
    
    def get_required_columns(self) -> List[str]:
        """Economic indicators may not need traditional OHLCV data."""
        return []  # Override in subclasses as needed
    
    def calculate_impact_score(self, 
                             actual: Optional[float], 
                             forecast: Optional[float], 
                             previous: Optional[float]) -> float:
        """
        Calculate economic event impact score.
        
        Args:
            actual: Actual reported value
            forecast: Forecasted value
            previous: Previous value
            
        Returns:
            Impact score (-1.0 to 1.0)
        """
        if actual is None:
            return 0.0
        
        # Compare to forecast if available
        if forecast is not None:
            if forecast == 0:
                return 1.0 if actual > 0 else -1.0
            deviation = (actual - forecast) / abs(forecast)
            return max(-1.0, min(1.0, deviation))
        
        # Compare to previous if forecast not available
        if previous is not None:
            if previous == 0:
                return 1.0 if actual > 0 else -1.0
            change = (actual - previous) / abs(previous)
            return max(-1.0, min(1.0, change))
        
        # No comparison available
        return 0.0
    
    def categorize_impact(self, impact_score: float) -> str:
        """
        Categorize impact score into levels.
        
        Args:
            impact_score: Impact score (-1.0 to 1.0)
            
        Returns:
            Impact category string
        """
        abs_impact = abs(impact_score)
        
        if abs_impact >= 0.5:
            return "HIGH"
        elif abs_impact >= 0.2:
            return "MEDIUM"
        else:
            return "LOW"


# Export all base classes
__all__ = [
    'BaseIndicator',
    'NumericIndicator', 
    'TechnicalIndicator',
    'EconomicIndicator'
]