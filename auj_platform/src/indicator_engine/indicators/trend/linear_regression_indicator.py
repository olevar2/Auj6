"""
Advanced Linear Regression Indicator with Trend Channel Analysis

Features:
- Linear regression line calculation
- R-squared correlation strength
- Price channel boundaries
- Trend line break detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import stats

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class LinearRegressionState(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"

@dataclass
class LinearRegressionResult:
    lr_value: float
    slope: float
    r_squared: float
    upper_channel: float
    lower_channel: float
    state: LinearRegressionState
    signal: SignalType
    confidence: float

class LinearRegressionIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 14, std_dev_multiplier: float = 2.0):
        self.period = period
        self.std_dev_multiplier = std_dev_multiplier
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period + 5:
                raise ValueError("Insufficient data")
            
            # Calculate Linear Regression
            lr_data = self._calculate_linear_regression(data)
            
            # Analyze state
            state = self._analyze_lr_state(data, lr_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, lr_data, state)
            
            latest_result = LinearRegressionResult(
                lr_value=lr_data['lr'].iloc[-1],
                slope=lr_data['slope'].iloc[-1],
                r_squared=lr_data['r_squared'].iloc[-1],
                upper_channel=lr_data['upper_channel'].iloc[-1],
                lower_channel=lr_data['lower_channel'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'lr': lr_data['lr'].tolist(),
                    'upper_channel': lr_data['upper_channel'].tolist(),
                    'lower_channel': lr_data['lower_channel'].tolist(),
                    'slope': lr_data['slope'].tolist(),
                    'r_squared': lr_data['r_squared'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Linear Regression: {e}")
            return self._get_default_result()
    
    def _calculate_linear_regression(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        df['close'] = data['close']
        df['lr'] = np.nan
        df['slope'] = np.nan
        df['r_squared'] = np.nan
        df['std_dev'] = np.nan
        df['upper_channel'] = np.nan
        df['lower_channel'] = np.nan
        
        for i in range(self.period - 1, len(data)):
            # Get data window
            y_values = data['close'].iloc[i - self.period + 1:i + 1].values
            x_values = np.arange(len(y_values))
            
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
            
            # Current regression value (end of line)
            lr_value = slope * (len(y_values) - 1) + intercept
            
            # Calculate standard deviation of residuals
            predicted_values = slope * x_values + intercept
            residuals = y_values - predicted_values
            std_dev = np.std(residuals)
            
            # Store values
            df['lr'].iloc[i] = lr_value
            df['slope'].iloc[i] = slope
            df['r_squared'].iloc[i] = r_value ** 2
            df['std_dev'].iloc[i] = std_dev
            df['upper_channel'].iloc[i] = lr_value + (self.std_dev_multiplier * std_dev)
            df['lower_channel'].iloc[i] = lr_value - (self.std_dev_multiplier * std_dev)
        
        return df
    
    def _analyze_lr_state(self, data: pd.DataFrame, lr_data: pd.DataFrame) -> LinearRegressionState:
        current_price = data['close'].iloc[-1]
        current_lr = lr_data['lr'].iloc[-1]
        slope = lr_data['slope'].iloc[-1]
        r_squared = lr_data['r_squared'].iloc[-1]
        upper_channel = lr_data['upper_channel'].iloc[-1]
        lower_channel = lr_data['lower_channel'].iloc[-1]
        
        if pd.isna(slope) or pd.isna(r_squared):
            return LinearRegressionState.SIDEWAYS
        
        # Check for breakouts
        if current_price > upper_channel:
            return LinearRegressionState.BREAKOUT_UP
        elif current_price < lower_channel:
            return LinearRegressionState.BREAKOUT_DOWN
        
        # Analyze trend based on slope and correlation
        if r_squared > 0.6:  # Strong correlation
            if slope > 0.001:  # Adjust threshold based on price scale
                return LinearRegressionState.STRONG_UPTREND
            elif slope < -0.001:
                return LinearRegressionState.STRONG_DOWNTREND
            else:
                return LinearRegressionState.SIDEWAYS
        else:  # Weak correlation
            if slope > 0.001:
                return LinearRegressionState.UPTREND
            elif slope < -0.001:
                return LinearRegressionState.DOWNTREND
            else:
                return LinearRegressionState.SIDEWAYS
    
    def _generate_signals(self, data: pd.DataFrame, lr_data: pd.DataFrame, 
                         state: LinearRegressionState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        current_lr = lr_data['lr'].iloc[-1]
        r_squared = lr_data['r_squared'].iloc[-1]
        
        if pd.isna(current_lr) or pd.isna(r_squared):
            return SignalType.NEUTRAL, 0.0
        
        # Base confidence on R-squared
        base_confidence = min(0.9, r_squared)
        
        # State-based signals
        if state == LinearRegressionState.BREAKOUT_UP:
            return SignalType.BUY, min(0.9, base_confidence + 0.2)
        elif state == LinearRegressionState.BREAKOUT_DOWN:
            return SignalType.SELL, min(0.9, base_confidence + 0.2)
        elif state == LinearRegressionState.STRONG_UPTREND:
            return SignalType.BUY, base_confidence * 0.8
        elif state == LinearRegressionState.STRONG_DOWNTREND:
            return SignalType.SELL, base_confidence * 0.8
        elif state in [LinearRegressionState.UPTREND]:
            return SignalType.BUY, base_confidence * 0.6
        elif state in [LinearRegressionState.DOWNTREND]:
            return SignalType.SELL, base_confidence * 0.6
        else:
            # Price position relative to regression line
            if current_price > current_lr:
                return SignalType.BUY, base_confidence * 0.4
            else:
                return SignalType.SELL, base_confidence * 0.4
    
    def _get_default_result(self) -> Dict:
        default_result = LinearRegressionResult(0.0, 0.0, 0.0, 0.0, 0.0, 
                                              LinearRegressionState.SIDEWAYS, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {
            'period': self.period,
            'std_dev_multiplier': self.std_dev_multiplier
        }
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)