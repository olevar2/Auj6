"""
Momentum Indicator - Simplified Implementation
=============================================

Focused momentum indicator with classic momentum calculation, adaptive periods,
and essential modular features for momentum analysis.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class MomentumIndicator(StandardIndicatorInterface):
    """
    Simplified Momentum Indicator Implementation
    
    Features:
    - Classic momentum calculation as primary output
    - Adaptive period optimization based on market cycles
    - Volume-weighted momentum calculations
    - Essential smoothing and volatility adjustment
    - Simplified threshold detection
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'period': 14,
            'adaptive_periods': True,
            'volume_weighted': True,
            'smoothing_enabled': True,
            'volatility_adjustment': True,
            'min_period': 7,
            'max_period': 28,
            'smoothing_period': 3
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="MomentumIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        
        self.history = {
            'momentum_values': [],
            'smoothed_momentum': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_period = max(self.parameters['period'], self.parameters['max_period'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'volume'],
            min_periods=max_period * 2 + 20,
            lookback_periods=100
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate simplified momentum with essential features"""
        try:
            # Optimize period if enabled
            if self.parameters['adaptive_periods']:
                period = self._optimize_momentum_period(data)
            else:
                period = self.parameters['period']
            
            # Calculate primary momentum
            momentum = self._calculate_classic_momentum(data, period)
            
            # Apply volume weighting if enabled
            if self.parameters['volume_weighted']:
                momentum = self._calculate_volume_weighted_momentum(data, period)
            
            # Apply volatility adjustment
            if self.parameters['volatility_adjustment']:
                momentum = self._apply_volatility_adjustment(momentum, data)
            
            # Apply smoothing if enabled
            if self.parameters['smoothing_enabled']:
                smoothed_momentum = momentum.rolling(window=self.parameters['smoothing_period']).mean()
            else:
                smoothed_momentum = momentum
            
            # Calculate simple thresholds
            upper_threshold, lower_threshold = self._calculate_simple_thresholds(momentum)
            
            # Generate signal
            signal, confidence = self._generate_momentum_signal(momentum, smoothed_momentum, upper_threshold, lower_threshold)
            
            # Update history
            current_momentum = float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else 0.0
            current_smoothed = float(smoothed_momentum.iloc[-1]) if not pd.isna(smoothed_momentum.iloc[-1]) else 0.0
            
            self.history['momentum_values'].append(current_momentum)
            self.history['smoothed_momentum'].append(current_smoothed)
            
            # Keep history limited
            if len(self.history['momentum_values']) > 100:
                self.history['momentum_values'] = self.history['momentum_values'][-100:]
                self.history['smoothed_momentum'] = self.history['smoothed_momentum'][-100:]
            
            result = {
                'momentum': current_momentum,
                'smoothed_momentum': current_smoothed,
                'signal': signal,
                'confidence': confidence,
                'period_used': period,
                'thresholds': {
                    'upper': upper_threshold,
                    'lower': lower_threshold
                },
                'trend_direction': 'bullish' if current_momentum > 0 else 'bearish',
                'momentum_strength': min(abs(current_momentum) * 10, 1.0),  # Normalize to 0-1
                'values_history': {
                    'momentum': momentum.tail(20).tolist(),
                    'smoothed_momentum': smoothed_momentum.tail(20).tolist()
                }
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Momentum: {str(e)}",
                cause=e
            )
    
    def _calculate_classic_momentum(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate classic momentum (price change over period)"""
        close = data['close']
        momentum = close / close.shift(period) - 1
        return momentum.fillna(0)
    
    def _optimize_momentum_period(self, data: pd.DataFrame) -> int:
        """Simplified period optimization based on volatility"""
        if len(data) < 40:
            return self.parameters['period']
        
        # Calculate recent volatility
        returns = data['close'].pct_change().tail(30)
        current_vol = returns.std()
        avg_vol = data['close'].pct_change().rolling(window=60).std().mean()
        
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        base_period = self.parameters['period']
        
        # Adjust based on volatility
        if vol_ratio > 1.5:  # High volatility - shorter period
            adjustment = 0.8
        elif vol_ratio < 0.7:  # Low volatility - longer period
            adjustment = 1.2
        else:
            adjustment = 1.0
        
        optimized_period = int(base_period * adjustment)
        
        # Apply bounds
        return max(self.parameters['min_period'], 
                  min(optimized_period, self.parameters['max_period']))
    
    def _calculate_volume_weighted_momentum(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate volume-weighted momentum"""
        close = data['close']
        
        if 'volume' not in data.columns:
            return self._calculate_classic_momentum(data, period)
        
        volume = data['volume']
        
        # Calculate volume-weighted average price over period
        vwap_current = (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        vwap_lagged = vwap_current.shift(period)
        
        # Volume-weighted momentum
        vw_momentum = (vwap_current - vwap_lagged) / vwap_lagged
        
        # Regular momentum for comparison
        regular_momentum = self._calculate_classic_momentum(data, period)
        
        # Blend based on volume significance
        volume_ratio = volume / volume.rolling(window=period).mean()
        volume_weight = np.clip((volume_ratio - 1) * 0.3, 0, 0.7)  # Cap the weight
        
        # Higher volume = more weight on volume-weighted momentum
        blended_momentum = regular_momentum * (1 - volume_weight) + vw_momentum * volume_weight
        
        return blended_momentum.fillna(0)
    
    def _apply_volatility_adjustment(self, momentum: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply simple volatility adjustment to momentum values"""
        if len(momentum) < 20:
            return momentum
        
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # Normalize momentum by volatility
        avg_volatility = volatility.rolling(window=40).mean()
        vol_ratio = volatility / (avg_volatility + 1e-8)
        
        # Adjust momentum values (prevent over-adjustment)
        adjusted_momentum = momentum / np.clip(vol_ratio, 0.5, 2.0)
        
        return adjusted_momentum
    
    def _calculate_simple_thresholds(self, momentum: pd.Series) -> Tuple[float, float]:
        """Calculate simple percentile-based thresholds"""
        if len(momentum) < 20:
            return 0.05, -0.05  # Default thresholds
        
        recent_momentum = momentum.tail(60).dropna()
        
        if len(recent_momentum) < 10:
            return 0.05, -0.05
        
        # Use 80th/20th percentile thresholds
        upper_threshold = np.percentile(recent_momentum, 80)
        lower_threshold = np.percentile(recent_momentum, 20)
        
        # Ensure reasonable separation
        momentum_std = recent_momentum.std()
        min_separation = momentum_std * 0.5
        
        if upper_threshold - lower_threshold < min_separation:
            mid_point = recent_momentum.median()
            upper_threshold = mid_point + min_separation / 2
            lower_threshold = mid_point - min_separation / 2
        
        # Apply reasonable bounds
        upper_threshold = max(0.02, min(upper_threshold, 0.3))
        lower_threshold = min(-0.02, max(lower_threshold, -0.3))
        
        return float(upper_threshold), float(lower_threshold)
    
    def _generate_momentum_signal(self, momentum: pd.Series, smoothed_momentum: pd.Series,
                                 upper_threshold: float, lower_threshold: float) -> Tuple[SignalType, float]:
        """Generate momentum signal"""
        current_momentum = momentum.iloc[-1] if not pd.isna(momentum.iloc[-1]) else 0.0
        current_smoothed = smoothed_momentum.iloc[-1] if not pd.isna(smoothed_momentum.iloc[-1]) else 0.0
        
        signal_components = []
        confidence_components = []
        
        # Basic momentum signals
        if current_momentum > upper_threshold:
            signal_components.append(0.8)
            confidence_components.append(0.7)
        elif current_momentum < lower_threshold:
            signal_components.append(-0.8)
            confidence_components.append(0.7)
        
        # Smoothed momentum confirmation
        if current_smoothed > 0 and current_momentum > 0:
            signal_components.append(0.5)
            confidence_components.append(0.6)
        elif current_smoothed < 0 and current_momentum < 0:
            signal_components.append(-0.5)
            confidence_components.append(0.6)
        
        # Momentum acceleration (simple)
        if len(momentum) >= 2:
            momentum_change = momentum.iloc[-1] - momentum.iloc[-2]
            if abs(momentum_change) > abs(current_momentum) * 0.1:  # Significant acceleration
                signal_components.append(0.3 if momentum_change > 0 else -0.3)
                confidence_components.append(0.5)
        
        # Calculate final signal
        if signal_components and confidence_components:
            weighted_signal = np.average(signal_components, weights=confidence_components)
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        # Determine signal type
        if weighted_signal > 0.5:
            signal = SignalType.BUY
        elif weighted_signal < -0.5:
            signal = SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'momentum_classic',
            'adaptive_periods': self.parameters['adaptive_periods'],
            'volume_weighted': self.parameters['volume_weighted'],
            'smoothing_enabled': self.parameters['smoothing_enabled'],
            'volatility_adjustment': self.parameters['volatility_adjustment'],
            'data_type': 'ohlcv',
            'complexity': 'moderate'
        })
        return base_metadata
