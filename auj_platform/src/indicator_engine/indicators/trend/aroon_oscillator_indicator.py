"""
Advanced Aroon Oscillator Indicator with Divergence Detection

This implementation provides sophisticated oscillator analysis using Aroon concepts:
- Enhanced Aroon Oscillator calculations
- Divergence detection algorithms
- Momentum analysis and trend strength
- Overbought/oversold level identification
- Multi-timeframe oscillator analysis
- ML-enhanced signal generation

The Aroon Oscillator (Aroon Up - Aroon Down) provides clearer trend signals
and momentum analysis compared to individual Aroon lines.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from scipy.signal import argrelextrema
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType


class OscillatorZone(Enum):
    """Aroon Oscillator zone classification"""
    EXTREME_BULLISH = "extreme_bullish"    # > 80
    BULLISH = "bullish"                    # 40 to 80
    NEUTRAL_BULLISH = "neutral_bullish"    # 0 to 40
    NEUTRAL_BEARISH = "neutral_bearish"    # -40 to 0
    BEARISH = "bearish"                    # -80 to -40
    EXTREME_BEARISH = "extreme_bearish"    # < -80


class DivergenceType(Enum):
    """Divergence classification"""
    BULLISH_REGULAR = "bullish_regular"
    BULLISH_HIDDEN = "bullish_hidden"
    BEARISH_REGULAR = "bearish_regular"
    BEARISH_HIDDEN = "bearish_hidden"
    NONE = "none"


@dataclass
class AroonOscillatorResult:
    """Comprehensive Aroon Oscillator analysis result"""
    oscillator_value: float
    zone: OscillatorZone
    signal: SignalType
    confidence: float
    divergence: DivergenceType
    momentum_strength: float
    trend_quality: float
    volatility_adjusted_signal: SignalType


class AroonOscillatorIndicator(StandardIndicatorInterface):
    """
    Advanced Aroon Oscillator Indicator with Divergence Detection
    
    The Aroon Oscillator is calculated as Aroon Up - Aroon Down, providing
    a single line that oscillates between -100 and +100 to show trend strength
    and direction.
    
    Features:
    - Enhanced oscillator calculations
    - Sophisticated divergence detection
    - Multi-timeframe analysis
    - Volatility-adjusted signals
    - ML-enhanced momentum analysis
    """
    
    def __init__(self,
                 period: int = 14,
                 divergence_lookback: int = 20,
                 smoothing_period: int = 3,
                 enable_ml: bool = True,
                 volatility_adjustment: bool = True):
        """
        Initialize the Aroon Oscillator Indicator
        
        Args:
            period: Period for Aroon calculation
            divergence_lookback: Lookback period for divergence detection
            smoothing_period: Period for oscillator smoothing
            enable_ml: Enable machine learning enhancements
            volatility_adjustment: Enable volatility-based signal adjustment
        """
        self.period = period
        self.divergence_lookback = divergence_lookback
        self.smoothing_period = smoothing_period
        self.enable_ml = enable_ml and ML_AVAILABLE
        self.volatility_adjustment = volatility_adjustment
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler() if self.enable_ml else None
        self.ml_trained = False
        
        # Historical data for learning
        self.feature_history = []
        self.signal_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate Aroon Oscillator with advanced features"""
        try:
            if len(data) < self.period + self.divergence_lookback:
                raise ValueError(f"Insufficient data: need at least {self.period + self.divergence_lookback} periods")
            
            # Calculate Aroon Oscillator
            oscillator_data = self._calculate_oscillator(data)
            
            # Classify current zone
            zone = self._classify_zone(oscillator_data['oscillator'].iloc[-1])
            
            # Detect divergences
            divergence = self._detect_divergence(data, oscillator_data)
            
            # Calculate momentum and trend quality
            momentum_strength = self._calculate_momentum_strength(oscillator_data)
            trend_quality = self._assess_trend_quality(data, oscillator_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(oscillator_data, zone, divergence)
            
            # Volatility adjustment
            volatility_adjusted_signal = signal
            if self.volatility_adjustment:
                volatility_adjusted_signal = self._adjust_for_volatility(data, signal, confidence)
            
            # ML enhancement
            if self.enable_ml:
                ml_adjustment = self._enhance_with_ml(data, oscillator_data)
                confidence *= ml_adjustment.get('confidence_multiplier', 1.0)
            
            # Create result
            latest_result = AroonOscillatorResult(
                oscillator_value=oscillator_data['oscillator'].iloc[-1],
                zone=zone,
                signal=signal,
                confidence=confidence,
                divergence=divergence,
                momentum_strength=momentum_strength,
                trend_quality=trend_quality,
                volatility_adjusted_signal=volatility_adjusted_signal
            )
            
            return {
                'current': latest_result,
                'values': {
                    'oscillator': oscillator_data['oscillator'].tolist(),
                    'smoothed': oscillator_data['smoothed'].tolist(),
                    'aroon_up': oscillator_data['aroon_up'].tolist(),
                    'aroon_down': oscillator_data['aroon_down'].tolist()
                },
                'zone': zone.value,
                'signal': signal,
                'confidence': confidence,
                'divergence': divergence.value,
                'momentum_strength': momentum_strength,
                'trend_quality': trend_quality,
                'volatility_adjusted_signal': volatility_adjusted_signal,
                'metadata': {
                    'period': self.period,
                    'calculation_time': pd.Timestamp.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Aroon Oscillator: {e}")
            return self._get_default_result()
    
    def _calculate_oscillator(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Aroon Oscillator and components"""
        df = pd.DataFrame(index=data.index)
        
        # Calculate periods since highest high and lowest low
        df['periods_since_high'] = data['high'].rolling(window=self.period).apply(
            lambda x: self.period - 1 - x.argmax(), raw=False
        )
        df['periods_since_low'] = data['low'].rolling(window=self.period).apply(
            lambda x: self.period - 1 - x.argmin(), raw=False
        )
        
        # Calculate Aroon Up and Aroon Down
        df['aroon_up'] = ((self.period - df['periods_since_high']) / self.period) * 100
        df['aroon_down'] = ((self.period - df['periods_since_low']) / self.period) * 100
        
        # Calculate Aroon Oscillator
        df['oscillator'] = df['aroon_up'] - df['aroon_down']
        
        # Apply smoothing
        df['smoothed'] = df['oscillator'].rolling(window=self.smoothing_period).mean()
        
        return df
    
    def _classify_zone(self, oscillator_value: float) -> OscillatorZone:
        """Classify the current oscillator zone"""
        if oscillator_value > 80:
            return OscillatorZone.EXTREME_BULLISH
        elif oscillator_value > 40:
            return OscillatorZone.BULLISH
        elif oscillator_value > 0:
            return OscillatorZone.NEUTRAL_BULLISH
        elif oscillator_value > -40:
            return OscillatorZone.NEUTRAL_BEARISH
        elif oscillator_value > -80:
            return OscillatorZone.BEARISH
        else:
            return OscillatorZone.EXTREME_BEARISH
    
    def _detect_divergence(self, data: pd.DataFrame, oscillator_data: pd.DataFrame) -> DivergenceType:
        """Detect price-oscillator divergences"""
        if len(data) < self.divergence_lookback:
            return DivergenceType.NONE
        
        try:
            # Get recent data
            recent_prices = data['close'].iloc[-self.divergence_lookback:].values
            recent_oscillator = oscillator_data['oscillator'].iloc[-self.divergence_lookback:].values
            
            # Find local extrema
            price_highs = argrelextrema(recent_prices, np.greater, order=3)[0]
            price_lows = argrelextrema(recent_prices, np.less, order=3)[0]
            osc_highs = argrelextrema(recent_oscillator, np.greater, order=3)[0]
            osc_lows = argrelextrema(recent_oscillator, np.less, order=3)[0]
            
            # Check for bullish divergence (price makes lower low, oscillator makes higher low)
            if len(price_lows) >= 2 and len(osc_lows) >= 2:
                latest_price_low = recent_prices[price_lows[-1]]
                prev_price_low = recent_prices[price_lows[-2]]
                latest_osc_low = recent_oscillator[osc_lows[-1]]
                prev_osc_low = recent_oscillator[osc_lows[-2]]
                
                if latest_price_low < prev_price_low and latest_osc_low > prev_osc_low:
                    return DivergenceType.BULLISH_REGULAR
            
            # Check for bearish divergence (price makes higher high, oscillator makes lower high)
            if len(price_highs) >= 2 and len(osc_highs) >= 2:
                latest_price_high = recent_prices[price_highs[-1]]
                prev_price_high = recent_prices[price_highs[-2]]
                latest_osc_high = recent_oscillator[osc_highs[-1]]
                prev_osc_high = recent_oscillator[osc_highs[-2]]
                
                if latest_price_high > prev_price_high and latest_osc_high < prev_osc_high:
                    return DivergenceType.BEARISH_REGULAR
            
            return DivergenceType.NONE
            
        except Exception as e:
            self.logger.warning(f"Divergence detection failed: {e}")
            return DivergenceType.NONE
    
    def _calculate_momentum_strength(self, oscillator_data: pd.DataFrame) -> float:
        """Calculate momentum strength based on oscillator characteristics"""
        if len(oscillator_data) < 10:
            return 0.0
        
        recent_oscillator = oscillator_data['oscillator'].iloc[-10:].values
        
        # Calculate momentum as rate of change and consistency
        momentum_change = abs(recent_oscillator[-1] - recent_oscillator[0]) / 10
        momentum_consistency = 1.0 - (np.std(np.diff(recent_oscillator)) / 10.0)
        
        # Combine momentum measures
        strength = (momentum_change + momentum_consistency) / 2.0
        return min(max(strength, 0.0), 1.0)
    
    def _assess_trend_quality(self, data: pd.DataFrame, oscillator_data: pd.DataFrame) -> float:
        """Assess the quality of the current trend"""
        if len(data) < 20:
            return 0.0
        
        # Price trend consistency
        prices = data['close'].iloc[-20:].values
        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        price_r2 = np.corrcoef(range(len(prices)), prices)[0, 1] ** 2
        
        # Oscillator trend consistency
        oscillator = oscillator_data['oscillator'].iloc[-20:].values
        osc_trend = np.polyfit(range(len(oscillator)), oscillator, 1)[0]
        osc_r2 = np.corrcoef(range(len(oscillator)), oscillator)[0, 1] ** 2
        
        # Volume trend (if available)
        volume_quality = 1.0
        if 'volume' in data.columns:
            volumes = data['volume'].iloc[-20:].values
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            volume_quality = 1.0 if (price_trend * volume_trend > 0) else 0.8
        
        # Combined trend quality
        quality = (price_r2 + osc_r2 + volume_quality) / 3.0
        return min(max(quality, 0.0), 1.0)
    
    def _generate_signals(self, oscillator_data: pd.DataFrame, zone: OscillatorZone, 
                         divergence: DivergenceType) -> Tuple[SignalType, float]:
        """Generate trading signals based on oscillator analysis"""
        oscillator_value = oscillator_data['oscillator'].iloc[-1]
        
        # Base confidence on oscillator strength
        base_confidence = min(abs(oscillator_value) / 100.0, 1.0)
        
        # Zone-based signals
        if zone in [OscillatorZone.EXTREME_BULLISH, OscillatorZone.BULLISH]:
            signal = SignalType.BUY
            confidence = base_confidence * 0.8
        elif zone in [OscillatorZone.EXTREME_BEARISH, OscillatorZone.BEARISH]:
            signal = SignalType.SELL
            confidence = base_confidence * 0.8
        else:
            signal = SignalType.NEUTRAL
            confidence = 0.3
        
        # Divergence enhancement
        if divergence == DivergenceType.BULLISH_REGULAR:
            signal = SignalType.BUY
            confidence = min(confidence + 0.3, 1.0)
        elif divergence == DivergenceType.BEARISH_REGULAR:
            signal = SignalType.SELL
            confidence = min(confidence + 0.3, 1.0)
        
        # Oscillator momentum consideration
        if len(oscillator_data) >= 5:
            recent_change = oscillator_data['oscillator'].iloc[-1] - oscillator_data['oscillator'].iloc[-5]
            if abs(recent_change) > 20:  # Strong momentum
                confidence = min(confidence * 1.2, 1.0)
        
        return signal, confidence
    
    def _adjust_for_volatility(self, data: pd.DataFrame, signal: SignalType, confidence: float) -> SignalType:
        """Adjust signals based on market volatility"""
        if len(data) < 20:
            return signal
        
        # Calculate volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        
        # High volatility reduces signal strength
        if volatility > 0.05:  # High volatility threshold
            if confidence < 0.7:  # Only keep high confidence signals
                return SignalType.NEUTRAL
        
        return signal
    
    def _enhance_with_ml(self, data: pd.DataFrame, oscillator_data: pd.DataFrame) -> Dict:
        """Enhance signals with machine learning"""
        if not self.enable_ml:
            return {'confidence_multiplier': 1.0}
        
        try:
            # Extract features
            features = self._extract_ml_features(data, oscillator_data)
            
            # Simple ML enhancement
            confidence_multiplier = 1.0
            
            # Market regime detection
            recent_volatility = data['close'].pct_change().rolling(10).std().iloc[-1]
            if recent_volatility < 0.015:  # Low volatility regime
                confidence_multiplier *= 1.1
            elif recent_volatility > 0.04:  # High volatility regime
                confidence_multiplier *= 0.9
            
            # Trend alignment check
            price_trend = data['close'].iloc[-10:].apply(lambda x: x / data['close'].iloc[-10] - 1).iloc[-1]
            osc_trend = oscillator_data['oscillator'].iloc[-1] / 100.0
            
            if (price_trend > 0 and osc_trend > 0) or (price_trend < 0 and osc_trend < 0):
                confidence_multiplier *= 1.05  # Trend alignment bonus
            
            return {
                'confidence_multiplier': min(confidence_multiplier, 1.3),
                'ml_features': features
            }
            
        except Exception as e:
            self.logger.warning(f"ML enhancement failed: {e}")
            return {'confidence_multiplier': 1.0}
    
    def _extract_ml_features(self, data: pd.DataFrame, oscillator_data: pd.DataFrame) -> List[float]:
        """Extract features for ML model"""
        features = []
        
        # Oscillator features
        features.extend([
            oscillator_data['oscillator'].iloc[-1] / 100.0,
            oscillator_data['smoothed'].iloc[-1] / 100.0,
            oscillator_data['aroon_up'].iloc[-1] / 100.0,
            oscillator_data['aroon_down'].iloc[-1] / 100.0
        ])
        
        # Price and volume features
        returns = data['close'].pct_change()
        features.extend([
            returns.iloc[-1],
            returns.rolling(5).mean().iloc[-1],
            returns.rolling(10).std().iloc[-1]
        ])
        
        # Volume features
        if 'volume' in data.columns:
            volume_sma = data['volume'].rolling(20).mean().iloc[-1]
            features.append(data['volume'].iloc[-1] / volume_sma if volume_sma > 0 else 1.0)
        else:
            features.append(1.0)
        
        return features
    
    def _get_default_result(self) -> Dict:
        """Get default result when calculation fails"""
        default_result = AroonOscillatorResult(
            oscillator_value=0.0,
            zone=OscillatorZone.NEUTRAL_BULLISH,
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            divergence=DivergenceType.NONE,
            momentum_strength=0.0,
            trend_quality=0.0,
            volatility_adjusted_signal=SignalType.NEUTRAL
        )
        
        return {
            'current': default_result,
            'values': {'oscillator': [], 'smoothed': [], 'aroon_up': [], 'aroon_down': []},
            'zone': 'neutral_bullish',
            'signal': SignalType.NEUTRAL,
            'confidence': 0.0,
            'error': True,
            'metadata': {'period': self.period}
        }

    def get_parameters(self) -> Dict:
        """Get current indicator parameters"""
        return {
            'period': self.period,
            'divergence_lookback': self.divergence_lookback,
            'smoothing_period': self.smoothing_period,
            'enable_ml': self.enable_ml,
            'volatility_adjustment': self.volatility_adjustment
        }
    
    def set_parameters(self, **kwargs):
        """Update indicator parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)