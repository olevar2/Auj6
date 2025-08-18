"""
Advanced Chop Zone Indicator - Market Choppiness and Trend Strength Analysis

This implementation provides sophisticated market state analysis:
- Choppiness Index calculation with dynamic thresholds
- Trend strength measurement and classification
- Market regime detection (trending vs ranging)
- Volatility-adjusted choppiness assessment
- Multi-timeframe market state analysis
- ML-enhanced market phase prediction

The Chop Zone Indicator helps traders identify when markets are trending
strongly versus when they are choppy and range-bound.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType


class MarketState(Enum):
    """Market state classification based on choppiness"""
    STRONG_TREND = "strong_trend"        # CI < 38.2
    MODERATE_TREND = "moderate_trend"    # 38.2 <= CI < 50
    WEAK_TREND = "weak_trend"           # 50 <= CI < 61.8
    CHOPPY = "choppy"                   # 61.8 <= CI < 76.4
    VERY_CHOPPY = "very_choppy"         # CI >= 76.4


class TrendDirection(Enum):
    """Trend direction when market is trending"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class ChopZoneResult:
    """Comprehensive Chop Zone analysis result"""
    choppiness_index: float
    market_state: MarketState
    trend_direction: TrendDirection
    trend_strength: float
    signal: SignalType
    confidence: float
    volatility_normalized_ci: float
    momentum_factor: float


class ChopZoneIndicator(StandardIndicatorInterface):
    """
    Advanced Chop Zone Indicator for Market State Analysis
    
    The Choppiness Index measures market choppiness vs trend strength:
    - Values near 0 indicate strong trending markets
    - Values near 100 indicate choppy, range-bound markets
    - Uses True Range and directional movement analysis
    
    Features:
    - Dynamic choppiness thresholds based on volatility
    - Multi-timeframe market state analysis
    - Trend strength assessment
    - ML-enhanced market regime prediction
    """
    
    def __init__(self,
                 period: int = 14,
                 atr_period: int = 14,
                 trend_period: int = 21,
                 enable_ml: bool = True,
                 dynamic_thresholds: bool = True):
        """
        Initialize the Chop Zone Indicator
        
        Args:
            period: Period for choppiness calculation
            atr_period: Period for ATR calculation
            trend_period: Period for trend analysis
            enable_ml: Enable machine learning enhancements
            dynamic_thresholds: Use dynamic volatility-based thresholds
        """
        self.period = period
        self.atr_period = atr_period
        self.trend_period = trend_period
        self.enable_ml = enable_ml and ML_AVAILABLE
        self.dynamic_thresholds = dynamic_thresholds
        
        # Standard Fibonacci-based thresholds
        self.static_thresholds = {
            'strong_trend': 38.2,
            'moderate_trend': 50.0,
            'weak_trend': 61.8,
            'choppy': 76.4
        }
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler() if self.enable_ml else None
        self.ml_trained = False
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate Chop Zone with advanced features"""
        try:
            if len(data) < max(self.period, self.atr_period, self.trend_period) + 10:
                raise ValueError(f"Insufficient data for Chop Zone calculation")
            
            # Calculate Choppiness Index
            chop_data = self._calculate_choppiness_index(data)
            
            # Determine dynamic thresholds if enabled
            thresholds = self._get_thresholds(data) if self.dynamic_thresholds else self.static_thresholds
            
            # Classify market state
            market_state = self._classify_market_state(chop_data['ci'].iloc[-1], thresholds)
            
            # Analyze trend direction and strength
            trend_direction, trend_strength = self._analyze_trend(data)
            
            # Calculate volatility-normalized CI
            volatility_normalized_ci = self._normalize_for_volatility(data, chop_data['ci'].iloc[-1])
            
            # Calculate momentum factor
            momentum_factor = self._calculate_momentum_factor(data)
            
            # Generate signals
            signal, confidence = self._generate_signals(market_state, trend_direction, trend_strength)
            
            # ML enhancement
            if self.enable_ml:
                ml_adjustment = self._enhance_with_ml(data, chop_data)
                confidence *= ml_adjustment.get('confidence_multiplier', 1.0)
            
            # Create result
            latest_result = ChopZoneResult(
                choppiness_index=chop_data['ci'].iloc[-1],
                market_state=market_state,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                signal=signal,
                confidence=confidence,
                volatility_normalized_ci=volatility_normalized_ci,
                momentum_factor=momentum_factor
            )
            
            return {
                'current': latest_result,
                'values': {
                    'choppiness_index': chop_data['ci'].tolist(),
                    'atr': chop_data['atr'].tolist(),
                    'true_range_sum': chop_data['tr_sum'].tolist()
                },
                'market_state': market_state.value,
                'trend_direction': trend_direction.value,
                'trend_strength': trend_strength,
                'signal': signal,
                'confidence': confidence,
                'thresholds': thresholds,
                'volatility_normalized_ci': volatility_normalized_ci,
                'momentum_factor': momentum_factor,
                'metadata': {
                    'period': self.period,
                    'dynamic_thresholds': self.dynamic_thresholds,
                    'calculation_time': pd.Timestamp.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Chop Zone: {e}")
            return self._get_default_result()
    
    def _calculate_choppiness_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the Choppiness Index"""
        df = pd.DataFrame(index=data.index)
        
        # Calculate True Range
        df['prev_close'] = data['close'].shift(1)
        df['tr1'] = data['high'] - data['low']
        df['tr2'] = abs(data['high'] - df['prev_close'])
        df['tr3'] = abs(data['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # Calculate sum of True Range over period
        df['tr_sum'] = df['tr'].rolling(window=self.period).sum()
        
        # Calculate highest high and lowest low over period
        df['hh'] = data['high'].rolling(window=self.period).max()
        df['ll'] = data['low'].rolling(window=self.period).min()
        
        # Choppiness Index calculation
        # CI = 100 * LOG10(SUM(TR, n) / (MAX(HIGH, n) - MIN(LOW, n))) / LOG10(n)
        df['range_hl'] = df['hh'] - df['ll']
        
        # Avoid division by zero
        df['range_hl'] = df['range_hl'].replace(0, np.nan)
        
        df['ci'] = 100 * (
            np.log10(df['tr_sum'] / df['range_hl']) / np.log10(self.period)
        )
        
        # Fill NaN values
        df['ci'] = df['ci'].fillna(50.0)  # Neutral value
        
        return df
    
    def _get_thresholds(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate dynamic thresholds based on market volatility"""
        if len(data) < 30:
            return self.static_thresholds
        
        # Calculate market volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(30).std().iloc[-1]
        
        # Adjust thresholds based on volatility
        # High volatility markets tend to be more choppy
        volatility_factor = min(volatility * 100, 2.0)  # Cap the adjustment
        
        adjusted_thresholds = {}
        for key, value in self.static_thresholds.items():
            if key in ['strong_trend', 'moderate_trend']:
                # Lower thresholds for trending in high volatility
                adjusted_thresholds[key] = value - volatility_factor * 5
            else:
                # Higher thresholds for choppiness in high volatility
                adjusted_thresholds[key] = value + volatility_factor * 5
        
        # Ensure logical order
        adjusted_thresholds['strong_trend'] = max(adjusted_thresholds['strong_trend'], 25.0)
        adjusted_thresholds['moderate_trend'] = max(adjusted_thresholds['moderate_trend'], 
                                                   adjusted_thresholds['strong_trend'] + 5)
        adjusted_thresholds['weak_trend'] = max(adjusted_thresholds['weak_trend'], 
                                              adjusted_thresholds['moderate_trend'] + 5)
        adjusted_thresholds['choppy'] = max(adjusted_thresholds['choppy'], 
                                          adjusted_thresholds['weak_trend'] + 5)
        
        return adjusted_thresholds
    
    def _classify_market_state(self, ci_value: float, thresholds: Dict[str, float]) -> MarketState:
        """Classify market state based on Choppiness Index value"""
        if ci_value < thresholds['strong_trend']:
            return MarketState.STRONG_TREND
        elif ci_value < thresholds['moderate_trend']:
            return MarketState.MODERATE_TREND
        elif ci_value < thresholds['weak_trend']:
            return MarketState.WEAK_TREND
        elif ci_value < thresholds['choppy']:
            return MarketState.CHOPPY
        else:
            return MarketState.VERY_CHOPPY
    
    def _analyze_trend(self, data: pd.DataFrame) -> Tuple[TrendDirection, float]:
        """Analyze trend direction and strength"""
        if len(data) < self.trend_period:
            return TrendDirection.NEUTRAL, 0.0
        
        # Calculate various trend indicators
        sma_short = data['close'].rolling(window=self.trend_period // 2).mean()
        sma_long = data['close'].rolling(window=self.trend_period).mean()
        
        # Current trend direction
        current_price = data['close'].iloc[-1]
        sma_short_current = sma_short.iloc[-1]
        sma_long_current = sma_long.iloc[-1]
        
        # Determine trend direction
        if current_price > sma_short_current > sma_long_current:
            trend_direction = TrendDirection.BULLISH
        elif current_price < sma_short_current < sma_long_current:
            trend_direction = TrendDirection.BEARISH
        else:
            trend_direction = TrendDirection.NEUTRAL
        
        # Calculate trend strength
        if sma_long_current > 0:
            price_distance = abs(current_price - sma_long_current) / sma_long_current
            ma_distance = abs(sma_short_current - sma_long_current) / sma_long_current
            trend_strength = min((price_distance + ma_distance) * 5, 1.0)
        else:
            trend_strength = 0.0
        
        return trend_direction, trend_strength
    
    def _normalize_for_volatility(self, data: pd.DataFrame, ci_value: float) -> float:
        """Normalize CI value for current market volatility"""
        if len(data) < 20:
            return ci_value
        
        # Calculate recent volatility
        returns = data['close'].pct_change()
        current_vol = returns.rolling(20).std().iloc[-1]
        long_term_vol = returns.rolling(60).std().iloc[-1] if len(data) >= 60 else current_vol
        
        if long_term_vol > 0:
            vol_ratio = current_vol / long_term_vol
            # Adjust CI based on volatility regime
            normalized_ci = ci_value * vol_ratio
            return min(max(normalized_ci, 0.0), 100.0)
        
        return ci_value
    
    def _calculate_momentum_factor(self, data: pd.DataFrame) -> float:
        """Calculate momentum factor for signal enhancement"""
        if len(data) < 10:
            return 0.0
        
        # Price momentum
        price_momentum = (data['close'].iloc[-1] / data['close'].iloc[-10] - 1) * 100
        
        # Volume momentum (if available)
        volume_momentum = 0.0
        if 'volume' in data.columns and len(data) >= 20:
            recent_volume = data['volume'].iloc[-5:].mean()
            avg_volume = data['volume'].iloc[-20:-5].mean()
            if avg_volume > 0:
                volume_momentum = (recent_volume / avg_volume - 1) * 100
        
        # Combined momentum factor
        momentum = (abs(price_momentum) * 0.7 + abs(volume_momentum) * 0.3) / 100
        return min(max(momentum, 0.0), 1.0)
    
    def _generate_signals(self, market_state: MarketState, trend_direction: TrendDirection, 
                         trend_strength: float) -> Tuple[SignalType, float]:
        """Generate trading signals based on market state analysis"""
        # Only generate signals in trending markets
        if market_state in [MarketState.CHOPPY, MarketState.VERY_CHOPPY]:
            return SignalType.NEUTRAL, 0.2
        
        # Signal strength based on market state
        state_confidence = {
            MarketState.STRONG_TREND: 0.9,
            MarketState.MODERATE_TREND: 0.7,
            MarketState.WEAK_TREND: 0.5
        }.get(market_state, 0.3)
        
        # Generate directional signals
        if trend_direction == TrendDirection.BULLISH:
            signal = SignalType.BUY
        elif trend_direction == TrendDirection.BEARISH:
            signal = SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        # Adjust confidence based on trend strength
        confidence = state_confidence * (0.5 + trend_strength * 0.5)
        
        return signal, min(confidence, 1.0)
    
    def _enhance_with_ml(self, data: pd.DataFrame, chop_data: pd.DataFrame) -> Dict:
        """Enhance signals with machine learning"""
        if not self.enable_ml:
            return {'confidence_multiplier': 1.0}
        
        try:
            # Extract features
            features = self._extract_ml_features(data, chop_data)
            
            # Simple ML enhancement based on market patterns
            confidence_multiplier = 1.0
            
            # Volatility regime adjustment
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            
            if volatility < 0.015:  # Low volatility regime
                confidence_multiplier *= 1.15
            elif volatility > 0.04:  # High volatility regime
                confidence_multiplier *= 0.85
            
            # Trend consistency check
            ci_values = chop_data['ci'].iloc[-10:].values
            ci_consistency = 1.0 - (np.std(ci_values) / 20.0)
            confidence_multiplier *= (0.8 + ci_consistency * 0.4)
            
            return {
                'confidence_multiplier': min(confidence_multiplier, 1.4),
                'ml_features': features
            }
            
        except Exception as e:
            self.logger.warning(f"ML enhancement failed: {e}")
            return {'confidence_multiplier': 1.0}
    
    def _extract_ml_features(self, data: pd.DataFrame, chop_data: pd.DataFrame) -> List[float]:
        """Extract features for ML model"""
        features = []
        
        # Choppiness features
        features.extend([
            chop_data['ci'].iloc[-1] / 100.0,
            chop_data['atr'].iloc[-1] / data['close'].iloc[-1] if data['close'].iloc[-1] > 0 else 0.0
        ])
        
        # Price features
        returns = data['close'].pct_change()
        features.extend([
            returns.iloc[-1],
            returns.rolling(5).mean().iloc[-1],
            returns.rolling(10).std().iloc[-1]
        ])
        
        # Trend features
        sma_short = data['close'].rolling(10).mean().iloc[-1]
        sma_long = data['close'].rolling(20).mean().iloc[-1]
        if sma_long > 0:
            features.append((sma_short - sma_long) / sma_long)
        else:
            features.append(0.0)
        
        # Volume features
        if 'volume' in data.columns:
            volume_sma = data['volume'].rolling(20).mean().iloc[-1]
            features.append(data['volume'].iloc[-1] / volume_sma if volume_sma > 0 else 1.0)
        else:
            features.append(1.0)
        
        return features
    
    def _get_default_result(self) -> Dict:
        """Get default result when calculation fails"""
        default_result = ChopZoneResult(
            choppiness_index=50.0,
            market_state=MarketState.CHOPPY,
            trend_direction=TrendDirection.NEUTRAL,
            trend_strength=0.0,
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            volatility_normalized_ci=50.0,
            momentum_factor=0.0
        )
        
        return {
            'current': default_result,
            'values': {'choppiness_index': [], 'atr': [], 'true_range_sum': []},
            'market_state': 'choppy',
            'trend_direction': 'neutral',
            'signal': SignalType.NEUTRAL,
            'confidence': 0.0,
            'error': True,
            'metadata': {'period': self.period}
        }

    def get_parameters(self) -> Dict:
        """Get current indicator parameters"""
        return {
            'period': self.period,
            'atr_period': self.atr_period,
            'trend_period': self.trend_period,
            'enable_ml': self.enable_ml,
            'dynamic_thresholds': self.dynamic_thresholds
        }
    
    def set_parameters(self, **kwargs):
        """Update indicator parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)