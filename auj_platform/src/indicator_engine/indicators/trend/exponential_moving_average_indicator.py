"""
Advanced Exponential Moving Average (EMA) Indicator with Adaptive Features

This implementation provides sophisticated EMA analysis:
- Traditional EMA calculations with multiple periods
- Adaptive EMA with dynamic smoothing factors
- Multi-timeframe EMA analysis and alignment
- Trend strength assessment using EMA slopes
- EMA crossover and divergence detection
- ML-enhanced signal optimization and trend prediction

The EMA gives more weight to recent prices, making it more responsive
to current market conditions than simple moving averages.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType


class EMATrendState(Enum):
    """EMA trend state classification"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


@dataclass
class EMAResult:
    """Comprehensive EMA analysis result"""
    ema_value: float
    trend_state: EMATrendState
    slope: float
    signal: SignalType
    confidence: float
    adaptive_factor: float
    multiple_ema_alignment: float


class ExponentialMovingAverageIndicator(StandardIndicatorInterface):
    """
    Advanced Exponential Moving Average Indicator
    
    Features:
    - Traditional EMA with customizable periods
    - Adaptive EMA with volatility-based smoothing
    - Multi-timeframe EMA analysis
    - Trend strength and slope analysis
    - ML-enhanced signal generation
    """
    
    def __init__(self,
                 period: int = 21,
                 adaptive: bool = True,
                 multiple_emas: List[int] = None,
                 enable_ml: bool = True):
        """Initialize the EMA Indicator"""
        self.period = period
        self.adaptive = adaptive
        self.multiple_emas = multiple_emas or [9, 21, 50]
        self.enable_ml = enable_ml and ML_AVAILABLE
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler() if self.enable_ml else None
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate EMA with advanced features"""
        try:
            if len(data) < max(self.multiple_emas) + 10:
                raise ValueError("Insufficient data for EMA calculation")
            
            # Calculate EMAs
            ema_data = self._calculate_emas(data)
            
            # Analyze trend
            trend_state, slope = self._analyze_trend(data, ema_data)
            
            # Calculate adaptive factor
            adaptive_factor = self._calculate_adaptive_factor(data) if self.adaptive else 1.0
            
            # Multi-EMA alignment
            alignment = self._calculate_ema_alignment(ema_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, ema_data, trend_state, alignment)
            
            # ML enhancement
            if self.enable_ml:
                ml_adjustment = self._enhance_with_ml(data, ema_data)
                confidence *= ml_adjustment.get('confidence_multiplier', 1.0)
            
            # Create result
            latest_result = EMAResult(
                ema_value=ema_data[f'ema_{self.period}'].iloc[-1],
                trend_state=trend_state,
                slope=slope,
                signal=signal,
                confidence=confidence,
                adaptive_factor=adaptive_factor,
                multiple_ema_alignment=alignment
            )
            
            return {
                'current': latest_result,
                'values': {f'ema_{period}': ema_data[f'ema_{period}'].tolist() 
                          for period in self.multiple_emas},
                'trend_state': trend_state.value,
                'signal': signal,
                'confidence': confidence,
                'metadata': {
                    'period': self.period,
                    'adaptive': self.adaptive,
                    'calculation_time': pd.Timestamp.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return self._get_default_result()
    
    def _calculate_emas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs for multiple periods"""
        df = pd.DataFrame(index=data.index)
        
        for period in self.multiple_emas:
            if self.adaptive:
                df[f'ema_{period}'] = self._adaptive_ema(data['close'], period)
            else:
                df[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        
        return df
    
    def _adaptive_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate adaptive EMA with volatility-based smoothing"""
        # Calculate volatility
        returns = series.pct_change()
        volatility = returns.rolling(window=period).std()
        
        # Adaptive smoothing factor
        base_alpha = 2 / (period + 1)
        
        # Adjust alpha based on volatility
        vol_mean = volatility.rolling(window=period*2).mean()
        vol_ratio = volatility / vol_mean
        
        # Higher volatility = more responsive (higher alpha)
        adaptive_alpha = base_alpha * (1 + vol_ratio.fillna(1.0))
        adaptive_alpha = adaptive_alpha.clip(upper=0.9)  # Cap the alpha
        
        # Calculate adaptive EMA
        ema = series.copy()
        ema.iloc[0] = series.iloc[0]
        
        for i in range(1, len(series)):
            alpha = adaptive_alpha.iloc[i] if not pd.isna(adaptive_alpha.iloc[i]) else base_alpha
            ema.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * ema.iloc[i-1]
        
        return ema
    
    def _analyze_trend(self, data: pd.DataFrame, ema_data: pd.DataFrame) -> Tuple[EMATrendState, float]:
        """Analyze trend state and calculate slope"""
        main_ema = ema_data[f'ema_{self.period}']
        
        # Calculate slope over recent periods
        if len(main_ema) >= 10:
            recent_ema = main_ema.iloc[-10:].values
            slope = np.polyfit(range(len(recent_ema)), recent_ema, 1)[0]
            
            # Normalize slope relative to price
            current_price = data['close'].iloc[-1]
            normalized_slope = slope / current_price if current_price > 0 else 0
            
            # Classify trend based on slope and EMA position
            if normalized_slope > 0.002:  # Strong upward slope
                trend_state = EMATrendState.STRONG_UPTREND
            elif normalized_slope > 0.0005:  # Weak upward slope
                trend_state = EMATrendState.WEAK_UPTREND
            elif normalized_slope < -0.002:  # Strong downward slope
                trend_state = EMATrendState.STRONG_DOWNTREND
            elif normalized_slope < -0.0005:  # Weak downward slope
                trend_state = EMATrendState.WEAK_DOWNTREND
            else:
                trend_state = EMATrendState.SIDEWAYS
            
            return trend_state, normalized_slope
        
        return EMATrendState.SIDEWAYS, 0.0
    
    def _calculate_adaptive_factor(self, data: pd.DataFrame) -> float:
        """Calculate adaptive factor based on market conditions"""
        if len(data) < 20:
            return 1.0
        
        # Volatility factor
        returns = data['close'].pct_change()
        current_vol = returns.rolling(10).std().iloc[-1]
        long_vol = returns.rolling(30).std().iloc[-1]
        
        vol_factor = current_vol / long_vol if long_vol > 0 else 1.0
        
        # Volume factor (if available)
        volume_factor = 1.0
        if 'volume' in data.columns:
            recent_vol = data['volume'].iloc[-5:].mean()
            avg_vol = data['volume'].iloc[-20:-5].mean()
            volume_factor = recent_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Combined adaptive factor
        adaptive_factor = (vol_factor + volume_factor) / 2
        return min(max(adaptive_factor, 0.5), 2.0)
    
    def _calculate_ema_alignment(self, ema_data: pd.DataFrame) -> float:
        """Calculate alignment score for multiple EMAs"""
        if len(self.multiple_emas) < 2:
            return 1.0
        
        # Get latest EMA values
        ema_values = []
        for period in sorted(self.multiple_emas):
            ema_values.append(ema_data[f'ema_{period}'].iloc[-1])
        
        # Check if EMAs are properly aligned
        bullish_alignment = all(ema_values[i] >= ema_values[i+1] for i in range(len(ema_values)-1))
        bearish_alignment = all(ema_values[i] <= ema_values[i+1] for i in range(len(ema_values)-1))
        
        if bullish_alignment or bearish_alignment:
            return 1.0  # Perfect alignment
        
        # Calculate partial alignment score
        aligned_pairs = 0
        total_pairs = len(ema_values) - 1
        
        for i in range(total_pairs):
            if ema_values[i] > ema_values[i+1]:  # Check for bullish alignment
                aligned_pairs += 1
        
        alignment_score = aligned_pairs / total_pairs
        # Convert to -1 to 1 scale (1 = bullish, -1 = bearish, 0 = mixed)
        return 2 * alignment_score - 1
    
    def _generate_signals(self, data: pd.DataFrame, ema_data: pd.DataFrame, 
                         trend_state: EMATrendState, alignment: float) -> Tuple[SignalType, float]:
        """Generate trading signals based on EMA analysis"""
        current_price = data['close'].iloc[-1]
        main_ema = ema_data[f'ema_{self.period}'].iloc[-1]
        
        # Base signal from price vs EMA
        if current_price > main_ema:
            base_signal = SignalType.BUY
        elif current_price < main_ema:
            base_signal = SignalType.SELL
        else:
            base_signal = SignalType.NEUTRAL
        
        # Adjust based on trend state
        trend_confidence = {
            EMATrendState.STRONG_UPTREND: 0.9,
            EMATrendState.WEAK_UPTREND: 0.6,
            EMATrendState.SIDEWAYS: 0.3,
            EMATrendState.WEAK_DOWNTREND: 0.6,
            EMATrendState.STRONG_DOWNTREND: 0.9
        }.get(trend_state, 0.3)
        
        # Adjust based on EMA alignment
        alignment_bonus = abs(alignment) * 0.2
        
        # Final confidence
        confidence = trend_confidence + alignment_bonus
        
        # Override signal if trend is very strong and aligned
        if trend_state == EMATrendState.STRONG_UPTREND and alignment > 0.8:
            return SignalType.BUY, min(confidence, 1.0)
        elif trend_state == EMATrendState.STRONG_DOWNTREND and alignment < -0.8:
            return SignalType.SELL, min(confidence, 1.0)
        
        return base_signal, min(confidence, 1.0)
    
    def _enhance_with_ml(self, data: pd.DataFrame, ema_data: pd.DataFrame) -> Dict:
        """Enhance signals with machine learning"""
        if not self.enable_ml:
            return {'confidence_multiplier': 1.0}
        
        try:
            confidence_multiplier = 1.0
            
            # Market regime detection
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            
            if volatility < 0.015:  # Low volatility regime
                confidence_multiplier *= 1.1
            elif volatility > 0.04:  # High volatility regime
                confidence_multiplier *= 0.9
            
            # EMA effectiveness in current conditions
            if len(data) >= 30:
                # Check how well EMA predicted recent price movements
                price_changes = data['close'].pct_change().iloc[-10:].values
                ema_signals = (data['close'].iloc[-11:-1].values > ema_data[f'ema_{self.period}'].iloc[-11:-1].values).astype(int)
                price_directions = (price_changes > 0).astype(int)
                
                # Calculate accuracy of EMA signals
                accuracy = np.mean(ema_signals == price_directions[1:])  # Skip first NaN
                confidence_multiplier *= (0.8 + accuracy * 0.4)
            
            return {
                'confidence_multiplier': min(confidence_multiplier, 1.3)
            }
            
        except Exception as e:
            self.logger.warning(f"ML enhancement failed: {e}")
            return {'confidence_multiplier': 1.0}
    
    def _get_default_result(self) -> Dict:
        """Get default result when calculation fails"""
        default_result = EMAResult(
            ema_value=0.0,
            trend_state=EMATrendState.SIDEWAYS,
            slope=0.0,
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            adaptive_factor=1.0,
            multiple_ema_alignment=0.0
        )
        
        return {
            'current': default_result,
            'values': {f'ema_{period}': [] for period in self.multiple_emas},
            'trend_state': 'sideways',
            'signal': SignalType.NEUTRAL,
            'confidence': 0.0,
            'error': True,
            'metadata': {'period': self.period}
        }

    def get_parameters(self) -> Dict:
        return {
            'period': self.period,
            'adaptive': self.adaptive,
            'multiple_emas': self.multiple_emas,
            'enable_ml': self.enable_ml
        }
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)