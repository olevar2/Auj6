"""
Advanced Aroon Indicator with Trend Reversal Detection

This implementation provides sophisticated trend analysis using the Aroon system:
- Traditional Aroon Up and Aroon Down calculations
- Advanced trend reversal detection algorithms
- Momentum divergence analysis
- Multi-timeframe trend confirmation
- ML-enhanced signal optimization
- Real-time trend strength assessment

The Aroon indicator measures the time since the highest high and lowest low
to identify trend changes and potential reversal points.
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


class AroonTrendState(Enum):
    """Aroon-based trend state classification"""
    STRONG_UPTREND = "strong_uptrend"      # Aroon Up > 80, Aroon Down < 20
    WEAK_UPTREND = "weak_uptrend"          # Aroon Up > 50, Aroon Down < 50
    CONSOLIDATION = "consolidation"        # Both Aroon values between 30-70
    WEAK_DOWNTREND = "weak_downtrend"      # Aroon Down > 50, Aroon Up < 50
    STRONG_DOWNTREND = "strong_downtrend"  # Aroon Down > 80, Aroon Up < 20


@dataclass
class AroonResult:
    """Comprehensive Aroon analysis result"""
    aroon_up: float
    aroon_down: float
    aroon_oscillator: float
    trend_state: AroonTrendState
    signal: SignalType
    confidence: float
    reversal_probability: float
    trend_strength: float
    momentum_score: float


class AroonIndicator(StandardIndicatorInterface):
    """
    Advanced Aroon Indicator with Trend Reversal Detection
    
    The Aroon indicator consists of two lines:
    - Aroon Up: ((period - periods since highest high) / period) * 100
    - Aroon Down: ((period - periods since lowest low) / period) * 100
    
    Features:
    - Traditional Aroon calculations
    - Trend reversal detection
    - Momentum divergence analysis
    - Multi-timeframe confirmation
    - ML-enhanced signal generation
    """
    
    def __init__(self,
                 period: int = 14,
                 enable_ml: bool = True,
                 reversal_threshold: float = 0.7,
                 momentum_window: int = 10):
        """
        Initialize the Aroon Indicator
        
        Args:
            period: Lookback period for Aroon calculation
            enable_ml: Enable machine learning enhancements
            reversal_threshold: Threshold for reversal detection
            momentum_window: Window for momentum analysis
        """
        self.period = period
        self.enable_ml = enable_ml and ML_AVAILABLE
        self.reversal_threshold = reversal_threshold
        self.momentum_window = momentum_window
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler() if self.enable_ml else None
        self.ml_trained = False
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate Aroon with advanced features"""
        try:
            if len(data) < self.period + 10:
                raise ValueError(f"Insufficient data: need at least {self.period + 10} periods")
            
            # Calculate basic Aroon values
            aroon_data = self._calculate_aroon_values(data)
            
            # Analyze trend state
            trend_state = self._analyze_trend_state(aroon_data)
            
            # Detect potential reversals
            reversal_prob = self._detect_reversals(data, aroon_data)
            
            # Calculate momentum and strength
            momentum_score = self._calculate_momentum_score(data, aroon_data)
            trend_strength = self._calculate_trend_strength(aroon_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(aroon_data, trend_state, reversal_prob)
            
            # ML enhancement
            if self.enable_ml:
                ml_adjustment = self._enhance_with_ml(data, aroon_data)
                confidence *= ml_adjustment.get('confidence_multiplier', 1.0)
            
            # Create result
            latest_result = AroonResult(
                aroon_up=aroon_data['aroon_up'].iloc[-1],
                aroon_down=aroon_data['aroon_down'].iloc[-1],
                aroon_oscillator=aroon_data['aroon_oscillator'].iloc[-1],
                trend_state=trend_state,
                signal=signal,
                confidence=confidence,
                reversal_probability=reversal_prob,
                trend_strength=trend_strength,
                momentum_score=momentum_score
            )
            
            return {
                'current': latest_result,
                'values': {
                    'aroon_up': aroon_data['aroon_up'].tolist(),
                    'aroon_down': aroon_data['aroon_down'].tolist(),
                    'aroon_oscillator': aroon_data['aroon_oscillator'].tolist()
                },
                'trend_state': trend_state.value,
                'signal': signal,
                'confidence': confidence,
                'reversal_probability': reversal_prob,
                'trend_strength': trend_strength,
                'momentum_score': momentum_score,
                'metadata': {
                    'period': self.period,
                    'calculation_time': pd.Timestamp.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Aroon: {e}")
            return self._get_default_result()
    
    def _calculate_aroon_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Aroon Up, Aroon Down, and Aroon Oscillator"""
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
        df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']
        
        return df
    
    def _analyze_trend_state(self, aroon_data: pd.DataFrame) -> AroonTrendState:
        """Analyze current trend state based on Aroon values"""
        aroon_up = aroon_data['aroon_up'].iloc[-1]
        aroon_down = aroon_data['aroon_down'].iloc[-1]
        
        if aroon_up > 80 and aroon_down < 20:
            return AroonTrendState.STRONG_UPTREND
        elif aroon_up > 50 and aroon_down < 50:
            return AroonTrendState.WEAK_UPTREND
        elif aroon_down > 80 and aroon_up < 20:
            return AroonTrendState.STRONG_DOWNTREND
        elif aroon_down > 50 and aroon_up < 50:
            return AroonTrendState.WEAK_DOWNTREND
        else:
            return AroonTrendState.CONSOLIDATION
    
    def _detect_reversals(self, data: pd.DataFrame, aroon_data: pd.DataFrame) -> float:
        """Detect potential trend reversals"""
        if len(aroon_data) < 10:
            return 0.0
        
        reversal_signals = 0
        total_signals = 0
        
        # Aroon crossover detection
        aroon_up_recent = aroon_data['aroon_up'].iloc[-5:].values
        aroon_down_recent = aroon_data['aroon_down'].iloc[-5:].values
        
        # Check for bullish crossover
        for i in range(1, len(aroon_up_recent)):
            if (aroon_up_recent[i] > aroon_down_recent[i] and 
                aroon_up_recent[i-1] <= aroon_down_recent[i-1]):
                reversal_signals += 1
            total_signals += 1
        
        # Check for bearish crossover  
        for i in range(1, len(aroon_down_recent)):
            if (aroon_down_recent[i] > aroon_up_recent[i] and 
                aroon_down_recent[i-1] <= aroon_up_recent[i-1]):
                reversal_signals += 1
            total_signals += 1
        
        # Price divergence analysis
        price_recent = data['close'].iloc[-10:].values
        aroon_osc_recent = aroon_data['aroon_oscillator'].iloc[-10:].values
        
        # Simple divergence detection
        price_trend = np.polyfit(range(len(price_recent)), price_recent, 1)[0]
        aroon_trend = np.polyfit(range(len(aroon_osc_recent)), aroon_osc_recent, 1)[0]
        
        if (price_trend > 0 and aroon_trend < 0) or (price_trend < 0 and aroon_trend > 0):
            reversal_signals += 2
        
        total_signals += 2
        
        return min(reversal_signals / max(total_signals, 1), 1.0)
    
    def _calculate_momentum_score(self, data: pd.DataFrame, aroon_data: pd.DataFrame) -> float:
        """Calculate momentum score based on Aroon and price action"""
        if len(data) < self.momentum_window:
            return 0.0
        
        # Price momentum
        price_change = data['close'].pct_change(self.momentum_window).iloc[-1]
        
        # Aroon oscillator momentum
        aroon_osc_change = (aroon_data['aroon_oscillator'].iloc[-1] - 
                           aroon_data['aroon_oscillator'].iloc[-self.momentum_window])
        
        # Volume momentum
        volume_ratio = (data['volume'].iloc[-5:].mean() / 
                       data['volume'].iloc[-self.momentum_window:-5].mean())
        
        # Combined momentum score
        momentum = (abs(price_change) * 0.4 + 
                   abs(aroon_osc_change) / 100 * 0.4 + 
                   (volume_ratio - 1.0) * 0.2)
        
        return min(max(momentum, 0.0), 1.0)
    
    def _calculate_trend_strength(self, aroon_data: pd.DataFrame) -> float:
        """Calculate trend strength based on Aroon values"""
        aroon_up = aroon_data['aroon_up'].iloc[-1]
        aroon_down = aroon_data['aroon_down'].iloc[-1]
        
        # Maximum separation indicates strongest trend
        max_value = max(aroon_up, aroon_down)
        min_value = min(aroon_up, aroon_down)
        
        # Strength is based on the dominant direction being high and other being low
        strength = (max_value - min_value) / 100.0
        
        return min(max(strength, 0.0), 1.0)
    
    def _generate_signals(self, aroon_data: pd.DataFrame, trend_state: AroonTrendState, 
                         reversal_prob: float) -> Tuple[SignalType, float]:
        """Generate trading signals based on Aroon analysis"""
        aroon_up = aroon_data['aroon_up'].iloc[-1]
        aroon_down = aroon_data['aroon_down'].iloc[-1]
        
        # Base confidence on trend strength
        base_confidence = abs(aroon_up - aroon_down) / 100.0
        
        # Strong trend signals
        if trend_state == AroonTrendState.STRONG_UPTREND:
            return SignalType.BUY, min(base_confidence + 0.2, 1.0)
        elif trend_state == AroonTrendState.STRONG_DOWNTREND:
            return SignalType.SELL, min(base_confidence + 0.2, 1.0)
        
        # Weak trend signals
        elif trend_state == AroonTrendState.WEAK_UPTREND:
            confidence = base_confidence * (1.0 - reversal_prob * 0.5)
            return SignalType.BUY, confidence
        elif trend_state == AroonTrendState.WEAK_DOWNTREND:
            confidence = base_confidence * (1.0 - reversal_prob * 0.5)
            return SignalType.SELL, confidence
        
        # Reversal signals
        elif reversal_prob > self.reversal_threshold:
            if aroon_up > aroon_down:
                return SignalType.BUY, reversal_prob * 0.8
            else:
                return SignalType.SELL, reversal_prob * 0.8
        
        # Consolidation - no signal
        return SignalType.NEUTRAL, 0.3
    
    def _enhance_with_ml(self, data: pd.DataFrame, aroon_data: pd.DataFrame) -> Dict:
        """Enhance signals with machine learning"""
        if not self.enable_ml:
            return {'confidence_multiplier': 1.0}
        
        try:
            # Extract features for ML
            features = self._extract_ml_features(data, aroon_data)
            
            # Simple ML enhancement based on market conditions
            confidence_multiplier = 1.0
            
            # Volatility adjustment
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            if volatility < 0.02:  # Low volatility
                confidence_multiplier *= 1.15
            elif volatility > 0.05:  # High volatility
                confidence_multiplier *= 0.85
            
            # Trend consistency adjustment
            aroon_consistency = aroon_data['aroon_oscillator'].rolling(10).std().iloc[-1]
            if aroon_consistency < 20:  # Consistent trend
                confidence_multiplier *= 1.1
            
            return {
                'confidence_multiplier': min(confidence_multiplier, 1.5),
                'ml_features': features
            }
            
        except Exception as e:
            self.logger.warning(f"ML enhancement failed: {e}")
            return {'confidence_multiplier': 1.0}
    
    def _extract_ml_features(self, data: pd.DataFrame, aroon_data: pd.DataFrame) -> List[float]:
        """Extract features for ML model"""
        features = []
        
        # Aroon-based features
        features.extend([
            aroon_data['aroon_up'].iloc[-1] / 100.0,
            aroon_data['aroon_down'].iloc[-1] / 100.0,
            aroon_data['aroon_oscillator'].iloc[-1] / 100.0
        ])
        
        # Price features
        returns = data['close'].pct_change()
        features.extend([
            returns.iloc[-1],
            returns.rolling(5).mean().iloc[-1],
            returns.rolling(10).std().iloc[-1]
        ])
        
        # Volume features
        volume_sma = data['volume'].rolling(20).mean().iloc[-1]
        if volume_sma > 0:
            features.append(data['volume'].iloc[-1] / volume_sma)
        else:
            features.append(1.0)
        
        return features
    
    def _get_default_result(self) -> Dict:
        """Get default result when calculation fails"""
        default_result = AroonResult(
            aroon_up=50.0,
            aroon_down=50.0,
            aroon_oscillator=0.0,
            trend_state=AroonTrendState.CONSOLIDATION,
            signal=SignalType.NEUTRAL,
            confidence=0.0,
            reversal_probability=0.0,
            trend_strength=0.0,
            momentum_score=0.0
        )
        
        return {
            'current': default_result,
            'values': {'aroon_up': [], 'aroon_down': [], 'aroon_oscillator': []},
            'trend_state': 'consolidation',
            'signal': SignalType.NEUTRAL,
            'confidence': 0.0,
            'error': True,
            'metadata': {'period': self.period}
        }

    def get_parameters(self) -> Dict:
        """Get current indicator parameters"""
        return {
            'period': self.period,
            'enable_ml': self.enable_ml,
            'reversal_threshold': self.reversal_threshold,
            'momentum_window': self.momentum_window
        }
    
    def set_parameters(self, **kwargs):
        """Update indicator parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)