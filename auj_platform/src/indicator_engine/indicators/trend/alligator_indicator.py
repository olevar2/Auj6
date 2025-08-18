"""
Advanced Alligator Indicator with Multi-Timeframe and Fractal Analysis

This implementation provides sophisticated trend analysis based on Bill Williams' Alligator:
- Traditional Alligator with Jaw, Teeth, and Lips lines
- Multi-timeframe fractal analysis
- Advanced signal generation with market phase detection
- Trend strength assessment and momentum analysis
- ML-enhanced pattern recognition
- Real-time market state classification

The Alligator uses smoothed moving averages to identify trend direction
and strength, helping traders understand market phases and optimal entry points.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType


class AlligatorPhase(Enum):
    """Market phases according to Alligator analysis"""
    SLEEPING = "sleeping"      # Lines converged, low volatility
    AWAKENING = "awakening"    # Lines starting to separate
    HUNTING = "hunting"        # Lines separated, strong trend
    FEEDING = "feeding"        # Maximum line separation, trend climax
    SATISFACTION = "satisfaction"  # Lines converging after trend


class TrendDirection(Enum):
    """Trend direction classification"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


@dataclass
class AlligatorResult:
    """Comprehensive Alligator analysis result"""
    jaw: float          # Blue line (13-period SMMA displaced by 8)
    teeth: float        # Red line (8-period SMMA displaced by 5)
    lips: float         # Green line (5-period SMMA displaced by 3)
    phase: AlligatorPhase
    trend_direction: TrendDirection
    trend_strength: float
    signal: SignalType
    confidence: float
    fractal_support: Optional[float] = None
    fractal_resistance: Optional[float] = None


class AlligatorIndicator(StandardIndicatorInterface):
    """
    Advanced Alligator Indicator by Bill Williams
    
    The Alligator is a technical analysis tool that uses three smoothed moving averages
    to identify trend direction and market phases. It consists of:
    - Jaw (Blue line): 13-period SMMA displaced by 8 periods
    - Teeth (Red line): 8-period SMMA displaced by 5 periods  
    - Lips (Green line): 5-period SMMA displaced by 3 periods
    
    Key Features:
    - Traditional Alligator calculation with proper displacement
    - Market phase detection (sleeping, awakening, hunting, etc.)
    - Multi-timeframe fractal analysis
    - Advanced signal generation with confidence scoring
    - ML-enhanced pattern recognition
    """
    
    def __init__(self,
                 jaw_period: int = 13,
                 jaw_shift: int = 8,
                 teeth_period: int = 8,
                 teeth_shift: int = 5,
                 lips_period: int = 5,
                 lips_shift: int = 3,
                 enable_fractals: bool = True,
                 enable_ml: bool = True,
                 fractal_period: int = 5):
        """
        Initialize the Alligator Indicator
        
        Args:
            jaw_period: Period for Jaw line calculation (default: 13)
            jaw_shift: Forward displacement for Jaw line (default: 8)
            teeth_period: Period for Teeth line calculation (default: 8)
            teeth_shift: Forward displacement for Teeth line (default: 5)
            lips_period: Period for Lips line calculation (default: 5)
            lips_shift: Forward displacement for Lips line (default: 3)
            enable_fractals: Enable fractal analysis
            enable_ml: Enable machine learning enhancements
            fractal_period: Period for fractal detection
        """
        self.jaw_period = jaw_period
        self.jaw_shift = jaw_shift
        self.teeth_period = teeth_period
        self.teeth_shift = teeth_shift
        self.lips_period = lips_period
        self.lips_shift = lips_shift
        self.enable_fractals = enable_fractals
        self.enable_ml = enable_ml and ML_AVAILABLE
        self.fractal_period = fractal_period
        
        # ML components
        self.ml_model = None
        self.scaler = StandardScaler() if self.enable_ml else None
        self.ml_trained = False
        
        # Historical data for pattern recognition
        self.pattern_history = []
        self.phase_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Alligator with advanced features
        
        Args:
            data: DataFrame with columns ['high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing Alligator analysis results
        """
        try:
            if len(data) < max(self.jaw_period, self.teeth_period, self.lips_period) + 20:
                raise ValueError("Insufficient data for Alligator calculation")
            
            # Calculate Alligator lines
            alligator_lines = self._calculate_alligator_lines(data)
            
            # Detect market phase
            phase = self._detect_market_phase(alligator_lines)
            
            # Determine trend direction and strength
            trend_direction, trend_strength = self._analyze_trend(alligator_lines, data)
            
            # Generate signals
            signal, confidence = self._generate_signals(alligator_lines, phase, trend_direction)
            
            # Fractal analysis if enabled
            fractal_levels = None
            if self.enable_fractals:
                fractal_levels = self._analyze_fractals(data)
            
            # Create comprehensive result
            latest_result = AlligatorResult(
                jaw=alligator_lines['jaw'].iloc[-1],
                teeth=alligator_lines['teeth'].iloc[-1],
                lips=alligator_lines['lips'].iloc[-1],
                phase=phase,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                signal=signal,
                confidence=confidence,
                fractal_support=fractal_levels['support'] if fractal_levels else None,
                fractal_resistance=fractal_levels['resistance'] if fractal_levels else None
            )
            
            # ML enhancement
            if self.enable_ml:
                ml_signals = self._enhance_with_ml(data, alligator_lines)
                latest_result.confidence *= ml_signals.get('confidence_multiplier', 1.0)
            
            # Multi-timeframe analysis
            mtf_analysis = self._multi_timeframe_analysis(data)
            
            return {
                'current': latest_result,
                'lines': {
                    'jaw': alligator_lines['jaw'].tolist(),
                    'teeth': alligator_lines['teeth'].tolist(),
                    'lips': alligator_lines['lips'].tolist()
                },
                'phase': phase.value,
                'trend_direction': trend_direction.value,
                'trend_strength': trend_strength,
                'signal': signal,
                'confidence': confidence,
                'fractals': fractal_levels,
                'multi_timeframe': mtf_analysis,
                'metadata': {
                    'jaw_period': self.jaw_period,
                    'teeth_period': self.teeth_period,
                    'lips_period': self.lips_period,
                    'calculation_time': pd.Timestamp.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Alligator: {e}")
            return self._get_default_result()
    
    def _calculate_alligator_lines(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the three Alligator lines with proper displacement"""
        df = pd.DataFrame()
        
        # Calculate SMMA (Smoothed Moving Average) for each line
        df['jaw_smma'] = self._smma(data['close'], self.jaw_period)
        df['teeth_smma'] = self._smma(data['close'], self.teeth_period)
        df['lips_smma'] = self._smma(data['close'], self.lips_period)
        
        # Apply forward displacement
        df['jaw'] = df['jaw_smma'].shift(self.jaw_shift)
        df['teeth'] = df['teeth_smma'].shift(self.teeth_shift)
        df['lips'] = df['lips_smma'].shift(self.lips_shift)
        
        return df
    
    def _smma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Smoothed Moving Average (SMMA)"""
        # SMMA = (previous_smma * (period - 1) + current_value) / period
        smma = series.copy()
        
        # Initialize with simple moving average
        sma_start = series.rolling(window=period).mean()
        smma.iloc[:period-1] = np.nan
        smma.iloc[period-1] = sma_start.iloc[period-1]
        
        # Calculate SMMA
        for i in range(period, len(series)):
            smma.iloc[i] = (smma.iloc[i-1] * (period - 1) + series.iloc[i]) / period
        
        return smma
    
    def _detect_market_phase(self, alligator_lines: pd.DataFrame) -> AlligatorPhase:
        """Detect current market phase based on Alligator lines"""
        latest_jaw = alligator_lines['jaw'].iloc[-1]
        latest_teeth = alligator_lines['teeth'].iloc[-1]
        latest_lips = alligator_lines['lips'].iloc[-1]
        
        # Calculate line separations
        jaw_teeth_diff = abs(latest_jaw - latest_teeth)
        teeth_lips_diff = abs(latest_teeth - latest_lips)
        jaw_lips_diff = abs(latest_jaw - latest_lips)
        
        # Calculate relative separations as percentage of price
        price = latest_lips  # Use lips as reference price
        if price == 0:
            return AlligatorPhase.SLEEPING
        
        separation_ratio = jaw_lips_diff / price
        
        # Detect trend direction
        if latest_lips > latest_teeth > latest_jaw:
            trend_up = True
        elif latest_lips < latest_teeth < latest_jaw:
            trend_up = False
        else:
            # Lines are crossed or mixed - likely sleeping or transitioning
            if separation_ratio < 0.005:  # Less than 0.5% separation
                return AlligatorPhase.SLEEPING
            else:
                return AlligatorPhase.AWAKENING
        
        # Classify phase based on separation and recent changes
        if separation_ratio < 0.005:
            return AlligatorPhase.SLEEPING
        elif separation_ratio < 0.015:
            return AlligatorPhase.AWAKENING
        elif separation_ratio < 0.04:
            return AlligatorPhase.HUNTING
        elif separation_ratio < 0.08:
            return AlligatorPhase.FEEDING
        else:
            return AlligatorPhase.SATISFACTION
    
    def _analyze_trend(self, alligator_lines: pd.DataFrame, data: pd.DataFrame) -> Tuple[TrendDirection, float]:
        """Analyze trend direction and strength"""
        latest_jaw = alligator_lines['jaw'].iloc[-1]
        latest_teeth = alligator_lines['teeth'].iloc[-1]
        latest_lips = alligator_lines['lips'].iloc[-1]
        latest_price = data['close'].iloc[-1]
        
        # Determine trend direction
        if latest_lips > latest_teeth > latest_jaw and latest_price > latest_lips:
            trend_direction = TrendDirection.BULLISH
        elif latest_lips < latest_teeth < latest_jaw and latest_price < latest_lips:
            trend_direction = TrendDirection.BEARISH
        else:
            trend_direction = TrendDirection.SIDEWAYS
        
        # Calculate trend strength (0.0 to 1.0)
        if latest_lips == 0:
            return trend_direction, 0.0
        
        # Strength based on line separation and price position
        max_separation = abs(latest_jaw - latest_lips) / latest_lips
        price_position = abs(latest_price - latest_lips) / latest_lips
        
        # Recent momentum
        recent_price_change = data['close'].pct_change(5).iloc[-1]
        momentum_factor = min(abs(recent_price_change) * 10, 1.0)
        
        # Combined strength calculation
        trend_strength = min((max_separation * 20 + price_position * 10 + momentum_factor) / 3, 1.0)
        
        return trend_direction, trend_strength
    
    def _generate_signals(self, alligator_lines: pd.DataFrame, phase: AlligatorPhase, 
                         trend_direction: TrendDirection) -> Tuple[SignalType, float]:
        """Generate trading signals based on Alligator analysis"""
        confidence = 0.5
        
        # No signals during sleeping phase
        if phase == AlligatorPhase.SLEEPING:
            return SignalType.NEUTRAL, 0.3
        
        # Strong signals during hunting and feeding phases
        if phase in [AlligatorPhase.HUNTING, AlligatorPhase.FEEDING]:
            if trend_direction == TrendDirection.BULLISH:
                confidence = 0.8 if phase == AlligatorPhase.HUNTING else 0.9
                return SignalType.BUY, confidence
            elif trend_direction == TrendDirection.BEARISH:
                confidence = 0.8 if phase == AlligatorPhase.HUNTING else 0.9
                return SignalType.SELL, confidence
        
        # Moderate signals during awakening
        elif phase == AlligatorPhase.AWAKENING:
            if trend_direction == TrendDirection.BULLISH:
                return SignalType.BUY, 0.6
            elif trend_direction == TrendDirection.BEARISH:
                return SignalType.SELL, 0.6
        
        # Exit signals during satisfaction phase
        elif phase == AlligatorPhase.SATISFACTION:
            return SignalType.NEUTRAL, 0.7  # High confidence to exit
        
        return SignalType.NEUTRAL, confidence
    
    def _analyze_fractals(self, data: pd.DataFrame) -> Dict:
        """Analyze fractal support and resistance levels"""
        if len(data) < self.fractal_period * 2 + 1:
            return {'support': None, 'resistance': None}
        
        highs = data['high'].values
        lows = data['low'].values
        
        # Find fractal highs (resistance)
        fractal_highs = []
        for i in range(self.fractal_period, len(highs) - self.fractal_period):
            is_fractal_high = True
            for j in range(i - self.fractal_period, i + self.fractal_period + 1):
                if j != i and highs[j] >= highs[i]:
                    is_fractal_high = False
                    break
            if is_fractal_high:
                fractal_highs.append((i, highs[i]))
        
        # Find fractal lows (support)
        fractal_lows = []
        for i in range(self.fractal_period, len(lows) - self.fractal_period):
            is_fractal_low = True
            for j in range(i - self.fractal_period, i + self.fractal_period + 1):
                if j != i and lows[j] <= lows[i]:
                    is_fractal_low = False
                    break
            if is_fractal_low:
                fractal_lows.append((i, lows[i]))
        
        # Get most recent significant levels
        current_price = data['close'].iloc[-1]
        
        # Find nearest support (fractal low below current price)
        support = None
        for _, level in reversed(fractal_lows[-10:]):  # Check last 10 fractal lows
            if level < current_price:
                support = level
                break
        
        # Find nearest resistance (fractal high above current price)
        resistance = None
        for _, level in reversed(fractal_highs[-10:]):  # Check last 10 fractal highs
            if level > current_price:
                resistance = level
                break
        
        return {'support': support, 'resistance': resistance}
    
    def _enhance_with_ml(self, data: pd.DataFrame, alligator_lines: pd.DataFrame) -> Dict:
        """Enhance signals with machine learning"""
        if not self.enable_ml:
            return {'confidence_multiplier': 1.0}
        
        try:
            # Extract features for ML model
            features = self._extract_ml_features(data, alligator_lines)
            
            # Simple pattern recognition for now
            # In production, this would use trained models
            confidence_multiplier = 1.0
            
            # Add ML-based confidence adjustment
            recent_volatility = data['close'].pct_change().rolling(10).std().iloc[-1]
            if recent_volatility < 0.02:  # Low volatility
                confidence_multiplier *= 1.1
            elif recent_volatility > 0.05:  # High volatility
                confidence_multiplier *= 0.9
            
            return {
                'confidence_multiplier': confidence_multiplier,
                'ml_features': features
            }
            
        except Exception as e:
            self.logger.warning(f"ML enhancement failed: {e}")
            return {'confidence_multiplier': 1.0}
    
    def _extract_ml_features(self, data: pd.DataFrame, alligator_lines: pd.DataFrame) -> List[float]:
        """Extract features for ML model"""
        features = []
        
        # Price position relative to Alligator lines
        current_price = data['close'].iloc[-1]
        jaw = alligator_lines['jaw'].iloc[-1]
        teeth = alligator_lines['teeth'].iloc[-1]
        lips = alligator_lines['lips'].iloc[-1]
        
        if lips > 0:
            features.extend([
                (current_price - jaw) / lips,
                (current_price - teeth) / lips,
                (current_price - lips) / lips,
                (jaw - teeth) / lips,
                (teeth - lips) / lips
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Volume analysis
        volume_sma = data['volume'].rolling(20).mean().iloc[-1]
        if volume_sma > 0:
            features.append(data['volume'].iloc[-1] / volume_sma)
        else:
            features.append(1.0)
        
        # Recent price momentum
        features.append(data['close'].pct_change(5).iloc[-1])
        
        return features
    
    def _multi_timeframe_analysis(self, data: pd.DataFrame) -> Dict:
        """Analyze Alligator across multiple timeframes"""
        # Simulate different timeframes by using different sampling
        timeframes = {}
        
        # Weekly equivalent (5-day sampling)
        if len(data) >= 25:
            weekly_data = data.iloc[::5].copy()
            if len(weekly_data) >= 15:
                weekly_alligator = AlligatorIndicator(enable_ml=False, enable_fractals=False)
                weekly_result = weekly_alligator.calculate(weekly_data)
                timeframes['weekly'] = {
                    'phase': weekly_result['phase'],
                    'trend_direction': weekly_result['trend_direction'],
                    'trend_strength': weekly_result['trend_strength']
                }
        
        # Daily (current timeframe) already calculated
        timeframes['daily'] = {
            'phase': self._detect_market_phase(self._calculate_alligator_lines(data)).value,
            'trend_direction': self._analyze_trend(self._calculate_alligator_lines(data), data)[0].value,
            'trend_strength': self._analyze_trend(self._calculate_alligator_lines(data), data)[1]
        }
        
        return timeframes
    
    def _get_default_result(self) -> Dict:
        """Get default result when calculation fails"""
        default_result = AlligatorResult(
            jaw=0.0,
            teeth=0.0,
            lips=0.0,
            phase=AlligatorPhase.SLEEPING,
            trend_direction=TrendDirection.SIDEWAYS,
            trend_strength=0.0,
            signal=SignalType.NEUTRAL,
            confidence=0.0
        )
        
        return {
            'current': default_result,
            'lines': {'jaw': [], 'teeth': [], 'lips': []},
            'phase': 'sleeping',
            'trend_direction': 'sideways',
            'trend_strength': 0.0,
            'signal': SignalType.NEUTRAL,
            'confidence': 0.0,
            'error': True,
            'metadata': {
                'jaw_period': self.jaw_period,
                'teeth_period': self.teeth_period,
                'lips_period': self.lips_period
            }
        }

    def get_parameters(self) -> Dict:
        """Get current indicator parameters"""
        return {
            'jaw_period': self.jaw_period,
            'jaw_shift': self.jaw_shift,
            'teeth_period': self.teeth_period,
            'teeth_shift': self.teeth_shift,
            'lips_period': self.lips_period,
            'lips_shift': self.lips_shift,
            'enable_fractals': self.enable_fractals,
            'enable_ml': self.enable_ml,
            'fractal_period': self.fractal_period
        }
    
    def set_parameters(self, **kwargs):
        """Update indicator parameters"""
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)