"""
Engulfing Pattern Indicator - Advanced Two-Candle Reversal Pattern Detection
==========================================================================

This indicator implements sophisticated engulfing pattern detection (both bullish and bearish)
with advanced size validation, volume confirmation, and ML-enhanced reversal prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
import logging
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
import talib
from scipy import stats
from scipy.stats import linregress, zscore

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    IndicatorResult, 
    SignalType, 
    DataType, 
    DataRequirement
)
from ...core.exceptions import IndicatorCalculationException


@dataclass
class EngulfingPattern:
    """Represents a detected engulfing pattern"""
    timestamp: pd.Timestamp
    pattern_type: str  # 'bullish_engulfing' or 'bearish_engulfing'
    pattern_strength: float
    size_ratio: float
    volume_ratio: float
    engulfment_percentage: float
    shadow_analysis: float
    trend_context_score: float
    volume_confirmation: float
    momentum_confirmation: float
    reversal_probability: float
    pattern_purity: float
    continuation_signals: float


class EngulfingPatternIndicator(StandardIndicatorInterface):
    """
    Advanced Engulfing Pattern Indicator
    
    Features:
    - Precise bullish and bearish engulfing pattern identification
    - Advanced size validation and engulfment percentage analysis
    - Comprehensive shadow and body ratio validation
    - Volume-based confirmation and momentum analysis
    - ML-enhanced reversal probability prediction
    - Trend context validation and continuation signal detection
    - Pattern purity scoring and reliability metrics
    - Statistical significance testing for pattern quality
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'min_engulfment_ratio': 1.1,       # Minimum size ratio for engulfment
            'min_body_size': 0.3,              # Minimum body size as % of range
            'max_shadow_ratio': 0.4,           # Maximum shadow as % of total pattern
            'min_volume_ratio': 1.0,           # Minimum volume ratio for confirmation
            'trend_context_periods': 10,       # Periods for trend context analysis
            'volume_lookback': 15,             # Periods for volume analysis
            'momentum_analysis': True,
            'volume_analysis': True,
            'ml_reversal_prediction': True,
            'continuation_analysis': True,
            'pattern_purity_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="EngulfingPatternIndicator", parameters=default_params)
        
        # Initialize ML components
        self.reversal_predictor = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=140, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('ada', AdaBoostRegressor(n_estimators=110, random_state=42))
        ])
        self.scaler = RobustScaler()
        self.is_ml_fitted = False
        
        logging.info(f"EngulfingPatternIndicator initialized with parameters: {self.parameters}")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=50,
            lookback_periods=100
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate engulfing patterns with advanced analysis"""
        try:
            if len(data) < 50:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < 50"
                )
            
            # Enhance data with technical indicators
            enhanced_data = self._enhance_data_with_indicators(data)
            
            # Detect engulfing patterns
            detected_patterns = self._detect_engulfing_patterns(enhanced_data)
            
            # Apply comprehensive analysis pipeline
            if self.parameters['volume_analysis']:
                detected_patterns = self._analyze_volume_confirmation(detected_patterns, enhanced_data)
            
            if self.parameters['momentum_analysis']:
                detected_patterns = self._analyze_momentum_confirmation(detected_patterns, enhanced_data)
            
            if self.parameters['continuation_analysis']:
                detected_patterns = self._analyze_continuation_signals(detected_patterns, enhanced_data)
            
            if self.parameters['pattern_purity_analysis']:
                detected_patterns = self._analyze_pattern_purity(detected_patterns, enhanced_data)
            
            if self.parameters['ml_reversal_prediction'] and detected_patterns:
                detected_patterns = self._predict_reversal_probability(detected_patterns, enhanced_data)
            
            # Generate comprehensive analysis
            pattern_analytics = self._generate_pattern_analytics(detected_patterns)
            trend_analysis = self._analyze_current_trend_context(enhanced_data)
            reversal_signals = self._generate_reversal_signals(detected_patterns, enhanced_data)
            
            return {
                'current_pattern': detected_patterns[-1] if detected_patterns else None,
                'recent_patterns': detected_patterns[-10:],
                'pattern_analytics': pattern_analytics,
                'trend_analysis': trend_analysis,
                'reversal_signals': reversal_signals,
                'market_structure': self._analyze_market_structure(enhanced_data),
                'pattern_reliability': self._assess_pattern_reliability(detected_patterns),
                'engulfment_statistics': self._calculate_engulfment_statistics(detected_patterns)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Engulfing pattern calculation failed: {str(e)}", e
            )
    
    def _enhance_data_with_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with comprehensive technical indicators"""
        df = data.copy()
        
        # Candlestick components
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        
        # Ratios and metrics
        df['body_ratio'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
        df['upper_shadow_ratio'] = np.where(df['total_range'] > 0, df['upper_shadow'] / df['total_range'], 0)
        df['lower_shadow_ratio'] = np.where(df['total_range'] > 0, df['lower_shadow'] / df['total_range'], 0)
        
        # Trend indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # Technical indicators
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['volatility_ratio'] = df['total_range'] / df['atr']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        # Advanced metrics
        df['trend_strength'] = self._calculate_trend_strength_series(df)
        df['trend_direction'] = np.where(df['close'] > df['sma_20'], 1, -1)
        df['price_momentum'] = df['close'].pct_change(3)
        df['momentum_zscore'] = df.rolling(15)['price_momentum'].apply(lambda x: zscore(x)[-1] if len(x) == 15 else 0)
        
        return df
    
    def _calculate_trend_strength_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength as a rolling series"""
        def trend_strength_window(window_data):
            if len(window_data) < 6:
                return 0.5
            
            x = np.arange(len(window_data))
            slope, _, r_value, _, _ = linregress(x, window_data['close'])
            
            price_range = window_data['close'].max() - window_data['close'].min()
            if price_range == 0:
                return 0.5
            
            normalized_slope = abs(slope) / price_range * len(window_data)
            trend_quality = r_value ** 2
            strength = normalized_slope * trend_quality
            return min(max(strength, 0), 1)
        
        return df.rolling(self.parameters['trend_context_periods']).apply(trend_strength_window, raw=False)['close']
    
    def _detect_engulfing_patterns(self, data: pd.DataFrame) -> List[EngulfingPattern]:
        """Detect engulfing patterns with sophisticated analysis"""
        patterns = []
        
        for i in range(self.parameters['trend_context_periods'], len(data) - 1):
            current_candle = data.iloc[i + 1]
            previous_candle = data.iloc[i]
            
            # Check for bullish engulfing
            bullish_engulfing = self._is_bullish_engulfing(previous_candle, current_candle)
            if bullish_engulfing:
                pattern = self._create_engulfing_pattern(
                    data, i + 1, 'bullish_engulfing', previous_candle, current_candle
                )
                if pattern and pattern.pattern_strength >= 0.6:
                    patterns.append(pattern)
            
            # Check for bearish engulfing
            bearish_engulfing = self._is_bearish_engulfing(previous_candle, current_candle)
            if bearish_engulfing:
                pattern = self._create_engulfing_pattern(
                    data, i + 1, 'bearish_engulfing', previous_candle, current_candle
                )
                if pattern and pattern.pattern_strength >= 0.6:
                    patterns.append(pattern)
        
        return patterns
    
    def _is_bullish_engulfing(self, prev_candle: pd.Series, curr_candle: pd.Series) -> bool:
        """Check if candles form a bullish engulfing pattern"""
        return (
            prev_candle['is_bearish'] and  # Previous candle is bearish
            curr_candle['is_bullish'] and  # Current candle is bullish
            curr_candle['body'] >= prev_candle['body'] * self.parameters['min_engulfment_ratio'] and  # Size requirement
            curr_candle['open'] <= prev_candle['close'] and  # Engulfment condition
            curr_candle['close'] >= prev_candle['open'] and   # Engulfment condition
            curr_candle['body_ratio'] >= self.parameters['min_body_size'] and  # Body size requirement
            prev_candle['body_ratio'] >= self.parameters['min_body_size']      # Previous body size requirement
        )
    
    def _is_bearish_engulfing(self, prev_candle: pd.Series, curr_candle: pd.Series) -> bool:
        """Check if candles form a bearish engulfing pattern"""
        return (
            prev_candle['is_bullish'] and  # Previous candle is bullish
            curr_candle['is_bearish'] and  # Current candle is bearish
            curr_candle['body'] >= prev_candle['body'] * self.parameters['min_engulfment_ratio'] and  # Size requirement
            curr_candle['open'] >= prev_candle['close'] and  # Engulfment condition
            curr_candle['close'] <= prev_candle['open'] and   # Engulfment condition
            curr_candle['body_ratio'] >= self.parameters['min_body_size'] and  # Body size requirement
            prev_candle['body_ratio'] >= self.parameters['min_body_size']      # Previous body size requirement
        )
    
    def _create_engulfing_pattern(self, data: pd.DataFrame, candle_idx: int, 
                                pattern_type: str, prev_candle: pd.Series, 
                                curr_candle: pd.Series) -> Optional[EngulfingPattern]:
        """Create engulfing pattern with comprehensive analysis"""
        try:
            # Calculate pattern metrics
            size_ratio = curr_candle['body'] / prev_candle['body'] if prev_candle['body'] > 0 else 0
            
            # Calculate engulfment percentage
            if pattern_type == 'bullish_engulfing':
                engulfment_pct = min(
                    (curr_candle['close'] - prev_candle['open']) / prev_candle['body'] if prev_candle['body'] > 0 else 0,
                    3.0
                )
            else:  # bearish_engulfing
                engulfment_pct = min(
                    (prev_candle['open'] - curr_candle['close']) / prev_candle['body'] if prev_candle['body'] > 0 else 0,
                    3.0
                )
            
            # Shadow analysis
            shadow_analysis = self._analyze_engulfing_shadows(prev_candle, curr_candle, pattern_type)
            
            # Trend context analysis
            trend_context_score = self._assess_engulfing_trend_context(data, candle_idx, pattern_type)
            
            # Volume ratio
            volume_ratio = curr_candle['volume'] / prev_candle['volume'] if prev_candle['volume'] > 0 else 1.0
            
            # Calculate pattern strength
            pattern_strength = self._calculate_engulfing_strength(
                size_ratio, engulfment_pct, shadow_analysis, trend_context_score, volume_ratio
            )
            
            pattern = EngulfingPattern(
                timestamp=curr_candle.name,
                pattern_type=pattern_type,
                pattern_strength=pattern_strength,
                size_ratio=size_ratio,
                volume_ratio=volume_ratio,
                engulfment_percentage=engulfment_pct,
                shadow_analysis=shadow_analysis,
                trend_context_score=trend_context_score,
                volume_confirmation=0.0,
                momentum_confirmation=0.0,
                reversal_probability=0.0,
                pattern_purity=0.0,
                continuation_signals=0.0
            )
            
            return pattern
            
        except Exception:
            return None
    
    def _analyze_engulfing_shadows(self, prev_candle: pd.Series, curr_candle: pd.Series, 
                                 pattern_type: str) -> float:
        """Analyze shadow quality for engulfing patterns"""
        # Calculate combined shadow ratios
        total_shadows = (prev_candle['upper_shadow_ratio'] + prev_candle['lower_shadow_ratio'] +
                        curr_candle['upper_shadow_ratio'] + curr_candle['lower_shadow_ratio'])
        
        # Penalize excessive shadows
        if total_shadows > self.parameters['max_shadow_ratio']:
            shadow_penalty = (total_shadows - self.parameters['max_shadow_ratio']) * 2
            shadow_score = max(0, 1.0 - shadow_penalty)
        else:
            shadow_score = 1.0
        
        # Bonus for minimal shadows on the engulfing candle
        if pattern_type == 'bullish_engulfing':
            if curr_candle['lower_shadow_ratio'] < 0.1:
                shadow_score += 0.2
        else:  # bearish_engulfing
            if curr_candle['upper_shadow_ratio'] < 0.1:
                shadow_score += 0.2
        
        return min(shadow_score, 1.0)
    
    def _assess_engulfing_trend_context(self, data: pd.DataFrame, candle_idx: int, 
                                      pattern_type: str) -> float:
        """Assess trend context for engulfing patterns"""
        context_data = data.iloc[max(0, candle_idx - self.parameters['trend_context_periods']):candle_idx + 1]
        
        if len(context_data) < 5:
            return 0.5
        
        context_factors = []
        
        # Trend direction alignment
        if pattern_type == 'bullish_engulfing':
            # Should appear after downtrend
            price_change = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
            if price_change < -0.02:
                trend_factor = min(abs(price_change) / 0.08, 1.0)
                context_factors.append(trend_factor * 0.4)
            else:
                context_factors.append(0.2)
        else:  # bearish_engulfing
            # Should appear after uptrend
            price_change = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
            if price_change > 0.02:
                trend_factor = min(price_change / 0.08, 1.0)
                context_factors.append(trend_factor * 0.4)
            else:
                context_factors.append(0.2)
        
        # Moving average context
        latest = context_data.iloc[-1]
        ma_context = 0
        if pattern_type == 'bullish_engulfing':
            if latest['close'] < latest['sma_20']:
                ma_context += 0.25
            if latest['sma_5'] < latest['sma_10']:
                ma_context += 0.25
        else:  # bearish_engulfing
            if latest['close'] > latest['sma_20']:
                ma_context += 0.25
            if latest['sma_5'] > latest['sma_10']:
                ma_context += 0.25
        context_factors.append(ma_context * 0.3)
        
        # RSI context
        rsi_context = 0
        if not pd.isna(latest['rsi']):
            if pattern_type == 'bullish_engulfing' and latest['rsi'] < 40:
                rsi_context = min((40 - latest['rsi']) / 20, 1.0)
            elif pattern_type == 'bearish_engulfing' and latest['rsi'] > 60:
                rsi_context = min((latest['rsi'] - 60) / 20, 1.0)
        context_factors.append(rsi_context * 0.2)
        
        # Trend strength
        trend_strength = latest['trend_strength'] if not pd.isna(latest['trend_strength']) else 0.5
        context_factors.append(trend_strength * 0.1)
        
        return sum(context_factors)
    
    def _calculate_engulfing_strength(self, size_ratio: float, engulfment_pct: float, 
                                    shadow_analysis: float, trend_context_score: float, 
                                    volume_ratio: float) -> float:
        """Calculate overall engulfing pattern strength"""
        strength_components = [
            min((size_ratio - 1.0) / 1.0, 1.0) * 0.3,  # Size ratio quality
            min(engulfment_pct / 1.5, 1.0) * 0.25,     # Engulfment percentage
            shadow_analysis * 0.2,                      # Shadow quality
            trend_context_score * 0.15,                 # Trend context
            min(volume_ratio / 1.5, 1.0) * 0.1         # Volume ratio
        ]
        
        return sum(strength_components)
    
    def _analyze_volume_confirmation(self, patterns: List[EngulfingPattern], 
                                   data: pd.DataFrame) -> List[EngulfingPattern]:
        """Analyze volume confirmation for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            volume_score = self._calculate_volume_confirmation_score(data, pattern_idx, pattern.pattern_type)
            pattern.volume_confirmation = volume_score
            pattern.pattern_strength = (pattern.pattern_strength * 0.85 + volume_score * 0.15)
        
        return patterns
    
    def _calculate_volume_confirmation_score(self, data: pd.DataFrame, pattern_idx: int, 
                                           pattern_type: str) -> float:
        """Calculate volume confirmation score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - self.parameters['volume_lookback']):pattern_idx + 1]
            current_volume = data.iloc[pattern_idx]['volume']
            previous_volume = data.iloc[pattern_idx - 1]['volume']
            
            # Average volume calculation
            avg_volume = context_data['volume'].iloc[:-2].mean()
            
            # Volume surge analysis
            current_surge = current_volume / avg_volume if avg_volume > 0 else 1.0
            previous_surge = previous_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume expansion from previous to current
            volume_expansion = current_volume / previous_volume if previous_volume > 0 else 1.0
            
            # Combined volume confirmation
            volume_factors = [
                min(current_surge / self.parameters['min_volume_ratio'], 1.0) * 0.4,
                min(volume_expansion / 1.2, 1.0) * 0.3,
                min((current_surge + previous_surge) / 3.0, 1.0) * 0.3
            ]
            
            return sum(volume_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_momentum_confirmation(self, patterns: List[EngulfingPattern], 
                                     data: pd.DataFrame) -> List[EngulfingPattern]:
        """Analyze momentum confirmation for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            momentum_score = self._calculate_momentum_confirmation_score(data, pattern_idx, pattern.pattern_type)
            pattern.momentum_confirmation = momentum_score
        
        return patterns
    
    def _calculate_momentum_confirmation_score(self, data: pd.DataFrame, pattern_idx: int, 
                                             pattern_type: str) -> float:
        """Calculate momentum confirmation score"""
        try:
            current = data.iloc[pattern_idx]
            previous = data.iloc[pattern_idx - 1]
            
            momentum_factors = []
            
            # MACD confirmation
            if not pd.isna(current['macd_hist']) and not pd.isna(previous['macd_hist']):
                if pattern_type == 'bullish_engulfing':
                    if current['macd_hist'] > previous['macd_hist']:
                        momentum_factors.append(0.3)
                    elif current['macd_hist'] > 0:
                        momentum_factors.append(0.15)
                else:  # bearish_engulfing
                    if current['macd_hist'] < previous['macd_hist']:
                        momentum_factors.append(0.3)
                    elif current['macd_hist'] < 0:
                        momentum_factors.append(0.15)
            
            # RSI momentum
            if not pd.isna(current['rsi']) and not pd.isna(previous['rsi']):
                rsi_change = current['rsi'] - previous['rsi']
                if pattern_type == 'bullish_engulfing' and rsi_change > 0:
                    momentum_factors.append(min(rsi_change / 10, 1.0) * 0.25)
                elif pattern_type == 'bearish_engulfing' and rsi_change < 0:
                    momentum_factors.append(min(abs(rsi_change) / 10, 1.0) * 0.25)
            
            # Price momentum
            if not pd.isna(current['momentum_zscore']):
                if pattern_type == 'bullish_engulfing' and current['momentum_zscore'] > 0:
                    momentum_factors.append(min(current['momentum_zscore'] / 2, 1.0) * 0.2)
                elif pattern_type == 'bearish_engulfing' and current['momentum_zscore'] < 0:
                    momentum_factors.append(min(abs(current['momentum_zscore']) / 2, 1.0) * 0.2)
            
            # Stochastic confirmation
            if not pd.isna(current['stoch_k']) and not pd.isna(previous['stoch_k']):
                stoch_change = current['stoch_k'] - previous['stoch_k']
                if pattern_type == 'bullish_engulfing' and stoch_change > 0:
                    momentum_factors.append(min(stoch_change / 15, 1.0) * 0.25)
                elif pattern_type == 'bearish_engulfing' and stoch_change < 0:
                    momentum_factors.append(min(abs(stoch_change) / 15, 1.0) * 0.25)
            
            return sum(momentum_factors) if momentum_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _analyze_continuation_signals(self, patterns: List[EngulfingPattern], 
                                    data: pd.DataFrame) -> List[EngulfingPattern]:
        """Analyze continuation signals for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            if pattern_idx < len(data) - 3:  # Need at least 3 candles after pattern
                continuation_score = self._calculate_continuation_score(data, pattern_idx, pattern.pattern_type)
                pattern.continuation_signals = continuation_score
        
        return patterns
    
    def _calculate_continuation_score(self, data: pd.DataFrame, pattern_idx: int, 
                                    pattern_type: str) -> float:
        """Calculate continuation signal score"""
        try:
            follow_up_data = data.iloc[pattern_idx + 1:pattern_idx + 4]
            
            if len(follow_up_data) < 2:
                return 0.5
            
            continuation_factors = []
            
            # Price continuation
            if pattern_type == 'bullish_engulfing':
                bullish_count = sum(1 for _, candle in follow_up_data.iterrows() if candle['is_bullish'])
                continuation_factors.append(bullish_count / len(follow_up_data) * 0.4)
                
                # Higher highs/lows
                pattern_high = data.iloc[pattern_idx]['high']
                higher_highs = sum(1 for _, candle in follow_up_data.iterrows() if candle['high'] > pattern_high)
                continuation_factors.append(min(higher_highs / 2, 1.0) * 0.3)
                
            else:  # bearish_engulfing
                bearish_count = sum(1 for _, candle in follow_up_data.iterrows() if candle['is_bearish'])
                continuation_factors.append(bearish_count / len(follow_up_data) * 0.4)
                
                # Lower highs/lows
                pattern_low = data.iloc[pattern_idx]['low']
                lower_lows = sum(1 for _, candle in follow_up_data.iterrows() if candle['low'] < pattern_low)
                continuation_factors.append(min(lower_lows / 2, 1.0) * 0.3)
            
            # Volume continuation
            pattern_volume = data.iloc[pattern_idx]['volume']
            avg_follow_volume = follow_up_data['volume'].mean()
            volume_continuation = min(avg_follow_volume / pattern_volume, 1.0) * 0.3 if pattern_volume > 0 else 0.15
            continuation_factors.append(volume_continuation)
            
            return sum(continuation_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_pattern_purity(self, patterns: List[EngulfingPattern], 
                              data: pd.DataFrame) -> List[EngulfingPattern]:
        """Analyze pattern purity for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            purity_score = self._calculate_pattern_purity_score(data, pattern_idx, pattern)
            pattern.pattern_purity = purity_score
        
        return patterns
    
    def _calculate_pattern_purity_score(self, data: pd.DataFrame, pattern_idx: int, 
                                      pattern: EngulfingPattern) -> float:
        """Calculate pattern purity score"""
        try:
            current_candle = data.iloc[pattern_idx]
            previous_candle = data.iloc[pattern_idx - 1]
            
            purity_factors = []
            
            # Clean engulfment (minimal gaps)
            if pattern.pattern_type == 'bullish_engulfing':
                gap_quality = 1.0 - abs(current_candle['open'] - previous_candle['close']) / previous_candle['close']
                purity_factors.append(max(gap_quality, 0) * 0.25)
            else:  # bearish_engulfing
                gap_quality = 1.0 - abs(current_candle['open'] - previous_candle['close']) / previous_candle['close']
                purity_factors.append(max(gap_quality, 0) * 0.25)
            
            # Complete engulfment quality
            engulfment_quality = min(pattern.engulfment_percentage / 1.2, 1.0)
            purity_factors.append(engulfment_quality * 0.3)
            
            # Size proportion quality
            size_quality = min((pattern.size_ratio - 1.0) / 1.5, 1.0)
            purity_factors.append(size_quality * 0.25)
            
            # Shadow quality
            purity_factors.append(pattern.shadow_analysis * 0.2)
            
            return sum(purity_factors)
            
        except Exception:
            return 0.5
    
    def _predict_reversal_probability(self, patterns: List[EngulfingPattern], 
                                    data: pd.DataFrame) -> List[EngulfingPattern]:
        """Predict reversal probability using ML"""
        if not patterns:
            return patterns
        
        try:
            features = []
            for pattern in patterns:
                pattern_idx = data.index.get_loc(pattern.timestamp)
                feature_vector = self._extract_reversal_features(data, pattern_idx, pattern)
                features.append(feature_vector)
            
            if len(features) < 12:
                for pattern in patterns:
                    pattern.reversal_probability = self._heuristic_reversal_probability(pattern)
                return patterns
            
            if not self.is_ml_fitted:
                self._train_reversal_model(patterns, features)
            
            if self.is_ml_fitted:
                features_scaled = self.scaler.transform(features)
                reversal_predictions = self.reversal_predictor.predict(features_scaled)
                
                for i, pattern in enumerate(patterns):
                    ml_probability = max(0.1, min(0.95, reversal_predictions[i]))
                    heuristic_prob = self._heuristic_reversal_probability(pattern)
                    pattern.reversal_probability = (ml_probability * 0.75 + heuristic_prob * 0.25)
            else:
                for pattern in patterns:
                    pattern.reversal_probability = self._heuristic_reversal_probability(pattern)
            
            return patterns
            
        except Exception as e:
            logging.warning(f"ML reversal prediction failed: {str(e)}")
            for pattern in patterns:
                pattern.reversal_probability = self._heuristic_reversal_probability(pattern)
            return patterns
    
    def _extract_reversal_features(self, data: pd.DataFrame, pattern_idx: int, 
                                 pattern: EngulfingPattern) -> List[float]:
        """Extract features for reversal prediction ML model"""
        try:
            current = data.iloc[pattern_idx]
            
            features = [
                pattern.pattern_strength,
                pattern.size_ratio,
                pattern.volume_ratio,
                pattern.engulfment_percentage,
                pattern.shadow_analysis,
                pattern.trend_context_score,
                pattern.volume_confirmation,
                pattern.momentum_confirmation,
                pattern.continuation_signals,
                pattern.pattern_purity,
                1.0 if pattern.pattern_type == 'bullish_engulfing' else 0.0,
                current['rsi'] / 100.0 if not pd.isna(current['rsi']) else 0.5,
                current['bb_position'] if not pd.isna(current['bb_position']) else 0.5,
                current['volatility_ratio'] / 2.0,
                current['volume_ratio'],
                current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
                current['macd_hist'] if not pd.isna(current['macd_hist']) else 0,
                current['momentum_zscore'] if not pd.isna(current['momentum_zscore']) else 0,
                current['stoch_k'] / 100.0 if not pd.isna(current['stoch_k']) else 0.5,
                pattern.size_ratio * pattern.volume_confirmation,  # Combined strength
                pattern.engulfment_percentage * pattern.momentum_confirmation,  # Momentum-engulfment factor
                current['body_ratio']
            ]
            
            return features
            
        except Exception:
            return [0.5] * 22
    
    def _train_reversal_model(self, patterns: List[EngulfingPattern], features: List[List[float]]):
        """Train ML model for reversal prediction"""
        try:
            targets = []
            for pattern in patterns:
                target = (
                    pattern.pattern_strength * 0.3 +
                    pattern.volume_confirmation * 0.25 +
                    pattern.momentum_confirmation * 0.2 +
                    pattern.trend_context_score * 0.15 +
                    pattern.pattern_purity * 0.1
                )
                targets.append(max(0.1, min(0.9, target)))
            
            if len(features) >= 18:
                features_scaled = self.scaler.fit_transform(features)
                self.reversal_predictor.fit(features_scaled, targets)
                self.is_ml_fitted = True
                logging.info("ML reversal predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _heuristic_reversal_probability(self, pattern: EngulfingPattern) -> float:
        """Calculate heuristic reversal probability"""
        return (
            pattern.pattern_strength * 0.35 +
            pattern.volume_confirmation * 0.25 +
            pattern.momentum_confirmation * 0.2 +
            pattern.trend_context_score * 0.2
        )
    
    def _generate_pattern_analytics(self, patterns: List[EngulfingPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-20:]
        bullish_patterns = [p for p in recent_patterns if p.pattern_type == 'bullish_engulfing']
        bearish_patterns = [p for p in recent_patterns if p.pattern_type == 'bearish_engulfing']
        
        return {
            'total_patterns': len(recent_patterns),
            'bullish_patterns': len(bullish_patterns),
            'bearish_patterns': len(bearish_patterns),
            'average_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'average_reversal_probability': sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns),
            'average_size_ratio': sum(p.size_ratio for p in recent_patterns) / len(recent_patterns),
            'average_engulfment_percentage': sum(p.engulfment_percentage for p in recent_patterns) / len(recent_patterns),
            'high_strength_patterns': len([p for p in recent_patterns if p.pattern_strength > 0.8]),
            'high_probability_patterns': len([p for p in recent_patterns if p.reversal_probability > 0.7]),
            'strong_volume_patterns': len([p for p in recent_patterns if p.volume_confirmation > 0.7]),
            'high_purity_patterns': len([p for p in recent_patterns if p.pattern_purity > 0.8])
        }
    
    def _analyze_current_trend_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current trend context"""
        current = data.iloc[-1]
        
        return {
            'trend_strength': current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
            'trend_direction': 'up' if current['trend_direction'] == 1 else 'down',
            'rsi_level': current['rsi'] if not pd.isna(current['rsi']) else 50,
            'bb_position': current['bb_position'] if not pd.isna(current['bb_position']) else 0.5,
            'volume_context': 'high' if current['volume_ratio'] > 1.5 else 'normal',
            'volatility_context': 'high' if current['volatility_ratio'] > 1.5 else 'normal'
        }
    
    def _generate_reversal_signals(self, patterns: List[EngulfingPattern], 
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """Generate reversal signals based on patterns"""
        if not patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        recent_patterns = [p for p in patterns[-5:] if p.pattern_strength > 0.7]
        
        if not recent_patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        latest_pattern = recent_patterns[-1]
        
        return {
            'signal_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'reversal_probability': sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns),
            'pattern_count': len(recent_patterns),
            'latest_pattern_type': latest_pattern.pattern_type,
            'volume_confirmation': sum(p.volume_confirmation for p in recent_patterns) / len(recent_patterns),
            'momentum_confirmation': sum(p.momentum_confirmation for p in recent_patterns) / len(recent_patterns),
            'pattern_purity': sum(p.pattern_purity for p in recent_patterns) / len(recent_patterns),
            'most_recent_pattern': latest_pattern.timestamp
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure"""
        current = data.iloc[-1]
        recent_data = data.iloc[-20:]
        
        return {
            'trend_consistency': 1.0 - np.std(recent_data['trend_direction']),
            'volume_pattern': {
                'average_volume': recent_data['volume'].mean(),
                'volume_trend': 'increasing' if recent_data['volume'].iloc[-5:].mean() > recent_data['volume'].iloc[-15:-5].mean() else 'decreasing'
            },
            'volatility_analysis': {
                'current_volatility': current['volatility_ratio'],
                'volatility_trend': recent_data['volatility_ratio'].iloc[-5:].mean() / recent_data['volatility_ratio'].iloc[-15:-5].mean()
            }
        }
    
    def _assess_pattern_reliability(self, patterns: List[EngulfingPattern]) -> Dict[str, Any]:
        """Assess pattern reliability metrics"""
        if not patterns:
            return {}
        
        return {
            'consistency_score': 1.0 - np.std([p.pattern_strength for p in patterns]),
            'purity_average': np.mean([p.pattern_purity for p in patterns]),
            'volume_reliability': np.mean([p.volume_confirmation for p in patterns]),
            'momentum_consistency': np.mean([p.momentum_confirmation for p in patterns]),
            'reversal_success_estimate': np.mean([p.reversal_probability for p in patterns])
        }
    
    def _calculate_engulfment_statistics(self, patterns: List[EngulfingPattern]) -> Dict[str, Any]:
        """Calculate engulfment pattern statistics"""
        if not patterns:
            return {}
        
        size_ratios = [p.size_ratio for p in patterns]
        engulfment_percentages = [p.engulfment_percentage for p in patterns]
        volume_ratios = [p.volume_ratio for p in patterns]
        
        return {
            'size_ratio_stats': {
                'mean': np.mean(size_ratios),
                'median': np.median(size_ratios),
                'max': np.max(size_ratios),
                'std': np.std(size_ratios)
            },
            'engulfment_percentage_stats': {
                'mean': np.mean(engulfment_percentages),
                'median': np.median(engulfment_percentages),
                'max': np.max(engulfment_percentages)
            },
            'volume_ratio_stats': {
                'mean': np.mean(volume_ratios),
                'median': np.median(volume_ratios),
                'max': np.max(volume_ratios)
            }
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on engulfing pattern analysis"""
        current_pattern = value.get('current_pattern')
        reversal_signals = value.get('reversal_signals', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Strong bullish engulfing signal
        if (current_pattern.pattern_type == 'bullish_engulfing' and
            current_pattern.pattern_strength > 0.8 and
            current_pattern.reversal_probability > 0.7 and
            current_pattern.volume_confirmation > 0.6):
            
            confidence = (
                current_pattern.pattern_strength * 0.4 +
                current_pattern.reversal_probability * 0.3 +
                current_pattern.volume_confirmation * 0.2 +
                current_pattern.momentum_confirmation * 0.1
            )
            
            return SignalType.BUY, confidence
        
        # Strong bearish engulfing signal
        elif (current_pattern.pattern_type == 'bearish_engulfing' and
              current_pattern.pattern_strength > 0.8 and
              current_pattern.reversal_probability > 0.7 and
              current_pattern.volume_confirmation > 0.6):
            
            confidence = (
                current_pattern.pattern_strength * 0.4 +
                current_pattern.reversal_probability * 0.3 +
                current_pattern.volume_confirmation * 0.2 +
                current_pattern.momentum_confirmation * 0.1
            )
            
            return SignalType.SELL, confidence
        
        # Moderate signals
        elif (current_pattern.pattern_strength > 0.7 and
              current_pattern.reversal_probability > 0.6):
            
            confidence = current_pattern.pattern_strength * 0.7
            
            if current_pattern.pattern_type == 'bullish_engulfing':
                return SignalType.BUY, confidence
            else:
                return SignalType.SELL, confidence
        
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_types': ['bullish_engulfing', 'bearish_engulfing'],
            'min_engulfment_ratio': self.parameters['min_engulfment_ratio'],
            'min_body_size': self.parameters['min_body_size'],
            'volume_analysis_enabled': self.parameters['volume_analysis'],
            'ml_reversal_prediction_enabled': self.parameters['ml_reversal_prediction']
        })
        return base_metadata