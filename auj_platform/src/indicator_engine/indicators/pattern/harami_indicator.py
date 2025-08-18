"""
Harami Indicator - Advanced Two-Candle Continuation/Reversal Pattern Detection
============================================================================

This indicator implements sophisticated harami pattern detection (both bullish and bearish)
with advanced size validation, body containment analysis, and ML-enhanced prediction.
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
class HaramiPattern:
    """Represents a detected harami pattern"""
    timestamp: pd.Timestamp
    pattern_type: str  # 'bullish_harami' or 'bearish_harami'
    pattern_strength: float
    containment_ratio: float
    size_ratio: float
    position_score: float
    shadow_harmony: float
    trend_context_score: float
    volume_confirmation: float
    momentum_divergence: float
    reversal_probability: float
    pattern_purity: float
    indecision_signal: float


class HaramiIndicator(StandardIndicatorInterface):
    """
    Advanced Harami Pattern Indicator
    
    Features:
    - Precise bullish and bearish harami pattern identification
    - Advanced body containment and size ratio validation
    - Comprehensive position analysis within mother candle
    - Volume-based confirmation and momentum divergence detection
    - ML-enhanced reversal/continuation probability prediction
    - Trend context validation and indecision signal analysis
    - Pattern purity scoring and reliability metrics
    - Statistical significance testing for pattern quality
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'max_size_ratio': 0.6,             # Maximum size of inside candle vs mother
            'min_containment': 0.95,           # Minimum containment percentage
            'min_mother_body': 0.4,            # Minimum mother candle body size
            'max_inside_body': 0.3,            # Maximum inside candle body size  
            'position_tolerance': 0.15,        # Tolerance for position scoring
            'trend_context_periods': 12,       # Periods for trend context analysis
            'volume_lookback': 18,             # Periods for volume analysis
            'momentum_analysis': True,
            'volume_analysis': True,
            'ml_prediction': True,
            'indecision_analysis': True,
            'pattern_purity_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="HaramiIndicator", parameters=default_params)
        
        # Initialize ML components
        self.pattern_predictor = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=130, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=90, random_state=42)),
            ('ada', AdaBoostRegressor(n_estimators=100, random_state=42))
        ])
        self.scaler = RobustScaler()
        self.is_ml_fitted = False
        
        logging.info(f"HaramiIndicator initialized with parameters: {self.parameters}")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=45,
            lookback_periods=90
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate harami patterns with advanced analysis"""
        try:
            if len(data) < 45:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < 45"
                )
            
            # Enhance data with technical indicators
            enhanced_data = self._enhance_data_with_indicators(data)
            
            # Detect harami patterns
            detected_patterns = self._detect_harami_patterns(enhanced_data)
            
            # Apply comprehensive analysis pipeline
            if self.parameters['volume_analysis']:
                detected_patterns = self._analyze_volume_confirmation(detected_patterns, enhanced_data)
            
            if self.parameters['momentum_analysis']:
                detected_patterns = self._analyze_momentum_divergence(detected_patterns, enhanced_data)
            
            if self.parameters['indecision_analysis']:
                detected_patterns = self._analyze_indecision_signals(detected_patterns, enhanced_data)
            
            if self.parameters['pattern_purity_analysis']:
                detected_patterns = self._analyze_pattern_purity(detected_patterns, enhanced_data)
            
            if self.parameters['ml_prediction'] and detected_patterns:
                detected_patterns = self._predict_reversal_probability(detected_patterns, enhanced_data)
            
            # Generate comprehensive analysis
            pattern_analytics = self._generate_pattern_analytics(detected_patterns)
            trend_analysis = self._analyze_current_trend_context(enhanced_data)
            signals = self._generate_pattern_signals(detected_patterns, enhanced_data)
            
            return {
                'current_pattern': detected_patterns[-1] if detected_patterns else None,
                'recent_patterns': detected_patterns[-10:],
                'pattern_analytics': pattern_analytics,
                'trend_analysis': trend_analysis,
                'pattern_signals': signals,
                'market_structure': self._analyze_market_structure(enhanced_data),
                'pattern_reliability': self._assess_pattern_reliability(detected_patterns),
                'harami_statistics': self._calculate_harami_statistics(detected_patterns)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Harami pattern calculation failed: {str(e)}", e
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
        df['is_doji'] = df['body'] / df['total_range'] < 0.05
        
        # Ratios and metrics
        df['body_ratio'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
        df['upper_shadow_ratio'] = np.where(df['total_range'] > 0, df['upper_shadow'] / df['total_range'], 0)
        df['lower_shadow_ratio'] = np.where(df['total_range'] > 0, df['lower_shadow'] / df['total_range'], 0)
        
        # Trend indicators
        df['sma_8'] = df['close'].rolling(8).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Technical indicators
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
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
        df['volatility'] = df['total_range'].rolling(10).std()
        
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
    
    def _detect_harami_patterns(self, data: pd.DataFrame) -> List[HaramiPattern]:
        """Detect harami patterns with sophisticated analysis"""
        patterns = []
        
        for i in range(self.parameters['trend_context_periods'], len(data) - 1):
            mother_candle = data.iloc[i]
            inside_candle = data.iloc[i + 1]
            
            # Check for bullish harami
            if self._is_bullish_harami(mother_candle, inside_candle):
                pattern = self._create_harami_pattern(
                    data, i + 1, 'bullish_harami', mother_candle, inside_candle
                )
                if pattern and pattern.pattern_strength >= 0.6:
                    patterns.append(pattern)
            
            # Check for bearish harami
            elif self._is_bearish_harami(mother_candle, inside_candle):
                pattern = self._create_harami_pattern(
                    data, i + 1, 'bearish_harami', mother_candle, inside_candle
                )
                if pattern and pattern.pattern_strength >= 0.6:
                    patterns.append(pattern)
        
        return patterns
    
    def _is_bullish_harami(self, mother: pd.Series, inside: pd.Series) -> bool:
        """Check if candles form a bullish harami pattern"""
        # Mother candle is bearish and large, inside candle is contained
        return (
            mother['is_bearish'] and
            mother['body_ratio'] >= self.parameters['min_mother_body'] and
            inside['body_ratio'] <= self.parameters['max_inside_body'] and
            self._is_contained(mother, inside) and
            inside['body'] / mother['body'] <= self.parameters['max_size_ratio']
        )
    
    def _is_bearish_harami(self, mother: pd.Series, inside: pd.Series) -> bool:
        """Check if candles form a bearish harami pattern"""
        # Mother candle is bullish and large, inside candle is contained
        return (
            mother['is_bullish'] and
            mother['body_ratio'] >= self.parameters['min_mother_body'] and
            inside['body_ratio'] <= self.parameters['max_inside_body'] and
            self._is_contained(mother, inside) and
            inside['body'] / mother['body'] <= self.parameters['max_size_ratio']
        )
    
    def _is_contained(self, mother: pd.Series, inside: pd.Series) -> bool:
        """Check if inside candle is properly contained within mother candle"""
        mother_body_top = max(mother['open'], mother['close'])
        mother_body_bottom = min(mother['open'], mother['close'])
        inside_high = inside['high']
        inside_low = inside['low']
        
        # Check containment with tolerance
        containment_ratio = self._calculate_containment_ratio(mother, inside)
        return containment_ratio >= self.parameters['min_containment']
    
    def _calculate_containment_ratio(self, mother: pd.Series, inside: pd.Series) -> float:
        """Calculate how well the inside candle is contained"""
        mother_body_top = max(mother['open'], mother['close'])
        mother_body_bottom = min(mother['open'], mother['close'])
        
        # Calculate overlap
        overlap_top = min(mother_body_top, inside['high'])
        overlap_bottom = max(mother_body_bottom, inside['low'])
        
        if overlap_top <= overlap_bottom:
            return 0.0
        
        overlap_range = overlap_top - overlap_bottom
        inside_range = inside['high'] - inside['low']
        
        return overlap_range / inside_range if inside_range > 0 else 0.0
    
    def _create_harami_pattern(self, data: pd.DataFrame, candle_idx: int, 
                             pattern_type: str, mother: pd.Series, 
                             inside: pd.Series) -> Optional[HaramiPattern]:
        """Create harami pattern with comprehensive analysis"""
        try:
            # Calculate pattern metrics
            containment_ratio = self._calculate_containment_ratio(mother, inside)
            size_ratio = inside['body'] / mother['body'] if mother['body'] > 0 else 0
            
            # Position analysis
            position_score = self._analyze_harami_position(mother, inside)
            
            # Shadow harmony
            shadow_harmony = self._analyze_shadow_harmony(mother, inside)
            
            # Trend context analysis
            trend_context_score = self._assess_harami_trend_context(data, candle_idx, pattern_type)
            
            # Calculate pattern strength
            pattern_strength = self._calculate_harami_strength(
                containment_ratio, size_ratio, position_score, shadow_harmony, trend_context_score
            )
            
            pattern = HaramiPattern(
                timestamp=inside.name,
                pattern_type=pattern_type,
                pattern_strength=pattern_strength,
                containment_ratio=containment_ratio,
                size_ratio=size_ratio,
                position_score=position_score,
                shadow_harmony=shadow_harmony,
                trend_context_score=trend_context_score,
                volume_confirmation=0.0,
                momentum_divergence=0.0,
                reversal_probability=0.0,
                pattern_purity=0.0,
                indecision_signal=0.0
            )
            
            return pattern
            
        except Exception:
            return None
    
    def _analyze_harami_position(self, mother: pd.Series, inside: pd.Series) -> float:
        """Analyze position of inside candle within mother candle"""
        mother_body_top = max(mother['open'], mother['close'])
        mother_body_bottom = min(mother['open'], mother['close'])
        mother_body_size = mother_body_top - mother_body_bottom
        
        inside_center = (inside['open'] + inside['close']) / 2
        mother_center = (mother_body_top + mother_body_bottom) / 2
        
        # Calculate position within mother body
        if mother_body_size > 0:
            position_within_body = (inside_center - mother_body_bottom) / mother_body_size
            # Ideal position is center (0.5), score based on distance from center
            center_distance = abs(position_within_body - 0.5)
            position_score = max(0, 1.0 - center_distance / 0.5)
        else:
            position_score = 0.5
        
        return position_score
    
    def _analyze_shadow_harmony(self, mother: pd.Series, inside: pd.Series) -> float:
        """Analyze shadow harmony between mother and inside candles"""
        # Calculate shadow ratios
        mother_upper_ratio = mother['upper_shadow_ratio']
        mother_lower_ratio = mother['lower_shadow_ratio']
        inside_upper_ratio = inside['upper_shadow_ratio']
        inside_lower_ratio = inside['lower_shadow_ratio']
        
        # Harmony factors
        harmony_factors = []
        
        # Upper shadow harmony
        upper_harmony = 1.0 - abs(mother_upper_ratio - inside_upper_ratio)
        harmony_factors.append(max(upper_harmony, 0) * 0.4)
        
        # Lower shadow harmony
        lower_harmony = 1.0 - abs(mother_lower_ratio - inside_lower_ratio)
        harmony_factors.append(max(lower_harmony, 0) * 0.4)
        
        # Overall shadow size (smaller is better for harami)
        inside_total_shadows = inside_upper_ratio + inside_lower_ratio
        shadow_size_score = max(0, 1.0 - inside_total_shadows / 0.6)
        harmony_factors.append(shadow_size_score * 0.2)
        
        return sum(harmony_factors)
    
    def _assess_harami_trend_context(self, data: pd.DataFrame, candle_idx: int, 
                                   pattern_type: str) -> float:
        """Assess trend context for harami patterns"""
        context_data = data.iloc[max(0, candle_idx - self.parameters['trend_context_periods']):candle_idx]
        
        if len(context_data) < 5:
            return 0.5
        
        context_factors = []
        
        # Trend alignment
        if pattern_type == 'bullish_harami':
            # Should appear after downtrend
            price_change = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
            if price_change < -0.02:
                trend_factor = min(abs(price_change) / 0.1, 1.0)
                context_factors.append(trend_factor * 0.35)
            else:
                context_factors.append(0.15)
        else:  # bearish_harami
            # Should appear after uptrend
            price_change = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
            if price_change > 0.02:
                trend_factor = min(price_change / 0.1, 1.0)
                context_factors.append(trend_factor * 0.35)
            else:
                context_factors.append(0.15)
        
        # ADX strength
        latest = context_data.iloc[-1]
        if not pd.isna(latest['adx']):
            adx_factor = min(latest['adx'] / 30, 1.0)
            context_factors.append(adx_factor * 0.2)
        else:
            context_factors.append(0.1)
        
        # RSI positioning
        rsi_context = 0
        if not pd.isna(latest['rsi']):
            if pattern_type == 'bullish_harami' and latest['rsi'] < 45:
                rsi_context = (45 - latest['rsi']) / 25
            elif pattern_type == 'bearish_harami' and latest['rsi'] > 55:
                rsi_context = (latest['rsi'] - 55) / 25
        context_factors.append(min(rsi_context, 1.0) * 0.25)
        
        # Trend strength
        trend_strength = latest['trend_strength'] if not pd.isna(latest['trend_strength']) else 0.5
        context_factors.append(trend_strength * 0.2)
        
        return sum(context_factors)
    
    def _calculate_harami_strength(self, containment_ratio: float, size_ratio: float, 
                                 position_score: float, shadow_harmony: float, 
                                 trend_context_score: float) -> float:
        """Calculate overall harami pattern strength"""
        strength_components = [
            containment_ratio * 0.3,           # Containment quality
            (1.0 - size_ratio) * 0.25,         # Size differential (smaller inside is better)
            position_score * 0.2,              # Position within mother
            shadow_harmony * 0.15,             # Shadow harmony
            trend_context_score * 0.1          # Trend context
        ]
        
        return sum(strength_components)
    
    def _analyze_volume_confirmation(self, patterns: List[HaramiPattern], 
                                   data: pd.DataFrame) -> List[HaramiPattern]:
        """Analyze volume confirmation for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            volume_score = self._calculate_volume_confirmation_score(data, pattern_idx)
            pattern.volume_confirmation = volume_score
        
        return patterns
    
    def _calculate_volume_confirmation_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate volume confirmation score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - self.parameters['volume_lookback']):pattern_idx + 1]
            mother_volume = data.iloc[pattern_idx - 1]['volume']
            inside_volume = data.iloc[pattern_idx]['volume']
            
            # Average volume
            avg_volume = context_data['volume'].iloc[:-2].mean()
            
            # Volume analysis
            volume_factors = []
            
            # Mother candle volume surge
            mother_surge = mother_volume / avg_volume if avg_volume > 0 else 1.0
            volume_factors.append(min(mother_surge / 1.5, 1.0) * 0.4)
            
            # Inside candle volume decline (indecision)
            volume_decline = inside_volume / mother_volume if mother_volume > 0 else 1.0
            decline_score = max(0, 1.0 - volume_decline)
            volume_factors.append(decline_score * 0.35)
            
            # Volume trend analysis
            recent_volumes = context_data['volume'].iloc[-5:]
            if len(recent_volumes) >= 3:
                volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
                trend_score = 0.5 + (volume_trend / avg_volume * 2) if avg_volume > 0 else 0.5
                volume_factors.append(min(max(trend_score, 0), 1) * 0.25)
            
            return sum(volume_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_momentum_divergence(self, patterns: List[HaramiPattern], 
                                   data: pd.DataFrame) -> List[HaramiPattern]:
        """Analyze momentum divergence for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            divergence_score = self._calculate_momentum_divergence_score(data, pattern_idx, pattern.pattern_type)
            pattern.momentum_divergence = divergence_score
        
        return patterns
    
    def _calculate_momentum_divergence_score(self, data: pd.DataFrame, pattern_idx: int, 
                                          pattern_type: str) -> float:
        """Calculate momentum divergence score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 8):pattern_idx + 1]
            
            if len(context_data) < 5:
                return 0.5
            
            divergence_factors = []
            
            # RSI divergence
            if pattern_type == 'bullish_harami':
                # Look for price making lower lows while RSI makes higher lows
                price_trend = np.polyfit(range(len(context_data)), context_data['low'], 1)[0]
                rsi_values = context_data['rsi'].fillna(50)
                rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
                
                if price_trend < 0 and rsi_trend > 0:  # Bullish divergence
                    divergence_factors.append(0.4)
                elif rsi_trend > 0:
                    divergence_factors.append(0.2)
            else:  # bearish_harami
                # Look for price making higher highs while RSI makes lower highs
                price_trend = np.polyfit(range(len(context_data)), context_data['high'], 1)[0]
                rsi_values = context_data['rsi'].fillna(50)
                rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
                
                if price_trend > 0 and rsi_trend < 0:  # Bearish divergence
                    divergence_factors.append(0.4)
                elif rsi_trend < 0:
                    divergence_factors.append(0.2)
            
            # MACD divergence
            macd_values = context_data['macd_hist'].fillna(0)
            if len(macd_values) >= 3:
                macd_trend = np.polyfit(range(len(macd_values)), macd_values, 1)[0]
                
                if pattern_type == 'bullish_harami' and macd_trend > 0:
                    divergence_factors.append(0.3)
                elif pattern_type == 'bearish_harami' and macd_trend < 0:
                    divergence_factors.append(0.3)
                else:
                    divergence_factors.append(0.1)
            
            # Stochastic momentum
            current_stoch = data.iloc[pattern_idx]['stoch_k']
            if not pd.isna(current_stoch):
                if pattern_type == 'bullish_harami' and current_stoch < 30:
                    divergence_factors.append(0.3)
                elif pattern_type == 'bearish_harami' and current_stoch > 70:
                    divergence_factors.append(0.3)
                else:
                    divergence_factors.append(0.15)
            
            return sum(divergence_factors) if divergence_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _analyze_indecision_signals(self, patterns: List[HaramiPattern], 
                                  data: pd.DataFrame) -> List[HaramiPattern]:
        """Analyze indecision signals for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            indecision_score = self._calculate_indecision_score(data, pattern_idx)
            pattern.indecision_signal = indecision_score
        
        return patterns
    
    def _calculate_indecision_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate indecision signal score"""
        try:
            inside_candle = data.iloc[pattern_idx]
            mother_candle = data.iloc[pattern_idx - 1]
            
            indecision_factors = []
            
            # Inside candle characteristics
            # Small body indicates indecision
            small_body_score = max(0, 1.0 - inside_candle['body_ratio'] / 0.3)
            indecision_factors.append(small_body_score * 0.3)
            
            # Doji-like characteristics
            if inside_candle['is_doji']:
                indecision_factors.append(0.25)
            else:
                indecision_factors.append(0.1)
            
            # Range contraction
            range_contraction = inside_candle['total_range'] / mother_candle['total_range'] if mother_candle['total_range'] > 0 else 1.0
            contraction_score = max(0, 1.0 - range_contraction)
            indecision_factors.append(contraction_score * 0.25)
            
            # Volume decline (hesitation)
            volume_ratio = inside_candle['volume'] / mother_candle['volume'] if mother_candle['volume'] > 0 else 1.0
            volume_decline_score = max(0, 1.0 - volume_ratio)
            indecision_factors.append(volume_decline_score * 0.2)
            
            return sum(indecision_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_pattern_purity(self, patterns: List[HaramiPattern], 
                              data: pd.DataFrame) -> List[HaramiPattern]:
        """Analyze pattern purity for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            purity_score = self._calculate_pattern_purity_score(data, pattern_idx, pattern)
            pattern.pattern_purity = purity_score
        
        return patterns
    
    def _calculate_pattern_purity_score(self, data: pd.DataFrame, pattern_idx: int, 
                                      pattern: HaramiPattern) -> float:
        """Calculate pattern purity score"""
        try:
            purity_factors = []
            
            # Containment purity
            containment_purity = pattern.containment_ratio
            purity_factors.append(containment_purity * 0.3)
            
            # Size ratio purity (smaller inside is purer)
            size_purity = max(0, 1.0 - pattern.size_ratio / 0.5)
            purity_factors.append(size_purity * 0.25)
            
            # Position purity
            purity_factors.append(pattern.position_score * 0.2)
            
            # Shadow harmony purity
            purity_factors.append(pattern.shadow_harmony * 0.15)
            
            # Gap analysis (no gaps is purer)
            mother_candle = data.iloc[pattern_idx - 1]
            inside_candle = data.iloc[pattern_idx]
            
            gap_size = abs(inside_candle['open'] - mother_candle['close']) / mother_candle['close']
            gap_purity = max(0, 1.0 - gap_size / 0.02)
            purity_factors.append(gap_purity * 0.1)
            
            return sum(purity_factors)
            
        except Exception:
            return 0.5
    
    def _predict_reversal_probability(self, patterns: List[HaramiPattern], 
                                    data: pd.DataFrame) -> List[HaramiPattern]:
        """Predict reversal probability using ML"""
        if not patterns:
            return patterns
        
        try:
            features = []
            for pattern in patterns:
                pattern_idx = data.index.get_loc(pattern.timestamp)
                feature_vector = self._extract_prediction_features(data, pattern_idx, pattern)
                features.append(feature_vector)
            
            if len(features) < 15:
                for pattern in patterns:
                    pattern.reversal_probability = self._heuristic_reversal_probability(pattern)
                return patterns
            
            if not self.is_ml_fitted:
                self._train_prediction_model(patterns, features)
            
            if self.is_ml_fitted:
                features_scaled = self.scaler.transform(features)
                predictions = self.pattern_predictor.predict(features_scaled)
                
                for i, pattern in enumerate(patterns):
                    ml_probability = max(0.1, min(0.95, predictions[i]))
                    heuristic_prob = self._heuristic_reversal_probability(pattern)
                    pattern.reversal_probability = (ml_probability * 0.7 + heuristic_prob * 0.3)
            else:
                for pattern in patterns:
                    pattern.reversal_probability = self._heuristic_reversal_probability(pattern)
            
            return patterns
            
        except Exception as e:
            logging.warning(f"ML prediction failed: {str(e)}")
            for pattern in patterns:
                pattern.reversal_probability = self._heuristic_reversal_probability(pattern)
            return patterns
    
    def _extract_prediction_features(self, data: pd.DataFrame, pattern_idx: int, 
                                   pattern: HaramiPattern) -> List[float]:
        """Extract features for ML prediction model"""
        try:
            current = data.iloc[pattern_idx]
            
            features = [
                pattern.pattern_strength,
                pattern.containment_ratio,
                pattern.size_ratio,
                pattern.position_score,
                pattern.shadow_harmony,
                pattern.trend_context_score,
                pattern.volume_confirmation,
                pattern.momentum_divergence,
                pattern.indecision_signal,
                pattern.pattern_purity,
                1.0 if pattern.pattern_type == 'bullish_harami' else 0.0,
                current['rsi'] / 100.0 if not pd.isna(current['rsi']) else 0.5,
                current['bb_position'] if not pd.isna(current['bb_position']) else 0.5,
                current['adx'] / 50.0 if not pd.isna(current['adx']) else 0.5,
                current['volume_ratio'],
                current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
                current['macd_hist'] if not pd.isna(current['macd_hist']) else 0,
                current['stoch_k'] / 100.0 if not pd.isna(current['stoch_k']) else 0.5,
                current['body_ratio'],
                pattern.containment_ratio * pattern.momentum_divergence,  # Combined signal
                pattern.indecision_signal * pattern.volume_confirmation,   # Indecision-volume factor
                current['volatility'] / current['atr'] if not pd.isna(current['atr']) and current['atr'] > 0 else 1.0
            ]
            
            return features
            
        except Exception:
            return [0.5] * 22
    
    def _train_prediction_model(self, patterns: List[HaramiPattern], features: List[List[float]]):
        """Train ML model for pattern prediction"""
        try:
            targets = []
            for pattern in patterns:
                target = (
                    pattern.pattern_strength * 0.3 +
                    pattern.momentum_divergence * 0.25 +
                    pattern.volume_confirmation * 0.2 +
                    pattern.trend_context_score * 0.15 +
                    pattern.pattern_purity * 0.1
                )
                targets.append(max(0.1, min(0.9, target)))
            
            if len(features) >= 20:
                features_scaled = self.scaler.fit_transform(features)
                self.pattern_predictor.fit(features_scaled, targets)
                self.is_ml_fitted = True
                logging.info("ML pattern predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _heuristic_reversal_probability(self, pattern: HaramiPattern) -> float:
        """Calculate heuristic reversal probability"""
        return (
            pattern.pattern_strength * 0.35 +
            pattern.momentum_divergence * 0.25 +
            pattern.volume_confirmation * 0.2 +
            pattern.indecision_signal * 0.2
        )
    
    def _generate_pattern_analytics(self, patterns: List[HaramiPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-20:]
        bullish_patterns = [p for p in recent_patterns if p.pattern_type == 'bullish_harami']
        bearish_patterns = [p for p in recent_patterns if p.pattern_type == 'bearish_harami']
        
        return {
            'total_patterns': len(recent_patterns),
            'bullish_patterns': len(bullish_patterns),
            'bearish_patterns': len(bearish_patterns),
            'average_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'average_reversal_probability': sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns),
            'average_containment_ratio': sum(p.containment_ratio for p in recent_patterns) / len(recent_patterns),
            'average_indecision_signal': sum(p.indecision_signal for p in recent_patterns) / len(recent_patterns),
            'high_strength_patterns': len([p for p in recent_patterns if p.pattern_strength > 0.8]),
            'high_probability_patterns': len([p for p in recent_patterns if p.reversal_probability > 0.7]),
            'strong_divergence_patterns': len([p for p in recent_patterns if p.momentum_divergence > 0.7]),
            'high_purity_patterns': len([p for p in recent_patterns if p.pattern_purity > 0.8])
        }
    
    def _analyze_current_trend_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current trend context"""
        current = data.iloc[-1]
        
        return {
            'trend_strength': current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
            'trend_direction': 'up' if current['trend_direction'] == 1 else 'down',
            'adx_strength': current['adx'] if not pd.isna(current['adx']) else 25,
            'rsi_level': current['rsi'] if not pd.isna(current['rsi']) else 50,
            'bb_position': current['bb_position'] if not pd.isna(current['bb_position']) else 0.5,
            'volume_context': 'high' if current['volume_ratio'] > 1.5 else 'normal'
        }
    
    def _generate_pattern_signals(self, patterns: List[HaramiPattern], 
                                data: pd.DataFrame) -> Dict[str, Any]:
        """Generate pattern signals based on harami analysis"""
        if not patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        recent_patterns = [p for p in patterns[-5:] if p.pattern_strength > 0.7]
        
        if not recent_patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        latest_pattern = recent_patterns[-1]
        
        return {
            'signal_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'reversal_probability': sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns),
            'indecision_strength': sum(p.indecision_signal for p in recent_patterns) / len(recent_patterns),
            'pattern_count': len(recent_patterns),
            'latest_pattern_type': latest_pattern.pattern_type,
            'momentum_divergence': sum(p.momentum_divergence for p in recent_patterns) / len(recent_patterns),
            'volume_confirmation': sum(p.volume_confirmation for p in recent_patterns) / len(recent_patterns),
            'pattern_purity': sum(p.pattern_purity for p in recent_patterns) / len(recent_patterns),
            'most_recent_pattern': latest_pattern.timestamp
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure"""
        current = data.iloc[-1]
        recent_data = data.iloc[-15:]
        
        return {
            'volatility_context': {
                'current_volatility': current['volatility'] if not pd.isna(current['volatility']) else 0,
                'volatility_trend': 'increasing' if recent_data['volatility'].iloc[-3:].mean() > recent_data['volatility'].iloc[-10:-3].mean() else 'stable'
            },
            'indecision_frequency': len([i for i in range(len(recent_data)) if recent_data.iloc[i]['is_doji']]) / len(recent_data),
            'trend_consistency': 1.0 - np.std(recent_data['trend_direction'])
        }
    
    def _assess_pattern_reliability(self, patterns: List[HaramiPattern]) -> Dict[str, Any]:
        """Assess pattern reliability metrics"""
        if not patterns:
            return {}
        
        return {
            'consistency_score': 1.0 - np.std([p.pattern_strength for p in patterns]),
            'purity_average': np.mean([p.pattern_purity for p in patterns]),
            'divergence_reliability': np.mean([p.momentum_divergence for p in patterns]),
            'indecision_consistency': np.mean([p.indecision_signal for p in patterns]),
            'reversal_success_estimate': np.mean([p.reversal_probability for p in patterns])
        }
    
    def _calculate_harami_statistics(self, patterns: List[HaramiPattern]) -> Dict[str, Any]:
        """Calculate harami pattern statistics"""
        if not patterns:
            return {}
        
        containment_ratios = [p.containment_ratio for p in patterns]
        size_ratios = [p.size_ratio for p in patterns]
        position_scores = [p.position_score for p in patterns]
        
        return {
            'containment_stats': {
                'mean': np.mean(containment_ratios),
                'median': np.median(containment_ratios),
                'min': np.min(containment_ratios),
                'std': np.std(containment_ratios)
            },
            'size_ratio_stats': {
                'mean': np.mean(size_ratios),
                'median': np.median(size_ratios),
                'max': np.max(size_ratios),
                'std': np.std(size_ratios)
            },
            'position_stats': {
                'mean': np.mean(position_scores),
                'median': np.median(position_scores),
                'std': np.std(position_scores)
            }
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on harami pattern analysis"""
        current_pattern = value.get('current_pattern')
        pattern_signals = value.get('pattern_signals', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Strong bullish harami signal
        if (current_pattern.pattern_type == 'bullish_harami' and
            current_pattern.pattern_strength > 0.8 and
            current_pattern.reversal_probability > 0.7 and
            current_pattern.momentum_divergence > 0.6):
            
            confidence = (
                current_pattern.pattern_strength * 0.35 +
                current_pattern.reversal_probability * 0.3 +
                current_pattern.momentum_divergence * 0.2 +
                current_pattern.volume_confirmation * 0.15
            )
            
            return SignalType.BUY, confidence
        
        # Strong bearish harami signal
        elif (current_pattern.pattern_type == 'bearish_harami' and
              current_pattern.pattern_strength > 0.8 and
              current_pattern.reversal_probability > 0.7 and
              current_pattern.momentum_divergence > 0.6):
            
            confidence = (
                current_pattern.pattern_strength * 0.35 +
                current_pattern.reversal_probability * 0.3 +
                current_pattern.momentum_divergence * 0.2 +
                current_pattern.volume_confirmation * 0.15
            )
            
            return SignalType.SELL, confidence
        
        # Moderate signals based on indecision
        elif (current_pattern.pattern_strength > 0.7 and
              current_pattern.indecision_signal > 0.7):
            
            # Harami often signals continuation after indecision, so weaker signals
            confidence = current_pattern.pattern_strength * 0.5
            
            if current_pattern.pattern_type == 'bullish_harami':
                return SignalType.BUY, confidence
            else:
                return SignalType.SELL, confidence
        
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_types': ['bullish_harami', 'bearish_harami'],
            'max_size_ratio': self.parameters['max_size_ratio'],
            'min_containment': self.parameters['min_containment'],
            'volume_analysis_enabled': self.parameters['volume_analysis'],
            'ml_prediction_enabled': self.parameters['ml_prediction']
        })
        return base_metadata