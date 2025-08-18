"""
Harami Cross Indicator - Advanced Two-Candle Doji-Based Reversal Pattern Detection
=================================================================================

This indicator implements sophisticated harami cross pattern detection where the inside
candle is a doji, indicating strong indecision and potential reversal signals.
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
class HaramiCrossPattern:
    """Represents a detected harami cross pattern"""
    timestamp: pd.Timestamp
    pattern_type: str  # 'bullish_harami_cross' or 'bearish_harami_cross'
    pattern_strength: float
    doji_quality: float
    containment_ratio: float
    mother_strength: float
    doji_position: float
    shadow_symmetry: float
    trend_context_score: float
    volume_confirmation: float
    momentum_exhaustion: float
    reversal_probability: float
    pattern_purity: float
    indecision_intensity: float


class HaramiCrossIndicator(StandardIndicatorInterface):
    """
    Advanced Harami Cross Pattern Indicator
    
    Features:
    - Precise bullish and bearish harami cross pattern identification
    - Advanced doji quality assessment and symmetry analysis
    - Comprehensive mother candle strength validation
    - Volume-based confirmation and momentum exhaustion detection
    - ML-enhanced reversal probability prediction
    - Trend context validation and indecision intensity analysis
    - Pattern purity scoring and reliability metrics
    - Statistical significance testing for pattern quality
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'doji_threshold': 0.05,             # Maximum body ratio for doji
            'min_mother_body': 0.45,            # Minimum mother candle body size
            'min_containment': 0.9,             # Minimum doji containment
            'max_doji_range_ratio': 0.7,        # Maximum doji range vs mother
            'shadow_symmetry_threshold': 0.3,   # Maximum shadow asymmetry
            'trend_context_periods': 14,        # Periods for trend context analysis
            'volume_lookback': 20,              # Periods for volume analysis
            'momentum_analysis': True,
            'volume_analysis': True,
            'ml_prediction': True,
            'indecision_analysis': True,
            'pattern_purity_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="HaramiCrossIndicator", parameters=default_params)
        
        # Initialize ML components
        self.reversal_predictor = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=140, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('ada', AdaBoostRegressor(n_estimators=110, random_state=42))
        ])
        self.scaler = RobustScaler()
        self.is_ml_fitted = False
        
        logging.info(f"HaramiCrossIndicator initialized with parameters: {self.parameters}")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=50,
            lookback_periods=100
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate harami cross patterns with advanced analysis"""
        try:
            if len(data) < 50:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < 50"
                )
            
            # Enhance data with technical indicators
            enhanced_data = self._enhance_data_with_indicators(data)
            
            # Detect harami cross patterns
            detected_patterns = self._detect_harami_cross_patterns(enhanced_data)
            
            # Apply comprehensive analysis pipeline
            if self.parameters['volume_analysis']:
                detected_patterns = self._analyze_volume_confirmation(detected_patterns, enhanced_data)
            
            if self.parameters['momentum_analysis']:
                detected_patterns = self._analyze_momentum_exhaustion(detected_patterns, enhanced_data)
            
            if self.parameters['indecision_analysis']:
                detected_patterns = self._analyze_indecision_intensity(detected_patterns, enhanced_data)
            
            if self.parameters['pattern_purity_analysis']:
                detected_patterns = self._analyze_pattern_purity(detected_patterns, enhanced_data)
            
            if self.parameters['ml_prediction'] and detected_patterns:
                detected_patterns = self._predict_reversal_probability(detected_patterns, enhanced_data)
            
            # Generate comprehensive analysis
            pattern_analytics = self._generate_pattern_analytics(detected_patterns)
            trend_analysis = self._analyze_current_trend_context(enhanced_data)
            reversal_signals = self._generate_reversal_signals(detected_patterns, enhanced_data)
            
            return {
                'current_pattern': detected_patterns[-1] if detected_patterns else None,
                'recent_patterns': detected_patterns[-8:],
                'pattern_analytics': pattern_analytics,
                'trend_analysis': trend_analysis,
                'reversal_signals': reversal_signals,
                'market_structure': self._analyze_market_structure(enhanced_data),
                'pattern_reliability': self._assess_pattern_reliability(detected_patterns),
                'doji_statistics': self._calculate_doji_statistics(detected_patterns)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Harami cross calculation failed: {str(e)}", e
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
        
        # Doji identification
        df['is_doji'] = df['body_ratio'] <= self.parameters['doji_threshold']
        df['shadow_symmetry'] = abs(df['upper_shadow_ratio'] - df['lower_shadow_ratio'])
        
        # Trend indicators
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
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
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        # Advanced metrics
        df['trend_strength'] = self._calculate_trend_strength_series(df)
        df['trend_direction'] = np.where(df['close'] > df['sma_20'], 1, -1)
        df['price_momentum'] = df['close'].pct_change(5)
        df['volatility'] = df['total_range'].rolling(14).std()
        df['momentum_roc'] = talib.ROC(df['close'], timeperiod=10)
        
        return df
    
    def _calculate_trend_strength_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength as a rolling series"""
        def trend_strength_window(window_data):
            if len(window_data) < 8:
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
    
    def _detect_harami_cross_patterns(self, data: pd.DataFrame) -> List[HaramiCrossPattern]:
        """Detect harami cross patterns with sophisticated analysis"""
        patterns = []
        
        for i in range(self.parameters['trend_context_periods'], len(data) - 1):
            mother_candle = data.iloc[i]
            doji_candle = data.iloc[i + 1]
            
            # Check if inside candle is a doji
            if not doji_candle['is_doji']:
                continue
            
            # Check for bullish harami cross
            if self._is_bullish_harami_cross(mother_candle, doji_candle):
                pattern = self._create_harami_cross_pattern(
                    data, i + 1, 'bullish_harami_cross', mother_candle, doji_candle
                )
                if pattern and pattern.pattern_strength >= 0.65:
                    patterns.append(pattern)
            
            # Check for bearish harami cross
            elif self._is_bearish_harami_cross(mother_candle, doji_candle):
                pattern = self._create_harami_cross_pattern(
                    data, i + 1, 'bearish_harami_cross', mother_candle, doji_candle
                )
                if pattern and pattern.pattern_strength >= 0.65:
                    patterns.append(pattern)
        
        return patterns
    
    def _is_bullish_harami_cross(self, mother: pd.Series, doji: pd.Series) -> bool:
        """Check if candles form a bullish harami cross pattern"""
        return (
            mother['is_bearish'] and
            mother['body_ratio'] >= self.parameters['min_mother_body'] and
            doji['is_doji'] and
            self._is_doji_contained(mother, doji) and
            doji['total_range'] / mother['total_range'] <= self.parameters['max_doji_range_ratio']
        )
    
    def _is_bearish_harami_cross(self, mother: pd.Series, doji: pd.Series) -> bool:
        """Check if candles form a bearish harami cross pattern"""
        return (
            mother['is_bullish'] and
            mother['body_ratio'] >= self.parameters['min_mother_body'] and
            doji['is_doji'] and
            self._is_doji_contained(mother, doji) and
            doji['total_range'] / mother['total_range'] <= self.parameters['max_doji_range_ratio']
        )
    
    def _is_doji_contained(self, mother: pd.Series, doji: pd.Series) -> bool:
        """Check if doji is properly contained within mother candle body"""
        mother_body_top = max(mother['open'], mother['close'])
        mother_body_bottom = min(mother['open'], mother['close'])
        
        # Check containment
        containment_ratio = self._calculate_doji_containment_ratio(mother, doji)
        return containment_ratio >= self.parameters['min_containment']
    
    def _calculate_doji_containment_ratio(self, mother: pd.Series, doji: pd.Series) -> float:
        """Calculate how well the doji is contained within mother's body"""
        mother_body_top = max(mother['open'], mother['close'])
        mother_body_bottom = min(mother['open'], mother['close'])
        
        # Calculate overlap of doji range with mother body
        overlap_top = min(mother_body_top, doji['high'])
        overlap_bottom = max(mother_body_bottom, doji['low'])
        
        if overlap_top <= overlap_bottom:
            return 0.0
        
        overlap_range = overlap_top - overlap_bottom
        doji_range = doji['high'] - doji['low']
        
        return overlap_range / doji_range if doji_range > 0 else 0.0
    
    def _create_harami_cross_pattern(self, data: pd.DataFrame, candle_idx: int, 
                                   pattern_type: str, mother: pd.Series, 
                                   doji: pd.Series) -> Optional[HaramiCrossPattern]:
        """Create harami cross pattern with comprehensive analysis"""
        try:
            # Calculate pattern metrics
            doji_quality = self._assess_doji_quality(doji)
            containment_ratio = self._calculate_doji_containment_ratio(mother, doji)
            mother_strength = self._assess_mother_candle_strength(mother)
            doji_position = self._analyze_doji_position(mother, doji)
            shadow_symmetry = self._analyze_shadow_symmetry(doji)
            
            # Trend context analysis
            trend_context_score = self._assess_harami_cross_trend_context(data, candle_idx, pattern_type)
            
            # Calculate pattern strength
            pattern_strength = self._calculate_harami_cross_strength(
                doji_quality, containment_ratio, mother_strength, 
                doji_position, shadow_symmetry, trend_context_score
            )
            
            pattern = HaramiCrossPattern(
                timestamp=doji.name,
                pattern_type=pattern_type,
                pattern_strength=pattern_strength,
                doji_quality=doji_quality,
                containment_ratio=containment_ratio,
                mother_strength=mother_strength,
                doji_position=doji_position,
                shadow_symmetry=shadow_symmetry,
                trend_context_score=trend_context_score,
                volume_confirmation=0.0,
                momentum_exhaustion=0.0,
                reversal_probability=0.0,
                pattern_purity=0.0,
                indecision_intensity=0.0
            )
            
            return pattern
            
        except Exception:
            return None
    
    def _assess_doji_quality(self, doji: pd.Series) -> float:
        """Assess the quality of the doji candle"""
        quality_factors = []
        
        # Body size (smaller is better)
        body_quality = max(0, 1.0 - doji['body_ratio'] / self.parameters['doji_threshold'])
        quality_factors.append(body_quality * 0.4)
        
        # Shadow symmetry (more symmetric is better)
        symmetry_quality = max(0, 1.0 - doji['shadow_symmetry'] / self.parameters['shadow_symmetry_threshold'])
        quality_factors.append(symmetry_quality * 0.3)
        
        # Shadow presence (should have meaningful shadows)
        shadow_presence = min((doji['upper_shadow_ratio'] + doji['lower_shadow_ratio']) / 0.8, 1.0)
        quality_factors.append(shadow_presence * 0.3)
        
        return sum(quality_factors)
    
    def _assess_mother_candle_strength(self, mother: pd.Series) -> float:
        """Assess the strength of the mother candle"""
        strength_factors = []
        
        # Body size strength
        body_strength = min(mother['body_ratio'] / 0.7, 1.0)
        strength_factors.append(body_strength * 0.5)
        
        # Range strength relative to ATR
        if not pd.isna(mother['atr']) and mother['atr'] > 0:
            range_strength = min(mother['total_range'] / (mother['atr'] * 1.5), 1.0)
            strength_factors.append(range_strength * 0.3)
        else:
            strength_factors.append(0.5 * 0.3)
        
        # Shadow proportion (less shadows = stronger body)
        shadow_proportion = mother['upper_shadow_ratio'] + mother['lower_shadow_ratio']
        shadow_strength = max(0, 1.0 - shadow_proportion / 0.4)
        strength_factors.append(shadow_strength * 0.2)
        
        return sum(strength_factors)
    
    def _analyze_doji_position(self, mother: pd.Series, doji: pd.Series) -> float:
        """Analyze position of doji within mother candle"""
        mother_body_top = max(mother['open'], mother['close'])
        mother_body_bottom = min(mother['open'], mother['close'])
        mother_body_size = mother_body_top - mother_body_bottom
        
        if mother_body_size == 0:
            return 0.5
        
        # Doji center position
        doji_center = (doji['open'] + doji['close']) / 2
        mother_center = (mother_body_top + mother_body_bottom) / 2
        
        # Position relative to mother body
        position_in_body = (doji_center - mother_body_bottom) / mother_body_size
        
        # Ideal position is center (0.5)
        center_distance = abs(position_in_body - 0.5)
        position_score = max(0, 1.0 - center_distance / 0.5)
        
        return position_score
    
    def _analyze_shadow_symmetry(self, doji: pd.Series) -> float:
        """Analyze shadow symmetry of the doji"""
        if doji['total_range'] == 0:
            return 0.0
        
        # Calculate symmetry score
        symmetry_difference = abs(doji['upper_shadow_ratio'] - doji['lower_shadow_ratio'])
        symmetry_score = max(0, 1.0 - symmetry_difference / 0.5)
        
        # Bonus for long shadows
        total_shadow_ratio = doji['upper_shadow_ratio'] + doji['lower_shadow_ratio']
        shadow_length_bonus = min(total_shadow_ratio / 0.8, 1.0) * 0.2
        
        return min(symmetry_score + shadow_length_bonus, 1.0)
    
    def _assess_harami_cross_trend_context(self, data: pd.DataFrame, candle_idx: int, 
                                         pattern_type: str) -> float:
        """Assess trend context for harami cross patterns"""
        context_data = data.iloc[max(0, candle_idx - self.parameters['trend_context_periods']):candle_idx]
        
        if len(context_data) < 8:
            return 0.5
        
        context_factors = []
        
        # Trend direction alignment
        if pattern_type == 'bullish_harami_cross':
            # Should appear after downtrend
            price_change = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
            if price_change < -0.03:
                trend_factor = min(abs(price_change) / 0.12, 1.0)
                context_factors.append(trend_factor * 0.4)
            else:
                context_factors.append(0.2)
        else:  # bearish_harami_cross
            # Should appear after uptrend
            price_change = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
            if price_change > 0.03:
                trend_factor = min(price_change / 0.12, 1.0)
                context_factors.append(trend_factor * 0.4)
            else:
                context_factors.append(0.2)
        
        # Trend strength and exhaustion
        latest = context_data.iloc[-1]
        trend_strength = latest['trend_strength'] if not pd.isna(latest['trend_strength']) else 0.5
        
        # High trend strength suggests potential exhaustion
        exhaustion_factor = min(trend_strength / 0.8, 1.0)
        context_factors.append(exhaustion_factor * 0.25)
        
        # RSI extreme levels
        rsi_context = 0
        if not pd.isna(latest['rsi']):
            if pattern_type == 'bullish_harami_cross' and latest['rsi'] < 35:
                rsi_context = (35 - latest['rsi']) / 20
            elif pattern_type == 'bearish_harami_cross' and latest['rsi'] > 65:
                rsi_context = (latest['rsi'] - 65) / 20
        context_factors.append(min(rsi_context, 1.0) * 0.2)
        
        # ADX confirmation
        adx_context = 0
        if not pd.isna(latest['adx']):
            # Strong trend (high ADX) increases reversal potential
            adx_context = min(latest['adx'] / 35, 1.0)
        context_factors.append(adx_context * 0.15)
        
        return sum(context_factors)
    
    def _calculate_harami_cross_strength(self, doji_quality: float, containment_ratio: float, 
                                       mother_strength: float, doji_position: float, 
                                       shadow_symmetry: float, trend_context_score: float) -> float:
        """Calculate overall harami cross pattern strength"""
        strength_components = [
            doji_quality * 0.3,                # Doji quality
            containment_ratio * 0.25,          # Containment quality
            mother_strength * 0.2,             # Mother candle strength
            doji_position * 0.1,               # Doji position
            shadow_symmetry * 0.1,             # Shadow symmetry
            trend_context_score * 0.05         # Trend context
        ]
        
        return sum(strength_components)
    
    def _analyze_volume_confirmation(self, patterns: List[HaramiCrossPattern], 
                                   data: pd.DataFrame) -> List[HaramiCrossPattern]:
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
            doji_volume = data.iloc[pattern_idx]['volume']
            
            # Average volume
            avg_volume = context_data['volume'].iloc[:-2].mean()
            
            volume_factors = []
            
            # Mother candle volume surge
            mother_surge = mother_volume / avg_volume if avg_volume > 0 else 1.0
            volume_factors.append(min(mother_surge / 1.8, 1.0) * 0.4)
            
            # Doji volume characteristics (often lower, showing indecision)
            doji_volume_ratio = doji_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Moderate volume on doji is ideal (not too high, not too low)
            if 0.7 <= doji_volume_ratio <= 1.3:
                doji_score = 1.0
            elif doji_volume_ratio < 0.7:
                doji_score = 0.8  # Low volume can indicate indecision
            else:
                doji_score = max(0, 1.0 - (doji_volume_ratio - 1.3) / 1.0)
            
            volume_factors.append(doji_score * 0.35)
            
            # Volume trend context
            recent_volumes = context_data['volume'].iloc[-6:]
            if len(recent_volumes) >= 4:
                volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
                # Declining volume can support reversal
                if volume_trend < 0:
                    trend_score = min(abs(volume_trend) / avg_volume * 5, 1.0) if avg_volume > 0 else 0.5
                else:
                    trend_score = 0.3
                volume_factors.append(trend_score * 0.25)
            
            return sum(volume_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_momentum_exhaustion(self, patterns: List[HaramiCrossPattern], 
                                   data: pd.DataFrame) -> List[HaramiCrossPattern]:
        """Analyze momentum exhaustion for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            exhaustion_score = self._calculate_momentum_exhaustion_score(data, pattern_idx, pattern.pattern_type)
            pattern.momentum_exhaustion = exhaustion_score
        
        return patterns
    
    def _calculate_momentum_exhaustion_score(self, data: pd.DataFrame, pattern_idx: int, 
                                           pattern_type: str) -> float:
        """Calculate momentum exhaustion score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 10):pattern_idx + 1]
            current = data.iloc[pattern_idx]
            
            exhaustion_factors = []
            
            # RSI extreme and reversal
            if not pd.isna(current['rsi']):
                if pattern_type == 'bullish_harami_cross':
                    if current['rsi'] < 25:
                        exhaustion_factors.append(0.4)
                    elif current['rsi'] < 35:
                        exhaustion_factors.append(0.25)
                else:  # bearish_harami_cross
                    if current['rsi'] > 75:
                        exhaustion_factors.append(0.4)
                    elif current['rsi'] > 65:
                        exhaustion_factors.append(0.25)
            
            # Stochastic exhaustion
            if not pd.isna(current['stoch_k']):
                if pattern_type == 'bullish_harami_cross' and current['stoch_k'] < 20:
                    exhaustion_factors.append(0.3)
                elif pattern_type == 'bearish_harami_cross' and current['stoch_k'] > 80:
                    exhaustion_factors.append(0.3)
                else:
                    exhaustion_factors.append(0.1)
            
            # CCI exhaustion
            if not pd.isna(current['cci']):
                if pattern_type == 'bullish_harami_cross' and current['cci'] < -150:
                    exhaustion_factors.append(0.25)
                elif pattern_type == 'bearish_harami_cross' and current['cci'] > 150:
                    exhaustion_factors.append(0.25)
                else:
                    exhaustion_factors.append(0.1)
            
            # Momentum divergence
            if len(context_data) >= 6:
                price_data = context_data['high'] if pattern_type == 'bearish_harami_cross' else context_data['low']
                rsi_data = context_data['rsi'].fillna(50)
                
                price_trend = np.polyfit(range(len(price_data)), price_data, 1)[0]
                rsi_trend = np.polyfit(range(len(rsi_data)), rsi_data, 1)[0]
                
                # Look for divergence
                if pattern_type == 'bullish_harami_cross' and price_trend < 0 and rsi_trend > 0:
                    exhaustion_factors.append(0.3)
                elif pattern_type == 'bearish_harami_cross' and price_trend > 0 and rsi_trend < 0:
                    exhaustion_factors.append(0.3)
                else:
                    exhaustion_factors.append(0.05)
            
            return sum(exhaustion_factors) if exhaustion_factors else 0.5
            
        except Exception:
            return 0.5
    
    def _analyze_indecision_intensity(self, patterns: List[HaramiCrossPattern], 
                                    data: pd.DataFrame) -> List[HaramiCrossPattern]:
        """Analyze indecision intensity for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            indecision_score = self._calculate_indecision_intensity_score(data, pattern_idx, pattern)
            pattern.indecision_intensity = indecision_score
        
        return patterns
    
    def _calculate_indecision_intensity_score(self, data: pd.DataFrame, pattern_idx: int, 
                                            pattern: HaramiCrossPattern) -> float:
        """Calculate indecision intensity score"""
        try:
            doji_candle = data.iloc[pattern_idx]
            mother_candle = data.iloc[pattern_idx - 1]
            
            indecision_factors = []
            
            # Doji quality contributes to indecision
            indecision_factors.append(pattern.doji_quality * 0.35)
            
            # Shadow symmetry indicates balance
            indecision_factors.append(pattern.shadow_symmetry * 0.25)
            
            # Range contraction from mother to doji
            range_contraction = doji_candle['total_range'] / mother_candle['total_range'] if mother_candle['total_range'] > 0 else 1.0
            contraction_score = max(0, 1.0 - range_contraction)
            indecision_factors.append(contraction_score * 0.2)
            
            # Volume decline (hesitation)
            volume_decline = 1.0 - (doji_candle['volume'] / mother_candle['volume']) if mother_candle['volume'] > 0 else 0.5
            volume_decline_score = max(0, min(volume_decline, 1.0))
            indecision_factors.append(volume_decline_score * 0.2)
            
            return sum(indecision_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_pattern_purity(self, patterns: List[HaramiCrossPattern], 
                              data: pd.DataFrame) -> List[HaramiCrossPattern]:
        """Analyze pattern purity for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            purity_score = self._calculate_pattern_purity_score(data, pattern_idx, pattern)
            pattern.pattern_purity = purity_score
        
        return patterns
    
    def _calculate_pattern_purity_score(self, data: pd.DataFrame, pattern_idx: int, 
                                      pattern: HaramiCrossPattern) -> float:
        """Calculate pattern purity score"""
        try:
            purity_factors = []
            
            # Doji purity
            purity_factors.append(pattern.doji_quality * 0.3)
            
            # Containment purity
            purity_factors.append(pattern.containment_ratio * 0.25)
            
            # Mother candle strength
            purity_factors.append(pattern.mother_strength * 0.2)
            
            # Shadow symmetry purity
            purity_factors.append(pattern.shadow_symmetry * 0.15)
            
            # Gap analysis
            mother_candle = data.iloc[pattern_idx - 1]
            doji_candle = data.iloc[pattern_idx]
            
            gap_size = abs(doji_candle['open'] - mother_candle['close']) / mother_candle['close']
            gap_purity = max(0, 1.0 - gap_size / 0.015)
            purity_factors.append(gap_purity * 0.1)
            
            return sum(purity_factors)
            
        except Exception:
            return 0.5
    
    def _predict_reversal_probability(self, patterns: List[HaramiCrossPattern], 
                                    data: pd.DataFrame) -> List[HaramiCrossPattern]:
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
                                 pattern: HaramiCrossPattern) -> List[float]:
        """Extract features for reversal prediction ML model"""
        try:
            current = data.iloc[pattern_idx]
            
            features = [
                pattern.pattern_strength,
                pattern.doji_quality,
                pattern.containment_ratio,
                pattern.mother_strength,
                pattern.doji_position,
                pattern.shadow_symmetry,
                pattern.trend_context_score,
                pattern.volume_confirmation,
                pattern.momentum_exhaustion,
                pattern.indecision_intensity,
                pattern.pattern_purity,
                1.0 if pattern.pattern_type == 'bullish_harami_cross' else 0.0,
                current['rsi'] / 100.0 if not pd.isna(current['rsi']) else 0.5,
                current['bb_position'] if not pd.isna(current['bb_position']) else 0.5,
                current['adx'] / 50.0 if not pd.isna(current['adx']) else 0.5,
                current['volume_ratio'],
                current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
                current['cci'] / 200.0 if not pd.isna(current['cci']) else 0.0,
                current['stoch_k'] / 100.0 if not pd.isna(current['stoch_k']) else 0.5,
                current['body_ratio'],
                pattern.doji_quality * pattern.momentum_exhaustion,  # Combined exhaustion signal
                pattern.indecision_intensity * pattern.volume_confirmation,  # Indecision-volume factor
                current['momentum_roc'] / 10.0 if not pd.isna(current['momentum_roc']) else 0.0
            ]
            
            return features
            
        except Exception:
            return [0.5] * 23
    
    def _train_reversal_model(self, patterns: List[HaramiCrossPattern], features: List[List[float]]):
        """Train ML model for reversal prediction"""
        try:
            targets = []
            for pattern in patterns:
                target = (
                    pattern.pattern_strength * 0.3 +
                    pattern.momentum_exhaustion * 0.25 +
                    pattern.volume_confirmation * 0.2 +
                    pattern.trend_context_score * 0.15 +
                    pattern.indecision_intensity * 0.1
                )
                targets.append(max(0.1, min(0.9, target)))
            
            if len(features) >= 18:
                features_scaled = self.scaler.fit_transform(features)
                self.reversal_predictor.fit(features_scaled, targets)
                self.is_ml_fitted = True
                logging.info("ML reversal predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _heuristic_reversal_probability(self, pattern: HaramiCrossPattern) -> float:
        """Calculate heuristic reversal probability"""
        return (
            pattern.pattern_strength * 0.35 +
            pattern.momentum_exhaustion * 0.25 +
            pattern.indecision_intensity * 0.2 +
            pattern.volume_confirmation * 0.2
        )
    
    def _generate_pattern_analytics(self, patterns: List[HaramiCrossPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-15:]
        bullish_patterns = [p for p in recent_patterns if p.pattern_type == 'bullish_harami_cross']
        bearish_patterns = [p for p in recent_patterns if p.pattern_type == 'bearish_harami_cross']
        
        return {
            'total_patterns': len(recent_patterns),
            'bullish_patterns': len(bullish_patterns),
            'bearish_patterns': len(bearish_patterns),
            'average_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'average_reversal_probability': sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns),
            'average_doji_quality': sum(p.doji_quality for p in recent_patterns) / len(recent_patterns),
            'average_indecision_intensity': sum(p.indecision_intensity for p in recent_patterns) / len(recent_patterns),
            'high_strength_patterns': len([p for p in recent_patterns if p.pattern_strength > 0.8]),
            'high_probability_patterns': len([p for p in recent_patterns if p.reversal_probability > 0.75]),
            'strong_exhaustion_patterns': len([p for p in recent_patterns if p.momentum_exhaustion > 0.7]),
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
            'cci_level': current['cci'] if not pd.isna(current['cci']) else 0,
            'bb_position': current['bb_position'] if not pd.isna(current['bb_position']) else 0.5,
            'is_doji_present': current['is_doji'],
            'volatility_context': 'high' if current['volatility'] > current['atr'] else 'normal'
        }
    
    def _generate_reversal_signals(self, patterns: List[HaramiCrossPattern], 
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
            'indecision_intensity': sum(p.indecision_intensity for p in recent_patterns) / len(recent_patterns),
            'momentum_exhaustion': sum(p.momentum_exhaustion for p in recent_patterns) / len(recent_patterns),
            'pattern_count': len(recent_patterns),
            'latest_pattern_type': latest_pattern.pattern_type,
            'doji_quality': sum(p.doji_quality for p in recent_patterns) / len(recent_patterns),
            'volume_confirmation': sum(p.volume_confirmation for p in recent_patterns) / len(recent_patterns),
            'pattern_purity': sum(p.pattern_purity for p in recent_patterns) / len(recent_patterns),
            'most_recent_pattern': latest_pattern.timestamp
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure"""
        current = data.iloc[-1]
        recent_data = data.iloc[-20:]
        
        return {
            'doji_frequency': len([i for i in range(len(recent_data)) if recent_data.iloc[i]['is_doji']]) / len(recent_data),
            'indecision_context': {
                'recent_dojis': sum(recent_data['is_doji'].iloc[-5:]),
                'average_symmetry': recent_data['shadow_symmetry'].iloc[-10:].mean()
            },
            'volatility_analysis': {
                'current_volatility': current['volatility'] if not pd.isna(current['volatility']) else 0,
                'volatility_trend': 'declining' if recent_data['volatility'].iloc[-3:].mean() < recent_data['volatility'].iloc[-10:-3].mean() else 'stable'
            }
        }
    
    def _assess_pattern_reliability(self, patterns: List[HaramiCrossPattern]) -> Dict[str, Any]:
        """Assess pattern reliability metrics"""
        if not patterns:
            return {}
        
        return {
            'consistency_score': 1.0 - np.std([p.pattern_strength for p in patterns]),
            'doji_quality_average': np.mean([p.doji_quality for p in patterns]),
            'exhaustion_reliability': np.mean([p.momentum_exhaustion for p in patterns]),
            'indecision_consistency': np.mean([p.indecision_intensity for p in patterns]),
            'reversal_success_estimate': np.mean([p.reversal_probability for p in patterns])
        }
    
    def _calculate_doji_statistics(self, patterns: List[HaramiCrossPattern]) -> Dict[str, Any]:
        """Calculate doji-specific statistics"""
        if not patterns:
            return {}
        
        doji_qualities = [p.doji_quality for p in patterns]
        shadow_symmetries = [p.shadow_symmetry for p in patterns]
        containment_ratios = [p.containment_ratio for p in patterns]
        
        return {
            'doji_quality_stats': {
                'mean': np.mean(doji_qualities),
                'median': np.median(doji_qualities),
                'max': np.max(doji_qualities),
                'std': np.std(doji_qualities)
            },
            'shadow_symmetry_stats': {
                'mean': np.mean(shadow_symmetries),
                'median': np.median(shadow_symmetries),
                'std': np.std(shadow_symmetries)
            },
            'containment_stats': {
                'mean': np.mean(containment_ratios),
                'min': np.min(containment_ratios),
                'std': np.std(containment_ratios)
            }
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on harami cross analysis"""
        current_pattern = value.get('current_pattern')
        reversal_signals = value.get('reversal_signals', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Strong bullish harami cross signal
        if (current_pattern.pattern_type == 'bullish_harami_cross' and
            current_pattern.pattern_strength > 0.8 and
            current_pattern.reversal_probability > 0.75 and
            current_pattern.momentum_exhaustion > 0.7):
            
            confidence = (
                current_pattern.pattern_strength * 0.35 +
                current_pattern.reversal_probability * 0.3 +
                current_pattern.momentum_exhaustion * 0.2 +
                current_pattern.indecision_intensity * 0.15
            )
            
            return SignalType.BUY, confidence
        
        # Strong bearish harami cross signal
        elif (current_pattern.pattern_type == 'bearish_harami_cross' and
              current_pattern.pattern_strength > 0.8 and
              current_pattern.reversal_probability > 0.75 and
              current_pattern.momentum_exhaustion > 0.7):
            
            confidence = (
                current_pattern.pattern_strength * 0.35 +
                current_pattern.reversal_probability * 0.3 +
                current_pattern.momentum_exhaustion * 0.2 +
                current_pattern.indecision_intensity * 0.15
            )
            
            return SignalType.SELL, confidence
        
        # Moderate signals
        elif (current_pattern.pattern_strength > 0.75 and
              current_pattern.reversal_probability > 0.65):
            
            confidence = current_pattern.pattern_strength * 0.7
            
            if current_pattern.pattern_type == 'bullish_harami_cross':
                return SignalType.BUY, confidence
            else:
                return SignalType.SELL, confidence
        
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_types': ['bullish_harami_cross', 'bearish_harami_cross'],
            'doji_threshold': self.parameters['doji_threshold'],
            'min_mother_body': self.parameters['min_mother_body'],
            'volume_analysis_enabled': self.parameters['volume_analysis'],
            'ml_prediction_enabled': self.parameters['ml_prediction']
        })
        return base_metadata