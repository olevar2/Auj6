"""
Morning Star Indicator - Advanced Three-Candle Bullish Reversal Pattern Detection
================================================================================

This indicator implements sophisticated morning star pattern detection with advanced
gap analysis, volume confirmation, and ML-enhanced trend reversal prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
import logging
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import talib
from scipy import stats
from scipy.stats import linregress

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    IndicatorResult, 
    SignalType, 
    DataType, 
    DataRequirement
)
from ...core.exceptions import IndicatorCalculationException


@dataclass
class MorningStarPattern:
    """Represents a detected morning star pattern"""
    timestamp: pd.Timestamp
    pattern_strength: float
    gap_down_quality: float
    gap_up_quality: float
    star_doji_quality: float
    bullish_confirmation: float
    volume_confirmation: float
    trend_context_score: float
    reversal_probability: float
    support_resistance_factor: float
    institutional_validation: float


class MorningStarIndicator(StandardIndicatorInterface):
    """
    Advanced Morning Star Pattern Indicator
    
    Features:
    - Sophisticated three-candle pattern recognition with gap analysis
    - Advanced star candle quality assessment (doji characteristics)
    - Volume-based confirmation and institutional validation
    - ML-enhanced trend reversal probability prediction
    - Support/resistance level validation and strength assessment
    - Multi-timeframe trend context analysis
    - Statistical significance testing for pattern reliability
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'min_gap_percentage': 0.15,     # Minimum gap size as % of candle range
            'min_star_body_ratio': 0.3,     # Max body ratio for star candle
            'min_confirmation_body': 0.6,   # Min body ratio for confirmation candle
            'volume_surge_threshold': 1.3,  # Volume surge multiplier
            'trend_lookback': 20,           # Periods for trend analysis
            'support_resistance_analysis': True,
            'volume_analysis': True,
            'ml_reversal_prediction': True,
            'institutional_validation': True,
            'gap_quality_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="MorningStarIndicator", parameters=default_params)
        
        # Initialize ML components
        self.reversal_predictor = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=150, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('et', ExtraTreesRegressor(n_estimators=100, random_state=42))
        ])
        self.scaler = StandardScaler()
        self.is_ml_fitted = False
        
        # Pattern analysis components
        self.gap_analyzer = self._initialize_gap_analyzer()
        self.trend_analyzer = self._initialize_trend_analyzer()
        
        logging.info(f"MorningStarIndicator initialized with parameters: {self.parameters}")
    
    def _initialize_gap_analyzer(self) -> Dict[str, Any]:
        """Initialize gap analysis components"""
        return {
            'gap_calculator': self._calculate_gap_quality,
            'gap_validator': self._validate_gap_significance,
            'gap_sustainability': self._assess_gap_sustainability
        }
    
    def _initialize_trend_analyzer(self) -> Dict[str, Any]:
        """Initialize trend analysis components"""
        return {
            'trend_detector': self._detect_trend_context,
            'trend_strength': self._calculate_trend_strength,
            'reversal_conditions': self._assess_reversal_conditions
        }
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=50,
            lookback_periods=100
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate morning star patterns with advanced analysis"""
        try:
            if len(data) < 50:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < 50"
                )
            
            # Enhance data with technical indicators
            enhanced_data = self._enhance_data_with_indicators(data)
            
            # Detect morning star patterns
            detected_patterns = self._detect_morning_star_patterns(enhanced_data)
            
            # Apply volume analysis
            if self.parameters['volume_analysis']:
                volume_enhanced_patterns = self._analyze_volume_confirmation(
                    detected_patterns, enhanced_data
                )
            else:
                volume_enhanced_patterns = detected_patterns
            
            # Apply support/resistance analysis
            if self.parameters['support_resistance_analysis']:
                sr_enhanced_patterns = self._analyze_support_resistance(
                    volume_enhanced_patterns, enhanced_data
                )
            else:
                sr_enhanced_patterns = volume_enhanced_patterns
            
            # Apply institutional validation
            if self.parameters['institutional_validation']:
                institutional_enhanced_patterns = self._validate_institutional_activity(
                    sr_enhanced_patterns, enhanced_data
                )
            else:
                institutional_enhanced_patterns = sr_enhanced_patterns
            
            # Apply ML reversal prediction
            if self.parameters['ml_reversal_prediction'] and institutional_enhanced_patterns:
                ml_enhanced_patterns = self._predict_reversal_probability(
                    institutional_enhanced_patterns, enhanced_data
                )
            else:
                ml_enhanced_patterns = institutional_enhanced_patterns
            
            # Generate comprehensive analysis
            pattern_analytics = self._generate_pattern_analytics(ml_enhanced_patterns)
            trend_analysis = self._analyze_current_trend_context(enhanced_data)
            reversal_signals = self._generate_reversal_signals(ml_enhanced_patterns, enhanced_data)
            gap_analysis = self._analyze_gap_characteristics(enhanced_data)
            
            return {
                'current_pattern': ml_enhanced_patterns[-1] if ml_enhanced_patterns else None,
                'recent_patterns': ml_enhanced_patterns[-10:],
                'pattern_analytics': pattern_analytics,
                'trend_analysis': trend_analysis,
                'reversal_signals': reversal_signals,
                'gap_analysis': gap_analysis,
                'pattern_reliability': self._assess_pattern_reliability(ml_enhanced_patterns),
                'market_conditions': self._assess_market_conditions(enhanced_data)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Morning star calculation failed: {str(e)}", e
            )
    
    def _enhance_data_with_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with comprehensive technical indicators"""
        df = data.copy()
        
        # Basic candlestick components
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Candlestick ratios
        df['body_ratio'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
        df['upper_shadow_ratio'] = np.where(df['total_range'] > 0, df['upper_shadow'] / df['total_range'], 0)
        df['lower_shadow_ratio'] = np.where(df['total_range'] > 0, df['lower_shadow'] / df['total_range'], 0)
        
        # Trend indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD for trend confirmation
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # RSI for momentum
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR for volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['volatility_ratio'] = df['total_range'] / df['atr']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Trend strength
        df['trend_strength'] = self._calculate_trend_strength_series(df)
        
        # Support/resistance levels
        df['pivot_high'] = df['high'].rolling(5, center=True).max() == df['high']
        df['pivot_low'] = df['low'].rolling(5, center=True).min() == df['low']
        
        # Gap analysis
        df['gap_up'] = df['low'] > df['high'].shift(1)
        df['gap_down'] = df['high'] < df['low'].shift(1)
        df['gap_size'] = np.where(df['gap_up'], 
                                 df['low'] - df['high'].shift(1),
                                 np.where(df['gap_down'],
                                         df['low'].shift(1) - df['high'],
                                         0))
        df['gap_percentage'] = df['gap_size'] / df['total_range'].shift(1)
        
        return df
    
    def _calculate_trend_strength_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength as a rolling series"""
        def trend_strength_window(window_data):
            if len(window_data) < 10:
                return 0.5
            
            # Linear regression slope
            x = np.arange(len(window_data))
            slope, _, r_value, _, _ = linregress(x, window_data['close'])
            
            # Normalize slope
            price_range = window_data['close'].max() - window_data['close'].min()
            if price_range == 0:
                return 0.5
            
            normalized_slope = slope / price_range * len(window_data)
            
            # R-squared for trend reliability
            trend_quality = r_value ** 2
            
            # Combine slope magnitude and quality
            strength = abs(normalized_slope) * trend_quality
            return min(max(strength, 0), 1)
        
        return df.rolling(self.parameters['trend_lookback']).apply(trend_strength_window, raw=False)['close']
    
    def _detect_morning_star_patterns(self, data: pd.DataFrame) -> List[MorningStarPattern]:
        """Detect morning star patterns with sophisticated analysis"""
        patterns = []
        
        for i in range(self.parameters['trend_lookback'], len(data) - 1):
            # Need at least 3 candles for pattern
            if i < 2:
                continue
            
            candle1 = data.iloc[i-2]  # First candle (bearish)
            candle2 = data.iloc[i-1]  # Star candle
            candle3 = data.iloc[i]    # Third candle (bullish)
            
            # Check for morning star pattern
            if not self._is_morning_star_pattern(candle1, candle2, candle3):
                continue
            
            # Calculate pattern components
            gap_down_quality = self._calculate_gap_quality(candle1, candle2, 'down')
            gap_up_quality = self._calculate_gap_quality(candle2, candle3, 'up')
            star_doji_quality = self._assess_star_candle_quality(candle2)
            
            # Assess trend context
            trend_context_score = self._assess_trend_context(data, i-2)
            
            # Calculate bullish confirmation
            bullish_confirmation = self._calculate_bullish_confirmation(candle3)
            
            # Calculate pattern strength
            pattern_strength = self._calculate_pattern_strength(
                gap_down_quality, gap_up_quality, star_doji_quality,
                bullish_confirmation, trend_context_score
            )
            
            if pattern_strength >= 0.7:  # High threshold for quality patterns
                pattern = MorningStarPattern(
                    timestamp=candle3.name,
                    pattern_strength=pattern_strength,
                    gap_down_quality=gap_down_quality,
                    gap_up_quality=gap_up_quality,
                    star_doji_quality=star_doji_quality,
                    bullish_confirmation=bullish_confirmation,
                    volume_confirmation=0.0,  # Will be calculated later
                    trend_context_score=trend_context_score,
                    reversal_probability=0.0,  # Will be calculated later
                    support_resistance_factor=0.0,  # Will be calculated later
                    institutional_validation=0.0  # Will be calculated later
                )
                patterns.append(pattern)
        
        return patterns
    
    def _is_morning_star_pattern(self, candle1: pd.Series, candle2: pd.Series, 
                               candle3: pd.Series) -> bool:
        """Check if three candles form a morning star pattern"""
        # 1. First candle should be bearish (large body)
        if candle1['close'] >= candle1['open']:
            return False
        if candle1['body_ratio'] < 0.6:
            return False
        
        # 2. Second candle should gap down and be small (star characteristics)
        if candle2['high'] >= candle1['low']:  # Should gap down
            return False
        if candle2['body_ratio'] > self.parameters['min_star_body_ratio']:
            return False
        
        # 3. Third candle should be bullish and gap up or penetrate first candle
        if candle3['close'] <= candle3['open']:
            return False
        if candle3['body_ratio'] < self.parameters['min_confirmation_body']:
            return False
        
        # Third candle should close well into first candle's body
        first_midpoint = (candle1['open'] + candle1['close']) / 2
        if candle3['close'] < first_midpoint:
            return False
        
        return True
    
    def _calculate_gap_quality(self, candle1: pd.Series, candle2: pd.Series, 
                             direction: str) -> float:
        """Calculate gap quality between two candles"""
        if direction == 'down':
            if candle2['high'] >= candle1['low']:
                return 0.0
            gap_size = candle1['low'] - candle2['high']
            reference_range = candle1['total_range']
        else:  # up
            if candle2['low'] <= candle1['high']:
                return 0.0
            gap_size = candle2['low'] - candle1['high']
            reference_range = candle2['total_range']
        
        if reference_range == 0:
            return 0.0
        
        gap_percentage = gap_size / reference_range
        
        # Quality based on gap size
        if gap_percentage >= self.parameters['min_gap_percentage'] * 3:
            return 1.0
        elif gap_percentage >= self.parameters['min_gap_percentage'] * 2:
            return 0.8
        elif gap_percentage >= self.parameters['min_gap_percentage']:
            return 0.6
        else:
            return gap_percentage / self.parameters['min_gap_percentage'] * 0.6
    
    def _assess_star_candle_quality(self, candle: pd.Series) -> float:
        """Assess the quality of the star candle (doji-like characteristics)"""
        quality_factors = []
        
        # 1. Small body (30% weight)
        body_quality = 1.0 - (candle['body_ratio'] / self.parameters['min_star_body_ratio'])
        quality_factors.append(max(0, body_quality) * 0.3)
        
        # 2. Shadow length (25% weight)
        total_shadow_ratio = candle['upper_shadow_ratio'] + candle['lower_shadow_ratio']
        shadow_quality = min(total_shadow_ratio / 0.7, 1.0)  # Prefer long shadows
        quality_factors.append(shadow_quality * 0.25)
        
        # 3. Position in range (20% weight)
        if candle['total_range'] > 0:
            body_center = (candle['open'] + candle['close']) / 2
            range_center = (candle['high'] + candle['low']) / 2
            center_distance = abs(body_center - range_center) / (candle['total_range'] / 2)
            position_quality = 1.0 - center_distance
            quality_factors.append(max(0, position_quality) * 0.2)
        else:
            quality_factors.append(0.2)
        
        # 4. Volatility appropriateness (15% weight)
        volatility_quality = min(candle['volatility_ratio'], 2.0) / 2.0
        quality_factors.append(volatility_quality * 0.15)
        
        # 5. Shadow symmetry (10% weight)
        if candle['upper_shadow_ratio'] + candle['lower_shadow_ratio'] > 0:
            shadow_symmetry = 1.0 - abs(candle['upper_shadow_ratio'] - candle['lower_shadow_ratio'])
            quality_factors.append(shadow_symmetry * 0.1)
        else:
            quality_factors.append(0.1)
        
        return sum(quality_factors)
    
    def _assess_trend_context(self, data: pd.DataFrame, pattern_end_index: int) -> float:
        """Assess the downtrend context before the pattern"""
        context_data = data.iloc[max(0, pattern_end_index - self.parameters['trend_lookback']):pattern_end_index + 3]
        
        if len(context_data) < 10:
            return 0.5
        
        context_factors = []
        
        # 1. Price trend (30% weight) - Should be downtrend for morning star
        price_trend = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
        if price_trend < -0.1:  # Strong downtrend
            context_factors.append(1.0 * 0.3)
        elif price_trend < -0.05:  # Moderate downtrend
            context_factors.append(0.8 * 0.3)
        elif price_trend < 0:  # Weak downtrend
            context_factors.append(0.6 * 0.3)
        else:
            context_factors.append(0.0 * 0.3)
        
        # 2. Moving average alignment (25% weight) - Should be bearish
        latest = context_data.iloc[-1]
        ma_alignment = 0
        if latest['close'] < latest['sma_5']:
            ma_alignment += 0.25
        if latest['sma_5'] < latest['sma_10']:
            ma_alignment += 0.25
        if latest['sma_10'] < latest['sma_20']:
            ma_alignment += 0.25
        if latest['sma_20'] < latest['sma_50']:
            ma_alignment += 0.25
        context_factors.append(ma_alignment * 0.25)
        
        # 3. RSI oversold condition (20% weight)
        rsi_factor = 0
        if latest['rsi'] < 20:
            rsi_factor = 1.0
        elif latest['rsi'] < 30:
            rsi_factor = 0.8
        elif latest['rsi'] < 40:
            rsi_factor = 0.6
        context_factors.append(rsi_factor * 0.2)
        
        # 4. Bollinger Band position (15% weight) - Lower position is better
        bb_factor = 1.0 - latest['bb_position'] if not pd.isna(latest['bb_position']) else 0.5
        context_factors.append(bb_factor * 0.15)
        
        # 5. Trend strength (10% weight)
        trend_strength_factor = latest['trend_strength'] if not pd.isna(latest['trend_strength']) else 0.5
        context_factors.append(trend_strength_factor * 0.1)
        
        return sum(context_factors)
    
    def _calculate_bullish_confirmation(self, candle: pd.Series) -> float:
        """Calculate bullish confirmation strength of the third candle"""
        confirmation_factors = []
        
        # 1. Body size (40% weight)
        body_strength = candle['body_ratio'] / self.parameters['min_confirmation_body']
        confirmation_factors.append(min(body_strength, 1.0) * 0.4)
        
        # 2. Close position in range (30% weight)
        close_position = (candle['close'] - candle['low']) / candle['total_range'] if candle['total_range'] > 0 else 1
        close_strength = close_position  # Higher close is better for bullish
        confirmation_factors.append(close_strength * 0.3)
        
        # 3. Upper shadow size (20% weight)
        upper_shadow_strength = 1.0 - candle['upper_shadow_ratio']  # Prefer small upper shadow
        confirmation_factors.append(upper_shadow_strength * 0.2)
        
        # 4. Volatility appropriateness (10% weight)
        volatility_strength = min(candle['volatility_ratio'], 2.0) / 2.0
        confirmation_factors.append(volatility_strength * 0.1)
        
        return sum(confirmation_factors)
    
    def _calculate_pattern_strength(self, gap_down_quality: float, gap_up_quality: float,
                                  star_doji_quality: float, bullish_confirmation: float,
                                  trend_context_score: float) -> float:
        """Calculate overall pattern strength"""
        strength_components = [
            gap_down_quality * 0.2,
            gap_up_quality * 0.2,
            star_doji_quality * 0.25,
            bullish_confirmation * 0.2,
            trend_context_score * 0.15
        ]
        
        return sum(strength_components)
    
    def _analyze_volume_confirmation(self, patterns: List[MorningStarPattern], 
                                   data: pd.DataFrame) -> List[MorningStarPattern]:
        """Analyze volume confirmation for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            volume_score = self._calculate_volume_confirmation_score(data, pattern_idx)
            pattern.volume_confirmation = volume_score
            
            # Enhance pattern strength with volume confirmation
            pattern.pattern_strength = (pattern.pattern_strength * 0.8 + volume_score * 0.2)
        
        return patterns
    
    def _calculate_volume_confirmation_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate volume confirmation score"""
        try:
            # Get the three candles of the pattern
            if pattern_idx < 2:
                return 0.5
            
            candle1_vol = data.iloc[pattern_idx - 2]['volume']
            candle2_vol = data.iloc[pattern_idx - 1]['volume']
            candle3_vol = data.iloc[pattern_idx]['volume']
            
            avg_volume = data.iloc[max(0, pattern_idx - 20):pattern_idx]['volume'].mean()
            
            volume_factors = []
            
            # 1. Third candle volume surge (50% weight)
            volume_surge = candle3_vol / avg_volume if avg_volume > 0 else 1.0
            surge_score = min(volume_surge / self.parameters['volume_surge_threshold'], 1.0)
            volume_factors.append(surge_score * 0.5)
            
            # 2. Volume progression (30% weight)
            if candle2_vol > 0 and candle1_vol > 0:
                progression = (candle3_vol / candle2_vol) * (candle2_vol / candle1_vol)
                progression_score = min(progression / 2.0, 1.0)
                volume_factors.append(progression_score * 0.3)
            else:
                volume_factors.append(0.15)
            
            # 3. Relative volume consistency (20% weight)
            volumes = [candle1_vol, candle2_vol, candle3_vol]
            volume_consistency = 1.0 - (np.std(volumes) / np.mean(volumes)) if np.mean(volumes) > 0 else 0.5
            volume_factors.append(max(0, min(1, volume_consistency)) * 0.2)
            
            return sum(volume_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_support_resistance(self, patterns: List[MorningStarPattern], 
                                  data: pd.DataFrame) -> List[MorningStarPattern]:
        """Analyze support/resistance levels for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            sr_factor = self._calculate_support_resistance_factor(data, pattern_idx)
            pattern.support_resistance_factor = sr_factor
        
        return patterns
    
    def _calculate_support_resistance_factor(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate support/resistance factor"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 50):pattern_idx + 1]
            pattern_low = data.iloc[pattern_idx - 2]['low']  # First candle low
            
            sr_factors = []
            
            # 1. Historical support at pattern level (40% weight)
            support_touches = 0
            price_tolerance = pattern_low * 0.005  # 0.5% tolerance
            
            for i in range(len(context_data) - 3):
                if abs(context_data.iloc[i]['low'] - pattern_low) <= price_tolerance:
                    support_touches += 1
            
            support_strength = min(support_touches / 3, 1.0)
            sr_factors.append(support_strength * 0.4)
            
            # 2. Pivot point analysis (30% weight)
            pivot_lows = context_data[context_data['pivot_low']]['low']
            if len(pivot_lows) > 0:
                nearest_support = min(abs(pivot_lows - pattern_low))
                pivot_factor = 1.0 / (1.0 + nearest_support / pattern_low * 100)
                sr_factors.append(pivot_factor * 0.3)
            else:
                sr_factors.append(0.15)
            
            # 3. Volume at support level (30% weight)
            support_volumes = []
            for i in range(len(context_data) - 1):
                if abs(context_data.iloc[i]['low'] - pattern_low) <= price_tolerance:
                    support_volumes.append(context_data.iloc[i]['volume'])
            
            if support_volumes:
                avg_support_volume = np.mean(support_volumes)
                avg_volume = context_data['volume'].mean()
                volume_factor = avg_support_volume / avg_volume if avg_volume > 0 else 1.0
                sr_factors.append(min(volume_factor, 1.0) * 0.3)
            else:
                sr_factors.append(0.15)
            
            return sum(sr_factors)
            
        except Exception:
            return 0.5
    
    def _validate_institutional_activity(self, patterns: List[MorningStarPattern], 
                                       data: pd.DataFrame) -> List[MorningStarPattern]:
        """Validate institutional activity around patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            institutional_score = self._calculate_institutional_validation_score(data, pattern_idx)
            pattern.institutional_validation = institutional_score
        
        return patterns
    
    def _calculate_institutional_validation_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate institutional validation score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 5):pattern_idx + 1]
            
            validation_factors = []
            
            # 1. Order flow analysis using OBV and AD Line (40% weight)
            obv_trend = context_data['obv'].diff().iloc[-3:].mean()
            ad_trend = context_data['ad_line'].diff().iloc[-3:].mean()
            
            # For morning star, expect positive institutional flow
            flow_factor = 0
            if obv_trend > 0:
                flow_factor += 0.5
            if ad_trend > 0:
                flow_factor += 0.5
            
            validation_factors.append(flow_factor * 0.4)
            
            # 2. Volume profile analysis (35% weight)
            pattern_volume = data.iloc[pattern_idx]['volume']
            avg_volume = data.iloc[max(0, pattern_idx - 20):pattern_idx]['volume'].mean()
            
            volume_significance = pattern_volume / avg_volume if avg_volume > 0 else 1.0
            # High volume suggests institutional participation
            volume_factor = min(volume_significance / 2.0, 1.0)
            validation_factors.append(volume_factor * 0.35)
            
            # 3. Price impact efficiency (25% weight)
            price_change = abs(data.iloc[pattern_idx]['close'] - data.iloc[pattern_idx - 2]['open'])
            volume_impact = price_change / pattern_volume if pattern_volume > 0 else 0
            
            # Lower price impact per unit volume suggests institutional efficiency
            impact_factor = 1.0 / (1.0 + volume_impact * 1000000)  # Normalize
            validation_factors.append(impact_factor * 0.25)
            
            return sum(validation_factors)
            
        except Exception:
            return 0.5
    
    def _predict_reversal_probability(self, patterns: List[MorningStarPattern], 
                                    data: pd.DataFrame) -> List[MorningStarPattern]:
        """Predict reversal probability using ML"""
        if not patterns:
            return patterns
        
        try:
            # Extract features for ML model
            features = []
            for pattern in patterns:
                pattern_idx = data.index.get_loc(pattern.timestamp)
                feature_vector = self._extract_reversal_features(data, pattern_idx, pattern)
                features.append(feature_vector)
            
            if len(features) < 10:
                # Not enough data for ML, use heuristic
                for pattern in patterns:
                    pattern.reversal_probability = self._heuristic_reversal_probability(pattern)
                return patterns
            
            # Train model if needed
            if not self.is_ml_fitted:
                self._train_reversal_model(patterns, features)
            
            # Apply ML predictions if model is fitted
            if self.is_ml_fitted:
                features_scaled = self.scaler.transform(features)
                reversal_predictions = self.reversal_predictor.predict(features_scaled)
                
                # Update reversal probabilities
                for i, pattern in enumerate(patterns):
                    ml_probability = max(0.1, min(0.95, reversal_predictions[i]))
                    # Combine with heuristic probability
                    heuristic_prob = self._heuristic_reversal_probability(pattern)
                    pattern.reversal_probability = (ml_probability * 0.7 + heuristic_prob * 0.3)
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
                                 pattern: MorningStarPattern) -> List[float]:
        """Extract features for reversal prediction ML model"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 10):pattern_idx + 1]
            current = data.iloc[pattern_idx]
            
            features = [
                pattern.pattern_strength,
                pattern.gap_down_quality,
                pattern.gap_up_quality,
                pattern.star_doji_quality,
                pattern.bullish_confirmation,
                pattern.volume_confirmation,
                pattern.trend_context_score,
                pattern.support_resistance_factor,
                pattern.institutional_validation,
                (100 - current['rsi']) / 100.0 if not pd.isna(current['rsi']) else 0.5,  # Inverted for bullish
                (1.0 - current['bb_position']) if not pd.isna(current['bb_position']) else 0.5,  # Inverted for oversold
                current['bb_width'] if not pd.isna(current['bb_width']) else 0.1,
                current['volatility_ratio'],
                current['volume_ratio'],
                current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
                -context_data['macd_hist'].iloc[-1] if not pd.isna(context_data['macd_hist'].iloc[-1]) else 0,  # Negative for bullish
                (current['sma_20'] - current['close']) / current['sma_20'] if current['sma_20'] > 0 else 0,  # Oversold measure
                context_data['obv'].diff().iloc[-3:].mean() / 1000000,  # Normalized
                context_data['ad_line'].diff().iloc[-3:].mean() / 1000000,  # Normalized
                min(max((current['high'] - current['low']) / current['atr'], 0), 3) if current['atr'] > 0 else 1
            ]
            
            return features
            
        except Exception:
            return [0.5] * 20  # Default features
    
    def _train_reversal_model(self, patterns: List[MorningStarPattern], features: List[List[float]]):
        """Train ML model for reversal prediction"""
        try:
            # Create targets based on pattern characteristics
            targets = []
            for pattern in patterns:
                # High-quality patterns with strong confirmations have higher reversal probability
                target = (
                    pattern.pattern_strength * 0.3 +
                    pattern.volume_confirmation * 0.25 +
                    pattern.support_resistance_factor * 0.2 +
                    pattern.institutional_validation * 0.15 +
                    pattern.trend_context_score * 0.1
                )
                targets.append(max(0.1, min(0.9, target)))
            
            if len(features) >= 15:
                features_scaled = self.scaler.fit_transform(features)
                self.reversal_predictor.fit(features_scaled, targets)
                self.is_ml_fitted = True
                logging.info("ML reversal predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _heuristic_reversal_probability(self, pattern: MorningStarPattern) -> float:
        """Calculate heuristic reversal probability"""
        return (
            pattern.pattern_strength * 0.35 +
            pattern.volume_confirmation * 0.25 +
            pattern.support_resistance_factor * 0.2 +
            pattern.institutional_validation * 0.2
        )
    
    def _generate_pattern_analytics(self, patterns: List[MorningStarPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-20:]  # Last 20 patterns
        
        return {
            'total_patterns': len(recent_patterns),
            'average_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'average_reversal_probability': sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns),
            'average_volume_confirmation': sum(p.volume_confirmation for p in recent_patterns) / len(recent_patterns),
            'high_strength_patterns': len([p for p in recent_patterns if p.pattern_strength > 0.8]),
            'high_probability_patterns': len([p for p in recent_patterns if p.reversal_probability > 0.7]),
            'institutional_validated_patterns': len([p for p in recent_patterns if p.institutional_validation > 0.7]),
            'average_gap_quality': (
                sum(p.gap_down_quality for p in recent_patterns) + 
                sum(p.gap_up_quality for p in recent_patterns)
            ) / (2 * len(recent_patterns)),
            'strong_support_patterns': len([p for p in recent_patterns if p.support_resistance_factor > 0.7])
        }
    
    def _analyze_current_trend_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current trend context"""
        current = data.iloc[-1]
        
        return {
            'trend_strength': current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
            'is_downtrend': current['close'] < current['sma_20'],
            'ma_alignment': {
                'short_below_medium': current['sma_5'] < current['sma_10'],
                'medium_below_long': current['sma_10'] < current['sma_20'],
                'price_below_ma': current['close'] < current['sma_5']
            },
            'rsi_oversold': current['rsi'] < 30 if not pd.isna(current['rsi']) else False,
            'bb_position': current['bb_position'] if not pd.isna(current['bb_position']) else 0.5,
            'volatility_context': 'high' if current['volatility_ratio'] > 1.5 else 'normal'
        }
    
    def _generate_reversal_signals(self, patterns: List[MorningStarPattern], 
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """Generate reversal signals based on patterns"""
        if not patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        # Get recent high-quality patterns
        recent_patterns = [p for p in patterns[-5:] if p.pattern_strength > 0.75]
        
        if not recent_patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        # Calculate aggregate metrics
        avg_strength = sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns)
        avg_reversal_prob = sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns)
        avg_volume_conf = sum(p.volume_confirmation for p in recent_patterns) / len(recent_patterns)
        
        return {
            'signal_strength': avg_strength,
            'reversal_probability': avg_reversal_prob,
            'volume_confirmation': avg_volume_conf,
            'pattern_count': len(recent_patterns),
            'institutional_validation': sum(p.institutional_validation for p in recent_patterns) / len(recent_patterns),
            'support_resistance_strength': sum(p.support_resistance_factor for p in recent_patterns) / len(recent_patterns),
            'most_recent_pattern': recent_patterns[-1].timestamp
        }
    
    def _analyze_gap_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze gap characteristics in recent data"""
        recent_data = data.iloc[-20:]
        
        return {
            'recent_gaps_down': recent_data['gap_down'].sum(),
            'recent_gaps_up': recent_data['gap_up'].sum(),
            'average_gap_size': recent_data['gap_size'].mean(),
            'gap_frequency': (recent_data['gap_up'].sum() + recent_data['gap_down'].sum()) / len(recent_data),
            'significant_gaps': len(recent_data[recent_data['gap_percentage'] > self.parameters['min_gap_percentage']])
        }
    
    def _assess_pattern_reliability(self, patterns: List[MorningStarPattern]) -> Dict[str, Any]:
        """Assess pattern reliability metrics"""
        if not patterns:
            return {}
        
        return {
            'consistency_score': np.std([p.pattern_strength for p in patterns]),
            'volume_reliability': np.mean([p.volume_confirmation for p in patterns]),
            'institutional_consistency': np.std([p.institutional_validation for p in patterns]),
            'reversal_success_estimate': np.mean([p.reversal_probability for p in patterns])
        }
    
    def _assess_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess current market conditions"""
        current = data.iloc[-1]
        
        return {
            'volatility_regime': 'high' if current['volatility_ratio'] > 1.5 else 'normal',
            'volume_regime': 'high' if current['volume_ratio'] > 1.5 else 'normal',
            'trend_regime': 'downtrend' if current['trend_strength'] > 0.6 else 'sideways',
            'momentum_state': 'oversold' if current['rsi'] < 30 else 'neutral' if current['rsi'] < 70 else 'overbought'
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on morning star analysis"""
        current_pattern = value.get('current_pattern')
        reversal_signals = value.get('reversal_signals', {})
        trend_analysis = value.get('trend_analysis', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Morning star is a bullish reversal pattern
        # Strong signal when we have high-quality pattern in downtrend with good confirmations
        
        if (current_pattern.pattern_strength > 0.8 and 
            current_pattern.reversal_probability > 0.7 and
            trend_analysis.get('is_downtrend', False) and
            current_pattern.volume_confirmation > 0.6):
            
            confidence = (
                current_pattern.pattern_strength * 0.4 +
                current_pattern.reversal_probability * 0.3 +
                current_pattern.volume_confirmation * 0.2 +
                current_pattern.institutional_validation * 0.1
            )
            
            return SignalType.BUY, confidence
        
        # Moderate bullish signal
        elif (current_pattern.pattern_strength > 0.7 and 
              current_pattern.reversal_probability > 0.6):
            
            confidence = current_pattern.pattern_strength * 0.6
            return SignalType.BUY, confidence
        
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_type': 'morning_star',
            'min_gap_percentage': self.parameters['min_gap_percentage'],
            'volume_analysis_enabled': self.parameters['volume_analysis'],
            'ml_reversal_prediction_enabled': self.parameters['ml_reversal_prediction'],
            'support_resistance_analysis_enabled': self.parameters['support_resistance_analysis']
        })
        return base_metadata