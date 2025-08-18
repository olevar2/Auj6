"""
Three White Soldiers Indicator - Advanced Three-Candle Bullish Reversal Pattern Detection
========================================================================================

This indicator implements sophisticated three white soldiers pattern detection, identifying
powerful bullish reversal signals through consecutive advancing bullish candles.
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
class ThreeWhiteSoldiersPattern:
    """Represents a detected three white soldiers pattern"""
    timestamp: pd.Timestamp
    pattern_strength: float
    candle_progression: float
    body_consistency: float
    shadow_analysis: float
    volume_progression: float
    trend_context_score: float
    momentum_acceleration: float
    breakout_potential: float
    gap_analysis: float
    price_ascent_rate: float
    volume_confirmation: float
    pattern_purity: float
    strength_signals: float


class ThreeWhiteSoldiersIndicator(StandardIndicatorInterface):
    """
    Advanced Three White Soldiers Pattern Indicator
    
    Features:
    - Precise three-candle bullish sequence detection
    - Advanced body consistency and progression analysis
    - Comprehensive volume trend and acceleration tracking
    - ML-enhanced breakout potential prediction
    - Gap analysis and price ascent rate calculation
    - Shadow analysis for pattern purity validation
    - Momentum acceleration and strength detection
    - Trend context validation for pattern significance
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'min_body_ratio': 0.45,              # Minimum body to range ratio for each candle
            'max_lower_shadow_ratio': 0.25,      # Maximum lower shadow for clean bullish candles
            'min_advance_progression': 0.6,      # Minimum price advance consistency
            'max_gap_tolerance': 0.008,          # Maximum gap between consecutive candles
            'volume_trend_weight': 0.3,          # Weight for volume trend analysis
            'trend_context_periods': 20,         # Periods for trend context analysis
            'momentum_lookback': 14,             # Periods for momentum analysis
            'pattern_validation': True,
            'volume_analysis': True,
            'ml_prediction': True,
            'gap_analysis': True,
            'strength_detection': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="ThreeWhiteSoldiersIndicator", parameters=default_params)
        
        # Initialize ML components
        self.breakout_predictor = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=130, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=90, random_state=42)),
            ('ada', AdaBoostRegressor(n_estimators=100, random_state=42))
        ])
        self.scaler = RobustScaler()
        self.is_ml_fitted = False
        
        logging.info(f"ThreeWhiteSoldiersIndicator initialized with parameters: {self.parameters}")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=60,
            lookback_periods=120
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate three white soldiers patterns with advanced analysis"""
        try:
            if len(data) < 60:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < 60"
                )
            
            # Enhance data with technical indicators
            enhanced_data = self._enhance_data_with_indicators(data)
            
            # Detect three white soldiers patterns
            detected_patterns = self._detect_three_white_soldiers_patterns(enhanced_data)
            
            # Apply comprehensive analysis pipeline
            if self.parameters['volume_analysis']:
                detected_patterns = self._analyze_volume_progression(detected_patterns, enhanced_data)
            
            if self.parameters['gap_analysis']:
                detected_patterns = self._analyze_gap_characteristics(detected_patterns, enhanced_data)
            
            if self.parameters['strength_detection']:
                detected_patterns = self._analyze_strength_signals(detected_patterns, enhanced_data)
            
            if self.parameters['ml_prediction'] and detected_patterns:
                detected_patterns = self._predict_breakout_potential(detected_patterns, enhanced_data)
            
            # Generate comprehensive analysis
            pattern_analytics = self._generate_pattern_analytics(detected_patterns)
            trend_analysis = self._analyze_current_trend_context(enhanced_data)
            breakout_signals = self._generate_breakout_signals(detected_patterns, enhanced_data)
            
            return {
                'current_pattern': detected_patterns[-1] if detected_patterns else None,
                'recent_patterns': detected_patterns[-6:],
                'pattern_analytics': pattern_analytics,
                'trend_analysis': trend_analysis,
                'breakout_signals': breakout_signals,
                'market_structure': self._analyze_market_structure(enhanced_data),
                'pattern_reliability': self._assess_pattern_reliability(detected_patterns),
                'ascent_statistics': self._calculate_ascent_statistics(detected_patterns)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Three white soldiers calculation failed: {str(e)}", e
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
        
        # Price change metrics
        df['open_to_open_change'] = df['open'].pct_change()
        df['close_to_close_change'] = df['close'].pct_change()
        df['high_to_high_change'] = df['high'].pct_change()
        df['low_to_low_change'] = df['low'].pct_change()
        
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
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
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
        df['price_velocity'] = df['close'].diff() / df['close'].shift(1)
        
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
    
    def _detect_three_white_soldiers_patterns(self, data: pd.DataFrame) -> List[ThreeWhiteSoldiersPattern]:
        """Detect three white soldiers patterns with sophisticated analysis"""
        patterns = []
        
        for i in range(self.parameters['trend_context_periods'], len(data) - 2):
            candle1 = data.iloc[i]
            candle2 = data.iloc[i + 1]
            candle3 = data.iloc[i + 2]
            
            # Check basic three white soldiers criteria
            if self._is_three_white_soldiers_sequence(candle1, candle2, candle3):
                pattern = self._create_three_white_soldiers_pattern(
                    data, i + 2, candle1, candle2, candle3
                )
                if pattern and pattern.pattern_strength >= 0.65:
                    patterns.append(pattern)
        
        return patterns
    
    def _is_three_white_soldiers_sequence(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
        """Check if three candles form a three white soldiers pattern"""
        # All candles must be bullish
        if not (c1['is_bullish'] and c2['is_bullish'] and c3['is_bullish']):
            return False
        
        # Each candle must have sufficient body
        if not (c1['body_ratio'] >= self.parameters['min_body_ratio'] and
                c2['body_ratio'] >= self.parameters['min_body_ratio'] and
                c3['body_ratio'] >= self.parameters['min_body_ratio']):
            return False
        
        # Limited lower shadows for clean bullish candles
        if not (c1['lower_shadow_ratio'] <= self.parameters['max_lower_shadow_ratio'] and
                c2['lower_shadow_ratio'] <= self.parameters['max_lower_shadow_ratio'] and
                c3['lower_shadow_ratio'] <= self.parameters['max_lower_shadow_ratio']):
            return False
        
        # Progressive advance in closing prices
        if not (c1['close'] < c2['close'] < c3['close']):
            return False
        
        # Check for appropriate gaps (not too large)
        gap1 = abs(c2['open'] - c1['close']) / c1['close']
        gap2 = abs(c3['open'] - c2['close']) / c2['close']
        
        if gap1 > self.parameters['max_gap_tolerance'] or gap2 > self.parameters['max_gap_tolerance']:
            return False
        
        return True
    
    def _create_three_white_soldiers_pattern(self, data: pd.DataFrame, candle_idx: int,
                                           c1: pd.Series, c2: pd.Series, c3: pd.Series) -> Optional[ThreeWhiteSoldiersPattern]:
        """Create three white soldiers pattern with comprehensive analysis"""
        try:
            # Calculate pattern metrics
            candle_progression = self._analyze_candle_progression(c1, c2, c3)
            body_consistency = self._analyze_body_consistency(c1, c2, c3)
            shadow_analysis = self._analyze_shadow_characteristics(c1, c2, c3)
            gap_analysis = self._analyze_gap_sequence(c1, c2, c3)
            price_ascent_rate = self._calculate_price_ascent_rate(c1, c2, c3)
            
            # Trend context analysis
            trend_context_score = self._assess_three_soldiers_trend_context(data, candle_idx)
            
            # Momentum analysis
            momentum_acceleration = self._analyze_momentum_acceleration(data, candle_idx)
            
            # Calculate pattern strength
            pattern_strength = self._calculate_three_soldiers_strength(
                candle_progression, body_consistency, shadow_analysis,
                trend_context_score, momentum_acceleration, gap_analysis
            )
            
            pattern = ThreeWhiteSoldiersPattern(
                timestamp=c3.name,
                pattern_strength=pattern_strength,
                candle_progression=candle_progression,
                body_consistency=body_consistency,
                shadow_analysis=shadow_analysis,
                volume_progression=0.0,
                trend_context_score=trend_context_score,
                momentum_acceleration=momentum_acceleration,
                breakout_potential=0.0,
                gap_analysis=gap_analysis,
                price_ascent_rate=price_ascent_rate,
                volume_confirmation=0.0,
                pattern_purity=0.0,
                strength_signals=0.0
            )
            
            return pattern
            
        except Exception:
            return None
    
    def _analyze_candle_progression(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> float:
        """Analyze the progression quality of the three candles"""
        progression_factors = []
        
        # Opening price progression (ideally opening within previous body)
        c2_open_in_c1_body = min(c1['open'], c1['close']) <= c2['open'] <= max(c1['open'], c1['close'])
        c3_open_in_c2_body = min(c2['open'], c2['close']) <= c3['open'] <= max(c2['open'], c2['close'])
        
        if c2_open_in_c1_body:
            progression_factors.append(0.25)
        elif c2['open'] > c1['close']:  # Gap up is acceptable
            progression_factors.append(0.15)
        
        if c3_open_in_c2_body:
            progression_factors.append(0.25)
        elif c3['open'] > c2['close']:  # Gap up is acceptable
            progression_factors.append(0.15)
        
        # Rising close sequence strength
        advance1 = (c2['close'] - c1['close']) / c1['close']
        advance2 = (c3['close'] - c2['close']) / c2['close']
        
        avg_advance = (advance1 + advance2) / 2
        advance_score = min(avg_advance / 0.03, 1.0)  # 3% average advance = max score
        progression_factors.append(advance_score * 0.3)
        
        # Consistent advance pattern
        if advance1 > 0 and advance2 > 0:
            consistency = 1.0 - abs(advance1 - advance2) / max(advance1, advance2)
            progression_factors.append(consistency * 0.2)
        
        return sum(progression_factors)
    
    def _analyze_body_consistency(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> float:
        """Analyze body size consistency across the three candles"""
        body_ratios = [c1['body_ratio'], c2['body_ratio'], c3['body_ratio']]
        
        # All bodies should be substantial
        min_body_score = min(body_ratios) / self.parameters['min_body_ratio']
        
        # Consistency in body sizes (standard deviation should be low)
        body_consistency = 1.0 - (np.std(body_ratios) / np.mean(body_ratios))
        
        # Ideal case: slightly increasing body sizes showing acceleration
        if c2['body_ratio'] >= c1['body_ratio'] and c3['body_ratio'] >= c2['body_ratio']:
            acceleration_bonus = 0.2
        elif c2['body_ratio'] >= c1['body_ratio'] or c3['body_ratio'] >= c2['body_ratio']:
            acceleration_bonus = 0.1
        else:
            acceleration_bonus = 0.0
        
        return min_body_score * 0.5 + body_consistency * 0.3 + acceleration_bonus
    
    def _analyze_shadow_characteristics(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> float:
        """Analyze shadow characteristics for pattern purity"""
        shadow_factors = []
        
        # Lower shadows should be minimal (clean bullish candles)
        lower_shadow_scores = []
        for candle in [c1, c2, c3]:
            shadow_score = max(0, 1.0 - candle['lower_shadow_ratio'] / self.parameters['max_lower_shadow_ratio'])
            lower_shadow_scores.append(shadow_score)
        
        avg_lower_shadow_score = np.mean(lower_shadow_scores)
        shadow_factors.append(avg_lower_shadow_score * 0.4)
        
        # Upper shadows analysis
        upper_shadow_ratios = [c1['upper_shadow_ratio'], c2['upper_shadow_ratio'], c3['upper_shadow_ratio']]
        
        # Moderate upper shadows are acceptable (showing some resistance testing)
        upper_shadow_scores = []
        for ratio in upper_shadow_ratios:
            if ratio <= 0.15:  # Ideal range
                upper_shadow_scores.append(1.0)
            elif ratio <= 0.3:  # Acceptable range
                upper_shadow_scores.append(0.7)
            else:
                upper_shadow_scores.append(max(0, 1.0 - (ratio - 0.3) / 0.2))
        
        avg_upper_shadow_score = np.mean(upper_shadow_scores)
        shadow_factors.append(avg_upper_shadow_score * 0.35)
        
        # Shadow progression (declining upper shadows show increasing strength)
        if c3['upper_shadow_ratio'] <= c2['upper_shadow_ratio'] <= c1['upper_shadow_ratio']:
            shadow_factors.append(0.25)
        elif c3['upper_shadow_ratio'] <= c2['upper_shadow_ratio'] or c2['upper_shadow_ratio'] <= c1['upper_shadow_ratio']:
            shadow_factors.append(0.15)
        
        return sum(shadow_factors)
    
    def _analyze_gap_sequence(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> float:
        """Analyze gap characteristics in the sequence"""
        gap_factors = []
        
        # Calculate gaps
        gap1 = (c2['open'] - c1['close']) / c1['close']
        gap2 = (c3['open'] - c2['close']) / c2['close']
        
        # Small gaps up are positive for the pattern
        for gap in [gap1, gap2]:
            if 0 <= gap <= 0.005:  # Small gap up
                gap_factors.append(0.25)
            elif 0 <= gap <= 0.015:  # Moderate gap up
                gap_factors.append(0.15)
            elif gap < 0:  # Gap down (negative for pattern)
                gap_factors.append(max(0, 0.1 + gap / 0.01))  # Penalty for gap down
            else:  # Large gap up
                gap_factors.append(max(0, 0.1 - (gap - 0.015) / 0.01))
        
        # Gap consistency
        if abs(gap1 - gap2) < 0.005:
            gap_factors.append(0.2)  # Consistent gap behavior
        
        # Progressive gapping up
        if gap1 > 0 and gap2 > gap1:
            gap_factors.append(0.3)  # Accelerating strength
        
        return sum(gap_factors)
    
    def _calculate_price_ascent_rate(self, c1: pd.Series, c2: pd.Series, c3: pd.Series) -> float:
        """Calculate the rate of price ascent across the pattern"""
        total_advance = (c3['close'] - c1['close']) / c1['close']
        
        # Normalize ascent rate (5% total advance = 1.0 score)
        ascent_rate = min(total_advance / 0.05, 1.0) if total_advance > 0 else 0.0
        
        # Acceleration analysis
        advance1 = (c2['close'] - c1['close']) / c1['close']
        advance2 = (c3['close'] - c2['close']) / c2['close']
        
        if advance2 > advance1:  # Accelerating advance
            acceleration_bonus = min((advance2 - advance1) / 0.02, 0.3)
        else:
            acceleration_bonus = 0.0
        
        return ascent_rate + acceleration_bonus
    
    def _assess_three_soldiers_trend_context(self, data: pd.DataFrame, candle_idx: int) -> float:
        """Assess trend context for three white soldiers patterns"""
        context_data = data.iloc[max(0, candle_idx - self.parameters['trend_context_periods']):candle_idx]
        
        if len(context_data) < 10:
            return 0.5
        
        context_factors = []
        
        # Pattern should appear after downtrend for maximum impact
        price_change = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
        if price_change < -0.05:  # Downtrend context
            trend_factor = min(abs(price_change) / 0.2, 1.0)
            context_factors.append(trend_factor * 0.4)
        else:
            context_factors.append(0.2)  # Reduced effectiveness in uptrend
        
        # Low level context (near lows)
        recent_low = context_data['low'].min()
        current_close = context_data['close'].iloc[-1]
        low_proximity = current_close / recent_low if recent_low > 0 else 1.0
        
        if low_proximity < 1.05:
            context_factors.append(0.3)
        elif low_proximity < 1.1:
            context_factors.append(0.2)
        else:
            context_factors.append(0.1)
        
        # RSI context (should be low)
        latest = context_data.iloc[-1]
        if not pd.isna(latest['rsi']):
            if latest['rsi'] < 40:
                rsi_context = (40 - latest['rsi']) / 20
                context_factors.append(rsi_context * 0.2)
            else:
                context_factors.append(0.05)
        
        # Volume context
        if latest['volume_ratio'] > 1.2:
            context_factors.append(0.1)
        
        return sum(context_factors)
    
    def _analyze_momentum_acceleration(self, data: pd.DataFrame, candle_idx: int) -> float:
        """Analyze momentum acceleration during the pattern"""
        momentum_data = data.iloc[max(0, candle_idx - self.parameters['momentum_lookback']):candle_idx + 1]
        
        if len(momentum_data) < 6:
            return 0.5
        
        acceleration_factors = []
        
        # RSI rise acceleration
        rsi_values = momentum_data['rsi'].dropna()
        if len(rsi_values) >= 4:
            rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
            if rsi_trend > 0.5:  # Rising RSI
                acceleration_factors.append(min(rsi_trend / 2.0, 1.0) * 0.3)
            else:
                acceleration_factors.append(0.1)
        
        # MACD histogram analysis
        macd_hist = momentum_data['macd_hist'].dropna()
        if len(macd_hist) >= 4:
            if macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3]:
                acceleration_factors.append(0.25)  # Accelerating bullish momentum
            elif macd_hist.iloc[-1] > 0:
                acceleration_factors.append(0.15)
        
        # Price velocity analysis
        velocity_data = momentum_data['price_velocity'].dropna()
        if len(velocity_data) >= 3:
            recent_velocity = velocity_data.iloc[-3:].mean()
            if recent_velocity > 0.01:  # Strong positive velocity
                acceleration_factors.append(min(recent_velocity / 0.03, 1.0) * 0.25)
        
        # Stochastic analysis
        if not pd.isna(momentum_data.iloc[-1]['stoch_k']):
            stoch_k = momentum_data.iloc[-1]['stoch_k']
            if stoch_k > 70:  # Overbought momentum
                acceleration_factors.append((stoch_k - 70) / 30 * 0.2)
        
        return sum(acceleration_factors)
    
    def _calculate_three_soldiers_strength(self, candle_progression: float, body_consistency: float,
                                         shadow_analysis: float, trend_context_score: float,
                                         momentum_acceleration: float, gap_analysis: float) -> float:
        """Calculate overall three white soldiers pattern strength"""
        strength_components = [
            candle_progression * 0.25,         # Progression quality
            body_consistency * 0.2,            # Body consistency
            shadow_analysis * 0.15,            # Shadow analysis
            trend_context_score * 0.15,        # Trend context
            momentum_acceleration * 0.15,      # Momentum acceleration
            gap_analysis * 0.1                 # Gap analysis
        ]
        
        return sum(strength_components)
    
    def _analyze_volume_progression(self, patterns: List[ThreeWhiteSoldiersPattern], 
                                  data: pd.DataFrame) -> List[ThreeWhiteSoldiersPattern]:
        """Analyze volume progression for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            volume_scores = self._calculate_volume_progression_score(data, pattern_idx)
            pattern.volume_progression = volume_scores['progression']
            pattern.volume_confirmation = volume_scores['confirmation']
        
        return patterns
    
    def _calculate_volume_progression_score(self, data: pd.DataFrame, pattern_idx: int) -> Dict[str, float]:
        """Calculate volume progression and confirmation scores"""
        try:
            # Get the three candles of the pattern
            c1 = data.iloc[pattern_idx - 2]
            c2 = data.iloc[pattern_idx - 1]
            c3 = data.iloc[pattern_idx]
            
            # Average volume context
            context_data = data.iloc[max(0, pattern_idx - 20):pattern_idx - 2]
            avg_volume = context_data['volume'].mean()
            
            volume_scores = {}
            
            # Volume progression analysis
            progression_factors = []
            
            # Each candle should have above-average volume
            for candle in [c1, c2, c3]:
                volume_ratio = candle['volume'] / avg_volume if avg_volume > 0 else 1.0
                if volume_ratio > 1.3:
                    progression_factors.append(0.25)
                elif volume_ratio > 1.1:
                    progression_factors.append(0.15)
                else:
                    progression_factors.append(0.05)
            
            # Increasing volume trend across the pattern
            if c3['volume'] > c2['volume'] > c1['volume']:
                progression_factors.append(0.25)  # Perfect progression
            elif c3['volume'] > c2['volume'] or c2['volume'] > c1['volume']:
                progression_factors.append(0.15)  # Partial progression
            
            volume_scores['progression'] = sum(progression_factors)
            
            # Volume confirmation analysis
            confirmation_factors = []
            
            # High volume on the final candle (accumulation)
            final_volume_ratio = c3['volume'] / avg_volume if avg_volume > 0 else 1.0
            confirmation_factors.append(min(final_volume_ratio / 2.0, 1.0) * 0.4)
            
            # Volume trend in context
            recent_volumes = data.iloc[pattern_idx - 5:pattern_idx + 1]['volume']
            if len(recent_volumes) >= 4:
                volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
                if volume_trend > 0:  # Increasing volume trend
                    confirmation_factors.append(min(volume_trend / avg_volume, 1.0) * 0.3)
            
            # Volume relative to volatility
            pattern_volatility = np.std([c1['total_range'], c2['total_range'], c3['total_range']])
            pattern_volume = np.mean([c1['volume'], c2['volume'], c3['volume']])
            
            if pattern_volatility > 0:
                vol_volatility_ratio = pattern_volume / (avg_volume * (1 + pattern_volatility))
                confirmation_factors.append(min(vol_volatility_ratio, 1.0) * 0.3)
            
            volume_scores['confirmation'] = sum(confirmation_factors)
            
            return volume_scores
            
        except Exception:
            return {'progression': 0.5, 'confirmation': 0.5}
    
    def _analyze_gap_characteristics(self, patterns: List[ThreeWhiteSoldiersPattern], 
                                   data: pd.DataFrame) -> List[ThreeWhiteSoldiersPattern]:
        """Analyze gap characteristics for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            gap_score = self._calculate_detailed_gap_analysis(data, pattern_idx)
            pattern.gap_analysis = gap_score
        
        return patterns
    
    def _calculate_detailed_gap_analysis(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate detailed gap analysis score"""
        try:
            c1 = data.iloc[pattern_idx - 2]
            c2 = data.iloc[pattern_idx - 1]
            c3 = data.iloc[pattern_idx]
            
            gap_factors = []
            
            # Gap sizes
            gap1 = (c2['open'] - c1['close']) / c1['close'] if c1['close'] != 0 else 0
            gap2 = (c3['open'] - c2['close']) / c2['close'] if c2['close'] != 0 else 0
            
            # Optimal gap analysis
            for gap in [gap1, gap2]:
                if 0.002 <= gap <= 0.008:  # Ideal small gap up
                    gap_factors.append(0.3)
                elif 0 <= gap <= 0.002:  # Very small gap
                    gap_factors.append(0.2)
                elif 0.008 < gap <= 0.02:  # Moderate gap
                    gap_factors.append(0.15)
                elif gap < 0:  # Gap down (negative)
                    gap_factors.append(max(0, 0.1 + gap / 0.01))
                else:  # Large gap up
                    gap_factors.append(0.05)
            
            # Gap progression analysis
            if gap2 > gap1 > 0:  # Accelerating gaps
                gap_factors.append(0.25)
            elif gap1 > 0 and gap2 > 0:  # Consistent gaps
                gap_factors.append(0.15)
            
            # Context analysis
            avg_daily_change = data.iloc[pattern_idx - 20:pattern_idx]['close'].pct_change().abs().mean()
            
            for gap in [gap1, gap2]:
                if gap > 0 and gap < avg_daily_change * 2:  # Reasonable gap relative to volatility
                    gap_factors.append(0.1)
            
            return sum(gap_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_strength_signals(self, patterns: List[ThreeWhiteSoldiersPattern], 
                                data: pd.DataFrame) -> List[ThreeWhiteSoldiersPattern]:
        """Analyze strength signals for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            strength_score = self._calculate_strength_signals_score(data, pattern_idx)
            pattern.strength_signals = strength_score
            pattern.pattern_purity = self._calculate_pattern_purity_score(data, pattern_idx, pattern)
        
        return patterns
    
    def _calculate_strength_signals_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate strength signals score"""
        try:
            current = data.iloc[pattern_idx]
            context_data = data.iloc[max(0, pattern_idx - 10):pattern_idx + 1]
            
            strength_factors = []
            
            # RSI strength condition
            if not pd.isna(current['rsi']):
                if current['rsi'] > 60:
                    strength_factors.append(0.25)
                elif current['rsi'] > 50:
                    strength_factors.append(0.15)
                elif current['rsi'] < 30:  # Oversold bounce
                    strength_factors.append(0.3)
            
            # Williams %R strength
            if not pd.isna(current['williams_r']):
                if current['williams_r'] > -20:
                    strength_factors.append(0.25)
                elif current['williams_r'] > -30:
                    strength_factors.append(0.15)
            
            # Bollinger Band position
            if not pd.isna(current['bb_position']):
                if current['bb_position'] > 0.8:  # Near upper band
                    strength_factors.append(0.2)
                elif current['bb_position'] > 0.6:
                    strength_factors.append(0.15)
            
            # Volume strength (increasing volume during advance)
            volumes = context_data['volume'].iloc[-3:]
            if len(volumes) >= 3 and volumes.iloc[-1] > volumes.iloc[0]:
                strength_factors.append(0.2)
            
            # Momentum confirmation
            if len(context_data) >= 6:
                price_highs = context_data['high'].iloc[-6:]
                rsi_values = context_data['rsi'].iloc[-6:].dropna()
                
                if len(rsi_values) >= 4:
                    price_trend = np.polyfit(range(len(price_highs)), price_highs, 1)[0]
                    rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
                    
                    # Price making higher highs and RSI confirming
                    if price_trend > 0 and rsi_trend > 0:
                        strength_factors.append(0.25)
            
            # ADX trend strength
            if not pd.isna(current['adx']) and current['adx'] > 25:
                strength_factors.append(min(current['adx'] / 50, 1.0) * 0.15)
            
            return sum(strength_factors)
            
        except Exception:
            return 0.5
    
    def _calculate_pattern_purity_score(self, data: pd.DataFrame, pattern_idx: int, 
                                      pattern: ThreeWhiteSoldiersPattern) -> float:
        """Calculate pattern purity score"""
        try:
            purity_factors = []
            
            # Core pattern strength
            purity_factors.append(pattern.pattern_strength * 0.3)
            
            # Body consistency
            purity_factors.append(pattern.body_consistency * 0.25)
            
            # Shadow analysis
            purity_factors.append(pattern.shadow_analysis * 0.2)
            
            # Candle progression
            purity_factors.append(pattern.candle_progression * 0.25)
            
            return sum(purity_factors)
            
        except Exception:
            return 0.5
    
    def _predict_breakout_potential(self, patterns: List[ThreeWhiteSoldiersPattern], 
                                  data: pd.DataFrame) -> List[ThreeWhiteSoldiersPattern]:
        """Predict breakout potential using ML"""
        if not patterns:
            return patterns
        
        try:
            features = []
            for pattern in patterns:
                pattern_idx = data.index.get_loc(pattern.timestamp)
                feature_vector = self._extract_breakout_features(data, pattern_idx, pattern)
                features.append(feature_vector)
            
            if len(features) < 10:
                for pattern in patterns:
                    pattern.breakout_potential = self._heuristic_breakout_potential(pattern)
                return patterns
            
            if not self.is_ml_fitted:
                self._train_breakout_model(patterns, features)
            
            if self.is_ml_fitted:
                features_scaled = self.scaler.transform(features)
                breakout_predictions = self.breakout_predictor.predict(features_scaled)
                
                for i, pattern in enumerate(patterns):
                    ml_potential = max(0.1, min(0.95, breakout_predictions[i]))
                    heuristic_potential = self._heuristic_breakout_potential(pattern)
                    pattern.breakout_potential = (ml_potential * 0.7 + heuristic_potential * 0.3)
            else:
                for pattern in patterns:
                    pattern.breakout_potential = self._heuristic_breakout_potential(pattern)
            
            return patterns
            
        except Exception as e:
            logging.warning(f"ML breakout prediction failed: {str(e)}")
            for pattern in patterns:
                pattern.breakout_potential = self._heuristic_breakout_potential(pattern)
            return patterns
    
    def _extract_breakout_features(self, data: pd.DataFrame, pattern_idx: int, 
                                 pattern: ThreeWhiteSoldiersPattern) -> List[float]:
        """Extract features for breakout prediction ML model"""
        try:
            current = data.iloc[pattern_idx]
            
            features = [
                pattern.pattern_strength,
                pattern.candle_progression,
                pattern.body_consistency,
                pattern.shadow_analysis,
                pattern.volume_progression,
                pattern.trend_context_score,
                pattern.momentum_acceleration,
                pattern.gap_analysis,
                pattern.price_ascent_rate,
                pattern.volume_confirmation,
                pattern.strength_signals,
                current['rsi'] / 100.0 if not pd.isna(current['rsi']) else 0.5,
                current['bb_position'] if not pd.isna(current['bb_position']) else 0.5,
                current['adx'] / 50.0 if not pd.isna(current['adx']) else 0.5,
                current['volume_ratio'],
                current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
                current['cci'] / 200.0 if not pd.isna(current['cci']) else 0.0,
                current['williams_r'] / -100.0 if not pd.isna(current['williams_r']) else 0.5,
                current['body_ratio'],
                pattern.momentum_acceleration * pattern.strength_signals,  # Combined signal
                pattern.volume_progression * pattern.volume_confirmation,  # Volume factor
                current['momentum_roc'] / 10.0 if not pd.isna(current['momentum_roc']) else 0.0
            ]
            
            return features
            
        except Exception:
            return [0.5] * 22
    
    def _train_breakout_model(self, patterns: List[ThreeWhiteSoldiersPattern], features: List[List[float]]):
        """Train ML model for breakout prediction"""
        try:
            targets = []
            for pattern in patterns:
                # Target combines multiple factors for breakout potential
                target = (
                    pattern.pattern_strength * 0.3 +
                    pattern.strength_signals * 0.25 +
                    pattern.volume_confirmation * 0.2 +
                    pattern.trend_context_score * 0.15 +
                    pattern.momentum_acceleration * 0.1
                )
                targets.append(max(0.1, min(0.9, target)))
            
            if len(features) >= 15:
                features_scaled = self.scaler.fit_transform(features)
                self.breakout_predictor.fit(features_scaled, targets)
                self.is_ml_fitted = True
                logging.info("ML breakout predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _heuristic_breakout_potential(self, pattern: ThreeWhiteSoldiersPattern) -> float:
        """Calculate heuristic breakout potential"""
        return (
            pattern.pattern_strength * 0.3 +
            pattern.strength_signals * 0.25 +
            pattern.volume_confirmation * 0.2 +
            pattern.trend_context_score * 0.15 +
            pattern.momentum_acceleration * 0.1
        )
    
    def _generate_pattern_analytics(self, patterns: List[ThreeWhiteSoldiersPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-10:]
        
        return {
            'total_patterns': len(recent_patterns),
            'average_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'average_breakout_potential': sum(p.breakout_potential for p in recent_patterns) / len(recent_patterns),
            'average_ascent_rate': sum(p.price_ascent_rate for p in recent_patterns) / len(recent_patterns),
            'high_strength_patterns': len([p for p in recent_patterns if p.pattern_strength > 0.8]),
            'high_potential_patterns': len([p for p in recent_patterns if p.breakout_potential > 0.75]),
            'strong_strength_patterns': len([p for p in recent_patterns if p.strength_signals > 0.7]),
            'high_volume_patterns': len([p for p in recent_patterns if p.volume_progression > 0.75]),
            'pattern_purity_average': sum(p.pattern_purity for p in recent_patterns) / len(recent_patterns),
            'momentum_acceleration_average': sum(p.momentum_acceleration for p in recent_patterns) / len(recent_patterns)
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
            'volume_context': 'high' if current['volume_ratio'] > 1.5 else 'normal',
            'volatility_context': 'high' if current['volatility'] > current['atr'] else 'normal',
            'bullish_momentum': current['momentum_roc'] > 0.02 if not pd.isna(current['momentum_roc']) else False
        }
    
    def _generate_breakout_signals(self, patterns: List[ThreeWhiteSoldiersPattern], 
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """Generate breakout signals based on patterns"""
        if not patterns:
            return {'signal_strength': 0.0, 'breakout_probability': 0.0}
        
        recent_patterns = [p for p in patterns[-3:] if p.pattern_strength > 0.7]
        
        if not recent_patterns:
            return {'signal_strength': 0.0, 'breakout_probability': 0.0}
        
        latest_pattern = recent_patterns[-1]
        
        return {
            'signal_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'breakout_probability': sum(p.breakout_potential for p in recent_patterns) / len(recent_patterns),
            'strength_level': sum(p.strength_signals for p in recent_patterns) / len(recent_patterns),
            'volume_confirmation': sum(p.volume_confirmation for p in recent_patterns) / len(recent_patterns),
            'pattern_count': len(recent_patterns),
            'ascent_rate': sum(p.price_ascent_rate for p in recent_patterns) / len(recent_patterns),
            'momentum_acceleration': sum(p.momentum_acceleration for p in recent_patterns) / len(recent_patterns),
            'most_recent_pattern': latest_pattern.timestamp
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure"""
        recent_data = data.iloc[-20:]
        
        return {
            'bullish_candle_frequency': len([i for i in range(len(recent_data)) if recent_data.iloc[i]['is_bullish']]) / len(recent_data),
            'average_body_ratio': recent_data['body_ratio'].mean(),
            'volatility_trend': 'increasing' if recent_data['volatility'].iloc[-5:].mean() > recent_data['volatility'].iloc[-15:-5].mean() else 'stable',
            'volume_trend': 'increasing' if recent_data['volume_ratio'].iloc[-5:].mean() > 1.2 else 'normal'
        }
    
    def _assess_pattern_reliability(self, patterns: List[ThreeWhiteSoldiersPattern]) -> Dict[str, Any]:
        """Assess pattern reliability metrics"""
        if not patterns:
            return {}
        
        return {
            'consistency_score': 1.0 - np.std([p.pattern_strength for p in patterns]),
            'strength_reliability': np.mean([p.strength_signals for p in patterns]),
            'volume_reliability': np.mean([p.volume_confirmation for p in patterns]),
            'breakout_success_estimate': np.mean([p.breakout_potential for p in patterns])
        }
    
    def _calculate_ascent_statistics(self, patterns: List[ThreeWhiteSoldiersPattern]) -> Dict[str, Any]:
        """Calculate ascent-specific statistics"""
        if not patterns:
            return {}
        
        ascent_rates = [p.price_ascent_rate for p in patterns]
        momentum_accelerations = [p.momentum_acceleration for p in patterns]
        
        return {
            'ascent_rate_stats': {
                'mean': np.mean(ascent_rates),
                'median': np.median(ascent_rates),
                'max': np.max(ascent_rates),
                'std': np.std(ascent_rates)
            },
            'momentum_stats': {
                'mean': np.mean(momentum_accelerations),
                'median': np.median(momentum_accelerations),
                'std': np.std(momentum_accelerations)
            }
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on three white soldiers analysis"""
        current_pattern = value.get('current_pattern')
        breakout_signals = value.get('breakout_signals', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Strong bullish breakout signal
        if (current_pattern.pattern_strength > 0.8 and
            current_pattern.breakout_potential > 0.75 and
            current_pattern.strength_signals > 0.6):
            
            confidence = (
                current_pattern.pattern_strength * 0.35 +
                current_pattern.breakout_potential * 0.3 +
                current_pattern.strength_signals * 0.2 +
                current_pattern.volume_confirmation * 0.15
            )
            
            return SignalType.BUY, confidence
        
        # Moderate bullish signal
        elif (current_pattern.pattern_strength > 0.7 and
              current_pattern.breakout_potential > 0.65):
            
            confidence = current_pattern.pattern_strength * 0.75
            
            return SignalType.BUY, confidence
        
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_type': 'three_white_soldiers',
            'min_body_ratio': self.parameters['min_body_ratio'],
            'max_lower_shadow_ratio': self.parameters['max_lower_shadow_ratio'],
            'volume_analysis_enabled': self.parameters['volume_analysis'],
            'ml_prediction_enabled': self.parameters['ml_prediction']
        })
        return base_metadata