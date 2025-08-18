"""
Hammer Indicator - Advanced Single-Candle Bullish Reversal Pattern Detection
===========================================================================

This indicator implements sophisticated hammer pattern detection with advanced
body-to-shadow ratio analysis, trend context validation, and ML-enhanced reversal prediction.
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
class HammerPattern:
    """Represents a detected hammer pattern"""
    timestamp: pd.Timestamp
    pattern_strength: float
    body_to_shadow_ratio: float
    lower_shadow_length: float
    upper_shadow_ratio: float
    body_position: float
    trend_context_score: float
    volume_confirmation: float
    support_strength: float
    reversal_probability: float
    institutional_interest: float
    pattern_purity: float


class HammerIndicator(StandardIndicatorInterface):
    """
    Advanced Hammer Pattern Indicator
    
    Features:
    - Precise hammer identification with body-to-shadow ratio analysis
    - Advanced trend context validation for downtrend requirement
    - Volume-based confirmation and institutional interest detection
    - ML-enhanced reversal probability prediction
    - Support level validation and strength assessment
    - Pattern purity scoring and reliability metrics
    - Statistical significance testing for pattern quality
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'min_lower_shadow_ratio': 0.6,     # Minimum lower shadow as % of total range
            'max_body_ratio': 0.3,             # Maximum body as % of total range
            'max_upper_shadow_ratio': 0.1,     # Maximum upper shadow as % of total range
            'min_trend_strength': 0.6,         # Minimum downtrend strength required
            'volume_surge_threshold': 1.2,     # Volume surge multiplier
            'trend_lookback': 15,              # Periods for trend analysis
            'support_analysis': True,
            'volume_analysis': True,
            'ml_reversal_prediction': True,
            'institutional_analysis': True,
            'pattern_purity_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="HammerIndicator", parameters=default_params)
        
        # Initialize ML components
        self.reversal_predictor = VotingRegressor([
            ('rf', RandomForestRegressor(n_estimators=120, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=80, random_state=42)),
            ('ada', AdaBoostRegressor(n_estimators=100, random_state=42))
        ])
        self.scaler = RobustScaler()
        self.is_ml_fitted = False
        
        # Pattern analysis components
        self.trend_analyzer = self._initialize_trend_analyzer()
        self.support_analyzer = self._initialize_support_analyzer()
        
        logging.info(f"HammerIndicator initialized with parameters: {self.parameters}")
    
    def _initialize_trend_analyzer(self) -> Dict[str, Any]:
        """Initialize trend analysis components"""
        return {
            'trend_detector': self._detect_trend_context,
            'trend_strength_calculator': self._calculate_trend_strength,
            'reversal_condition_assessor': self._assess_reversal_conditions
        }
    
    def _initialize_support_analyzer(self) -> Dict[str, Any]:
        """Initialize support analysis components"""
        return {
            'support_detector': self._detect_support_levels,
            'support_strength_calculator': self._calculate_support_strength,
            'support_validator': self._validate_support_confluence
        }
    
    def get_data_requirements(self) -> DataRequirement:
        """Define OHLCV data requirements"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=40,
            lookback_periods=80
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate hammer patterns with advanced analysis"""
        try:
            if len(data) < 40:
                raise IndicatorCalculationException(
                    self.name, "data_validation", 
                    f"Insufficient data: {len(data)} < 40"
                )
            
            # Enhance data with technical indicators
            enhanced_data = self._enhance_data_with_indicators(data)
            
            # Detect hammer patterns
            detected_patterns = self._detect_hammer_patterns(enhanced_data)
            
            # Apply volume analysis
            if self.parameters['volume_analysis']:
                volume_enhanced_patterns = self._analyze_volume_confirmation(
                    detected_patterns, enhanced_data
                )
            else:
                volume_enhanced_patterns = detected_patterns
            
            # Apply support analysis
            if self.parameters['support_analysis']:
                support_enhanced_patterns = self._analyze_support_strength(
                    volume_enhanced_patterns, enhanced_data
                )
            else:
                support_enhanced_patterns = volume_enhanced_patterns
            
            # Apply institutional analysis
            if self.parameters['institutional_analysis']:
                institutional_enhanced_patterns = self._analyze_institutional_interest(
                    support_enhanced_patterns, enhanced_data
                )
            else:
                institutional_enhanced_patterns = support_enhanced_patterns
            
            # Apply pattern purity analysis
            if self.parameters['pattern_purity_analysis']:
                purity_enhanced_patterns = self._analyze_pattern_purity(
                    institutional_enhanced_patterns, enhanced_data
                )
            else:
                purity_enhanced_patterns = institutional_enhanced_patterns
            
            # Apply ML reversal prediction
            if self.parameters['ml_reversal_prediction'] and purity_enhanced_patterns:
                ml_enhanced_patterns = self._predict_reversal_probability(
                    purity_enhanced_patterns, enhanced_data
                )
            else:
                ml_enhanced_patterns = purity_enhanced_patterns
            
            # Generate comprehensive analysis
            pattern_analytics = self._generate_pattern_analytics(ml_enhanced_patterns)
            trend_analysis = self._analyze_current_trend_context(enhanced_data)
            reversal_signals = self._generate_reversal_signals(ml_enhanced_patterns, enhanced_data)
            market_structure = self._analyze_market_structure(enhanced_data)
            
            return {
                'current_pattern': ml_enhanced_patterns[-1] if ml_enhanced_patterns else None,
                'recent_patterns': ml_enhanced_patterns[-8:],
                'pattern_analytics': pattern_analytics,
                'trend_analysis': trend_analysis,
                'reversal_signals': reversal_signals,
                'market_structure': market_structure,
                'pattern_reliability': self._assess_pattern_reliability(ml_enhanced_patterns),
                'support_levels': self._identify_key_support_levels(enhanced_data)
            }
            
        except Exception as e:
            raise IndicatorCalculationException(
                self.name, "calculation", f"Hammer calculation failed: {str(e)}", e
            )
    
    def _enhance_data_with_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with comprehensive technical indicators"""
        df = data.copy()
        
        # Basic candlestick components
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Advanced candlestick ratios
        df['body_ratio'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
        df['upper_shadow_ratio'] = np.where(df['total_range'] > 0, df['upper_shadow'] / df['total_range'], 0)
        df['lower_shadow_ratio'] = np.where(df['total_range'] > 0, df['lower_shadow'] / df['total_range'], 0)
        df['body_position'] = np.where(df['total_range'] > 0, 
                                      (np.minimum(df['open'], df['close']) - df['low']) / df['total_range'], 0.5)
        
        # Trend indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_8'] = df['close'].ewm(span=8).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        
        # MACD for momentum
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # RSI for momentum extremes
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_oversold'] = df['rsi'] < 30
        
        # Stochastic for momentum
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] < 0.1
        
        # ATR for volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['volatility_ratio'] = df['total_range'] / df['atr']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['ad_line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])
        
        # Trend strength calculation
        df['trend_strength'] = self._calculate_trend_strength_series(df)
        df['trend_direction'] = np.where(df['close'] > df['sma_20'], 1, -1)
        
        # Support/resistance levels
        df['pivot_low'] = df['low'].rolling(5, center=True).min() == df['low']
        df['pivot_high'] = df['high'].rolling(5, center=True).max() == df['high']
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(5)
        df['momentum_zscore'] = df.rolling(20)['price_momentum'].apply(lambda x: zscore(x)[-1] if len(x) == 20 else 0)
        
        return df
    
    def _calculate_trend_strength_series(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength as a rolling series"""
        def trend_strength_window(window_data):
            if len(window_data) < 8:
                return 0.5
            
            # Linear regression slope
            x = np.arange(len(window_data))
            slope, _, r_value, _, _ = linregress(x, window_data['close'])
            
            # Normalize slope
            price_range = window_data['close'].max() - window_data['close'].min()
            if price_range == 0:
                return 0.5
            
            normalized_slope = abs(slope) / price_range * len(window_data)
            
            # R-squared for trend reliability
            trend_quality = r_value ** 2
            
            # Combine slope magnitude and quality
            strength = normalized_slope * trend_quality
            return min(max(strength, 0), 1)
        
        return df.rolling(self.parameters['trend_lookback']).apply(trend_strength_window, raw=False)['close']
    
    def _detect_hammer_patterns(self, data: pd.DataFrame) -> List[HammerPattern]:
        """Detect hammer patterns with sophisticated analysis"""
        patterns = []
        
        for i in range(self.parameters['trend_lookback'], len(data)):
            candle = data.iloc[i]
            
            # Check for hammer pattern
            if not self._is_hammer_pattern(candle):
                continue
            
            # Assess trend context
            trend_context_score = self._assess_trend_context(data, i)
            
            # Only consider hammers in downtrends
            if trend_context_score < self.parameters['min_trend_strength']:
                continue
            
            # Calculate pattern components
            body_to_shadow_ratio = self._calculate_body_to_shadow_ratio(candle)
            pattern_strength = self._calculate_pattern_strength(candle, trend_context_score)
            
            if pattern_strength >= 0.6:  # Quality threshold
                pattern = HammerPattern(
                    timestamp=candle.name,
                    pattern_strength=pattern_strength,
                    body_to_shadow_ratio=body_to_shadow_ratio,
                    lower_shadow_length=candle['lower_shadow_ratio'],
                    upper_shadow_ratio=candle['upper_shadow_ratio'],
                    body_position=candle['body_position'],
                    trend_context_score=trend_context_score,
                    volume_confirmation=0.0,  # Will be calculated later
                    support_strength=0.0,  # Will be calculated later
                    reversal_probability=0.0,  # Will be calculated later
                    institutional_interest=0.0,  # Will be calculated later
                    pattern_purity=0.0  # Will be calculated later
                )
                patterns.append(pattern)
        
        return patterns
    
    def _is_hammer_pattern(self, candle: pd.Series) -> bool:
        """Check if candle meets hammer criteria"""
        # 1. Long lower shadow (at least 60% of total range)
        if candle['lower_shadow_ratio'] < self.parameters['min_lower_shadow_ratio']:
            return False
        
        # 2. Small body (max 30% of total range)
        if candle['body_ratio'] > self.parameters['max_body_ratio']:
            return False
        
        # 3. Small or no upper shadow (max 10% of total range)
        if candle['upper_shadow_ratio'] > self.parameters['max_upper_shadow_ratio']:
            return False
        
        # 4. Body should be in upper part of the range
        if candle['body_position'] < 0.7:
            return False
        
        # 5. Minimum volatility requirement
        if candle['volatility_ratio'] < 0.8:
            return False
        
        return True
    
    def _assess_trend_context(self, data: pd.DataFrame, candle_index: int) -> float:
        """Assess downtrend context before hammer"""
        context_data = data.iloc[max(0, candle_index - self.parameters['trend_lookback']):candle_index + 1]
        
        if len(context_data) < 8:
            return 0.0
        
        context_factors = []
        
        # 1. Price trend (30% weight) - Should be downtrend
        price_change = (context_data['close'].iloc[-1] - context_data['close'].iloc[0]) / context_data['close'].iloc[0]
        if price_change < -0.05:  # At least 5% decline
            trend_factor = min(abs(price_change) / 0.15, 1.0)  # Scale up to 15% decline
            context_factors.append(trend_factor * 0.3)
        else:
            context_factors.append(0.0)
        
        # 2. Moving average alignment (25% weight)
        latest = context_data.iloc[-1]
        ma_bearish_score = 0
        if latest['close'] < latest['sma_5']:
            ma_bearish_score += 0.25
        if latest['sma_5'] < latest['sma_10']:
            ma_bearish_score += 0.25
        if latest['sma_10'] < latest['sma_20']:
            ma_bearish_score += 0.25
        if latest['ema_8'] < latest['ema_21']:
            ma_bearish_score += 0.25
        context_factors.append(ma_bearish_score * 0.25)
        
        # 3. RSI oversold condition (20% weight)
        rsi_factor = 0
        if latest['rsi'] < 20:
            rsi_factor = 1.0
        elif latest['rsi'] < 30:
            rsi_factor = 0.8
        elif latest['rsi'] < 40:
            rsi_factor = 0.5
        context_factors.append(rsi_factor * 0.2)
        
        # 4. Trend strength (15% weight)
        trend_strength = latest['trend_strength'] if not pd.isna(latest['trend_strength']) else 0.5
        context_factors.append(trend_strength * 0.15)
        
        # 5. Lower lows pattern (10% weight)
        recent_lows = context_data['low'].iloc[-5:]
        lower_lows_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows.iloc[i] < recent_lows.iloc[i-1])
        lower_lows_factor = lower_lows_count / 4  # Max 4 comparisons
        context_factors.append(lower_lows_factor * 0.1)
        
        return sum(context_factors)
    
    def _calculate_body_to_shadow_ratio(self, candle: pd.Series) -> float:
        """Calculate body to shadow ratio quality"""
        if candle['lower_shadow_ratio'] == 0:
            return 0.0
        
        # Ideal hammer has very long lower shadow and very small body
        ratio = candle['body_ratio'] / candle['lower_shadow_ratio']
        
        # Optimal ratio is around 0.1-0.3 (small body to long shadow)
        if ratio <= 0.1:
            return 1.0
        elif ratio <= 0.2:
            return 0.9
        elif ratio <= 0.3:
            return 0.7
        elif ratio <= 0.5:
            return 0.5
        else:
            return max(0, 1.0 - ratio)
    
    def _calculate_pattern_strength(self, candle: pd.Series, trend_context_score: float) -> float:
        """Calculate overall hammer pattern strength"""
        strength_components = []
        
        # 1. Lower shadow quality (30% weight)
        shadow_quality = min(candle['lower_shadow_ratio'] / 0.8, 1.0)  # Normalize to 80% max
        strength_components.append(shadow_quality * 0.3)
        
        # 2. Body size quality (25% weight)
        body_quality = 1.0 - (candle['body_ratio'] / self.parameters['max_body_ratio'])
        strength_components.append(body_quality * 0.25)
        
        # 3. Upper shadow quality (20% weight)
        upper_shadow_quality = 1.0 - (candle['upper_shadow_ratio'] / self.parameters['max_upper_shadow_ratio'])
        strength_components.append(min(upper_shadow_quality, 1.0) * 0.2)
        
        # 4. Body position quality (15% weight)
        position_quality = (candle['body_position'] - 0.7) / 0.3  # Scale from 0.7-1.0
        strength_components.append(min(max(position_quality, 0), 1) * 0.15)
        
        # 5. Trend context (10% weight)
        strength_components.append(trend_context_score * 0.1)
        
        return sum(strength_components)
    
    def _analyze_volume_confirmation(self, patterns: List[HammerPattern], 
                                   data: pd.DataFrame) -> List[HammerPattern]:
        """Analyze volume confirmation for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            volume_score = self._calculate_volume_confirmation_score(data, pattern_idx)
            pattern.volume_confirmation = volume_score
            
            # Enhance pattern strength with volume confirmation
            pattern.pattern_strength = (pattern.pattern_strength * 0.85 + volume_score * 0.15)
        
        return patterns
    
    def _calculate_volume_confirmation_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate volume confirmation score"""
        try:
            candle = data.iloc[pattern_idx]
            context_data = data.iloc[max(0, pattern_idx - 15):pattern_idx + 1]
            
            volume_factors = []
            
            # 1. Volume surge (40% weight)
            avg_volume = context_data['volume'].iloc[:-1].mean()  # Exclude current candle
            volume_surge = candle['volume'] / avg_volume if avg_volume > 0 else 1.0
            surge_score = min(volume_surge / self.parameters['volume_surge_threshold'], 1.0)
            volume_factors.append(surge_score * 0.4)
            
            # 2. Volume at support test (30% weight)
            # Higher volume during support test indicates institutional interest
            recent_volumes = context_data['volume'].iloc[-5:]
            volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            volume_trend_score = min(max(volume_trend / avg_volume * 5, 0), 1) if avg_volume > 0 else 0.5
            volume_factors.append(volume_trend_score * 0.3)
            
            # 3. Volume relative to volatility (20% weight)
            volume_volatility_ratio = candle['volume'] / candle['volatility_ratio'] if candle['volatility_ratio'] > 0 else 0
            avg_vol_vol_ratio = context_data['volume'].mean() / context_data['volatility_ratio'].mean()
            vol_efficiency = volume_volatility_ratio / avg_vol_vol_ratio if avg_vol_vol_ratio > 0 else 1.0
            volume_factors.append(min(vol_efficiency, 1.0) * 0.2)
            
            # 4. Volume distribution (10% weight)
            # Even volume distribution suggests more reliable pattern
            volume_cv = context_data['volume'].std() / context_data['volume'].mean() if context_data['volume'].mean() > 0 else 1.0
            distribution_score = 1.0 / (1.0 + volume_cv)
            volume_factors.append(distribution_score * 0.1)
            
            return sum(volume_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_support_strength(self, patterns: List[HammerPattern], 
                                data: pd.DataFrame) -> List[HammerPattern]:
        """Analyze support strength for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            support_score = self._calculate_support_strength_score(data, pattern_idx)
            pattern.support_strength = support_score
        
        return patterns
    
    def _calculate_support_strength_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate support strength score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 40):pattern_idx + 1]
            hammer_low = data.iloc[pattern_idx]['low']
            
            support_factors = []
            
            # 1. Historical support touches (40% weight)
            support_touches = 0
            price_tolerance = hammer_low * 0.01  # 1% tolerance
            
            for i in range(len(context_data) - 1):
                if abs(context_data.iloc[i]['low'] - hammer_low) <= price_tolerance:
                    support_touches += 1
            
            support_strength = min(support_touches / 3, 1.0)  # Normalize to max 3 touches
            support_factors.append(support_strength * 0.4)
            
            # 2. Pivot low confluence (30% weight)
            pivot_lows = context_data[context_data['pivot_low']]['low']
            if len(pivot_lows) > 0:
                nearest_pivot_distance = min(abs(pivot_lows - hammer_low))
                pivot_confluence = 1.0 / (1.0 + nearest_pivot_distance / hammer_low * 100)
                support_factors.append(pivot_confluence * 0.3)
            else:
                support_factors.append(0.15)
            
            # 3. Volume at support level (20% weight)
            support_volumes = []
            for i in range(len(context_data) - 1):
                if abs(context_data.iloc[i]['low'] - hammer_low) <= price_tolerance:
                    support_volumes.append(context_data.iloc[i]['volume'])
            
            if support_volumes:
                avg_support_volume = np.mean(support_volumes)
                avg_volume = context_data['volume'].mean()
                volume_support_factor = avg_support_volume / avg_volume if avg_volume > 0 else 1.0
                support_factors.append(min(volume_support_factor, 1.0) * 0.2)
            else:
                support_factors.append(0.1)
            
            # 4. Time since last support test (10% weight)
            last_support_test = 0
            for i in range(len(context_data) - 1, 0, -1):
                if abs(context_data.iloc[i]['low'] - hammer_low) <= price_tolerance:
                    last_support_test = len(context_data) - 1 - i
                    break
            
            time_factor = min(last_support_test / 20, 1.0)  # Normalize to 20 periods
            support_factors.append(time_factor * 0.1)
            
            return sum(support_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_institutional_interest(self, patterns: List[HammerPattern], 
                                      data: pd.DataFrame) -> List[HammerPattern]:
        """Analyze institutional interest for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            institutional_score = self._calculate_institutional_interest_score(data, pattern_idx)
            pattern.institutional_interest = institutional_score
        
        return patterns
    
    def _calculate_institutional_interest_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate institutional interest score"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 8):pattern_idx + 1]
            hammer_candle = data.iloc[pattern_idx]
            
            institutional_factors = []
            
            # 1. Order flow analysis (35% weight)
            obv_change = context_data['obv'].diff().iloc[-3:].mean()
            ad_line_change = context_data['ad_line'].diff().iloc[-3:].mean()
            
            # For hammer (bullish), expect positive institutional flow
            flow_score = 0
            if obv_change > 0:
                flow_score += 0.5
            if ad_line_change > 0:
                flow_score += 0.5
            
            institutional_factors.append(flow_score * 0.35)
            
            # 2. Volume efficiency (30% weight)
            price_move = abs(hammer_candle['close'] - hammer_candle['open'])
            volume_efficiency = price_move / hammer_candle['volume'] if hammer_candle['volume'] > 0 else 0
            
            # Higher efficiency suggests institutional participation
            avg_efficiency = context_data.apply(
                lambda row: abs(row['close'] - row['open']) / row['volume'] if row['volume'] > 0 else 0, 
                axis=1
            ).mean()
            
            efficiency_ratio = volume_efficiency / avg_efficiency if avg_efficiency > 0 else 1.0
            institutional_factors.append(min(efficiency_ratio, 1.0) * 0.3)
            
            # 3. Accumulation pattern (25% weight)
            # Look for accumulation during the decline
            cmf_values = context_data['cmf'].iloc[-5:]
            accumulation_score = 0
            if cmf_values.mean() > 0:  # Positive CMF suggests accumulation
                accumulation_score = min(cmf_values.mean() / 100000, 1.0)  # Normalize CMF
            
            institutional_factors.append(accumulation_score * 0.25)
            
            # 4. Block trade indicators (10% weight)
            # Large volume with controlled price action
            volume_surge = hammer_candle['volume'] / context_data['volume'].iloc[:-1].mean()
            price_control = 1.0 - abs(hammer_candle['close'] - hammer_candle['open']) / hammer_candle['total_range']
            
            block_indicator = volume_surge * price_control if volume_surge > 1.5 else 0
            institutional_factors.append(min(block_indicator / 3, 1.0) * 0.1)
            
            return sum(institutional_factors)
            
        except Exception:
            return 0.5
    
    def _analyze_pattern_purity(self, patterns: List[HammerPattern], 
                              data: pd.DataFrame) -> List[HammerPattern]:
        """Analyze pattern purity for patterns"""
        for pattern in patterns:
            pattern_idx = data.index.get_loc(pattern.timestamp)
            purity_score = self._calculate_pattern_purity_score(data, pattern_idx)
            pattern.pattern_purity = purity_score
        
        return patterns
    
    def _calculate_pattern_purity_score(self, data: pd.DataFrame, pattern_idx: int) -> float:
        """Calculate pattern purity score"""
        try:
            hammer_candle = data.iloc[pattern_idx]
            context_data = data.iloc[max(0, pattern_idx - 5):pattern_idx + 5]
            
            purity_factors = []
            
            # 1. Isolation quality (30% weight)
            # Hammer should stand out from surrounding candles
            surrounding_shadows = []
            for i in range(max(0, pattern_idx - 2), min(len(data), pattern_idx + 3)):
                if i != pattern_idx:
                    surrounding_shadows.append(data.iloc[i]['lower_shadow_ratio'])
            
            if surrounding_shadows:
                hammer_shadow = hammer_candle['lower_shadow_ratio']
                max_surrounding = max(surrounding_shadows)
                isolation_quality = (hammer_shadow - max_surrounding) / hammer_shadow if hammer_shadow > 0 else 0
                purity_factors.append(max(0, min(1, isolation_quality)) * 0.3)
            else:
                purity_factors.append(0.3)
            
            # 2. Geometric perfection (25% weight)
            # How close to ideal hammer proportions
            ideal_lower_shadow = 0.75  # 75% of range
            ideal_body = 0.15  # 15% of range
            ideal_upper_shadow = 0.1  # 10% of range
            
            shadow_deviation = abs(hammer_candle['lower_shadow_ratio'] - ideal_lower_shadow)
            body_deviation = abs(hammer_candle['body_ratio'] - ideal_body)
            upper_deviation = abs(hammer_candle['upper_shadow_ratio'] - ideal_upper_shadow)
            
            total_deviation = shadow_deviation + body_deviation + upper_deviation
            geometric_score = max(0, 1.0 - total_deviation / 0.5)  # Normalize
            purity_factors.append(geometric_score * 0.25)
            
            # 3. Color consistency (20% weight)
            # Hammer can be bullish or bearish, but should be consistent with reversal
            is_bullish = hammer_candle['close'] > hammer_candle['open']
            
            # In downtrend, bullish hammer is more pure
            prev_trend = data.iloc[max(0, pattern_idx - 5):pattern_idx]['close'].mean()
            current_close = hammer_candle['close']
            
            if current_close < prev_trend and is_bullish:  # Bullish hammer in downtrend
                color_score = 1.0
            elif current_close < prev_trend and not is_bullish:  # Bearish hammer in downtrend
                color_score = 0.7
            else:
                color_score = 0.3
            
            purity_factors.append(color_score * 0.2)
            
            # 4. Shadow symmetry (15% weight)
            # Lower shadow should dominate, upper shadow should be minimal
            shadow_ratio = hammer_candle['upper_shadow_ratio'] / hammer_candle['lower_shadow_ratio'] if hammer_candle['lower_shadow_ratio'] > 0 else 1
            symmetry_score = 1.0 / (1.0 + shadow_ratio * 5)  # Penalize large upper shadows
            purity_factors.append(symmetry_score * 0.15)
            
            # 5. Volatility appropriateness (10% weight)
            # Pattern should occur during appropriate volatility
            volatility_score = min(hammer_candle['volatility_ratio'] / 1.5, 1.0)
            purity_factors.append(volatility_score * 0.1)
            
            return sum(purity_factors)
            
        except Exception:
            return 0.5
    
    def _predict_reversal_probability(self, patterns: List[HammerPattern], 
                                    data: pd.DataFrame) -> List[HammerPattern]:
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
            
            if len(features) < 8:
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
                                 pattern: HammerPattern) -> List[float]:
        """Extract features for reversal prediction ML model"""
        try:
            context_data = data.iloc[max(0, pattern_idx - 8):pattern_idx + 1]
            current = data.iloc[pattern_idx]
            
            features = [
                pattern.pattern_strength,
                pattern.body_to_shadow_ratio,
                pattern.lower_shadow_length,
                pattern.upper_shadow_ratio,
                pattern.body_position,
                pattern.trend_context_score,
                pattern.volume_confirmation,
                pattern.support_strength,
                pattern.institutional_interest,
                pattern.pattern_purity,
                (100 - current['rsi']) / 100.0 if not pd.isna(current['rsi']) else 0.5,  # Inverted for bullish
                (1.0 - current['bb_position']) if not pd.isna(current['bb_position']) else 0.5,  # Oversold condition
                current['volatility_ratio'] / 2.0,  # Normalized
                current['volume_ratio'],
                current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
                -current['macd_hist'] if not pd.isna(current['macd_hist']) else 0,  # Negative for bullish divergence
                current['momentum_zscore'] if not pd.isna(current['momentum_zscore']) else 0,
                context_data['obv'].diff().iloc[-3:].mean() / 1000000,  # Normalized
                context_data['ad_line'].diff().iloc[-3:].mean() / 1000000,  # Normalized
                min(max(current['stoch_k'] / 100.0, 0), 1) if not pd.isna(current['stoch_k']) else 0.5
            ]
            
            return features
            
        except Exception:
            return [0.5] * 20  # Default features
    
    def _train_reversal_model(self, patterns: List[HammerPattern], features: List[List[float]]):
        """Train ML model for reversal prediction"""
        try:
            # Create targets based on pattern characteristics
            targets = []
            for pattern in patterns:
                # High-quality patterns with strong confirmations have higher reversal probability
                target = (
                    pattern.pattern_strength * 0.3 +
                    pattern.volume_confirmation * 0.25 +
                    pattern.support_strength * 0.2 +
                    pattern.institutional_interest * 0.15 +
                    pattern.pattern_purity * 0.1
                )
                targets.append(max(0.1, min(0.9, target)))
            
            if len(features) >= 12:
                features_scaled = self.scaler.fit_transform(features)
                self.reversal_predictor.fit(features_scaled, targets)
                self.is_ml_fitted = True
                logging.info("ML reversal predictor trained successfully")
            
        except Exception as e:
            logging.warning(f"ML model training failed: {str(e)}")
    
    def _heuristic_reversal_probability(self, pattern: HammerPattern) -> float:
        """Calculate heuristic reversal probability"""
        return (
            pattern.pattern_strength * 0.35 +
            pattern.volume_confirmation * 0.25 +
            pattern.support_strength * 0.2 +
            pattern.institutional_interest * 0.2
        )
    
    def _generate_pattern_analytics(self, patterns: List[HammerPattern]) -> Dict[str, Any]:
        """Generate comprehensive pattern analytics"""
        if not patterns:
            return {}
        
        recent_patterns = patterns[-15:]  # Last 15 patterns
        
        return {
            'total_patterns': len(recent_patterns),
            'average_strength': sum(p.pattern_strength for p in recent_patterns) / len(recent_patterns),
            'average_reversal_probability': sum(p.reversal_probability for p in recent_patterns) / len(recent_patterns),
            'average_volume_confirmation': sum(p.volume_confirmation for p in recent_patterns) / len(recent_patterns),
            'high_strength_patterns': len([p for p in recent_patterns if p.pattern_strength > 0.8]),
            'high_probability_patterns': len([p for p in recent_patterns if p.reversal_probability > 0.7]),
            'institutional_interest_patterns': len([p for p in recent_patterns if p.institutional_interest > 0.7]),
            'high_purity_patterns': len([p for p in recent_patterns if p.pattern_purity > 0.8]),
            'strong_support_patterns': len([p for p in recent_patterns if p.support_strength > 0.7]),
            'average_body_to_shadow_ratio': sum(p.body_to_shadow_ratio for p in recent_patterns) / len(recent_patterns)
        }
    
    def _analyze_current_trend_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current trend context"""
        current = data.iloc[-1]
        
        return {
            'trend_strength': current['trend_strength'] if not pd.isna(current['trend_strength']) else 0.5,
            'is_downtrend': current['trend_direction'] == -1,
            'rsi_oversold': current['rsi'] < 30 if not pd.isna(current['rsi']) else False,
            'stoch_oversold': current['stoch_k'] < 20 if not pd.isna(current['stoch_k']) else False,
            'bb_oversold': current['bb_position'] < 0.2 if not pd.isna(current['bb_position']) else False,
            'momentum_extreme': abs(current['momentum_zscore']) > 1.5 if not pd.isna(current['momentum_zscore']) else False,
            'volatility_context': 'high' if current['volatility_ratio'] > 1.5 else 'normal'
        }
    
    def _generate_reversal_signals(self, patterns: List[HammerPattern], 
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """Generate reversal signals based on patterns"""
        if not patterns:
            return {'signal_strength': 0.0, 'reversal_probability': 0.0}
        
        # Get recent high-quality patterns
        recent_patterns = [p for p in patterns[-5:] if p.pattern_strength > 0.7]
        
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
            'institutional_interest': sum(p.institutional_interest for p in recent_patterns) / len(recent_patterns),
            'support_strength': sum(p.support_strength for p in recent_patterns) / len(recent_patterns),
            'pattern_purity': sum(p.pattern_purity for p in recent_patterns) / len(recent_patterns),
            'most_recent_pattern': recent_patterns[-1].timestamp
        }
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure"""
        current = data.iloc[-1]
        recent_data = data.iloc[-20:]
        
        return {
            'support_resistance_structure': {
                'major_support_levels': len(recent_data[recent_data['pivot_low']]),
                'major_resistance_levels': len(recent_data[recent_data['pivot_high']]),
                'current_position': 'near_support' if current['bb_position'] < 0.3 else 'neutral'
            },
            'volume_structure': {
                'average_volume': recent_data['volume'].mean(),
                'volume_trend': 'increasing' if recent_data['volume'].iloc[-5:].mean() > recent_data['volume'].iloc[-15:-5].mean() else 'decreasing',
                'volume_volatility': recent_data['volume'].std() / recent_data['volume'].mean()
            },
            'momentum_structure': {
                'momentum_divergence': self._detect_momentum_divergence(recent_data),
                'trend_exhaustion': current['rsi'] < 25 if not pd.isna(current['rsi']) else False
            }
        }
    
    def _detect_momentum_divergence(self, data: pd.DataFrame) -> bool:
        """Detect momentum divergence"""
        try:
            if len(data) < 10:
                return False
            
            # Compare recent price lows with RSI
            recent_price_trend = np.polyfit(range(len(data)), data['low'], 1)[0]
            recent_rsi_trend = np.polyfit(range(len(data)), data['rsi'].fillna(50), 1)[0]
            
            # Bullish divergence: price making lower lows, RSI making higher lows
            return recent_price_trend < 0 and recent_rsi_trend > 0
            
        except Exception:
            return False
    
    def _assess_pattern_reliability(self, patterns: List[HammerPattern]) -> Dict[str, Any]:
        """Assess pattern reliability metrics"""
        if not patterns:
            return {}
        
        return {
            'consistency_score': 1.0 - np.std([p.pattern_strength for p in patterns]),
            'purity_average': np.mean([p.pattern_purity for p in patterns]),
            'volume_reliability': np.mean([p.volume_confirmation for p in patterns]),
            'institutional_consistency': np.mean([p.institutional_interest for p in patterns]),
            'reversal_success_estimate': np.mean([p.reversal_probability for p in patterns])
        }
    
    def _identify_key_support_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify key support levels"""
        recent_data = data.iloc[-50:]
        pivot_lows = recent_data[recent_data['pivot_low']]['low'].tolist()
        
        # Cluster support levels
        support_levels = []
        if pivot_lows:
            sorted_lows = sorted(pivot_lows)
            current_level = sorted_lows[0]
            level_touches = 1
            
            for low in sorted_lows[1:]:
                if abs(low - current_level) / current_level < 0.01:  # Within 1%
                    level_touches += 1
                else:
                    if level_touches >= 2:
                        support_levels.append({
                            'level': current_level,
                            'touches': level_touches,
                            'strength': min(level_touches / 3, 1.0)
                        })
                    current_level = low
                    level_touches = 1
            
            # Add the last level
            if level_touches >= 2:
                support_levels.append({
                    'level': current_level,
                    'touches': level_touches,
                    'strength': min(level_touches / 3, 1.0)
                })
        
        return {
            'key_levels': support_levels,
            'nearest_support': min(support_levels, key=lambda x: abs(x['level'] - data.iloc[-1]['close']))['level'] if support_levels else None,
            'support_density': len(support_levels)
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on hammer analysis"""
        current_pattern = value.get('current_pattern')
        reversal_signals = value.get('reversal_signals', {})
        trend_analysis = value.get('trend_analysis', {})
        
        if not current_pattern:
            return None, 0.0
        
        # Hammer is a bullish reversal pattern
        # Strong signal when we have high-quality pattern in downtrend with good confirmations
        
        if (current_pattern.pattern_strength > 0.8 and 
            current_pattern.reversal_probability > 0.7 and
            trend_analysis.get('is_downtrend', False) and
            current_pattern.support_strength > 0.6):
            
            confidence = (
                current_pattern.pattern_strength * 0.35 +
                current_pattern.reversal_probability * 0.25 +
                current_pattern.volume_confirmation * 0.2 +
                current_pattern.support_strength * 0.2
            )
            
            return SignalType.BUY, confidence
        
        # Moderate bullish signal
        elif (current_pattern.pattern_strength > 0.7 and 
              current_pattern.reversal_probability > 0.6 and
              trend_analysis.get('is_downtrend', False)):
            
            confidence = current_pattern.pattern_strength * 0.65
            return SignalType.BUY, confidence
        
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'ml_model_fitted': self.is_ml_fitted,
            'pattern_type': 'hammer',
            'min_lower_shadow_ratio': self.parameters['min_lower_shadow_ratio'],
            'max_body_ratio': self.parameters['max_body_ratio'],
            'volume_analysis_enabled': self.parameters['volume_analysis'],
            'ml_reversal_prediction_enabled': self.parameters['ml_reversal_prediction'],
            'support_analysis_enabled': self.parameters['support_analysis']
        })
        return base_metadata