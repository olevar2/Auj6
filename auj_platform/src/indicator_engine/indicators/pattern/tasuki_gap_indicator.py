"""
Tasuki Gap Pattern Indicator

The Tasuki Gap is a continuation pattern that occurs within gaps, indicating the gap is likely
to be sustained and the trend will continue.

Pattern Types:
1. Upside Tasuki Gap: Occurs in uptrend
   - First candle: White/bullish candle
   - Second candle: Another white candle that gaps up
   - Third candle: Black candle that opens within second candle's body and closes in the gap

2. Downside Tasuki Gap: Occurs in downtrend
   - First candle: Black/bearish candle
   - Second candle: Another black candle that gaps down
   - Third candle: White candle that opens within second candle's body and closes in the gap

Trading Significance:
- Continuation pattern - trend likely to resume
- Gap represents strong momentum
- Third candle tests gap but fails to close it
- High reliability when volume confirms
- Best used in strong trending markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class TasukiGapSignal:
    """Signal data for Tasuki Gap pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_type: str  # 'upside_tasuki', 'downside_tasuki'
    gap_size: float
    gap_test_strength: float
    trend_strength: float
    volume_confirmation: bool
    continuation_probability: float
    momentum_persistence: float
    gap_sustainability: float
    confidence_level: float
    entry_price: float
    stop_loss: float
    take_profit: float
class TasukiGapIndicator(BasePatternIndicator):
    """
    Tasuki Gap Pattern Recognition Indicator
    
    Identifies upside and downside Tasuki Gap patterns with sophisticated
    gap analysis and trend continuation probability assessment.
    """
    
    def __init__(self, 
                 min_gap_ratio: float = 0.002,
                 min_trend_strength: float = 0.3,
                 trend_analysis_periods: int = 20,
                 volume_confirmation_threshold: float = 1.1,
                 atr_periods: int = 14,
                 momentum_periods: int = 10,
                 ml_training_periods: int = 120):
        """
        Initialize Tasuki Gap indicator
        
        Args:
            min_gap_ratio: Minimum gap size as ratio of price
            min_trend_strength: Minimum trend strength for pattern validity
            trend_analysis_periods: Periods for trend analysis
            volume_confirmation_threshold: Volume threshold for confirmation
            atr_periods: Periods for ATR calculation
            momentum_periods: Periods for momentum analysis
            ml_training_periods: Periods for ML model training
        """
        super().__init__()
        self.min_gap_ratio = min_gap_ratio
        self.min_trend_strength = min_trend_strength
        self.trend_analysis_periods = trend_analysis_periods
        self.volume_confirmation_threshold = volume_confirmation_threshold
        self.atr_periods = atr_periods
        self.momentum_periods = momentum_periods
        self.ml_training_periods = ml_training_periods        self.signals: List[TasukiGapSignal] = []
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize machine learning model for pattern validation"""
        self.ml_model = RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            class_weight='balanced'
        )
    
    def _calculate_atr(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate Average True Range"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr_list = []
        for i in range(1, len(data)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        tr_array = np.array(tr_list)
        atr = np.zeros(len(data))
        
        if len(tr_array) >= self.atr_periods:
            atr[self.atr_periods:] = pd.Series(tr_array).rolling(
                window=self.atr_periods
            ).mean().values[self.atr_periods-1:]
        
        return atr    
    def _analyze_trend_strength(self, data: pd.DataFrame) -> Tuple[float, str]:
        """Analyze trend strength and direction"""
        if len(data) < self.trend_analysis_periods:
            return 0.0, 'sideways'
        
        recent_data = data.tail(self.trend_analysis_periods)
        close_prices = recent_data['close']
        
        # Linear regression for trend
        x = np.arange(len(close_prices))
        slope, _, r_value, _, _ = stats.linregress(x, close_prices)
        
        r_squared = r_value ** 2
        
        # Determine trend direction and strength
        if r_squared < self.min_trend_strength:
            return r_squared, 'sideways'
        elif slope > 0:
            return r_squared, 'uptrend'
        else:
            return r_squared, 'downtrend'
    
    def _detect_gap(self, candle1: pd.Series, candle2: pd.Series) -> Tuple[bool, float, str]:
        """Detect gap between two consecutive candles"""
        c1_high = candle1['high']
        c1_low = candle1['low']
        c2_high = candle2['high']
        c2_low = candle2['low']
        
        # Check for gap up
        if c2_low > c1_high:
            gap_size = (c2_low - c1_high) / c1_high
            if gap_size >= self.min_gap_ratio:
                return True, gap_size, 'gap_up'
        
        # Check for gap down
        elif c2_high < c1_low:
            gap_size = (c1_low - c2_high) / c1_low
            if gap_size >= self.min_gap_ratio:
                return True, gap_size, 'gap_down'
        
        return False, 0.0, 'no_gap'    
    def _identify_tasuki_gap_pattern(self, data: pd.DataFrame, 
                                   index: int) -> Tuple[Optional[str], Dict]:
        """Identify Tasuki Gap pattern at given index"""
        if index < 2:
            return None, {}
        
        # Get the three candles
        first_candle = data.iloc[index - 2]
        second_candle = data.iloc[index - 1]
        third_candle = data.iloc[index]
        
        # Analyze trend context
        trend_strength, trend_direction = self._analyze_trend_strength(
            data.iloc[:index-2]  # Trend before the pattern
        )
        
        if trend_direction == 'sideways':
            return None, {}
        
        # Check for gap between first and second candles
        has_gap, gap_size, gap_direction = self._detect_gap(first_candle, second_candle)
        
        if not has_gap:
            return None, {}
        
        # Analyze candle characteristics
        first_is_bullish = first_candle['close'] > first_candle['open']
        second_is_bullish = second_candle['close'] > second_candle['open']
        third_is_bullish = third_candle['close'] > third_candle['open']
        
        pattern_type = None
        pattern_data = {}
        
        # Check for Upside Tasuki Gap
        if (trend_direction == 'uptrend' and gap_direction == 'gap_up' and
            first_is_bullish and second_is_bullish and not third_is_bullish):
            
            # Third candle should open within second candle's body
            second_body_top = max(second_candle['open'], second_candle['close'])
            second_body_bottom = min(second_candle['open'], second_candle['close'])
            
            third_opens_in_body = (second_body_bottom <= third_candle['open'] <= second_body_top)
            
            # Third candle should close in the gap
            gap_bottom = first_candle['high']
            gap_top = second_candle['low']
            third_closes_in_gap = (gap_bottom <= third_candle['close'] <= gap_top)
            
            if third_opens_in_body and third_closes_in_gap:
                pattern_type = 'upside_tasuki'
        
        # Check for Downside Tasuki Gap
        elif (trend_direction == 'downtrend' and gap_direction == 'gap_down' and
              not first_is_bullish and not second_is_bullish and third_is_bullish):
            
            # Third candle should open within second candle's body
            second_body_top = max(second_candle['open'], second_candle['close'])
            second_body_bottom = min(second_candle['open'], second_candle['close'])
            
            third_opens_in_body = (second_body_bottom <= third_candle['open'] <= second_body_top)
            
            # Third candle should close in the gap
            gap_top = first_candle['low']
            gap_bottom = second_candle['high']
            third_closes_in_gap = (gap_bottom <= third_candle['close'] <= gap_top)
            
            if third_opens_in_body and third_closes_in_gap:
                pattern_type = 'downside_tasuki'
        
        if pattern_type:
            pattern_data = {
                'first_candle': first_candle,
                'second_candle': second_candle,
                'third_candle': third_candle,
                'gap_size': gap_size,
                'gap_direction': gap_direction,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction
            }
        
        return pattern_type, pattern_data    
    def _calculate_gap_test_strength(self, pattern_data: Dict) -> float:
        """Calculate how well the third candle tests the gap"""
        third_candle = pattern_data['third_candle']
        gap_size = pattern_data['gap_size']
        
        if pattern_data['gap_direction'] == 'gap_up':
            # For upside gap, measure how far third candle penetrates the gap
            gap_bottom = pattern_data['first_candle']['high']
            gap_top = pattern_data['second_candle']['low']
            gap_range = gap_top - gap_bottom
            
            if gap_range > 0:
                penetration = (third_candle['close'] - gap_bottom) / gap_range
                # Good test is partial penetration (0.3 to 0.7)
                test_strength = 1 - abs(penetration - 0.5) * 2
            else:
                test_strength = 0.5
        
        else:  # gap_down
            # For downside gap, measure how far third candle penetrates the gap
            gap_top = pattern_data['first_candle']['low']
            gap_bottom = pattern_data['second_candle']['high']
            gap_range = gap_top - gap_bottom
            
            if gap_range > 0:
                penetration = (gap_top - third_candle['close']) / gap_range
                # Good test is partial penetration (0.3 to 0.7)
                test_strength = 1 - abs(penetration - 0.5) * 2
            else:
                test_strength = 0.5
        
        return max(0.0, min(1.0, test_strength))
    
    def _analyze_volume_confirmation(self, data: pd.DataFrame) -> bool:
        """Analyze volume for pattern confirmation"""
        if 'volume' not in data.columns or len(data) < 20:
            return False
        
        volumes = data['volume']
        vol_ma = volumes.rolling(window=20).mean()
        
        # Check volume on the three pattern candles
        pattern_volumes = volumes.tail(3)
        avg_vol = vol_ma.tail(3).mean()
        
        if avg_vol > 0:
            # At least two of the three candles should have above-average volume
            above_avg_count = sum(vol >= (avg_vol * self.volume_confirmation_threshold) 
                                for vol in pattern_volumes)
            return above_avg_count >= 2
        
        return False    
    def _calculate_continuation_probability(self, pattern_data: Dict,
                                          gap_test_strength: float,
                                          volume_confirmation: bool) -> float:
        """Calculate probability of trend continuation"""
        trend_strength = pattern_data['trend_strength']
        gap_size = pattern_data['gap_size']
        
        # Base probability from trend strength
        base_prob = trend_strength
        
        # Gap size bonus (larger gaps more likely to hold)
        gap_bonus = min(0.2, gap_size * 50)  # Scale gap size
        
        # Gap test quality (good test increases continuation probability)
        test_bonus = gap_test_strength * 0.15
        
        # Volume confirmation bonus
        volume_bonus = 0.1 if volume_confirmation else 0
        
        continuation_prob = base_prob + gap_bonus + test_bonus + volume_bonus
        
        return min(1.0, continuation_prob)
    
    def _calculate_momentum_persistence(self, data: pd.DataFrame,
                                      pattern_type: str) -> float:
        """Calculate momentum persistence likelihood"""
        if len(data) < self.momentum_periods:
            return 0.5
        
        recent_data = data.tail(self.momentum_periods)
        close_prices = recent_data['close']
        
        # Calculate momentum indicators
        roc = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        
        # Price acceleration
        if len(close_prices) >= 3:
            mid_point = len(close_prices) // 2
            first_half_roc = ((close_prices.iloc[mid_point] - close_prices.iloc[0]) / 
                             close_prices.iloc[0])
            second_half_roc = ((close_prices.iloc[-1] - close_prices.iloc[mid_point]) / 
                              close_prices.iloc[mid_point])
            acceleration = second_half_roc - first_half_roc
        else:
            acceleration = 0
        
        # Calculate persistence score
        if pattern_type == 'upside_tasuki':
            # For upside pattern, we want positive momentum
            momentum_score = max(0, np.tanh(roc * 10)) * 0.7
            accel_score = max(0, np.tanh(acceleration * 20)) * 0.3
        else:
            # For downside pattern, we want negative momentum
            momentum_score = max(0, np.tanh(-roc * 10)) * 0.7
            accel_score = max(0, np.tanh(-acceleration * 20)) * 0.3
        
        return momentum_score + accel_score    
    def _calculate_gap_sustainability(self, pattern_data: Dict,
                                    gap_test_strength: float) -> float:
        """Calculate gap sustainability score"""
        gap_size = pattern_data['gap_size']
        trend_strength = pattern_data['trend_strength']
        
        # Larger gaps with strong trends are more sustainable
        size_factor = min(1.0, gap_size * 100)  # Scale gap size
        trend_factor = trend_strength
        
        # Good gap test (partial fill) actually increases sustainability
        # as it confirms the gap's support/resistance
        test_factor = gap_test_strength
        
        sustainability = (size_factor * 0.4 + trend_factor * 0.4 + test_factor * 0.2)
        
        return sustainability
    
    def _prepare_ml_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Prepare features for ML model"""
        if index < 20:
            return np.array([])
        
        features = []
        
        # Pattern structure features
        pattern_type, pattern_data = self._identify_tasuki_gap_pattern(data, index)
        
        if not pattern_type:
            return np.array([])
        
        features.extend([
            pattern_data['gap_size'],
            pattern_data['trend_strength'],
            1 if pattern_type == 'upside_tasuki' else 0
        ])
        
        # Gap test strength
        gap_test_strength = self._calculate_gap_test_strength(pattern_data)
        features.append(gap_test_strength)
        
        # Volume features
        if 'volume' in data.columns:
            volumes = data['volume']
            vol_ma = volumes.rolling(window=20).mean()
            recent_vol_ratio = volumes.iloc[index] / vol_ma.iloc[index] if vol_ma.iloc[index] > 0 else 1
            features.append(recent_vol_ratio)
        else:
            features.append(1.0)
        
        # Momentum features
        momentum_persistence = self._calculate_momentum_persistence(
            data.iloc[index-self.momentum_periods+1:index+1], pattern_type
        )
        features.append(momentum_persistence)
        
        # Volatility features
        recent_closes = data.iloc[index-9:index+1]['close']
        volatility = recent_closes.std() / recent_closes.mean()
        features.append(volatility)
        
        return np.array(features)    
    def _train_ml_model(self, data: pd.DataFrame):
        """Train ML model for pattern validation"""
        if len(data) < self.ml_training_periods:
            return
        
        features_list = []
        labels_list = []
        
        for i in range(22, len(data) - 10):
            features = self._prepare_ml_features(data, i)
            if len(features) == 0:
                continue
            
            # Create label based on trend continuation
            pattern_type, _ = self._identify_tasuki_gap_pattern(data, i)
            if not pattern_type:
                continue
            
            current_close = data.iloc[i]['close']
            future_data = data.iloc[i+1:i+11]  # Next 10 periods
            
            if pattern_type == 'upside_tasuki':
                # Check for continued upward movement
                future_high = future_data['high'].max()
                price_move = (future_high - current_close) / current_close
                # Success if price moves up > 1.5%
                label = 1 if price_move > 0.015 else 0
            else:
                # Check for continued downward movement
                future_low = future_data['low'].min()
                price_move = (current_close - future_low) / current_close
                # Success if price moves down > 1.5%
                label = 1 if price_move > 0.015 else 0
            
            features_list.append(features)
            labels_list.append(label)
        
        if len(features_list) < 20:
            return
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y)
    
    def _get_ml_confidence(self, data: pd.DataFrame, index: int) -> float:
        """Get ML model confidence"""
        if self.ml_model is None:
            return 0.5
        
        features = self._prepare_ml_features(data, index)
        if len(features) == 0:
            return 0.5
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            return probabilities[1] if len(probabilities) > 1 else 0.5
        except:
            return 0.5    
    def _determine_signal_type(self, pattern_type: str) -> SignalType:
        """Determine trading signal based on pattern type"""
        if pattern_type == 'upside_tasuki':
            return SignalType.BUY
        elif pattern_type == 'downside_tasuki':
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _calculate_trading_levels(self, data: pd.DataFrame,
                                current_price: float,
                                pattern_type: str,
                                pattern_data: Dict) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        atr_values = self._calculate_atr(data)
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0
        gap_size = pattern_data['gap_size']
        
        if current_atr == 0:
            # Fallback to percentage levels
            if pattern_type == 'upside_tasuki':
                stop_loss = current_price * 0.97
                take_profit = current_price * 1.05
            else:
                stop_loss = current_price * 1.03
                take_profit = current_price * 0.95
        else:
            # ATR-based levels
            if pattern_type == 'upside_tasuki':
                # Stop below the gap
                gap_bottom = pattern_data['first_candle']['high']
                stop_loss = min(gap_bottom - current_atr, current_price - current_atr * 1.5)
                take_profit = current_price + (current_atr * 2.5)
            else:
                # Stop above the gap
                gap_top = pattern_data['first_candle']['low']
                stop_loss = max(gap_top + current_atr, current_price + current_atr * 1.5)
                take_profit = current_price - (current_atr * 2.5)
        
        return stop_loss, take_profit
    
    def update(self, data: pd.DataFrame) -> Optional[TasukiGapSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            TasukiGapSignal if pattern detected, None otherwise
        """
        if len(data) < max(self.atr_periods, self.trend_analysis_periods, 20):
            return None
        
        # Train ML model if enough data
        if len(data) >= self.ml_training_periods and self.ml_model is not None:
            self._train_ml_model(data)
        
        # Check for Tasuki Gap pattern
        current_index = len(data) - 1
        pattern_type, pattern_data = self._identify_tasuki_gap_pattern(data, current_index)
        
        if not pattern_type:
            return None        
        # Calculate pattern characteristics
        gap_test_strength = self._calculate_gap_test_strength(pattern_data)
        volume_confirmation = self._analyze_volume_confirmation(data)
        continuation_probability = self._calculate_continuation_probability(
            pattern_data, gap_test_strength, volume_confirmation
        )
        momentum_persistence = self._calculate_momentum_persistence(data, pattern_type)
        gap_sustainability = self._calculate_gap_sustainability(pattern_data, gap_test_strength)
        
        # Get ML confidence
        ml_confidence = self._get_ml_confidence(data, current_index)
        
        # Calculate overall confidence
        confidence_level = (
            continuation_probability * 0.3 +
            momentum_persistence * 0.25 +
            gap_sustainability * 0.2 +
            ml_confidence * 0.15 +
            (0.1 if volume_confirmation else 0)
        )
        
        # Determine signal
        signal_type = self._determine_signal_type(pattern_type)
        
        # Calculate trading levels
        current_price = data.iloc[-1]['close']
        stop_loss, take_profit = self._calculate_trading_levels(
            data, current_price, pattern_type, pattern_data
        )
        
        signal = TasukiGapSignal(
            timestamp=data.index[-1],
            signal_type=signal_type,
            pattern_type=pattern_type,
            gap_size=pattern_data['gap_size'],
            gap_test_strength=gap_test_strength,
            trend_strength=pattern_data['trend_strength'],
            volume_confirmation=volume_confirmation,
            continuation_probability=continuation_probability,
            momentum_persistence=momentum_persistence,
            gap_sustainability=gap_sustainability,
            confidence_level=confidence_level,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.signals.append(signal)
        return signal
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[TasukiGapSignal]:
        """Get signals within date range"""
        signals = self.signals
        
        if start_date:
            signals = [s for s in signals if s.timestamp >= start_date]
        
        if end_date:
            signals = [s for s in signals if s.timestamp <= end_date]
        
        return signals    
    def get_pattern_statistics(self) -> Dict:
        """Get pattern detection statistics"""
        if not self.signals:
            return {}
        
        pattern_types = [s.pattern_type for s in self.signals]
        
        return {
            'total_signals': len(self.signals),
            'avg_gap_size': np.mean([s.gap_size for s in self.signals]),
            'avg_gap_test_strength': np.mean([s.gap_test_strength for s in self.signals]),
            'avg_trend_strength': np.mean([s.trend_strength for s in self.signals]),
            'avg_continuation_probability': np.mean([s.continuation_probability for s in self.signals]),
            'avg_momentum_persistence': np.mean([s.momentum_persistence for s in self.signals]),
            'avg_gap_sustainability': np.mean([s.gap_sustainability for s in self.signals]),
            'avg_confidence_level': np.mean([s.confidence_level for s in self.signals]),
            'pattern_type_distribution': {
                ptype: pattern_types.count(ptype) / len(pattern_types)
                for ptype in set(pattern_types)
            },
            'volume_confirmation_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals),
            'high_continuation_probability': len([s for s in self.signals if s.continuation_probability > 0.7]),
            'high_confidence_signals': len([s for s in self.signals if s.confidence_level > 0.7]),
            'sustainable_gaps': len([s for s in self.signals if s.gap_sustainability > 0.6])
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()