"""
Kicking Pattern Indicator

The Kicking pattern is a powerful two-candle reversal pattern consisting of two opposite Marubozu candles
that gap away from each other, indicating a strong shift in market sentiment.

Pattern Types:
1. Bullish Kicking: Bearish Marubozu followed by gap up and Bullish Marubozu
2. Bearish Kicking: Bullish Marubozu followed by gap down and Bearish Marubozu

Pattern Characteristics:
- Two consecutive Marubozu candles of opposite colors
- Gap between the candles (no overlap)
- Strong volume on both candles
- Indicates dramatic sentiment shift
- High reliability reversal signal

Trading Significance:
- Strong reversal pattern with high success rate
- Indicates institutional participation
- Gap shows urgency and conviction
- Best used after extended trends
- Volume confirmation is crucial
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class KickingSignal:
    """Signal data for Kicking pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_type: str  # 'bullish_kicking', 'bearish_kicking'
    gap_size: float
    first_candle_strength: float
    second_candle_strength: float
    volume_confirmation: bool
    trend_reversal_strength: float
    institutional_participation: float
    momentum_shift: float
    reliability_score: float
    confidence_level: float
    entry_price: float
    stop_loss: float
    take_profit: float
class KickingIndicator(BasePatternIndicator):
    """
    Kicking Pattern Recognition Indicator
    
    Identifies bullish and bearish kicking patterns with sophisticated
    gap analysis and institutional participation detection.
    """
    
    def __init__(self, 
                 min_marubozu_body_ratio: float = 0.8,
                 max_marubozu_shadow_ratio: float = 0.03,
                 min_gap_ratio: float = 0.001,
                 volume_confirmation_threshold: float = 1.5,
                 trend_reversal_periods: int = 20,
                 atr_periods: int = 14,
                 institutional_volume_threshold: float = 2.0,
                 ml_training_periods: int = 100):
        """
        Initialize Kicking indicator
        
        Args:
            min_marubozu_body_ratio: Minimum body ratio for Marubozu qualification
            max_marubozu_shadow_ratio: Maximum shadow ratio for Marubozu
            min_gap_ratio: Minimum gap size as ratio of price
            volume_confirmation_threshold: Volume threshold for confirmation
            trend_reversal_periods: Periods for trend reversal analysis
            atr_periods: Periods for ATR calculation
            institutional_volume_threshold: Threshold for institutional participation
            ml_training_periods: Periods for ML model training
        """
        super().__init__()
        self.min_marubozu_body_ratio = min_marubozu_body_ratio
        self.max_marubozu_shadow_ratio = max_marubozu_shadow_ratio
        self.min_gap_ratio = min_gap_ratio        self.volume_confirmation_threshold = volume_confirmation_threshold
        self.trend_reversal_periods = trend_reversal_periods
        self.atr_periods = atr_periods
        self.institutional_volume_threshold = institutional_volume_threshold
        self.ml_training_periods = ml_training_periods
        self.signals: List[KickingSignal] = []
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize machine learning model for pattern validation"""
        self.ml_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.08,
            random_state=42,
            subsample=0.9
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
    def _is_marubozu(self, candle: pd.Series) -> Tuple[bool, str, float]:
        """Check if candle is a Marubozu and return type and strength"""
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']
        
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        if total_range == 0:
            return False, 'none', 0.0
        
        body_ratio = body_size / total_range
        shadow_ratio = (upper_shadow + lower_shadow) / total_range
        
        # Check Marubozu criteria
        is_marubozu = (body_ratio >= self.min_marubozu_body_ratio and 
                      shadow_ratio <= self.max_marubozu_shadow_ratio)
        
        if not is_marubozu:
            return False, 'none', 0.0
        
        # Determine type
        if close_price > open_price:
            marubozu_type = 'bullish'
        else:
            marubozu_type = 'bearish'
        
        # Calculate strength
        strength = (body_ratio + (1 - shadow_ratio)) / 2
        
        return True, marubozu_type, strength
    
    def _detect_gap(self, candle1: pd.Series, candle2: pd.Series) -> Tuple[bool, float, str]:
        """Detect gap between two candles"""
        # First candle ranges
        c1_high = candle1['high']
        c1_low = candle1['low']
        
        # Second candle ranges
        c2_high = candle2['high']
        c2_low = candle2['low']
        
        # Check for gap up (bullish gap)
        if c2_low > c1_high:
            gap_size = (c2_low - c1_high) / c1_high
            if gap_size >= self.min_gap_ratio:
                return True, gap_size, 'gap_up'
        
        # Check for gap down (bearish gap)
        elif c2_high < c1_low:
            gap_size = (c1_low - c2_high) / c1_low
            if gap_size >= self.min_gap_ratio:
                return True, gap_size, 'gap_down'
        
        return False, 0.0, 'no_gap'    
    def _identify_kicking_pattern(self, data: pd.DataFrame, 
                                index: int) -> Tuple[Optional[str], Dict]:
        """Identify kicking pattern at given index"""
        if index < 1:
            return None, {}
        
        # Get the two candles
        first_candle = data.iloc[index - 1]
        second_candle = data.iloc[index]
        
        # Check if both are Marubozu
        is_first_marubozu, first_type, first_strength = self._is_marubozu(first_candle)
        is_second_marubozu, second_type, second_strength = self._is_marubozu(second_candle)
        
        if not (is_first_marubozu and is_second_marubozu):
            return None, {}
        
        # Check for gap
        has_gap, gap_size, gap_direction = self._detect_gap(first_candle, second_candle)
        
        if not has_gap:
            return None, {}
        
        # Determine kicking pattern type
        pattern_type = None
        
        if (first_type == 'bearish' and second_type == 'bullish' and 
            gap_direction == 'gap_up'):
            pattern_type = 'bullish_kicking'
        elif (first_type == 'bullish' and second_type == 'bearish' and 
              gap_direction == 'gap_down'):
            pattern_type = 'bearish_kicking'
        
        if not pattern_type:
            return None, {}
        
        pattern_data = {
            'first_candle_strength': first_strength,
            'second_candle_strength': second_strength,
            'gap_size': gap_size,
            'gap_direction': gap_direction,
            'first_candle': first_candle,
            'second_candle': second_candle
        }
        
        return pattern_type, pattern_data    
    def _analyze_volume_confirmation(self, data: pd.DataFrame, 
                                   pattern_data: Dict) -> bool:
        """Analyze volume for kicking pattern confirmation"""
        if 'volume' not in data.columns or len(data) < 20:
            return False
        
        volumes = data['volume']
        vol_ma = volumes.rolling(window=20).mean()
        
        # Get volumes of the two pattern candles
        first_candle_idx = len(data) - 2
        second_candle_idx = len(data) - 1
        
        if (first_candle_idx < len(vol_ma) and second_candle_idx < len(vol_ma) and
            vol_ma.iloc[first_candle_idx] > 0 and vol_ma.iloc[second_candle_idx] > 0):
            
            first_vol = volumes.iloc[first_candle_idx]
            second_vol = volumes.iloc[second_candle_idx]
            avg_vol = (vol_ma.iloc[first_candle_idx] + vol_ma.iloc[second_candle_idx]) / 2
            
            # Both candles should have above-average volume
            first_vol_confirm = first_vol >= (avg_vol * self.volume_confirmation_threshold)
            second_vol_confirm = second_vol >= (avg_vol * self.volume_confirmation_threshold)
            
            return first_vol_confirm and second_vol_confirm
        
        return False
    
    def _calculate_institutional_participation(self, data: pd.DataFrame) -> float:
        """Calculate likelihood of institutional participation"""
        if 'volume' not in data.columns or len(data) < 20:
            return 0.5
        
        volumes = data['volume']
        vol_ma = volumes.rolling(window=20).mean()
        vol_std = volumes.rolling(window=20).std()
        
        # Check recent volume spikes
        recent_volumes = volumes.tail(2)
        recent_vol_ma = vol_ma.tail(2).mean()
        recent_vol_std = vol_std.tail(2).mean()
        
        if recent_vol_ma > 0 and recent_vol_std > 0:
            # Calculate volume z-score
            max_recent_vol = recent_volumes.max()
            vol_z_score = (max_recent_vol - recent_vol_ma) / recent_vol_std
            
            # High z-score indicates unusual volume (institutional activity)
            institutional_score = min(1.0, vol_z_score / 3.0)  # Normalize to 0-1
            
            # Additional boost if both candles have very high volume
            avg_recent_vol = recent_volumes.mean()
            if avg_recent_vol >= (recent_vol_ma * self.institutional_volume_threshold):
                institutional_score = min(1.0, institutional_score + 0.3)
            
            return max(0.0, institutional_score)
        
        return 0.5    
    def _calculate_trend_reversal_strength(self, data: pd.DataFrame, 
                                         pattern_type: str) -> float:
        """Calculate strength of trend reversal"""
        if len(data) < self.trend_reversal_periods:
            return 0.5
        
        # Analyze trend before the pattern
        pre_pattern_data = data.iloc[:-2]  # Exclude the two pattern candles
        trend_data = pre_pattern_data.tail(self.trend_reversal_periods)
        close_prices = trend_data['close']
        
        # Calculate trend using linear regression
        x = np.arange(len(close_prices))
        slope, _, r_value, _, _ = stats.linregress(x, close_prices)
        
        r_squared = r_value ** 2
        
        # Determine if pattern opposes the trend
        is_uptrend = slope > 0
        is_bullish_kicking = pattern_type == 'bullish_kicking'
        
        # Reversal strength is higher when:
        # 1. There was a strong trend (high R-squared)
        # 2. Pattern opposes the trend direction
        if r_squared > 0.3:  # Strong trend
            if (is_uptrend and not is_bullish_kicking) or (not is_uptrend and is_bullish_kicking):
                # Pattern opposes trend - strong reversal signal
                reversal_strength = r_squared
            else:
                # Pattern continues trend - weaker signal
                reversal_strength = 0.3
        else:
            # Weak or sideways trend
            reversal_strength = 0.5
        
        return min(1.0, reversal_strength)
    
    def _calculate_momentum_shift(self, data: pd.DataFrame, pattern_type: str) -> float:
        """Calculate momentum shift strength"""
        if len(data) < 10:
            return 0.5
        
        # Calculate momentum before and during pattern
        pre_pattern_closes = data.iloc[:-2]['close'].tail(5)
        pattern_closes = data.iloc[-2:]['close']
        
        # Pre-pattern momentum
        if len(pre_pattern_closes) >= 2:
            pre_momentum = (pre_pattern_closes.iloc[-1] - pre_pattern_closes.iloc[0]) / pre_pattern_closes.iloc[0]
        else:
            pre_momentum = 0
        
        # Pattern momentum
        pattern_momentum = (pattern_closes.iloc[-1] - pattern_closes.iloc[0]) / pattern_closes.iloc[0]
        
        # Check for momentum shift
        if pattern_type == 'bullish_kicking':
            # For bullish kicking, we want positive pattern momentum
            # Stronger if it reverses negative pre-pattern momentum
            if pre_momentum < 0 and pattern_momentum > 0:
                momentum_shift = abs(pre_momentum) + pattern_momentum
            else:
                momentum_shift = pattern_momentum
        else:
            # For bearish kicking, we want negative pattern momentum
            # Stronger if it reverses positive pre-pattern momentum
            if pre_momentum > 0 and pattern_momentum < 0:
                momentum_shift = pre_momentum + abs(pattern_momentum)
            else:
                momentum_shift = abs(pattern_momentum)
        
        return min(1.0, momentum_shift * 5)  # Scale to 0-1 range    
    def _calculate_reliability_score(self, pattern_data: Dict, 
                                   volume_confirmation: bool,
                                   trend_reversal_strength: float,
                                   institutional_participation: float,
                                   momentum_shift: float) -> float:
        """Calculate overall pattern reliability score"""
        # Base score from pattern quality
        first_strength = pattern_data['first_candle_strength']
        second_strength = pattern_data['second_candle_strength']
        gap_size = pattern_data['gap_size']
        
        pattern_quality = (first_strength + second_strength) / 2
        gap_bonus = min(0.3, gap_size * 100)  # Bonus for larger gaps
        
        base_score = pattern_quality + gap_bonus
        
        # Volume confirmation bonus
        volume_bonus = 0.2 if volume_confirmation else 0
        
        # Trend reversal bonus
        reversal_bonus = trend_reversal_strength * 0.25
        
        # Institutional participation bonus
        institutional_bonus = institutional_participation * 0.15
        
        # Momentum shift bonus
        momentum_bonus = momentum_shift * 0.1
        
        total_score = (base_score * 0.4 + volume_bonus + reversal_bonus + 
                      institutional_bonus + momentum_bonus)
        
        return min(1.0, total_score)
    
    def _prepare_ml_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Prepare features for ML model"""
        if index < 20:
            return np.array([])
        
        features = []
        
        # Pattern structure features
        first_candle = data.iloc[index - 1]
        second_candle = data.iloc[index]
        
        _, _, first_strength = self._is_marubozu(first_candle)
        _, _, second_strength = self._is_marubozu(second_candle)
        _, gap_size, _ = self._detect_gap(first_candle, second_candle)
        
        features.extend([first_strength, second_strength, gap_size])
        
        # Volume features
        if 'volume' in data.columns:
            volumes = data['volume']
            vol_ma = volumes.rolling(window=20).mean()
            recent_vol_ratio = volumes.iloc[index] / vol_ma.iloc[index] if vol_ma.iloc[index] > 0 else 1
            features.append(recent_vol_ratio)
        else:
            features.append(1.0)
        
        # Trend features
        recent_data = data.iloc[index-19:index-1]
        close_prices = recent_data['close']
        
        if len(close_prices) >= 10:
            x = np.arange(len(close_prices))
            slope, _, r_value, _, _ = stats.linregress(x, close_prices)
            features.extend([slope, r_value ** 2])
        else:
            features.extend([0, 0])
        
        # Momentum features
        if len(close_prices) >= 5:
            momentum = (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5]
            features.append(momentum)
        else:
            features.append(0)
        
        return np.array(features)    
    def _train_ml_model(self, data: pd.DataFrame):
        """Train ML model for pattern validation"""
        if len(data) < self.ml_training_periods:
            return
        
        features_list = []
        labels_list = []
        
        for i in range(21, len(data) - 10):
            # Check if there's a kicking pattern at this position
            pattern_type, pattern_data = self._identify_kicking_pattern(data, i)
            
            if not pattern_type:
                continue
            
            features = self._prepare_ml_features(data, i)
            if len(features) == 0:
                continue
            
            # Create label based on future price movement
            current_close = data.iloc[i]['close']
            future_data = data.iloc[i+1:i+11]  # Next 10 periods
            
            if pattern_type == 'bullish_kicking':
                # Check for upward movement
                future_high = future_data['high'].max()
                price_move = (future_high - current_close) / current_close
            else:
                # Check for downward movement
                future_low = future_data['low'].min()
                price_move = (current_close - future_low) / current_close
            
            # Label as successful if price moves > 2% in expected direction
            label = 1 if price_move > 0.02 else 0
            
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
        """Determine trading signal based on kicking pattern type"""
        if pattern_type == 'bullish_kicking':
            return SignalType.BUY
        elif pattern_type == 'bearish_kicking':
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _calculate_trading_levels(self, data: pd.DataFrame,
                                current_price: float,
                                pattern_type: str,
                                gap_size: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        atr_values = self._calculate_atr(data)
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0
        
        if current_atr == 0:
            # Fallback to percentage levels
            if pattern_type == 'bullish_kicking':
                stop_loss = current_price * 0.96
                take_profit = current_price * 1.06
            else:
                stop_loss = current_price * 1.04
                take_profit = current_price * 0.94
        else:
            # ATR-based levels with gap consideration
            gap_adjustment = max(1.0, gap_size * 50)  # Adjust for gap size
            
            if pattern_type == 'bullish_kicking':
                stop_loss = current_price - (current_atr * 2.0 * gap_adjustment)
                take_profit = current_price + (current_atr * 3.0)
            else:
                stop_loss = current_price + (current_atr * 2.0 * gap_adjustment)
                take_profit = current_price - (current_atr * 3.0)
        
        return stop_loss, take_profit
    
    def update(self, data: pd.DataFrame) -> Optional[KickingSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            KickingSignal if pattern detected, None otherwise
        """
        if len(data) < max(self.atr_periods, self.trend_reversal_periods, 20):
            return None
        
        # Train ML model if enough data
        if len(data) >= self.ml_training_periods and self.ml_model is not None:
            self._train_ml_model(data)
        
        # Check for kicking pattern
        current_index = len(data) - 1
        pattern_type, pattern_data = self._identify_kicking_pattern(data, current_index)
        
        if not pattern_type:
            return None        
        # Analyze pattern characteristics
        volume_confirmation = self._analyze_volume_confirmation(data, pattern_data)
        trend_reversal_strength = self._calculate_trend_reversal_strength(data, pattern_type)
        institutional_participation = self._calculate_institutional_participation(data)
        momentum_shift = self._calculate_momentum_shift(data, pattern_type)
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(
            pattern_data, volume_confirmation, trend_reversal_strength,
            institutional_participation, momentum_shift
        )
        
        # Get ML confidence
        ml_confidence = self._get_ml_confidence(data, current_index)
        
        # Calculate overall confidence
        confidence_level = (reliability_score * 0.7 + ml_confidence * 0.3)
        
        # Determine signal
        signal_type = self._determine_signal_type(pattern_type)
        
        # Calculate trading levels
        current_price = data.iloc[-1]['close']
        stop_loss, take_profit = self._calculate_trading_levels(
            data, current_price, pattern_type, pattern_data['gap_size']
        )
        
        signal = KickingSignal(
            timestamp=data.index[-1],
            signal_type=signal_type,
            pattern_type=pattern_type,
            gap_size=pattern_data['gap_size'],
            first_candle_strength=pattern_data['first_candle_strength'],
            second_candle_strength=pattern_data['second_candle_strength'],
            volume_confirmation=volume_confirmation,
            trend_reversal_strength=trend_reversal_strength,
            institutional_participation=institutional_participation,
            momentum_shift=momentum_shift,
            reliability_score=reliability_score,
            confidence_level=confidence_level,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.signals.append(signal)
        return signal
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[KickingSignal]:
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
            'avg_reliability_score': np.mean([s.reliability_score for s in self.signals]),
            'avg_confidence_level': np.mean([s.confidence_level for s in self.signals]),
            'avg_trend_reversal_strength': np.mean([s.trend_reversal_strength for s in self.signals]),
            'avg_institutional_participation': np.mean([s.institutional_participation for s in self.signals]),
            'pattern_type_distribution': {
                ptype: pattern_types.count(ptype) / len(pattern_types)
                for ptype in set(pattern_types)
            },
            'volume_confirmation_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals),
            'high_reliability_signals': len([s for s in self.signals if s.reliability_score > 0.8]),
            'high_confidence_signals': len([s for s in self.signals if s.confidence_level > 0.7]),
            'institutional_signals': len([s for s in self.signals if s.institutional_participation > 0.6])
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()
        self.ml_model = None
        self.scaler = StandardScaler()
        self._initialize_ml_model()