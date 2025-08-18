"""
Rising Three Methods Pattern Indicator

The Rising Three Methods is a five-candle bullish continuation pattern that appears during uptrends.

Pattern Recognition:
1. First candle: Long bullish candle (white/green)
2. Next three candles: Small bearish candles (black/red) that:
   - Don't close below the open of the first candle
   - Don't close above the close of the first candle
   - Show pullback/consolidation within the first candle's range
3. Fifth candle: Bullish candle that:
   - Opens above the close of the fourth candle
   - Closes above the close of the first candle
   - Confirms continuation of the uptrend

Trading Significance:
- Indicates temporary consolidation in an uptrend
- The three small bearish candles represent profit-taking
- The final bullish candle confirms trend continuation
- High reliability when appearing after strong upward momentum
- Volume should increase on the final candle
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class RisingThreeMethodsSignal:
    """Signal data for Rising Three Methods pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_type: str  # 'rising_three_methods'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    base_candle: Dict
    consolidation_candles: List[Dict]
    breakout_candle: Dict
    volume_confirmation: bool
    trend_strength: float
    reliability_score: float


class RisingThreeMethodsIndicator(BasePatternIndicator):
    """
    Rising Three Methods Pattern Recognition Indicator
    
    Identifies five-candle bullish continuation patterns during uptrends.
    """
    
    def __init__(self, 
                 min_base_body_ratio: float = 0.002,
                 max_consolidation_body_ratio: float = 0.001,
                 volume_threshold: float = 1.3,
                 atr_periods: int = 14,
                 trend_periods: int = 10):
        """
        Initialize Rising Three Methods indicator
        
        Args:
            min_base_body_ratio: Minimum body size for base candle as % of price
            max_consolidation_body_ratio: Maximum body size for consolidation candles as % of price
            volume_threshold: Volume multiplier for confirmation
            atr_periods: Periods for ATR calculation
            trend_periods: Periods to analyze trend strength
        """
        super().__init__()
        self.min_base_body_ratio = min_base_body_ratio
        self.max_consolidation_body_ratio = max_consolidation_body_ratio
        self.volume_threshold = volume_threshold
        self.atr_periods = atr_periods
        self.trend_periods = trend_periods
        self.signals: List[RisingThreeMethodsSignal] = []
        
    def _calculate_atr(self, data: pd.DataFrame, periods: int) -> np.ndarray:
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
        atr[periods:] = pd.Series(tr_array).rolling(window=periods-1).mean().values[periods-1:]
        
        return atr
    
    def _calculate_ema(self, data: pd.Series, periods: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=periods).mean()
    
    def _is_bullish_candle(self, candle: Dict) -> bool:
        """Check if candle is bullish"""
        return candle['close'] > candle['open']
    
    def _is_bearish_candle(self, candle: Dict) -> bool:
        """Check if candle is bearish"""
        return candle['close'] < candle['open']
    
    def _get_candle_body_size(self, candle: Dict) -> float:
        """Calculate candle body size as percentage of price"""
        return abs(candle['close'] - candle['open']) / candle['open']
    
    def _get_candle_range(self, candle: Dict) -> float:
        """Calculate candle high-low range"""
        return candle['high'] - candle['low']
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength over recent periods"""
        if len(data) < self.trend_periods:
            return 0.0
        
        closes = data['close'].tail(self.trend_periods)
        ema_short = self._calculate_ema(closes, 5).iloc[-1]
        ema_long = self._calculate_ema(closes, self.trend_periods).iloc[-1]
        
        if ema_long == 0:
            return 0.0
        
        # Trend strength based on EMA slope and distance
        trend_strength = (ema_short - ema_long) / ema_long
        
        # Additional confirmation from price action
        price_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
        
        # Combine both measures
        combined_strength = (trend_strength + price_change) / 2
        
        return max(0.0, min(1.0, combined_strength * 10))  # Scale to 0-1
    
    def _check_base_candle(self, candle: Dict) -> Tuple[bool, float]:
        """
        Check if candle qualifies as a strong base candle
        
        Returns:
            Tuple of (is_valid, strength)
        """
        if not self._is_bullish_candle(candle):
            return False, 0.0
        
        body_ratio = self._get_candle_body_size(candle)
        if body_ratio < self.min_base_body_ratio:
            return False, 0.0
        
        # Check for minimal upper shadow (strong close near high)
        upper_shadow = candle['high'] - candle['close']
        total_range = self._get_candle_range(candle)
        
        if total_range > 0:
            shadow_ratio = upper_shadow / total_range
            strength = 1.0 - shadow_ratio  # Less shadow = stronger
        else:
            strength = 0.5
        
        # Bonus for large body
        strength *= (1 + body_ratio)
        
        return True, max(0.1, min(1.0, strength))
    
    def _check_consolidation_candles(self, candles: List[Dict], base_candle: Dict) -> Tuple[bool, float]:
        """
        Check if three candles form proper consolidation
        
        Args:
            candles: List of three consolidation candles
            base_candle: The base bullish candle
            
        Returns:
            Tuple of (is_valid, quality_score)
        """
        if len(candles) != 3:
            return False, 0.0
        
        base_open = base_candle['open']
        base_close = base_candle['close']
        base_high = base_candle['high']
        base_low = base_candle['low']
        
        quality_scores = []
        
        for candle in candles:
            # Preferably bearish, but small bullish candles can be acceptable
            if self._is_bearish_candle(candle):
                direction_score = 1.0
            elif self._is_bullish_candle(candle):
                # Small bullish candles are acceptable but less ideal
                body_ratio = self._get_candle_body_size(candle)
                if body_ratio <= self.max_consolidation_body_ratio:
                    direction_score = 0.7
                else:
                    return False, 0.0
            else:
                # Doji candles are acceptable
                direction_score = 0.8
            
            # Check body size constraint
            body_ratio = self._get_candle_body_size(candle)
            if body_ratio > self.max_consolidation_body_ratio:
                return False, 0.0
            
            # Must stay within base candle range
            if candle['close'] > base_close or candle['close'] < base_open:
                return False, 0.0
            
            # Check if candle stays mostly within base range
            if candle['high'] > base_high or candle['low'] < base_low:
                range_score = 0.5  # Partial penalty for exceeding base range
            else:
                range_score = 1.0
            
            # Calculate position within base range (closer to middle is better)
            base_range = base_close - base_open
            if base_range > 0:
                candle_position = (candle['close'] - base_open) / base_range
                position_score = 1.0 - abs(candle_position - 0.5) * 2  # Best at 50% level
            else:
                position_score = 0.5
            
            candle_score = direction_score * 0.4 + range_score * 0.4 + position_score * 0.2
            quality_scores.append(candle_score)
        
        # Overall quality is average of individual candle scores
        overall_quality = np.mean(quality_scores)
        
        # Bonus for decreasing volume during consolidation (if available)
        volumes = [c.get('volume', 0) for c in candles]
        if all(v > 0 for v in volumes) and len(set(volumes)) > 1:
            if volumes[2] < volumes[0]:  # Volume decreases
                overall_quality *= 1.1
        
        return True, max(0.1, min(1.0, overall_quality))
    
    def _check_breakout_candle(self, candle: Dict, base_candle: Dict, 
                              last_consolidation: Dict) -> Tuple[bool, float]:
        """
        Check if candle provides valid breakout confirmation
        
        Returns:
            Tuple of (is_valid, strength)
        """
        if not self._is_bullish_candle(candle):
            return False, 0.0
        
        # Must open above last consolidation candle's close
        if candle['open'] <= last_consolidation['close']:
            return False, 0.0
        
        # Must close above base candle's close
        if candle['close'] <= base_candle['close']:
            return False, 0.0
        
        # Calculate breakout strength
        base_close = base_candle['close']
        breakout_distance = candle['close'] - base_close
        base_range = base_candle['close'] - base_candle['open']
        
        if base_range <= 0:
            return False, 0.0
        
        # Strength based on how much it exceeds the base candle
        strength = breakout_distance / base_range
        
        # Check body size of breakout candle
        body_ratio = self._get_candle_body_size(candle)
        if body_ratio >= self.min_base_body_ratio:
            strength *= 1.2  # Bonus for strong breakout candle
        
        # Check for minimal upper shadow
        upper_shadow = candle['high'] - candle['close']
        total_range = self._get_candle_range(candle)
        
        if total_range > 0:
            shadow_ratio = upper_shadow / total_range
            strength *= (1.0 - shadow_ratio * 0.5)  # Penalty for large upper shadow
        
        return True, max(0.1, min(2.0, strength))
    
    def _check_volume_confirmation(self, candles: List[Dict], avg_volume: float) -> bool:
        """Check volume pattern for confirmation"""
        if len(candles) != 5:
            return False
        
        base_volume = candles[0].get('volume', 0)
        breakout_volume = candles[4].get('volume', 0)
        
        # Base candle should have good volume
        base_volume_ok = base_volume >= avg_volume
        
        # Breakout candle should have elevated volume
        breakout_volume_ok = breakout_volume >= (avg_volume * self.volume_threshold)
        
        return base_volume_ok and breakout_volume_ok
    
    def _calculate_stop_loss(self, candles: List[Dict], atr: float) -> float:
        """Calculate stop loss level"""
        # Stop below the lowest low of the consolidation candles
        consolidation_lows = [candles[i]['low'] for i in range(1, 4)]
        consolidation_low = min(consolidation_lows)
        
        return consolidation_low - (atr * 1.5)
    
    def _calculate_take_profit(self, entry_price: float, stop_loss: float,
                             base_candle: Dict, risk_reward: float = 2.5) -> float:
        """Calculate take profit level"""
        risk = abs(entry_price - stop_loss)
        
        # Base target from risk-reward
        base_target = entry_price + (risk * risk_reward)
        
        # Enhanced target based on base candle size
        base_range = base_candle['close'] - base_candle['open']
        enhanced_target = entry_price + (base_range * 2)
        
        # Use the more conservative target
        return min(base_target, enhanced_target)
    
    def _calculate_reliability_score(self, base_strength: float, consolidation_quality: float,
                                   breakout_strength: float, volume_confirmation: bool,
                                   trend_strength: float) -> float:
        """Calculate overall pattern reliability score"""
        # Weight different components
        score = (
            base_strength * 0.25 +
            consolidation_quality * 0.25 +
            breakout_strength * 0.25 +
            trend_strength * 0.15
        )
        
        # Volume confirmation bonus
        if volume_confirmation:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _check_rising_three_methods_pattern(self, candles: List[Dict], 
                                          trend_strength: float) -> Tuple[bool, Dict]:
        """
        Check for complete Rising Three Methods pattern
        
        Returns:
            Tuple of (is_pattern, pattern_data)
        """
        if len(candles) != 5:
            return False, {}
        
        base_candle = candles[0]
        consolidation_candles = candles[1:4]
        breakout_candle = candles[4]
        
        # Check base candle
        is_valid_base, base_strength = self._check_base_candle(base_candle)
        if not is_valid_base:
            return False, {}
        
        # Check consolidation phase
        is_valid_consolidation, consolidation_quality = self._check_consolidation_candles(
            consolidation_candles, base_candle
        )
        if not is_valid_consolidation:
            return False, {}
        
        # Check breakout candle
        is_valid_breakout, breakout_strength = self._check_breakout_candle(
            breakout_candle, base_candle, consolidation_candles[2]
        )
        if not is_valid_breakout:
            return False, {}
        
        # Ensure we're in an uptrend context
        if trend_strength < 0.2:
            return False, {}
        
        pattern_data = {
            'base_strength': base_strength,
            'consolidation_quality': consolidation_quality,
            'breakout_strength': breakout_strength,
            'overall_strength': (base_strength + consolidation_quality + breakout_strength) / 3
        }
        
        return True, pattern_data
    
    def update(self, data: pd.DataFrame) -> Optional[RisingThreeMethodsSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            RisingThreeMethodsSignal if pattern detected, None otherwise
        """
        if len(data) < 5 + max(self.atr_periods, self.trend_periods):
            return None
        
        # Calculate ATR for stop loss calculation
        atr_values = self._calculate_atr(data, self.atr_periods)
        current_atr = atr_values[-1]
        
        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(data)
        
        # Get recent candles for pattern analysis
        recent_data = data.tail(5)
        candles = []
        
        for _, row in recent_data.iterrows():
            candle = {
                'timestamp': row.name,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row.get('volume', 0)
            }
            candles.append(candle)
        
        # Check for Rising Three Methods pattern
        is_pattern, pattern_data = self._check_rising_three_methods_pattern(candles, trend_strength)
        
        if not is_pattern:
            return None
        
        # Calculate average volume for confirmation
        volume_data = data['volume'].tail(20) if 'volume' in data.columns else pd.Series([0] * 20)
        avg_volume = volume_data.mean()
        
        volume_confirmation = self._check_volume_confirmation(candles, avg_volume)
        
        # Calculate trade parameters
        entry_price = candles[4]['close']
        stop_loss = self._calculate_stop_loss(candles, current_atr)
        take_profit = self._calculate_take_profit(entry_price, stop_loss, candles[0])
        
        # Calculate reliability score
        reliability = self._calculate_reliability_score(
            pattern_data['base_strength'],
            pattern_data['consolidation_quality'],
            pattern_data['breakout_strength'],
            volume_confirmation,
            trend_strength
        )
        
        signal = RisingThreeMethodsSignal(
            timestamp=candles[4]['timestamp'],
            signal_type=SignalType.BUY,
            pattern_type='rising_three_methods',
            strength=pattern_data['overall_strength'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            base_candle=candles[0],
            consolidation_candles=candles[1:4],
            breakout_candle=candles[4],
            volume_confirmation=volume_confirmation,
            trend_strength=trend_strength,
            reliability_score=reliability
        )
        
        self.signals.append(signal)
        return signal
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[RisingThreeMethodsSignal]:
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
        
        return {
            'total_patterns': len(self.signals),
            'avg_strength': np.mean([s.strength for s in self.signals]),
            'avg_reliability': np.mean([s.reliability_score for s in self.signals]),
            'avg_trend_strength': np.mean([s.trend_strength for s in self.signals]),
            'volume_confirmed_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals),
            'strong_trend_rate': len([s for s in self.signals if s.trend_strength > 0.5]) / len(self.signals)
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()
