"""
Belt Hold Pattern Indicator

The Belt Hold (also known as Yorikiri in Japanese) is a single-candle reversal pattern.

Pattern Recognition:
1. Bullish Belt Hold (Belt Hold Line):
   - Single bullish candle in a downtrend
   - Opens at or near the low of the day
   - Closes near the high with minimal upper shadow
   - Long body with little to no lower shadow
   - Shows strong buying pressure from the open

2. Bearish Belt Hold:
   - Single bearish candle in an uptrend
   - Opens at or near the high of the day
   - Closes near the low with minimal lower shadow
   - Long body with little to no upper shadow
   - Shows strong selling pressure from the open

Trading Significance:
- Indicates immediate momentum shift at trend extremes
- The opening at extreme suggests gap or strong momentum
- Minimal shadows show sustained directional pressure
- More reliable when appearing after extended trends
- Volume confirmation enhances pattern reliability
- Longer body indicates stronger reversal potential
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class BeltHoldSignal:
    """Signal data for Belt Hold pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_type: str  # 'bullish_belt_hold' or 'bearish_belt_hold'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    belt_candle: Dict
    body_ratio: float
    shadow_ratio: float
    gap_analysis: Dict
    volume_confirmation: bool
    trend_context: str
    reliability_score: float


class BeltHoldIndicator(BasePatternIndicator):
    """
    Belt Hold Pattern Recognition Indicator
    
    Identifies single-candle reversal patterns with minimal shadows and strong directional pressure.
    """
    
    def __init__(self, 
                 min_body_ratio: float = 0.003,
                 max_shadow_ratio: float = 0.15,
                 min_range_ratio: float = 0.002,
                 volume_threshold: float = 1.2,
                 atr_periods: int = 14,
                 trend_periods: int = 20):
        """
        Initialize Belt Hold indicator
        
        Args:
            min_body_ratio: Minimum body size as percentage of price
            max_shadow_ratio: Maximum shadow as percentage of total range
            min_range_ratio: Minimum candle range as percentage of price
            volume_threshold: Volume multiplier for confirmation
            atr_periods: Periods for ATR calculation
            trend_periods: Periods to determine trend context
        """
        super().__init__()
        self.min_body_ratio = min_body_ratio
        self.max_shadow_ratio = max_shadow_ratio
        self.min_range_ratio = min_range_ratio
        self.volume_threshold = volume_threshold
        self.atr_periods = atr_periods
        self.trend_periods = trend_periods
        self.signals: List[BeltHoldSignal] = []
        
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
    
    def _get_candle_range_ratio(self, candle: Dict) -> float:
        """Calculate candle range as percentage of price"""
        range_size = self._get_candle_range(candle)
        return range_size / candle['open'] if candle['open'] > 0 else 0
    
    def _determine_trend_context(self, data: pd.DataFrame) -> str:
        """Determine if we're in uptrend, downtrend, or sideways"""
        if len(data) < self.trend_periods:
            return 'unknown'
        
        closes = data['close'].tail(self.trend_periods)
        ema_short = self._calculate_ema(closes, 8).iloc[-1]
        ema_long = self._calculate_ema(closes, self.trend_periods).iloc[-1]
        
        # Price trend analysis
        price_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
        
        if ema_short > ema_long and price_change > 0.02:
            return 'uptrend'
        elif ema_short < ema_long and price_change < -0.02:
            return 'downtrend'
        else:
            return 'sideways'    
    def _analyze_gap_context(self, current_candle: Dict, previous_candle: Dict) -> Dict:
        """Analyze gap context for belt hold pattern"""
        gap_up = current_candle['open'] - previous_candle['high']
        gap_down = previous_candle['low'] - current_candle['open']
        
        # Determine gap type and size
        if gap_up > 0:
            gap_type = 'gap_up'
            gap_size = gap_up / previous_candle['close']
        elif gap_down > 0:
            gap_type = 'gap_down'
            gap_size = gap_down / previous_candle['close']
        else:
            gap_type = 'no_gap'
            gap_size = 0.0
        
        return {
            'gap_type': gap_type,
            'gap_size': gap_size,
            'has_significant_gap': gap_size > 0.001
        }
    
    def _check_bullish_belt_hold(self, candle: Dict, trend_context: str, 
                                gap_analysis: Dict) -> Tuple[bool, float]:
        """
        Check for bullish belt hold pattern
        
        Returns:
            Tuple of (is_pattern, strength)
        """
        # Must be bullish candle
        if not self._is_bullish_candle(candle):
            return False, 0.0
        
        # Prefer downtrend or sideways context
        if trend_context == 'uptrend':
            return False, 0.0
        
        # Check body size
        body_ratio = self._get_candle_body_size(candle)
        if body_ratio < self.min_body_ratio:
            return False, 0.0
        
        # Check range size
        range_ratio = self._get_candle_range_ratio(candle)
        if range_ratio < self.min_range_ratio:
            return False, 0.0
        
        total_range = self._get_candle_range(candle)
        if total_range <= 0:
            return False, 0.0
        
        # Check that it opens near the low
        lower_shadow = candle['open'] - candle['low']
        lower_shadow_ratio = lower_shadow / total_range
        
        if lower_shadow_ratio > self.max_shadow_ratio:
            return False, 0.0
        
        # Check that it closes near the high
        upper_shadow = candle['high'] - candle['close']
        upper_shadow_ratio = upper_shadow / total_range
        
        if upper_shadow_ratio > self.max_shadow_ratio:
            return False, 0.0
        
        # Calculate pattern strength
        strength = body_ratio * 0.4  # Body size importance
        strength += (1.0 - lower_shadow_ratio) * 0.3  # Proximity to low
        strength += (1.0 - upper_shadow_ratio) * 0.2  # Proximity to high
        
        # Bonus for trend context
        if trend_context == 'downtrend':
            strength *= 1.3
        elif trend_context == 'sideways':
            strength *= 1.1
        
        # Gap bonus
        if gap_analysis['gap_type'] == 'gap_down' and gap_analysis['has_significant_gap']:
            strength *= 1.2
        
        return True, max(0.1, min(2.0, strength))
    
    def _check_bearish_belt_hold(self, candle: Dict, trend_context: str,
                               gap_analysis: Dict) -> Tuple[bool, float]:
        """
        Check for bearish belt hold pattern
        
        Returns:
            Tuple of (is_pattern, strength)
        """
        # Must be bearish candle
        if not self._is_bearish_candle(candle):
            return False, 0.0
        
        # Prefer uptrend or sideways context
        if trend_context == 'downtrend':
            return False, 0.0
        
        # Check body size
        body_ratio = self._get_candle_body_size(candle)
        if body_ratio < self.min_body_ratio:
            return False, 0.0
        
        # Check range size
        range_ratio = self._get_candle_range_ratio(candle)
        if range_ratio < self.min_range_ratio:
            return False, 0.0
        
        total_range = self._get_candle_range(candle)
        if total_range <= 0:
            return False, 0.0
        
        # Check that it opens near the high
        upper_shadow = candle['high'] - candle['open']
        upper_shadow_ratio = upper_shadow / total_range
        
        if upper_shadow_ratio > self.max_shadow_ratio:
            return False, 0.0
        
        # Check that it closes near the low
        lower_shadow = candle['close'] - candle['low']
        lower_shadow_ratio = lower_shadow / total_range
        
        if lower_shadow_ratio > self.max_shadow_ratio:
            return False, 0.0
        
        # Calculate pattern strength
        strength = body_ratio * 0.4  # Body size importance
        strength += (1.0 - upper_shadow_ratio) * 0.3  # Proximity to high
        strength += (1.0 - lower_shadow_ratio) * 0.2  # Proximity to low
        
        # Bonus for trend context
        if trend_context == 'uptrend':
            strength *= 1.3
        elif trend_context == 'sideways':
            strength *= 1.1
        
        # Gap bonus
        if gap_analysis['gap_type'] == 'gap_up' and gap_analysis['has_significant_gap']:
            strength *= 1.2
        
        return True, max(0.1, min(2.0, strength))    
    def _check_volume_confirmation(self, candle: Dict, avg_volume: float) -> bool:
        """Check volume confirmation for the pattern"""
        candle_volume = candle.get('volume', 0)
        return candle_volume >= (avg_volume * self.volume_threshold)
    
    def _calculate_stop_loss(self, pattern_type: str, candle: Dict, atr: float) -> float:
        """Calculate stop loss level"""
        if pattern_type == 'bullish_belt_hold':
            # Stop below the low of the belt hold candle
            return candle['low'] - (atr * 1.0)
        else:  # bearish_belt_hold
            # Stop above the high of the belt hold candle
            return candle['high'] + (atr * 1.0)
    
    def _calculate_take_profit(self, pattern_type: str, entry_price: float, 
                             stop_loss: float, risk_reward: float = 2.0) -> float:
        """Calculate take profit level"""
        risk = abs(entry_price - stop_loss)
        
        if pattern_type == 'bullish_belt_hold':
            return entry_price + (risk * risk_reward)
        else:  # bearish_belt_hold
            return entry_price - (risk * risk_reward)
    
    def _calculate_shadow_ratio(self, candle: Dict, shadow_type: str) -> float:
        """Calculate shadow ratio for the candle"""
        total_range = self._get_candle_range(candle)
        if total_range <= 0:
            return 0.0
        
        if shadow_type == 'upper':
            if self._is_bullish_candle(candle):
                shadow = candle['high'] - candle['close']
            else:
                shadow = candle['high'] - candle['open']
        else:  # lower
            if self._is_bullish_candle(candle):
                shadow = candle['open'] - candle['low']
            else:
                shadow = candle['close'] - candle['low']
        
        return shadow / total_range
    
    def _calculate_reliability_score(self, pattern_type: str, strength: float,
                                   body_ratio: float, shadow_ratio: float,
                                   volume_confirmation: bool, gap_analysis: Dict,
                                   trend_context: str) -> float:
        """Calculate pattern reliability score"""
        score = strength * 0.4
        
        # Body size adds reliability
        score += min(body_ratio * 50, 0.2)
        
        # Low shadow ratio adds reliability
        score += (1.0 - shadow_ratio) * 0.2
        
        # Volume confirmation adds reliability
        if volume_confirmation:
            score += 0.2
        
        # Gap context adds reliability
        if gap_analysis['has_significant_gap']:
            score += 0.1
        
        # Trend context alignment
        if ((pattern_type == 'bullish_belt_hold' and trend_context == 'downtrend') or
            (pattern_type == 'bearish_belt_hold' and trend_context == 'uptrend')):
            score += 0.1
        
        return max(0.0, min(1.0, score))    
    def update(self, data: pd.DataFrame) -> Optional[BeltHoldSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            BeltHoldSignal if pattern detected, None otherwise
        """
        if len(data) < 2 + max(self.atr_periods, self.trend_periods):
            return None
        
        # Calculate ATR for stop loss calculation
        atr_values = self._calculate_atr(data, self.atr_periods)
        current_atr = atr_values[-1]
        
        # Determine trend context
        trend_context = self._determine_trend_context(data)
        
        # Get current and previous candles
        current_row = data.iloc[-1]
        previous_row = data.iloc[-2]
        
        current_candle = {
            'timestamp': current_row.name,
            'open': current_row['open'],
            'high': current_row['high'],
            'low': current_row['low'],
            'close': current_row['close'],
            'volume': current_row.get('volume', 0)
        }
        
        previous_candle = {
            'open': previous_row['open'],
            'high': previous_row['high'],
            'low': previous_row['low'],
            'close': previous_row['close']
        }
        
        # Analyze gap context
        gap_analysis = self._analyze_gap_context(current_candle, previous_candle)
        
        # Calculate average volume for confirmation
        volume_data = data['volume'].tail(20) if 'volume' in data.columns else pd.Series([0] * 20)
        avg_volume = volume_data.mean()
        
        # Check for bullish belt hold
        is_bullish, bullish_strength = self._check_bullish_belt_hold(current_candle, trend_context, gap_analysis)
        
        if is_bullish:
            volume_confirmation = self._check_volume_confirmation(current_candle, avg_volume)
            body_ratio = self._get_candle_body_size(current_candle)
            shadow_ratio = max(
                self._calculate_shadow_ratio(current_candle, 'upper'),
                self._calculate_shadow_ratio(current_candle, 'lower')
            )
            
            entry_price = current_candle['close']
            stop_loss = self._calculate_stop_loss('bullish_belt_hold', current_candle, current_atr)
            take_profit = self._calculate_take_profit('bullish_belt_hold', entry_price, stop_loss)
            
            reliability = self._calculate_reliability_score(
                'bullish_belt_hold', bullish_strength, body_ratio, shadow_ratio,
                volume_confirmation, gap_analysis, trend_context
            )
            
            signal = BeltHoldSignal(
                timestamp=current_candle['timestamp'],
                signal_type=SignalType.BUY,
                pattern_type='bullish_belt_hold',
                strength=bullish_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                belt_candle=current_candle,
                body_ratio=body_ratio,
                shadow_ratio=shadow_ratio,
                gap_analysis=gap_analysis,
                volume_confirmation=volume_confirmation,
                trend_context=trend_context,
                reliability_score=reliability
            )
            
            self.signals.append(signal)
            return signal
        
        # Check for bearish belt hold
        is_bearish, bearish_strength = self._check_bearish_belt_hold(current_candle, trend_context, gap_analysis)
        
        if is_bearish:
            volume_confirmation = self._check_volume_confirmation(current_candle, avg_volume)
            body_ratio = self._get_candle_body_size(current_candle)
            shadow_ratio = max(
                self._calculate_shadow_ratio(current_candle, 'upper'),
                self._calculate_shadow_ratio(current_candle, 'lower')
            )
            
            entry_price = current_candle['close']
            stop_loss = self._calculate_stop_loss('bearish_belt_hold', current_candle, current_atr)
            take_profit = self._calculate_take_profit('bearish_belt_hold', entry_price, stop_loss)
            
            reliability = self._calculate_reliability_score(
                'bearish_belt_hold', bearish_strength, body_ratio, shadow_ratio,
                volume_confirmation, gap_analysis, trend_context
            )
            
            signal = BeltHoldSignal(
                timestamp=current_candle['timestamp'],
                signal_type=SignalType.SELL,
                pattern_type='bearish_belt_hold',
                strength=bearish_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                belt_candle=current_candle,
                body_ratio=body_ratio,
                shadow_ratio=shadow_ratio,
                gap_analysis=gap_analysis,
                volume_confirmation=volume_confirmation,
                trend_context=trend_context,
                reliability_score=reliability
            )
            
            self.signals.append(signal)
            return signal
        
        return None    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[BeltHoldSignal]:
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
        
        bullish_signals = [s for s in self.signals if s.pattern_type == 'bullish_belt_hold']
        bearish_signals = [s for s in self.signals if s.pattern_type == 'bearish_belt_hold']
        
        return {
            'total_patterns': len(self.signals),
            'bullish_belt_holds': len(bullish_signals),
            'bearish_belt_holds': len(bearish_signals),
            'avg_bullish_strength': np.mean([s.strength for s in bullish_signals]) if bullish_signals else 0,
            'avg_bearish_strength': np.mean([s.strength for s in bearish_signals]) if bearish_signals else 0,
            'avg_body_ratio': np.mean([s.body_ratio for s in self.signals]),
            'avg_shadow_ratio': np.mean([s.shadow_ratio for s in self.signals]),
            'avg_reliability': np.mean([s.reliability_score for s in self.signals]),
            'volume_confirmed_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals),
            'gap_occurrence_rate': len([s for s in self.signals if s.gap_analysis['has_significant_gap']]) / len(self.signals)
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()