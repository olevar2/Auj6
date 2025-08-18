"""
Three Line Strike Pattern Indicator

The Three Line Strike is a four-candle reversal pattern that can signal both bullish and bearish reversals.

Pattern Recognition:
1. Bullish Three Line Strike:
   - Three consecutive bearish candles (declining pattern)
   - Fourth candle opens below the close of the third candle
   - Fourth candle closes above the open of the first candle
   - The fourth candle "strikes through" the entire three-candle decline

2. Bearish Three Line Strike:
   - Three consecutive bullish candles (rising pattern)
   - Fourth candle opens above the close of the third candle
   - Fourth candle closes below the open of the first candle
   - The fourth candle "strikes through" the entire three-candle rise

Trading Significance:
- Indicates strong momentum reversal
- The "strike" candle shows rejection of the prior trend
- Often marks beginning of new trend direction
- High reliability when combined with volume confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class ThreeLineStrikeSignal:
    """Signal data for Three Line Strike pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_type: str  # 'bullish_strike' or 'bearish_strike'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confirmation_candles: List[Dict]
    strike_candle: Dict
    volume_confirmation: bool
    reliability_score: float


class ThreeLineStrikeIndicator(BasePatternIndicator):
    """
    Three Line Strike Pattern Recognition Indicator
    
    Identifies four-candle reversal patterns where the final candle
    completely engulfs the prior three-candle trend.
    """
    
    def __init__(self, 
                 min_body_size: float = 0.001,
                 volume_threshold: float = 1.2,
                 atr_periods: int = 14,
                 confirmation_periods: int = 3):
        """
        Initialize Three Line Strike indicator
        
        Args:
            min_body_size: Minimum candle body size as percentage of price
            volume_threshold: Volume multiplier for confirmation
            atr_periods: Periods for ATR calculation
            confirmation_periods: Periods to wait for pattern confirmation
        """
        super().__init__()
        self.min_body_size = min_body_size
        self.volume_threshold = volume_threshold
        self.atr_periods = atr_periods
        self.confirmation_periods = confirmation_periods
        self.signals: List[ThreeLineStrikeSignal] = []
        
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
    
    def _is_bearish_candle(self, candle: Dict) -> bool:
        """Check if candle is bearish"""
        return candle['close'] < candle['open']
    
    def _is_bullish_candle(self, candle: Dict) -> bool:
        """Check if candle is bullish"""
        return candle['close'] > candle['open']
    
    def _get_candle_body_size(self, candle: Dict) -> float:
        """Calculate candle body size as percentage of price"""
        return abs(candle['close'] - candle['open']) / candle['open']
    
    def _get_candle_range(self, candle: Dict) -> float:
        """Calculate candle high-low range"""
        return candle['high'] - candle['low']
    
    def _has_sufficient_body(self, candle: Dict) -> bool:
        """Check if candle has sufficient body size"""
        return self._get_candle_body_size(candle) >= self.min_body_size
    
    def _check_three_consecutive_bearish(self, candles: List[Dict]) -> bool:
        """Check for three consecutive bearish candles with declining pattern"""
        if len(candles) != 3:
            return False
        
        # All must be bearish with sufficient body
        for candle in candles:
            if not self._is_bearish_candle(candle) or not self._has_sufficient_body(candle):
                return False
        
        # Each candle should close lower than the previous
        for i in range(1, 3):
            if candles[i]['close'] >= candles[i-1]['close']:
                return False
        
        return True
    
    def _check_three_consecutive_bullish(self, candles: List[Dict]) -> bool:
        """Check for three consecutive bullish candles with rising pattern"""
        if len(candles) != 3:
            return False
        
        # All must be bullish with sufficient body
        for candle in candles:
            if not self._is_bullish_candle(candle) or not self._has_sufficient_body(candle):
                return False
        
        # Each candle should close higher than the previous
        for i in range(1, 3):
            if candles[i]['close'] <= candles[i-1]['close']:
                return False
        
        return True
    
    def _check_bullish_strike_pattern(self, candles: List[Dict]) -> Tuple[bool, float]:
        """
        Check for bullish three line strike pattern
        
        Returns:
            Tuple of (is_pattern, strength)
        """
        if len(candles) != 4:
            return False, 0.0
        
        first_three = candles[:3]
        strike_candle = candles[3]
        
        # Check first three are consecutive bearish
        if not self._check_three_consecutive_bearish(first_three):
            return False, 0.0
        
        # Strike candle must be bullish
        if not self._is_bullish_candle(strike_candle):
            return False, 0.0
        
        # Strike candle should open below third candle's close
        if strike_candle['open'] >= candles[2]['close']:
            return False, 0.0
        
        # Strike candle must close above first candle's open
        if strike_candle['close'] <= candles[0]['open']:
            return False, 0.0
        
        # Calculate pattern strength
        decline_range = candles[0]['open'] - candles[2]['close']
        strike_recovery = strike_candle['close'] - strike_candle['open']
        total_recovery = strike_candle['close'] - candles[2]['close']
        
        if decline_range <= 0:
            return False, 0.0
        
        # Strength based on how much the strike candle recovers
        strength = min(total_recovery / decline_range, 2.0)
        
        # Bonus for large strike candle body
        strike_body_ratio = self._get_candle_body_size(strike_candle)
        strength *= (1 + strike_body_ratio)
        
        return True, max(0.1, min(1.0, strength))
    
    def _check_bearish_strike_pattern(self, candles: List[Dict]) -> Tuple[bool, float]:
        """
        Check for bearish three line strike pattern
        
        Returns:
            Tuple of (is_pattern, strength)
        """
        if len(candles) != 4:
            return False, 0.0
        
        first_three = candles[:3]
        strike_candle = candles[3]
        
        # Check first three are consecutive bullish
        if not self._check_three_consecutive_bullish(first_three):
            return False, 0.0
        
        # Strike candle must be bearish
        if not self._is_bearish_candle(strike_candle):
            return False, 0.0
        
        # Strike candle should open above third candle's close
        if strike_candle['open'] <= candles[2]['close']:
            return False, 0.0
        
        # Strike candle must close below first candle's open
        if strike_candle['close'] >= candles[0]['open']:
            return False, 0.0
        
        # Calculate pattern strength
        rise_range = candles[2]['close'] - candles[0]['open']
        strike_decline = strike_candle['open'] - strike_candle['close']
        total_decline = candles[2]['close'] - strike_candle['close']
        
        if rise_range <= 0:
            return False, 0.0
        
        # Strength based on how much the strike candle declines
        strength = min(total_decline / rise_range, 2.0)
        
        # Bonus for large strike candle body
        strike_body_ratio = self._get_candle_body_size(strike_candle)
        strength *= (1 + strike_body_ratio)
        
        return True, max(0.1, min(1.0, strength))
    
    def _check_volume_confirmation(self, candles: List[Dict], avg_volume: float) -> bool:
        """Check if strike candle has volume confirmation"""
        if len(candles) < 4:
            return False
        
        strike_volume = candles[3].get('volume', 0)
        return strike_volume >= (avg_volume * self.volume_threshold)
    
    def _calculate_stop_loss(self, pattern_type: str, candles: List[Dict], atr: float) -> float:
        """Calculate stop loss level"""
        if pattern_type == 'bullish_strike':
            # Stop below the lowest low of the pattern
            pattern_low = min(candle['low'] for candle in candles)
            return pattern_low - (atr * 1.5)
        else:  # bearish_strike
            # Stop above the highest high of the pattern
            pattern_high = max(candle['high'] for candle in candles)
            return pattern_high + (atr * 1.5)
    
    def _calculate_take_profit(self, pattern_type: str, entry_price: float, 
                             stop_loss: float, risk_reward: float = 2.0) -> float:
        """Calculate take profit level"""
        risk = abs(entry_price - stop_loss)
        
        if pattern_type == 'bullish_strike':
            return entry_price + (risk * risk_reward)
        else:  # bearish_strike
            return entry_price - (risk * risk_reward)
    
    def _calculate_reliability_score(self, pattern_type: str, strength: float,
                                   volume_confirmation: bool, candles: List[Dict]) -> float:
        """Calculate pattern reliability score"""
        score = strength * 0.4
        
        # Volume confirmation adds reliability
        if volume_confirmation:
            score += 0.3
        
        # Check for clean pattern (minimal wicks on strike candle)
        strike_candle = candles[3]
        body_size = abs(strike_candle['close'] - strike_candle['open'])
        total_range = strike_candle['high'] - strike_candle['low']
        
        if total_range > 0:
            body_to_range_ratio = body_size / total_range
            score += body_to_range_ratio * 0.2
        
        # Pattern consistency bonus
        if pattern_type == 'bullish_strike':
            # Check if the three bearish candles show consistent decline
            closes = [candle['close'] for candle in candles[:3]]
            if all(closes[i] < closes[i-1] for i in range(1, 3)):
                score += 0.1
        else:
            # Check if the three bullish candles show consistent rise
            closes = [candle['close'] for candle in candles[:3]]
            if all(closes[i] > closes[i-1] for i in range(1, 3)):
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def update(self, data: pd.DataFrame) -> Optional[ThreeLineStrikeSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            ThreeLineStrikeSignal if pattern detected, None otherwise
        """
        if len(data) < 4 + self.atr_periods:
            return None
        
        # Calculate ATR for stop loss calculation
        atr_values = self._calculate_atr(data, self.atr_periods)
        current_atr = atr_values[-1]
        
        # Get recent candles for pattern analysis
        recent_data = data.tail(4)
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
        
        # Calculate average volume for confirmation
        volume_data = data['volume'].tail(20) if 'volume' in data.columns else pd.Series([0] * 20)
        avg_volume = volume_data.mean()
        
        # Check for bullish strike pattern
        is_bullish_strike, bullish_strength = self._check_bullish_strike_pattern(candles)
        
        if is_bullish_strike:
            volume_conf = self._check_volume_confirmation(candles, avg_volume)
            entry_price = candles[3]['close']
            stop_loss = self._calculate_stop_loss('bullish_strike', candles, current_atr)
            take_profit = self._calculate_take_profit('bullish_strike', entry_price, stop_loss)
            reliability = self._calculate_reliability_score('bullish_strike', bullish_strength, 
                                                          volume_conf, candles)
            
            signal = ThreeLineStrikeSignal(
                timestamp=candles[3]['timestamp'],
                signal_type=SignalType.BUY,
                pattern_type='bullish_strike',
                strength=bullish_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confirmation_candles=candles[:3],
                strike_candle=candles[3],
                volume_confirmation=volume_conf,
                reliability_score=reliability
            )
            
            self.signals.append(signal)
            return signal
        
        # Check for bearish strike pattern
        is_bearish_strike, bearish_strength = self._check_bearish_strike_pattern(candles)
        
        if is_bearish_strike:
            volume_conf = self._check_volume_confirmation(candles, avg_volume)
            entry_price = candles[3]['close']
            stop_loss = self._calculate_stop_loss('bearish_strike', candles, current_atr)
            take_profit = self._calculate_take_profit('bearish_strike', entry_price, stop_loss)
            reliability = self._calculate_reliability_score('bearish_strike', bearish_strength,
                                                          volume_conf, candles)
            
            signal = ThreeLineStrikeSignal(
                timestamp=candles[3]['timestamp'],
                signal_type=SignalType.SELL,
                pattern_type='bearish_strike',
                strength=bearish_strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confirmation_candles=candles[:3],
                strike_candle=candles[3],
                volume_confirmation=volume_conf,
                reliability_score=reliability
            )
            
            self.signals.append(signal)
            return signal
        
        return None
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[ThreeLineStrikeSignal]:
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
        
        bullish_signals = [s for s in self.signals if s.pattern_type == 'bullish_strike']
        bearish_signals = [s for s in self.signals if s.pattern_type == 'bearish_strike']
        
        return {
            'total_patterns': len(self.signals),
            'bullish_strikes': len(bullish_signals),
            'bearish_strikes': len(bearish_signals),
            'avg_bullish_strength': np.mean([s.strength for s in bullish_signals]) if bullish_signals else 0,
            'avg_bearish_strength': np.mean([s.strength for s in bearish_signals]) if bearish_signals else 0,
            'avg_reliability': np.mean([s.reliability_score for s in self.signals]),
            'volume_confirmed_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals)
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()
