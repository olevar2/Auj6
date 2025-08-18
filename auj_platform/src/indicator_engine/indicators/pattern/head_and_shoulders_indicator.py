"""
Head and Shoulders Pattern Indicator

Head and Shoulders is a classic reversal pattern that signals the end of an uptrend.

Pattern Recognition:
1. Head and Shoulders Top (Bearish Reversal):
   - Left Shoulder: First peak with moderate volume
   - Head: Higher peak (new high) with declining volume
   - Right Shoulder: Lower peak similar to left shoulder, low volume
   - Neckline: Support line connecting lows between shoulders and head
   - Breakout: Price breaks below neckline with increased volume

2. Inverse Head and Shoulders (Bullish Reversal):
   - Left Shoulder: First trough with moderate volume
   - Head: Lower trough (new low) with declining volume
   - Right Shoulder: Higher trough similar to left shoulder, low volume
   - Neckline: Resistance line connecting highs between shoulders and head
   - Breakout: Price breaks above neckline with increased volume

Trading Significance:
- Highly reliable reversal pattern
- Volume should decrease during head formation
- Volume should increase on neckline breakout
- Price target = distance from head to neckline projected from breakout
- Pattern failure if price returns above/below neckline after breakout
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from scipy import stats
from ...core.signal_type import SignalType
from .base_pattern_indicator import BasePatternIndicator


@dataclass
class HeadAndShouldersSignal:
    """Signal data for Head and Shoulders pattern"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    pattern_type: str  # 'head_and_shoulders_top', 'inverse_head_and_shoulders'
    formation_stage: str  # 'left_shoulder', 'head', 'right_shoulder', 'neckline_break', 'confirmed'
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    left_shoulder: Dict
    head: Dict
    right_shoulder: Dict
    neckline: Dict
    volume_profile: Dict
    pattern_symmetry: float
    volume_confirmation: bool
    reliability_score: float


class HeadAndShouldersIndicator(BasePatternIndicator):
    """
    Head and Shoulders Pattern Recognition Indicator
    
    Identifies both regular and inverse head and shoulders patterns
    with volume analysis and symmetry validation.
    """
    
    def __init__(self, 
                 min_pattern_periods: int = 15,
                 max_pattern_periods: int = 60,
                 symmetry_tolerance: float = 0.1,
                 shoulder_height_tolerance: float = 0.05,
                 volume_decline_threshold: float = 0.8,
                 volume_breakout_threshold: float = 1.4,
                 neckline_min_r_squared: float = 0.6,
                 atr_periods: int = 14):
        """
        Initialize Head and Shoulders indicator
        
        Args:
            min_pattern_periods: Minimum periods for pattern formation
            max_pattern_periods: Maximum periods for pattern formation
            symmetry_tolerance: Tolerance for pattern symmetry (0-1)
            shoulder_height_tolerance: Tolerance for shoulder height similarity
            volume_decline_threshold: Volume decline factor during head formation
            volume_breakout_threshold: Volume increase factor for breakout confirmation
            neckline_min_r_squared: Minimum R-squared for neckline validation
            atr_periods: Periods for ATR calculation
        """
        super().__init__()
        self.min_pattern_periods = min_pattern_periods
        self.max_pattern_periods = max_pattern_periods
        self.symmetry_tolerance = symmetry_tolerance
        self.shoulder_height_tolerance = shoulder_height_tolerance
        self.volume_decline_threshold = volume_decline_threshold
        self.volume_breakout_threshold = volume_breakout_threshold
        self.neckline_min_r_squared = neckline_min_r_squared
        self.atr_periods = atr_periods
        self.signals: List[HeadAndShouldersSignal] = []
        
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
    
    def _find_peaks_and_troughs(self, data: pd.DataFrame, window: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """Find significant peaks and troughs in price data"""
        highs = data['high'].values
        lows = data['low'].values
        volumes = data.get('volume', pd.Series([1] * len(data))).values
        timestamps = data.index
        
        peaks = []
        troughs = []
        
        for i in range(window, len(data) - window):
            # Check for peak (local maximum)
            is_peak = all(highs[i] >= highs[j] for j in range(i - window, i + window + 1) if j != i)
            if is_peak:
                peaks.append({
                    'index': i,
                    'timestamp': timestamps[i],
                    'price': highs[i],
                    'volume': volumes[i]
                })
            
            # Check for trough (local minimum)
            is_trough = all(lows[i] <= lows[j] for j in range(i - window, i + window + 1) if j != i)
            if is_trough:
                troughs.append({
                    'index': i,
                    'timestamp': timestamps[i],
                    'price': lows[i],
                    'volume': volumes[i]
                })
        
        return peaks, troughs
    
    def _validate_head_and_shoulders_top(self, peaks: List[Dict], troughs: List[Dict]) -> Optional[Dict]:
        """Validate head and shoulders top pattern"""
        if len(peaks) < 3 or len(troughs) < 2:
            return None
        
        # Get the last three peaks and two troughs
        left_shoulder = peaks[-3]
        head = peaks[-2]
        right_shoulder = peaks[-1]
        
        left_trough = troughs[-2]
        right_trough = troughs[-1]
        
        # Validate pattern structure
        # Head should be the highest point
        if not (head['price'] > left_shoulder['price'] and head['price'] > right_shoulder['price']):
            return None
        
        # Shoulders should be approximately equal height
        shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price']) / left_shoulder['price']
        if shoulder_diff > self.shoulder_height_tolerance:
            return None
        
        # Check temporal order
        if not (left_shoulder['index'] < head['index'] < right_shoulder['index']):
            return None
        
        if not (left_trough['index'] < right_trough['index']):
            return None
        
        return {
            'type': 'head_and_shoulders_top',
            'left_shoulder': left_shoulder,
            'head': head,
            'right_shoulder': right_shoulder,
            'left_trough': left_trough,
            'right_trough': right_trough
        }
    
    def _validate_inverse_head_and_shoulders(self, peaks: List[Dict], troughs: List[Dict]) -> Optional[Dict]:
        """Validate inverse head and shoulders pattern"""
        if len(troughs) < 3 or len(peaks) < 2:
            return None
        
        # Get the last three troughs and two peaks
        left_shoulder = troughs[-3]
        head = troughs[-2]
        right_shoulder = troughs[-1]
        
        left_peak = peaks[-2]
        right_peak = peaks[-1]
        
        # Validate pattern structure
        # Head should be the lowest point
        if not (head['price'] < left_shoulder['price'] and head['price'] < right_shoulder['price']):
            return None
        
        # Shoulders should be approximately equal depth
        shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price']) / left_shoulder['price']
        if shoulder_diff > self.shoulder_height_tolerance:
            return None
        
        # Check temporal order
        if not (left_shoulder['index'] < head['index'] < right_shoulder['index']):
            return None
        
        if not (left_peak['index'] < right_peak['index']):
            return None
        
        return {
            'type': 'inverse_head_and_shoulders',
            'left_shoulder': left_shoulder,
            'head': head,
            'right_shoulder': right_shoulder,
            'left_peak': left_peak,
            'right_peak': right_peak
        }
    
    def _calculate_neckline(self, pattern: Dict) -> Optional[Dict]:
        """Calculate neckline for the pattern"""
        if pattern['type'] == 'head_and_shoulders_top':
            # Connect the two troughs
            point1 = pattern['left_trough']
            point2 = pattern['right_trough']
        else:
            # Connect the two peaks
            point1 = pattern['left_peak']
            point2 = pattern['right_peak']
        
        # Calculate neckline slope and intercept
        x1, y1 = point1['index'], point1['price']
        x2, y2 = point2['index'], point2['price']
        
        if x2 == x1:
            return None
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # Calculate R-squared for neckline quality
        x_values = np.array([x1, x2])
        y_values = np.array([y1, y2])
        predicted = slope * x_values + intercept
        
        ss_res = np.sum((y_values - predicted) ** 2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'point1': point1,
            'point2': point2
        }
    
    def _calculate_pattern_symmetry(self, pattern: Dict) -> float:
        """Calculate symmetry score for the pattern"""
        left_shoulder = pattern['left_shoulder']
        head = pattern['head']
        right_shoulder = pattern['right_shoulder']
        
        # Time symmetry
        left_to_head = head['index'] - left_shoulder['index']
        head_to_right = right_shoulder['index'] - head['index']
        
        time_ratio = min(left_to_head, head_to_right) / max(left_to_head, head_to_right)
        
        # Price symmetry (shoulder heights)
        if pattern['type'] == 'head_and_shoulders_top':
            price_diff = abs(left_shoulder['price'] - right_shoulder['price'])
            avg_price = (left_shoulder['price'] + right_shoulder['price']) / 2
        else:
            price_diff = abs(left_shoulder['price'] - right_shoulder['price'])
            avg_price = (left_shoulder['price'] + right_shoulder['price']) / 2
        
        price_symmetry = 1 - (price_diff / avg_price) if avg_price > 0 else 0
        
        # Combined symmetry score
        symmetry = (time_ratio + price_symmetry) / 2
        return max(0.0, min(1.0, symmetry))
    
    def _analyze_volume_profile(self, pattern: Dict, data: pd.DataFrame) -> Dict:
        """Analyze volume during pattern formation"""
        if 'volume' not in data.columns:
            return {'volume_confirmation': False, 'volume_decline': False}
        
        left_shoulder = pattern['left_shoulder']
        head = pattern['head']
        right_shoulder = pattern['right_shoulder']
        
        # Get volume at each point
        left_vol = data.iloc[left_shoulder['index']]['volume']
        head_vol = data.iloc[head['index']]['volume']
        right_vol = data.iloc[right_shoulder['index']]['volume']
        
        # Volume should typically decline from left shoulder to head to right shoulder
        volume_decline = (head_vol < left_vol * self.volume_decline_threshold and 
                         right_vol < head_vol)
        
        # Calculate average volume over pattern period
        start_idx = left_shoulder['index']
        end_idx = right_shoulder['index']
        pattern_avg_volume = data.iloc[start_idx:end_idx]['volume'].mean()
        
        # Get recent volume for breakout confirmation
        recent_volume = data['volume'].iloc[-1]
        volume_breakout = recent_volume >= (pattern_avg_volume * self.volume_breakout_threshold)
        
        return {
            'left_shoulder_volume': left_vol,
            'head_volume': head_vol,
            'right_shoulder_volume': right_vol,
            'volume_decline': volume_decline,
            'volume_breakout': volume_breakout,
            'pattern_avg_volume': pattern_avg_volume,
            'volume_confirmation': volume_decline and volume_breakout
        }
    
    def _check_neckline_break(self, pattern: Dict, neckline: Dict, 
                             current_data: pd.Series, current_index: int) -> Tuple[bool, str]:
        """Check if neckline has been broken"""
        if not neckline:
            return False, 'none'
        
        current_neckline_price = neckline['slope'] * current_index + neckline['intercept']
        current_close = current_data['close']
        
        if pattern['type'] == 'head_and_shoulders_top':
            # Bearish breakout below neckline
            if current_close < current_neckline_price:
                return True, 'bearish'
        else:
            # Bullish breakout above neckline
            if current_close > current_neckline_price:
                return True, 'bullish'
        
        return False, 'none'
    
    def _calculate_price_targets(self, pattern: Dict, neckline: Dict, 
                               current_price: float, current_index: int) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        if not neckline:
            return current_price * 0.98, current_price * 1.02
        
        head = pattern['head']
        current_neckline_price = neckline['slope'] * current_index + neckline['intercept']
        
        # Calculate pattern height (head to neckline)
        head_neckline_price = neckline['slope'] * head['index'] + neckline['intercept']
        pattern_height = abs(head['price'] - head_neckline_price)
        
        if pattern['type'] == 'head_and_shoulders_top':
            # Bearish pattern
            stop_loss = max(pattern['left_shoulder']['price'], pattern['right_shoulder']['price'])
            take_profit = current_neckline_price - pattern_height
        else:
            # Bullish pattern
            stop_loss = min(pattern['left_shoulder']['price'], pattern['right_shoulder']['price'])
            take_profit = current_neckline_price + pattern_height
        
        return stop_loss, take_profit
    
    def _calculate_reliability_score(self, pattern: Dict, neckline: Dict, 
                                   volume_profile: Dict, symmetry: float) -> float:
        """Calculate overall pattern reliability score"""
        score = 0.0
        
        # Pattern symmetry (25%)
        score += symmetry * 0.25
        
        # Neckline quality (25%)
        if neckline and neckline['r_squared'] >= self.neckline_min_r_squared:
            score += neckline['r_squared'] * 0.25
        
        # Volume confirmation (30%)
        if volume_profile['volume_confirmation']:
            score += 0.30
        elif volume_profile['volume_decline']:
            score += 0.15
        
        # Pattern structure (20%)
        head = pattern['head']
        left_shoulder = pattern['left_shoulder']
        right_shoulder = pattern['right_shoulder']
        
        if pattern['type'] == 'head_and_shoulders_top':
            head_prominence = (head['price'] - max(left_shoulder['price'], right_shoulder['price'])) / head['price']
        else:
            head_prominence = (min(left_shoulder['price'], right_shoulder['price']) - head['price']) / head['price']
        
        prominence_score = min(1.0, head_prominence * 10)  # Scale prominence
        score += prominence_score * 0.20
        
        return max(0.0, min(1.0, score))
    
    def update(self, data: pd.DataFrame) -> Optional[HeadAndShouldersSignal]:
        """
        Update indicator with new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            HeadAndShouldersSignal if pattern detected, None otherwise
        """
        if len(data) < self.min_pattern_periods + self.atr_periods:
            return None
        
        # Calculate ATR for stop loss calculation
        atr_values = self._calculate_atr(data, self.atr_periods)
        current_atr = atr_values[-1]
        
        # Analyze recent data window
        analysis_window = data.tail(self.max_pattern_periods)
        
        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(analysis_window)
        
        # Try to identify head and shoulders top pattern
        top_pattern = self._validate_head_and_shoulders_top(peaks, troughs)
        inverse_pattern = self._validate_inverse_head_and_shoulders(peaks, troughs)
        
        pattern = top_pattern or inverse_pattern
        if not pattern:
            return None
        
        # Calculate neckline
        neckline = self._calculate_neckline(pattern)
        if not neckline or neckline['r_squared'] < self.neckline_min_r_squared:
            return None
        
        # Calculate pattern metrics
        symmetry = self._calculate_pattern_symmetry(pattern)
        if symmetry < (1 - self.symmetry_tolerance):
            return None
        
        # Analyze volume profile
        volume_profile = self._analyze_volume_profile(pattern, analysis_window)
        
        # Check for neckline break
        current_data = data.iloc[-1]
        current_index = len(data) - 1
        
        neckline_broken, break_direction = self._check_neckline_break(
            pattern, neckline, current_data, current_index
        )
        
        # Determine formation stage
        if neckline_broken:
            formation_stage = 'neckline_break'
            if volume_profile['volume_breakout']:
                formation_stage = 'confirmed'
        else:
            # Check which part we're currently in
            right_shoulder_idx = pattern['right_shoulder']['index']
            if current_index <= right_shoulder_idx + 2:
                formation_stage = 'right_shoulder'
            else:
                formation_stage = 'right_shoulder'
        
        # Calculate price targets
        current_price = current_data['close']
        stop_loss, take_profit = self._calculate_price_targets(
            pattern, neckline, current_price, current_index
        )
        
        # Calculate reliability score
        reliability = self._calculate_reliability_score(
            pattern, neckline, volume_profile, symmetry
        )
        
        # Determine signal type
        if neckline_broken:
            if pattern['type'] == 'head_and_shoulders_top':
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.BUY
        else:
            signal_type = SignalType.HOLD
        
        signal = HeadAndShouldersSignal(
            timestamp=data.index[-1],
            signal_type=signal_type,
            pattern_type=pattern['type'],
            formation_stage=formation_stage,
            strength=reliability,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            left_shoulder=pattern['left_shoulder'],
            head=pattern['head'],
            right_shoulder=pattern['right_shoulder'],
            neckline=neckline,
            volume_profile=volume_profile,
            pattern_symmetry=symmetry,
            volume_confirmation=volume_profile['volume_confirmation'],
            reliability_score=reliability
        )
        
        self.signals.append(signal)
        return signal
    
    def get_signals(self, start_date: Optional[pd.Timestamp] = None,
                   end_date: Optional[pd.Timestamp] = None) -> List[HeadAndShouldersSignal]:
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
        
        top_patterns = [s for s in self.signals if s.pattern_type == 'head_and_shoulders_top']
        inverse_patterns = [s for s in self.signals if s.pattern_type == 'inverse_head_and_shoulders']
        
        return {
            'total_patterns': len(self.signals),
            'head_and_shoulders_top': len(top_patterns),
            'inverse_head_and_shoulders': len(inverse_patterns),
            'avg_symmetry': np.mean([s.pattern_symmetry for s in self.signals]),
            'avg_reliability': np.mean([s.reliability_score for s in self.signals]),
            'volume_confirmed_rate': len([s for s in self.signals if s.volume_confirmation]) / len(self.signals),
            'neckline_breaks': len([s for s in self.signals if s.formation_stage in ['neckline_break', 'confirmed']]),
            'confirmed_patterns': len([s for s in self.signals if s.formation_stage == 'confirmed'])
        }
    
    def reset(self):
        """Reset indicator state"""
        self.signals.clear()