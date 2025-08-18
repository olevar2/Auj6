"""
Advanced Zone Indicator with Support/Resistance and Price Action Analysis

Features:
- Dynamic zone identification using price action
- Multiple timeframe zone confirmation
- Zone strength scoring based on touches and volume
- Breakout and retest detection
- Zone confluence analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class ZoneType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    DEMAND = "demand"
    SUPPLY = "supply"

class ZoneStrength(Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class ZoneState(Enum):
    ZONE_HOLDING = "zone_holding"
    ZONE_TESTED = "zone_tested"
    ZONE_BROKEN = "zone_broken"
    ZONE_RETESTED = "zone_retested"
    BREAKOUT_CONFIRMED = "breakout_confirmed"
    FALSE_BREAKOUT = "false_breakout"

@dataclass
class PriceZone:
    zone_type: ZoneType
    upper_bound: float
    lower_bound: float
    center_price: float
    strength: ZoneStrength
    touch_count: int
    volume_strength: float
    age: int

@dataclass
class ZoneResult:
    active_zones: List[PriceZone]
    current_zone: PriceZone
    nearest_support: float
    nearest_resistance: float
    zone_state: ZoneState
    confluence_score: float
    signal: SignalType
    confidence: float

class ZoneIndicator(StandardIndicatorInterface):
    def __init__(self, lookback_period: int = 50, zone_thickness: float = 0.002, 
                 min_touches: int = 2, max_zones: int = 10):
        self.lookback_period = lookback_period
        self.zone_thickness = zone_thickness  # As percentage of price
        self.min_touches = min_touches
        self.max_zones = max_zones
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.lookback_period:
                raise ValueError("Insufficient data")
            
            # Identify price zones
            zones_data = self._identify_price_zones(data)
            
            # Analyze current zone state
            zone_state = self._analyze_zone_state(data, zones_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, zones_data, zone_state)
            
            # Find nearest support/resistance
            current_price = data['close'].iloc[-1]
            nearest_support, nearest_resistance = self._find_nearest_levels(
                zones_data['zones'], current_price
            )
            
            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(zones_data['zones'], current_price)
            
            # Find current zone
            current_zone = self._find_current_zone(zones_data['zones'], current_price)
            
            latest_result = ZoneResult(
                active_zones=zones_data['zones'],
                current_zone=current_zone,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                zone_state=zone_state,
                confluence_score=confluence_score,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'support_levels': [zone.lower_bound for zone in zones_data['zones'] 
                                     if zone.zone_type in [ZoneType.SUPPORT, ZoneType.DEMAND]],
                    'resistance_levels': [zone.upper_bound for zone in zones_data['zones'] 
                                        if zone.zone_type in [ZoneType.RESISTANCE, ZoneType.SUPPLY]],
                    'zone_strengths': [zone.strength.value for zone in zones_data['zones']],
                    'confluence_score': [confluence_score] * len(data)
                },
                'signal': signal,
                'confidence': confidence,
                'zone_state': zone_state.value,
                'zones_count': len(zones_data['zones'])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Zone Indicator: {e}")
            return self._get_default_result()
    
    def _identify_price_zones(self, data: pd.DataFrame) -> Dict:
        """Identify significant price zones"""
        zones = []
        
        # Use recent data for zone identification
        recent_data = data.tail(self.lookback_period)
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(recent_data['high'], True)
        swing_lows = self._find_swing_points(recent_data['low'], False)
        
        # Create resistance zones from swing highs
        for idx, price in swing_highs:
            zone = self._create_zone(recent_data, idx, price, ZoneType.RESISTANCE)
            if zone and zone.touch_count >= self.min_touches:
                zones.append(zone)
        
        # Create support zones from swing lows
        for idx, price in swing_lows:
            zone = self._create_zone(recent_data, idx, price, ZoneType.SUPPORT)
            if zone and zone.touch_count >= self.min_touches:
                zones.append(zone)
        
        # Add volume-based demand/supply zones
        volume_zones = self._identify_volume_zones(recent_data)
        zones.extend(volume_zones)
        
        # Filter and rank zones
        zones = self._filter_and_rank_zones(zones)
        
        return {'zones': zones}
    
    def _find_swing_points(self, price_series: pd.Series, is_high: bool) -> List[Tuple[int, float]]:
        """Find swing highs or lows"""
        swing_points = []
        window = 5  # Look for peaks/valleys in 5-bar window
        
        for i in range(window, len(price_series) - window):
            current_price = price_series.iloc[i]
            
            if is_high:
                # Check if current price is highest in window
                left_max = price_series.iloc[i-window:i].max()
                right_max = price_series.iloc[i+1:i+window+1].max()
                
                if current_price > left_max and current_price > right_max:
                    swing_points.append((i, current_price))
            else:
                # Check if current price is lowest in window
                left_min = price_series.iloc[i-window:i].min()
                right_min = price_series.iloc[i+1:i+window+1].min()
                
                if current_price < left_min and current_price < right_min:
                    swing_points.append((i, current_price))
        
        return swing_points
    
    def _create_zone(self, data: pd.DataFrame, idx: int, center_price: float, 
                    zone_type: ZoneType) -> PriceZone:
        """Create a price zone around a significant level"""
        
        # Calculate zone boundaries
        thickness = center_price * self.zone_thickness
        upper_bound = center_price + thickness
        lower_bound = center_price - thickness
        
        # Count touches and calculate volume strength
        touch_count = 0
        total_volume = 0
        
        for i in range(len(data)):
            if lower_bound <= data['low'].iloc[i] <= upper_bound or \
               lower_bound <= data['high'].iloc[i] <= upper_bound:
                touch_count += 1
                if 'volume' in data.columns:
                    total_volume += data['volume'].iloc[i]
        
        # Calculate volume strength
        avg_volume = data['volume'].mean() if 'volume' in data.columns else 1
        volume_strength = (total_volume / touch_count) / avg_volume if touch_count > 0 else 1
        
        # Determine zone strength
        strength = self._determine_zone_strength(touch_count, volume_strength)
        
        # Calculate age
        age = len(data) - idx
        
        return PriceZone(
            zone_type=zone_type,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            center_price=center_price,
            strength=strength,
            touch_count=touch_count,
            volume_strength=volume_strength,
            age=age
        )
    
    def _identify_volume_zones(self, data: pd.DataFrame) -> List[PriceZone]:
        """Identify demand/supply zones based on volume analysis"""
        volume_zones = []
        
        if 'volume' not in data.columns:
            return volume_zones
        
        # Find high volume bars
        volume_threshold = data['volume'].quantile(0.8)
        high_volume_bars = data[data['volume'] > volume_threshold]
        
        for idx, row in high_volume_bars.iterrows():
            # Determine if it's demand or supply zone
            price_change = row['close'] - row['open']
            
            if price_change > 0:  # Bullish bar - demand zone
                zone_type = ZoneType.DEMAND
                center_price = (row['low'] + row['open']) / 2
            else:  # Bearish bar - supply zone
                zone_type = ZoneType.SUPPLY
                center_price = (row['high'] + row['open']) / 2
            
            # Create zone
            thickness = center_price * self.zone_thickness
            zone = PriceZone(
                zone_type=zone_type,
                upper_bound=center_price + thickness,
                lower_bound=center_price - thickness,
                center_price=center_price,
                strength=ZoneStrength.MEDIUM,
                touch_count=1,
                volume_strength=row['volume'] / data['volume'].mean(),
                age=len(data) - data.index.get_loc(idx)
            )
            
            volume_zones.append(zone)
        
        return volume_zones
    
    def _determine_zone_strength(self, touch_count: int, volume_strength: float) -> ZoneStrength:
        """Determine zone strength based on touches and volume"""
        score = touch_count + (volume_strength - 1)  # Normalize volume strength
        
        if score >= 6:
            return ZoneStrength.VERY_STRONG
        elif score >= 4:
            return ZoneStrength.STRONG
        elif score >= 2:
            return ZoneStrength.MEDIUM
        else:
            return ZoneStrength.WEAK
    
    def _filter_and_rank_zones(self, zones: List[PriceZone]) -> List[PriceZone]:
        """Filter overlapping zones and rank by strength"""
        if not zones:
            return zones
        
        # Remove overlapping zones (keep stronger one)
        filtered_zones = []
        zones_sorted = sorted(zones, key=lambda z: z.center_price)
        
        for zone in zones_sorted:
            overlapping = False
            for existing_zone in filtered_zones:
                if self._zones_overlap(zone, existing_zone):
                    overlapping = True
                    # If new zone is stronger, replace existing
                    if zone.strength.value > existing_zone.strength.value:
                        filtered_zones.remove(existing_zone)
                        filtered_zones.append(zone)
                    break
            
            if not overlapping:
                filtered_zones.append(zone)
        
        # Limit number of zones and sort by strength
        filtered_zones.sort(key=lambda z: (z.strength.value, z.touch_count), reverse=True)
        
        return filtered_zones[:self.max_zones]
    
    def _zones_overlap(self, zone1: PriceZone, zone2: PriceZone) -> bool:
        """Check if two zones overlap"""
        return not (zone1.upper_bound < zone2.lower_bound or zone2.upper_bound < zone1.lower_bound)
    
    def _analyze_zone_state(self, data: pd.DataFrame, zones_data: Dict) -> ZoneState:
        """Analyze current zone interaction state"""
        current_price = data['close'].iloc[-1]
        zones = zones_data['zones']
        
        # Find if price is currently in a zone
        current_zone = self._find_current_zone(zones, current_price)
        
        if current_zone:
            return ZoneState.ZONE_TESTED
        
        # Check for recent breakouts
        recent_data = data.tail(10)
        for zone in zones:
            if self._check_zone_breakout(recent_data, zone):
                return ZoneState.BREAKOUT_CONFIRMED
        
        return ZoneState.ZONE_HOLDING
    
    def _find_current_zone(self, zones: List[PriceZone], current_price: float) -> PriceZone:
        """Find zone that contains current price"""
        for zone in zones:
            if zone.lower_bound <= current_price <= zone.upper_bound:
                return zone
        return None
    
    def _check_zone_breakout(self, recent_data: pd.DataFrame, zone: PriceZone) -> bool:
        """Check if zone has been broken recently"""
        for i, row in recent_data.iterrows():
            if zone.zone_type in [ZoneType.RESISTANCE, ZoneType.SUPPLY]:
                if row['close'] > zone.upper_bound:
                    return True
            else:  # Support or Demand
                if row['close'] < zone.lower_bound:
                    return True
        return False
    
    def _find_nearest_levels(self, zones: List[PriceZone], current_price: float) -> Tuple[float, float]:
        """Find nearest support and resistance levels"""
        support_levels = []
        resistance_levels = []
        
        for zone in zones:
            if zone.zone_type in [ZoneType.SUPPORT, ZoneType.DEMAND]:
                if zone.center_price < current_price:
                    support_levels.append(zone.center_price)
            else:  # Resistance or Supply
                if zone.center_price > current_price:
                    resistance_levels.append(zone.center_price)
        
        nearest_support = max(support_levels) if support_levels else 0.0
        nearest_resistance = min(resistance_levels) if resistance_levels else 0.0
        
        return nearest_support, nearest_resistance
    
    def _calculate_confluence_score(self, zones: List[PriceZone], current_price: float) -> float:
        """Calculate confluence score based on nearby zones"""
        confluence = 0.0
        price_range = current_price * 0.01  # 1% range
        
        for zone in zones:
            distance = abs(zone.center_price - current_price)
            if distance <= price_range:
                # Closer zones and stronger zones contribute more
                weight = (1 - distance / price_range) * (1 + zone.touch_count * 0.1)
                confluence += weight
        
        return min(1.0, confluence / 3.0)  # Normalize to 0-1
    
    def _generate_signals(self, data: pd.DataFrame, zones_data: Dict, 
                         zone_state: ZoneState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        zones = zones_data['zones']
        
        # Find nearest support/resistance
        nearest_support, nearest_resistance = self._find_nearest_levels(zones, current_price)
        
        # Calculate distances for confidence
        support_distance = abs(current_price - nearest_support) / current_price if nearest_support > 0 else 1
        resistance_distance = abs(current_price - nearest_resistance) / current_price if nearest_resistance > 0 else 1
        
        # Base confidence on proximity to zones
        base_confidence = 1 - min(support_distance, resistance_distance) if min(support_distance, resistance_distance) < 0.02 else 0.3
        
        # State-based signals
        if zone_state == ZoneState.BREAKOUT_CONFIRMED:
            if resistance_distance < support_distance:
                return SignalType.BUY, min(0.9, base_confidence + 0.2)
            else:
                return SignalType.SELL, min(0.9, base_confidence + 0.2)
        elif zone_state == ZoneState.ZONE_TESTED:
            # At zone - expect bounce
            if support_distance < resistance_distance:
                return SignalType.BUY, base_confidence * 0.8
            else:
                return SignalType.SELL, base_confidence * 0.8
        elif support_distance < 0.005:  # Very close to support
            return SignalType.BUY, base_confidence * 0.7
        elif resistance_distance < 0.005:  # Very close to resistance
            return SignalType.SELL, base_confidence * 0.7
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_zone = PriceZone(ZoneType.SUPPORT, 0.0, 0.0, 0.0, ZoneStrength.WEAK, 0, 1.0, 0)
        default_result = ZoneResult([], default_zone, 0.0, 0.0, ZoneState.ZONE_HOLDING, 
                                  0.0, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {
            'lookback_period': self.lookback_period,
            'zone_thickness': self.zone_thickness,
            'min_touches': self.min_touches,
            'max_zones': self.max_zones
        }
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)