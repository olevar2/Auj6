"""
Economic Event Impact Indicator
Analyzes the potential impact of economic events on currency pairs and generates trading signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import asyncio
import logging

from ....core.data_contracts import EconomicEvent, EconomicEventImpact
from ..base.base_indicator import EconomicIndicator

logger = logging.getLogger(__name__)

@dataclass
class EventImpactSignal:
    """Signal generated from economic event impact analysis"""
    currency: str
    event_impact: str  # HIGH, MEDIUM, LOW
    signal_strength: float  # 0.0 to 1.0
    signal_direction: str   # BUY, SELL, NEUTRAL
    confidence: float       # 0.0 to 1.0
    time_window: Tuple[datetime, datetime]  # Start and end of impact window
    risk_adjustment: float  # Risk multiplier (0.5 to 2.0)

class EconomicEventImpactIndicator(EconomicIndicator):
    """
    Advanced indicator that analyzes economic calendar events and generates trading signals
    based on historical impact patterns, event importance, and market conditions.
    """
    
    def __init__(self, 
                 look_ahead_hours: int = 24,
                 impact_decay_hours: int = 4,
                 high_impact_threshold: float = 0.7,
                 medium_impact_threshold: float = 0.4):
        """
        Initialize Economic Event Impact Indicator
        
        Args:
            look_ahead_hours: Hours to look ahead for upcoming events
            impact_decay_hours: Hours over which event impact decays
            high_impact_threshold: Threshold for high impact signals
            medium_impact_threshold: Threshold for medium impact signals
        """
        super().__init__()
        self.look_ahead_hours = look_ahead_hours
        self.impact_decay_hours = impact_decay_hours
        self.high_impact_threshold = high_impact_threshold
        self.medium_impact_threshold = medium_impact_threshold
        
        # Event impact patterns learned from historical data
        self.impact_patterns = {
            'NFP': {'volatility_increase': 2.5, 'trend_duration': 6, 'reversal_probability': 0.3},
            'CPI': {'volatility_increase': 2.0, 'trend_duration': 4, 'reversal_probability': 0.25},
            'Interest Rate Decision': {'volatility_increase': 3.0, 'trend_duration': 8, 'reversal_probability': 0.4},
            'GDP': {'volatility_increase': 1.5, 'trend_duration': 3, 'reversal_probability': 0.2},
            'Unemployment Rate': {'volatility_increase': 1.8, 'trend_duration': 4, 'reversal_probability': 0.22},
            'Retail Sales': {'volatility_increase': 1.3, 'trend_duration': 2, 'reversal_probability': 0.18}
        }
        
        # Currency strength factors
        self.currency_strength_factors = {
            'USD': 1.0, 'EUR': 0.9, 'GBP': 0.85, 'JPY': 0.8,
            'AUD': 0.7, 'CAD': 0.65, 'CHF': 0.75, 'NZD': 0.6
        }
        
    async def calculate(self, data: pd.DataFrame, 
                       economic_events: List[EconomicEvent] = None,
                       pair: str = None) -> Dict:
        """
        Calculate economic event impact signals
        
        Args:
            data: Price data DataFrame
            economic_events: List of economic events
            pair: Currency pair (e.g., 'EURUSD')
            
        Returns:
            Dictionary containing impact signals and analysis
        """
        try:
            if economic_events is None or not economic_events:
                return self._generate_neutral_signal()
                
            current_time = datetime.now()
            
            # Filter relevant events within look-ahead window
            relevant_events = self._filter_relevant_events(
                economic_events, current_time, pair
            )
            
            if not relevant_events:
                return self._generate_neutral_signal()
            
            # Analyze event impacts
            impact_analysis = await self._analyze_event_impacts(
                relevant_events, data, pair
            )
            
            # Generate trading signals
            signals = self._generate_trading_signals(impact_analysis, pair)
            
            # Calculate risk adjustments
            risk_adjustments = self._calculate_risk_adjustments(
                relevant_events, impact_analysis
            )
            
            return {
                'signals': signals,
                'impact_analysis': impact_analysis,
                'risk_adjustments': risk_adjustments,
                'event_count': len(relevant_events),
                'max_impact_level': max([e.impact_level for e in relevant_events]) 
                    if relevant_events else 'LOW',
                'calculation_time': current_time,
                'pair': pair
            }
            
        except Exception as e:
            logger.error(f"Error calculating economic event impact: {e}")
            return self._generate_error_signal(str(e))
    
    def _filter_relevant_events(self, events: List[EconomicEvent], 
                               current_time: datetime, pair: str) -> List[EconomicEvent]:
        """Filter events relevant to the trading pair and time window"""
        relevant_events = []
        
        # Extract currencies from pair
        if pair and len(pair) >= 6:
            base_currency = pair[:3]
            quote_currency = pair[3:6]
            pair_currencies = {base_currency, quote_currency}
        else:
            pair_currencies = set()
        
        end_time = current_time + timedelta(hours=self.look_ahead_hours)
        
        for event in events:
            # Check time relevance
            if current_time <= event.time <= end_time:
                # Check currency relevance
                if not pair_currencies or event.currency in pair_currencies:
                    # Check impact significance
                    if event.impact_level in ['HIGH', 'MEDIUM', 'CRITICAL']:
                        relevant_events.append(event)
        
        return sorted(relevant_events, key=lambda x: x.time)
    
    async def _analyze_event_impacts(self, events: List[EconomicEvent], 
                                   data: pd.DataFrame, pair: str) -> Dict:
        """Analyze the potential impact of economic events"""
        analysis = {
            'individual_impacts': [],
            'cumulative_impact': 0.0,
            'volatility_forecast': 1.0,
            'directional_bias': 'NEUTRAL',
            'confidence_level': 0.0
        }
        
        cumulative_impact = 0.0
        volatility_multiplier = 1.0
        bullish_signals = 0
        bearish_signals = 0
        
        for event in events:
            # Get historical impact pattern
            pattern = self.impact_patterns.get(
                event.name, 
                {'volatility_increase': 1.2, 'trend_duration': 2, 'reversal_probability': 0.15}
            )
            
            # Calculate event impact score
            impact_score = self._calculate_event_impact_score(event, pattern)
            
            # Determine directional bias
            direction = self._determine_event_direction(event, pair)
            
            # Calculate time decay factor
            time_to_event = (event.time - datetime.now()).total_seconds() / 3600
            decay_factor = max(0.1, 1.0 - (time_to_event / self.look_ahead_hours))
            
            # Apply decay to impact
            adjusted_impact = impact_score * decay_factor
            
            cumulative_impact += adjusted_impact
            volatility_multiplier *= pattern['volatility_increase'] ** 0.5
            
            if direction == 'BULLISH':
                bullish_signals += adjusted_impact
            elif direction == 'BEARISH':
                bearish_signals += adjusted_impact
            
            analysis['individual_impacts'].append({
                'event': event.name,
                'currency': event.currency,
                'impact_score': impact_score,
                'adjusted_impact': adjusted_impact,
                'direction': direction,
                'time': event.time,
                'pattern': pattern
            })
        
        # Calculate overall analysis
        analysis['cumulative_impact'] = min(cumulative_impact, 2.0)  # Cap at 2.0
        analysis['volatility_forecast'] = min(volatility_multiplier, 3.0)  # Cap at 3.0x
        
        # Determine directional bias
        if bullish_signals > bearish_signals * 1.2:
            analysis['directional_bias'] = 'BULLISH'
        elif bearish_signals > bullish_signals * 1.2:
            analysis['directional_bias'] = 'BEARISH'
        else:
            analysis['directional_bias'] = 'NEUTRAL'
        
        # Calculate confidence level
        analysis['confidence_level'] = min(
            (cumulative_impact / len(events)) * 0.8 + 
            (abs(bullish_signals - bearish_signals) / max(bullish_signals + bearish_signals, 0.1)) * 0.2,
            1.0
        )
        
        return analysis
    
    def _calculate_event_impact_score(self, event: EconomicEvent, pattern: Dict) -> float:
        """Calculate impact score for an individual event"""
        base_score = {
            'CRITICAL': 1.0,
            'HIGH': 0.8,
            'MEDIUM': 0.5,
            'LOW': 0.2
        }.get(event.impact_level, 0.2)
        
        # Apply currency strength factor
        currency_factor = self.currency_strength_factors.get(event.currency, 0.5)
        
        # Apply pattern volatility factor
        volatility_factor = min(pattern.get('volatility_increase', 1.0) / 2.0, 1.0)
        
        return base_score * currency_factor * volatility_factor
    
    def _determine_event_direction(self, event: EconomicEvent, pair: str) -> str:
        """Determine if event is bullish or bearish for the currency pair"""
        if not pair or len(pair) < 6:
            return 'NEUTRAL'
        
        base_currency = pair[:3]
        
        # Simplistic directional mapping (would be enhanced with actual/forecast comparison)
        positive_indicators = [
            'employment', 'gdp', 'retail sales', 'industrial production',
            'consumer confidence', 'business sentiment'
        ]
        
        negative_indicators = [
            'unemployment', 'inflation', 'trade deficit'
        ]
        
        event_name_lower = event.name.lower()
        
        # Determine if event is positive or negative for economy
        is_positive = any(indicator in event_name_lower for indicator in positive_indicators)
        is_negative = any(indicator in event_name_lower for indicator in negative_indicators)
        
        if event.currency == base_currency:
            if is_positive:
                return 'BULLISH'
            elif is_negative:
                return 'BEARISH'
        else:  # Quote currency
            if is_positive:
                return 'BEARISH'  # Strong quote currency = bearish for pair
            elif is_negative:
                return 'BULLISH'  # Weak quote currency = bullish for pair
        
        return 'NEUTRAL'
    
    def _generate_trading_signals(self, analysis: Dict, pair: str) -> List[EventImpactSignal]:
        """Generate trading signals based on impact analysis"""
        signals = []
        
        if analysis['cumulative_impact'] < self.medium_impact_threshold:
            return signals  # No significant impact expected
        
        # Determine signal strength
        if analysis['cumulative_impact'] >= self.high_impact_threshold:
            signal_strength = min(analysis['cumulative_impact'], 1.0)
        else:
            signal_strength = analysis['cumulative_impact'] / self.high_impact_threshold * 0.7
        
        # Determine signal direction
        if analysis['directional_bias'] == 'BULLISH':
            signal_direction = 'BUY'
        elif analysis['directional_bias'] == 'BEARISH':
            signal_direction = 'SELL'
        else:
            signal_direction = 'NEUTRAL'
        
        # Calculate time window
        current_time = datetime.now()
        window_start = current_time
        window_end = current_time + timedelta(hours=self.impact_decay_hours)
        
        # Create signal
        if signal_direction != 'NEUTRAL':
            signal = EventImpactSignal(
                currency=pair[:3] if pair else 'USD',
                event_impact=self._classify_impact_level(analysis['cumulative_impact']),
                signal_strength=signal_strength,
                signal_direction=signal_direction,
                confidence=analysis['confidence_level'],
                time_window=(window_start, window_end),
                risk_adjustment=min(analysis['volatility_forecast'], 2.0)
            )
            signals.append(signal)
        
        return signals
    
    def _classify_impact_level(self, cumulative_impact: float) -> str:
        """Classify impact level based on cumulative impact score"""
        if cumulative_impact >= self.high_impact_threshold:
            return 'HIGH'
        elif cumulative_impact >= self.medium_impact_threshold:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_risk_adjustments(self, events: List[EconomicEvent], 
                                  analysis: Dict) -> Dict:
        """Calculate risk management adjustments"""
        return {
            'position_size_multiplier': max(0.5, 1.0 - (analysis['cumulative_impact'] * 0.3)),
            'stop_loss_multiplier': analysis['volatility_forecast'],
            'take_profit_multiplier': 1.0 + (analysis['cumulative_impact'] * 0.5),
            'max_exposure_reduction': min(0.5, analysis['cumulative_impact'] * 0.4),
            'recommended_timeframe': self._recommend_timeframe(analysis)
        }
    
    def _recommend_timeframe(self, analysis: Dict) -> str:
        """Recommend trading timeframe based on event impact"""
        if analysis['cumulative_impact'] >= 0.8:
            return 'H4'  # Longer timeframe for high impact
        elif analysis['cumulative_impact'] >= 0.5:
            return 'H1'  # Medium timeframe
        else:
            return 'M15'  # Shorter timeframe for lower impact
    
    def _generate_neutral_signal(self) -> Dict:
        """Generate neutral signal when no significant events"""
        return {
            'signals': [],
            'impact_analysis': {
                'individual_impacts': [],
                'cumulative_impact': 0.0,
                'volatility_forecast': 1.0,
                'directional_bias': 'NEUTRAL',
                'confidence_level': 0.0
            },
            'risk_adjustments': {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0,
                'max_exposure_reduction': 0.0,
                'recommended_timeframe': 'H1'
            },
            'event_count': 0,
            'max_impact_level': 'NONE',
            'calculation_time': datetime.now(),
            'pair': None
        }
    
    def _generate_error_signal(self, error_message: str) -> Dict:
        """Generate error signal"""
        return {
            'signals': [],
            'impact_analysis': None,
            'risk_adjustments': None,
            'event_count': 0,
            'max_impact_level': 'ERROR',
            'calculation_time': datetime.now(),
            'error': error_message
        }