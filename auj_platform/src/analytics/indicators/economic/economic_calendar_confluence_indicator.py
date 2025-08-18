"""
Economic Calendar Confluence Indicator
Identifies confluence zones where multiple economic events and technical factors align.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
import logging

from ....core.data_contracts import EconomicEvent
from ..base.base_indicator import EconomicIndicator

logger = logging.getLogger(__name__)

@dataclass
class ConfluenceZone:
    """Confluence zone identification"""
    zone_start: datetime
    zone_end: datetime
    confluence_strength: float      # 0.0 to 1.0
    contributing_factors: List[str]
    economic_events: List[str]
    technical_factors: List[str]
    recommended_action: str         # BUY, SELL, AVOID, WAIT
    risk_level: str                # LOW, MEDIUM, HIGH

class EconomicCalendarConfluenceIndicator(EconomicIndicator):
    """
    Advanced indicator that identifies confluence zones where economic events
    align with technical analysis factors to create high-probability trading opportunities.
    """
    
    def __init__(self,
                 confluence_threshold: float = 0.6,
                 zone_window_hours: int = 4,
                 technical_weight: float = 0.4,
                 fundamental_weight: float = 0.6):
        """
        Initialize Economic Calendar Confluence Indicator
        
        Args:
            confluence_threshold: Minimum strength for confluence zones
            zone_window_hours: Hours around events to consider for confluence
            technical_weight: Weight for technical factors (0.0 to 1.0)
            fundamental_weight: Weight for fundamental factors (0.0 to 1.0)
        """
        super().__init__()
        self.confluence_threshold = confluence_threshold
        self.zone_window_hours = zone_window_hours
        self.technical_weight = technical_weight
        self.fundamental_weight = fundamental_weight
        
        # Technical factor weights
        self.technical_factors = {
            'support_resistance': 0.8,
            'moving_average': 0.6,
            'fibonacci_level': 0.7,
            'trend_line': 0.6,
            'volume_profile': 0.5,
            'momentum_divergence': 0.7,
            'overbought_oversold': 0.5
        }
        
        # Event impact multipliers
        self.event_impact_multipliers = {
            'CRITICAL': 1.0,
            'HIGH': 0.8,
            'MEDIUM': 0.5,
            'LOW': 0.2
        }
        
        # Currency pair volatility factors
        self.pair_volatility_factors = {
            'EURUSD': 1.0, 'GBPUSD': 1.2, 'USDJPY': 0.9, 'USDCHF': 0.8,
            'AUDUSD': 1.1, 'USDCAD': 1.0, 'NZDUSD': 1.3, 'EURGBP': 0.7,
            'EURJPY': 1.1, 'GBPJPY': 1.4
        }
    
    async def calculate(self, data: pd.DataFrame,
                       economic_events: List[EconomicEvent] = None,
                       pair: str = None,
                       technical_analysis: Dict = None) -> Dict:
        """
        Identify confluence zones with economic events and technical factors
        
        Args:
            data: Price data DataFrame
            economic_events: List of economic events
            pair: Currency pair
            technical_analysis: Technical analysis results
            
        Returns:
            Dictionary containing confluence analysis
        """
        try:
            if economic_events is None:
                economic_events = []
            
            if technical_analysis is None:
                technical_analysis = await self._generate_technical_analysis(data, pair)
            
            current_time = datetime.now()
            
            # Filter relevant events
            relevant_events = self._filter_relevant_events(
                economic_events, current_time, pair
            )
            
            # Identify potential confluence zones
            confluence_zones = await self._identify_confluence_zones(
                relevant_events, technical_analysis, data, current_time
            )
            
            # Score and rank confluence zones
            scored_zones = self._score_confluence_zones(confluence_zones, pair)
            
            # Filter zones above threshold
            significant_zones = [
                zone for zone in scored_zones
                if zone.confluence_strength >= self.confluence_threshold
            ]
            
            # Generate trading recommendations
            recommendations = self._generate_trading_recommendations(
                significant_zones, data, pair
            )
            
            # Calculate risk assessment
            risk_assessment = self._assess_confluence_risks(
                significant_zones, relevant_events, data
            )
            
            return {
                'confluence_zones': significant_zones,
                'recommendations': recommendations,
                'risk_assessment': risk_assessment,
                'technical_analysis': technical_analysis,
                'relevant_events': len(relevant_events),
                'zone_count': len(significant_zones),
                'calculation_time': current_time,
                'pair': pair
            }
            
        except Exception as e:
            logger.error(f"Error in confluence analysis: {e}")
            return self._generate_error_result(str(e))
    
    def _filter_relevant_events(self, events: List[EconomicEvent],
                               current_time: datetime, pair: str) -> List[EconomicEvent]:
        """Filter events relevant for confluence analysis"""
        if not pair or len(pair) < 6:
            return events
        
        base_currency = pair[:3]
        quote_currency = pair[3:6]
        relevant_currencies = {base_currency, quote_currency}
        
        # Look ahead 72 hours for confluence opportunities
        end_time = current_time + timedelta(hours=72)
        
        return [
            event for event in events
            if (event.currency in relevant_currencies and
                current_time <= event.time <= end_time and
                event.impact_level in ['MEDIUM', 'HIGH', 'CRITICAL'])
        ]
    
    async def _generate_technical_analysis(self, data: pd.DataFrame, 
                                         pair: str) -> Dict:
        """Generate simplified technical analysis"""
        if len(data) < 50:
            return self._get_default_technical_analysis()
        
        try:
            # Calculate key technical levels
            close = data['close']
            high = data['high']
            low = data['low']
            
            # Moving averages
            ma20 = close.rolling(window=20).mean().iloc[-1]
            ma50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else ma20
            
            current_price = close.iloc[-1]
            
            # Support and resistance levels
            recent_high = high.tail(20).max()
            recent_low = low.tail(20).min()
            
            # RSI calculation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # Technical factor analysis
            technical_factors = []
            
            # Support/Resistance proximity
            if abs(current_price - recent_high) / current_price < 0.001:
                technical_factors.append('near_resistance')
            elif abs(current_price - recent_low) / current_price < 0.001:
                technical_factors.append('near_support')
            
            # Moving average alignment
            if current_price > ma20 > ma50:
                technical_factors.append('bullish_ma_alignment')
            elif current_price < ma20 < ma50:
                technical_factors.append('bearish_ma_alignment')
            
            # Overbought/Oversold
            if current_rsi > 70:
                technical_factors.append('overbought')
            elif current_rsi < 30:
                technical_factors.append('oversold')
            
            return {
                'current_price': current_price,
                'ma20': ma20,
                'ma50': ma50,
                'resistance_level': recent_high,
                'support_level': recent_low,
                'rsi': current_rsi,
                'technical_factors': technical_factors,
                'trend_direction': self._determine_trend_direction(data)
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return self._get_default_technical_analysis()
    
    def _get_default_technical_analysis(self) -> Dict:
        """Get default technical analysis when calculation fails"""
        return {
            'current_price': 1.0,
            'ma20': 1.0,
            'ma50': 1.0,
            'resistance_level': 1.01,
            'support_level': 0.99,
            'rsi': 50.0,
            'technical_factors': [],
            'trend_direction': 'SIDEWAYS'
        }
    
    def _determine_trend_direction(self, data: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        if len(data) < 20:
            return 'SIDEWAYS'
        
        close = data['close']
        ma_short = close.rolling(window=10).mean()
        ma_long = close.rolling(window=20).mean()
        
        if ma_short.iloc[-1] > ma_long.iloc[-1] * 1.002:
            return 'UPTREND'
        elif ma_short.iloc[-1] < ma_long.iloc[-1] * 0.998:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    async def _identify_confluence_zones(self, events: List[EconomicEvent],
                                       technical_analysis: Dict,
                                       data: pd.DataFrame,
                                       current_time: datetime) -> List[ConfluenceZone]:
        """Identify potential confluence zones"""
        confluence_zones = []
        
        # Group events by time proximity
        event_clusters = self._cluster_events_by_time(events)
        
        for cluster in event_clusters:
            # Create confluence zone around event cluster
            cluster_time = cluster[0].time  # Use first event time as reference
            zone_start = cluster_time - timedelta(hours=self.zone_window_hours // 2)
            zone_end = cluster_time + timedelta(hours=self.zone_window_hours // 2)
            
            # Analyze confluence factors
            confluence_analysis = await self._analyze_confluence_factors(
                cluster, technical_analysis, data, cluster_time
            )
            
            if confluence_analysis['strength'] > 0.3:  # Minimum threshold
                confluence_zone = ConfluenceZone(
                    zone_start=zone_start,
                    zone_end=zone_end,
                    confluence_strength=confluence_analysis['strength'],
                    contributing_factors=confluence_analysis['factors'],
                    economic_events=[event.name for event in cluster],
                    technical_factors=confluence_analysis['technical_factors'],
                    recommended_action=confluence_analysis['action'],
                    risk_level=confluence_analysis['risk_level']
                )
                
                confluence_zones.append(confluence_zone)
        
        return confluence_zones
    
    def _cluster_events_by_time(self, events: List[EconomicEvent]) -> List[List[EconomicEvent]]:
        """Cluster events by time proximity"""
        if not events:
            return []
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x.time)
        
        clusters = []
        current_cluster = [sorted_events[0]]
        
        for i in range(1, len(sorted_events)):
            current_event = sorted_events[i]
            last_event = current_cluster[-1]
            
            # If events are within zone window, add to current cluster
            time_diff = abs((current_event.time - last_event.time).total_seconds() / 3600)
            
            if time_diff <= self.zone_window_hours:
                current_cluster.append(current_event)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [current_event]
        
        # Add last cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters
    
    async def _analyze_confluence_factors(self, event_cluster: List[EconomicEvent],
                                        technical_analysis: Dict,
                                        data: pd.DataFrame,
                                        cluster_time: datetime) -> Dict:
        """Analyze confluence factors for an event cluster"""
        analysis = {
            'strength': 0.0,
            'factors': [],
            'technical_factors': [],
            'action': 'WAIT',
            'risk_level': 'MEDIUM'
        }
        
        # Fundamental factors (economic events)
        fundamental_score = 0.0
        high_impact_events = [e for e in event_cluster if e.impact_level in ['HIGH', 'CRITICAL']]
        
        if high_impact_events:
            fundamental_score += 0.6
            analysis['factors'].append('high_impact_events')
        
        # Multiple events confluence
        if len(event_cluster) > 1:
            fundamental_score += 0.3
            analysis['factors'].append('multiple_events')
        
        # Technical factors
        technical_score = 0.0
        technical_factors = technical_analysis.get('technical_factors', [])
        
        for factor in technical_factors:
            if factor in ['near_resistance', 'near_support']:
                technical_score += 0.4
                analysis['technical_factors'].append(factor)
            elif factor in ['overbought', 'oversold']:
                technical_score += 0.3
                analysis['technical_factors'].append(factor)
            elif factor in ['bullish_ma_alignment', 'bearish_ma_alignment']:
                technical_score += 0.2
                analysis['technical_factors'].append(factor)
        
        # Combine scores
        analysis['strength'] = (
            fundamental_score * self.fundamental_weight +
            technical_score * self.technical_weight
        )
        
        # Determine recommended action
        if analysis['strength'] >= 0.8:
            # Strong confluence - determine direction
            if 'near_support' in analysis['technical_factors'] and high_impact_events:
                analysis['action'] = 'BUY'
            elif 'near_resistance' in analysis['technical_factors'] and high_impact_events:
                analysis['action'] = 'SELL'
            else:
                analysis['action'] = 'WAIT'
        elif analysis['strength'] >= 0.6:
            analysis['action'] = 'WATCH'
        else:
            analysis['action'] = 'AVOID'
        
        # Determine risk level
        if len(high_impact_events) > 1:
            analysis['risk_level'] = 'HIGH'
        elif len(high_impact_events) == 1:
            analysis['risk_level'] = 'MEDIUM'
        else:
            analysis['risk_level'] = 'LOW'
        
        return analysis
    
    def _score_confluence_zones(self, zones: List[ConfluenceZone], 
                               pair: str) -> List[ConfluenceZone]:
        """Score and rank confluence zones"""
        # Apply pair-specific volatility factors
        volatility_factor = self.pair_volatility_factors.get(pair, 1.0)
        
        for zone in zones:
            # Adjust strength based on pair volatility
            zone.confluence_strength *= volatility_factor
            
            # Cap at 1.0
            zone.confluence_strength = min(zone.confluence_strength, 1.0)
        
        # Sort by strength (highest first)
        return sorted(zones, key=lambda x: x.confluence_strength, reverse=True)
    
    def _generate_trading_recommendations(self, zones: List[ConfluenceZone],
                                        data: pd.DataFrame, pair: str) -> List[Dict]:
        """Generate specific trading recommendations"""
        recommendations = []
        
        for zone in zones[:3]:  # Top 3 zones
            recommendation = {
                'zone_id': f"confluence_{len(recommendations) + 1}",
                'time_window': (zone.zone_start, zone.zone_end),
                'action': zone.recommended_action,
                'strength': zone.confluence_strength,
                'risk_level': zone.risk_level,
                'entry_strategy': self._determine_entry_strategy(zone),
                'risk_management': self._determine_risk_management(zone),
                'profit_targets': self._calculate_profit_targets(zone, data)
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _determine_entry_strategy(self, zone: ConfluenceZone) -> Dict:
        """Determine entry strategy for confluence zone"""
        if zone.recommended_action in ['BUY', 'SELL']:
            return {
                'entry_type': 'market_on_event',
                'confirmation_required': zone.confluence_strength < 0.8,
                'position_size': 'normal' if zone.risk_level == 'MEDIUM' else 'reduced',
                'timeframe': 'H1' if zone.risk_level == 'LOW' else 'H4'
            }
        else:
            return {
                'entry_type': 'wait_and_watch',
                'confirmation_required': True,
                'position_size': 'small',
                'timeframe': 'M15'
            }
    
    def _determine_risk_management(self, zone: ConfluenceZone) -> Dict:
        """Determine risk management for confluence zone"""
        risk_multipliers = {
            'LOW': 1.0,
            'MEDIUM': 1.5,
            'HIGH': 2.0
        }
        
        multiplier = risk_multipliers.get(zone.risk_level, 1.5)
        
        return {
            'stop_loss_multiplier': multiplier,
            'take_profit_multiplier': 1.0 / multiplier,
            'trailing_stop': zone.confluence_strength > 0.8,
            'max_exposure': 0.02 / multiplier  # Percentage of account
        }
    
    def _calculate_profit_targets(self, zone: ConfluenceZone, 
                                 data: pd.DataFrame) -> List[float]:
        """Calculate profit targets based on confluence strength"""
        if len(data) < 20:
            return [1.5, 2.5, 4.0]  # Default ratios
        
        # Calculate average true range for targets
        high = data['high'].tail(20)
        low = data['low'].tail(20)
        close = data['close'].tail(20)
        
        atr = (high - low).mean()
        current_price = close.iloc[-1]
        
        # Adjust targets based on confluence strength
        base_targets = [1.5, 2.5, 4.0]
        strength_multiplier = zone.confluence_strength
        
        return [target * strength_multiplier for target in base_targets]
    
    def _assess_confluence_risks(self, zones: List[ConfluenceZone],
                               events: List[EconomicEvent],
                               data: pd.DataFrame) -> Dict:
        """Assess overall risks in confluence analysis"""
        risk_assessment = {
            'overall_risk': 'MEDIUM',
            'risk_factors': [],
            'mitigation_strategies': [],
            'max_recommended_exposure': 0.05
        }
        
        # Count high-risk zones
        high_risk_zones = len([z for z in zones if z.risk_level == 'HIGH'])
        
        if high_risk_zones > 1:
            risk_assessment['overall_risk'] = 'HIGH'
            risk_assessment['risk_factors'].append('Multiple high-risk confluence zones')
            risk_assessment['max_recommended_exposure'] = 0.02
        
        # Check for clustered high-impact events
        critical_events = [e for e in events if e.impact_level == 'CRITICAL']
        if len(critical_events) > 1:
            risk_assessment['risk_factors'].append('Multiple critical economic events')
            risk_assessment['mitigation_strategies'].append('Reduce position sizes')
        
        # Add standard mitigation strategies
        risk_assessment['mitigation_strategies'].extend([
            'Use proper stop losses',
            'Diversify across multiple timeframes',
            'Monitor news flow closely'
        ])
        
        return risk_assessment
    
    def _generate_error_result(self, error_message: str) -> Dict:
        """Generate error result"""
        return {
            'confluence_zones': [],
            'recommendations': [],
            'risk_assessment': {
                'overall_risk': 'UNKNOWN',
                'risk_factors': ['Analysis error'],
                'mitigation_strategies': [],
                'max_recommended_exposure': 0.01
            },
            'technical_analysis': {},
            'relevant_events': 0,
            'zone_count': 0,
            'calculation_time': datetime.now(),
            'error': error_message
        }