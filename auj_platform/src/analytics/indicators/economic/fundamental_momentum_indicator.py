"""
Fundamental Momentum Indicator
Tracks fundamental momentum based on economic event impacts and data releases.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
import logging
from collections import deque

from ....core.data_contracts import EconomicEvent
from ..base.base_indicator import EconomicIndicator

logger = logging.getLogger(__name__)

@dataclass
class FundamentalMomentumSignal:
    """Fundamental momentum signal"""
    currency: str
    momentum_score: float           # -1.0 to 1.0
    momentum_strength: str          # WEAK, MODERATE, STRONG, VERY_STRONG
    trend_direction: str            # BULLISH, BEARISH, NEUTRAL
    acceleration: float             # Rate of change in momentum
    sustainability_score: float     # 0.0 to 1.0
    contributing_events: List[str]
    timeframe: str                 # SHORT, MEDIUM, LONG

class FundamentalMomentumIndicator(EconomicIndicator):
    """
    Advanced indicator that tracks fundamental momentum by analyzing
    economic event impacts over time to identify building economic trends.
    """
    
    def __init__(self,
                 momentum_window_days: int = 30,
                 acceleration_window_days: int = 7,
                 strong_momentum_threshold: float = 0.6,
                 weak_momentum_threshold: float = 0.2):
        """
        Initialize Fundamental Momentum Indicator
        
        Args:
            momentum_window_days: Days to analyze for momentum calculation
            acceleration_window_days: Days to analyze for acceleration
            strong_momentum_threshold: Threshold for strong momentum signals
            weak_momentum_threshold: Threshold for weak momentum signals
        """
        super().__init__()
        self.momentum_window_days = momentum_window_days
        self.acceleration_window_days = acceleration_window_days
        self.strong_momentum_threshold = strong_momentum_threshold
        self.weak_momentum_threshold = weak_momentum_threshold
        
        # Event scoring system
        self.event_scores = {
            'GDP': {'positive': 0.8, 'negative': -0.8, 'weight': 1.0},
            'Employment': {'positive': 0.7, 'negative': -0.7, 'weight': 0.9},
            'NFP': {'positive': 0.9, 'negative': -0.9, 'weight': 1.0},
            'CPI': {'positive': -0.6, 'negative': 0.6, 'weight': 0.8},  # Inverted for inflation
            'Interest Rate': {'positive': 0.9, 'negative': -0.9, 'weight': 1.0},
            'PMI': {'positive': 0.6, 'negative': -0.6, 'weight': 0.7},
            'Retail Sales': {'positive': 0.5, 'negative': -0.5, 'weight': 0.6},
            'Industrial Production': {'positive': 0.6, 'negative': -0.6, 'weight': 0.7},
            'Consumer Confidence': {'positive': 0.4, 'negative': -0.4, 'weight': 0.5},
            'Trade Balance': {'positive': 0.5, 'negative': -0.5, 'weight': 0.6}
        }
        
        # Momentum decay factors (how momentum decays over time)
        self.momentum_decay = {
            'CRITICAL': 0.95,  # Slow decay for critical events
            'HIGH': 0.90,
            'MEDIUM': 0.85,
            'LOW': 0.80
        }
        
        # Currency-specific factors
        self.currency_momentum_factors = {
            'USD': {'central_bank_weight': 1.0, 'economic_sensitivity': 1.0},
            'EUR': {'central_bank_weight': 0.9, 'economic_sensitivity': 0.9},
            'GBP': {'central_bank_weight': 0.8, 'economic_sensitivity': 1.1},
            'JPY': {'central_bank_weight': 0.9, 'economic_sensitivity': 0.8},
            'AUD': {'central_bank_weight': 0.7, 'economic_sensitivity': 1.2},
            'CAD': {'central_bank_weight': 0.7, 'economic_sensitivity': 1.1},
            'CHF': {'central_bank_weight': 0.8, 'economic_sensitivity': 0.7},
            'NZD': {'central_bank_weight': 0.6, 'economic_sensitivity': 1.3}
        }
        
        # Historical momentum tracking
        self.momentum_history = {}  # Currency -> deque of momentum scores
        
    async def calculate(self, data: pd.DataFrame,
                       economic_events: List[EconomicEvent] = None,
                       historical_events: List[EconomicEvent] = None,
                       pair: str = None) -> Dict:
        """
        Calculate fundamental momentum for currencies
        
        Args:
            data: Price data DataFrame
            economic_events: Recent/upcoming economic events
            historical_events: Historical events for momentum calculation
            pair: Currency pair
            
        Returns:
            Dictionary containing momentum analysis
        """
        try:
            if not economic_events:
                economic_events = []
            
            if not historical_events:
                historical_events = await self._fetch_synthetic_historical_events()
            
            current_time = datetime.now()
            
            # Extract currencies from pair
            currencies = self._extract_currencies_from_pair(pair)
            
            # Calculate momentum for each currency
            momentum_analysis = {}
            for currency in currencies:
                momentum_data = await self._calculate_currency_momentum(
                    currency, historical_events, current_time
                )
                momentum_analysis[currency] = momentum_data
            
            # Generate comparative signals for the pair
            signals = self._generate_momentum_signals(momentum_analysis, pair)
            
            # Calculate momentum trends and acceleration
            trend_analysis = self._analyze_momentum_trends(momentum_analysis)
            
            # Assess sustainability of current momentum
            sustainability_analysis = self._assess_momentum_sustainability(
                momentum_analysis, economic_events
            )
            
            # Generate trading recommendations
            recommendations = self._generate_momentum_recommendations(
                signals, trend_analysis, sustainability_analysis
            )
            
            return {
                'signals': signals,
                'momentum_analysis': momentum_analysis,
                'trend_analysis': trend_analysis,
                'sustainability_analysis': sustainability_analysis,
                'recommendations': recommendations,
                'calculation_time': current_time,
                'pair': pair
            }
            
        except Exception as e:
            logger.error(f"Error calculating fundamental momentum: {e}")
            return self._generate_error_result(str(e))
    
    def _extract_currencies_from_pair(self, pair: str) -> List[str]:
        """Extract currencies from trading pair"""
        if not pair or len(pair) < 6:
            return ['USD', 'EUR']  # Default currencies
        
        return [pair[:3], pair[3:6]]
    
    async def _fetch_synthetic_historical_events(self) -> List[EconomicEvent]:
        """Generate synthetic historical events for demonstration"""
        current_time = datetime.now()
        historical_events = []
        
        # Generate events for the past 30 days
        for days_ago in range(1, 31):
            event_time = current_time - timedelta(days=days_ago)
            
            # GDP events (monthly)
            if days_ago % 30 == 0:
                historical_events.append(EconomicEvent(
                    name='GDP',
                    time=event_time,
                    currency='USD',
                    impact_level='HIGH',
                    actual=2.3,
                    forecast=2.1,
                    previous=2.0
                ))
            
            # Employment events (weekly)
            if days_ago % 7 == 0:
                historical_events.append(EconomicEvent(
                    name='Employment',
                    time=event_time,
                    currency='USD',
                    impact_level='MEDIUM',
                    actual=220000,
                    forecast=215000,
                    previous=210000
                ))
            
            # CPI events (monthly)
            if days_ago % 30 == 15:
                historical_events.append(EconomicEvent(
                    name='CPI',
                    time=event_time,
                    currency='EUR',
                    impact_level='HIGH',
                    actual=2.1,
                    forecast=2.0,
                    previous=1.9
                ))
        
        return historical_events
    
    async def _calculate_currency_momentum(self, currency: str,
                                         historical_events: List[EconomicEvent],
                                         current_time: datetime) -> Dict:
        """Calculate momentum for a specific currency"""
        # Filter events for this currency
        currency_events = [
            event for event in historical_events
            if event.currency == currency
        ]
        
        # Calculate time-weighted momentum scores
        momentum_scores = []
        cutoff_time = current_time - timedelta(days=self.momentum_window_days)
        
        for event in currency_events:
            if event.time >= cutoff_time:
                # Calculate event impact score
                event_score = self._calculate_event_impact_score(event)
                
                # Apply time decay
                days_ago = (current_time - event.time).days
                decay_factor = self._calculate_time_decay(event, days_ago)
                
                # Apply currency-specific factors
                currency_factor = self._get_currency_momentum_factor(currency, event)
                
                final_score = event_score * decay_factor * currency_factor
                momentum_scores.append({
                    'score': final_score,
                    'event': event.name,
                    'time': event.time,
                    'impact_level': event.impact_level
                })
        
        # Calculate overall momentum
        if momentum_scores:
            recent_scores = [s['score'] for s in momentum_scores[-10:]]  # Last 10 events
            momentum_score = np.mean(recent_scores)
            momentum_volatility = np.std(recent_scores) if len(recent_scores) > 1 else 0
        else:
            momentum_score = 0.0
            momentum_volatility = 0.0
        
        # Calculate acceleration (change in momentum)
        acceleration = self._calculate_momentum_acceleration(
            currency, momentum_score, current_time
        )
        
        # Update momentum history
        self._update_momentum_history(currency, momentum_score, current_time)
        
        return {
            'currency': currency,
            'momentum_score': momentum_score,
            'momentum_volatility': momentum_volatility,
            'acceleration': acceleration,
            'contributing_events': [s['event'] for s in momentum_scores[-5:]],
            'score_history': momentum_scores,
            'last_updated': current_time
        }
    
    def _calculate_event_impact_score(self, event: EconomicEvent) -> float:
        """Calculate impact score for an economic event"""
        # Base score from actual vs forecast
        if hasattr(event, 'actual') and hasattr(event, 'forecast'):
            if event.actual is not None and event.forecast is not None:
                # Calculate surprise factor
                if event.forecast != 0:
                    surprise = (event.actual - event.forecast) / abs(event.forecast)
                else:
                    surprise = 0.1 if event.actual > 0 else -0.1
            else:
                surprise = 0.0
        else:
            surprise = 0.0
        
        # Get event type scoring
        event_config = None
        for event_type, config in self.event_scores.items():
            if event_type.lower() in event.name.lower():
                event_config = config
                break
        
        if not event_config:
            event_config = {'positive': 0.3, 'negative': -0.3, 'weight': 0.5}
        
        # Calculate directional score
        if surprise > 0:
            base_score = event_config['positive'] * min(surprise * 2, 1.0)
        else:
            base_score = event_config['negative'] * min(abs(surprise) * 2, 1.0)
        
        # Apply impact level multiplier
        impact_multiplier = {
            'CRITICAL': 1.0,
            'HIGH': 0.8,
            'MEDIUM': 0.5,
            'LOW': 0.2
        }.get(event.impact_level, 0.3)
        
        return base_score * impact_multiplier * event_config['weight']
    
    def _calculate_time_decay(self, event: EconomicEvent, days_ago: int) -> float:
        """Calculate time decay factor for event impact"""
        decay_rate = self.momentum_decay.get(event.impact_level, 0.85)
        return decay_rate ** days_ago
    
    def _get_currency_momentum_factor(self, currency: str, event: EconomicEvent) -> float:
        """Get currency-specific momentum factor"""
        currency_config = self.currency_momentum_factors.get(
            currency, {'central_bank_weight': 0.8, 'economic_sensitivity': 1.0}
        )
        
        # Determine if event is central bank related
        is_central_bank = any(
            term in event.name.lower() 
            for term in ['rate', 'monetary', 'fed', 'ecb', 'boe', 'boj']
        )
        
        if is_central_bank:
            return currency_config['central_bank_weight']
        else:
            return currency_config['economic_sensitivity']
    
    def _calculate_momentum_acceleration(self, currency: str,
                                       current_momentum: float,
                                       current_time: datetime) -> float:
        """Calculate momentum acceleration (rate of change)"""
        if currency not in self.momentum_history:
            return 0.0
        
        history = self.momentum_history[currency]
        if len(history) < 2:
            return 0.0
        
        # Get momentum from acceleration window ago
        cutoff_time = current_time - timedelta(days=self.acceleration_window_days)
        past_momentum = None
        
        for momentum_data in reversed(history):
            if momentum_data['time'] <= cutoff_time:
                past_momentum = momentum_data['score']
                break
        
        if past_momentum is not None:
            return current_momentum - past_momentum
        else:
            return 0.0
    
    def _update_momentum_history(self, currency: str, momentum_score: float,
                                current_time: datetime):
        """Update momentum history for a currency"""
        if currency not in self.momentum_history:
            self.momentum_history[currency] = deque(maxlen=100)  # Keep last 100 records
        
        self.momentum_history[currency].append({
            'score': momentum_score,
            'time': current_time
        })
    
    def _generate_momentum_signals(self, momentum_analysis: Dict, 
                                 pair: str) -> List[FundamentalMomentumSignal]:
        """Generate momentum trading signals"""
        signals = []
        
        if len(momentum_analysis) < 2:
            return signals
        
        currencies = list(momentum_analysis.keys())
        base_currency = currencies[0]
        quote_currency = currencies[1]
        
        base_momentum = momentum_analysis[base_currency]
        quote_momentum = momentum_analysis[quote_currency]
        
        # Calculate relative momentum
        relative_momentum = (
            base_momentum['momentum_score'] - quote_momentum['momentum_score']
        )
        
        # Calculate momentum strength
        momentum_strength = abs(relative_momentum)
        
        if momentum_strength >= self.weak_momentum_threshold:
            # Determine strength category
            if momentum_strength >= self.strong_momentum_threshold:
                strength_category = 'VERY_STRONG'
            elif momentum_strength >= (self.strong_momentum_threshold * 0.75):
                strength_category = 'STRONG'
            elif momentum_strength >= (self.strong_momentum_threshold * 0.5):
                strength_category = 'MODERATE'
            else:
                strength_category = 'WEAK'
            
            # Determine trend direction
            if relative_momentum > 0.1:
                trend_direction = 'BULLISH'
            elif relative_momentum < -0.1:
                trend_direction = 'BEARISH'
            else:
                trend_direction = 'NEUTRAL'
            
            # Calculate acceleration
            base_acceleration = base_momentum.get('acceleration', 0.0)
            quote_acceleration = quote_momentum.get('acceleration', 0.0)
            relative_acceleration = base_acceleration - quote_acceleration
            
            # Calculate sustainability
            sustainability = self._calculate_signal_sustainability(
                base_momentum, quote_momentum
            )
            
            # Determine timeframe based on momentum strength
            if momentum_strength >= self.strong_momentum_threshold:
                timeframe = 'LONG'
            elif momentum_strength >= (self.strong_momentum_threshold * 0.6):
                timeframe = 'MEDIUM'
            else:
                timeframe = 'SHORT'
            
            # Create signal
            signal = FundamentalMomentumSignal(
                currency=base_currency,
                momentum_score=relative_momentum,
                momentum_strength=strength_category,
                trend_direction=trend_direction,
                acceleration=relative_acceleration,
                sustainability_score=sustainability,
                contributing_events=(
                    base_momentum.get('contributing_events', []) +
                    quote_momentum.get('contributing_events', [])
                ),
                timeframe=timeframe
            )
            
            signals.append(signal)
        
        return signals
    
    def _calculate_signal_sustainability(self, base_momentum: Dict,
                                       quote_momentum: Dict) -> float:
        """Calculate sustainability score for momentum signal"""
        # Base sustainability on momentum volatility and acceleration
        base_volatility = base_momentum.get('momentum_volatility', 0.5)
        quote_volatility = quote_momentum.get('momentum_volatility', 0.5)
        
        base_acceleration = abs(base_momentum.get('acceleration', 0.0))
        quote_acceleration = abs(quote_momentum.get('acceleration', 0.0))
        
        # Lower volatility = higher sustainability
        volatility_score = 1.0 - min((base_volatility + quote_volatility) / 2, 1.0)
        
        # Positive acceleration = higher sustainability
        acceleration_score = min((base_acceleration + quote_acceleration) / 2, 1.0)
        
        return (volatility_score * 0.6 + acceleration_score * 0.4)
    
    def _analyze_momentum_trends(self, momentum_analysis: Dict) -> Dict:
        """Analyze momentum trends across currencies"""
        trend_analysis = {
            'strongest_momentum': None,
            'weakest_momentum': None,
            'momentum_leaders': [],
            'momentum_laggards': [],
            'cross_currency_correlations': {}
        }
        
        if not momentum_analysis:
            return trend_analysis
        
        # Find strongest and weakest momentum
        momentum_scores = {
            currency: data['momentum_score']
            for currency, data in momentum_analysis.items()
        }
        
        if momentum_scores:
            strongest_currency = max(momentum_scores, key=momentum_scores.get)
            weakest_currency = min(momentum_scores, key=momentum_scores.get)
            
            trend_analysis['strongest_momentum'] = {
                'currency': strongest_currency,
                'score': momentum_scores[strongest_currency]
            }
            
            trend_analysis['weakest_momentum'] = {
                'currency': weakest_currency,
                'score': momentum_scores[weakest_currency]
            }
        
        # Identify leaders and laggards
        for currency, score in momentum_scores.items():
            if score > 0.3:
                trend_analysis['momentum_leaders'].append(currency)
            elif score < -0.3:
                trend_analysis['momentum_laggards'].append(currency)
        
        return trend_analysis
    
    def _assess_momentum_sustainability(self, momentum_analysis: Dict,
                                      upcoming_events: List[EconomicEvent]) -> Dict:
        """Assess sustainability of current momentum"""
        sustainability = {
            'overall_sustainability': 'MEDIUM',
            'risk_factors': [],
            'support_factors': [],
            'upcoming_catalysts': []
        }
        
        # Check for upcoming high-impact events
        high_impact_events = [
            event for event in upcoming_events
            if event.impact_level in ['HIGH', 'CRITICAL']
        ]
        
        if high_impact_events:
            sustainability['upcoming_catalysts'] = [
                f"{event.name} ({event.currency})" for event in high_impact_events
            ]
        
        # Analyze momentum consistency
        volatility_scores = [
            data.get('momentum_volatility', 0.5)
            for data in momentum_analysis.values()
        ]
        
        avg_volatility = np.mean(volatility_scores) if volatility_scores else 0.5
        
        if avg_volatility > 0.8:
            sustainability['overall_sustainability'] = 'LOW'
            sustainability['risk_factors'].append('High momentum volatility')
        elif avg_volatility < 0.3:
            sustainability['overall_sustainability'] = 'HIGH'
            sustainability['support_factors'].append('Low momentum volatility')
        
        # Check acceleration trends
        accelerations = [
            abs(data.get('acceleration', 0.0))
            for data in momentum_analysis.values()
        ]
        
        avg_acceleration = np.mean(accelerations) if accelerations else 0.0
        
        if avg_acceleration > 0.5:
            sustainability['support_factors'].append('Strong momentum acceleration')
        elif avg_acceleration < 0.1:
            sustainability['risk_factors'].append('Weak momentum acceleration')
        
        return sustainability
    
    def _generate_momentum_recommendations(self, signals: List[FundamentalMomentumSignal],
                                         trend_analysis: Dict,
                                         sustainability_analysis: Dict) -> List[Dict]:
        """Generate trading recommendations based on momentum analysis"""
        recommendations = []
        
        for signal in signals:
            recommendation = {
                'signal_id': f"momentum_{len(recommendations) + 1}",
                'currency': signal.currency,
                'action': self._determine_action_from_signal(signal),
                'strength': signal.momentum_strength,
                'timeframe': signal.timeframe,
                'entry_timing': self._determine_entry_timing(signal),
                'risk_management': self._determine_momentum_risk_management(signal),
                'profit_expectations': self._calculate_profit_expectations(signal),
                'sustainability_score': signal.sustainability_score
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _determine_action_from_signal(self, signal: FundamentalMomentumSignal) -> str:
        """Determine trading action from momentum signal"""
        if signal.trend_direction == 'BULLISH' and signal.momentum_strength in ['STRONG', 'VERY_STRONG']:
            return 'BUY'
        elif signal.trend_direction == 'BEARISH' and signal.momentum_strength in ['STRONG', 'VERY_STRONG']:
            return 'SELL'
        elif signal.momentum_strength in ['MODERATE', 'STRONG']:
            return 'WATCH'
        else:
            return 'HOLD'
    
    def _determine_entry_timing(self, signal: FundamentalMomentumSignal) -> str:
        """Determine optimal entry timing"""
        if signal.acceleration > 0.3:
            return 'IMMEDIATE'
        elif signal.acceleration > 0.1:
            return 'NEAR_TERM'
        else:
            return 'PATIENT'
    
    def _determine_momentum_risk_management(self, signal: FundamentalMomentumSignal) -> Dict:
        """Determine risk management for momentum trade"""
        base_risk = 0.02  # 2% base risk
        
        if signal.momentum_strength == 'VERY_STRONG':
            position_size = base_risk * 1.5
        elif signal.momentum_strength == 'STRONG':
            position_size = base_risk * 1.2
        elif signal.momentum_strength == 'MODERATE':
            position_size = base_risk * 1.0
        else:
            position_size = base_risk * 0.7
        
        # Adjust for sustainability
        position_size *= signal.sustainability_score
        
        return {
            'position_size': position_size,
            'stop_loss_distance': f"{(1.0 - signal.sustainability_score) * 100 + 50} pips",
            'take_profit_distance': f"{signal.sustainability_score * 200 + 100} pips",
            'trailing_stop': signal.momentum_strength in ['STRONG', 'VERY_STRONG']
        }
    
    def _calculate_profit_expectations(self, signal: FundamentalMomentumSignal) -> Dict:
        """Calculate profit expectations for momentum trade"""
        strength_multipliers = {
            'VERY_STRONG': 3.0,
            'STRONG': 2.5,
            'MODERATE': 2.0,
            'WEAK': 1.5
        }
        
        base_expectation = strength_multipliers.get(signal.momentum_strength, 1.5)
        sustainability_factor = signal.sustainability_score
        
        return {
            'profit_target_ratio': base_expectation * sustainability_factor,
            'time_horizon': signal.timeframe,
            'probability_of_success': min(0.9, 0.5 + sustainability_factor * 0.4)
        }
    
    def _generate_error_result(self, error_message: str) -> Dict:
        """Generate error result"""
        return {
            'signals': [],
            'momentum_analysis': {},
            'trend_analysis': {},
            'sustainability_analysis': {},
            'recommendations': [],
            'calculation_time': datetime.now(),
            'error': error_message
        }