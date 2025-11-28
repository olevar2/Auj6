"""
Economic Calendar Agent Implementation for AUJ Platform.

This module contains specialized agents that integrate economic calendar data
into their market analysis and trading decisions.

FIXES IMPLEMENTED:
- Fixed config_manager initialization in __init__
- Removed confusing commented code
- Ensured proper parent class initialization
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from decimal import Decimal

from .base_agent import BaseAgent, AnalysisResult, AgentState
from ..core.data_contracts import TradeSignal, MarketConditions, ConfidenceLevel
from ..core.exceptions import AgentError
from ..core.logging_setup import get_logger
from ..analytics.indicators.economic.economic_event_impact_indicator import EconomicEventImpactIndicator
from ..analytics.indicators.economic.economic_calendar_confluence_indicator import EconomicCalendarConfluenceIndicator
from ..data_providers.unified_news_economic_provider import UnifiedNewsEconomicProvider

logger = get_logger(__name__)


class EconomicCalendarEnhancedAgent(BaseAgent):
    """
    Enhanced base agent class that integrates economic calendar analysis.
    
    This class extends the base agent to include economic calendar data
    in market analysis and decision-making processes.
    
    FIXED: Proper config_manager initialization via parent class
    """
    
    def __init__(self, 
                 name: str,
                 assigned_indicators: List[str],
                 config: Dict[str, Any],
                 economic_provider: Optional[UnifiedNewsEconomicProvider] = None):
        """
        Initialize economic calendar enhanced agent.
        
        Args:
            name: Agent name
            assigned_indicators: List of assigned indicator names
            config: Agent configuration
            economic_provider: Economic calendar data provider
        """
        # FIXED: Parent class initialization provides config_manager
        super().__init__(name, assigned_indicators, config)
        
        self.economic_provider = economic_provider or UnifiedNewsEconomicProvider()
        self.economic_indicators = {}
        
        # Initialize economic indicators if available
        self._initialize_economic_indicators()
        
        # FIXED: Now config_manager is properly available from parent class
        self.economic_config = self.config_manager.get_dict('economic_analysis', {
            'lookback_hours': 24,
            'lookahead_hours': 8,
            'minimum_impact': 'MEDIUM',
            'consider_sentiment': True,
            'volatility_adjustment': True
        })
        
        logger.info(f"Initialized EconomicCalendarEnhancedAgent: {name}")
    
    def _initialize_economic_indicators(self):
        """Initialize economic calendar indicators."""
        try:
            if 'economic_event_impact_indicator' in self.assigned_indicators:
                self.economic_indicators['impact'] = EconomicEventImpactIndicator()
                
            if 'economic_calendar_confluence_indicator' in self.assigned_indicators:
                self.economic_indicators['confluence'] = EconomicCalendarConfluenceIndicator()
                
            logger.info(f"Initialized {len(self.economic_indicators)} economic indicators")
            
        except Exception as e:
            logger.error(f"Error initializing economic indicators: {e}")
    
    async def get_economic_context(self, 
                                 symbol: str, 
                                 lookback_hours: int = 24, 
                                 lookahead_hours: int = 8) -> Dict[str, Any]:
        """
        Get economic calendar context for analysis.
        
        Args:
            symbol: Trading symbol
            lookback_hours: Hours to look back for events
            lookahead_hours: Hours to look ahead for events
            
        Returns:
            Economic context data
        """
        try:
            # Get currency from symbol (e.g., EURUSD -> EUR, USD)
            base_currency = symbol[:3] if len(symbol) >= 6 else 'USD'
            quote_currency = symbol[3:6] if len(symbol) >= 6 else 'USD'
            
            # Time ranges
            start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
            end_time = datetime.utcnow() + timedelta(hours=lookahead_hours)
            
            # Get economic events
            events = await self.economic_provider.get_economic_calendar(
                start_date=start_time,
                end_date=end_time,
                currencies=[base_currency, quote_currency],
                importance=['HIGH', 'CRITICAL']
            )
            
            # Analyze economic context
            context = {
                'total_events': len(events),
                'recent_events': [e for e in events if e.event_time <= datetime.utcnow()],
                'upcoming_events': [e for e in events if e.event_time > datetime.utcnow()],
                'high_impact_recent': [],
                'high_impact_upcoming': [],
                'currencies_affected': [base_currency, quote_currency],
                'economic_bias': 'NEUTRAL',
                'volatility_expectation': 'NORMAL',
                'confidence': 0.5
            }
            
            # Analyze recent high-impact events
            for event in context['recent_events']:
                if event.importance in ['HIGH', 'CRITICAL']:
                    context['high_impact_recent'].append({
                        'event': event.event_name,
                        'currency': event.currency,
                        'actual': event.actual_value,
                        'forecast': event.forecast_value,
                        'impact': event.importance,
                        'hours_ago': (datetime.utcnow() - event.event_time).total_seconds() / 3600
                    })
            
            # Analyze upcoming high-impact events
            for event in context['upcoming_events']:
                if event.importance in ['HIGH', 'CRITICAL']:
                    context['high_impact_upcoming'].append({
                        'event': event.event_name,
                        'currency': event.currency,
                        'forecast': event.forecast_value,
                        'previous': event.previous_value,
                        'impact': event.importance,
                        'hours_until': (event.event_time - datetime.utcnow()).total_seconds() / 3600
                    })
            
            # Determine economic bias and volatility expectation
            if context['high_impact_recent'] or context['high_impact_upcoming']:
                context['volatility_expectation'] = 'HIGH'
                context['confidence'] = 0.8
                
                # Simple bias calculation based on event outcomes
                positive_events = 0
                total_events = 0
                
                for event in context['high_impact_recent']:
                    if event.get('actual') and event.get('forecast'):
                        if event['actual'] > event['forecast']:
                            positive_events += 1
                        total_events += 1
                
                if total_events > 0:
                    bias_ratio = positive_events / total_events
                    if bias_ratio > 0.6:
                        context['economic_bias'] = 'BULLISH'
                    elif bias_ratio < 0.4:
                        context['economic_bias'] = 'BEARISH'
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting economic context: {e}")
            return {
                'total_events': 0,
                'recent_events': [],
                'upcoming_events': [],
                'economic_bias': 'NEUTRAL',
                'volatility_expectation': 'NORMAL',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def integrate_economic_analysis(self, 
                                  base_analysis: Dict[str, Any], 
                                  economic_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate economic calendar analysis with base technical analysis.
        
        Args:
            base_analysis: Base technical analysis results
            economic_context: Economic calendar context
            
        Returns:
            Enhanced analysis with economic integration
        """
        try:
            enhanced_analysis = base_analysis.copy()
            
            # Adjust confidence based on economic events
            economic_confidence_adjustment = 0.0
            
            if economic_context.get('high_impact_upcoming'):
                # Reduce confidence if major events are upcoming
                upcoming_hours = min([e['hours_until'] for e in economic_context['high_impact_upcoming']])
                if upcoming_hours < 2:
                    economic_confidence_adjustment -= 0.3
                elif upcoming_hours < 6:
                    economic_confidence_adjustment -= 0.15
            
            if economic_context.get('high_impact_recent'):
                # Adjust based on recent event outcomes
                recent_hours = min([e['hours_ago'] for e in economic_context['high_impact_recent']])
                if recent_hours < 1:
                    economic_confidence_adjustment -= 0.2  # High uncertainty immediately after
                elif recent_hours < 4:
                    economic_confidence_adjustment += 0.1  # Some clarity emerging
            
            # Apply economic bias
            if economic_context['economic_bias'] != 'NEUTRAL':
                base_signal = base_analysis.get('signal', 'HOLD')
                economic_bias = economic_context['economic_bias']
                
                if base_signal == 'BUY' and economic_bias == 'BULLISH':
                    economic_confidence_adjustment += 0.15
                elif base_signal == 'SELL' and economic_bias == 'BEARISH':
                    economic_confidence_adjustment += 0.15
                elif base_signal == 'BUY' and economic_bias == 'BEARISH':
                    economic_confidence_adjustment -= 0.2
                elif base_signal == 'SELL' and economic_bias == 'BULLISH':
                    economic_confidence_adjustment -= 0.2
            
            # Apply volatility expectations
            if economic_context.get('volatility_expectation') == 'HIGH':
                enhanced_analysis['position_size_multiplier'] = 0.7  # Reduce position size
                enhanced_analysis['stop_loss_multiplier'] = 1.5     # Wider stops
                
            # Update confidence
            original_confidence = base_analysis.get('confidence', 0.5)
            enhanced_confidence = max(0.1, min(1.0, original_confidence + economic_confidence_adjustment))
            enhanced_analysis['confidence'] = enhanced_confidence
            
            # Add economic analysis section
            enhanced_analysis['economic_analysis'] = {
                'bias': economic_context['economic_bias'],
                'volatility_expectation': economic_context.get('volatility_expectation', 'NORMAL'),
                'confidence_adjustment': economic_confidence_adjustment,
                'recent_events_count': len(economic_context.get('high_impact_recent', [])),
                'upcoming_events_count': len(economic_context.get('high_impact_upcoming', [])),
                'economic_confidence': economic_context.get('confidence', 0.5)
            }
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Error integrating economic analysis: {e}")
            return base_analysis


class EconomicSessionExpert(EconomicCalendarEnhancedAgent):
    """
    Enhanced SessionExpert agent with economic calendar integration.
    
    FIXED: Proper parent class initialization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize EconomicSessionExpert."""
        assigned_indicators = [
            "session_analysis_indicator",
            "session_volume_profile_indicator",
            "session_price_range_indicator",
            "time_of_day_indicator",
            "economic_event_impact_indicator",
            "economic_calendar_confluence_indicator",
            "news_sentiment_impact_indicator",
            "fundamental_momentum_indicator"
        ]
        
        super().__init__("EconomicSessionExpert", assigned_indicators, config)
        
    async def analyze_market(self, 
                           symbol: str, 
                           market_data: pd.DataFrame, 
                           market_conditions: MarketConditions) -> AnalysisResult:
        """Analyze market with session timing and economic calendar integration."""
        try:
            # Get economic context
            economic_context = await self.get_economic_context(symbol)
            
            # Base session analysis
            base_analysis = self._perform_session_analysis(symbol, market_data, market_conditions)
            
            # Integrate economic calendar analysis
            enhanced_analysis = self.integrate_economic_analysis(base_analysis, economic_context)
            
            # Generate final decision
            decision = self._make_session_decision(enhanced_analysis, economic_context)
            
            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision['signal'],
                confidence=decision['confidence'],
                reasoning=decision['reasoning'],
                indicators_used=self.assigned_indicators,
                technical_analysis=enhanced_analysis,
                risk_assessment=decision['risk_assessment'],
                supporting_data={'economic_context': economic_context}
            )
            
        except Exception as e:
            logger.error(f"Error in EconomicSessionExpert analysis: {e}")
            raise AgentError(f"Session analysis failed: {e}")
    
    def _perform_session_analysis(self, 
                                symbol: str, 
                                market_data: pd.DataFrame, 
                                market_conditions: MarketConditions) -> Dict[str, Any]:
        """Perform base session analysis."""
        current_time = datetime.utcnow()
        
        return {
            'signal': 'HOLD',
            'confidence': 0.6,
            'session': self._determine_current_session(current_time),
            'session_strength': 0.7,
            'volume_profile': 'NORMAL',
            'price_action': 'RANGING'
        }
    
    def _determine_current_session(self, current_time: datetime) -> str:
        """Determine current trading session."""
        hour = current_time.hour
        
        if 22 <= hour or hour < 5:
            return 'SYDNEY'
        elif 5 <= hour < 9:
            return 'TOKYO'
        elif 9 <= hour < 17:
            return 'LONDON'
        else:
            return 'NEW_YORK'
    
    def _make_session_decision(self, 
                             analysis: Dict[str, Any], 
                             economic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Make final session-based trading decision."""
        
        base_signal = analysis.get('signal', 'HOLD')
        base_confidence = analysis.get('confidence', 0.5)
        
        # Session-specific logic
        current_session = analysis.get('session', 'UNKNOWN')
        
        reasoning_parts = [
            f"Session: {current_session}",
            f"Base signal: {base_signal}",
            f"Economic bias: {economic_context.get('economic_bias', 'NEUTRAL')}"
        ]
        
        if economic_context.get('high_impact_upcoming'):
            reasoning_parts.append("Caution: Major economic events upcoming")
        
        risk_level = 'MEDIUM'
        if economic_context.get('volatility_expectation') == 'HIGH':
            risk_level = 'HIGH'
        
        return {
            'signal': base_signal,
            'confidence': base_confidence,
            'reasoning': '; '.join(reasoning_parts),
            'risk_assessment': {
                'risk_level': risk_level,
                'session_risk': current_session,
                'economic_risk': economic_context.get('volatility_expectation', 'NORMAL')
            }
        }


class EconomicRiskGenius(EconomicCalendarEnhancedAgent):
    """
    Enhanced RiskGenius agent with economic volatility prediction.
    
    FIXED: Proper parent class initialization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize EconomicRiskGenius."""
        assigned_indicators = [
            "average_true_range_indicator",
            "bollinger_bands_indicator",
            "historical_volatility_indicator",
            "volatility_prediction_indicator",
            "event_volatility_predictor"
        ]
        
        super().__init__("EconomicRiskGenius", assigned_indicators, config)
    
    async def analyze_market(self, 
                           symbol: str, 
                           market_data: pd.DataFrame, 
                           market_conditions: MarketConditions) -> AnalysisResult:
        """Analyze market with enhanced volatility and economic risk assessment."""
        try:
            # Get economic context for volatility prediction
            economic_context = await self.get_economic_context(symbol, lookback_hours=48, lookahead_hours=24)
            
            # Base volatility analysis
            base_analysis = self._perform_volatility_analysis(symbol, market_data, market_conditions)
            
            # Economic volatility prediction
            economic_volatility = self._predict_economic_volatility(economic_context)
            
            # Integrate analyses
            enhanced_analysis = self.integrate_economic_analysis(base_analysis, economic_context)
            enhanced_analysis['economic_volatility'] = economic_volatility
            
            # Generate risk assessment
            risk_decision = self._make_risk_decision(enhanced_analysis, economic_context)
            
            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=risk_decision['risk_signal'],
                confidence=risk_decision['confidence'],
                reasoning=risk_decision['reasoning'],
                indicators_used=self.assigned_indicators,
                technical_analysis=enhanced_analysis,
                risk_assessment=risk_decision['risk_assessment'],
                supporting_data={'economic_context': economic_context}
            )
            
        except Exception as e:
            logger.error(f"Error in EconomicRiskGenius analysis: {e}")
            raise AgentError(f"Risk analysis failed: {e}")
    
    def _perform_volatility_analysis(self, 
                                   symbol: str, 
                                   market_data: pd.DataFrame, 
                                   market_conditions: MarketConditions) -> Dict[str, Any]:
        """Perform base volatility analysis."""
        return {
            'current_volatility': 0.15,
            'historical_volatility': 0.12,
            'volatility_trend': 'INCREASING',
            'volatility_percentile': 75,
            'signal': 'HOLD',
            'confidence': 0.6
        }
    
    def _predict_economic_volatility(self, economic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict volatility based on economic events."""
        
        base_volatility_multiplier = 1.0
        volatility_drivers = []
        
        # Analyze upcoming events
        for event in economic_context.get('high_impact_upcoming', []):
            hours_until = event['hours_until']
            
            if hours_until < 1:
                base_volatility_multiplier *= 1.8
                volatility_drivers.append(f"Major event in <1h: {event['event']}")
            elif hours_until < 6:
                base_volatility_multiplier *= 1.4
                volatility_drivers.append(f"Major event in <6h: {event['event']}")
            elif hours_until < 24:
                base_volatility_multiplier *= 1.2
                volatility_drivers.append(f"Major event in <24h: {event['event']}")
        
        # Analyze recent events
        for event in economic_context.get('high_impact_recent', []):
            hours_ago = event['hours_ago']
            
            if hours_ago < 2:
                base_volatility_multiplier *= 1.6
                volatility_drivers.append(f"Recent major event: {event['event']}")
            elif hours_ago < 8:
                base_volatility_multiplier *= 1.3
                volatility_drivers.append(f"Recent event settling: {event['event']}")
        
        return {
            'predicted_volatility_multiplier': min(base_volatility_multiplier, 3.0),
            'drivers': volatility_drivers,
            'confidence': 0.8 if volatility_drivers else 0.5
        }
    
    def _make_risk_decision(self, 
                          analysis: Dict[str, Any], 
                          economic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Make risk-based decision."""
        
        volatility_multiplier = analysis.get('economic_volatility', {}).get('predicted_volatility_multiplier', 1.0)
        
        if volatility_multiplier > 2.0:
            risk_signal = 'REDUCE_EXPOSURE'
            confidence = 0.9
        elif volatility_multiplier > 1.5:
            risk_signal = 'CAUTIOUS'
            confidence = 0.8
        elif volatility_multiplier < 0.8:
            risk_signal = 'INCREASE_EXPOSURE'
            confidence = 0.7
        else:
            risk_signal = 'NORMAL'
            confidence = 0.6
        
        return {
            'risk_signal': risk_signal,
            'confidence': confidence,
            'reasoning': f"Economic volatility multiplier: {volatility_multiplier:.2f}",
            'risk_assessment': {
                'volatility_forecast': volatility_multiplier,
                'risk_level': 'HIGH' if volatility_multiplier > 1.5 else 'MEDIUM',
                'position_size_adjustment': 1.0 / volatility_multiplier
            }
        }
