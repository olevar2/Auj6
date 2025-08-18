"""
Base Agent Class for the AUJ Platform.

This module defines the abstract base class that all expert agents must implement,
providing a standardized interface for agent behavior, decision making, and performance tracking.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import pandas as pd
from decimal import Decimal
import uuid

from ..core.exceptions import AgentError, ValidationError, IndicatorCalculationError
from ..core.data_contracts import (
    TradeSignal, AgentDecision, MarketConditions, AgentRank,
    ConfidenceLevel, TradeDirection, MarketRegime
)
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)


class AgentState(str, Enum):
    """States an agent can be in."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    ANALYZING = "ANALYZING"
    DECISION_PENDING = "DECISION_PENDING"
    ERROR = "ERROR"
    LEARNING = "LEARNING"


class AnalysisResult:
    """Container for agent analysis results."""

    def __init__(self,
                 agent_name: str,
                 symbol: str,
                 decision: str,
                 confidence: float,
                 reasoning: str,
                 indicators_used: List[str],
                 technical_analysis: Dict[str, Any],
                 risk_assessment: Dict[str, Any],
                 supporting_data: Optional[Dict[str, Any]] = None):
        """
        Initialize analysis result.

        Args:
            agent_name: Name of the agent
            symbol: Trading symbol analyzed
            decision: Trading decision (BUY, SELL, HOLD, NO_SIGNAL)
            confidence: Confidence level (0.0 to 1.0)
            reasoning: Explanation of the decision
            indicators_used: List of indicators used in analysis
            technical_analysis: Technical analysis data
            risk_assessment: Risk assessment data
            supporting_data: Additional supporting data
        """
        self.agent_name = agent_name
        self.symbol = symbol
        self.decision = decision
        self.confidence = confidence
        self.reasoning = reasoning
        self.indicators_used = indicators_used
        self.technical_analysis = technical_analysis
        self.risk_assessment = risk_assessment
        self.supporting_data = supporting_data or {}
        self.timestamp = datetime.utcnow()
        self.analysis_id = str(uuid.uuid4())

    def to_agent_decision(self) -> AgentDecision:
        """Convert to AgentDecision data contract."""
        return AgentDecision(
            agent_name=self.agent_name,
            timestamp=self.timestamp,
            symbol=self.symbol,
            decision=self.decision,
            confidence=self.confidence,
            reasoning=self.reasoning,
            indicators_analyzed=self.indicators_used,
            technical_analysis=self.technical_analysis,
            risk_assessment=self.risk_assessment,
            supporting_data=self.supporting_data
        )


class BaseAgent(ABC):
    """
    Abstract base class for all expert agents in the AUJ Platform.

    Each agent specializes in a specific aspect of market analysis and must implement
    the analyze_market method to provide trading recommendations.
    """

    def __init__(self,
                 name: str,
                 specialization: str,
                 assigned_indicators: List[str],
                 config_manager: UnifiedConfigManager,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.

        Args:
            name: Agent name (e.g., "StrategyExpert", "RiskGenius")
            specialization: Area of expertise description
            assigned_indicators: List of indicators this agent uses
            config_manager: Unified configuration manager instance
            config: Agent-specific configuration (deprecated, use config_manager)
        """
        self.name = name
        self.specialization = specialization
        self.assigned_indicators = assigned_indicators
        self.config_manager = config_manager
        # Backward compatibility - will be removed in next version
        self.config = config or {}

        # Agent state and performance tracking
        self.state = AgentState.INACTIVE
        self.current_rank = AgentRank.GAMMA  # Start at lowest rank
        self.total_analyses = 0
        self.successful_analyses = 0
        self.total_signals_generated = 0
        self.successful_signals = 0
        self.last_analysis_time: Optional[datetime] = None
        self.last_error: Optional[str] = None

        # Performance metrics
        self.analysis_times: List[float] = []
        self.confidence_scores: List[float] = []
        self.recent_decisions: List[AnalysisResult] = []
        self.max_recent_decisions = 100

        # Learning and adaptation
        self.learning_rate = 0.01
        self.performance_decay_factor = 0.95
        self.min_confidence_threshold = 0.3
        
        # Messaging integration
        self.message_broker = None
        self.messaging_enabled = False

        logger.info(f"Initialized agent: {name} - {specialization}")

    @abstractmethod
    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """
        Perform market analysis and generate trading recommendation.

        This is the core method that each agent must implement based on their specialization.

        Args:
            symbol: Trading symbol to analyze
            market_data: Available market data (OHLCV, tick, etc.)
            market_conditions: Current market conditions assessment
            indicator_results: Calculated indicator values for assigned indicators

        Returns:
            AnalysisResult containing the agent's recommendation

        Raises:
            AgentError: If analysis fails
        """
        pass

    @abstractmethod
    def get_required_data_types(self) -> List[str]:
        """
        Define what types of market data this agent requires.

        Returns:
            List of required data types (e.g., ["OHLCV", "TICK"])
        """
        pass

    @abstractmethod
    def get_minimum_data_points(self) -> int:
        """
        Define minimum number of data points needed for analysis.

        Returns:
            Minimum required data points
        """
        pass

    def validate_inputs(self,
                       symbol: str,
                       market_data: Dict[str, pd.DataFrame],
                       indicator_results: Dict[str, Any]) -> bool:
        """
        Validate inputs before analysis.

        Args:
            symbol: Trading symbol
            market_data: Market data
            indicator_results: Indicator results

        Returns:
            True if inputs are valid

        Raises:
            ValidationError: If validation fails
        """
        # Check symbol
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Invalid symbol provided")

        # Check required data types
        required_data_types = self.get_required_data_types()
        for data_type in required_data_types:
            if data_type not in market_data:
                raise ValidationError(f"Missing required data type: {data_type}")

            if market_data[data_type].empty:
                raise ValidationError(f"Empty data for type: {data_type}")

            if len(market_data[data_type]) < self.get_minimum_data_points():
                raise ValidationError(f"Insufficient data points for {data_type}")

        # Check assigned indicators
        missing_indicators = set(self.assigned_indicators) - set(indicator_results.keys())
        if missing_indicators:
            logger.warning(f"Missing indicators for {self.name}: {missing_indicators}")

        return True

    async def perform_analysis(self,
                             symbol: str,
                             market_data: Dict[str, pd.DataFrame],
                             market_conditions: MarketConditions,
                             indicator_results: Dict[str, Any]) -> Optional[AnalysisResult]:
        """
        Main analysis method with error handling and performance tracking.

        Args:
            symbol: Trading symbol
            market_data: Market data
            market_conditions: Market conditions
            indicator_results: Indicator results

        Returns:
            AnalysisResult or None if analysis fails
        """
        start_time = datetime.utcnow()
        self.state = AgentState.ANALYZING

        try:
            # Validate inputs
            self.validate_inputs(symbol, market_data, indicator_results)

            # Perform the actual analysis
            result = await self.analyze_market(symbol, market_data, market_conditions, indicator_results)

            # Validate result
            if not self._validate_analysis_result(result):
                raise AgentError(f"Invalid analysis result from {self.name}")

            # Update performance metrics
            analysis_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_performance_metrics(result, analysis_time, success=True)

            # Store recent decision
            self._store_recent_decision(result)

            self.state = AgentState.ACTIVE
            self.last_analysis_time = datetime.utcnow()

            logger.debug(f"Agent {self.name} completed analysis for {symbol}")
            return result

        except Exception as e:
            # Handle error
            analysis_time = (datetime.utcnow() - start_time).total_seconds()
            self._handle_analysis_error(e, analysis_time)
            return None

    def _validate_analysis_result(self, result: AnalysisResult) -> bool:
        """Validate the analysis result."""
        if not isinstance(result, AnalysisResult):
            return False

        if result.confidence < 0.0 or result.confidence > 1.0:
            return False

        if result.decision not in ["BUY", "SELL", "HOLD", "NO_SIGNAL"]:
            return False

        if not result.reasoning or len(result.reasoning.strip()) < 10:
            return False

        return True

    def _update_performance_metrics(self, result: AnalysisResult, analysis_time: float, success: bool):
        """Update agent performance metrics."""
        self.total_analyses += 1
        if success:
            self.successful_analyses += 1

        # Track analysis times
        self.analysis_times.append(analysis_time)
        if len(self.analysis_times) > 100:  # Keep last 100 times
            self.analysis_times.pop(0)

        # Track confidence scores
        if result:
            self.confidence_scores.append(result.confidence)
            if len(self.confidence_scores) > 100:  # Keep last 100 scores
                self.confidence_scores.pop(0)

    def _store_recent_decision(self, result: AnalysisResult):
        """Store recent decision for learning purposes."""
        self.recent_decisions.append(result)
        if len(self.recent_decisions) > self.max_recent_decisions:
            self.recent_decisions.pop(0)

    def _handle_analysis_error(self, error: Exception, analysis_time: float):
        """Handle analysis errors."""
        error_msg = f"Agent {self.name} analysis failed: {str(error)}"
        self.last_error = error_msg
        self.state = AgentState.ERROR

        # Update performance metrics
        self._update_performance_metrics(None, analysis_time, success=False)

        logger.error(error_msg)

        if isinstance(error, (AgentError, ValidationError)):
            raise
        else:
            raise AgentError(f"Analysis failed: {str(error)}")

    def generate_trade_signal(self, analysis_result: AnalysisResult) -> Optional[TradeSignal]:
        """
        Generate a trade signal from analysis result.

        Args:
            analysis_result: Result of market analysis

        Returns:
            TradeSignal if decision warrants a signal, None otherwise
        """
        # Only generate signals for BUY/SELL decisions with sufficient confidence
        if analysis_result.decision not in ["BUY", "SELL"]:
            return None

        if analysis_result.confidence < self.min_confidence_threshold:
            logger.debug(f"Agent {self.name} confidence too low: {analysis_result.confidence}")
            return None

        # Determine confidence level
        confidence_level = self._determine_confidence_level(analysis_result.confidence)

        # Create trade signal
        signal = TradeSignal(
            symbol=analysis_result.symbol,
            direction=TradeDirection.BUY if analysis_result.decision == "BUY" else TradeDirection.SELL,
            confidence=analysis_result.confidence,
            confidence_level=confidence_level,
            timeframe=self.config_manager.get_str(f'agents.{self.name.lower()}.default_timeframe', '1H'),
            strategy=f"{self.name}_Strategy",
            generating_agent=self.name,
            indicators_used=analysis_result.indicators_used,
            metadata={
                "analysis_id": analysis_result.analysis_id,
                "reasoning": analysis_result.reasoning,
                "technical_analysis": analysis_result.technical_analysis,
                "risk_assessment": analysis_result.risk_assessment
            }
        )

        self.total_signals_generated += 1
        logger.info(f"Agent {self.name} generated signal: {signal.direction.value} {signal.symbol}")

        return signal

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to ConfidenceLevel enum."""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.65:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.35:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def update_rank(self, new_rank: AgentRank):
        """Update agent's hierarchy rank."""
        old_rank = self.current_rank
        self.current_rank = new_rank
        logger.info(f"Agent {self.name} rank updated: {old_rank.value} -> {new_rank.value}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        success_rate = self.successful_analyses / max(self.total_analyses, 1)
        avg_analysis_time = sum(self.analysis_times) / max(len(self.analysis_times), 1)
        avg_confidence = sum(self.confidence_scores) / max(len(self.confidence_scores), 1)

        return {
            "agent_name": self.name,
            "specialization": self.specialization,
            "current_rank": self.current_rank.value,
            "state": self.state.value,
            "total_analyses": self.total_analyses,
            "successful_analyses": self.successful_analyses,
            "success_rate": success_rate,
            "total_signals_generated": self.total_signals_generated,
            "successful_signals": self.successful_signals,
            "signal_success_rate": self.successful_signals / max(self.total_signals_generated, 1),
            "avg_analysis_time_seconds": avg_analysis_time,
            "avg_confidence": avg_confidence,
            "last_analysis_time": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "last_error": self.last_error,
            "assigned_indicators_count": len(self.assigned_indicators),
            "recent_decisions_count": len(self.recent_decisions)
        }

    def get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get recent performance metrics."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        recent_decisions = [
            d for d in self.recent_decisions
            if d.timestamp >= cutoff_time
        ]

        if not recent_decisions:
            return {
                "agent_name": self.name,
                "period_days": days,
                "total_decisions": 0,
                "avg_confidence": 0.0,
                "decision_distribution": {}
            }

        # Analyze recent decisions
        decision_counts = {}
        confidence_sum = 0

        for decision in recent_decisions:
            decision_counts[decision.decision] = decision_counts.get(decision.decision, 0) + 1
            confidence_sum += decision.confidence

        return {
            "agent_name": self.name,
            "period_days": days,
            "total_decisions": len(recent_decisions),
            "avg_confidence": confidence_sum / len(recent_decisions),
            "decision_distribution": decision_counts,
            "most_common_decision": max(decision_counts.items(), key=lambda x: x[1])[0] if decision_counts else None
        }

    def adapt_parameters(self, performance_feedback: Dict[str, Any]):
        """
        Adapt agent parameters based on performance feedback.

        Args:
            performance_feedback: Performance metrics and feedback
        """
        # Adjust confidence threshold based on recent performance
        recent_success_rate = performance_feedback.get('recent_success_rate', 0.5)

        if recent_success_rate > 0.7:
            # Good performance, can be slightly more aggressive
            self.min_confidence_threshold = max(0.2, self.min_confidence_threshold - 0.05)
        elif recent_success_rate < 0.4:
            # Poor performance, be more conservative
            self.min_confidence_threshold = min(0.6, self.min_confidence_threshold + 0.05)

        logger.debug(f"Agent {self.name} adapted confidence threshold to {self.min_confidence_threshold}")

    def reset_performance_metrics(self):
        """Reset performance metrics (for testing or reinitialization)."""
        self.total_analyses = 0
        self.successful_analyses = 0
        self.total_signals_generated = 0
        self.successful_signals = 0
        self.analysis_times.clear()
        self.confidence_scores.clear()
        self.recent_decisions.clear()
        self.last_error = None
        self.state = AgentState.INACTIVE

        logger.info(f"Performance metrics reset for agent {self.name}")
    
    def set_message_broker(self, message_broker):
        """Set the message broker for agent communication."""
        self.message_broker = message_broker
        self.messaging_enabled = message_broker is not None
        if self.messaging_enabled:
            logger.info(f"Message broker enabled for agent {self.name}")
    
    async def publish_analysis_result(self, analysis_result: AnalysisResult):
        """Publish analysis result to message queue if messaging is enabled."""
        if not self.messaging_enabled or not self.message_broker:
            return
        
        try:
            # Prepare message
            message_data = {
                "agent_name": self.name,
                "analysis_id": analysis_result.analysis_id,
                "symbol": analysis_result.symbol,
                "decision": analysis_result.decision,
                "confidence": analysis_result.confidence,
                "timestamp": analysis_result.timestamp.isoformat(),
                "reasoning": analysis_result.reasoning[:500],  # Limit reasoning length
                "indicators_used": analysis_result.indicators_used,
                "specialization": self.specialization,
                "rank": self.current_rank.value
            }
            
            # Determine routing key based on confidence and decision
            routing_key = "trading.signal.normal"
            if analysis_result.confidence >= 0.8 and analysis_result.decision in ["BUY", "SELL"]:
                routing_key = "trading.signal.high"
            
            # Publish message
            await self.message_broker.publish_message(
                message_body=str(message_data),
                exchange_name="auj.platform",
                routing_key=routing_key,
                priority=min(10, int(analysis_result.confidence * 10))
            )
            
            logger.debug(f"Agent {self.name} published analysis result to {routing_key}")
            
        except Exception as e:
            logger.error(f"Failed to publish analysis result for agent {self.name}: {e}")
    
    async def publish_agent_status(self, status_type: str = "status_update"):
        """Publish agent status update to message queue."""
        if not self.messaging_enabled or not self.message_broker:
            return
        
        try:
            status_data = {
                "agent_name": self.name,
                "status_type": status_type,
                "state": self.state.value,
                "rank": self.current_rank.value,
                "specialization": self.specialization,
                "timestamp": datetime.utcnow().isoformat(),
                "performance_summary": {
                    "success_rate": self.successful_analyses / max(self.total_analyses, 1),
                    "total_analyses": self.total_analyses,
                    "avg_confidence": sum(self.confidence_scores) / max(len(self.confidence_scores), 1)
                }
            }
            
            await self.message_broker.publish_message(
                message_body=str(status_data),
                exchange_name="auj.platform",
                routing_key=f"system.status.agent.{self.name.lower()}",
                priority=3
            )
            
            logger.debug(f"Agent {self.name} published status update")
            
        except Exception as e:
            logger.error(f"Failed to publish status for agent {self.name}: {e}")

    def is_ready_for_analysis(self) -> bool:
        """Check if agent is ready to perform analysis."""
        return (
            self.state in [AgentState.ACTIVE, AgentState.INACTIVE] and
            self.current_rank != AgentRank.INACTIVE and
            len(self.assigned_indicators) > 0
        )

    def get_assigned_indicators(self) -> List[str]:
        """Get list of assigned indicators."""
        return self.assigned_indicators.copy()

    def add_indicator(self, indicator_name: str):
        """Add an indicator to the agent's toolkit."""
        if indicator_name not in self.assigned_indicators:
            self.assigned_indicators.append(indicator_name)
            logger.info(f"Added indicator {indicator_name} to agent {self.name}")

    def remove_indicator(self, indicator_name: str):
        """Remove an indicator from the agent's toolkit."""
        if indicator_name in self.assigned_indicators:
            self.assigned_indicators.remove(indicator_name)
            logger.info(f"Removed indicator {indicator_name} from agent {self.name}")

    def __str__(self) -> str:
        return f"{self.name} ({self.specialization}) - {self.current_rank.value}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', rank='{self.current_rank.value}')"
