"""
Session Expert Agent for the AUJ Platform.

This agent specializes in trading session analysis and market timing.
It focuses on 20 timing and session-specific indicators.
"""

from datetime import datetime, time
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from decimal import Decimal
import yaml
import os

from .base_agent import BaseAgent, AnalysisResult, AgentState
from ..core.data_contracts import MarketConditions, TradeDirection, ConfidenceLevel
from ..core.exceptions import AgentError, ValidationError
from ..core.logging_setup import get_logger
from ..core.unified_config import UnifiedConfigManager

logger = get_logger(__name__)


class SessionExpert(BaseAgent):
    """
    Session Expert Agent - Trading session analysis and market timing.

    Specializes in:
    - Trading session analysis (Asian, European, American)
    - Market opening/closing effects
    - Time-based momentum patterns
    - Session overlap opportunities
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_manager: Optional[UnifiedConfigManager] = None):
        """Initialize the Session Expert Agent."""

        assigned_indicators = [
            # ALL Gann Methods (12)
            "gann_angles_indicator",
            "gann_box_indicator",
            "gann_fan_indicator",
            "gann_grid_indicator",
            "gann_pattern_detector",
            "gann_price_time_indicator",
            "gann_square_indicator",
            "gann_square_of_nine_indicator",
            "gann_time_cycle_indicator",
            "gann_time_price_square_indicator",
            "price_time_relationships_indicator",
            "square_of_nine_calculator_indicator",

            # Economic Indicators (5)
            "economic_calendar_confluence_indicator",
            "economic_event_impact_indicator",
            "event_volatility_predictor",
            "fundamental_momentum_indicator",
            "news_sentiment_impact_indicator"
        ]

        super().__init__(
            name="SessionExpert",
            specialization="Trading session analysis and market timing",
            assigned_indicators=assigned_indicators,
            config=config,
            config_manager=config_manager
        )

        # Session timing configuration from unified config
        self.asian_session = {
            "start": time(*map(int, self.config_manager.get_str('agents.session_expert.session_times.asian_start', '23:00').split(':'))),
            "end": time(*map(int, self.config_manager.get_str('agents.session_expert.session_times.asian_end', '08:00').split(':')))
        }
        self.european_session = {
            "start": time(*map(int, self.config_manager.get_str('agents.session_expert.session_times.european_start', '07:00').split(':'))),
            "end": time(*map(int, self.config_manager.get_str('agents.session_expert.session_times.european_end', '16:00').split(':')))
        }
        self.american_session = {
            "start": time(*map(int, self.config_manager.get_str('agents.session_expert.session_times.american_start', '13:00').split(':'))),
            "end": time(*map(int, self.config_manager.get_str('agents.session_expert.session_times.american_end', '22:00').split(':')))
        }

        self.session_overlap_bonus = self.config_manager.get_float('agents.session_expert.session_analysis.overlap_bonus', 1.2)
        self.opening_range_minutes = self.config_manager.get_int('agents.session_expert.session_analysis.opening_range_minutes', 30)

        logger.info(f"SessionExpert initialized with {len(assigned_indicators)} indicators")

    def _load_agent_config(self) -> Dict[str, Any]:
        """Load agent configuration from YAML file with fallback defaults."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'agents', 'session_expert.yaml'
            )

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded SessionExpert configuration from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file not found at {config_path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading SessionExpert configuration: {e}")

        # Fallback defaults
        return {
            'session_times': {
                'asian_start': '23:00',
                'asian_end': '08:00',
                'european_start': '07:00',
                'european_end': '16:00',
                'american_start': '13:00',
                'american_end': '22:00'
            },
            'session_analysis': {
                'overlap_bonus': 0.2,
                'opening_range_minutes': 30,
                'breakout_threshold': 0.6,
                'momentum_threshold': 0.5
            },
            'timing_weights': {
                'session_weight': 1.0,
                'overlap_weight': 1.3,
                'opening_weight': 1.2
            }
        }

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform session and timing analysis."""
        try:
            # Session Analysis
            session_analysis = self._analyze_trading_sessions(indicator_results)

            # Timing Analysis
            timing_analysis = self._analyze_market_timing(indicator_results)

            # Generate Decision
            decision = self._generate_session_decision(session_analysis, timing_analysis)

            # Calculate Confidence
            confidence = self._calculate_session_confidence(session_analysis, timing_analysis)

            # Generate Reasoning
            reasoning = self._generate_session_reasoning(decision, session_analysis, timing_analysis)

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "session_analysis": session_analysis,
                    "timing_analysis": timing_analysis
                },
                risk_assessment=self._assess_session_risk(session_analysis),
                supporting_data={
                    "current_session": session_analysis.get("current_session", "UNKNOWN"),
                    "session_momentum": session_analysis.get("momentum_score", 0.0),
                    "timing_score": timing_analysis.get("timing_score", 0.0)
                }
            )

        except Exception as e:
            logger.error(f"SessionExpert analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Session analysis failed: {str(e)}")

    def _analyze_trading_sessions(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current trading session characteristics."""
        session_analysis = {
            "current_session": "UNKNOWN",
            "session_momentum": "NEUTRAL",
            "session_volatility": "MEDIUM",
            "overlap_active": False,
            "momentum_score": 0.0,
            "session_strength": "MEDIUM"
        }

        # Current session detection
        if "market_session_indicator" in indicator_results:
            session_data = indicator_results["market_session_indicator"]
            session_analysis["current_session"] = session_data.get("current_session", "UNKNOWN")
            session_analysis["overlap_active"] = session_data.get("overlap_active", False)

        # Session momentum
        if "session_momentum_indicator" in indicator_results:
            momentum_data = indicator_results["session_momentum_indicator"]
            momentum_score = momentum_data.get("momentum", 0.0)
            session_analysis["momentum_score"] = momentum_score

            if momentum_score > 0.6:
                session_analysis["session_momentum"] = "STRONG_BULLISH"
            elif momentum_score < -0.6:
                session_analysis["session_momentum"] = "STRONG_BEARISH"
            elif momentum_score > 0.2:
                session_analysis["session_momentum"] = "BULLISH"
            elif momentum_score < -0.2:
                session_analysis["session_momentum"] = "BEARISH"

        # Session volatility
        if "session_volatility_indicator" in indicator_results:
            vol_data = indicator_results["session_volatility_indicator"]
            vol_level = vol_data.get("volatility_level", "MEDIUM")
            session_analysis["session_volatility"] = vol_level

        return session_analysis

    def _analyze_market_timing(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market timing factors."""
        timing_analysis = {
            "opening_breakout": False,
            "closing_effect": "NEUTRAL",
            "gap_analysis": "NO_GAP",
            "timing_score": 0.0,
            "optimal_entry_time": False,
            "vwap_position": "NEUTRAL"
        }

        # Opening range breakout
        if "opening_range_breakout_indicator" in indicator_results:
            orb_data = indicator_results["opening_range_breakout_indicator"]
            timing_analysis["opening_breakout"] = orb_data.get("breakout_detected", False)
            timing_analysis["optimal_entry_time"] = orb_data.get("optimal_entry", False)

        # Gap analysis
        if "gap_analysis_indicator" in indicator_results:
            gap_data = indicator_results["gap_analysis_indicator"]
            gap_type = gap_data.get("gap_type", "NO_GAP")
            timing_analysis["gap_analysis"] = gap_type

        # VWAP positioning
        if "volume_weighted_average_price_indicator" in indicator_results:
            vwap_data = indicator_results["volume_weighted_average_price_indicator"]
            vwap_position = vwap_data.get("position", "NEUTRAL")
            timing_analysis["vwap_position"] = vwap_position

        # Calculate overall timing score
        timing_analysis["timing_score"] = self._calculate_timing_score(timing_analysis)

        return timing_analysis

    def _calculate_timing_score(self, timing_analysis: Dict[str, Any]) -> float:
        """Calculate overall timing score."""
        score = 0.0

        # Opening breakout bonus
        if timing_analysis["opening_breakout"]:
            score += 0.3

        # Optimal entry time bonus
        if timing_analysis["optimal_entry_time"]:
            score += 0.2

        # VWAP positioning
        vwap_pos = timing_analysis["vwap_position"]
        if vwap_pos == "ABOVE":
            score += 0.1
        elif vwap_pos == "BELOW":
            score -= 0.1

        # Gap analysis
        gap_type = timing_analysis["gap_analysis"]
        if gap_type == "BULLISH_GAP":
            score += 0.15
        elif gap_type == "BEARISH_GAP":
            score -= 0.15

        return max(-1.0, min(1.0, score))

    def _generate_session_decision(self, session_analysis: Dict[str, Any], timing_analysis: Dict[str, Any]) -> str:
        """Generate session-based trading decision."""
        session_momentum = session_analysis["session_momentum"]
        timing_score = timing_analysis["timing_score"]
        overlap_active = session_analysis["overlap_active"]

        # Strong session momentum with good timing
        if session_momentum in ["STRONG_BULLISH", "STRONG_BEARISH"] and timing_score > 0.3:
            if session_momentum == "STRONG_BULLISH":
                return "BUY"
            else:
                return "SELL"

        # Session overlap with moderate signals
        if overlap_active and timing_score > 0.2:
            if session_momentum in ["BULLISH", "STRONG_BULLISH"]:
                return "BUY"
            elif session_momentum in ["BEARISH", "STRONG_BEARISH"]:
                return "SELL"

        # Opening breakout signals
        if timing_analysis["opening_breakout"] and timing_analysis["optimal_entry_time"]:
            if session_analysis["momentum_score"] > 0:
                return "BUY"
            elif session_analysis["momentum_score"] < 0:
                return "SELL"

        return "NO_SIGNAL"

    def _calculate_session_confidence(self, session_analysis: Dict[str, Any], timing_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in session analysis."""
        factors = []

        # Session momentum confidence
        momentum_score = abs(session_analysis.get("momentum_score", 0.0))
        factors.append(momentum_score * 0.4)

        # Timing score confidence
        timing_score = abs(timing_analysis.get("timing_score", 0.0))
        factors.append(timing_score * 0.3)

        # Session overlap bonus
        if session_analysis.get("overlap_active", False):
            factors.append(self.session_overlap_bonus)

        # Opening breakout confidence
        if timing_analysis.get("opening_breakout", False):
            factors.append(0.1)

        return min(sum(factors), 1.0)

    def _assess_session_risk(self, session_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess session-specific risks."""
        volatility = session_analysis.get("session_volatility", "MEDIUM")

        risk_level = "MEDIUM"
        if volatility == "HIGH":
            risk_level = "HIGH"
        elif volatility == "LOW":
            risk_level = "LOW"

        return {
            "session_risk": risk_level,
            "volatility_risk": volatility,
            "timing_risk": "MEDIUM"
        }

    def _generate_session_reasoning(self, decision: str, session_analysis: Dict[str, Any], timing_analysis: Dict[str, Any]) -> str:
        """Generate reasoning for session analysis."""
        parts = [f"Session analysis decision: {decision}."]

        current_session = session_analysis.get("current_session", "UNKNOWN")
        parts.append(f"Current session: {current_session}.")

        session_momentum = session_analysis.get("session_momentum", "NEUTRAL")
        parts.append(f"Session momentum: {session_momentum}.")

        if timing_analysis.get("opening_breakout", False):
            parts.append("Opening range breakout detected.")

        if session_analysis.get("overlap_active", False):
            parts.append("Session overlap active.")

        return " ".join(parts)

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually used in analysis."""
        return [ind for ind in self.assigned_indicators if ind in indicator_results]

    def get_required_data_types(self) -> List[str]:
        """Define required data types."""
        return ["OHLCV", "TICK"]

    def get_minimum_data_points(self) -> int:
        """Define minimum data points needed."""
        return 30  # Need recent data for session analysis
