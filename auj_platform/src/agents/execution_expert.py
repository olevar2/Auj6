"""
Execution Expert Agent for the AUJ Platform.

This agent specializes in trade execution optimization and order management.
It focuses on 15 execution-specific indicators and slippage analysis.
"""

from datetime import datetime
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


class ExecutionExpert(BaseAgent):
    """
    Execution Expert Agent - Trade execution optimization and order management.

    Specializes in:
    - Order book analysis and liquidity assessment
    - Slippage prediction and minimization
    - Optimal execution timing
    - Market impact analysis
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_manager: Optional[UnifiedConfigManager] = None, messaging_service: Optional[Any] = None):
        """Initialize the Execution Expert Agent."""
        assigned_indicators = [
            # Execution-Focused Volume Indicators
            "accumulation_distribution_line_indicator",
            "anchored_vwap_indicator",
            "chaikin_money_flow_indicator",
            "ease_of_movement_indicator",
            "klinger_oscillator_indicator",
            "negative_volume_index_indicator",
            "on_balance_volume_indicator",
            "positive_volume_index_indicator",
            "price_volume_rank_indicator",
            "price_volume_trend_indicator",
            "volume_breakout_detector",
            "volume_delta_indicator",
            "volume_oscillator_indicator",
            "volume_profile_indicator",
            "volume_rate_of_change_indicator",
            "vpt_trend_state_indicator",
            "vwap_indicator",

            # Execution-Related Signals
            "accumulation_distribution_signal"
        ]

        super().__init__(
            name="ExecutionExpert",
            specialization="Trade execution optimization and order management",
            assigned_indicators=assigned_indicators,
            config=config,
            config_manager=config_manager
        )

        # Store messaging service
        self.messaging_service = messaging_service

        # Load configuration from YAML file
        self._load_agent_config()

        logger.info(f"ExecutionExpert initialized with {len(assigned_indicators)} indicators")

    def _load_agent_config(self):
        """Load agent-specific configuration from YAML file."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'agents', 'execution_expert.yaml'
            )

            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    self.agent_config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Configuration file not found: {config_path}, using defaults")
                self.agent_config = {}

        except Exception as e:
            logger.error(f"Error loading agent configuration: {e}, using defaults")
            self.agent_config = {}

        # Set parameters from unified config with fallback defaults
        self.max_market_impact = self.config_manager.get_float('agents.execution_expert.max_market_impact', 0.05)
        self.min_liquidity_threshold = self.config_manager.get_int('agents.execution_expert.min_liquidity_threshold', 10000)
        self.optimal_spread_threshold = self.config_manager.get_float('agents.execution_expert.optimal_spread_threshold', 0.002)
        self.slippage_tolerance = self.config_manager.get_float('agents.execution_expert.slippage_tolerance', 0.001)

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform execution analysis and optimization with graceful fallback for missing data."""
        try:
            # Check data availability and determine analysis mode
            order_book_available = self._check_data_availability(market_data, 'ORDER_BOOK')
            tick_data_available = self._check_data_availability(market_data, 'TICK')
            ohlcv_available = self._check_data_availability(market_data, 'OHLCV')

            # Validate minimum data requirements
            if not ohlcv_available:
                raise AgentError(f"No OHLCV data available for {symbol} - cannot perform any analysis")

            # Log data availability and switch to fallback mode if needed
            missing_data = []
            if not order_book_available:
                missing_data.append('ORDER_BOOK')
            if not tick_data_available:
                missing_data.append('TICK')

            if missing_data:
                logger.warning(f"ExecutionExpert for {symbol}: Missing {', '.join(missing_data)} data. "
                             f"Operating in fallback mode using OHLCV data only. "
                             f"Analysis accuracy will be reduced.")

            # Perform analysis based on available data
            if order_book_available and tick_data_available:
                # Full execution analysis with all data types
                liquidity_analysis = self._analyze_liquidity(indicator_results)
                execution_analysis = self._analyze_execution_quality(indicator_results)
                impact_analysis = self._analyze_market_impact(indicator_results)
            else:
                # Fallback analysis using OHLCV data
                liquidity_analysis = self._analyze_liquidity_fallback(indicator_results, market_data.get('OHLCV'))
                execution_analysis = self._analyze_execution_quality_fallback(indicator_results, market_data.get('OHLCV'))
                impact_analysis = self._analyze_market_impact_fallback(indicator_results, market_data.get('OHLCV'))

                # Adjust confidence for fallback mode
                liquidity_analysis['data_quality'] = 'FALLBACK'
                execution_analysis['data_quality'] = 'FALLBACK'
                impact_analysis['data_quality'] = 'FALLBACK'

            # Generate Decision
            decision = self._generate_execution_decision(liquidity_analysis, execution_analysis, impact_analysis)

            # Calculate Confidence (reduced for fallback mode)
            confidence = self._calculate_execution_confidence(liquidity_analysis, execution_analysis, impact_analysis)
            if missing_data:
                confidence *= 0.7  # Reduce confidence by 30% when using fallback

            # Generate Reasoning
            reasoning = self._generate_execution_reasoning(decision, liquidity_analysis, execution_analysis, impact_analysis)
            if missing_data:
                reasoning += f" Note: Analysis performed in fallback mode due to missing {', '.join(missing_data)} data."

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "liquidity_analysis": liquidity_analysis,
                    "execution_analysis": execution_analysis,
                    "impact_analysis": impact_analysis
                },
                risk_assessment=self._assess_execution_risk(liquidity_analysis, impact_analysis),
                supporting_data={
                    "liquidity_score": liquidity_analysis.get("liquidity_score", 0.0),
                    "execution_quality": execution_analysis.get("quality_score", 0.0),
                    "market_impact": impact_analysis.get("expected_impact", 0.0),
                    "optimal_timing": execution_analysis.get("optimal_timing", False),
                    "fallback_mode": bool(missing_data)
                }
            )

        except Exception as e:
            logger.error(f"ExecutionExpert analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Execution analysis failed: {str(e)}")

    def _check_data_availability(self, market_data: Dict[str, pd.DataFrame], data_type: str) -> bool:
        """Check if specific data type is available and valid."""
        data = market_data.get(data_type)
        return data is not None and not data.empty

    def _analyze_liquidity(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market liquidity conditions."""
        liquidity_analysis = {
            "liquidity_score": 0.0,
            "bid_ask_spread": 0.0,
            "order_book_depth": "UNKNOWN",
            "volume_profile": "NORMAL",
            "liquidity_risk": "MEDIUM",
            "optimal_size": 0.0
        }

        # Liquidity indicator
        if "liquidity_indicator" in indicator_results:
            liq_data = indicator_results["liquidity_indicator"]
            liquidity_analysis["liquidity_score"] = liq_data.get("liquidity_score", 0.0)

            # Determine liquidity risk
            if liquidity_analysis["liquidity_score"] > 0.8:
                liquidity_analysis["liquidity_risk"] = "LOW"
            elif liquidity_analysis["liquidity_score"] < 0.3:
                liquidity_analysis["liquidity_risk"] = "HIGH"

        # Bid-ask spread analysis
        if "bid_ask_spread_indicator" in indicator_results:
            spread_data = indicator_results["bid_ask_spread_indicator"]
            liquidity_analysis["bid_ask_spread"] = spread_data.get("spread", 0.0)

        # Order book depth
        if "order_book_depth_indicator" in indicator_results:
            depth_data = indicator_results["order_book_depth_indicator"]
            depth_level = depth_data.get("depth_level", "UNKNOWN")
            liquidity_analysis["order_book_depth"] = depth_level

        # Volume profile analysis
        if "volume_profile_indicator" in indicator_results:
            vol_data = indicator_results["volume_profile_indicator"]
            vol_profile = vol_data.get("profile_type", "NORMAL")
            liquidity_analysis["volume_profile"] = vol_profile
            liquidity_analysis["optimal_size"] = vol_data.get("optimal_trade_size", 0.0)

        return liquidity_analysis

    def _analyze_execution_quality(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution quality metrics."""
        execution_analysis = {
            "quality_score": 0.0,
            "fill_ratio": 1.0,
            "price_improvement": 0.0,
            "execution_shortfall": 0.0,
            "twap_performance": "NEUTRAL",
            "optimal_timing": False,
            "execution_recommendation": "MARKET"
        }

        # Execution quality score
        if "execution_quality_indicator" in indicator_results:
            qual_data = indicator_results["execution_quality_indicator"]
            execution_analysis["quality_score"] = qual_data.get("quality_score", 0.0)
            execution_analysis["optimal_timing"] = qual_data.get("optimal_timing", False)

        # Fill ratio analysis
        if "fill_ratio_indicator" in indicator_results:
            fill_data = indicator_results["fill_ratio_indicator"]
            execution_analysis["fill_ratio"] = fill_data.get("fill_ratio", 1.0)

        # Price improvement
        if "price_improvement_indicator" in indicator_results:
            improvement_data = indicator_results["price_improvement_indicator"]
            execution_analysis["price_improvement"] = improvement_data.get("improvement", 0.0)

        # Execution shortfall
        if "execution_shortfall_indicator" in indicator_results:
            shortfall_data = indicator_results["execution_shortfall_indicator"]
            execution_analysis["execution_shortfall"] = shortfall_data.get("shortfall", 0.0)

        # TWAP performance
        if "twap_vwap_ratio_indicator" in indicator_results:
            twap_data = indicator_results["twap_vwap_ratio_indicator"]
            twap_ratio = twap_data.get("ratio", 1.0)
            if twap_ratio > 1.02:
                execution_analysis["twap_performance"] = "UNDERPERFORMING"
            elif twap_ratio < 0.98:
                execution_analysis["twap_performance"] = "OUTPERFORMING"

        # Determine execution recommendation
        execution_analysis["execution_recommendation"] = self._determine_execution_method(execution_analysis)

        return execution_analysis

    def _analyze_market_impact(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market impact and slippage."""
        impact_analysis = {
            "expected_impact": 0.0,
            "slippage_estimate": 0.0,
            "impact_risk": "MEDIUM",
            "microstructure_signal": "NEUTRAL",
            "order_flow_direction": "BALANCED",
            "impact_tolerance": True
        }

        # Market impact indicator
        if "market_impact_indicator" in indicator_results:
            impact_data = indicator_results["market_impact_indicator"]
            impact_analysis["expected_impact"] = impact_data.get("expected_impact", 0.0)

            # Check if impact is within tolerance
            if impact_analysis["expected_impact"] > self.max_market_impact:
                impact_analysis["impact_tolerance"] = False
                impact_analysis["impact_risk"] = "HIGH"
            elif impact_analysis["expected_impact"] < self.max_market_impact * 0.5:
                impact_analysis["impact_risk"] = "LOW"

        # Slippage analysis
        if "slippage_indicator" in indicator_results:
            slip_data = indicator_results["slippage_indicator"]
            impact_analysis["slippage_estimate"] = slip_data.get("expected_slippage", 0.0)

        # Microstructure analysis
        if "market_microstructure_indicator" in indicator_results:
            micro_data = indicator_results["market_microstructure_indicator"]
            impact_analysis["microstructure_signal"] = micro_data.get("signal", "NEUTRAL")

        # Order flow analysis
        if "order_flow_indicator" in indicator_results:
            flow_data = indicator_results["order_flow_indicator"]
            impact_analysis["order_flow_direction"] = flow_data.get("direction", "BALANCED")

        return impact_analysis

    def _analyze_liquidity_fallback(self, indicator_results: Dict[str, Any], ohlcv_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Fallback liquidity analysis using OHLCV data only."""
        liquidity_analysis = {
            "liquidity_score": 0.5,  # Default moderate liquidity
            "bid_ask_spread": 0.001,  # Estimated spread
            "order_book_depth": "UNKNOWN",
            "volume_profile": "NORMAL",
            "liquidity_risk": "MEDIUM",
            "optimal_size": 0.0,
            "data_quality": "FALLBACK"
        }

        # Estimate liquidity from volume patterns
        if ohlcv_data is not None and not ohlcv_data.empty:
            recent_volume = ohlcv_data['volume'].tail(10)
            avg_volume = recent_volume.mean()
            volume_std = recent_volume.std()
            latest_volume = recent_volume.iloc[-1]

            # Volume-based liquidity scoring
            if latest_volume > avg_volume + volume_std:
                liquidity_analysis["liquidity_score"] = 0.7
                liquidity_analysis["liquidity_risk"] = "LOW"
                liquidity_analysis["volume_profile"] = "HIGH"
            elif latest_volume < avg_volume - volume_std:
                liquidity_analysis["liquidity_score"] = 0.3
                liquidity_analysis["liquidity_risk"] = "HIGH"
                liquidity_analysis["volume_profile"] = "LOW"

            # Estimate spread from high-low range
            latest = ohlcv_data.iloc[-1]
            hl_spread = (latest['high'] - latest['low']) / latest['close']
            liquidity_analysis["bid_ask_spread"] = max(0.0005, hl_spread * 0.3)  # Conservative estimate

            # Estimate optimal trade size as percentage of average volume
            liquidity_analysis["optimal_size"] = avg_volume * 0.1  # 10% of average volume

        # Use any available liquidity indicators
        if "liquidity_indicator" in indicator_results:
            liq_data = indicator_results["liquidity_indicator"]
            liquidity_analysis["liquidity_score"] = liq_data.get("liquidity_score", liquidity_analysis["liquidity_score"])

        return liquidity_analysis

    def _analyze_execution_quality_fallback(self, indicator_results: Dict[str, Any], ohlcv_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Fallback execution quality analysis using OHLCV data only."""
        execution_analysis = {
            "quality_score": 0.5,  # Default moderate quality
            "fill_ratio": 0.9,  # Conservative estimate
            "price_improvement": 0.0,
            "execution_shortfall": 0.001,  # Small default shortfall
            "twap_performance": "NEUTRAL",
            "optimal_timing": False,
            "execution_recommendation": "LIMIT",  # Conservative default
            "data_quality": "FALLBACK"
        }

        # Analyze recent price volatility for timing
        if ohlcv_data is not None and not ohlcv_data.empty:
            price_changes = ohlcv_data['close'].pct_change().tail(10)
            volatility = price_changes.std()

            # High volatility suggests poor execution conditions
            if volatility > 0.02:  # 2% volatility
                execution_analysis["quality_score"] = 0.3
                execution_analysis["execution_recommendation"] = "TWAP"
                execution_analysis["fill_ratio"] = 0.8
            elif volatility < 0.005:  # 0.5% volatility
                execution_analysis["quality_score"] = 0.7
                execution_analysis["optimal_timing"] = True
                execution_analysis["execution_recommendation"] = "MARKET"
                execution_analysis["fill_ratio"] = 0.95

        # Use any available execution quality indicators
        if "execution_quality_indicator" in indicator_results:
            qual_data = indicator_results["execution_quality_indicator"]
            execution_analysis["quality_score"] = qual_data.get("quality_score", execution_analysis["quality_score"])
            execution_analysis["optimal_timing"] = qual_data.get("optimal_timing", execution_analysis["optimal_timing"])

        return execution_analysis

    def _analyze_market_impact_fallback(self, indicator_results: Dict[str, Any], ohlcv_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Fallback market impact analysis using OHLCV data only."""
        impact_analysis = {
            "expected_impact": 0.002,  # Conservative 20bp default
            "slippage_estimate": 0.001,  # 10bp default slippage
            "impact_risk": "MEDIUM",
            "microstructure_signal": "NEUTRAL",
            "order_flow_direction": "BALANCED",
            "impact_tolerance": True,
            "data_quality": "FALLBACK"
        }

        # Estimate impact from volume and volatility
        if ohlcv_data is not None and not ohlcv_data.empty:
            recent_data = ohlcv_data.tail(5)
            avg_volume = recent_data['volume'].mean()
            price_volatility = recent_data['close'].pct_change().std()

            # Higher volatility and lower volume = higher impact
            vol_factor = min(price_volatility / 0.01, 2.0)  # Cap at 2x
            volume_factor = max(0.5, 1.0 / (avg_volume / ohlcv_data['volume'].mean()))

            base_impact = 0.001  # 10bp base
            impact_analysis["expected_impact"] = min(base_impact * vol_factor * volume_factor, 0.01)  # Cap at 100bp
            impact_analysis["slippage_estimate"] = impact_analysis["expected_impact"] * 0.5

            # Adjust risk assessment
            if impact_analysis["expected_impact"] > self.max_market_impact:
                impact_analysis["impact_tolerance"] = False
                impact_analysis["impact_risk"] = "HIGH"
            elif impact_analysis["expected_impact"] < self.max_market_impact * 0.5:
                impact_analysis["impact_risk"] = "LOW"

        # Use any available impact indicators
        if "market_impact_indicator" in indicator_results:
            impact_data = indicator_results["market_impact_indicator"]
            impact_analysis["expected_impact"] = impact_data.get("expected_impact", impact_analysis["expected_impact"])

        if "slippage_indicator" in indicator_results:
            slip_data = indicator_results["slippage_indicator"]
            impact_analysis["slippage_estimate"] = slip_data.get("expected_slippage", impact_analysis["slippage_estimate"])

        return impact_analysis

    def _determine_execution_method(self, execution_analysis: Dict[str, Any]) -> str:
        """Determine optimal execution method."""
        quality_score = execution_analysis.get("quality_score", 0.0)
        fill_ratio = execution_analysis.get("fill_ratio", 1.0)
        optimal_timing = execution_analysis.get("optimal_timing", False)

        # High quality execution conditions
        if quality_score > 0.8 and fill_ratio > 0.95 and optimal_timing:
            return "MARKET"

        # Moderate quality - use limit orders
        if quality_score > 0.5 and fill_ratio > 0.8:
            return "LIMIT"

        # Poor conditions - use TWAP/VWAP
        if quality_score < 0.5 or fill_ratio < 0.7:
            return "TWAP"

        return "LIMIT"

    def _generate_execution_decision(self, liquidity_analysis: Dict[str, Any], execution_analysis: Dict[str, Any], impact_analysis: Dict[str, Any]) -> str:
        """Generate execution decision."""
        liquidity_score = liquidity_analysis.get("liquidity_score", 0.0)
        quality_score = execution_analysis.get("quality_score", 0.0)
        impact_tolerance = impact_analysis.get("impact_tolerance", True)
        optimal_timing = execution_analysis.get("optimal_timing", False)

        # Excellent execution conditions
        if liquidity_score > 0.8 and quality_score > 0.8 and impact_tolerance and optimal_timing:
            return "EXECUTE_AGGRESSIVE"

        # Good execution conditions
        if liquidity_score > 0.6 and quality_score > 0.6 and impact_tolerance:
            return "EXECUTE_MODERATE"

        # Poor liquidity or high impact
        if liquidity_score < 0.4 or not impact_tolerance:
            return "EXECUTE_PASSIVE"

        # Moderate conditions
        if liquidity_score > 0.5 and quality_score > 0.5:
            return "EXECUTE_MODERATE"

        return "WAIT"

    def _calculate_execution_confidence(self, liquidity_analysis: Dict[str, Any], execution_analysis: Dict[str, Any], impact_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in execution analysis."""
        factors = []

        # Liquidity confidence
        liquidity_score = liquidity_analysis.get("liquidity_score", 0.0)
        factors.append(liquidity_score * 0.4)

        # Execution quality confidence
        quality_score = execution_analysis.get("quality_score", 0.0)
        factors.append(quality_score * 0.3)

        # Impact confidence (inverse)
        expected_impact = impact_analysis.get("expected_impact", 0.0)
        impact_confidence = max(0, 1 - (expected_impact / self.max_market_impact))
        factors.append(impact_confidence * 0.3)

        return min(sum(factors), 1.0)

    def _assess_execution_risk(self, liquidity_analysis: Dict[str, Any], impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess execution-related risks."""
        liquidity_risk = liquidity_analysis.get("liquidity_risk", "MEDIUM")
        impact_risk = impact_analysis.get("impact_risk", "MEDIUM")

        # Overall risk is highest of individual risks
        if liquidity_risk == "HIGH" or impact_risk == "HIGH":
            overall_risk = "HIGH"
        elif liquidity_risk == "LOW" and impact_risk == "LOW":
            overall_risk = "LOW"
        else:
            overall_risk = "MEDIUM"

        return {
            "execution_risk": overall_risk,
            "liquidity_risk": liquidity_risk,
            "impact_risk": impact_risk,
            "slippage_risk": "MEDIUM"
        }

    def _generate_execution_reasoning(self, decision: str, liquidity_analysis: Dict[str, Any], execution_analysis: Dict[str, Any], impact_analysis: Dict[str, Any]) -> str:
        """Generate reasoning for execution analysis."""
        parts = [f"Execution analysis decision: {decision}."]

        liquidity_score = liquidity_analysis.get("liquidity_score", 0.0)
        parts.append(f"Liquidity score: {liquidity_score:.3f}.")

        quality_score = execution_analysis.get("quality_score", 0.0)
        parts.append(f"Execution quality: {quality_score:.3f}.")

        expected_impact = impact_analysis.get("expected_impact", 0.0)
        parts.append(f"Expected market impact: {expected_impact:.4f}.")

        execution_method = execution_analysis.get("execution_recommendation", "MARKET")
        parts.append(f"Recommended execution method: {execution_method}.")

        return " ".join(parts)

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually used in analysis."""
        return [ind for ind in self.assigned_indicators if ind in indicator_results]

    def get_required_data_types(self) -> List[str]:
        """Define required data types."""
        return ["OHLCV", "TICK", "ORDER_BOOK"]

    def get_minimum_data_points(self) -> int:
        """Define minimum data points needed."""
        return 20  # Need recent data for execution analysis
