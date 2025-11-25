"""
Pair Specialist Agent for the AUJ Platform.

This agent specializes in correlation analysis and pairs trading.
It focuses on 20 correlation and statistical arbitrage indicators.
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

logger = get_logger(__name__)


class PairSpecialist(BaseAgent):
    """
    Pair Specialist Agent - Correlation analysis and pairs trading.

    Specializes in:
    - Correlation matrix analysis
    - Cointegration testing
    - Statistical arbitrage opportunities
    - Relative strength analysis
    """

    def __init__(self, config_manager, config: Optional[Dict[str, Any]] = None):
        """Initialize the Pair Specialist Agent."""
        # Load configuration first
        self.config = config or self._load_agent_config()
        self.config_manager = config_manager

        assigned_indicators = [
            # Correlation & Relationship Analysis
            "correlation_analysis_indicator",
            "correlation_coefficient_indicator",
            "correlation_matrix_indicator",
            "cointegration_indicator",
            "beta_coefficient_indicator",

            # Cross-Pair Analysis
            "linear_regression_indicator",
            "linear_regression_channels_indicator",
            "rsquared_indicator",
            "intermarket_correlation_indicator",

            # Relative Performance
            "relative_strength_mansfield_indicator",
            "price_oscillator_indicator"
        ]

        super().__init__(
            name="PairSpecialist",
            specialization="Correlation analysis and pairs trading",
            assigned_indicators=assigned_indicators,
            config_manager=config_manager,
            config=self.config
        )

        # Configuration parameters
        self.correlation_threshold = self.config['correlation_analysis']['correlation_threshold']
        self.cointegration_threshold = self.config['cointegration_testing']['cointegration_threshold']
        self.pairs_trading_enabled = self.config['pairs_trading']['enabled']

        logger.info(f"PairSpecialist initialized with {len(assigned_indicators)} indicators")

    def _load_agent_config(self) -> Dict[str, Any]:
        """Load agent configuration from YAML file with fallback defaults."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'agents', 'pair_specialist.yaml'
            )

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded PairSpecialist configuration from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file not found at {config_path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading PairSpecialist configuration: {e}")

        # Fallback defaults
        return {
            'correlation_analysis': {
                'correlation_threshold': 0.7,
                'min_correlation_period': 30,
                'rolling_correlation_window': 60
            },
            'cointegration_testing': {
                'cointegration_threshold': 0.05,
                'lookback_period': 252,
                'confidence_level': 0.95
            },
            'pairs_trading': {
                'enabled': True,
                'z_score_entry': 2.0,
                'z_score_exit': 0.5,
                'max_holding_period': 30
            },
            'risk_management': {
                'max_position_correlation': 0.8,
                'diversification_threshold': 0.6
            }
        }

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform correlation and pairs analysis."""
        try:
            # Correlation Analysis
            correlation_analysis = self._analyze_correlations(indicator_results)

            # Statistical Arbitrage
            arbitrage_analysis = self._analyze_statistical_arbitrage(indicator_results)

            # Generate Decision
            decision = self._generate_pairs_decision(correlation_analysis, arbitrage_analysis)

            # Calculate Confidence
            confidence = self._calculate_pairs_confidence(correlation_analysis, arbitrage_analysis)

            # Generate Reasoning
            reasoning = self._generate_pairs_reasoning(decision, correlation_analysis, arbitrage_analysis)

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "correlation_analysis": correlation_analysis,
                    "arbitrage_analysis": arbitrage_analysis
                },
                risk_assessment=self._assess_pairs_risk(correlation_analysis),
                supporting_data={
                    "correlation_strength": correlation_analysis.get("avg_correlation", 0.0),
                    "arbitrage_opportunity": arbitrage_analysis.get("opportunity_score", 0.0)
                }
            )

        except Exception as e:
            logger.error(f"PairSpecialist analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Pairs analysis failed: {str(e)}")

    def _analyze_correlations(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlation indicators."""
        correlation_analysis = {
            "correlation_matrix": {},
            "avg_correlation": 0.0,
            "correlation_trend": "STABLE",
            "relative_strength": "NEUTRAL",
            "correlation_quality": "MEDIUM"
        }

        # Process correlation indicators
        if "correlation_analysis_indicator" in indicator_results:
            corr_data = indicator_results["correlation_analysis_indicator"]
            correlation_analysis["correlation_matrix"] = corr_data.get("matrix", {})
            correlation_analysis["avg_correlation"] = corr_data.get("average", 0.0)
            correlation_analysis["correlation_trend"] = corr_data.get("trend", "STABLE")

        # Relative strength analysis
        if "relative_vigor_index_indicator" in indicator_results:
            rvi_data = indicator_results["relative_vigor_index_indicator"]
            rvi_value = rvi_data.get("rvi", 0.0)
            if rvi_value > 0.5:
                correlation_analysis["relative_strength"] = "STRONG"
            elif rvi_value < -0.5:
                correlation_analysis["relative_strength"] = "WEAK"

        return correlation_analysis

    def _analyze_statistical_arbitrage(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze statistical arbitrage opportunities."""
        arbitrage_analysis = {
            "cointegration_score": 0.0,
            "mean_reversion_signal": "NEUTRAL",
            "arbitrage_opportunity": False,
            "opportunity_score": 0.0,
            "expected_return": 0.0
        }

        # Cointegration analysis
        if "cointegration_indicator" in indicator_results:
            coint_data = indicator_results["cointegration_indicator"]
            arbitrage_analysis["cointegration_score"] = coint_data.get("p_value", 1.0)
            if coint_data.get("p_value", 1.0) < self.cointegration_threshold:
                arbitrage_analysis["arbitrage_opportunity"] = True

        # Mean reversion signals
        if "detrended_price_oscillator_indicator" in indicator_results:
            dpo_data = indicator_results["detrended_price_oscillator_indicator"]
            dpo_value = dpo_data.get("dpo", 0.0)
            if abs(dpo_value) > 2.0:  # Strong mean reversion signal
                arbitrage_analysis["mean_reversion_signal"] = "STRONG"
                arbitrage_analysis["opportunity_score"] = min(abs(dpo_value) / 5.0, 1.0)

        return arbitrage_analysis

    def _generate_pairs_decision(self, correlation_analysis: Dict[str, Any], arbitrage_analysis: Dict[str, Any]) -> str:
        """Generate pairs trading decision."""
        if not self.pairs_trading_enabled:
            return "HOLD"

        # Check for arbitrage opportunities
        if arbitrage_analysis["arbitrage_opportunity"] and arbitrage_analysis["opportunity_score"] > 0.6:
            mean_reversion = arbitrage_analysis["mean_reversion_signal"]
            if mean_reversion == "STRONG":
                return "BUY"  # Assuming mean reversion buy signal

        # Correlation-based signals
        avg_corr = correlation_analysis["avg_correlation"]
        if abs(avg_corr) > self.correlation_threshold:
            return "HOLD"  # Let correlation guide but don't force direction

        return "NO_SIGNAL"

    def _calculate_pairs_confidence(self, correlation_analysis: Dict[str, Any], arbitrage_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in pairs analysis."""
        factors = []

        # Correlation confidence
        avg_corr = abs(correlation_analysis.get("avg_correlation", 0.0))
        factors.append(min(avg_corr, 1.0) * 0.4)

        # Arbitrage opportunity confidence
        opportunity_score = arbitrage_analysis.get("opportunity_score", 0.0)
        factors.append(opportunity_score * 0.6)

        return sum(factors)

    def _assess_pairs_risk(self, correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk in pairs trading."""
        return {
            "correlation_risk": "MEDIUM",
            "breakdown_risk": 0.3,
            "recommended_hedge_ratio": 1.0
        }

    def _generate_pairs_reasoning(self, decision: str, correlation_analysis: Dict[str, Any], arbitrage_analysis: Dict[str, Any]) -> str:
        """Generate reasoning for pairs analysis."""
        parts = [f"Pairs analysis decision: {decision}."]

        avg_corr = correlation_analysis.get("avg_correlation", 0.0)
        parts.append(f"Average correlation: {avg_corr:.3f}.")

        if arbitrage_analysis["arbitrage_opportunity"]:
            parts.append("Statistical arbitrage opportunity detected.")

        return " ".join(parts)

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually used in analysis."""
        return [ind for ind in self.assigned_indicators if ind in indicator_results]

    def get_required_data_types(self) -> List[str]:
        """Define required data types."""
        return ["OHLCV"]

    def get_minimum_data_points(self) -> int:
        """Define minimum data points needed."""
        return 60  # Need more data for correlation analysis
