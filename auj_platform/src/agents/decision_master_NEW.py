"""
Decision Master Agent for the AUJ Platform.

This agent serves as the final decision maker, integrating all agent inputs.
It focuses on 18 decision synthesis and portfolio-level indicators with AI ensemble modeling.
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
from ..ai_ensemble import EnsembleManager, VotingMethod

logger = get_logger(__name__)


class DecisionMaster(BaseAgent):
    """
    Decision Master Agent - Final decision synthesis and integration.

    Specializes in:
    - Multi-agent decision synthesis
    - Portfolio-level risk management
    - Confidence aggregation
    - Final trade authorization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_manager=None):
        from ..core.unified_config import UnifiedConfigManager
        self.config_manager = config_manager or UnifiedConfigManager()
        """Initialize the Decision Master Agent with AI Ensemble capabilities."""
        # Load configuration first
        self.config = self._load_agent_config()

        assigned_indicators = [
            # Core AI Enhanced Indicators
            "advanced_ml_engine_indicator",
            "ai_commodity_channel_index_indicator",
            "attractor_point_indicator",
            "chaos_geometry_predictor_indicator",
            "crystallographic_lattice_detector_indicator",
            "genetic_algorithm_optimizer_indicator",
            "lstm_price_predictor_indicator",
            "ml_signal_generator_indicator",
            "neural_harmonic_resonance_indicator",
            "neural_network_predictor_indicator",
            "thermodynamic_entropy_engine_indicator",

            # Market Intelligence
            "social_media_post_indicator",
            "news_article_indicator",
            "biorhythm_market_synth_indicator",
            "custom_ai_composite_indicator",
            "self_similarity_detector_indicator",

            # Decision Support Indicators
            "average_true_range_indicator",
            "directional_movement_system_indicator",
            "donchian_channels_indicator",
            "force_index_indicator",
            "grid_line_indicator",
            "timeframe_config_indicator"
        ]

        super().__init__(
            name="DecisionMaster",
            specialization="AI-enhanced multi-agent decision synthesis and final authorization",
            assigned_indicators=assigned_indicators,
            config=config
        )

        # Initialize AI Ensemble Manager
        ensemble_config = self.config_manager.get_dict('ensemble_config', {}) if config else {}
        self.ensemble_manager = EnsembleManager(ensemble_config)

        # Decision thresholds from configuration
        self.min_consensus_threshold = self.config['decision_thresholds']['min_consensus_threshold']
        self.min_confidence_threshold = self.config['decision_thresholds']['min_confidence_threshold']
        self.max_portfolio_risk = self.config['decision_thresholds']['max_portfolio_risk']
        self.risk_free_rate = self.config['decision_thresholds']['risk_free_rate']

        # AI-specific parameters from configuration
        self.ai_confidence_threshold = self.config['ai_ensemble']['confidence_threshold']
        self.ensemble_training_frequency = self.config['ai_ensemble']['training_frequency']
        self.decision_count = 0

        logger.info(f"DecisionMaster initialized with {len(assigned_indicators)} indicators and AI ensemble")

    def _load_agent_config(self) -> Dict[str, Any]:
        """Load agent configuration from YAML file with fallback defaults."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'agents', 'decision_master.yaml'
            )

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded DecisionMaster configuration from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file not found at {config_path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading DecisionMaster configuration: {e}")

        # Fallback defaults
        return {
            'decision_thresholds': {
                'min_consensus_threshold': 0.6,
                'min_confidence_threshold': 0.5,
                'max_portfolio_risk': 0.02,
                'risk_free_rate': 0.02
            },
            'ai_ensemble': {
                'confidence_threshold': 0.7,
                'training_frequency': 50,
                'voting_method': 'weighted',
                'model_selection_threshold': 0.8
            },
            'portfolio_management': {
                'max_position_size': 0.1,
                'max_correlation': 0.7,
                'rebalancing_threshold': 0.05
            },
            'risk_management': {
                'stop_loss_threshold': 0.02,
                'take_profit_threshold': 0.04,
                'max_drawdown_threshold': 0.15
            }
        }

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform final decision synthesis with AI-enhanced analysis."""
        try:
            # Check data availability for AI processing
            data_availability = self._assess_data_availability(market_data)

            # AI-Enhanced Agent consensus analysis
            consensus_analysis = self._analyze_agent_consensus_ai(indicator_results, data_availability)

            # AI-Enhanced Portfolio risk analysis
            portfolio_analysis = self._analyze_portfolio_risk_ai(indicator_results, market_data)

            # AI-Enhanced Decision synthesis
            synthesis_analysis = self._synthesize_final_decision_ai(indicator_results, consensus_analysis, portfolio_analysis, market_data)

            # Generate Final Decision with AI confidence
            decision = self._generate_master_decision(consensus_analysis, portfolio_analysis, synthesis_analysis)

            # Calculate AI-Enhanced Confidence
            confidence = self._calculate_master_confidence_ai(consensus_analysis, portfolio_analysis, synthesis_analysis, data_availability)

            # Generate AI-Enhanced Reasoning
            reasoning = self._generate_master_reasoning_ai(decision, consensus_analysis, portfolio_analysis, synthesis_analysis, data_availability)

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "consensus_analysis": consensus_analysis,
                    "portfolio_analysis": portfolio_analysis,
                    "synthesis_analysis": synthesis_analysis,
                    "data_availability": data_availability
                },
                risk_assessment=self._assess_master_risk_ai(portfolio_analysis, synthesis_analysis, data_availability),
                supporting_data={
                    "agent_consensus": consensus_analysis.get("consensus_score", 0.0),
                    "portfolio_risk": portfolio_analysis.get("portfolio_risk", 0.0),
                    "final_authorization": synthesis_analysis.get("authorized", False),
                    "position_size": portfolio_analysis.get("recommended_size", 0.0),
                    "ai_confidence": synthesis_analysis.get("ai_confidence", 0.0),
                    "data_quality_score": data_availability.get("quality_score", 0.5)
                }
            )

        except Exception as e:
            logger.error(f"DecisionMaster analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Master decision analysis failed: {str(e)}")

    def _analyze_agent_consensus(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consensus among agents."""
        consensus_analysis = {
            "consensus_score": 0.0,
            "agent_agreement": "LOW",
            "confidence_aggregation": 0.0,
            "voting_result": "NO_CONSENSUS",
            "dissenting_agents": [],
            "consensus_direction": "NEUTRAL"
        }

        # Consensus indicator
        if "consensus_indicator" in indicator_results:
            consensus_data = indicator_results["consensus_indicator"]
            consensus_analysis["consensus_score"] = consensus_data.get("consensus_score", 0.0)
            consensus_analysis["voting_result"] = consensus_data.get("voting_result", "NO_CONSENSUS")
            consensus_analysis["dissenting_agents"] = consensus_data.get("dissenting_agents", [])

        # Confidence aggregation
        if "confidence_aggregation_indicator" in indicator_results:
            conf_data = indicator_results["confidence_aggregation_indicator"]
            consensus_analysis["confidence_aggregation"] = conf_data.get("aggregated_confidence", 0.0)

        # Ensemble voting
        if "ensemble_voting_indicator" in indicator_results:
            voting_data = indicator_results["ensemble_voting_indicator"]
            consensus_analysis["consensus_direction"] = voting_data.get("final_direction", "NEUTRAL")

        # Determine agreement level
        consensus_score = consensus_analysis["consensus_score"]
        if consensus_score > 0.8:
            consensus_analysis["agent_agreement"] = "HIGH"
        elif consensus_score > 0.6:
            consensus_analysis["agent_agreement"] = "MEDIUM"

        return consensus_analysis

    def _analyze_portfolio_risk(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio-level risk metrics."""
        portfolio_analysis = {
            "portfolio_risk": 0.0,
            "position_size": 0.0,
            "recommended_size": 0.0,
            "risk_budget": 0.0,
            "diversification_score": 0.0,
            "concentration_risk": "LOW",
            "var_estimate": 0.0,
            "max_drawdown_risk": 0.0
        }

        # Portfolio risk indicator
        if "portfolio_risk_indicator" in indicator_results:
            risk_data = indicator_results["portfolio_risk_indicator"]
            portfolio_analysis["portfolio_risk"] = risk_data.get("portfolio_risk", 0.0)
            portfolio_analysis["risk_budget"] = risk_data.get("available_risk_budget", 0.0)

        # Position sizing
        if "position_sizing_indicator" in indicator_results:
            size_data = indicator_results["position_sizing_indicator"]
            portfolio_analysis["recommended_size"] = size_data.get("optimal_size", 0.0)

        # Kelly criterion
        if "kelly_criterion_indicator" in indicator_results:
            kelly_data = indicator_results["kelly_criterion_indicator"]
            kelly_size = kelly_data.get("kelly_fraction", 0.0)
            # Use conservative fraction of Kelly
            portfolio_analysis["position_size"] = min(kelly_size * 0.25, portfolio_analysis["recommended_size"])

        # Diversification
        if "portfolio_diversification_indicator" in indicator_results:
            div_data = indicator_results["portfolio_diversification_indicator"]
            portfolio_analysis["diversification_score"] = div_data.get("diversification_score", 0.0)

        # Concentration risk
        if "concentration_risk_indicator" in indicator_results:
            conc_data = indicator_results["concentration_risk_indicator"]
            conc_level = conc_data.get("concentration_level", "LOW")
            portfolio_analysis["concentration_risk"] = conc_level

        # VaR estimation
        if "var_indicator" in indicator_results:
            var_data = indicator_results["var_indicator"]
            portfolio_analysis["var_estimate"] = var_data.get("var_95", 0.0)

        return portfolio_analysis

    def _synthesize_final_decision(self, indicator_results: Dict[str, Any], consensus_analysis: Dict[str, Any], portfolio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final decision from all inputs."""
        synthesis_analysis = {
            "authorized": False,
            "authorization_score": 0.0,
            "risk_adjusted_confidence": 0.0,
            "regime_stability": "STABLE",
            "model_confidence": 0.0,
            "final_recommendation": "NO_ACTION",
            "risk_override": False
        }

        # Decision synthesis
        if "decision_synthesis_indicator" in indicator_results:
            synth_data = indicator_results["decision_synthesis_indicator"]
            synthesis_analysis["authorization_score"] = synth_data.get("authorization_score", 0.0)
            synthesis_analysis["final_recommendation"] = synth_data.get("final_recommendation", "NO_ACTION")

        # Regime stability
        if "regime_stability_indicator" in indicator_results:
            regime_data = indicator_results["regime_stability_indicator"]
            synthesis_analysis["regime_stability"] = regime_data.get("stability_level", "STABLE")

        # Model confidence
        if "model_confidence_indicator" in indicator_results:
            model_data = indicator_results["model_confidence_indicator"]
            synthesis_analysis["model_confidence"] = model_data.get("confidence_level", 0.0)

        # Calculate risk-adjusted confidence
        consensus_conf = consensus_analysis.get("confidence_aggregation", 0.0)
        portfolio_risk = portfolio_analysis.get("portfolio_risk", 0.0)
        risk_penalty = min(portfolio_risk / self.max_portfolio_risk, 1.0)
        synthesis_analysis["risk_adjusted_confidence"] = consensus_conf * (1 - risk_penalty * 0.5)

        # Authorization logic
        min_consensus = consensus_analysis.get("consensus_score", 0.0) >= self.min_consensus_threshold
        min_confidence = synthesis_analysis["risk_adjusted_confidence"] >= self.min_confidence_threshold
        risk_acceptable = portfolio_analysis.get("portfolio_risk", 0.0) <= self.max_portfolio_risk
        regime_stable = synthesis_analysis["regime_stability"] in ["STABLE", "TRANSITIONING"]

        synthesis_analysis["authorized"] = min_consensus and min_confidence and risk_acceptable and regime_stable

        # Risk override check
        if not risk_acceptable:
            synthesis_analysis["risk_override"] = True
            synthesis_analysis["authorized"] = False

        return synthesis_analysis

    def _generate_master_decision(self, consensus_analysis: Dict[str, Any], portfolio_analysis: Dict[str, Any], synthesis_analysis: Dict[str, Any]) -> str:
        """Generate the final master decision."""
        authorized = synthesis_analysis.get("authorized", False)
        final_recommendation = synthesis_analysis.get("final_recommendation", "NO_ACTION")
        consensus_direction = consensus_analysis.get("consensus_direction", "NEUTRAL")

        if not authorized:
            return "NO_SIGNAL"

        # Use synthesis recommendation if available
        if final_recommendation in ["BUY", "SELL"]:
            return final_recommendation

        # Fall back to consensus direction
        if consensus_direction in ["BULLISH", "BUY"]:
            return "BUY"
        elif consensus_direction in ["BEARISH", "SELL"]:
            return "SELL"

        return "NO_SIGNAL"

    def _calculate_master_confidence(self, consensus_analysis: Dict[str, Any], portfolio_analysis: Dict[str, Any], synthesis_analysis: Dict[str, Any]) -> float:
        """Calculate final master confidence."""
        if not synthesis_analysis.get("authorized", False):
            return 0.0

        factors = []

        # Risk-adjusted confidence
        risk_adj_conf = synthesis_analysis.get("risk_adjusted_confidence", 0.0)
        factors.append(risk_adj_conf * 0.4)

        # Consensus strength
        consensus_score = consensus_analysis.get("consensus_score", 0.0)
        factors.append(consensus_score * 0.3)

        # Model confidence
        model_conf = synthesis_analysis.get("model_confidence", 0.0)
        factors.append(model_conf * 0.2)

        # Portfolio health
        div_score = portfolio_analysis.get("diversification_score", 0.0)
        factors.append(div_score * 0.1)

        return min(sum(factors), 1.0)

    def _assess_master_risk(self, portfolio_analysis: Dict[str, Any], synthesis_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess master-level risks."""
        portfolio_risk = portfolio_analysis.get("portfolio_risk", 0.0)
        concentration_risk = portfolio_analysis.get("concentration_risk", "LOW")
        regime_stability = synthesis_analysis.get("regime_stability", "STABLE")

        # Overall risk assessment
        if portfolio_risk > self.max_portfolio_risk or concentration_risk == "HIGH":
            overall_risk = "HIGH"
        elif regime_stability == "UNSTABLE":
            overall_risk = "HIGH"
        elif portfolio_risk > self.max_portfolio_risk * 0.7:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        return {
            "master_risk": overall_risk,
            "portfolio_risk": f"{portfolio_risk:.4f}",
            "concentration_risk": concentration_risk,
            "regime_risk": regime_stability,
            "model_risk": "MEDIUM"
        }

    def _generate_master_reasoning(self, decision: str, consensus_analysis: Dict[str, Any], portfolio_analysis: Dict[str, Any], synthesis_analysis: Dict[str, Any]) -> str:
        """Generate reasoning for master decision."""
        parts = [f"Master decision: {decision}."]

        consensus_score = consensus_analysis.get("consensus_score", 0.0)
        parts.append(f"Agent consensus: {consensus_score:.3f}.")

        portfolio_risk = portfolio_analysis.get("portfolio_risk", 0.0)
        parts.append(f"Portfolio risk: {portfolio_risk:.4f}.")

        authorized = synthesis_analysis.get("authorized", False)
        parts.append(f"Trade authorized: {authorized}.")

        if synthesis_analysis.get("risk_override", False):
            parts.append("Risk override activated.")

        regime = synthesis_analysis.get("regime_stability", "STABLE")
        parts.append(f"Market regime: {regime}.")

        return " ".join(parts)

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually used in analysis."""
        return [ind for ind in self.assigned_indicators if ind in indicator_results]

    def get_required_data_types(self) -> List[str]:
        """Define required data types."""
        return ["AGENT_DECISIONS"]  # Special data type for agent decisions

    def get_minimum_data_points(self) -> int:
        """Define minimum data points needed."""
        return 1  # Needs current agent decisions

    def _assess_data_availability(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess quality and availability of different data sources for AI processing."""
        data_availability = {
            "ohlcv_available": False,
            "tick_available": False,
            "order_book_available": False,
            "volume_available": False,
            "quality_score": 0.0,
            "data_sources": [],
            "data_completeness": 0.0
        }

        total_score = 0.0
        max_score = 0.0

        # Check OHLCV data (base requirement)
        if 'OHLCV' in market_data and not market_data['OHLCV'].empty:
            data_availability["ohlcv_available"] = True
            data_availability["data_sources"].append("OHLCV")
            ohlcv_data = market_data['OHLCV']

            # Quality check
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            completeness = sum(col in ohlcv_data.columns for col in required_cols) / len(required_cols)
            total_score += completeness * 0.4
            data_availability["volume_available"] = 'volume' in ohlcv_data.columns
        max_score += 0.4

        # Check tick data (premium feature)
        if 'TICK' in market_data and not market_data['TICK'].empty:
            data_availability["tick_available"] = True
            data_availability["data_sources"].append("TICK")
            tick_data = market_data['TICK']

            # Quality check for tick data
            tick_cols = ['price', 'volume', 'timestamp']
            completeness = sum(col in tick_data.columns for col in tick_cols) / len(tick_cols)
            total_score += completeness * 0.4
        max_score += 0.4

        # Check order book data (premium feature)
        if 'ORDER_BOOK' in market_data and not market_data['ORDER_BOOK'].empty:
            data_availability["order_book_available"] = True
            data_availability["data_sources"].append("ORDER_BOOK")
            total_score += 0.2
        max_score += 0.2

        # Calculate overall quality score
        data_availability["quality_score"] = total_score / max_score if max_score > 0 else 0.0
        data_availability["data_completeness"] = len(data_availability["data_sources"]) / 3  # Max 3 sources

        return data_availability

    def _analyze_agent_consensus_ai(self, indicator_results: Dict[str, Any], data_availability: Dict[str, Any]) -> Dict[str, Any]:
        """AI-enhanced agent consensus analysis with data quality weighting."""
        consensus_analysis = {
            "consensus_score": 0.0,
            "agent_agreement": "LOW",
            "confidence_aggregation": 0.0,
            "voting_result": "NO_CONSENSUS",
            "dissenting_agents": [],
            "consensus_direction": "NEUTRAL",
            "ai_weighted_consensus": 0.0,
            "data_quality_adjustment": 0.0
        }

        # Apply data quality weighting to consensus
        data_quality_multiplier = 0.5 + (data_availability.get("quality_score", 0.0) * 0.5)

        # Enhanced consensus calculation with AI
        base_consensus = self._calculate_base_consensus(indicator_results)
        consensus_analysis.update(base_consensus)

        # AI enhancement based on data availability
        if data_availability.get("tick_available", False):
            # High-frequency data available - increase confidence
            consensus_analysis["ai_weighted_consensus"] = min(1.0, base_consensus.get("consensus_score", 0.0) * 1.2)
        elif data_availability.get("ohlcv_available", False):
            # Standard data - normal processing
            consensus_analysis["ai_weighted_consensus"] = base_consensus.get("consensus_score", 0.0)
        else:
            # Limited data - reduce confidence
            consensus_analysis["ai_weighted_consensus"] = base_consensus.get("consensus_score", 0.0) * 0.7

        consensus_analysis["data_quality_adjustment"] = data_quality_multiplier

        return consensus_analysis

    def _calculate_base_consensus(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate base consensus from traditional indicators."""
        # Legacy consensus analysis
        if "consensus_indicator" in indicator_results:
            consensus_data = indicator_results["consensus_indicator"]
            return {
                "consensus_score": consensus_data.get("consensus_score", 0.0),
                "voting_result": consensus_data.get("voting_result", "NO_CONSENSUS"),
                "dissenting_agents": consensus_data.get("dissenting_agents", [])
            }

        return {
            "consensus_score": 0.0,
            "voting_result": "NO_CONSENSUS",
            "dissenting_agents": []
        }

    def _analyze_portfolio_risk_ai(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """AI-enhanced portfolio risk analysis using available data."""
        portfolio_analysis = {
            "portfolio_risk": 0.0,
            "position_size": 0.0,
            "recommended_size": 0.0,
            "risk_budget": 0.0,
            "diversification_score": 0.0,
            "concentration_risk": "LOW",
            "var_estimate": 0.0,
            "max_drawdown_risk": 0.0,
            "ai_risk_adjustment": 0.0,
            "dynamic_risk_scaling": 1.0
        }

        # Base portfolio analysis
        base_analysis = self._calculate_base_portfolio_risk(indicator_results)
        portfolio_analysis.update(base_analysis)

        # AI enhancement with market data
        ai_risk_factor = self._calculate_ai_risk_factor(market_data)
        portfolio_analysis["ai_risk_adjustment"] = ai_risk_factor

        # Dynamic risk scaling based on market conditions
        if market_data.get('OHLCV') is not None:
            volatility_factor = self._calculate_market_volatility_factor(market_data['OHLCV'])
            portfolio_analysis["dynamic_risk_scaling"] = max(0.5, min(2.0, 1.0 / volatility_factor))

        # Adjust recommended size based on AI analysis
        base_size = portfolio_analysis.get("recommended_size", 0.0)
        portfolio_analysis["recommended_size"] = base_size * portfolio_analysis["dynamic_risk_scaling"]

        return portfolio_analysis

    def _calculate_ai_risk_factor(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate AI-based risk factor from market data patterns."""
        risk_factors = []

        # Volume analysis
        if 'OHLCV' in market_data and not market_data['OHLCV'].empty:
            ohlcv = market_data['OHLCV']
            if len(ohlcv) > 10:
                volume_volatility = ohlcv['volume'].std() / ohlcv['volume'].mean()
                risk_factors.append(min(volume_volatility, 2.0))

        # Price volatility
        if 'OHLCV' in market_data and not market_data['OHLCV'].empty:
            ohlcv = market_data['OHLCV']
            if len(ohlcv) > 5:
                price_volatility = ohlcv['close'].pct_change().std()
                risk_factors.append(min(price_volatility * 100, 2.0))

        # Tick data volatility (if available)
        if 'TICK' in market_data and not market_data['TICK'].empty:
            tick_data = market_data['TICK']
            if len(tick_data) > 50:
                tick_volatility = tick_data['price'].pct_change().std()
                risk_factors.append(min(tick_volatility * 50, 1.5))

        return sum(risk_factors) / len(risk_factors) if risk_factors else 1.0

    def _calculate_market_volatility_factor(self, ohlcv_data: pd.DataFrame) -> float:
        """Calculate market volatility factor for risk scaling."""
        if ohlcv_data.empty or len(ohlcv_data) < 5:
            return 1.0

        # Calculate ATR-based volatility
        high_low = ohlcv_data['high'] - ohlcv_data['low']
        high_close = abs(ohlcv_data['high'] - ohlcv_data['close'].shift(1))
        low_close = abs(ohlcv_data['low'] - ohlcv_data['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1] if len(ohlcv_data) >= 14 else true_range.mean()

        # Normalize by current price
        current_price = ohlcv_data['close'].iloc[-1]
        volatility_factor = (atr / current_price) * 100  # Convert to percentage

        return max(0.1, min(5.0, volatility_factor))

    def _calculate_base_portfolio_risk(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate base portfolio risk from traditional indicators."""
        base_analysis = {
            "portfolio_risk": 0.0,
            "recommended_size": 0.0,
            "risk_budget": 0.0,
            "diversification_score": 0.0
        }

        # Portfolio risk indicator
        if "portfolio_risk_indicator" in indicator_results:
            risk_data = indicator_results["portfolio_risk_indicator"]
            base_analysis["portfolio_risk"] = risk_data.get("portfolio_risk", 0.0)
            base_analysis["risk_budget"] = risk_data.get("available_risk_budget", 0.0)

        # Position sizing
        if "position_sizing_indicator" in indicator_results:
            size_data = indicator_results["position_sizing_indicator"]
            base_analysis["recommended_size"] = size_data.get("optimal_size", 0.0)

        # Diversification
        if "portfolio_diversification_indicator" in indicator_results:
            div_data = indicator_results["portfolio_diversification_indicator"]
            base_analysis["diversification_score"] = div_data.get("diversification_score", 0.0)

        return base_analysis

    def _synthesize_final_decision_ai(self, indicator_results: Dict[str, Any], consensus_analysis: Dict[str, Any], portfolio_analysis: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """AI-enhanced decision synthesis with real-time data processing."""
        synthesis_analysis = {
            "authorized": False,
            "authorization_score": 0.0,
            "risk_adjusted_confidence": 0.0,
            "regime_stability": "STABLE",
            "model_confidence": 0.0,
            "final_recommendation": "NO_ACTION",
            "risk_override": False,
            "ai_confidence": 0.0,
            "data_driven_adjustment": 0.0
        }

        # Base synthesis
        base_synthesis = self._calculate_base_synthesis(indicator_results, consensus_analysis, portfolio_analysis)
        synthesis_analysis.update(base_synthesis)

        # AI enhancement using real market data
        ai_confidence = self._calculate_ai_confidence(market_data, consensus_analysis, portfolio_analysis)
        synthesis_analysis["ai_confidence"] = ai_confidence

        # Data-driven confidence adjustment
        data_quality = len([k for k in market_data.keys() if not market_data[k].empty]) / 3
        synthesis_analysis["data_driven_adjustment"] = data_quality * 0.2

        # Enhanced authorization logic
        ai_weighted_consensus = consensus_analysis.get("ai_weighted_consensus", 0.0)
        risk_adjusted_conf = synthesis_analysis["risk_adjusted_confidence"]

        synthesis_analysis["authorized"] = (
            ai_weighted_consensus >= self.min_consensus_threshold and
            (risk_adjusted_conf + synthesis_analysis["data_driven_adjustment"]) >= self.min_confidence_threshold and
            portfolio_analysis.get("portfolio_risk", 0.0) <= self.max_portfolio_risk and
            ai_confidence > 0.3
        )

        return synthesis_analysis

    def _calculate_base_synthesis(self, indicator_results: Dict[str, Any], consensus_analysis: Dict[str, Any], portfolio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate base synthesis from traditional methods."""
        base_synthesis = {
            "authorization_score": 0.0,
            "risk_adjusted_confidence": 0.0,
            "regime_stability": "STABLE",
            "model_confidence": 0.0,
            "final_recommendation": "NO_ACTION"
        }

        # Decision synthesis
        if "decision_synthesis_indicator" in indicator_results:
            synth_data = indicator_results["decision_synthesis_indicator"]
            base_synthesis["authorization_score"] = synth_data.get("authorization_score", 0.0)
            base_synthesis["final_recommendation"] = synth_data.get("final_recommendation", "NO_ACTION")

        # Model confidence
        if "model_confidence_indicator" in indicator_results:
            model_data = indicator_results["model_confidence_indicator"]
            base_synthesis["model_confidence"] = model_data.get("confidence_level", 0.0)

        # Risk-adjusted confidence
        consensus_conf = consensus_analysis.get("confidence_aggregation", 0.0)
        portfolio_risk = portfolio_analysis.get("portfolio_risk", 0.0)
        risk_penalty = min(portfolio_risk / self.max_portfolio_risk, 1.0)
        base_synthesis["risk_adjusted_confidence"] = consensus_conf * (1 - risk_penalty * 0.5)

        return base_synthesis

    def _calculate_ai_confidence(self, market_data: Dict[str, pd.DataFrame], consensus_analysis: Dict[str, Any], portfolio_analysis: Dict[str, Any]) -> float:
        """Calculate AI confidence based on data patterns and consensus quality."""
        confidence_factors = []

        # Data availability factor
        data_sources = len([k for k in market_data.keys() if not market_data[k].empty])
        confidence_factors.append(min(data_sources / 3, 1.0) * 0.3)

        # Consensus strength factor
        consensus_strength = consensus_analysis.get("ai_weighted_consensus", 0.0)
        confidence_factors.append(consensus_strength * 0.4)

        # Portfolio health factor
        diversification = portfolio_analysis.get("diversification_score", 0.0)
        confidence_factors.append(diversification * 0.2)

        # Market stability factor (from OHLCV if available)
        if 'OHLCV' in market_data and not market_data['OHLCV'].empty:
            stability_factor = self._calculate_market_stability(market_data['OHLCV'])
            confidence_factors.append(stability_factor * 0.1)

        return sum(confidence_factors)

    def _calculate_market_stability(self, ohlcv_data: pd.DataFrame) -> float:
        """Calculate market stability factor from price data."""
        if ohlcv_data.empty or len(ohlcv_data) < 10:
            return 0.5

        # Calculate price stability
        returns = ohlcv_data['close'].pct_change().dropna()
        volatility = returns.std()

        # Normalize volatility (lower volatility = higher stability)
        stability = max(0.0, min(1.0, 1.0 - (volatility * 50)))

        return stability

    def _prepare_ensemble_features(self, indicator_results: Dict[str, Any], consensus_analysis: Dict[str, Any], portfolio_analysis: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for AI ensemble training and prediction."""
        features = {}

        # Agent consensus features
        features["consensus_score"] = consensus_analysis.get("consensus_score", 0.0)
        features["confidence_aggregation"] = consensus_analysis.get("confidence_aggregation", 0.0)
        features["agent_agreement"] = 1.0 if consensus_analysis.get("agent_agreement") == "HIGH" else 0.5 if consensus_analysis.get("agent_agreement") == "MEDIUM" else 0.0

        # Portfolio risk features
        features["portfolio_risk"] = portfolio_analysis.get("portfolio_risk", 0.0)
        features["diversification_score"] = portfolio_analysis.get("diversification_score", 0.0)
        features["var_estimate"] = portfolio_analysis.get("var_estimate", 0.0)
        features["concentration_risk"] = 1.0 if portfolio_analysis.get("concentration_risk") == "HIGH" else 0.5 if portfolio_analysis.get("concentration_risk") == "MEDIUM" else 0.0

        # Model confidence features
        features["model_confidence"] = indicator_results.get("model_confidence_indicator", {}).get("confidence_level", 0.0)
        features["regime_stability"] = indicator_results.get("regime_stability_indicator", {}).get("stability_score", 0.0)

        # Synthetic features (combinations)
        features["risk_adjusted_consensus"] = features["consensus_score"] * (1 - features["portfolio_risk"])
        features["confidence_risk_ratio"] = features["confidence_aggregation"] / max(features["portfolio_risk"], 0.01)

        # Convert to DataFrame
        return pd.DataFrame([features])

    def _calculate_market_volatility(self, ohlcv_data: pd.DataFrame) -> float:
        """Calculate current market volatility."""
        if ohlcv_data.empty or len(ohlcv_data) < 10:
            return 0.02  # Default 2% volatility

        returns = ohlcv_data['close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 0 else 0.02

        return min(volatility, 0.1)  # Cap at 10% daily volatility

    def _synthesize_final_decision_ai_full(self, indicator_results: Dict[str, Any], consensus_analysis: Dict[str, Any],
                                     portfolio_analysis: Dict[str, Any], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """AI-enhanced final decision synthesis using ensemble methods."""
        try:
            # Prepare features for ensemble prediction
            features = self._prepare_ensemble_features(indicator_results, consensus_analysis, portfolio_analysis)

            # Check if ensemble needs training or retraining
            if self.decision_count % self.ensemble_training_frequency == 0:
                self._train_ensemble_if_needed(features)

            # Get ensemble prediction
            ensemble_prediction = None
            try:
                ensemble_prediction = self.ensemble_manager.predict_ensemble(features, VotingMethod.ADAPTIVE)
            except Exception as e:
                logger.warning(f"Ensemble prediction failed, using fallback: {str(e)}")
                ensemble_prediction = None

            # Base synthesis
            synthesis_analysis = self._synthesize_final_decision(indicator_results, consensus_analysis, portfolio_analysis)

            # AI enhancements
            synthesis_analysis["ai_enhanced"] = True
            synthesis_analysis["ensemble_prediction"] = None
            synthesis_analysis["ai_confidence"] = 0.5  # Default

            if ensemble_prediction:
                synthesis_analysis["ensemble_prediction"] = {
                    "prediction": ensemble_prediction.final_prediction,
                    "confidence": ensemble_prediction.ensemble_confidence,
                    "consensus_strength": ensemble_prediction.consensus_strength,
                    "overfitting_risk": ensemble_prediction.overfitting_risk
                }

                # Combine traditional and AI predictions
                traditional_decision = synthesis_analysis.get("final_decision", "HOLD")
                ai_decision = ensemble_prediction.final_prediction
                ai_confidence = ensemble_prediction.ensemble_confidence

                # Decision reconciliation
                if traditional_decision == ai_decision:
                    # Agreement - boost confidence
                    synthesis_analysis["final_decision"] = traditional_decision
                    synthesis_analysis["ai_confidence"] = min(ai_confidence * 1.2, 1.0)
                    synthesis_analysis["decision_source"] = "CONSENSUS"
                elif ai_confidence > 0.8 and ensemble_prediction.overfitting_risk < 0.3:
                    # High AI confidence, low overfitting risk - trust AI
                    synthesis_analysis["final_decision"] = ai_decision
                    synthesis_analysis["ai_confidence"] = ai_confidence * 0.9
                    synthesis_analysis["decision_source"] = "AI_OVERRIDE"
                else:
                    # Disagreement - be conservative
                    synthesis_analysis["final_decision"] = "HOLD"
                    synthesis_analysis["ai_confidence"] = max(ai_confidence * 0.6, 0.3)
                    synthesis_analysis["decision_source"] = "CONSERVATIVE"

                # Overfitting risk assessment
                if ensemble_prediction.overfitting_risk > 0.7:
                    synthesis_analysis["ai_confidence"] *= 0.7
                    synthesis_analysis["overfitting_warning"] = True
            else:
                # Fallback to traditional decision
                synthesis_analysis["final_decision"] = synthesis_analysis.get("final_decision", "HOLD")
                synthesis_analysis["ai_confidence"] = 0.5
                synthesis_analysis["decision_source"] = "TRADITIONAL_FALLBACK"

            # Final authorization based on AI confidence
            ai_conf = synthesis_analysis["ai_confidence"]
            traditional_auth = synthesis_analysis.get("authorized", False)

            if ai_conf >= self.ai_confidence_threshold and traditional_auth:
                synthesis_analysis["authorized"] = True
                synthesis_analysis["authorization_source"] = "AI_ENHANCED"
            elif ai_conf >= 0.6 and traditional_auth:
                synthesis_analysis["authorized"] = True
                synthesis_analysis["authorization_source"] = "MODERATE_AI"
            else:
                synthesis_analysis["authorized"] = False
                synthesis_analysis["authorization_source"] = "AI_REJECTED"

            self.decision_count += 1

            return synthesis_analysis

        except Exception as e:
            logger.error(f"AI synthesis failed, using traditional approach: {str(e)}")
            synthesis = self._synthesize_final_decision(indicator_results, consensus_analysis, portfolio_analysis)
            synthesis["ai_enhanced"] = False
            synthesis["ai_error"] = str(e)
            return synthesis

    def _train_ensemble_if_needed(self, features: pd.DataFrame) -> None:
        """Train or retrain the ensemble if needed."""
        try:
            # Check ensemble status
            status = self.ensemble_manager.get_ensemble_status()

            if status["health_status"] in ["unhealthy_insufficient_training", "warning_high_overfitting_risk"]:
                logger.info("Retraining ensemble due to health status")

                # Generate synthetic training data for demonstration
                # In production, this would use historical decision outcomes
                training_features, training_targets = self._generate_training_data()

                if len(training_features) > 10:
                    self.ensemble_manager.train_ensemble(training_features, training_targets)
                    logger.info("Ensemble retraining completed")
                else:
                    logger.warning("Insufficient training data for ensemble")

        except Exception as e:
            logger.error(f"Ensemble training failed: {str(e)}")

    def _generate_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic training data for ensemble."""
        # This is a simplified version - in production would use historical data
        np.random.seed(42)  # For reproducibility

        n_samples = 50
        features = []
        targets = []

        for _ in range(n_samples):
            # Generate synthetic features
            consensus_score = np.random.uniform(0.3, 0.9)
            confidence_agg = np.random.uniform(0.4, 0.8)
            portfolio_risk = np.random.uniform(0.005, 0.025)
            diversification = np.random.uniform(0.6, 0.9)

            feature_row = {
                "consensus_score": consensus_score,
                "confidence_aggregation": confidence_agg,
                "agent_agreement": 1.0 if consensus_score > 0.7 else 0.5,
                "portfolio_risk": portfolio_risk,
                "diversification_score": diversification,
                "var_estimate": portfolio_risk * 1.5,
                "concentration_risk": 0.0 if diversification > 0.8 else 0.5,
                "model_confidence": np.random.uniform(0.5, 0.9),
                "regime_stability": np.random.uniform(0.4, 0.8),
                "risk_adjusted_consensus": consensus_score * (1 - portfolio_risk),
                "confidence_risk_ratio": confidence_agg / max(portfolio_risk, 0.01)
            }

            # Generate target based on logical rules
            if (consensus_score > 0.7 and confidence_agg > 0.6 and
                portfolio_risk < 0.015 and diversification > 0.7):
                target = "BUY" if np.random.random() > 0.3 else "SELL"
            elif consensus_score < 0.4 or portfolio_risk > 0.02:
                target = "HOLD"
            else:
                target = np.random.choice(["BUY", "SELL", "HOLD"], p=[0.3, 0.3, 0.4])

            features.append(feature_row)
            targets.append(target)

        features_df = pd.DataFrame(features)
        targets_series = pd.Series(targets)

        return features_df, targets_series

    def _calculate_master_confidence_ai(self, consensus_analysis: Dict[str, Any], portfolio_analysis: Dict[str, Any],
                                       synthesis_analysis: Dict[str, Any], data_availability: Dict[str, Any]) -> float:
        """Calculate AI-enhanced master confidence."""
        confidence_components = []

        # Traditional confidence (40% weight)
        traditional_conf = self._calculate_master_confidence(consensus_analysis, portfolio_analysis, synthesis_analysis)
        confidence_components.append(traditional_conf * 0.4)

        # AI confidence (30% weight)
        ai_conf = synthesis_analysis.get("ai_confidence", 0.5)
        confidence_components.append(ai_conf * 0.3)

        # Data quality confidence (20% weight)
        data_quality = data_availability.get("quality_score", 0.5)
        confidence_components.append(data_quality * 0.2)

        # Ensemble consensus confidence (10% weight)
        if synthesis_analysis.get("ensemble_prediction"):
            ensemble_consensus = synthesis_analysis["ensemble_prediction"].get("consensus_strength", 0.5)
            confidence_components.append(ensemble_consensus * 0.1)
        else:
            confidence_components.append(0.05)  # Half weight for no ensemble

        # Overfitting penalty
        total_confidence = sum(confidence_components)

        if synthesis_analysis.get("ensemble_prediction"):
            overfitting_risk = synthesis_analysis["ensemble_prediction"].get("overfitting_risk", 0.0)
            if overfitting_risk > 0.5:
                total_confidence *= (1 - overfitting_risk * 0.3)

        # Decision source adjustment
        decision_source = synthesis_analysis.get("decision_source", "TRADITIONAL")
        if decision_source == "CONSENSUS":
            total_confidence *= 1.1  # Boost for agreement
        elif decision_source == "CONSERVATIVE":
            total_confidence *= 0.8  # Penalty for disagreement

        return max(0.0, min(total_confidence, 1.0))

    def _generate_master_reasoning_ai(self, decision: str, consensus_analysis: Dict[str, Any],
                                    portfolio_analysis: Dict[str, Any], synthesis_analysis: Dict[str, Any],
                                    data_availability: Dict[str, Any]) -> str:
        """Generate AI-enhanced reasoning for the master decision."""
        reasoning_parts = []

        # Base reasoning
        base_reasoning = self._generate_master_reasoning(decision, consensus_analysis, portfolio_analysis, synthesis_analysis)
        reasoning_parts.append(base_reasoning)

        # AI enhancement details
        if synthesis_analysis.get("ai_enhanced", False):
            decision_source = synthesis_analysis.get("decision_source", "TRADITIONAL")
            ai_confidence = synthesis_analysis.get("ai_confidence", 0.0)

            reasoning_parts.append(f"AI-enhanced analysis with {ai_confidence:.2f} confidence.")

            if decision_source == "CONSENSUS":
                reasoning_parts.append("Traditional and AI models agree on decision.")
            elif decision_source == "AI_OVERRIDE":
                reasoning_parts.append("AI model override due to high confidence and low overfitting risk.")
            elif decision_source == "CONSERVATIVE":
                reasoning_parts.append("Conservative approach due to model disagreement.")
            elif decision_source == "TRADITIONAL_FALLBACK":
                reasoning_parts.append("Using traditional analysis due to AI model unavailability.")

            # Ensemble details
            if synthesis_analysis.get("ensemble_prediction"):
                ensemble_info = synthesis_analysis["ensemble_prediction"]
                consensus_strength = ensemble_info.get("consensus_strength", 0.0)
                overfitting_risk = ensemble_info.get("overfitting_risk", 0.0)

                reasoning_parts.append(f"Ensemble consensus: {consensus_strength:.2f}, overfitting risk: {overfitting_risk:.2f}.")

                if overfitting_risk > 0.7:
                    reasoning_parts.append("High overfitting risk detected - confidence reduced.")

        # Data quality context
        data_quality = data_availability.get("quality_score", 0.5)
        data_sources = data_availability.get("available_sources", 0)
        reasoning_parts.append(f"Decision based on {data_sources} data sources with {data_quality:.2f} quality score.")

        # Authorization reasoning
        auth_source = synthesis_analysis.get("authorization_source", "TRADITIONAL")
        if auth_source == "AI_ENHANCED":
            reasoning_parts.append("Trade authorized with AI enhancement.")
        elif auth_source == "MODERATE_AI":
            reasoning_parts.append("Trade authorized with moderate AI confidence.")
        elif auth_source == "AI_REJECTED":
            reasoning_parts.append("Trade rejected due to insufficient AI confidence.")

        return " ".join(reasoning_parts)

    def _assess_master_risk_ai(self, portfolio_analysis: Dict[str, Any], synthesis_analysis: Dict[str, Any],
                              data_availability: Dict[str, Any]) -> Dict[str, Any]:
        """AI-enhanced master risk assessment."""
        # Base risk assessment
        risk_assessment = self._assess_master_risk(portfolio_analysis, synthesis_analysis)

        # AI enhancements
        risk_assessment["ai_enhanced"] = True

        # AI-based risk adjustments
        if synthesis_analysis.get("ensemble_prediction"):
            overfitting_risk = synthesis_analysis["ensemble_prediction"].get("overfitting_risk", 0.0)

            if overfitting_risk > 0.5:
                risk_assessment["ai_model_risk"] = "HIGH"
                risk_assessment["recommended_position_adjustment"] = 0.7  # Reduce position
            else:
                risk_assessment["ai_model_risk"] = "LOW"
                risk_assessment["recommended_position_adjustment"] = 1.0
        else:
            risk_assessment["ai_model_risk"] = "UNKNOWN"
            risk_assessment["recommended_position_adjustment"] = 0.8  # Conservative without AI

        # Data quality risk
        data_quality = data_availability.get("quality_score", 0.5)
        if data_quality < 0.6:
            risk_assessment["data_quality_risk"] = "HIGH"
            risk_assessment["recommended_position_adjustment"] *= 0.8
        else:
            risk_assessment["data_quality_risk"] = "LOW"

        # Decision source risk
        decision_source = synthesis_analysis.get("decision_source", "TRADITIONAL")
        if decision_source == "CONSERVATIVE":
            risk_assessment["decision_conflict_risk"] = "HIGH"
            risk_assessment["recommended_position_adjustment"] *= 0.6
        else:
            risk_assessment["decision_conflict_risk"] = "LOW"

        return risk_assessment

    def update_ensemble_performance(self, actual_outcome: str, prediction_timestamp: datetime) -> None:
        """Update ensemble performance with actual trading outcomes."""
        try:
            self.ensemble_manager.update_model_performance(actual_outcome, prediction_timestamp)
            logger.info(f"Ensemble performance updated with actual outcome: {actual_outcome}")
        except Exception as e:
            logger.error(f"Failed to update ensemble performance: {str(e)}")

    def get_ai_status(self) -> Dict[str, Any]:
        """Get comprehensive AI system status."""
        try:
            ensemble_status = self.ensemble_manager.get_ensemble_status()

            return {
                "ai_system_active": True,
                "ensemble_status": ensemble_status,
                "decision_count": self.decision_count,
                "next_training": self.ensemble_training_frequency - (self.decision_count % self.ensemble_training_frequency),
                "ai_confidence_threshold": self.ai_confidence_threshold
            }
        except Exception as e:
            logger.error(f"Failed to get AI status: {str(e)}")
            return {
                "ai_system_active": False,
                "error": str(e)
            }
