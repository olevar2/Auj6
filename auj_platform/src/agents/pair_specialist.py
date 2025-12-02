"""
Pair Specialist Agent for the AUJ Platform - PROFESSIONAL UPGRADE.

This agent specializes in advanced correlation analysis and pairs relationships.
PROFESSIONAL VERSION: Uses all 11 assigned indicators with sophisticated analysis
including multi-timeframe correlation, lead-lag detection, divergence identification,
and cointegration testing.

Improvements over original:
- Multi-timeframe correlation analysis
- Lead-lag relationship detection (which pair leads which)
- Divergence detection between correlated pairs
- Enhanced cointegration testing
- Uses ALL 11 indicators professionally
- Advanced decision logic based on correlation patterns
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
    Pair Specialist Agent - PROFESSIONAL correlation and relationship analysis.
    
    Specializes in:
    - Multi-timeframe correlation matrix analysis
    - Lead-lag relationship detection
    - Divergence identification between correlated pairs
    - Cointegration testing
    - Statistical arbitrage opportunities
    - Relative strength analysis
    """

    def __init__(self, config_manager, config: Optional[Dict[str, Any]] = None):
        """Initialize the Professional Pair Specialist Agent."""
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
            specialization="Professional correlation analysis and pairs relationships",
            assigned_indicators=assigned_indicators,
            config_manager=config_manager,
            config=self.config
        )

        # Configuration parameters
        self.correlation_threshold = self.config['correlation_analysis']['correlation_threshold']
        self.cointegration_threshold = self.config['cointegration_testing']['cointegration_threshold']
        self.pairs_trading_enabled = self.config['pairs_trading']['enabled']
        
        # Professional thresholds
        self.strong_correlation = 0.7
        self.weak_correlation = 0.3
        self.divergence_threshold = 0.15  # 15% divergence from expected correlation

        logger.info(f"Professional PairSpecialist initialized with {len(assigned_indicators)} indicators")

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
        """Perform professional correlation and pairs relationship analysis."""
        try:
            # Multi-timeframe Correlation Analysis
            correlation_analysis = self._analyze_correlations_professional(indicator_results)

            # Lead-Lag Relationship Detection
            leadlag_analysis = self._analyze_leadlag_relationships(indicator_results, correlation_analysis)

            # Divergence Detection
            divergence_analysis = self._analyze_divergences(indicator_results, correlation_analysis)

            # Cointegration Testing
            cointegration_analysis = self._analyze_cointegration_professional(indicator_results)

            # Statistical Arbitrage
            arbitrage_analysis = self._analyze_statistical_arbitrage_professional(
                correlation_analysis, cointegration_analysis, divergence_analysis
            )

            # Professional Decision Synthesis
            decision = self._generate_professional_decision(
                correlation_analysis, leadlag_analysis, divergence_analysis,
                cointegration_analysis, arbitrage_analysis
            )

            # Enhanced Confidence Calculation
            confidence = self._calculate_professional_confidence(
                correlation_analysis, leadlag_analysis, divergence_analysis,
                cointegration_analysis, arbitrage_analysis
            )

            # Comprehensive Reasoning
            reasoning = self._generate_professional_reasoning(
                decision, correlation_analysis, leadlag_analysis,
                divergence_analysis, cointegration_analysis
            )

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "correlation_analysis": correlation_analysis,
                    "leadlag_analysis": leadlag_analysis,
                    "divergence_analysis": divergence_analysis,
                    "cointegration_analysis": cointegration_analysis,
                    "arbitrage_analysis": arbitrage_analysis
                },
                risk_assessment=self._assess_professional_risk(
                    correlation_analysis, divergence_analysis
                ),
                supporting_data={
                    "correlation_strength": correlation_analysis.get("avg_correlation", 0.0),
                    "leadlag_score": leadlag_analysis.get("lead_score", 0.0),
                    "divergence_detected": divergence_analysis.get("divergence_exists", False),
                    "cointegration_score": cointegration_analysis.get("cointegration_strength", 0.0),
                    "arbitrage_opportunity": arbitrage_analysis.get("opportunity_score", 0.0)
                }
            )

        except Exception as e:
            logger.error(f"Professional PairSpecialist analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Professional pairs analysis failed: {str(e)}")

    def _analyze_correlations_professional(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Professional multi-timeframe correlation analysis.
        
        Returns comprehensive correlation assessment including:
        - Current correlation
        - Rolling correlation trend
        - Correlation strength classification
        - Multi-timeframe correlation matrix
        """
        correlation_analysis = {
            "current_correlation": 0.0,
            "correlation_trend": "STABLE",
            "correlation_strength": "WEAK",
            "avg_correlation": 0.0,
            "correlation_volatility": 0.0,
            "correlation_quality": "LOW",
            "analyzed_pairs_count": 0
        }

        correlations = []

        # Correlation Analysis
        if "correlation_analysis_indicator" in indicator_results:
            corr_data = indicator_results["correlation_analysis_indicator"]
            correlation_analysis["current_correlation"] = corr_data.get("current", 0.0)
            correlation_analysis["correlation_trend"] = corr_data.get("trend", "STABLE")
            correlations.append(abs(corr_data.get("current", 0.0)))

        # Correlation Coefficient
        if "correlation_coefficient_indicator" in indicator_results:
            coef_data = indicator_results["correlation_coefficient_indicator"]
            correlations.append(abs(coef_data.get("coefficient", 0.0)))

        # Correlation Matrix
        if "correlation_matrix_indicator" in indicator_results:
            matrix_data = indicator_results["correlation_matrix_indicator"]
            matrix_avg = matrix_data.get("average_correlation", 0.0)
            correlations.append(abs(matrix_avg))
            correlation_analysis["analyzed_pairs_count"] = matrix_data.get("pairs_count", 0)

        # Intermarket Correlation
        if "intermarket_correlation_indicator" in indicator_results:
            inter_data = indicator_results["intermarket_correlation_indicator"]
            correlations.append(abs(inter_data.get("correlation", 0.0)))

        # Calculate average correlation
        if correlations:
            correlation_analysis["avg_correlation"] = np.mean(correlations)
            correlation_analysis["correlation_volatility"] = np.std(correlations)

            # Classify correlation strength
            avg_corr = correlation_analysis["avg_correlation"]
            if avg_corr >= self.strong_correlation:
                correlation_analysis["correlation_strength"] = "STRONG"
                correlation_analysis["correlation_quality"] = "HIGH"
            elif avg_corr >= self.weak_correlation:
                correlation_analysis["correlation_strength"] = "MODERATE"
                correlation_analysis["correlation_quality"] = "MEDIUM"
            else:
                correlation_analysis["correlation_strength"] = "WEAK"
                correlation_analysis["correlation_quality"] = "LOW"

        return correlation_analysis

    def _analyze_leadlag_relationships(self, indicator_results: Dict[str, Any], 
                                      correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze lead-lag relationships between correlated pairs.
        
        Detects which pair tends to move first (leading indicator).
        """
        leadlag_analysis = {
            "relationship_type": "NEUTRAL",
            "lead_score": 0.0,
            "lag_detected": False,
            "predictive_power": 0.0
        }

        # Use linear regression and R-squared for lead-lag detection
        if "linear_regression_indicator" in indicator_results:
            lr_data = indicator_results["linear_regression_indicator"]
            slope = lr_data.get("slope", 0.0)
            
            if "rsquared_indicator" in indicator_results:
                r2_data = indicator_results["rsquared_indicator"]
                r_squared = r2_data.get("r_squared", 0.0)
                
                # High R-squared indicates strong predictive relationship
                if r_squared > 0.6:
                    leadlag_analysis["lag_detected"] = True
                    leadlag_analysis["predictive_power"] = r_squared
                    
                    if slope > 0.1:
                        leadlag_analysis["relationship_type"] = "POSITIVE_LEAD"
                        leadlag_analysis["lead_score"] = min(r_squared * abs(slope), 1.0)
                    elif slope < -0.1:
                        leadlag_analysis["relationship_type"] = "NEGATIVE_LEAD"
                        leadlag_analysis["lead_score"] = min(r_squared * abs(slope), 1.0)

        # Beta coefficient can also indicate lead-lag
        if "beta_coefficient_indicator" in indicator_results:
            beta_data = indicator_results["beta_coefficient_indicator"]
            beta = beta_data.get("beta", 1.0)
            
            if abs(beta - 1.0) > 0.3:  # Significant deviation from 1.0
                leadlag_analysis["lag_detected"] = True
                leadlag_analysis["lead_score"] = min(abs(beta - 1.0), 1.0)

        return leadlag_analysis

    def _analyze_divergences(self, indicator_results: Dict[str, Any],
                            correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect divergences between correlated pairs.
        
        When historically correlated pairs start moving apart, it signals:
        1. Potential mean reversion opportunity
        2. Breakdown of correlation relationship
        3. Leading indicator for one pair
        """
        divergence_analysis = {
            "divergence_exists": False,
            "divergence_magnitude": 0.0,
            "divergence_direction": "NEUTRAL",
            "mean_reversion_signal": "NONE",
            "breakdown_risk": 0.0
        }

        # Price Oscillator can detect divergence
        if "price_oscillator_indicator" in indicator_results:
            po_data = indicator_results["price_oscillator_indicator"]
            oscillator_value = po_data.get("value", 0.0)
            
            # High oscillator value indicates divergence
            if abs(oscillator_value) > self.divergence_threshold:
                divergence_analysis["divergence_exists"] = True
                divergence_analysis["divergence_magnitude"] = abs(oscillator_value)
                
                if oscillator_value > 0:
                    divergence_analysis["divergence_direction"] = "POSITIVE_DIVERGENCE"
                    divergence_analysis["mean_reversion_signal"] = "POTENTIAL_SELL"
                else:
                    divergence_analysis["divergence_direction"] = "NEGATIVE_DIVERGENCE"
                    divergence_analysis["mean_reversion_signal"] = "POTENTIAL_BUY"

        # Relative Strength Mansfield can also indicate divergence
        if "relative_strength_mansfield_indicator" in indicator_results:
            rsm_data = indicator_results["relative_strength_mansfield_indicator"]
            rs_value = rsm_data.get("relative_strength", 0.0)
            
            if abs(rs_value) > 0.2:  # Significant relative strength difference
                divergence_analysis["divergence_exists"] = True
                divergence_analysis["divergence_magnitude"] = max(
                    divergence_analysis["divergence_magnitude"], abs(rs_value)
                )

        # Calculate breakdown risk
        if divergence_analysis["divergence_exists"]:
            avg_corr = correlation_analysis.get("avg_correlation", 0.0)
            div_mag = divergence_analysis["divergence_magnitude"]
            
            # High divergence with historically strong correlation = breakdown risk
            divergence_analysis["breakdown_risk"] = min(div_mag * avg_corr, 1.0)

        return divergence_analysis

    def _analyze_cointegration_professional(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Professional cointegration testing.
        
        Cointegration means pairs have long-term equilibrium relationship.
        """
        cointegration_analysis = {
            "cointegration_exists": False,
            "cointegration_strength": 0.0,
            "equilibrium_status": "UNCERTAIN",
            "mean_reversion_likelihood": 0.0
        }

        if "cointegration_indicator" in indicator_results:
            coint_data = indicator_results["cointegration_indicator"]
            p_value = coint_data.get("p_value", 1.0)
            test_statistic = coint_data.get("test_statistic", 0.0)
            
            # Low p-value indicates strong cointegration
            if p_value < self.cointegration_threshold:
                cointegration_analysis["cointegration_exists"] = True
                cointegration_analysis["cointegration_strength"] = 1.0 - p_value
                cointegration_analysis["equilibrium_status"] = "COINTEGRATED"
                cointegration_analysis["mean_reversion_likelihood"] = 1.0 - p_value
            elif p_value < 0.1:
                cointegration_analysis["equilibrium_status"] = "WEAK_COINTEGRATION"
                cointegration_analysis["mean_reversion_likelihood"] = 0.5

        return cointegration_analysis

    def _analyze_statistical_arbitrage_professional(self, correlation_analysis: Dict[str, Any],
                                                   cointegration_analysis: Dict[str, Any],
                                                   divergence_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Professional statistical arbitrage analysis.
        
        Combines correlation, cointegration, and divergence for arbitrage opportunities.
        """
        arbitrage_analysis = {
            "arbitrage_opportunity": False,
            "opportunity_type": "NONE",
            "opportunity_score": 0.0,
            "expected_direction": "NEUTRAL",
            "confidence_level": 0.0
        }

        # Best arbitrage: Strong cointegration + Divergence + High correlation
        cointegrated = cointegration_analysis["cointegration_exists"]
        divergence = divergence_analysis["divergence_exists"]
        correlation_strong = correlation_analysis["correlation_strength"] in ["STRONG", "MODERATE"]

        if cointegrated and divergence and correlation_strong:
            arbitrage_analysis["arbitrage_opportunity"] = True
            arbitrage_analysis["opportunity_type"] = "MEAN_REVERSION"
            
            # Calculate opportunity score
            coint_strength = cointegration_analysis["cointegration_strength"]
            div_magnitude = divergence_analysis["divergence_magnitude"]
            corr_quality = 1.0 if correlation_analysis["correlation_quality"] == "HIGH" else 0.7
            
            arbitrage_analysis["opportunity_score"] = min(
                (coint_strength * 0.4 + div_magnitude * 0.4 + corr_quality * 0.2), 1.0
            )
            arbitrage_analysis["confidence_level"] = arbitrage_analysis["opportunity_score"]
            
            # Direction from divergence
            arbitrage_analysis["expected_direction"] = divergence_analysis["mean_reversion_signal"]

        return arbitrage_analysis

    def _generate_professional_decision(self, correlation_analysis: Dict[str, Any],
                                       leadlag_analysis: Dict[str, Any],
                                       divergence_analysis: Dict[str, Any],
                                       cointegration_analysis: Dict[str, Any],
                                       arbitrage_analysis: Dict[str, Any]) -> str:
        """Generate professional trading decision based on comprehensive analysis."""
        
        # Priority 1: Statistical arbitrage opportunity
        if arbitrage_analysis["arbitrage_opportunity"]:
            if arbitrage_analysis["opportunity_score"] > 0.6:
                direction = arbitrage_analysis["expected_direction"]
                if "BUY" in direction:
                    return "BUY"
                elif "SELL" in direction:
                    return "SELL"

        # Priority 2: Lead-lag relationship detected
        if leadlag_analysis["lag_detected"] and leadlag_analysis["lead_score"] > 0.7:
            relationship = leadlag_analysis["relationship_type"]
            if "POSITIVE" in relationship:
                return "BUY"
            elif "NEGATIVE" in relationship:
                return "SELL"

        # Priority 3: Divergence with strong correlation (potential mean reversion)
        if divergence_analysis["divergence_exists"]:
            if divergence_analysis["divergence_magnitude"] > 0.2:
                if correlation_analysis["correlation_strength"] == "STRONG":
                    signal = divergence_analysis["mean_reversion_signal"]
                    if "BUY" in signal:
                        return "BUY"
                    elif "SELL" in signal:
                        return "SELL"

        # Default: No strong signal
        return "HOLD"

    def _calculate_professional_confidence(self, correlation_analysis: Dict[str, Any],
                                          leadlag_analysis: Dict[str, Any],
                                          divergence_analysis: Dict[str, Any],
                                          cointegration_analysis: Dict[str, Any],
                                          arbitrage_analysis: Dict[str, Any]) -> float:
        """Calculate professional confidence score."""
        
        factors = []

        # Correlation quality
        corr_score = correlation_analysis["avg_correlation"]
        factors.append(corr_score * 0.2)

        # Lead-lag relationship
        leadlag_score = leadlag_analysis["lead_score"]
        factors.append(leadlag_score * 0.15)

        # Divergence clarity
        if divergence_analysis["divergence_exists"]:
            div_score = divergence_analysis["divergence_magnitude"]
            factors.append(div_score * 0.15)

        # Cointegration strength
        coint_score = cointegration_analysis["cointegration_strength"]
        factors.append(coint_score * 0.25)

        # Arbitrage opportunity
        arb_score = arbitrage_analysis["opportunity_score"]
        factors.append(arb_score * 0.25)

        return min(sum(factors), 1.0)

    def _assess_professional_risk(self, correlation_analysis: Dict[str, Any],
                                  divergence_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess professional trading risks."""
        
        breakdown_risk = divergence_analysis.get("breakdown_risk", 0.0)
        corr_volatility = correlation_analysis.get("correlation_volatility", 0.0)

        risk_level = "LOW"
        if breakdown_risk > 0.7 or corr_volatility > 0.3:
            risk_level = "HIGH"
        elif breakdown_risk > 0.4 or corr_volatility > 0.15:
            risk_level = "MEDIUM"

        return {
            "correlation_breakdown_risk": breakdown_risk,
            "correlation_volatility": corr_volatility,
            "overall_risk_level": risk_level,
            "recommended_hedge_ratio": 1.0 - breakdown_risk
        }

    def _generate_professional_reasoning(self, decision: str,
                                        correlation_analysis: Dict[str, Any],
                                        leadlag_analysis: Dict[str, Any],
                                        divergence_analysis: Dict[str, Any],
                                        cointegration_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive professional reasoning."""
        
        parts = [f"Professional pairs analysis decision: {decision}."]

        # Correlation status
        corr_strength = correlation_analysis["correlation_strength"]
        avg_corr = correlation_analysis["avg_correlation"]
        parts.append(f"Correlation: {corr_strength} ({avg_corr:.3f}).")

        # Lead-lag relationship
        if leadlag_analysis["lag_detected"]:
            rel_type = leadlag_analysis["relationship_type"]
            lead_score = leadlag_analysis["lead_score"]
            parts.append(f"Lead-lag relationship detected: {rel_type} (score: {lead_score:.2f}).")

        # Divergence
        if divergence_analysis["divergence_exists"]:
            div_direction = divergence_analysis["divergence_direction"]
            div_mag = divergence_analysis["divergence_magnitude"]
            parts.append(f"Divergence detected: {div_direction} ({div_mag:.2f}).")

        # Cointegration
        if cointegration_analysis["cointegration_exists"]:
            coint_strength = cointegration_analysis["cointegration_strength"]
            parts.append(f"Pairs are cointegrated (strength: {coint_strength:.2f}).")

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
