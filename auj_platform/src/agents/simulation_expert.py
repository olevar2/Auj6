"""
Simulation Expert Agent for the AUJ Platform.

This agent specializes in Monte Carlo simulations and scenario analysis.
It focuses on 12 simulation and forecasting indicators.
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


class SimulationExpert(BaseAgent):
    """
    Simulation Expert Agent - Monte Carlo simulations and scenario analysis.

    Specializes in:
    - Monte Carlo price simulations
    - Scenario analysis and stress testing
    - Probability distributions
    - Risk scenario modeling
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_manager: Optional[UnifiedConfigManager] = None):
        """Initialize the Simulation Expert Agent."""

        assigned_indicators = [
            # Support/Resistance & Others (16)
            "standard_deviation_indicator",
            "support_resistance_indicator",
            "stochastic_oscillator_indicator",
            "synthetic_option_indicator",
            "time_segmented_volume_indicator",
            "triangular_moving_average_indicator",
            "triple_exponential_average_indicator",
            "true_strength_index_indicator",
            "ultimate_oscillator_indicator",
            "variable_moving_average_indicator",
            "weighted_moving_average_indicator",
            "williams_r_indicator",
            "renko_indicator",
            "zig_zag_indicator",
            "ai_parabolic_sar_indicator",
            "pivot_point_indicator"
        ]

        super().__init__(
            name="SimulationExpert",
            specialization="Monte Carlo simulations and scenario analysis",
            assigned_indicators=assigned_indicators,
            config=config,
            config_manager=config_manager
        )

        # Simulation parameters from unified configuration
        self.num_simulations = self.config_manager.get_int('agents.simulation_expert.simulation_parameters.num_simulations', 1000)
        self.confidence_levels = self.config_manager.get_list('agents.simulation_expert.confidence_levels', [0.95, 0.99])
        self.stress_scenarios = self.config_manager.get_dict('agents.simulation_expert.stress_scenarios', {})
        self.tail_threshold = self.config_manager.get_float('agents.simulation_expert.risk_metrics.tail_threshold', 0.05)

        logger.info(f"SimulationExpert initialized with {len(assigned_indicators)} indicators")

    def _load_agent_config(self) -> Dict[str, Any]:
        """Load agent configuration from YAML file with fallback defaults."""
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config', 'agents', 'simulation_expert.yaml'
            )

            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded SimulationExpert configuration from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file not found at {config_path}, using defaults")
        except Exception as e:
            logger.error(f"Error loading SimulationExpert configuration: {e}")

        # Fallback defaults
        return {
            'simulation_parameters': {
                'num_simulations': 10000,
                'num_scenarios': 1000,
                'confidence_interval': 0.95,
                'lookback_periods': 252
            },
            'confidence_levels': [0.95, 0.99, 0.999],
            'stress_scenarios': ['crash', 'rally', 'volatility_spike', 'correlation_breakdown'],
            'risk_metrics': {
                'tail_threshold': 0.05,
                'var_confidence': 0.95,
                'expected_shortfall_alpha': 0.05
            }
        }

    async def analyze_market(self,
                           symbol: str,
                           market_data: Dict[str, pd.DataFrame],
                           market_conditions: MarketConditions,
                           indicator_results: Dict[str, Any]) -> AnalysisResult:
        """Perform simulation-based analysis."""
        try:
            # Monte Carlo Analysis
            mc_analysis = self._analyze_monte_carlo(indicator_results)

            # Scenario Analysis
            scenario_analysis = self._analyze_scenarios(indicator_results)

            # Risk Simulation
            risk_analysis = self._analyze_risk_simulation(indicator_results)

            # Generate Decision
            decision = self._generate_simulation_decision(mc_analysis, scenario_analysis, risk_analysis)

            # Calculate Confidence
            confidence = self._calculate_simulation_confidence(mc_analysis, scenario_analysis, risk_analysis)

            # Generate Reasoning
            reasoning = self._generate_simulation_reasoning(decision, mc_analysis, scenario_analysis, risk_analysis)

            return AnalysisResult(
                agent_name=self.name,
                symbol=symbol,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                indicators_used=self._get_used_indicators(indicator_results),
                technical_analysis={
                    "monte_carlo_analysis": mc_analysis,
                    "scenario_analysis": scenario_analysis,
                    "risk_analysis": risk_analysis
                },
                risk_assessment=self._assess_simulation_risk(scenario_analysis, risk_analysis),
                supporting_data={
                    "expected_return": mc_analysis.get("expected_return", 0.0),
                    "win_probability": mc_analysis.get("win_probability", 0.5),
                    "worst_case_scenario": scenario_analysis.get("worst_case_loss", 0.0),
                    "tail_risk": risk_analysis.get("tail_risk_score", 0.0)
                }
            )

        except Exception as e:
            logger.error(f"SimulationExpert analysis failed for {symbol}: {str(e)}")
            raise AgentError(f"Simulation analysis failed: {str(e)}")

    def _analyze_monte_carlo(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results."""
        mc_analysis = {
            "expected_return": 0.0,
            "volatility": 0.0,
            "win_probability": 0.5,
            "loss_probability": 0.5,
            "expected_profit": 0.0,
            "expected_loss": 0.0,
            "sharpe_ratio": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "simulation_quality": "MEDIUM"
        }

        # Monte Carlo simulation results
        if "monte_carlo_simulation_indicator" in indicator_results:
            mc_data = indicator_results["monte_carlo_simulation_indicator"]
            mc_analysis["expected_return"] = mc_data.get("expected_return", 0.0)
            mc_analysis["volatility"] = mc_data.get("volatility", 0.0)
            mc_analysis["win_probability"] = mc_data.get("win_probability", 0.5)
            mc_analysis["loss_probability"] = 1 - mc_analysis["win_probability"]
            mc_analysis["sharpe_ratio"] = mc_data.get("sharpe_ratio", 0.0)
            mc_analysis["skewness"] = mc_data.get("skewness", 0.0)
            mc_analysis["kurtosis"] = mc_data.get("kurtosis", 0.0)

            # Calculate expected profit/loss
            if mc_analysis["expected_return"] > 0:
                mc_analysis["expected_profit"] = mc_analysis["expected_return"]
            else:
                mc_analysis["expected_loss"] = abs(mc_analysis["expected_return"])

        # Probability distribution analysis
        if "probability_distribution_indicator" in indicator_results:
            prob_data = indicator_results["probability_distribution_indicator"]
            distribution_quality = prob_data.get("distribution_quality", "MEDIUM")
            mc_analysis["simulation_quality"] = distribution_quality

        return mc_analysis

    def _analyze_scenarios(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scenario analysis results."""
        scenario_analysis = {
            "best_case_return": 0.0,
            "worst_case_loss": 0.0,
            "base_case_return": 0.0,
            "stress_test_results": {},
            "scenario_robustness": "MEDIUM",
            "black_swan_risk": 0.0,
            "correlation_breakdown_risk": 0.0
        }

        # Scenario analysis results
        if "scenario_analysis_indicator" in indicator_results:
            scenario_data = indicator_results["scenario_analysis_indicator"]
            scenario_analysis["best_case_return"] = scenario_data.get("best_case", 0.0)
            scenario_analysis["worst_case_loss"] = scenario_data.get("worst_case", 0.0)
            scenario_analysis["base_case_return"] = scenario_data.get("base_case", 0.0)
            scenario_analysis["scenario_robustness"] = scenario_data.get("robustness", "MEDIUM")

        # Stress test results
        if "stress_test_indicator" in indicator_results:
            stress_data = indicator_results["stress_test_indicator"]
            scenario_analysis["stress_test_results"] = stress_data.get("test_results", {})

        # Black swan analysis
        if "black_swan_indicator" in indicator_results:
            swan_data = indicator_results["black_swan_indicator"]
            scenario_analysis["black_swan_risk"] = swan_data.get("black_swan_probability", 0.0)

        # Correlation breakdown analysis
        if "correlation_breakdown_indicator" in indicator_results:
            corr_data = indicator_results["correlation_breakdown_indicator"]
            scenario_analysis["correlation_breakdown_risk"] = corr_data.get("breakdown_probability", 0.0)

        return scenario_analysis

    def _analyze_risk_simulation(self, indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk simulation results."""
        risk_analysis = {
            "var_95": 0.0,
            "var_99": 0.0,
            "expected_shortfall_95": 0.0,
            "expected_shortfall_99": 0.0,
            "tail_risk_score": 0.0,
            "extreme_loss_probability": 0.0,
            "fat_tail_detected": False,
            "risk_distribution": "NORMAL"
        }

        # VaR simulation results
        if "value_at_risk_simulation_indicator" in indicator_results:
            var_data = indicator_results["value_at_risk_simulation_indicator"]
            risk_analysis["var_95"] = var_data.get("var_95", 0.0)
            risk_analysis["var_99"] = var_data.get("var_99", 0.0)

        # Expected Shortfall simulation
        if "expected_shortfall_simulation_indicator" in indicator_results:
            es_data = indicator_results["expected_shortfall_simulation_indicator"]
            risk_analysis["expected_shortfall_95"] = es_data.get("es_95", 0.0)
            risk_analysis["expected_shortfall_99"] = es_data.get("es_99", 0.0)

        # Tail risk analysis
        if "tail_risk_indicator" in indicator_results:
            tail_data = indicator_results["tail_risk_indicator"]
            risk_analysis["tail_risk_score"] = tail_data.get("tail_risk_score", 0.0)
            risk_analysis["extreme_loss_probability"] = tail_data.get("extreme_loss_prob", 0.0)

        # Fat tail detection
        if "fat_tail_indicator" in indicator_results:
            fat_data = indicator_results["fat_tail_indicator"]
            risk_analysis["fat_tail_detected"] = fat_data.get("fat_tail_detected", False)
            if risk_analysis["fat_tail_detected"]:
                risk_analysis["risk_distribution"] = "FAT_TAIL"

        # Extreme value analysis
        if "extreme_value_indicator" in indicator_results:
            extreme_data = indicator_results["extreme_value_indicator"]
            extreme_prob = extreme_data.get("extreme_probability", 0.0)
            risk_analysis["extreme_loss_probability"] = max(risk_analysis["extreme_loss_probability"], extreme_prob)

        return risk_analysis

    def _generate_simulation_decision(self, mc_analysis: Dict[str, Any], scenario_analysis: Dict[str, Any], risk_analysis: Dict[str, Any]) -> str:
        """Generate decision based on simulation analysis."""
        expected_return = mc_analysis.get("expected_return", 0.0)
        win_probability = mc_analysis.get("win_probability", 0.5)
        worst_case_loss = scenario_analysis.get("worst_case_loss", 0.0)
        tail_risk_score = risk_analysis.get("tail_risk_score", 0.0)
        sharpe_ratio = mc_analysis.get("sharpe_ratio", 0.0)

        # Strong positive expectation with acceptable risk
        if expected_return > 0.01 and win_probability > 0.6 and tail_risk_score < 0.1 and sharpe_ratio > 1.0:
            return "BUY"

        # Moderate positive expectation
        if expected_return > 0.005 and win_probability > 0.55 and tail_risk_score < 0.15:
            return "BUY"

        # Negative expectation or high tail risk
        if expected_return < -0.005 or tail_risk_score > 0.2 or win_probability < 0.4:
            return "SELL"

        # High tail risk but positive expectation
        if tail_risk_score > 0.15 and expected_return > 0:
            return "HOLD"  # Wait for better conditions

        # Extreme scenarios
        if abs(worst_case_loss) > 0.1:  # 10% worst case loss
            return "AVOID"

        return "NO_SIGNAL"

    def _calculate_simulation_confidence(self, mc_analysis: Dict[str, Any], scenario_analysis: Dict[str, Any], risk_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in simulation analysis."""
        factors = []

        # Monte Carlo confidence
        simulation_quality = mc_analysis.get("simulation_quality", "MEDIUM")
        if simulation_quality == "HIGH":
            factors.append(0.3)
        elif simulation_quality == "MEDIUM":
            factors.append(0.2)
        else:
            factors.append(0.1)

        # Win probability confidence
        win_prob = mc_analysis.get("win_probability", 0.5)
        prob_confidence = abs(win_prob - 0.5) * 2  # Convert to 0-1 scale
        factors.append(prob_confidence * 0.2)

        # Scenario robustness
        robustness = scenario_analysis.get("scenario_robustness", "MEDIUM")
        if robustness == "HIGH":
            factors.append(0.2)
        elif robustness == "MEDIUM":
            factors.append(0.1)

        # Risk clarity (inverse of tail risk)
        tail_risk = risk_analysis.get("tail_risk_score", 0.0)
        risk_clarity = max(0, 1 - tail_risk * 2)  # Higher tail risk reduces confidence
        factors.append(risk_clarity * 0.3)

        return min(sum(factors), 1.0)

    def _assess_simulation_risk(self, scenario_analysis: Dict[str, Any], risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess simulation-based risks."""
        worst_case_loss = abs(scenario_analysis.get("worst_case_loss", 0.0))
        tail_risk_score = risk_analysis.get("tail_risk_score", 0.0)
        black_swan_risk = scenario_analysis.get("black_swan_risk", 0.0)
        fat_tail_detected = risk_analysis.get("fat_tail_detected", False)

        # Overall simulation risk
        if worst_case_loss > 0.05 or tail_risk_score > 0.15 or black_swan_risk > 0.05:
            sim_risk = "HIGH"
        elif worst_case_loss > 0.02 or tail_risk_score > 0.1:
            sim_risk = "MEDIUM"
        else:
            sim_risk = "LOW"

        # Tail risk assessment
        if fat_tail_detected or tail_risk_score > 0.2:
            tail_risk_level = "HIGH"
        elif tail_risk_score > 0.1:
            tail_risk_level = "MEDIUM"
        else:
            tail_risk_level = "LOW"

        return {
            "simulation_risk": sim_risk,
            "tail_risk": tail_risk_level,
            "scenario_risk": "HIGH" if worst_case_loss > 0.1 else "MEDIUM",
            "black_swan_risk": "HIGH" if black_swan_risk > 0.02 else "LOW",
            "distribution_risk": "HIGH" if fat_tail_detected else "MEDIUM"
        }

    def _generate_simulation_reasoning(self, decision: str, mc_analysis: Dict[str, Any], scenario_analysis: Dict[str, Any], risk_analysis: Dict[str, Any]) -> str:
        """Generate reasoning for simulation analysis."""
        parts = [f"Simulation analysis decision: {decision}."]

        expected_return = mc_analysis.get("expected_return", 0.0)
        parts.append(f"Expected return: {expected_return:.4f}.")

        win_probability = mc_analysis.get("win_probability", 0.5)
        parts.append(f"Win probability: {win_probability:.3f}.")

        worst_case_loss = scenario_analysis.get("worst_case_loss", 0.0)
        parts.append(f"Worst case loss: {worst_case_loss:.4f}.")

        tail_risk = risk_analysis.get("tail_risk_score", 0.0)
        parts.append(f"Tail risk score: {tail_risk:.3f}.")

        if risk_analysis.get("fat_tail_detected", False):
            parts.append("Fat tail distribution detected.")

        if scenario_analysis.get("black_swan_risk", 0.0) > 0.02:
            parts.append("Elevated black swan risk.")

        return " ".join(parts)

    def _get_used_indicators(self, indicator_results: Dict[str, Any]) -> List[str]:
        """Get list of indicators actually used in analysis."""
        return [ind for ind in self.assigned_indicators if ind in indicator_results]

    def get_required_data_types(self) -> List[str]:
        """Define required data types."""
        return ["OHLCV", "HISTORICAL"]  # Needs historical data for simulations

    def get_minimum_data_points(self) -> int:
        """Define minimum data points needed."""
        return 252  # Need at least 1 year of data for meaningful simulations
    # ============== DATA-AWARE ENHANCEMENT METHODS ==============

    def _assess_data_availability(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess data availability and quality for enhanced simulations."""
        data_availability = {
            "ohlcv_available": False,
            "historical_depth": 0,
            "tick_available": False,
            "volume_available": False,
            "data_quality_score": 0.0,
            "simulation_reliability": "LOW",
            "bootstrap_feasible": False
        }

        total_score = 0.0
        max_score = 0.0

        # Check OHLCV data depth
        if 'OHLCV' in market_data and not market_data['OHLCV'].empty:
            data_availability["ohlcv_available"] = True
            ohlcv_data = market_data['OHLCV']
            data_availability["historical_depth"] = len(ohlcv_data)

            # Quality score based on data depth
            if len(ohlcv_data) >= 252:  # 1 year
                total_score += 0.4
                data_availability["bootstrap_feasible"] = True
            elif len(ohlcv_data) >= 60:  # 2-3 months
                total_score += 0.3
            elif len(ohlcv_data) >= 20:  # 3-4 weeks
                total_score += 0.2

            # Volume data availability
            if 'volume' in ohlcv_data.columns and not ohlcv_data['volume'].isna().all():
                data_availability["volume_available"] = True
                total_score += 0.2

        max_score += 0.6

        # Check tick data for micro-structure simulations
        if 'TICK' in market_data and not market_data['TICK'].empty:
            data_availability["tick_available"] = True
            total_score += 0.3
        max_score += 0.3

        # Additional historical data
        if 'HISTORICAL' in market_data and not market_data['HISTORICAL'].empty:
            total_score += 0.1
        max_score += 0.1

        # Calculate quality score
        data_availability["data_quality_score"] = total_score / max_score if max_score > 0 else 0.0

        # Determine simulation reliability
        if data_availability["data_quality_score"] >= 0.8:
            data_availability["simulation_reliability"] = "HIGH"
        elif data_availability["data_quality_score"] >= 0.6:
            data_availability["simulation_reliability"] = "MEDIUM"
        else:
            data_availability["simulation_reliability"] = "LOW"

        return data_availability

    def _run_monte_carlo_simulation_enhanced(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], data_availability: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Monte Carlo simulation adapted to available data."""
        monte_carlo_analysis = {
            "expected_return": 0.0,
            "return_volatility": 0.0,
            "win_probability": 0.5,
            "confidence_intervals": {},
            "convergence_score": 0.0,
            "simulation_method": "TRADITIONAL",
            "accuracy_estimate": 0.5,
            "data_driven_adjustments": {}
        }

        # Base Monte Carlo analysis
        base_mc = self._analyze_monte_carlo(indicator_results)
        monte_carlo_analysis.update(base_mc)

        # Enhanced simulations based on data availability
        if data_availability.get("bootstrap_feasible", False) and data_availability.get("ohlcv_available", False):
            # Historical Bootstrap simulation
            historical_returns = self._calculate_historical_returns(market_data['OHLCV'])

            if len(historical_returns) > 30:
                bootstrap_results = self._run_bootstrap_simulation(historical_returns)
                monte_carlo_analysis.update(bootstrap_results)
                monte_carlo_analysis["simulation_method"] = "BOOTSTRAP_ENHANCED"
                monte_carlo_analysis["accuracy_estimate"] = min(0.9, 0.5 + len(historical_returns) / 500)

        # Micro-structure enhanced simulations
        if data_availability.get("tick_available", False):
            tick_enhanced_results = self._run_tick_enhanced_simulation(market_data['TICK'])
            monte_carlo_analysis["data_driven_adjustments"]["tick_enhancement"] = tick_enhanced_results
            monte_carlo_analysis["accuracy_estimate"] *= 1.1  # Boost accuracy with tick data

        # Volume-based enhancements
        if data_availability.get("volume_available", False):
            volume_weighted_results = self._run_volume_weighted_simulation(market_data['OHLCV'])
            monte_carlo_analysis["data_driven_adjustments"]["volume_weighting"] = volume_weighted_results

        # Convergence assessment based on data quality
        data_quality = data_availability.get("data_quality_score", 0.0)
        monte_carlo_analysis["convergence_score"] = min(1.0, base_mc.get("convergence_score", 0.5) * (0.5 + data_quality * 0.5))

        return monte_carlo_analysis

    def _calculate_historical_returns(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """Calculate historical returns for bootstrap simulation."""
        if 'close' not in ohlcv_data.columns or len(ohlcv_data) < 2:
            return np.array([])

        returns = ohlcv_data['close'].pct_change().dropna()
        return returns.values

    def _run_bootstrap_simulation(self, historical_returns: np.ndarray, n_simulations: int = 1000) -> Dict[str, Any]:
        """Run bootstrap simulation using historical returns."""
        if len(historical_returns) < 10:
            return {"expected_return": 0.0, "return_volatility": 0.0, "bootstrap_valid": False}

        # Bootstrap resampling
        simulated_returns = []
        for _ in range(n_simulations):
            # Random sampling with replacement
            bootstrap_sample = np.random.choice(historical_returns, size=len(historical_returns), replace=True)
            simulated_returns.append(np.mean(bootstrap_sample))

        simulated_returns = np.array(simulated_returns)

        return {
            "expected_return": np.mean(simulated_returns),
            "return_volatility": np.std(simulated_returns),
            "confidence_intervals": {
                "95%": [np.percentile(simulated_returns, 2.5), np.percentile(simulated_returns, 97.5)],
                "99%": [np.percentile(simulated_returns, 0.5), np.percentile(simulated_returns, 99.5)]
            },
            "bootstrap_valid": True
        }

    def _run_tick_enhanced_simulation(self, tick_data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced simulation using tick-level data."""
        if tick_data.empty or 'price' not in tick_data.columns:
            return {"tick_enhanced": False}

        # Calculate tick-level statistics
        tick_returns = tick_data['price'].pct_change().dropna()

        if len(tick_returns) < 100:
            return {"tick_enhanced": False}

        # Micro-structure measures
        tick_volatility = tick_returns.std()
        tick_autocorrelation = tick_returns.autocorr(lag=1) if len(tick_returns) > 1 else 0.0

        return {
            "tick_enhanced": True,
            "micro_volatility": tick_volatility,
            "tick_autocorrelation": tick_autocorrelation,
            "tick_observations": len(tick_returns),
            "high_frequency_adjustment": min(1.2, 1.0 + abs(tick_autocorrelation) * 0.5)
        }

    def _run_volume_weighted_simulation(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Volume-weighted simulation enhancements."""
        if 'volume' not in ohlcv_data.columns or ohlcv_data['volume'].isna().all():
            return {"volume_weighted": False}

        # Calculate volume-weighted returns
        returns = ohlcv_data['close'].pct_change().dropna()
        volumes = ohlcv_data['volume'].iloc[1:len(returns)+1]  # Align with returns

        if len(returns) != len(volumes) or len(returns) < 10:
            return {"volume_weighted": False}

        # Volume-weighted average return
        total_volume = volumes.sum()
        if total_volume == 0:
            return {"volume_weighted": False}

        vwap_return = (returns * volumes).sum() / total_volume
        volume_volatility = volumes.std() / volumes.mean() if volumes.mean() > 0 else 0.0

        return {
            "volume_weighted": True,
            "vwap_return": vwap_return,
            "volume_volatility": volume_volatility,
            "volume_weighted_adjustment": min(1.3, 1.0 + volume_volatility * 0.1)
        }
    def _run_scenario_analysis_enhanced(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], data_availability: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced scenario analysis with data-driven scenarios."""
        scenario_analysis = {
            "base_case": 0.0,
            "best_case": 0.0,
            "worst_case": 0.0,
            "scenario_probability_weighted": 0.0,
            "data_driven_scenarios": {},
            "historical_analog_scenarios": []
        }

        # Base scenario analysis
        base_scenarios = self._analyze_scenarios(indicator_results)
        scenario_analysis.update(base_scenarios)

        # Historical analog scenarios
        if data_availability.get("historical_depth", 0) >= 60:
            historical_scenarios = self._generate_historical_analog_scenarios(market_data['OHLCV'])
            scenario_analysis["historical_analog_scenarios"] = historical_scenarios
            scenario_analysis["data_driven_scenarios"]["historical_analogs"] = True

        # Volatility regime scenarios
        if data_availability.get("ohlcv_available", False):
            volatility_scenarios = self._generate_volatility_regime_scenarios(market_data['OHLCV'])
            scenario_analysis["data_driven_scenarios"]["volatility_regimes"] = volatility_scenarios

        return scenario_analysis

    def _run_stress_testing_enhanced(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], data_availability: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced stress testing with data-driven stress scenarios."""
        stress_test_analysis = {
            "market_crash_scenario": 0.0,
            "volatility_spike_scenario": 0.0,
            "liquidity_crisis_scenario": 0.0,
            "pass_rate": 0.0,
            "stress_test_confidence": 0.0,
            "data_driven_stress_tests": {}
        }

        # Base stress testing
        base_stress = self._analyze_risk_simulation(indicator_results)
        stress_test_analysis.update(base_stress)

        # Historical crisis scenarios
        if data_availability.get("historical_depth", 0) >= 252:
            crisis_scenarios = self._generate_historical_crisis_scenarios(market_data['OHLCV'])
            stress_test_analysis["data_driven_stress_tests"]["historical_crises"] = crisis_scenarios

        # Data quality adjustment for stress test confidence
        data_quality = data_availability.get("data_quality_score", 0.0)
        base_confidence = stress_test_analysis.get("stress_test_confidence", 0.5)
        stress_test_analysis["stress_test_confidence"] = min(1.0, base_confidence * (0.3 + data_quality * 0.7))

        return stress_test_analysis

    def _analyze_risk_distributions_enhanced(self, indicator_results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], data_availability: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced risk distribution analysis using empirical data."""
        risk_distribution_analysis = {
            "distribution_type": "NORMAL",
            "skewness": 0.0,
            "kurtosis": 3.0,
            "tail_risk": 0.0,
            "fat_tail_probability": 0.0,
            "empirical_distribution": False,
            "distribution_confidence": 0.5
        }

        # Empirical distribution analysis if sufficient data
        if data_availability.get("historical_depth", 0) >= 100 and data_availability.get("ohlcv_available", False):
            empirical_analysis = self._analyze_empirical_distribution(market_data['OHLCV'])
            risk_distribution_analysis.update(empirical_analysis)
            risk_distribution_analysis["empirical_distribution"] = True
            risk_distribution_analysis["distribution_confidence"] = min(0.9, 0.5 + data_availability["historical_depth"] / 500)

        return risk_distribution_analysis

    def _generate_historical_analog_scenarios(self, ohlcv_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate scenarios based on historical market patterns."""
        if len(ohlcv_data) < 60:
            return []

        returns = ohlcv_data['close'].pct_change().dropna()
        scenarios = []

        # Find periods of high volatility
        volatility = returns.rolling(20).std()
        high_vol_periods = volatility > volatility.quantile(0.8)

        if high_vol_periods.any():
            high_vol_returns = returns[high_vol_periods]
            scenarios.append({
                "scenario_name": "High Volatility Analog",
                "expected_return": high_vol_returns.mean(),
                "volatility": high_vol_returns.std(),
                "probability": 0.2,
                "based_on_periods": high_vol_periods.sum()
            })

        # Find periods of low volatility
        low_vol_periods = volatility < volatility.quantile(0.2)
        if low_vol_periods.any():
            low_vol_returns = returns[low_vol_periods]
            scenarios.append({
                "scenario_name": "Low Volatility Analog",
                "expected_return": low_vol_returns.mean(),
                "volatility": low_vol_returns.std(),
                "probability": 0.2,
                "based_on_periods": low_vol_periods.sum()
            })

        return scenarios

    def _generate_volatility_regime_scenarios(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate scenarios based on volatility regimes."""
        if len(ohlcv_data) < 30:
            return {"volatility_regimes": False}

        returns = ohlcv_data['close'].pct_change().dropna()
        volatility = returns.rolling(10).std()

        # Classify volatility regimes
        high_vol_threshold = volatility.quantile(0.75)
        low_vol_threshold = volatility.quantile(0.25)

        current_volatility = volatility.iloc[-1] if len(volatility) > 0 else 0.0

        if current_volatility > high_vol_threshold:
            current_regime = "HIGH_VOLATILITY"
            regime_probability = 0.3
        elif current_volatility < low_vol_threshold:
            current_regime = "LOW_VOLATILITY"
            regime_probability = 0.3
        else:
            current_regime = "MEDIUM_VOLATILITY"
            regime_probability = 0.4

        return {
            "volatility_regimes": True,
            "current_regime": current_regime,
            "regime_probability": regime_probability,
            "high_vol_threshold": high_vol_threshold,
            "low_vol_threshold": low_vol_threshold,
            "current_volatility": current_volatility
        }

    def _generate_historical_crisis_scenarios(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate stress scenarios based on historical crisis patterns."""
        if len(ohlcv_data) < 252:
            return {"historical_crises": False}

        returns = ohlcv_data['close'].pct_change().dropna()

        # Identify potential crisis periods (large negative returns)
        crisis_threshold = returns.quantile(0.05)  # Bottom 5% of returns
        crisis_returns = returns[returns <= crisis_threshold]

        if len(crisis_returns) == 0:
            return {"historical_crises": False}

        # Calculate crisis statistics
        average_crisis_return = crisis_returns.mean()
        worst_crisis_return = crisis_returns.min()
        crisis_frequency = len(crisis_returns) / len(returns)

        return {
            "historical_crises": True,
            "average_crisis_return": average_crisis_return,
            "worst_crisis_return": worst_crisis_return,
            "crisis_frequency": crisis_frequency,
            "crisis_periods_identified": len(crisis_returns),
            "crisis_severity_score": abs(average_crisis_return) * crisis_frequency * 10
        }

    def _analyze_empirical_distribution(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze empirical distribution of returns."""
        if len(ohlcv_data) < 30:
            return {"distribution_type": "INSUFFICIENT_DATA"}

        returns = ohlcv_data['close'].pct_change().dropna()

        if len(returns) < 30:
            return {"distribution_type": "INSUFFICIENT_DATA"}

        # Calculate distribution moments
        mean_return = returns.mean()
        std_return = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Tail risk analysis
        var_95 = returns.quantile(0.05)  # 5% VaR
        var_99 = returns.quantile(0.01)  # 1% VaR

        # Fat tail detection
        fat_tail_threshold = 3.0  # Normal distribution has kurtosis = 3
        fat_tail_detected = kurtosis > fat_tail_threshold

        # Distribution classification
        if abs(skewness) < 0.5 and abs(kurtosis - 3) < 1:
            distribution_type = "APPROXIMATELY_NORMAL"
        elif fat_tail_detected:
            distribution_type = "FAT_TAILED"
        elif abs(skewness) > 1:
            distribution_type = "SKEWED"
        else:
            distribution_type = "NON_NORMAL"

        return {
            "distribution_type": distribution_type,
            "mean_return": mean_return,
            "std_return": std_return,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "var_95": var_95,
            "var_99": var_99,
            "fat_tail_detected": fat_tail_detected,
            "tail_risk": abs(var_99),
            "sample_size": len(returns)
        }
    def _generate_simulation_decision_enhanced(self, monte_carlo_analysis: Dict[str, Any], scenario_analysis: Dict[str, Any], stress_test_analysis: Dict[str, Any], data_availability: Dict[str, Any]) -> str:
        """Generate enhanced decision based on comprehensive simulation results and data quality."""
        # Data quality factor
        data_quality = data_availability.get("data_quality_score", 0.0)
        simulation_reliability = data_availability.get("simulation_reliability", "LOW")

        # Conservative approach if data quality is poor
        if simulation_reliability == "LOW" or data_quality < 0.3:
            return "NO_SIGNAL"  # Conservative when data is unreliable

        # Enhanced decision logic
        expected_return = monte_carlo_analysis.get("expected_return", 0.0)
        win_probability = monte_carlo_analysis.get("win_probability", 0.5)
        stress_pass_rate = stress_test_analysis.get("pass_rate", 0.0)
        worst_case = scenario_analysis.get("worst_case_loss", 0.0)

        # Data-driven adjustments
        accuracy_estimate = monte_carlo_analysis.get("accuracy_estimate", 0.5)
        convergence_score = monte_carlo_analysis.get("convergence_score", 0.5)

        # Enhanced thresholds based on data quality
        base_return_threshold = 0.01
        base_probability_threshold = 0.6
        base_stress_threshold = 0.7

        # Adjust thresholds based on data quality and simulation accuracy
        return_threshold = base_return_threshold * (2.0 - data_quality)  # Higher threshold for poor data
        probability_threshold = base_probability_threshold + (1.0 - accuracy_estimate) * 0.2
        stress_threshold = base_stress_threshold - (1.0 - convergence_score) * 0.2

        # Decision logic
        if (expected_return > return_threshold and
            win_probability > probability_threshold and
            stress_pass_rate > stress_threshold and
            abs(worst_case) < 0.05):  # Max 5% worst case loss

            if expected_return > 0:
                return "BUY"
            else:
                return "SELL"
        elif simulation_reliability == "HIGH" and convergence_score > 0.8:
            # High confidence simulations can make moderate recommendations
            if expected_return > 0 and win_probability > 0.55:
                return "BUY"
            elif expected_return < 0 and win_probability < 0.45:
                return "SELL"

        return "NO_SIGNAL"

    def _calculate_simulation_confidence_enhanced(self, monte_carlo_analysis: Dict[str, Any], scenario_analysis: Dict[str, Any], stress_test_analysis: Dict[str, Any], data_availability: Dict[str, Any]) -> float:
        """Calculate enhanced confidence based on simulation quality and data availability."""
        confidence_factors = []

        # Base simulation confidence (40% weight)
        base_confidence = self._calculate_simulation_confidence(monte_carlo_analysis, scenario_analysis, stress_test_analysis)
        confidence_factors.append(base_confidence * 0.4)

        # Data quality confidence (30% weight)
        data_quality = data_availability.get("data_quality_score", 0.0)
        confidence_factors.append(data_quality * 0.3)

        # Simulation accuracy confidence (20% weight)
        accuracy_estimate = monte_carlo_analysis.get("accuracy_estimate", 0.5)
        confidence_factors.append(accuracy_estimate * 0.2)

        # Convergence confidence (10% weight)
        convergence_score = monte_carlo_analysis.get("convergence_score", 0.5)
        confidence_factors.append(convergence_score * 0.1)

        total_confidence = sum(confidence_factors)

        # Adjustments based on simulation method
        simulation_method = monte_carlo_analysis.get("simulation_method", "TRADITIONAL")
        if simulation_method == "BOOTSTRAP_ENHANCED":
            total_confidence *= 1.1  # Boost for enhanced methods

        # Penalty for insufficient data
        historical_depth = data_availability.get("historical_depth", 0)
        if historical_depth < 60:
            total_confidence *= 0.8  # Penalty for insufficient history

        return max(0.0, min(total_confidence, 1.0))

    def _generate_simulation_reasoning_enhanced(self, decision: str, monte_carlo_analysis: Dict[str, Any], scenario_analysis: Dict[str, Any], stress_test_analysis: Dict[str, Any], data_availability: Dict[str, Any]) -> str:
        """Generate enhanced reasoning including data quality and simulation method details."""
        reasoning_parts = []

        # Base reasoning
        base_reasoning = self._generate_simulation_reasoning(decision, monte_carlo_analysis, scenario_analysis, stress_test_analysis)
        reasoning_parts.append(base_reasoning)

        # Data quality context
        data_quality = data_availability.get("data_quality_score", 0.0)
        simulation_reliability = data_availability.get("simulation_reliability", "LOW")
        historical_depth = data_availability.get("historical_depth", 0)

        reasoning_parts.append(f"Data quality: {data_quality:.2f} ({simulation_reliability} reliability).")
        reasoning_parts.append(f"Historical depth: {historical_depth} observations.")

        # Simulation method details
        simulation_method = monte_carlo_analysis.get("simulation_method", "TRADITIONAL")
        accuracy_estimate = monte_carlo_analysis.get("accuracy_estimate", 0.5)
        reasoning_parts.append(f"Simulation method: {simulation_method} (accuracy: {accuracy_estimate:.2f}).")

        # Enhanced features used
        enhancements = []
        if data_availability.get("bootstrap_feasible", False):
            enhancements.append("bootstrap sampling")
        if data_availability.get("tick_available", False):
            enhancements.append("tick-level analysis")
        if data_availability.get("volume_available", False):
            enhancements.append("volume weighting")

        if enhancements:
            reasoning_parts.append(f"Enhanced with: {', '.join(enhancements)}.")

        # Convergence and reliability
        convergence_score = monte_carlo_analysis.get("convergence_score", 0.5)
        reasoning_parts.append(f"Simulation convergence: {convergence_score:.2f}.")

        # Risk assessment context
        if "historical_analog_scenarios" in scenario_analysis and scenario_analysis["historical_analog_scenarios"]:
            reasoning_parts.append(f"Based on {len(scenario_analysis['historical_analog_scenarios'])} historical analogs.")

        return " ".join(reasoning_parts)

    def _assess_simulation_risk_enhanced(self, monte_carlo_analysis: Dict[str, Any], scenario_analysis: Dict[str, Any], stress_test_analysis: Dict[str, Any], data_availability: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced risk assessment incorporating data quality factors."""
        # Base risk assessment
        base_risk = self._assess_simulation_risk(scenario_analysis, stress_test_analysis)

        # Enhanced risk assessment
        enhanced_risk = {
            "simulation_risk": base_risk.get("simulation_risk", "MEDIUM"),
            "data_quality_risk": "MEDIUM",
            "model_risk": "MEDIUM",
            "convergence_risk": "MEDIUM",
            "overall_simulation_confidence": "MEDIUM"
        }
        enhanced_risk.update(base_risk)

        # Data quality risk
        data_quality = data_availability.get("data_quality_score", 0.0)
        if data_quality >= 0.8:
            enhanced_risk["data_quality_risk"] = "LOW"
        elif data_quality >= 0.5:
            enhanced_risk["data_quality_risk"] = "MEDIUM"
        else:
            enhanced_risk["data_quality_risk"] = "HIGH"

        # Model risk based on simulation method and accuracy
        accuracy_estimate = monte_carlo_analysis.get("accuracy_estimate", 0.5)
        if accuracy_estimate >= 0.8:
            enhanced_risk["model_risk"] = "LOW"
        elif accuracy_estimate >= 0.6:
            enhanced_risk["model_risk"] = "MEDIUM"
        else:
            enhanced_risk["model_risk"] = "HIGH"

        # Convergence risk
        convergence_score = monte_carlo_analysis.get("convergence_score", 0.5)
        if convergence_score >= 0.8:
            enhanced_risk["convergence_risk"] = "LOW"
        elif convergence_score >= 0.6:
            enhanced_risk["convergence_risk"] = "MEDIUM"
        else:
            enhanced_risk["convergence_risk"] = "HIGH"

        # Overall confidence assessment
        risk_factors = [
            enhanced_risk["data_quality_risk"],
            enhanced_risk["model_risk"],
            enhanced_risk["convergence_risk"]
        ]

        high_risk_count = risk_factors.count("HIGH")
        if high_risk_count >= 2:
            enhanced_risk["overall_simulation_confidence"] = "LOW"
        elif high_risk_count == 1:
            enhanced_risk["overall_simulation_confidence"] = "MEDIUM"
        else:
            enhanced_risk["overall_simulation_confidence"] = "HIGH"

        return enhanced_risk
