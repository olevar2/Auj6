"""
Advanced Skewness Indicator with Distribution Analysis and Risk Assessment

This indicator implements comprehensive skewness analysis including:
- Multiple skewness measures (sample, Fisher, Pearson)
- Time-varying skewness with regime detection
- Higher-order moment analysis (kurtosis, hyperskewness)
- Distribution tail risk assessment
- Asymmetric volatility modeling
- Skewness-based trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class SkewnessResult:
    """Result container for skewness analysis"""
    sample_skewness: float
    fisher_skewness: float
    pearson_skewness: float
    time_varying_skewness: np.ndarray
    rolling_skewness: np.ndarray
    excess_kurtosis: float
    jarque_bera_statistic: float
    jarque_bera_p_value: float
    dagostino_test_statistic: float
    dagostino_test_p_value: float
    tail_risk_measures: Dict[str, float]
    asymmetric_volatility: Dict[str, float]
    skewness_confidence_interval: Tuple[float, float]
    distribution_moments: Dict[str, float]
    regime_probabilities: np.ndarray
    risk_adjusted_returns: np.ndarray


class SkewnessIndicator(StandardIndicatorInterface):
    """
    Advanced Skewness Indicator
    
    Implements comprehensive distribution asymmetry analysis with
    advanced statistical measures and risk assessment capabilities.
    """
    
    def __init__(self, 
                 window: int = 60,
                 rolling_window: int = 20,
                 confidence_level: float = 0.95):
        """
        Initialize Skewness Indicator
        
        Args:
            window: Main analysis window
            rolling_window: Rolling window for time-varying analysis
            confidence_level: Confidence level for statistical tests
        """
        super().__init__()
        self.window = window
        self.rolling_window = rolling_window
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate skewness analysis
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing skewness analysis results
        """
        try:
            if data.empty or len(data) < self.window:
                raise IndicatorCalculationError("Insufficient data for skewness analysis")
            
            # Calculate returns
            returns = np.diff(np.log(data['close'].values))
            
            if len(returns) < 20:
                raise IndicatorCalculationError("Insufficient return data")
            
            # Perform skewness analysis
            skewness_result = self._calculate_skewness_analysis(returns, data)
            
            # Generate trading signal
            signal = self._generate_signal(skewness_result, returns)
            
            return {
                'signal': signal,
                'sample_skewness': skewness_result.sample_skewness,
                'fisher_skewness': skewness_result.fisher_skewness,
                'pearson_skewness': skewness_result.pearson_skewness,
                'time_varying_skewness': skewness_result.time_varying_skewness.tolist(),
                'rolling_skewness': skewness_result.rolling_skewness.tolist(),
                'excess_kurtosis': skewness_result.excess_kurtosis,
                'jarque_bera_statistic': skewness_result.jarque_bera_statistic,
                'jarque_bera_p_value': skewness_result.jarque_bera_p_value,
                'dagostino_test_statistic': skewness_result.dagostino_test_statistic,
                'dagostino_test_p_value': skewness_result.dagostino_test_p_value,
                'tail_risk_measures': skewness_result.tail_risk_measures,
                'asymmetric_volatility': skewness_result.asymmetric_volatility,
                'skewness_confidence_interval': skewness_result.skewness_confidence_interval,
                'distribution_moments': skewness_result.distribution_moments,
                'regime_probabilities': skewness_result.regime_probabilities.tolist(),
                'risk_adjusted_returns': skewness_result.risk_adjusted_returns.tolist(),
                'strength': self._calculate_signal_strength(skewness_result),
                'confidence': self._calculate_confidence(skewness_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating skewness: {str(e)}")
            raise IndicatorCalculationError(f"Skewness calculation failed: {str(e)}")
    
    def _calculate_skewness_analysis(self, returns: np.ndarray, data: pd.DataFrame) -> SkewnessResult:
        """Perform comprehensive skewness analysis"""
        
        # Basic skewness measures
        sample_skewness = self._calculate_sample_skewness(returns)
        fisher_skewness = self._calculate_fisher_skewness(returns)
        pearson_skewness = self._calculate_pearson_skewness(returns)
        
        # Time-varying analysis
        time_varying_skew = self._calculate_time_varying_skewness(returns)
        rolling_skew = self._calculate_rolling_skewness(returns)
        
        # Higher-order moments
        excess_kurtosis = self._calculate_excess_kurtosis(returns)
        
        # Statistical tests
        jb_stat, jb_p = self._jarque_bera_test(returns)
        dag_stat, dag_p = self._dagostino_test(returns)
        
        # Tail risk measures
        tail_risk = self._calculate_tail_risk_measures(returns)
        
        # Asymmetric volatility
        asym_vol = self._calculate_asymmetric_volatility(returns)
        
        # Confidence intervals
        skew_ci = self._calculate_skewness_confidence_interval(sample_skewness, len(returns))
        
        # Distribution moments
        dist_moments = self._calculate_distribution_moments(returns)
        
        # Regime analysis
        regime_probs = self._analyze_skewness_regimes(returns)
        
        # Risk-adjusted returns
        risk_adj_returns = self._calculate_risk_adjusted_returns(returns, data)
        
        return SkewnessResult(
            sample_skewness=sample_skewness,
            fisher_skewness=fisher_skewness,
            pearson_skewness=pearson_skewness,
            time_varying_skewness=time_varying_skew,
            rolling_skewness=rolling_skew,
            excess_kurtosis=excess_kurtosis,
            jarque_bera_statistic=jb_stat,
            jarque_bera_p_value=jb_p,
            dagostino_test_statistic=dag_stat,
            dagostino_test_p_value=dag_p,
            tail_risk_measures=tail_risk,
            asymmetric_volatility=asym_vol,
            skewness_confidence_interval=skew_ci,
            distribution_moments=dist_moments,
            regime_probabilities=regime_probs,
            risk_adjusted_returns=risk_adj_returns
        )
    
    def _calculate_sample_skewness(self, returns: np.ndarray) -> float:
        """Calculate sample skewness"""
        return float(stats.skew(returns, bias=False))
    
    def _calculate_fisher_skewness(self, returns: np.ndarray) -> float:
        """Calculate Fisher skewness (moment-based)"""
        n = len(returns)
        if n < 3:
            return 0.0
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret == 0:
            return 0.0
        
        # Third central moment
        third_moment = np.mean((returns - mean_ret) ** 3)
        
        # Fisher skewness
        skewness = third_moment / (std_ret ** 3)
        
        # Bias correction
        bias_correction = np.sqrt(n * (n - 1)) / (n - 2)
        
        return skewness * bias_correction
    
    def _calculate_pearson_skewness(self, returns: np.ndarray) -> float:
        """Calculate Pearson skewness coefficients"""
        mean_ret = np.mean(returns)
        median_ret = np.median(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret == 0:
            return 0.0
        
        # Pearson's second skewness coefficient
        pearson_skew = 3 * (mean_ret - median_ret) / std_ret
        
        return pearson_skew
    
    def _calculate_time_varying_skewness(self, returns: np.ndarray) -> np.ndarray:
        """Calculate time-varying skewness using exponential weighting"""
        n = len(returns)
        lambda_param = 0.94  # RiskMetrics decay factor
        
        # Exponential weights
        weights = np.array([lambda_param ** i for i in range(n)])[::-1]
        weights = weights / np.sum(weights)
        
        # Calculate weighted moments
        time_varying_skew = np.zeros(n)
        
        for i in range(20, n):  # Start after sufficient observations
            subset_returns = returns[:i+1]
            subset_weights = weights[n-i-1:]
            
            # Weighted mean and variance
            weighted_mean = np.average(subset_returns, weights=subset_weights)
            weighted_var = np.average((subset_returns - weighted_mean) ** 2, weights=subset_weights)
            
            if weighted_var > 0:
                # Weighted third moment
                weighted_third = np.average((subset_returns - weighted_mean) ** 3, weights=subset_weights)
                time_varying_skew[i] = weighted_third / (weighted_var ** 1.5)
        
        # Fill initial values
        time_varying_skew[:20] = time_varying_skew[20] if n > 20 else 0
        
        return time_varying_skew
    
    def _calculate_rolling_skewness(self, returns: np.ndarray) -> np.ndarray:
        """Calculate rolling skewness"""
        n = len(returns)
        rolling_skew = np.zeros(n)
        
        for i in range(self.rolling_window, n):
            window_returns = returns[i-self.rolling_window:i]
            rolling_skew[i] = stats.skew(window_returns, bias=False)
        
        # Fill initial values
        rolling_skew[:self.rolling_window] = rolling_skew[self.rolling_window] if n > self.rolling_window else 0
        
        return rolling_skew
    
    def _calculate_excess_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis"""
        return float(stats.kurtosis(returns, bias=False, fisher=True))
    
    def _jarque_bera_test(self, returns: np.ndarray) -> Tuple[float, float]:
        """Perform Jarque-Bera test for normality"""
        try:
            jb_stat, jb_p = stats.jarque_bera(returns)
            return float(jb_stat), float(jb_p)
        except:
            return 0.0, 1.0
    
    def _dagostino_test(self, returns: np.ndarray) -> Tuple[float, float]:
        """Perform D'Agostino test for skewness"""
        try:
            dag_stat, dag_p = stats.normaltest(returns)
            return float(dag_stat), float(dag_p)
        except:
            return 0.0, 1.0
    
    def _calculate_tail_risk_measures(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate tail risk measures"""
        # Value at Risk (VaR) at different confidence levels
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        es_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else var_99
        
        # Tail ratio
        upper_tail = np.percentile(returns, 95)
        lower_tail = np.percentile(returns, 5)
        tail_ratio = abs(upper_tail) / abs(lower_tail) if lower_tail != 0 else 1.0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'tail_ratio': tail_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_asymmetric_volatility(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate asymmetric volatility measures"""
        # Separate positive and negative returns
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        # Upside and downside volatility
        upside_vol = np.std(positive_returns, ddof=1) if len(positive_returns) > 1 else 0.0
        downside_vol = np.std(negative_returns, ddof=1) if len(negative_returns) > 1 else 0.0
        
        # Volatility asymmetry
        vol_asymmetry = (upside_vol - downside_vol) / (upside_vol + downside_vol) if (upside_vol + downside_vol) > 0 else 0.0
        
        # Leverage effect (correlation between returns and squared returns)
        lagged_returns = returns[:-1]
        squared_returns = returns[1:] ** 2
        leverage_corr = np.corrcoef(lagged_returns, squared_returns)[0, 1] if len(lagged_returns) > 1 else 0.0
        
        return {
            'upside_volatility': upside_vol,
            'downside_volatility': downside_vol,
            'volatility_asymmetry': vol_asymmetry,
            'leverage_effect': leverage_corr
        }
    
    def _calculate_skewness_confidence_interval(self, skewness: float, n: int) -> Tuple[float, float]:
        """Calculate confidence interval for skewness"""
        if n < 8:
            return (skewness - 1, skewness + 1)
        
        # Standard error of skewness
        se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
        
        # Critical value
        alpha = 1 - self.confidence_level
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval
        lower = skewness - z_crit * se_skew
        upper = skewness + z_crit * se_skew
        
        return (lower, upper)
    
    def _calculate_distribution_moments(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive distribution moments"""
        mean_ret = np.mean(returns)
        var_ret = np.var(returns, ddof=1)
        std_ret = np.sqrt(var_ret)
        
        # Central moments
        third_moment = np.mean((returns - mean_ret) ** 3)
        fourth_moment = np.mean((returns - mean_ret) ** 4)
        
        # Standardized moments
        skewness = third_moment / (std_ret ** 3) if std_ret > 0 else 0
        kurtosis = fourth_moment / (var_ret ** 2) if var_ret > 0 else 3
        
        # Additional measures
        coefficient_of_variation = std_ret / abs(mean_ret) if mean_ret != 0 else np.inf
        
        return {
            'mean': mean_ret,
            'variance': var_ret,
            'standard_deviation': std_ret,
            'third_central_moment': third_moment,
            'fourth_central_moment': fourth_moment,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'excess_kurtosis': kurtosis - 3,
            'coefficient_of_variation': coefficient_of_variation
        }
    
    def _analyze_skewness_regimes(self, returns: np.ndarray) -> np.ndarray:
        """Analyze skewness regimes using Hidden Markov Model approach"""
        # Simplified regime detection based on rolling skewness
        rolling_skew = self._calculate_rolling_skewness(returns)
        
        # Define regimes: negative skew, neutral, positive skew
        negative_threshold = -0.5
        positive_threshold = 0.5
        
        regime_probs = np.zeros((len(returns), 3))  # [negative, neutral, positive]
        
        for i, skew in enumerate(rolling_skew):
            if skew < negative_threshold:
                regime_probs[i] = [0.8, 0.15, 0.05]  # Negative skew regime
            elif skew > positive_threshold:
                regime_probs[i] = [0.05, 0.15, 0.8]  # Positive skew regime
            else:
                regime_probs[i] = [0.2, 0.6, 0.2]   # Neutral regime
        
        return regime_probs
    
    def _calculate_risk_adjusted_returns(self, returns: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        """Calculate risk-adjusted returns considering skewness"""
        # Sortino ratio adjustment for asymmetric risk
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else np.std(returns, ddof=1)
        
        if downside_std == 0:
            return returns
        
        # Skewness-adjusted risk measure
        skewness = self._calculate_sample_skewness(returns)
        skew_adjustment = 1 + skewness / 6  # Simple adjustment factor
        
        adjusted_risk = downside_std * skew_adjustment
        risk_adjusted = returns / adjusted_risk
        
        return risk_adjusted
    
    def _generate_signal(self, result: SkewnessResult, returns: np.ndarray) -> SignalType:
        """Generate trading signal based on skewness analysis"""
        current_skew = result.sample_skewness
        rolling_skew = result.rolling_skewness
        tail_ratio = result.tail_risk_measures['tail_ratio']
        vol_asymmetry = result.asymmetric_volatility['volatility_asymmetry']
        
        # Recent skewness trend
        if len(rolling_skew) >= 10:
            recent_skew_change = rolling_skew[-1] - np.mean(rolling_skew[-10:-1])
        else:
            recent_skew_change = 0
        
        # Signal generation logic
        
        # Strong negative skewness suggests potential upside
        if current_skew < -1.0 and recent_skew_change > 0.2:
            # Tail risk is manageable
            if tail_ratio < 2.0:
                return SignalType.BUY
        
        # Strong positive skewness suggests potential downside
        elif current_skew > 1.0 and recent_skew_change < -0.2:
            # High tail risk
            if tail_ratio > 1.5:
                return SignalType.SELL
        
        # Regime-based signals
        current_regime = np.argmax(result.regime_probabilities[-1])
        
        # Negative skew regime (regime 0) with improving conditions
        if current_regime == 0 and recent_skew_change > 0:
            return SignalType.BUY
            
        # Positive skew regime (regime 2) with deteriorating conditions  
        elif current_regime == 2 and recent_skew_change < 0:
            return SignalType.SELL
        
        # Volatility asymmetry considerations
        if abs(vol_asymmetry) > 0.3:
            # High upside volatility relative to downside
            if vol_asymmetry > 0.3 and current_skew < 0:
                return SignalType.BUY
            # High downside volatility relative to upside
            elif vol_asymmetry < -0.3 and current_skew > 0:
                return SignalType.SELL
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: SkewnessResult) -> float:
        """Calculate signal strength based on skewness characteristics"""
        # Absolute skewness magnitude
        skew_strength = min(abs(result.sample_skewness) / 2.0, 1.0)
        
        # Statistical significance
        jb_significance = max(0, 1 - result.jarque_bera_p_value)
        
        # Tail risk consistency
        tail_consistency = min(result.tail_risk_measures['tail_ratio'] / 3.0, 1.0)
        
        # Rolling skewness stability
        if len(result.rolling_skewness) >= 10:
            skew_stability = 1 / (1 + np.std(result.rolling_skewness[-10:]))
        else:
            skew_stability = 0.5
        
        return (skew_strength + jb_significance + tail_consistency + skew_stability) / 4
    
    def _calculate_confidence(self, result: SkewnessResult) -> float:
        """Calculate confidence based on statistical tests and consistency"""
        # Confidence interval tightness
        ci_width = result.skewness_confidence_interval[1] - result.skewness_confidence_interval[0]
        ci_confidence = max(0, 1 - ci_width / 4)  # Normalize by reasonable range
        
        # Statistical test confidence
        jb_confidence = max(0, 1 - result.jarque_bera_p_value) if result.jarque_bera_p_value < 0.05 else 0.5
        dag_confidence = max(0, 1 - result.dagostino_test_p_value) if result.dagostino_test_p_value < 0.05 else 0.5
        
        # Regime consistency
        regime_entropy = -np.sum(result.regime_probabilities[-1] * np.log(result.regime_probabilities[-1] + 1e-10))
        max_entropy = np.log(3)  # Maximum entropy for 3 regimes
        regime_confidence = 1 - (regime_entropy / max_entropy)
        
        return (ci_confidence + jb_confidence + dag_confidence + regime_confidence) / 4