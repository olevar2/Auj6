"""
Advanced Variance Ratio Indicator with Market Efficiency Testing

This indicator implements comprehensive variance ratio analysis including:
- Lo-MacKinlay variance ratio test
- Multiple holding period analysis
- Market efficiency assessment
- Random walk hypothesis testing
- Autocorrelation and mean reversion detection
- Adaptive variance ratio with regime detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class VarianceRatioResult:
    """Result container for variance ratio analysis"""
    variance_ratios: Dict[int, float]
    lo_mackinlay_statistics: Dict[int, float]
    lo_mackinlay_p_values: Dict[int, float]
    joint_test_statistic: float
    joint_test_p_value: float
    autocorrelations: np.ndarray
    mean_reversion_strength: float
    momentum_strength: float
    market_efficiency_score: float
    random_walk_evidence: float
    regime_specific_vr: Dict[str, Dict[int, float]]
    adaptive_variance_ratios: np.ndarray
    heteroscedasticity_robust_stats: Dict[int, float]
    bootstrap_confidence_intervals: Dict[int, Tuple[float, float]]


class VarianceRatioIndicator(StandardIndicatorInterface):
    """
    Advanced Variance Ratio Indicator
    
    Implements comprehensive market efficiency testing using variance ratios
    with multiple statistical tests and regime-aware analysis.
    """
    
    def __init__(self, 
                 holding_periods: List[int] = None,
                 min_observations: int = 100,
                 bootstrap_samples: int = 1000):
        """
        Initialize Variance Ratio Indicator
        
        Args:
            holding_periods: List of holding periods for variance ratio calculation
            min_observations: Minimum number of observations required
            bootstrap_samples: Number of bootstrap samples for confidence intervals
        """
        super().__init__()
        self.holding_periods = holding_periods or [2, 4, 8, 16]
        self.min_observations = min_observations
        self.bootstrap_samples = bootstrap_samples
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate variance ratio analysis
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing variance ratio analysis results
        """
        try:
            if data.empty or len(data) < self.min_observations:
                raise IndicatorCalculationError("Insufficient data for variance ratio analysis")
            
            # Calculate log returns
            prices = data['close'].values
            log_returns = np.diff(np.log(prices))
            
            if len(log_returns) < max(self.holding_periods) * 2:
                raise IndicatorCalculationError("Insufficient returns for analysis")
            
            # Perform variance ratio analysis
            vr_result = self._calculate_variance_ratio_analysis(log_returns)
            
            # Generate trading signal
            signal = self._generate_signal(vr_result, log_returns)
            
            return {
                'signal': signal,
                'variance_ratios': vr_result.variance_ratios,
                'lo_mackinlay_statistics': vr_result.lo_mackinlay_statistics,
                'lo_mackinlay_p_values': vr_result.lo_mackinlay_p_values,
                'joint_test_statistic': vr_result.joint_test_statistic,
                'joint_test_p_value': vr_result.joint_test_p_value,
                'autocorrelations': vr_result.autocorrelations.tolist(),
                'mean_reversion_strength': vr_result.mean_reversion_strength,
                'momentum_strength': vr_result.momentum_strength,
                'market_efficiency_score': vr_result.market_efficiency_score,
                'random_walk_evidence': vr_result.random_walk_evidence,
                'regime_specific_vr': vr_result.regime_specific_vr,
                'adaptive_variance_ratios': vr_result.adaptive_variance_ratios.tolist(),
                'heteroscedasticity_robust_stats': vr_result.heteroscedasticity_robust_stats,
                'bootstrap_confidence_intervals': vr_result.bootstrap_confidence_intervals,
                'strength': self._calculate_signal_strength(vr_result),
                'confidence': self._calculate_confidence(vr_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating variance ratio: {str(e)}")
            raise IndicatorCalculationError(f"Variance ratio calculation failed: {str(e)}")
    
    def _calculate_variance_ratio_analysis(self, returns: np.ndarray) -> VarianceRatioResult:
        """Perform comprehensive variance ratio analysis"""
        
        # Basic variance ratios
        variance_ratios = self._calculate_variance_ratios(returns)
        
        # Lo-MacKinlay test statistics
        lo_mac_stats, lo_mac_p_values = self._calculate_lo_mackinlay_test(returns, variance_ratios)
        
        # Joint test
        joint_stat, joint_p = self._calculate_joint_test(lo_mac_stats)
        
        # Autocorrelations
        autocorrs = self._calculate_autocorrelations(returns)
        
        # Mean reversion and momentum strength
        mean_rev_strength = self._calculate_mean_reversion_strength(variance_ratios, autocorrs)
        momentum_strength = self._calculate_momentum_strength(variance_ratios, autocorrs)
        
        # Market efficiency score
        efficiency_score = self._calculate_market_efficiency_score(variance_ratios, lo_mac_p_values)
        
        # Random walk evidence
        rw_evidence = self._calculate_random_walk_evidence(variance_ratios, lo_mac_p_values)
        
        # Regime-specific analysis
        regime_vr = self._calculate_regime_specific_variance_ratios(returns)
        
        # Adaptive variance ratios
        adaptive_vr = self._calculate_adaptive_variance_ratios(returns)
        
        # Heteroscedasticity-robust statistics
        robust_stats = self._calculate_robust_statistics(returns, variance_ratios)
        
        # Bootstrap confidence intervals
        bootstrap_ci = self._calculate_bootstrap_confidence_intervals(returns)
        
        return VarianceRatioResult(
            variance_ratios=variance_ratios,
            lo_mackinlay_statistics=lo_mac_stats,
            lo_mackinlay_p_values=lo_mac_p_values,
            joint_test_statistic=joint_stat,
            joint_test_p_value=joint_p,
            autocorrelations=autocorrs,
            mean_reversion_strength=mean_rev_strength,
            momentum_strength=momentum_strength,
            market_efficiency_score=efficiency_score,
            random_walk_evidence=rw_evidence,
            regime_specific_vr=regime_vr,
            adaptive_variance_ratios=adaptive_vr,
            heteroscedasticity_robust_stats=robust_stats,
            bootstrap_confidence_intervals=bootstrap_ci
        )
    
    def _calculate_variance_ratios(self, returns: np.ndarray) -> Dict[int, float]:
        """Calculate variance ratios for different holding periods"""
        variance_ratios = {}
        
        # Base period variance (1-period)
        var_1 = np.var(returns, ddof=1)
        
        for k in self.holding_periods:
            if len(returns) < k * 10:  # Need sufficient data
                variance_ratios[k] = 1.0
                continue
            
            # k-period overlapping returns
            k_period_returns = []
            for i in range(len(returns) - k + 1):
                k_period_return = np.sum(returns[i:i+k])
                k_period_returns.append(k_period_return)
            
            if len(k_period_returns) > 1:
                var_k = np.var(k_period_returns, ddof=1)
                # Variance ratio: Var(k-period) / (k * Var(1-period))
                variance_ratios[k] = var_k / (k * var_1) if var_1 > 0 else 1.0
            else:
                variance_ratios[k] = 1.0
        
        return variance_ratios
    
    def _calculate_lo_mackinlay_test(self, returns: np.ndarray, variance_ratios: Dict[int, float]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Calculate Lo-MacKinlay test statistics and p-values"""
        n = len(returns)
        statistics = {}
        p_values = {}
        
        for k, vr in variance_ratios.items():
            # Lo-MacKinlay statistic
            # Asymptotic variance under random walk hypothesis
            theta = 2 * (2*k - 1) * (k - 1) / (3 * k * n)
            
            # Test statistic
            z_stat = (vr - 1) / np.sqrt(theta) if theta > 0 else 0
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            statistics[k] = z_stat
            p_values[k] = p_value
        
        return statistics, p_values
    
    def _calculate_joint_test(self, lo_mac_stats: Dict[int, float]) -> Tuple[float, float]:
        """Calculate joint test statistic for all holding periods"""
        if not lo_mac_stats:
            return 0.0, 1.0
        
        # Joint test statistic (sum of squared individual statistics)
        joint_stat = sum(z**2 for z in lo_mac_stats.values())
        
        # Degrees of freedom
        df = len(lo_mac_stats)
        
        # P-value from chi-square distribution
        p_value = 1 - stats.chi2.cdf(joint_stat, df)
        
        return joint_stat, p_value
    
    def _calculate_autocorrelations(self, returns: np.ndarray) -> np.ndarray:
        """Calculate autocorrelations for multiple lags"""
        max_lag = min(20, len(returns) // 4)
        autocorrs = np.zeros(max_lag)
        
        for lag in range(1, max_lag + 1):
            if len(returns) > lag:
                corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                autocorrs[lag-1] = corr if not np.isnan(corr) else 0
        
        return autocorrs
    
    def _calculate_mean_reversion_strength(self, variance_ratios: Dict[int, float], autocorrs: np.ndarray) -> float:
        """Calculate mean reversion strength"""
        # Mean reversion indicated by VR < 1 and negative autocorrelations
        vr_evidence = np.mean([1 - vr for vr in variance_ratios.values() if vr < 1])
        
        # Negative autocorrelation evidence
        negative_autocorrs = autocorrs[autocorrs < 0]
        autocorr_evidence = np.mean(np.abs(negative_autocorrs)) if len(negative_autocorrs) > 0 else 0
        
        return (vr_evidence + autocorr_evidence) / 2
    
    def _calculate_momentum_strength(self, variance_ratios: Dict[int, float], autocorrs: np.ndarray) -> float:
        """Calculate momentum strength"""
        # Momentum indicated by VR > 1 and positive autocorrelations
        vr_evidence = np.mean([vr - 1 for vr in variance_ratios.values() if vr > 1])
        
        # Positive autocorrelation evidence
        positive_autocorrs = autocorrs[autocorrs > 0]
        autocorr_evidence = np.mean(positive_autocorrs) if len(positive_autocorrs) > 0 else 0
        
        return (vr_evidence + autocorr_evidence) / 2
    
    def _calculate_market_efficiency_score(self, variance_ratios: Dict[int, float], p_values: Dict[int, float]) -> float:
        """Calculate market efficiency score"""
        # Efficiency indicated by VR close to 1 and high p-values
        vr_efficiency = 1 - np.mean([abs(vr - 1) for vr in variance_ratios.values()])
        
        # Statistical insignificance (high p-values) supports efficiency
        p_value_efficiency = np.mean(list(p_values.values()))
        
        efficiency_score = (vr_efficiency + p_value_efficiency) / 2
        return max(0, min(efficiency_score, 1))
    
    def _calculate_random_walk_evidence(self, variance_ratios: Dict[int, float], p_values: Dict[int, float]) -> float:
        """Calculate evidence for random walk hypothesis"""
        # Random walk: all VR should be close to 1
        vr_deviations = [abs(vr - 1) for vr in variance_ratios.values()]
        vr_score = 1 - np.mean(vr_deviations)
        
        # High p-values support random walk
        p_value_score = np.mean(list(p_values.values()))
        
        # Combine evidence
        random_walk_evidence = (vr_score + p_value_score) / 2
        return max(0, min(random_walk_evidence, 1))
    
    def _calculate_regime_specific_variance_ratios(self, returns: np.ndarray) -> Dict[str, Dict[int, float]]:
        """Calculate variance ratios for different market regimes"""
        if len(returns) < 60:
            return {'low_vol': {}, 'normal_vol': {}, 'high_vol': {}}
        
        # Classify regimes based on rolling volatility
        window = 20
        rolling_vol = np.zeros(len(returns))
        
        for i in range(window, len(returns)):
            rolling_vol[i] = np.std(returns[i-window:i], ddof=1)
        
        # Percentile-based regime classification
        vol_33 = np.percentile(rolling_vol[window:], 33)
        vol_67 = np.percentile(rolling_vol[window:], 67)
        
        # Separate returns by regime
        low_vol_mask = rolling_vol <= vol_33
        normal_vol_mask = (rolling_vol > vol_33) & (rolling_vol <= vol_67)
        high_vol_mask = rolling_vol > vol_67
        
        regime_returns = {
            'low_vol': returns[low_vol_mask],
            'normal_vol': returns[normal_vol_mask],
            'high_vol': returns[high_vol_mask]
        }
        
        regime_vr = {}
        for regime, regime_ret in regime_returns.items():
            if len(regime_ret) >= max(self.holding_periods) * 2:
                regime_vr[regime] = self._calculate_variance_ratios(regime_ret)
            else:
                regime_vr[regime] = {k: 1.0 for k in self.holding_periods}
        
        return regime_vr
    
    def _calculate_adaptive_variance_ratios(self, returns: np.ndarray) -> np.ndarray:
        """Calculate time-varying variance ratios"""
        window = 60
        n = len(returns)
        adaptive_vr = np.zeros(n)
        
        for i in range(window, n):
            window_returns = returns[i-window:i]
            
            # Calculate VR for holding period 4 (most common)
            if len(window_returns) >= 20:
                vr_dict = self._calculate_variance_ratios(window_returns)
                adaptive_vr[i] = vr_dict.get(4, 1.0)
            else:
                adaptive_vr[i] = 1.0
        
        # Fill initial values
        adaptive_vr[:window] = 1.0
        
        return adaptive_vr
    
    def _calculate_robust_statistics(self, returns: np.ndarray, variance_ratios: Dict[int, float]) -> Dict[int, float]:
        """Calculate heteroscedasticity-robust test statistics"""
        n = len(returns)
        robust_stats = {}
        
        for k, vr in variance_ratios.items():
            # Robust variance adjustment for heteroscedasticity
            # Simplified robust standard error calculation
            
            # Calculate k-period returns
            k_returns = []
            for i in range(len(returns) - k + 1):
                k_returns.append(np.sum(returns[i:i+k]))
            
            if len(k_returns) > 1:
                # Heteroscedasticity-consistent variance estimator
                k_returns = np.array(k_returns)
                mean_k = np.mean(k_returns)
                
                # Fourth moment for robust variance
                fourth_moment = np.mean((k_returns - mean_k) ** 4)
                second_moment = np.var(k_returns, ddof=1)
                
                # Robust adjustment factor
                robust_factor = fourth_moment / (second_moment ** 2) if second_moment > 0 else 1
                
                # Adjusted test statistic
                theta_robust = 2 * (2*k - 1) * (k - 1) * robust_factor / (3 * k * n)
                robust_stat = (vr - 1) / np.sqrt(theta_robust) if theta_robust > 0 else 0
                
                robust_stats[k] = robust_stat
            else:
                robust_stats[k] = 0.0
        
        return robust_stats
    
    def _calculate_bootstrap_confidence_intervals(self, returns: np.ndarray) -> Dict[int, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for variance ratios"""
        if len(returns) < 50:
            return {k: (0.8, 1.2) for k in self.holding_periods}
        
        bootstrap_vrs = {k: [] for k in self.holding_periods}
        
        # Bootstrap sampling
        n_samples = min(self.bootstrap_samples, 500)  # Limit for performance
        
        for _ in range(n_samples):
            # Bootstrap sample
            boot_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate VR for bootstrap sample
            try:
                boot_vr = self._calculate_variance_ratios(boot_returns)
                for k in self.holding_periods:
                    if k in boot_vr:
                        bootstrap_vrs[k].append(boot_vr[k])
            except:
                continue
        
        # Calculate confidence intervals
        confidence_intervals = {}
        alpha = 0.05  # 95% confidence interval
        
        for k in self.holding_periods:
            if bootstrap_vrs[k]:
                lower = np.percentile(bootstrap_vrs[k], (alpha/2) * 100)
                upper = np.percentile(bootstrap_vrs[k], (1 - alpha/2) * 100)
                confidence_intervals[k] = (lower, upper)
            else:
                confidence_intervals[k] = (0.8, 1.2)
        
        return confidence_intervals
    
    def _generate_signal(self, result: VarianceRatioResult, returns: np.ndarray) -> SignalType:
        """Generate trading signal based on variance ratio analysis"""
        # Current market state
        mean_reversion = result.mean_reversion_strength
        momentum = result.momentum_strength
        efficiency = result.market_efficiency_score
        
        # Recent adaptive variance ratio
        current_vr = result.adaptive_variance_ratios[-1] if len(result.adaptive_variance_ratios) > 0 else 1.0
        
        # Autocorrelation signals
        recent_autocorr = result.autocorrelations[0] if len(result.autocorrelations) > 0 else 0
        
        # Strong mean reversion signals
        if mean_reversion > 0.6 and current_vr < 0.8:
            # Recent positive returns in mean-reverting market
            if len(returns) >= 5:
                recent_returns = returns[-5:]
                if np.mean(recent_returns) > 0:
                    return SignalType.SELL  # Expect reversion
                else:
                    return SignalType.BUY   # Expect reversion
        
        # Strong momentum signals
        elif momentum > 0.6 and current_vr > 1.2:
            # Continue momentum direction
            if len(returns) >= 3:
                recent_trend = np.mean(returns[-3:])
                if recent_trend > 0:
                    return SignalType.BUY   # Continue upward momentum
                else:
                    return SignalType.SELL  # Continue downward momentum
        
        # Market efficiency changes
        if efficiency < 0.3:  # Market becoming inefficient
            # Look for regime-specific patterns
            regime_vr = result.regime_specific_vr
            
            # High volatility regime with strong patterns
            if 'high_vol' in regime_vr:
                high_vol_vr = regime_vr['high_vol']
                if high_vol_vr and np.mean(list(high_vol_vr.values())) < 0.8:
                    # Mean reversion in high volatility
                    recent_return = returns[-1] if len(returns) > 0 else 0
                    if recent_return > 0.02:  # Large positive return
                        return SignalType.SELL
                    elif recent_return < -0.02:  # Large negative return
                        return SignalType.BUY
        
        # Autocorrelation-based signals
        if abs(recent_autocorr) > 0.3:  # Significant autocorrelation
            if recent_autocorr > 0:  # Positive autocorrelation (momentum)
                recent_return = returns[-1] if len(returns) > 0 else 0
                if recent_return > 0:
                    return SignalType.BUY
                else:
                    return SignalType.SELL
            else:  # Negative autocorrelation (mean reversion)
                recent_return = returns[-1] if len(returns) > 0 else 0
                if recent_return > 0:
                    return SignalType.SELL
                else:
                    return SignalType.BUY
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: VarianceRatioResult) -> float:
        """Calculate signal strength based on variance ratio evidence"""
        # Pattern strength
        pattern_strength = max(result.mean_reversion_strength, result.momentum_strength)
        
        # Statistical significance
        significant_tests = sum(1 for p in result.lo_mackinlay_p_values.values() if p < 0.05)
        significance_strength = significant_tests / len(result.lo_mackinlay_p_values) if result.lo_mackinlay_p_values else 0
        
        # Joint test strength
        joint_strength = max(0, 1 - result.joint_test_p_value) if result.joint_test_p_value <= 1 else 0
        
        # Autocorrelation strength
        autocorr_strength = np.max(np.abs(result.autocorrelations)) if len(result.autocorrelations) > 0 else 0
        
        return (pattern_strength + significance_strength + joint_strength + autocorr_strength) / 4
    
    def _calculate_confidence(self, result: VarianceRatioResult) -> float:
        """Calculate confidence based on test reliability and consistency"""
        # Statistical test confidence
        test_confidence = 1 - result.joint_test_p_value if result.joint_test_p_value <= 1 else 0
        
        # Bootstrap confidence interval tightness
        ci_tightness = 0
        if result.bootstrap_confidence_intervals:
            widths = [(upper - lower) for lower, upper in result.bootstrap_confidence_intervals.values()]
            avg_width = np.mean(widths)
            ci_tightness = max(0, 1 - avg_width / 2)  # Normalize by reasonable width
        
        # Regime consistency
        regime_consistency = 0
        if result.regime_specific_vr:
            # Check if patterns are consistent across regimes
            regime_patterns = []
            for regime_vr in result.regime_specific_vr.values():
                if regime_vr:
                    avg_vr = np.mean(list(regime_vr.values()))
                    regime_patterns.append(avg_vr)
            
            if len(regime_patterns) > 1:
                pattern_std = np.std(regime_patterns)
                regime_consistency = 1 / (1 + pattern_std)
            else:
                regime_consistency = 0.5
        
        # Random walk rejection confidence
        rw_rejection = 1 - result.random_walk_evidence
        
        return (test_confidence + ci_tightness + regime_consistency + rw_rejection) / 4