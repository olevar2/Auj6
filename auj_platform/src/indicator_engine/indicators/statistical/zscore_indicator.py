"""
Advanced Z-Score Indicator with Multi-Scale Statistical Analysis

This indicator implements comprehensive Z-score analysis including:
- Multi-timeframe Z-score calculation
- Rolling and exponential Z-scores
- Distribution-adjusted Z-scores
- Outlier detection and classification
- Regime-aware normalization
- Statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from sklearn.preprocessing import RobustScaler
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class ZScoreResult:
    """Result container for Z-score analysis"""
    current_z_score: float
    rolling_z_scores: np.ndarray
    exponential_z_scores: np.ndarray
    robust_z_scores: np.ndarray
    percentile_z_scores: np.ndarray
    outlier_probabilities: np.ndarray
    extreme_event_indicators: np.ndarray
    regime_adjusted_z_scores: np.ndarray
    statistical_significance: np.ndarray
    tail_probabilities: np.ndarray
    distribution_parameters: Dict[str, float]
    normality_tests: Dict[str, float]
    outlier_classification: np.ndarray
    adaptive_thresholds: Dict[str, float]
    confidence_intervals: np.ndarray


class ZScoreIndicator(StandardIndicatorInterface):
    """
    Advanced Z-Score Indicator
    
    Implements comprehensive statistical analysis with multiple Z-score
    variants and advanced outlier detection capabilities.
    """
    
    def __init__(self, 
                 window: int = 60,
                 short_window: int = 20,
                 lambda_param: float = 0.94,
                 outlier_threshold: float = 2.0):
        """
        Initialize Z-Score Indicator
        
        Args:
            window: Main window for Z-score calculation
            short_window: Short-term window for comparison
            lambda_param: Decay parameter for exponential weighting
            outlier_threshold: Threshold for outlier classification
        """
        super().__init__()
        self.window = window
        self.short_window = short_window
        self.lambda_param = lambda_param
        self.outlier_threshold = outlier_threshold
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate Z-score analysis
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing Z-score analysis results
        """
        try:
            if data.empty or len(data) < self.window:
                raise IndicatorCalculationError("Insufficient data for Z-score analysis")
            
            # Extract price data and calculate returns
            prices = data['close'].values
            returns = np.diff(np.log(prices))
            
            if len(returns) < 20:
                raise IndicatorCalculationError("Insufficient return data")
            
            # Perform Z-score analysis
            zscore_result = self._calculate_zscore_analysis(returns, prices)
            
            # Generate trading signal
            signal = self._generate_signal(zscore_result, returns, prices)
            
            return {
                'signal': signal,
                'current_z_score': zscore_result.current_z_score,
                'rolling_z_scores': zscore_result.rolling_z_scores.tolist(),
                'exponential_z_scores': zscore_result.exponential_z_scores.tolist(),
                'robust_z_scores': zscore_result.robust_z_scores.tolist(),
                'percentile_z_scores': zscore_result.percentile_z_scores.tolist(),
                'outlier_probabilities': zscore_result.outlier_probabilities.tolist(),
                'extreme_event_indicators': zscore_result.extreme_event_indicators.tolist(),
                'regime_adjusted_z_scores': zscore_result.regime_adjusted_z_scores.tolist(),
                'statistical_significance': zscore_result.statistical_significance.tolist(),
                'tail_probabilities': zscore_result.tail_probabilities.tolist(),
                'distribution_parameters': zscore_result.distribution_parameters,
                'normality_tests': zscore_result.normality_tests,
                'outlier_classification': zscore_result.outlier_classification.tolist(),
                'adaptive_thresholds': zscore_result.adaptive_thresholds,
                'confidence_intervals': zscore_result.confidence_intervals.tolist(),
                'strength': self._calculate_signal_strength(zscore_result),
                'confidence': self._calculate_confidence(zscore_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Z-score: {str(e)}")
            raise IndicatorCalculationError(f"Z-score calculation failed: {str(e)}")
    
    def _calculate_zscore_analysis(self, returns: np.ndarray, prices: np.ndarray) -> ZScoreResult:
        """Perform comprehensive Z-score analysis"""
        
        # Basic rolling Z-scores
        rolling_z_scores = self._calculate_rolling_z_scores(returns)
        
        # Exponential Z-scores
        exponential_z_scores = self._calculate_exponential_z_scores(returns)
        
        # Robust Z-scores (using median and MAD)
        robust_z_scores = self._calculate_robust_z_scores(returns)
        
        # Percentile-based Z-scores
        percentile_z_scores = self._calculate_percentile_z_scores(returns)
        
        # Current Z-score
        current_z_score = rolling_z_scores[-1] if len(rolling_z_scores) > 0 else 0.0
        
        # Outlier probabilities
        outlier_probs = self._calculate_outlier_probabilities(rolling_z_scores)
        
        # Extreme event indicators
        extreme_events = self._detect_extreme_events(rolling_z_scores, returns)
        
        # Regime-adjusted Z-scores
        regime_z_scores = self._calculate_regime_adjusted_z_scores(returns, rolling_z_scores)
        
        # Statistical significance
        stat_significance = self._calculate_statistical_significance(rolling_z_scores)
        
        # Tail probabilities
        tail_probs = self._calculate_tail_probabilities(rolling_z_scores)
        
        # Distribution parameters
        dist_params = self._estimate_distribution_parameters(returns)
        
        # Normality tests
        normality_tests = self._perform_normality_tests(returns)
        
        # Outlier classification
        outlier_class = self._classify_outliers(rolling_z_scores, returns)
        
        # Adaptive thresholds
        adaptive_thresh = self._calculate_adaptive_thresholds(rolling_z_scores)
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(rolling_z_scores)
        
        return ZScoreResult(
            current_z_score=current_z_score,
            rolling_z_scores=rolling_z_scores,
            exponential_z_scores=exponential_z_scores,
            robust_z_scores=robust_z_scores,
            percentile_z_scores=percentile_z_scores,
            outlier_probabilities=outlier_probs,
            extreme_event_indicators=extreme_events,
            regime_adjusted_z_scores=regime_z_scores,
            statistical_significance=stat_significance,
            tail_probabilities=tail_probs,
            distribution_parameters=dist_params,
            normality_tests=normality_tests,
            outlier_classification=outlier_class,
            adaptive_thresholds=adaptive_thresh,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_rolling_z_scores(self, returns: np.ndarray) -> np.ndarray:
        """Calculate rolling Z-scores"""
        n = len(returns)
        z_scores = np.zeros(n)
        
        for i in range(self.window, n):
            window_returns = returns[i-self.window:i]
            current_return = returns[i]
            
            mean_ret = np.mean(window_returns)
            std_ret = np.std(window_returns, ddof=1)
            
            if std_ret > 0:
                z_scores[i] = (current_return - mean_ret) / std_ret
            else:
                z_scores[i] = 0.0
        
        # Fill initial values
        if self.window < n:
            initial_mean = np.mean(returns[:self.window])
            initial_std = np.std(returns[:self.window], ddof=1)
            
            for i in range(self.window):
                if initial_std > 0:
                    z_scores[i] = (returns[i] - initial_mean) / initial_std
        
        return z_scores
    
    def _calculate_exponential_z_scores(self, returns: np.ndarray) -> np.ndarray:
        """Calculate exponentially weighted Z-scores"""
        n = len(returns)
        z_scores = np.zeros(n)
        
        # Initialize
        exp_mean = returns[0] if n > 0 else 0
        exp_var = 0.01  # Small initial variance
        
        for i in range(n):
            # Update exponential mean and variance
            exp_mean = self.lambda_param * exp_mean + (1 - self.lambda_param) * returns[i]
            
            if i > 0:
                exp_var = self.lambda_param * exp_var + (1 - self.lambda_param) * (returns[i] - exp_mean) ** 2
            
            # Calculate Z-score
            exp_std = np.sqrt(exp_var)
            if exp_std > 0:
                z_scores[i] = (returns[i] - exp_mean) / exp_std
        
        return z_scores
    
    def _calculate_robust_z_scores(self, returns: np.ndarray) -> np.ndarray:
        """Calculate robust Z-scores using median and MAD"""
        n = len(returns)
        z_scores = np.zeros(n)
        
        for i in range(self.window, n):
            window_returns = returns[i-self.window:i]
            current_return = returns[i]
            
            # Median and MAD (Median Absolute Deviation)
            median_ret = np.median(window_returns)
            mad = np.median(np.abs(window_returns - median_ret))
            
            # Robust Z-score
            if mad > 0:
                z_scores[i] = (current_return - median_ret) / (1.4826 * mad)  # 1.4826 for normal consistency
            else:
                z_scores[i] = 0.0
        
        # Fill initial values
        if self.window < n:
            initial_median = np.median(returns[:self.window])
            initial_mad = np.median(np.abs(returns[:self.window] - initial_median))
            
            for i in range(self.window):
                if initial_mad > 0:
                    z_scores[i] = (returns[i] - initial_median) / (1.4826 * initial_mad)
        
        return z_scores
    
    def _calculate_percentile_z_scores(self, returns: np.ndarray) -> np.ndarray:
        """Calculate percentile-based Z-scores"""
        n = len(returns)
        z_scores = np.zeros(n)
        
        for i in range(self.window, n):
            window_returns = returns[i-self.window:i]
            current_return = returns[i]
            
            # Percentile rank
            percentile = stats.percentileofscore(window_returns, current_return) / 100
            
            # Convert to Z-score equivalent
            if 0 < percentile < 1:
                z_scores[i] = stats.norm.ppf(percentile)
            else:
                z_scores[i] = 0.0
        
        return z_scores
    
    def _calculate_outlier_probabilities(self, z_scores: np.ndarray) -> np.ndarray:
        """Calculate probability of each observation being an outlier"""
        # Two-tailed probability
        probabilities = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        return probabilities
    
    def _detect_extreme_events(self, z_scores: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Detect extreme events based on multiple criteria"""
        n = len(z_scores)
        extreme_events = np.zeros(n)
        
        for i in range(n):
            indicators = []
            
            # Z-score threshold
            indicators.append(abs(z_scores[i]) > self.outlier_threshold)
            
            # Percentile threshold
            if i >= 20:
                recent_returns = returns[max(0, i-20):i]
                percentile_99 = np.percentile(recent_returns, 99)
                percentile_1 = np.percentile(recent_returns, 1)
                indicators.append(returns[i] > percentile_99 or returns[i] < percentile_1)
            
            # Consecutive extreme moves
            if i >= 2:
                consecutive = (abs(z_scores[i]) > 1.5 and 
                             abs(z_scores[i-1]) > 1.5 and
                             abs(z_scores[i-2]) > 1.5)
                indicators.append(consecutive)
            
            # Aggregate indicator
            extreme_events[i] = sum(indicators) / len(indicators)
        
        return extreme_events
    
    def _calculate_regime_adjusted_z_scores(self, returns: np.ndarray, z_scores: np.ndarray) -> np.ndarray:
        """Calculate regime-adjusted Z-scores"""
        if len(returns) < 60:
            return z_scores.copy()
        
        # Detect volatility regimes
        window = 20
        rolling_vol = np.zeros(len(returns))
        
        for i in range(window, len(returns)):
            rolling_vol[i] = np.std(returns[i-window:i], ddof=1)
        
        # Regime classification
        vol_median = np.median(rolling_vol[window:])
        low_vol_regime = rolling_vol <= vol_median
        high_vol_regime = rolling_vol > vol_median
        
        # Adjust Z-scores by regime
        adjusted_z_scores = z_scores.copy()
        
        for i in range(len(z_scores)):
            if low_vol_regime[i]:
                # Lower threshold in low volatility regime
                adjusted_z_scores[i] = z_scores[i] * 1.2
            elif high_vol_regime[i]:
                # Higher threshold in high volatility regime
                adjusted_z_scores[i] = z_scores[i] * 0.8
        
        return adjusted_z_scores
    
    def _calculate_statistical_significance(self, z_scores: np.ndarray) -> np.ndarray:
        """Calculate statistical significance of Z-scores"""
        # Two-tailed p-values
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
        
        # Convert to significance levels
        significance = np.zeros(len(p_values))
        significance[p_values < 0.01] = 3  # Highly significant
        significance[(p_values >= 0.01) & (p_values < 0.05)] = 2  # Significant
        significance[(p_values >= 0.05) & (p_values < 0.1)] = 1  # Marginally significant
        
        return significance
    
    def _calculate_tail_probabilities(self, z_scores: np.ndarray) -> np.ndarray:
        """Calculate tail probabilities"""
        # Probability of being in the tail (beyond 2 standard deviations)
        tail_probs = np.zeros(len(z_scores))
        
        for i, z in enumerate(z_scores):
            if z > 0:
                tail_probs[i] = 1 - stats.norm.cdf(z)  # Right tail
            else:
                tail_probs[i] = stats.norm.cdf(z)      # Left tail
        
        return tail_probs
    
    def _estimate_distribution_parameters(self, returns: np.ndarray) -> Dict[str, float]:
        """Estimate distribution parameters"""
        # Basic moments
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        skewness = stats.skew(returns, bias=False)
        kurtosis = stats.kurtosis(returns, bias=False)
        
        # Fit normal distribution
        mu_norm, sigma_norm = stats.norm.fit(returns)
        
        # Fit t-distribution
        try:
            df_t, loc_t, scale_t = stats.t.fit(returns)
        except:
            df_t, loc_t, scale_t = 4, mean_ret, std_ret
        
        return {
            'mean': mean_ret,
            'std': std_ret,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'normal_mu': mu_norm,
            'normal_sigma': sigma_norm,
            't_df': df_t,
            't_loc': loc_t,
            't_scale': scale_t
        }
    
    def _perform_normality_tests(self, returns: np.ndarray) -> Dict[str, float]:
        """Perform various normality tests"""
        tests = {}
        
        try:
            # Shapiro-Wilk test
            if len(returns) <= 5000:
                sw_stat, sw_p = stats.shapiro(returns)
                tests['shapiro_wilk_statistic'] = sw_stat
                tests['shapiro_wilk_p_value'] = sw_p
        except:
            tests['shapiro_wilk_statistic'] = 0.0
            tests['shapiro_wilk_p_value'] = 1.0
        
        try:
            # Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(returns)
            tests['jarque_bera_statistic'] = jb_stat
            tests['jarque_bera_p_value'] = jb_p
        except:
            tests['jarque_bera_statistic'] = 0.0
            tests['jarque_bera_p_value'] = 1.0
        
        try:
            # Anderson-Darling test
            ad_stat, ad_crit, ad_sig = stats.anderson(returns, dist='norm')
            tests['anderson_darling_statistic'] = ad_stat
        except:
            tests['anderson_darling_statistic'] = 0.0
        
        return tests
    
    def _classify_outliers(self, z_scores: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Classify outliers by type and severity"""
        n = len(z_scores)
        classification = np.zeros(n)
        
        for i in range(n):
            z = z_scores[i]
            
            if abs(z) <= 1.96:
                classification[i] = 0  # Normal
            elif abs(z) <= 2.58:
                classification[i] = 1  # Mild outlier (95-99%)
            elif abs(z) <= 3.29:
                classification[i] = 2  # Moderate outlier (99-99.9%)
            else:
                classification[i] = 3  # Extreme outlier (>99.9%)
        
        return classification
    
    def _calculate_adaptive_thresholds(self, z_scores: np.ndarray) -> Dict[str, float]:
        """Calculate adaptive thresholds based on recent data"""
        if len(z_scores) < 20:
            return {
                'conservative': 2.0,
                'moderate': 2.5,
                'aggressive': 3.0
            }
        
        # Recent volatility of Z-scores
        recent_z_scores = z_scores[-60:] if len(z_scores) >= 60 else z_scores
        z_volatility = np.std(recent_z_scores)
        
        # Adaptive thresholds
        base_threshold = 2.0
        
        conservative = base_threshold * (1 + z_volatility / 2)
        moderate = base_threshold * (1 + z_volatility / 3)
        aggressive = base_threshold * (1 + z_volatility / 4)
        
        return {
            'conservative': min(conservative, 4.0),
            'moderate': min(moderate, 3.5),
            'aggressive': min(aggressive, 3.0)
        }
    
    def _calculate_confidence_intervals(self, z_scores: np.ndarray) -> np.ndarray:
        """Calculate confidence intervals for Z-scores"""
        n = len(z_scores)
        intervals = np.zeros((n, 2))  # [lower, upper]
        
        # Standard error for Z-scores (approximately 1 for large samples)
        se = 1.0
        
        # 95% confidence interval
        z_crit = 1.96
        
        for i in range(n):
            intervals[i, 0] = z_scores[i] - z_crit * se  # Lower bound
            intervals[i, 1] = z_scores[i] + z_crit * se  # Upper bound
        
        return intervals
    
    def _generate_signal(self, result: ZScoreResult, returns: np.ndarray, prices: np.ndarray) -> SignalType:
        """Generate trading signal based on Z-score analysis"""
        current_z = result.current_z_score
        robust_z = result.robust_z_scores[-1] if len(result.robust_z_scores) > 0 else 0
        outlier_prob = result.outlier_probabilities[-1] if len(result.outlier_probabilities) > 0 else 0
        extreme_event = result.extreme_event_indicators[-1] if len(result.extreme_event_indicators) > 0 else 0
        adaptive_thresh = result.adaptive_thresholds
        
        # Extreme value mean reversion signals
        if current_z > adaptive_thresh['moderate']:
            # Extremely high Z-score suggests overbought condition
            if outlier_prob < 0.01:  # Very rare event
                return SignalType.SELL
        elif current_z < -adaptive_thresh['moderate']:
            # Extremely low Z-score suggests oversold condition
            if outlier_prob < 0.01:  # Very rare event
                return SignalType.BUY
        
        # Robust Z-score confirmation
        if abs(current_z - robust_z) < 0.5:  # Consistent across methods
            if current_z > adaptive_thresh['conservative']:
                return SignalType.SELL
            elif current_z < -adaptive_thresh['conservative']:
                return SignalType.BUY
        
        # Extreme event signals
        if extreme_event > 0.7:  # Multiple extreme indicators
            # Direction based on Z-score
            if current_z > 0:
                return SignalType.SELL  # Extreme positive event
            else:
                return SignalType.BUY   # Extreme negative event
        
        # Regime-adjusted signals
        regime_z = result.regime_adjusted_z_scores[-1] if len(result.regime_adjusted_z_scores) > 0 else current_z
        
        if abs(regime_z) > 2.5:
            # Strong regime-adjusted signal
            if regime_z > 0:
                return SignalType.SELL
            else:
                return SignalType.BUY
        
        # Consecutive extreme moves (contrarian signal)
        if len(result.rolling_z_scores) >= 3:
            recent_z = result.rolling_z_scores[-3:]
            if all(z > 2.0 for z in recent_z):
                return SignalType.SELL  # Consecutive extreme positive
            elif all(z < -2.0 for z in recent_z):
                return SignalType.BUY   # Consecutive extreme negative
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: ZScoreResult) -> float:
        """Calculate signal strength based on Z-score characteristics"""
        current_z = abs(result.current_z_score)
        outlier_prob = 1 - result.outlier_probabilities[-1] if len(result.outlier_probabilities) > 0 else 0.5
        extreme_event = result.extreme_event_indicators[-1] if len(result.extreme_event_indicators) > 0 else 0
        significance = result.statistical_significance[-1] if len(result.statistical_significance) > 0 else 0
        
        # Z-score magnitude strength
        z_strength = min(current_z / 4.0, 1.0)  # Normalize by 4 standard deviations
        
        # Rarity strength (inverse of probability)
        rarity_strength = outlier_prob
        
        # Extreme event strength
        event_strength = extreme_event
        
        # Statistical significance strength
        sig_strength = significance / 3.0  # Normalize by max significance level
        
        return (z_strength + rarity_strength + event_strength + sig_strength) / 4
    
    def _calculate_confidence(self, result: ZScoreResult) -> float:
        """Calculate confidence based on statistical validity"""
        # Normality test confidence
        jb_p = result.normality_tests.get('jarque_bera_p_value', 0.5)
        normality_confidence = jb_p if jb_p > 0.05 else 0.3  # Lower confidence if non-normal
        
        # Method consistency
        current_z = result.current_z_score
        robust_z = result.robust_z_scores[-1] if len(result.robust_z_scores) > 0 else current_z
        exp_z = result.exponential_z_scores[-1] if len(result.exponential_z_scores) > 0 else current_z
        
        method_consistency = 1 / (1 + abs(current_z - robust_z) + abs(current_z - exp_z))
        
        # Distribution fit quality
        dist_params = result.distribution_parameters
        skewness = abs(dist_params.get('skewness', 0))
        kurtosis = abs(dist_params.get('kurtosis', 0))
        
        # Lower confidence for highly skewed or heavy-tailed distributions
        dist_quality = 1 / (1 + skewness + kurtosis / 3)
        
        # Confidence interval tightness
        if len(result.confidence_intervals) > 0:
            ci_width = result.confidence_intervals[-1, 1] - result.confidence_intervals[-1, 0]
            ci_tightness = max(0, 1 - ci_width / 8)  # Normalize by reasonable width
        else:
            ci_tightness = 0.5
        
        return (normality_confidence + method_consistency + dist_quality + ci_tightness) / 4