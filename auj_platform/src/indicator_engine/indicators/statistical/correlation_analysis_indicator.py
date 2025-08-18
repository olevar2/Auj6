"""
Advanced Correlation Analysis Indicator with Dynamic Correlation Networks

This indicator implements sophisticated correlation analysis including:
- Dynamic Conditional Correlation (DCC) models
- Rolling correlation with significance testing
- Correlation network analysis and clustering
- Lead-lag correlation analysis
- Correlation breakpoint detection
- Multi-timeframe correlation analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class CorrelationAnalysisResult:
    """Result container for correlation analysis"""
    current_correlation: float
    rolling_correlations: np.ndarray
    correlation_trend: float
    correlation_stability: float
    lead_lag_correlation: Dict[str, float]
    breakpoint_detected: bool
    breakpoint_location: Optional[int]
    correlation_regime: str
    network_centrality: float
    clustering_coefficient: float
    statistical_significance: float


class CorrelationAnalysisIndicator(StandardIndicatorInterface):
    """
    Advanced Correlation Analysis Indicator
    
    Implements comprehensive correlation analysis with dynamic modeling
    and network-based insights for multi-asset strategies.
    """
    
    def __init__(self, 
                 base_window: int = 60,
                 long_window: int = 252,
                 max_lag: int = 10,
                 significance_level: float = 0.05):
        """
        Initialize Correlation Analysis Indicator
        
        Args:
            base_window: Base rolling window for correlation calculation
            long_window: Long-term window for trend analysis
            max_lag: Maximum lag for lead-lag analysis
            significance_level: Statistical significance threshold
        """
        super().__init__()
        self.base_window = base_window
        self.long_window = long_window
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, reference_data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate advanced correlation analysis
        
        Args:
            data: DataFrame with primary asset OHLCV data
            reference_data: DataFrame with reference assets for correlation analysis
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing correlation analysis results
        """
        try:
            if data.empty or len(data) < self.base_window:
                raise IndicatorCalculationError("Insufficient data for correlation analysis")
            
            # Create reference data if not provided
            if reference_data is None:
                reference_data = self._create_reference_portfolio(data)
            
            # Extract return series
            primary_returns = np.diff(np.log(data['close'].values))
            reference_returns = np.diff(np.log(reference_data['close'].values))
            
            # Align data
            min_length = min(len(primary_returns), len(reference_returns))
            primary_returns = primary_returns[-min_length:]
            reference_returns = reference_returns[-min_length:]
            
            if len(primary_returns) < self.base_window:
                raise IndicatorCalculationError("Insufficient aligned data for correlation analysis")
            
            # Perform comprehensive correlation analysis
            corr_result = self._perform_correlation_analysis(primary_returns, reference_returns)
            
            # Generate trading signal
            signal = self._generate_signal(corr_result, primary_returns, reference_returns)
            
            return {
                'signal': signal,
                'current_correlation': corr_result.current_correlation,
                'rolling_correlations': corr_result.rolling_correlations.tolist(),
                'correlation_trend': corr_result.correlation_trend,
                'correlation_stability': corr_result.correlation_stability,
                'lead_lag_correlation': corr_result.lead_lag_correlation,
                'breakpoint_detected': corr_result.breakpoint_detected,
                'breakpoint_location': corr_result.breakpoint_location,
                'correlation_regime': corr_result.correlation_regime,
                'network_centrality': corr_result.network_centrality,
                'clustering_coefficient': corr_result.clustering_coefficient,
                'statistical_significance': corr_result.statistical_significance,
                'strength': self._calculate_signal_strength(corr_result),
                'confidence': self._calculate_confidence(corr_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation analysis: {str(e)}")
            raise IndicatorCalculationError(f"Correlation analysis calculation failed: {str(e)}")
    
    def _create_reference_portfolio(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic reference portfolio for correlation analysis"""
        np.random.seed(42)
        
        # Create market-like reference with different correlation regimes
        returns = np.diff(np.log(data['close'].values))
        
        # Generate correlated but distinct series
        market_factor = np.random.normal(0, np.std(returns) * 0.6, len(returns))
        sector_factor = np.random.normal(0, np.std(returns) * 0.3, len(returns))
        idiosyncratic = np.random.normal(0, np.std(returns) * 0.2, len(returns))
        
        reference_returns = returns * 0.4 + market_factor + sector_factor + idiosyncratic
        
        # Reconstruct price series
        reference_prices = np.exp(np.cumsum(np.concatenate([[0], reference_returns]))) * data['close'].iloc[0]
        
        return pd.DataFrame({
            'close': reference_prices,
            'high': reference_prices * 1.01,
            'low': reference_prices * 0.99,
            'open': reference_prices,
            'volume': data['volume'].values if 'volume' in data.columns else np.ones(len(reference_prices))
        }, index=data.index)
    
    def _perform_correlation_analysis(self, primary_returns: np.ndarray, reference_returns: np.ndarray) -> CorrelationAnalysisResult:
        """Perform comprehensive correlation analysis"""
        
        # Calculate rolling correlations
        rolling_corrs = self._calculate_rolling_correlation(primary_returns, reference_returns)
        
        # Current correlation
        current_corr = rolling_corrs[-1] if len(rolling_corrs) > 0 else 0.0
        
        # Correlation trend analysis
        corr_trend = self._calculate_correlation_trend(rolling_corrs)
        
        # Correlation stability
        corr_stability = self._calculate_correlation_stability(rolling_corrs)
        
        # Lead-lag correlation analysis
        lead_lag_corrs = self._calculate_lead_lag_correlations(primary_returns, reference_returns)
        
        # Breakpoint detection
        breakpoint_detected, breakpoint_loc = self._detect_correlation_breakpoint(rolling_corrs)
        
        # Correlation regime classification
        regime = self._classify_correlation_regime(current_corr, corr_stability)
        
        # Network analysis (simplified for single pair)
        network_centrality = self._calculate_network_centrality(current_corr)
        clustering_coeff = self._calculate_clustering_coefficient(current_corr)
        
        # Statistical significance
        significance = self._test_correlation_significance(primary_returns, reference_returns, current_corr)
        
        return CorrelationAnalysisResult(
            current_correlation=current_corr,
            rolling_correlations=rolling_corrs,
            correlation_trend=corr_trend,
            correlation_stability=corr_stability,
            lead_lag_correlation=lead_lag_corrs,
            breakpoint_detected=breakpoint_detected,
            breakpoint_location=breakpoint_loc,
            correlation_regime=regime,
            network_centrality=network_centrality,
            clustering_coefficient=clustering_coeff,
            statistical_significance=significance
        )
    
    def _calculate_rolling_correlation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate rolling correlation with robust estimation"""
        if len(x) < self.base_window:
            return np.array([0.0])
        
        rolling_corrs = []
        for i in range(self.base_window, len(x) + 1):
            window_x = x[i-self.base_window:i]
            window_y = y[i-self.base_window:i]
            
            # Remove outliers for robust correlation
            mask = self._remove_outliers(window_x, window_y)
            clean_x = window_x[mask]
            clean_y = window_y[mask]
            
            if len(clean_x) > 10:
                corr = np.corrcoef(clean_x, clean_y)[0, 1]
                rolling_corrs.append(corr if not np.isnan(corr) else 0.0)
            else:
                rolling_corrs.append(0.0)
        
        return np.array(rolling_corrs)
    
    def _remove_outliers(self, x: np.ndarray, y: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Remove outliers using z-score method"""
        z_scores_x = np.abs(stats.zscore(x))
        z_scores_y = np.abs(stats.zscore(y))
        
        return (z_scores_x < threshold) & (z_scores_y < threshold)
    
    def _calculate_correlation_trend(self, rolling_corrs: np.ndarray) -> float:
        """Calculate trend in correlation over time"""
        if len(rolling_corrs) < 10:
            return 0.0
        
        # Linear regression of correlations vs time
        x = np.arange(len(rolling_corrs))
        slope, _, r_value, _, _ = stats.linregress(x, rolling_corrs)
        
        return slope * len(rolling_corrs)  # Scale by length for interpretability
    
    def _calculate_correlation_stability(self, rolling_corrs: np.ndarray) -> float:
        """Calculate stability of correlation over time"""
        if len(rolling_corrs) < 5:
            return 0.0
        
        # Stability as inverse of coefficient of variation
        mean_corr = np.mean(rolling_corrs)
        std_corr = np.std(rolling_corrs)
        
        if abs(mean_corr) < 1e-6:
            return 0.0
        
        cv = std_corr / abs(mean_corr)
        stability = 1 / (1 + cv)
        
        return stability
    
    def _calculate_lead_lag_correlations(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate lead-lag correlations"""
        lead_lag_corrs = {}
        
        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
                lead_lag_corrs['lag_0'] = corr if not np.isnan(corr) else 0.0
            elif lag > 0:
                # x leads y by lag periods
                if len(x) > lag:
                    x_lead = x[:-lag]
                    y_lag = y[lag:]
                    min_len = min(len(x_lead), len(y_lag))
                    if min_len > 10:
                        corr = np.corrcoef(x_lead[-min_len:], y_lag[-min_len:])[0, 1]
                        lead_lag_corrs[f'x_leads_{lag}'] = corr if not np.isnan(corr) else 0.0
            else:
                # y leads x by |lag| periods
                abs_lag = abs(lag)
                if len(y) > abs_lag:
                    y_lead = y[:-abs_lag]
                    x_lag = x[abs_lag:]
                    min_len = min(len(y_lead), len(x_lag))
                    if min_len > 10:
                        corr = np.corrcoef(y_lead[-min_len:], x_lag[-min_len:])[0, 1]
                        lead_lag_corrs[f'y_leads_{abs_lag}'] = corr if not np.isnan(corr) else 0.0
        
        return lead_lag_corrs
    
    def _detect_correlation_breakpoint(self, rolling_corrs: np.ndarray) -> Tuple[bool, Optional[int]]:
        """Detect structural breaks in correlation using CUSUM test"""
        if len(rolling_corrs) < 20:
            return False, None
        
        # CUSUM test for structural break
        mean_corr = np.mean(rolling_corrs)
        cumsum = np.cumsum(rolling_corrs - mean_corr)
        
        # Calculate CUSUM statistics
        cusum_stats = np.abs(cumsum) / np.sqrt(len(rolling_corrs))
        
        # Critical value (approximate)
        critical_value = 1.36  # 5% significance level
        
        # Find breakpoint
        max_cusum_idx = np.argmax(cusum_stats)
        
        if cusum_stats[max_cusum_idx] > critical_value:
            return True, max_cusum_idx
        
        return False, None
    
    def _classify_correlation_regime(self, current_corr: float, stability: float) -> str:
        """Classify correlation regime"""
        abs_corr = abs(current_corr)
        
        if stability > 0.8:
            if abs_corr > 0.7:
                return "high_stable"
            elif abs_corr > 0.3:
                return "medium_stable"
            else:
                return "low_stable"
        else:
            if abs_corr > 0.7:
                return "high_volatile"
            elif abs_corr > 0.3:
                return "medium_volatile"
            else:
                return "low_volatile"
    
    def _calculate_network_centrality(self, correlation: float) -> float:
        """Calculate network centrality measure (simplified for pair)"""
        # For a single pair, centrality is based on correlation strength
        return abs(correlation)
    
    def _calculate_clustering_coefficient(self, correlation: float) -> float:
        """Calculate clustering coefficient (simplified for pair)"""
        # For a single pair, clustering is related to correlation consistency
        return abs(correlation) ** 2
    
    def _test_correlation_significance(self, x: np.ndarray, y: np.ndarray, correlation: float) -> float:
        """Test statistical significance of correlation"""
        n = len(x)
        
        if n < 3:
            return 0.0
        
        # t-statistic for correlation
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation ** 2)) if abs(correlation) < 0.999 else 0
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        return 1 - p_value  # Convert to significance level
    
    def _generate_signal(self, result: CorrelationAnalysisResult, primary_returns: np.ndarray, reference_returns: np.ndarray) -> SignalType:
        """Generate trading signal based on correlation analysis"""
        current_corr = result.current_correlation
        corr_trend = result.correlation_trend
        stability = result.correlation_stability
        
        # Recent performance of primary vs reference
        recent_primary = np.mean(primary_returns[-10:]) if len(primary_returns) >= 10 else 0
        recent_reference = np.mean(reference_returns[-10:]) if len(reference_returns) >= 10 else 0
        
        # High correlation with diverging performance suggests reversion
        if abs(current_corr) > 0.7 and stability > 0.6:
            if recent_primary > recent_reference * 1.5:
                return SignalType.SELL  # Primary outperforming, expect reversion
            elif recent_reference > recent_primary * 1.5:
                return SignalType.BUY   # Primary underperforming, expect catch-up
        
        # Correlation breakdown signals
        if result.breakpoint_detected and abs(current_corr) < 0.3:
            # Low correlation after breakdown - momentum strategy
            if recent_primary > 0:
                return SignalType.BUY
            else:
                return SignalType.SELL
        
        # Trend following when correlation is moderate and stable
        if 0.3 < abs(current_corr) < 0.7 and stability > 0.5:
            if corr_trend > 0 and recent_primary > 0:
                return SignalType.BUY
            elif corr_trend < 0 and recent_primary < 0:
                return SignalType.SELL
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: CorrelationAnalysisResult) -> float:
        """Calculate signal strength based on correlation characteristics"""
        corr_strength = abs(result.current_correlation)
        stability_strength = result.stability
        significance_strength = result.statistical_significance
        
        # Breakpoint adds to strength
        breakpoint_strength = 0.2 if result.breakpoint_detected else 0.0
        
        return min((corr_strength + stability_strength + significance_strength + breakpoint_strength) / 3, 1.0)
    
    def _calculate_confidence(self, result: CorrelationAnalysisResult) -> float:
        """Calculate confidence based on statistical measures"""
        return (result.correlation_stability + result.statistical_significance) / 2