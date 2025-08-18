"""
Fractal Market Hypothesis Indicator - Advanced Implementation

This indicator implements the Fractal Market Hypothesis (FMH) theory developed by
Edgar Peters, incorporating memory persistence analysis, regime change detection,
and multi-horizon investment behavior modeling. Features include:
- FMH theory implementation with liquidity analysis
- Memory persistence measurement using rescaled range statistics
- Multi-horizon investor behavior modeling
- Regime change detection with transition probability analysis
- Market stability and information incorporation efficiency
- Behavioral finance integration with sentiment cycles

Mission: Supporting humanitarian trading platform for poor and sick children through
maximum profitability via advanced fractal market hypothesis analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import warnings

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FMHResult:
    """Results container for Fractal Market Hypothesis"""
    liquidity_index: float
    memory_persistence: float
    information_efficiency: float
    market_stability: float
    regime_probability: float
    investor_horizon_dominance: str
    market_regime: str
    volatility_clustering: float
    trend_strength: float
    regime_confidence: float

class FractalMarketHypothesisIndicator(StandardIndicatorInterface):
    """
    Advanced Fractal Market Hypothesis Indicator
    
    Implements Edgar Peters' FMH theory with memory persistence analysis,
    regime change detection, and multi-horizon investment behavior modeling.
    """
    
    def __init__(self, 
                 short_window: int = 20,
                 medium_window: int = 50,
                 long_window: int = 200,
                 memory_window: int = 100,
                 regime_window: int = 60):
        super().__init__()
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.memory_window = memory_window
        self.regime_window = regime_window
        
        logger.info(f"Initialized FractalMarketHypothesisIndicator with memory_window={memory_window}")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            self._validate_data(data)
            
            if len(data) < self.long_window:
                return self._create_default_result()
            
            # Calculate FMH components
            liquidity_index = self._calculate_liquidity_index(data)
            memory_persistence = self._calculate_memory_persistence(data)
            information_efficiency = self._calculate_information_efficiency(data)
            market_stability = self._calculate_market_stability(data)
            
            # Regime analysis
            regime_analysis = self._analyze_market_regime(data)
            
            # Investor horizon analysis
            horizon_analysis = self._analyze_investor_horizons(data)
            
            # Volatility clustering
            volatility_clustering = self._calculate_volatility_clustering(data)
            
            # Trend strength
            trend_strength = self._calculate_trend_strength(data)
            
            result = FMHResult(
                liquidity_index=liquidity_index,
                memory_persistence=memory_persistence,
                information_efficiency=information_efficiency,
                market_stability=market_stability,
                regime_probability=regime_analysis['probability'],
                investor_horizon_dominance=horizon_analysis['dominance'],
                market_regime=regime_analysis['regime'],
                volatility_clustering=volatility_clustering,
                trend_strength=trend_strength,
                regime_confidence=regime_analysis['confidence']
            )
            
            return self._format_output(result, data.index[-1])
            
        except Exception as e:
            logger.error(f"Error in FMH calculation: {e}")
            raise IndicatorCalculationError(f"FractalMarketHypothesisIndicator calculation failed: {e}")

    def _calculate_liquidity_index(self, data: pd.DataFrame) -> float:
        """Calculate market liquidity index based on volume and price impact"""
        try:
            if len(data) < self.medium_window:
                return 0.5
            
            recent_data = data[-self.medium_window:]
            
            # Price impact measure
            if 'volume' in recent_data.columns:
                price_changes = recent_data['close'].pct_change().abs()
                volumes = recent_data['volume']
                
                # Normalize volumes
                vol_scaler = StandardScaler()
                norm_volumes = vol_scaler.fit_transform(volumes.values.reshape(-1, 1)).flatten()
                
                # Calculate price impact per unit volume
                valid_idx = (volumes > 0) & (~price_changes.isna())
                if valid_idx.sum() == 0:
                    return 0.5
                
                price_impact = price_changes[valid_idx] / (norm_volumes[valid_idx] + 1e-8)
                
                # Liquidity is inverse of price impact
                avg_impact = price_impact.mean()
                liquidity = 1.0 / (1.0 + avg_impact * 100)
            else:
                # Use price volatility as proxy
                returns = recent_data['close'].pct_change().dropna()
                volatility = returns.std()
                
                # Higher volatility indicates lower liquidity
                liquidity = 1.0 / (1.0 + volatility * 50)
            
            return np.clip(liquidity, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Liquidity index calculation error: {e}")
            return 0.5

    def _calculate_memory_persistence(self, data: pd.DataFrame) -> float:
        """Calculate memory persistence using rescaled range (R/S) analysis"""
        try:
            if len(data) < self.memory_window:
                return 0.5
            
            recent_data = data[-self.memory_window:]
            prices = recent_data['close'].values
            log_returns = np.diff(np.log(prices))
            
            if len(log_returns) < 20:
                return 0.5
            
            # R/S analysis for different time scales
            scales = np.unique(np.logspace(0.5, np.log10(len(log_returns) // 4), 8).astype(int))
            rs_values = []
            
            for scale in scales:
                if scale >= len(log_returns):
                    continue
                
                # Split into non-overlapping periods
                n_periods = len(log_returns) // scale
                rs_period_values = []
                
                for i in range(n_periods):
                    period_returns = log_returns[i*scale:(i+1)*scale]
                    
                    if len(period_returns) == 0:
                        continue
                    
                    # Calculate mean return
                    mean_return = np.mean(period_returns)
                    
                    # Calculate cumulative deviations from mean
                    deviations = period_returns - mean_return
                    cumulative_deviations = np.cumsum(deviations)
                    
                    # Range of cumulative deviations
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                    
                    # Standard deviation
                    S = np.std(period_returns)
                    
                    # R/S ratio
                    if S > 0:
                        rs_period_values.append(R / S)
                
                if rs_period_values:
                    rs_values.append(np.mean(rs_period_values))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Fit log-log relationship to get Hurst exponent
            log_scales = np.log(scales[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            slope, _, r_value, _, _ = stats.linregress(log_scales, log_rs)
            
            # Hurst exponent (memory persistence)
            if r_value ** 2 > 0.6:
                hurst = np.clip(slope, 0.0, 1.0)
            else:
                hurst = 0.5  # Default for random walk
            
            return hurst
            
        except Exception as e:
            logger.warning(f"Memory persistence calculation error: {e}")
            return 0.5

    def _calculate_information_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate information efficiency using variance ratio test"""
        try:
            if len(data) < self.medium_window:
                return 0.5
            
            recent_data = data[-self.medium_window:]
            prices = recent_data['close'].values
            log_returns = np.diff(np.log(prices))
            
            if len(log_returns) < 16:
                return 0.5
            
            # Variance ratio test for different periods
            base_period = 2
            test_periods = [4, 8, 16]
            variance_ratios = []
            
            for period in test_periods:
                if period >= len(log_returns):
                    continue
                
                # Calculate overlapping returns for base and test periods
                base_returns = []
                test_returns = []
                
                for i in range(len(log_returns) - period + 1):
                    if i + base_period <= len(log_returns):
                        base_returns.append(np.sum(log_returns[i:i+base_period]))
                    test_returns.append(np.sum(log_returns[i:i+period]))
                
                if len(base_returns) == 0 or len(test_returns) == 0:
                    continue
                
                # Calculate variance ratio
                base_var = np.var(base_returns)
                test_var = np.var(test_returns)
                
                if base_var > 0:
                    # Expected variance ratio for random walk
                    expected_ratio = period / base_period
                    actual_ratio = test_var / base_var
                    
                    # Efficiency measure (closer to 1.0 = more efficient)
                    efficiency = 1.0 - abs(actual_ratio - expected_ratio) / expected_ratio
                    variance_ratios.append(max(0.0, efficiency))
            
            if variance_ratios:
                return np.mean(variance_ratios)
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Information efficiency calculation error: {e}")
            return 0.5

    def _calculate_market_stability(self, data: pd.DataFrame) -> float:
        """Calculate market stability using volatility persistence and autocorrelation"""
        try:
            if len(data) < self.medium_window:
                return 0.5
            
            recent_data = data[-self.medium_window:]
            returns = recent_data['close'].pct_change().dropna()
            
            if len(returns) < 20:
                return 0.5
            
            # Volatility stability
            volatilities = returns.rolling(window=5).std().dropna()
            vol_stability = 1.0 - (volatilities.std() / (volatilities.mean() + 1e-8))
            
            # Return autocorrelation (lower is more stable)
            autocorr = returns.autocorr(lag=1)
            if pd.isna(autocorr):
                autocorr = 0.0
            
            autocorr_stability = 1.0 - abs(autocorr)
            
            # Combine measures
            stability = (vol_stability * 0.6 + autocorr_stability * 0.4)
            
            return np.clip(stability, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Market stability calculation error: {e}")
            return 0.5

    def _analyze_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market regime and transition probability"""
        try:
            if len(data) < self.regime_window:
                return {'regime': 'UNKNOWN', 'probability': 0.0, 'confidence': 0.0}
            
            recent_data = data[-self.regime_window:]
            returns = recent_data['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return {'regime': 'UNKNOWN', 'probability': 0.0, 'confidence': 0.0}
            
            # Calculate regime characteristics
            volatility = returns.std()
            trend = returns.mean()
            autocorr = returns.autocorr(lag=1)
            
            if pd.isna(autocorr):
                autocorr = 0.0
            
            # Regime classification based on FMH theory
            if volatility < 0.01 and abs(trend) < 0.001:
                regime = 'STABLE_EQUILIBRIUM'
                probability = 0.8
            elif volatility > 0.03:
                regime = 'HIGH_VOLATILITY'
                probability = 0.7
            elif abs(autocorr) > 0.3:
                regime = 'TRENDING'
                probability = 0.6
            else:
                regime = 'TRANSITIONAL'
                probability = 0.5
            
            # Calculate confidence based on data consistency
            confidence = min(len(returns) / self.regime_window, 1.0)
            
            return {
                'regime': regime,
                'probability': probability,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.warning(f"Regime analysis error: {e}")
            return {'regime': 'UNKNOWN', 'probability': 0.0, 'confidence': 0.0}

    def _analyze_investor_horizons(self, data: pd.DataFrame) -> Dict[str, str]:
        """Analyze dominant investor time horizons"""
        try:
            if len(data) < self.long_window:
                return {'dominance': 'UNKNOWN'}
            
            # Calculate momentum across different horizons
            short_momentum = self._calculate_momentum(data, self.short_window)
            medium_momentum = self._calculate_momentum(data, self.medium_window)
            long_momentum = self._calculate_momentum(data, self.long_window)
            
            # Determine dominant horizon
            momentums = {
                'SHORT_TERM': abs(short_momentum),
                'MEDIUM_TERM': abs(medium_momentum),
                'LONG_TERM': abs(long_momentum)
            }
            
            dominant = max(momentums, key=momentums.get)
            
            return {'dominance': dominant}
            
        except Exception as e:
            logger.warning(f"Investor horizon analysis error: {e}")
            return {'dominance': 'UNKNOWN'}

    def _calculate_momentum(self, data: pd.DataFrame, window: int) -> float:
        """Calculate momentum for a given time window"""
        try:
            if len(data) < window:
                return 0.0
            
            recent_data = data[-window:]
            return (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1) * 100
            
        except Exception as e:
            logger.warning(f"Momentum calculation error: {e}")
            return 0.0

    def _calculate_volatility_clustering(self, data: pd.DataFrame) -> float:
        """Calculate volatility clustering coefficient"""
        try:
            if len(data) < self.medium_window:
                return 0.0
            
            recent_data = data[-self.medium_window:]
            returns = recent_data['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return 0.0
            
            # Calculate rolling volatility
            vol_window = min(5, len(returns) // 4)
            volatilities = returns.rolling(window=vol_window).std().dropna()
            
            if len(volatilities) < 5:
                return 0.0
            
            # Autocorrelation of volatilities (clustering measure)
            vol_autocorr = volatilities.autocorr(lag=1)
            
            if pd.isna(vol_autocorr):
                return 0.0
            
            return np.clip(vol_autocorr, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Volatility clustering calculation error: {e}")
            return 0.0

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate overall trend strength"""
        try:
            if len(data) < self.medium_window:
                return 0.0
            
            recent_data = data[-self.medium_window:]
            prices = recent_data['close'].values
            
            # Linear regression slope
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            
            # Normalize slope by price level
            normalized_slope = slope / (np.mean(prices) + 1e-8)
            
            # Trend strength as combination of slope and R²
            trend_strength = abs(normalized_slope) * (r_value ** 2) * 100
            
            return np.clip(trend_strength, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Trend strength calculation error: {e}")
            return 0.0

    def _format_output(self, result: FMHResult, timestamp) -> Dict[str, Any]:
        """Format the calculation results for output"""
        return {
            'timestamp': timestamp,
            'indicator_name': 'FractalMarketHypothesis',
            'liquidity_index': round(result.liquidity_index, 4),
            'memory_persistence': round(result.memory_persistence, 4),
            'information_efficiency': round(result.information_efficiency, 4),
            'market_stability': round(result.market_stability, 4),
            'regime_probability': round(result.regime_probability, 4),
            'investor_horizon_dominance': result.investor_horizon_dominance,
            'market_regime': result.market_regime,
            'volatility_clustering': round(result.volatility_clustering, 4),
            'trend_strength': round(result.trend_strength, 4),
            'regime_confidence': round(result.regime_confidence, 4)
        }

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
        if len(data) < self.short_window:
            raise IndicatorCalculationError(f"Insufficient data: minimum {self.short_window} periods required")

    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data cases"""
        return {
            'timestamp': pd.Timestamp.now(),
            'indicator_name': 'FractalMarketHypothesis',
            'liquidity_index': 0.5,
            'memory_persistence': 0.5,
            'information_efficiency': 0.5,
            'market_stability': 0.5,
            'regime_probability': 0.0,
            'investor_horizon_dominance': 'INSUFFICIENT_DATA',
            'market_regime': 'UNKNOWN',
            'volatility_clustering': 0.0,
            'trend_strength': 0.0,
            'regime_confidence': 0.0
        }

    def get_required_columns(self) -> List[str]:
        return ['close']

    def get_indicator_name(self) -> str:
        return "FractalMarketHypothesis"