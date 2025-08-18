"""
Variable Moving Average Indicator

Implements a sophisticated variable moving average system that dynamically adjusts
its period based on market volatility, trend strength, and other market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

from auj_platform.src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import IndicatorCalculationError

logger = logging.getLogger(__name__)


@dataclass
class VariabilityMetrics:
    """Data structure for market variability analysis"""
    volatility_regime: str
    trend_strength: float
    market_efficiency: float
    noise_level: float
    cycle_detection: float
    adaptive_factor: float


@dataclass
class PeriodOptimization:
    """Data structure for period optimization results"""
    optimal_period: int
    optimization_score: float
    confidence_level: float
    stability_metric: float
    performance_history: List[float]


@dataclass
class VariableMASignal:
    """Data structure for variable MA signal analysis"""
    ma_value: float
    current_period: int
    period_efficiency: float
    trend_following_mode: str
    mean_reversion_mode: str
    signal_strength: float
    adaptation_speed: float


class VariableMovingAverageIndicator(StandardIndicatorInterface):
    """
    Advanced Variable Moving Average Indicator
    
    This indicator dynamically adjusts its averaging period based on market conditions,
    providing optimal smoothing during different market regimes while maintaining
    responsiveness to significant price movements.
    """
    
def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'base_period': 20,
            'min_period': 5,
            'max_period': 100,
            'volatility_window': 50,
            'efficiency_window': 30,
            'adaptation_speed': 0.1,
            'volatility_threshold_low': 0.01,
            'volatility_threshold_high': 0.05,
            'trend_threshold': 0.02,
            'optimization_lookback': 200,
            'performance_window': 50,
            'smoothing_methods': ['sma', 'ema', 'wma', 'hull'],
            'regime_detection_window': 100,
            'noise_filter_enabled': True,
            'cycle_analysis_enabled': True,
            'multi_timeframe_enabled': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="Variable Moving Average")
        
        # State tracking
        self.period_history = []
        self.performance_history = []
        self.variability_history = []
        self.optimization_results = None
        self.current_period = self.parameters['base_period']
        self.adaptation_momentum = 0.0
        
def _calculate_market_variability(self, data: pd.DataFrame) -> VariabilityMetrics:
        """Calculate comprehensive market variability metrics"""
        try:
            window = self.parameters['volatility_window']
            if len(data) < window:
                return VariabilityMetrics()
                    volatility_regime='unknown',
                    trend_strength=0.0,
                    market_efficiency=0.0,
                    noise_level=0.5,
                    cycle_detection=0.0,
                    adaptive_factor=1.0
(                )
            
            recent_data = data.tail(window)
            
            # 1. Volatility regime analysis
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            vol_low = self.parameters['volatility_threshold_low']
            vol_high = self.parameters['volatility_threshold_high']
            
            if volatility < vol_low:
                volatility_regime = 'low'
            elif volatility > vol_high:
                volatility_regime = 'high'
            else:
                volatility_regime = 'medium'
            
            # 2. Trend strength analysis
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            price_path = abs(recent_data['close'].diff()).sum()
            net_change = abs(recent_data['close'].iloc[-1] - recent_data['close'].iloc[0])
            
            trend_strength = net_change / price_path if price_path > 0 else 0
            
            # 3. Market efficiency (directional persistence)
            direction_changes = ((returns[:-1] * returns[1:]) < 0).sum()
            efficiency = 1 - (direction_changes / max(1, len(returns) - 1))
            
            # 4. Noise level (high frequency fluctuations)
            high_freq_moves = abs(recent_data['close'].diff()).rolling(3).mean()
            noise_level = high_freq_moves.std() / high_freq_moves.mean() if high_freq_moves.mean() > 0 else 0.5
            
            # 5. Cycle detection using autocorrelation
            if self.parameters['cycle_analysis_enabled']:
                cycle_strength = self._detect_market_cycles(recent_data)
            else:
                cycle_strength = 0.0
            
            # 6. Adaptive factor calculation
            # Combine all metrics to determine how much adaptation is needed
            adaptive_factor = ()
                (1.0 - trend_strength) * 0.3 +  # Less trending = more adaptation
                volatility * 10 * 0.3 +          # Higher volatility = more adaptation
                (1.0 - efficiency) * 0.2 +       # Less efficient = more adaptation
                noise_level * 0.2                # More noise = more adaptation
(            )
            
            adaptive_factor = max(0.1, min(2.0, adaptive_factor))
            
            return VariabilityMetrics()
                volatility_regime=volatility_regime,
                trend_strength=trend_strength,
                market_efficiency=efficiency,
                noise_level=noise_level,
                cycle_detection=cycle_strength,
                adaptive_factor=adaptive_factor
(            )
            
        except Exception as e:
            logger.error(f"Error calculating market variability: {str(e)}")
            return VariabilityMetrics()
                volatility_regime='unknown',
                trend_strength=0.0,
                market_efficiency=0.0,
                noise_level=0.5,
                cycle_detection=0.0,
                adaptive_factor=1.0
(            )
    
def _detect_market_cycles(self, data: pd.DataFrame) -> float:
        """Detect market cycles using autocorrelation analysis"""
        try:
            if len(data) < 50:
                return 0.0
            
            # Detrend the price series
            prices = data['close'].values
            x = np.arange(len(prices))
            
            # Linear detrending
            slope, intercept, _, _, _ = stats.linregress(x, prices)
            detrended = prices - (slope * x + intercept)
            
            # Calculate autocorrelation for different lags
            max_lag = min(30, len(detrended) // 4)
            autocorrs = []
            
            for lag in range(1, max_lag):
                if lag < len(detrended):
                    corr = np.corrcoef(detrended[:-lag], detrended[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(abs(corr))
            
            # Find peaks in autocorrelation (indicating cycles)
            if len(autocorrs) > 5:
                peaks, _ = find_peaks(autocorrs, height=0.1, distance=3)
                cycle_strength = np.mean([autocorrs[p] for p in peaks]) if len(peaks) > 0 else 0.0
            else:
                cycle_strength = 0.0
            
            return min(1.0, cycle_strength)
            
        except Exception as e:
            logger.error(f"Error detecting market cycles: {str(e)}")
            return 0.0
    
def _optimize_period(self, data: pd.DataFrame,:)
(                        variability: VariabilityMetrics) -> PeriodOptimization:
        """Optimize the moving average period based on current market conditions"""
        try:
            lookback = min(len(data), self.parameters['optimization_lookback'])
            optimization_data = data.tail(lookback)
            
            if len(optimization_data) < 50:
                return PeriodOptimization()
                    optimal_period=self.parameters['base_period'],
                    optimization_score=0.0,
                    confidence_level=0.0,
                    stability_metric=0.0,
                    performance_history=[]
(                )
            
            # Test different periods
            min_period = self.parameters['min_period']
            max_period = min(self.parameters['max_period'], len(optimization_data) // 3)
            
            period_scores = {}
            
            # Test periods with step size based on data length
            step_size = max(1, (max_period - min_period) // 20)
            test_periods = range(min_period, max_period + 1, step_size)
            
            for period in test_periods:
                score = self._evaluate_period_performance(optimization_data, period, variability)
                period_scores[period] = score
            
            # Find optimal period
            if period_scores:
                optimal_period = max(period_scores, key=period_scores.get)
                optimization_score = period_scores[optimal_period]
                
                # Calculate confidence level (how much better is optimal vs alternatives)
                scores = list(period_scores.values())
                if len(scores) > 1:
                    confidence = (optimization_score - np.mean(scores)) / (np.std(scores) + 1e-10)
                    confidence_level = min(1.0, max(0.0, confidence / 2))
                else:
                    confidence_level = 0.5
                
                # Stability metric (consistency across similar periods)
                nearby_periods = [p for p in period_scores.keys() ]
[                                if abs(p - optimal_period) <= 3]:
                nearby_scores = [period_scores[p] for p in nearby_periods]
                stability_metric = 1.0 - (np.std(nearby_scores) / (np.mean(nearby_scores) + 1e-10))
                stability_metric = max(0.0, min(1.0, stability_metric))
                
            else:
                optimal_period = self.parameters['base_period']
                optimization_score = 0.0
                confidence_level = 0.0
                stability_metric = 0.0
            
            return PeriodOptimization()
                optimal_period=optimal_period,
                optimization_score=optimization_score,
                confidence_level=confidence_level,
                stability_metric=stability_metric,
                performance_history=list(period_scores.values())
(            )
            
        except Exception as e:
            logger.error(f"Error optimizing period: {str(e)}")
            return PeriodOptimization()
                optimal_period=self.parameters['base_period'],
                optimization_score=0.0,
                confidence_level=0.0,
                stability_metric=0.0,
                performance_history=[]
(            )
    
def _evaluate_period_performance(self, data: pd.DataFrame, period: int,:)
(                                   variability: VariabilityMetrics) -> float:
        """Evaluate the performance of a specific MA period"""
        try:
            if len(data) < period + 10:
                return 0.0
            
            # Calculate moving average
            ma = data['close'].rolling(window=period).mean()
            
            # Performance metrics
            performance_factors = []
            
            # 1. Trend following performance
            ma_signals = np.where(data['close'] > ma, 1, -1)
            returns = data['close'].pct_change().fillna(0)
            
            # Align signals and returns
            aligned_signals = ma_signals[:-1]
            aligned_returns = returns.iloc[1:].values
            
            if len(aligned_signals) > 0 and len(aligned_returns) > 0:
                strategy_returns = aligned_signals * aligned_returns
                trend_performance = np.mean(strategy_returns)
                performance_factors.append(trend_performance * 100)
            
            # 2. Smoothness (penalize excessive noise)
            ma_changes = abs(ma.diff()).mean()
            price_changes = abs(data['close'].diff()).mean()
            smoothness = 1.0 - (ma_changes / (price_changes + 1e-10))
            performance_factors.append(max(0, smoothness))
            
            # 3. Responsiveness (ability to capture significant moves)
            significant_moves = abs(data['close'].pct_change()) > 0.02
            if significant_moves.any():
                move_data = data[significant_moves]
                move_ma = ma[significant_moves]
                
                # How quickly MA responds to significant moves
                response_lag = abs(move_data['close'] - move_ma).mean() / move_data['close'].mean()
                responsiveness = 1.0 / (1.0 + response_lag * 10)
                performance_factors.append(responsiveness)
            
            # 4. Regime-specific performance
            if variability.volatility_regime == 'low':
                # In low volatility, prefer longer periods (more smoothing)
                regime_bonus = 1.0 - (period / self.parameters['max_period'])
            elif variability.volatility_regime == 'high':
                # In high volatility, prefer shorter periods (more responsive)
                regime_bonus = period / self.parameters['max_period']
            else:
                # Medium volatility - balance
                optimal_range = (self.parameters['min_period'] + self.parameters['max_period']) / 2
                regime_bonus = 1.0 - abs(period - optimal_range) / optimal_range
            
            performance_factors.append(regime_bonus)
            
            # 5. Efficiency bonus (works better in trending markets)
            if variability.trend_strength > 0.5:
                efficiency_bonus = variability.market_efficiency
                performance_factors.append(efficiency_bonus)
            
            # Combine all factors
            total_performance = np.mean(performance_factors) if performance_factors else 0.0
            
            return max(0.0, total_performance)
            
        except Exception as e:
            logger.error(f"Error evaluating period performance: {str(e)}")
            return 0.0
    
def _calculate_variable_ma(self, data: pd.DataFrame, period: int,:)
(                             method: str = 'adaptive') -> pd.Series:
        """Calculate variable moving average with specified method"""
        try:
            if len(data) < period:
                return pd.Series(index=data.index, data=np.nan)
            
            prices = data['close']
            
            if method == 'sma':
                return prices.rolling(window=period).mean()
            
            elif method == 'ema':
                return prices.ewm(span=period).mean()
            
            elif method == 'wma':
                # Weighted moving average
                weights = np.arange(1, period + 1)
                
def wma_calc(x):
                    if len(x) < period:
                        return np.nan
                    return np.sum(weights * x[-period:]) / np.sum(weights)
                
                return prices.rolling(window=period).apply(wma_calc, raw=True)
            
            elif method == 'hull':
                # Hull moving average
                n_half = period // 2
                n_sqrt = int(np.sqrt(period))
                
                wma_half = prices.rolling(window=n_half).apply()
                    lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), 
                    raw=True
(                )
                wma_full = prices.rolling(window=period).apply()
                    lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), 
                    raw=True
(                )
                
                hull_raw = 2 * wma_half - wma_full
                hull_ma = hull_raw.rolling(window=n_sqrt).apply()
                    lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1)), 
                    raw=True
(                )
                
                return hull_ma
            
            elif method == 'adaptive':
                # Adaptive MA that changes period based on volatility
                adaptive_ma = pd.Series(index=data.index, dtype=float)
                
                for i in range(period, len(data)):
                    # Calculate local volatility
                    local_returns = prices.iloc[i-period:i].pct_change().dropna()
                    local_vol = local_returns.std() if len(local_returns) > 1 else 0.01
                    
                    # Adjust period based on volatility
                    vol_factor = min(2.0, max(0.5, 1.0 / (local_vol * 50 + 0.1)))
                    adjusted_period = int(period * vol_factor)
                    adjusted_period = max(3, min(period * 2, adjusted_period))
                    
                    # Calculate MA with adjusted period
                    if i >= adjusted_period:
                        adaptive_ma.iloc[i] = prices.iloc[i-adjusted_period:i].mean()
                
                return adaptive_ma
            
            else:
                # Default to SMA
                return prices.rolling(window=period).mean()
                
        except Exception as e:
            logger.error(f"Error calculating variable MA: {str(e)}")
            return pd.Series(index=data.index, data=np.nan)
    
def _update_adaptive_period(self, optimal_period: int, variability: VariabilityMetrics) -> int:
        """Update current period with momentum and constraints"""
        try:
            adaptation_speed = self.parameters['adaptation_speed']
            
            # Apply adaptation speed to smooth period changes
            period_change = optimal_period - self.current_period
            
            # Apply momentum to prevent erratic changes
            self.adaptation_momentum = ()
                self.adaptation_momentum * 0.7 + 
                period_change * 0.3
(            )
            
            # Calculate new period with momentum
            new_period = self.current_period + (self.adaptation_momentum * adaptation_speed)
            
            # Apply variability-based adjustment
            new_period *= variability.adaptive_factor
            
            # Ensure within bounds
            new_period = max(self.parameters['min_period'], )
(                           min(self.parameters['max_period'], new_period))
            
            return int(round(new_period))
            
        except Exception as e:
            logger.error(f"Error updating adaptive period: {str(e)}")
            return self.current_period    
def _calculate_multi_timeframe_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate signals across multiple timeframes"""
        try:
            if not self.parameters['multi_timeframe_enabled'] or len(data) < 100:
                return {'short_term': 'neutral', 'medium_term': 'neutral', 'long_term': 'neutral'}
            
            signals = {}
            
            # Define timeframe periods
            timeframes = {
                'short_term': self.current_period,
                'medium_term': self.current_period * 2,
                'long_term': self.current_period * 4
            }
            
            current_price = data['close'].iloc[-1]
            
            for timeframe, period in timeframes.items():
                if len(data) >= period:
                    ma = self._calculate_variable_ma(data, period, 'adaptive')
                    
                    if not ma.isna().all():
                        current_ma = ma.iloc[-1]
                        
                        # Determine signal
                        if current_price > current_ma * 1.01:  # 1% threshold:
                            signals[timeframe] = 'bullish'
                        elif current_price < current_ma * 0.99:
                            signals[timeframe] = 'bearish'
                        else:
                            signals[timeframe] = 'neutral'
                    else:
                        signals[timeframe] = 'neutral'
                else:
                    signals[timeframe] = 'neutral'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating multi-timeframe signals: {str(e)}")
            return {'short_term': 'neutral', 'medium_term': 'neutral', 'long_term': 'neutral'}
    
def _analyze_ma_quality(self, data: pd.DataFrame, ma: pd.Series,:)
(                           period: int) -> Dict[str, float]:
        """Analyze the quality of the moving average"""
        try:
            quality_metrics = {}
            
            if len(ma) < 20:
                return {'smoothness': 0.0, 'responsiveness': 0.0, 'stability': 0.0, 'overall': 0.0}
            
            # 1. Smoothness - how smooth is the MA
            ma_changes = ma.diff().dropna()
            price_changes = data['close'].diff().dropna()
            
            if len(ma_changes) > 0 and len(price_changes) > 0:
                smoothness = 1.0 - (ma_changes.std() / (price_changes.std() + 1e-10))
                smoothness = max(0.0, min(1.0, smoothness))
            else:
                smoothness = 0.0
            
            quality_metrics['smoothness'] = smoothness
            
            # 2. Responsiveness - how quickly MA responds to price changes
            significant_moves = abs(data['close'].pct_change()) > 0.02
            if significant_moves.any():
                responsive_data = data[significant_moves]
                responsive_ma = ma[significant_moves]
                
                # Correlation between price moves and MA moves
                if len(responsive_data) > 3:
                    price_momentum = responsive_data['close'].pct_change()
                    ma_momentum = responsive_ma.pct_change()
                    responsiveness = abs(price_momentum.corr(ma_momentum))
                    responsiveness = responsiveness if not np.isnan(responsiveness) else 0.0
                else:
                    responsiveness = 0.0
            else:
                responsiveness = 0.5
            
            quality_metrics['responsiveness'] = responsiveness
            
            # 3. Stability - how stable is the period selection
            period_variance = np.var(self.period_history[-20:]) if len(self.period_history) >= 20 else 0
            max_variance = (self.parameters['max_period'] - self.parameters['min_period']) ** 2 / 4
            stability = 1.0 - (period_variance / max_variance) if max_variance > 0 else 1.0
            stability = max(0.0, min(1.0, stability))
            
            quality_metrics['stability'] = stability
            
            # 4. Trend alignment - how well MA aligns with overall trend
            if len(data) >= 50:
                long_term_trend = (data['close'].iloc[-1] - data['close'].iloc[-50]) / data['close'].iloc[-50]
                ma_trend = (ma.iloc[-1] - ma.iloc[-20]) / ma.iloc[-20] if len(ma) >= 20 else 0
                
                trend_alignment = 1.0 - abs(np.sign(long_term_trend) - np.sign(ma_trend)) / 2
            else:
                trend_alignment = 0.5
            
            quality_metrics['trend_alignment'] = trend_alignment
            
            # Overall quality score
            weights = {'smoothness': 0.25, 'responsiveness': 0.25, 'stability': 0.25, 'trend_alignment': 0.25}
            overall_quality = sum(quality_metrics[metric] * weights[metric] for metric in weights)
            quality_metrics['overall'] = overall_quality
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing MA quality: {str(e)}")
            return {'smoothness': 0.0, 'responsiveness': 0.0, 'stability': 0.0, 'overall': 0.0}
    
def _detect_trading_modes(self, data: pd.DataFrame, ma: pd.Series,:)
(                            variability: VariabilityMetrics) -> Tuple[str, str]:
        """Detect optimal trading modes based on market conditions"""
        try:
            # Trend following mode detection
            if variability.trend_strength > 0.6 and variability.market_efficiency > 0.7:
                trend_mode = 'strong_trend_following'
            elif variability.trend_strength > 0.3:
                trend_mode = 'weak_trend_following'
            else:
                trend_mode = 'no_trend'
            
            # Mean reversion mode detection
            current_price = data['close'].iloc[-1]
            current_ma = ma.iloc[-1] if not np.isnan(ma.iloc[-1]) else current_price
            
            # Calculate deviation from MA
            price_deviation = abs(current_price - current_ma) / current_ma
            
            # Historical deviation analysis
            if len(data) >= 50:
                historical_deviations = abs((data['close'] - ma) / ma).dropna()
                if len(historical_deviations) > 0:
                    deviation_percentile = stats.percentileofscore(historical_deviations, price_deviation) / 100
                    
                    if deviation_percentile > 0.8:  # Extreme deviation:
                        if variability.noise_level < 0.5:  # Low noise environment:
                            mean_reversion_mode = 'strong_mean_reversion'
                        else:
                            mean_reversion_mode = 'weak_mean_reversion'
                    elif deviation_percentile > 0.6:
                        mean_reversion_mode = 'moderate_mean_reversion'
                    else:
                        mean_reversion_mode = 'no_mean_reversion'
                else:
                    mean_reversion_mode = 'no_mean_reversion'
            else:
                mean_reversion_mode = 'no_mean_reversion'
            
            return trend_mode, mean_reversion_mode
            
        except Exception as e:
            logger.error(f"Error detecting trading modes: {str(e)}")
            return 'no_trend', 'no_mean_reversion'
    
def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive variable moving average analysis
        """
        try:
            if len(data) < self.parameters['base_period']:
                return {
                    'variable_ma': 0.0,
                    'current_period': self.parameters['base_period'],
                    'variability_metrics': {},
                    'optimization_results': {},
                    'multi_timeframe_signals': {},
                    'quality_metrics': {},
                    'trading_modes': {},
                    'adaptation_speed': 0.0
                }
            
            # Calculate market variability
            variability = self._calculate_market_variability(data)
            self.variability_history.append(variability)
            
            # Keep variability history manageable
            if len(self.variability_history) > 100:
                self.variability_history = self.variability_history[-100:]
            
            # Optimize period based on current conditions
            optimization = self._optimize_period(data, variability)
            self.optimization_results = optimization
            
            # Update adaptive period
            new_period = self._update_adaptive_period(optimization.optimal_period, variability)
            
            # Apply noise filter if enabled
            if self.parameters['noise_filter_enabled']:
                # Smooth period changes to reduce noise
                if len(self.period_history) > 0:
                    period_change = abs(new_period - self.current_period)
                    max_change = max(1, self.current_period * 0.2)  # Max 20% change per update
                    
                    if period_change > max_change:
                        if new_period > self.current_period:
                            new_period = self.current_period + max_change
                        else:
                            new_period = self.current_period - max_change
                        
                        new_period = int(new_period)
            
            self.current_period = new_period
            self.period_history.append(new_period)
            
            # Keep period history manageable
            if len(self.period_history) > 200:
                self.period_history = self.period_history[-200:]
            
            # Calculate variable moving average
            ma_method = 'adaptive'  # Could be made configurable
            variable_ma = self._calculate_variable_ma(data, self.current_period, ma_method)
            current_ma = variable_ma.iloc[-1] if not variable_ma.isna().all() else 0.0
            
            # Multi-timeframe analysis
            mtf_signals = self._calculate_multi_timeframe_signals(data)
            
            # Quality analysis
            quality_metrics = self._analyze_ma_quality(data, variable_ma, self.current_period)
            
            # Trading mode detection
            trend_mode, mean_reversion_mode = self._detect_trading_modes(data, variable_ma, variability)
            
            # Calculate adaptation speed
            if len(self.period_history) >= 2:
                recent_changes = [abs(self.period_history[i] - self.period_history[i-1]) ]
[                                for i in range(-5, 0) if i < 0 and abs(i) <= len(self.period_history)]:
                adaptation_speed = np.mean(recent_changes) if recent_changes else 0.0
            else:
                adaptation_speed = 0.0
            
            # Performance tracking
            if len(data) >= self.parameters['performance_window']:
                performance_score = self._calculate_current_performance(data, variable_ma)
                self.performance_history.append(performance_score)
                
                # Keep performance history manageable
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-100:]
            
            # Advanced analytics
            volatility_forecast = self._forecast_volatility(data)
            period_forecast = self._forecast_optimal_period(data, variability)
            regime_stability = self._calculate_regime_stability()
            
            # Signal synthesis
            signal_synthesis = self._synthesize_signals(data, variable_ma, variability, mtf_signals)
            
            result = {
                'variable_ma': current_ma,
                'current_period': self.current_period,
                'variability_metrics': self._variability_to_dict(variability),
                'optimization_results': self._optimization_to_dict(optimization),
                'multi_timeframe_signals': mtf_signals,
                'quality_metrics': quality_metrics,
                'trading_modes': {
                    'trend_following': trend_mode,
                    'mean_reversion': mean_reversion_mode
                },
                'adaptation_speed': adaptation_speed,
                'volatility_forecast': volatility_forecast,
                'period_forecast': period_forecast,
                'regime_stability': regime_stability,
                'signal_synthesis': signal_synthesis,
                'ma_slope': variable_ma.diff().iloc[-1] if len(variable_ma) > 1 else 0.0,
                'ma_acceleration': variable_ma.diff().diff().iloc[-1] if len(variable_ma) > 2 else 0.0,
                'period_efficiency': optimization.optimization_score,
                'adaptation_momentum': self.adaptation_momentum,
                'performance_score': self.performance_history[-1] if self.performance_history else 0.0,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in variable MA calculation: {str(e)}")
            raise IndicatorCalculationError()
                indicator_name=self.name,
                calculation_step="variable_ma_calculation",
                message=str(e)
(            )
    
def _calculate_current_performance(self, data: pd.DataFrame, ma: pd.Series) -> float:
        """Calculate current performance of the variable MA"""
        try:
            window = self.parameters['performance_window']
            if len(data) < window or len(ma) < window:
                return 0.0
            
            recent_data = data.tail(window)
            recent_ma = ma.tail(window)
            
            # Generate signals
            signals = np.where(recent_data['close'] > recent_ma, 1, -1)
            returns = recent_data['close'].pct_change().fillna(0)
            
            # Calculate strategy returns
            strategy_returns = signals[:-1] * returns.iloc[1:].values
            
            # Performance metrics
            total_return = np.sum(strategy_returns)
            win_rate = (strategy_returns > 0).mean()
            
            # Risk-adjusted performance
            volatility = strategy_returns.std()
            sharpe_ratio = np.mean(strategy_returns) / (volatility + 1e-10)
            
            # Combined performance score
            performance = total_return + sharpe_ratio * 0.1 + win_rate * 0.1
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating current performance: {str(e)}")
            return 0.0
    
def _forecast_volatility(self, data: pd.DataFrame) -> Dict[str, float]:
        """Forecast short-term volatility"""
        try:
            if len(data) < 50:
                return {'forecast': 0.02, 'confidence': 0.0, 'trend': 'stable'}
            
            returns = data['close'].pct_change().dropna()
            
            # Simple GARCH-like volatility forecasting
            recent_vol = returns.tail(20).std()
            medium_vol = returns.tail(50).std()
            
            # Volatility trend
            if recent_vol > medium_vol * 1.1:
                vol_trend = 'increasing'
            elif recent_vol < medium_vol * 0.9:
                vol_trend = 'decreasing'
            else:
                vol_trend = 'stable'
            
            # Simple forecast (mean reversion)
            long_term_vol = returns.std()
            forecast = 0.7 * recent_vol + 0.3 * long_term_vol
            
            # Confidence based on consistency
            vol_history = [returns.iloc[i:i+10].std() for i in range(len(returns)-30, len(returns)-10, 5)]
            vol_consistency = 1.0 - (np.std(vol_history) / np.mean(vol_history)) if vol_history else 0.0
            
            return {
                'forecast': forecast,
                'confidence': max(0.0, min(1.0, vol_consistency)),
                'trend': vol_trend
            }
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {str(e)}")
            return {'forecast': 0.02, 'confidence': 0.0, 'trend': 'stable'}
    
def _forecast_optimal_period(self, data: pd.DataFrame,:)
(                               variability: VariabilityMetrics) -> Dict[str, Any]:
        """Forecast optimal period for next few periods"""
        try:
            if len(self.period_history) < 10:
                return {'forecast': self.current_period, 'confidence': 0.0, 'direction': 'stable'}
            
            # Trend in period changes
            recent_periods = self.period_history[-10:]
            period_trend = np.polyfit(range(len(recent_periods)), recent_periods, 1)[0]
            
            # Forecast next period
            if abs(period_trend) < 0.5:
                period_direction = 'stable'
                forecast_period = self.current_period
            elif period_trend > 0:
                period_direction = 'increasing'
                forecast_period = min(self.parameters['max_period'], )
(                                    self.current_period + abs(period_trend))
            else:
                period_direction = 'decreasing'
                forecast_period = max(self.parameters['min_period'], )
(                                    self.current_period - abs(period_trend))
            
            # Confidence based on trend consistency
            period_changes = np.diff(recent_periods)
            change_consistency = 1.0 - (np.std(period_changes) / (np.mean(np.abs(period_changes)) + 1e-10))
            confidence = max(0.0, min(1.0, change_consistency))
            
            return {
                'forecast': int(forecast_period),
                'confidence': confidence,
                'direction': period_direction,
                'trend_strength': abs(period_trend)
            }
            
        except Exception as e:
            logger.error(f"Error forecasting optimal period: {str(e)}")
            return {'forecast': self.current_period, 'confidence': 0.0, 'direction': 'stable'}
    
def _calculate_regime_stability(self) -> float:
        """Calculate stability of current market regime"""
        try:
            if len(self.variability_history) < 10:
                return 0.5
            
            # Analyze consistency of market regimes
            recent_regimes = [v.volatility_regime for v in self.variability_history[-10:]]
            
            # Count regime changes
            regime_changes = sum(1 for i in range(1, len(recent_regimes)) )
(                               if recent_regimes[i] != recent_regimes[i-1]):
            
            # Stability = 1 - (changes / max_possible_changes)
            stability = 1.0 - (regime_changes / max(1, len(recent_regimes) - 1))
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            logger.error(f"Error calculating regime stability: {str(e)}")
            return 0.5
    
def _synthesize_signals(self, data: pd.DataFrame, ma: pd.Series,:)
                          variability: VariabilityMetrics, 
(                          mtf_signals: Dict[str, str]) -> Dict[str, Any]:
        """Synthesize signals from all components"""
        try:
            current_price = data['close'].iloc[-1]
            current_ma = ma.iloc[-1] if not np.isnan(ma.iloc[-1]) else current_price
            
            # Price-MA relationship
            price_position = 'above' if current_price > current_ma else 'below'
            price_distance = abs(current_price - current_ma) / current_ma
            
            # Multi-timeframe consensus
            bullish_timeframes = sum(1 for signal in mtf_signals.values() if signal == 'bullish')
            bearish_timeframes = sum(1 for signal in mtf_signals.values() if signal == 'bearish')
            total_timeframes = len(mtf_signals)
            
            mtf_consensus = 'bullish' if bullish_timeframes > bearish_timeframes else 'bearish'
            mtf_strength = max(bullish_timeframes, bearish_timeframes) / total_timeframes
            
            # Volatility-adjusted signals
            if variability.volatility_regime == 'high':
                volatility_bias = 'reduce_exposure'
            elif variability.volatility_regime == 'low':
                volatility_bias = 'increase_exposure'
            else:
                volatility_bias = 'neutral'
            
            # Trend strength consideration
            if variability.trend_strength > 0.7:
                trend_bias = 'trend_following'
            elif variability.trend_strength < 0.3:
                trend_bias = 'mean_reversion'
            else:
                trend_bias = 'neutral'
            
            return {
                'price_position': price_position,
                'price_distance': price_distance,
                'mtf_consensus': mtf_consensus,
                'mtf_strength': mtf_strength,
                'volatility_bias': volatility_bias,
                'trend_bias': trend_bias,
                'regime_favorable': variability.market_efficiency > 0.6
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing signals: {str(e)}")
            return {}
    
def _variability_to_dict(self, variability: VariabilityMetrics) -> Dict[str, Any]:
        """Convert VariabilityMetrics to dictionary"""
        return {
            'volatility_regime': variability.volatility_regime,
            'trend_strength': variability.trend_strength,
            'market_efficiency': variability.market_efficiency,
            'noise_level': variability.noise_level,
            'cycle_detection': variability.cycle_detection,
            'adaptive_factor': variability.adaptive_factor
        }
    
def _optimization_to_dict(self, optimization: PeriodOptimization) -> Dict[str, Any]:
        """Convert PeriodOptimization to dictionary"""
        return {
            'optimal_period': optimization.optimal_period,
            'optimization_score': optimization.optimization_score,
            'confidence_level': optimization.confidence_level,
            'stability_metric': optimization.stability_metric,
            'performance_range': {
                'min': min(optimization.performance_history) if optimization.performance_history else 0.0,
                'max': max(optimization.performance_history) if optimization.performance_history else 0.0,
                'mean': np.mean(optimization.performance_history) if optimization.performance_history else 0.0
            }
        }
    
def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on variable MA analysis
        """
        try:
            current_price = data['close'].iloc[-1]
            variable_ma = value.get('variable_ma', 0)
            quality_metrics = value.get('quality_metrics', {})
            mtf_signals = value.get('multi_timeframe_signals', {})
            trading_modes = value.get('trading_modes', {})
            signal_synthesis = value.get('signal_synthesis', {})
            variability_metrics = value.get('variability_metrics', {})
            
            if variable_ma == 0:
                return SignalType.NEUTRAL, 0.0
            
            # Basic MA signal
            price_above_ma = current_price > variable_ma
            price_distance = abs(current_price - variable_ma) / variable_ma
            
            # Initialize signal
            signal_strength = 0.0
            signal_type = SignalType.NEUTRAL
            
            # Primary signal from MA position
            if price_above_ma:
                signal_type = SignalType.BUY
                signal_strength += 0.3
            else:
                signal_type = SignalType.SELL
                signal_strength += 0.3
            
            # Multi-timeframe confirmation
            mtf_consensus = signal_synthesis.get('mtf_consensus', 'neutral')
            mtf_strength = signal_synthesis.get('mtf_strength', 0.0)
            
            if (mtf_consensus == 'bullish' and signal_type == SignalType.BUY) or \:
               (mtf_consensus == 'bearish' and signal_type == SignalType.SELL):
                signal_strength += mtf_strength * 0.3
            
            # Quality adjustment
            overall_quality = quality_metrics.get('overall', 0.0)
            signal_strength *= overall_quality
            
            # Trading mode consideration
            trend_mode = trading_modes.get('trend_following', 'no_trend')
            mean_reversion_mode = trading_modes.get('mean_reversion', 'no_mean_reversion')
            
            if trend_mode in ['strong_trend_following', 'weak_trend_following']:
                signal_strength += 0.2
            
            # Mean reversion signals (contrarian to basic signal)
            if mean_reversion_mode == 'strong_mean_reversion' and price_distance > 0.03:
                # Strong mean reversion suggests contrarian signal
                signal_type = SignalType.SELL if signal_type == SignalType.BUY else SignalType.BUY
                signal_strength += 0.2
            
            # Volatility regime adjustment
            volatility_bias = signal_synthesis.get('volatility_bias', 'neutral')
            if volatility_bias == 'reduce_exposure':
                signal_strength *= 0.7
            elif volatility_bias == 'increase_exposure':
                signal_strength *= 1.2
            
            # Market efficiency consideration
            market_efficiency = variability_metrics.get('market_efficiency', 0.0)
            if market_efficiency < 0.3:  # Inefficient market:
                signal_strength *= 0.8
            
            # Adaptation stability
            adaptation_speed = value.get('adaptation_speed', 0.0)
            if adaptation_speed > 5:  # Too much adaptation = less reliable:
                signal_strength *= 0.9
            
            # Ensure signal strength is within bounds
            signal_strength = min(1.0, max(0.0, signal_strength))
            
            # Minimum threshold for signal generation
            if signal_strength < 0.4:
                return SignalType.NEUTRAL, 0.0
            
            return signal_type, signal_strength
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0
    
def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        
        vma_metadata = {
            'base_period': self.parameters['base_period'],
            'current_period': self.current_period,
            'period_range': [self.parameters['min_period'], self.parameters['max_period']],
            'adaptation_speed': self.parameters['adaptation_speed'],
            'period_history_length': len(self.period_history),
            'performance_history_length': len(self.performance_history),
            'variability_history_length': len(self.variability_history),
            'noise_filter_enabled': self.parameters['noise_filter_enabled'],
            'multi_timeframe_enabled': self.parameters['multi_timeframe_enabled'],
            'cycle_analysis_enabled': self.parameters['cycle_analysis_enabled']
        }
        
        base_metadata.update(vma_metadata)
        return base_metadata


def create_variable_moving_average_indicator(parameters: Optional[Dict[str, Any]] = None) -> VariableMovingAverageIndicator:
    """
    Factory function to create a VariableMovingAverageIndicator instance
    
    Args:
        parameters: Optional dictionary of parameters to customize the indicator
        
    Returns:
        Configured VariableMovingAverageIndicator instance
    """
    return VariableMovingAverageIndicator(parameters=parameters)
def get_data_requirements(self):
        """
        Get data requirements for VariableMovingAverageIndicator.
        
        Returns:
            list: List of DataRequirement objects
        """
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import DataRequirement, DataType
        
        # Standard OHLCV requirements for most indicators
        return []
            DataRequirement()
                data_type=DataType.OHLCV,
                required_columns=['open', 'high', 'low', 'close', 'volume'],
                min_periods=20  # Reasonable default for most indicators
(            )
[        ]



# Example usage
if __name__ == "__main__":
    # Create sample data with varying market conditions
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    
    # Create realistic price data with different regimes
    base_price = 100
    n_periods = len(dates)
    
    # Generate price series with regime changes
    returns = []
    regime_length = 100
    
    for i in range(n_periods):
        regime = (i // regime_length) % 4
        
        if regime == 0:  # Low volatility trending:
            ret = np.random.normal(0.0001, 0.005)
        elif regime == 1:  # High volatility trending:
            ret = np.random.normal(0.0002, 0.02)
        elif regime == 2:  # Low volatility sideways:
            ret = np.random.normal(0, 0.003)
        else:  # High volatility sideways
            ret = np.random.normal(0, 0.025)
        
        returns.append(ret)
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate volume data correlated with volatility
    volumes = []
    for i, ret in enumerate(returns):
        vol_factor = 1 + abs(ret) * 50  # Higher volatility = higher volume
        volume = np.random.lognormal(10, 0.3) * vol_factor
        volumes.append(volume)
    
    sample_data = pd.DataFrame({)
        'high': prices * np.random.uniform(1.0001, 1.005, n_periods),
        'low': prices * np.random.uniform(0.995, 0.9999, n_periods),
        'close': prices,
        'volume': volumes
(    }, index=dates)
    
    # Test the indicator
    indicator = create_variable_moving_average_indicator({)
        'base_period': 24,  # 24 hours
        'min_period': 6,
        'max_period': 72,
        'adaptation_speed': 0.15,
        'multi_timeframe_enabled': True,
        'cycle_analysis_enabled': True
(    })
    
    try:
        result = indicator.calculate(sample_data)
        print("Variable Moving Average Analysis Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Current MA Value: {result.value.get('variable_ma', 0):.2f}")
        print(f"Current Period: {result.value.get('current_period', 0)}")
        print(f"Adaptation Speed: {result.value.get('adaptation_speed', 0):.2f}")
        
        # Display variability metrics
        variability = result.value.get('variability_metrics', {})
        print(f"\nMarket Variability:")
        print(f"Volatility Regime: {variability.get('volatility_regime', 'unknown')}")
        print(f"Trend Strength: {variability.get('trend_strength', 0):.3f}")
        print(f"Market Efficiency: {variability.get('market_efficiency', 0):.3f}")
        print(f"Noise Level: {variability.get('noise_level', 0):.3f}")
        
        # Display optimization results
        optimization = result.value.get('optimization_results', {})
        print(f"\nOptimization Results:")
        print(f"Optimal Period: {optimization.get('optimal_period', 0)}")
        print(f"Optimization Score: {optimization.get('optimization_score', 0):.3f}")
        print(f"Confidence Level: {optimization.get('confidence_level', 0):.3f}")
        
        # Display multi-timeframe signals
        mtf_signals = result.value.get('multi_timeframe_signals', {})
        print(f"\nMulti-Timeframe Signals:")
        for timeframe, signal in mtf_signals.items():
            print(f"{timeframe}: {signal}")
        
        # Display quality metrics
        quality = result.value.get('quality_metrics', {})
        print(f"\nQuality Metrics:")
        print(f"Overall Quality: {quality.get('overall', 0):.3f}")
        print(f"Smoothness: {quality.get('smoothness', 0):.3f}")
        print(f"Responsiveness: {quality.get('responsiveness', 0):.3f}")
        print(f"Stability: {quality.get('stability', 0):.3f}")
        
    except Exception as e:
        print(f"Error testing indicator: {str(e)}")