"""
Triangular Moving Average Indicator

Implements a sophisticated triangular moving average system with multiple smoothing
techniques, adaptive periods, and advanced trend analysis capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

from auj_platform.src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import IndicatorCalculationError

logger = logging.getLogger(__name__)


@dataclass
class TriangularMASignal:
    """Data structure for triangular MA signal analysis"""
    ma_value: float
    trend_direction: str
    trend_strength: float
    smoothness_factor: float
    adaptive_period: int
    crossover_signal: str
    momentum_alignment: float


@dataclass
class TrendAnalysis:
    """Data structure for comprehensive trend analysis"""
    short_term_trend: str
    medium_term_trend: str
    long_term_trend: str
    trend_consistency: float
    trend_velocity: float
    trend_acceleration: float
    reversal_probability: float


@dataclass
class AdaptiveParameters:
    """Data structure for adaptive parameter management"""
    current_period: int
    volatility_adjustment: float
    market_regime: str
    efficiency_score: float
    optimization_confidence: float


class TriangularMovingAverageIndicator(StandardIndicatorInterface):
    """
    Advanced Triangular Moving Average Indicator
    
    This indicator provides sophisticated triangular moving average analysis with
    adaptive periods, multiple smoothing techniques, and comprehensive trend analysis.
    The triangular MA applies double smoothing for reduced lag and noise.
    """
    
def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'base_period': 20,
            'adaptive_enabled': True,
            'volatility_lookback': 50,
            'min_period': 10,
            'max_period': 50,
            'smoothing_factor': 2.0,
            'trend_sensitivity': 0.02,
            'crossover_periods': [10, 20, 50],
            'momentum_periods': [5, 10, 20],
            'optimization_window': 100,
            'reversal_threshold': 0.75,
            'trend_confirmation_bars': 3
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="Triangular Moving Average")
        
        # State tracking
        self.ma_history = []
        self.trend_history = []
        self.adaptive_params = None
        self.crossover_signals = []
        
def _calculate_triangular_ma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate triangular moving average with double smoothing"""
        try:
            if len(data) < period:
                return pd.Series(index=data.index, data=np.nan)
            
            # First smoothing - simple moving average
            first_ma = data.rolling(window=period).mean()
            
            # Second smoothing - moving average of the first MA
            # Period for second smoothing is half of the original (rounded up)
            second_period = max(1, (period + 1) // 2)
            triangular_ma = first_ma.rolling(window=second_period).mean()
            
            return triangular_ma
            
        except Exception as e:
            logger.error(f"Error calculating triangular MA: {str(e)}")
            return pd.Series(index=data.index, data=np.nan)
    
def _calculate_adaptive_period(self, data: pd.DataFrame) -> int:
        """Calculate adaptive period based on market volatility and efficiency"""
        try:
            if len(data) < self.parameters['volatility_lookback']:
                return self.parameters['base_period']
            
            lookback = self.parameters['volatility_lookback']
            recent_data = data.tail(lookback)
            
            # Calculate market volatility
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate price efficiency (trending vs sideways)
            price_change = abs(recent_data['close'].iloc[-1] - recent_data['close'].iloc[0])
            price_path = abs(recent_data['close'].diff()).sum()
            efficiency = price_change / price_path if price_path > 0 else 0
            
            # Calculate volume-weighted efficiency
            volume_weight = recent_data['volume'].corr(abs(recent_data['close'].diff()))
            volume_weight = max(0, volume_weight) if not np.isnan(volume_weight) else 0
            
            # Adaptive period calculation
            base_period = self.parameters['base_period']
            min_period = self.parameters['min_period']
            max_period = self.parameters['max_period']
            
            # Higher volatility -> shorter period (more responsive)
            volatility_factor = np.exp(-volatility * 50)  # Scale volatility
            
            # Higher efficiency -> longer period (trend following)
            efficiency_factor = 1 + efficiency
            
            # Volume confirmation factor
            volume_factor = 1 + volume_weight * 0.5
            
            # Combined adaptive period
            adaptive_period = int(base_period * volatility_factor * efficiency_factor * volume_factor)
            adaptive_period = max(min_period, min(max_period, adaptive_period))
            
            return adaptive_period
            
        except Exception as e:
            logger.error(f"Error calculating adaptive period: {str(e)}")
            return self.parameters['base_period']
    
def _analyze_trend_characteristics(self, ma_series: pd.Series,:)
(                                     price_data: pd.DataFrame) -> TrendAnalysis:
        """Analyze comprehensive trend characteristics"""
        try:
            if len(ma_series) < 20:
                return TrendAnalysis()
                    short_term_trend='neutral',
                    medium_term_trend='neutral',
                    long_term_trend='neutral',
                    trend_consistency=0.0,
                    trend_velocity=0.0,
                    trend_acceleration=0.0,
                    reversal_probability=0.5
(                )
            
            # Calculate trend directions for different timeframes
            short_term = ma_series.tail(5)
            medium_term = ma_series.tail(15)
            long_term = ma_series.tail(30) if len(ma_series) >= 30 else ma_series
            
            # Trend direction analysis
            short_trend = self._determine_trend_direction(short_term)
            medium_trend = self._determine_trend_direction(medium_term)
            long_trend = self._determine_trend_direction(long_term)
            
            # Trend consistency (how often direction changes)
            ma_diff = ma_series.diff().dropna()
            direction_changes = (ma_diff[:-1] * ma_diff[1:] < 0).sum()
            consistency = 1 - (direction_changes / max(1, len(ma_diff) - 1))
            
            # Trend velocity (rate of change)
            recent_velocity = ma_series.pct_change(periods=3).tail(5).mean()
            
            # Trend acceleration (change in velocity)
            velocity_series = ma_series.pct_change(periods=3)
            acceleration = velocity_series.diff().tail(3).mean()
            
            # Reversal probability
            reversal_prob = self._calculate_reversal_probability(ma_series, price_data)
            
            return TrendAnalysis()
                short_term_trend=short_trend,
                medium_term_trend=medium_trend,
                long_term_trend=long_trend,
                trend_consistency=max(0, min(1, consistency)),
                trend_velocity=recent_velocity,
                trend_acceleration=acceleration,
                reversal_probability=reversal_prob
(            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend characteristics: {str(e)}")
            return TrendAnalysis()
                short_term_trend='neutral',
                medium_term_trend='neutral', 
                long_term_trend='neutral',
                trend_consistency=0.0,
                trend_velocity=0.0,
                trend_acceleration=0.0,
                reversal_probability=0.5
(            )
    
def _determine_trend_direction(self, series: pd.Series) -> str:
        """Determine trend direction from series"""
        try:
            if len(series) < 2:
                return 'neutral'
            
            # Linear regression slope
            x = np.arange(len(series))
            y = series.values
            slope, _, r_value, _, _ = stats.linregress(x, y)
            
            # Minimum slope threshold
            threshold = self.parameters['trend_sensitivity']
            relative_slope = slope / series.mean() if series.mean() != 0 else 0
            
            # Consider R-squared for trend strength
            if abs(relative_slope) < threshold or r_value**2 < 0.5:
                return 'neutral'
            elif relative_slope > 0:
                return 'bullish'
            else:
                return 'bearish'
                
        except Exception as e:
            logger.error(f"Error determining trend direction: {str(e)}")
            return 'neutral'
    
def _calculate_reversal_probability(self, ma_series: pd.Series,:)
(                                      price_data: pd.DataFrame) -> float:
        """Calculate probability of trend reversal"""
        try:
            if len(ma_series) < 20:
                return 0.5
            
            # Factors indicating potential reversal
            reversal_factors = []
            
            # 1. Momentum divergence
            ma_momentum = ma_series.pct_change(periods=5)
            price_momentum = price_data['close'].pct_change(periods=5)
            
            recent_ma_mom = ma_momentum.tail(3).mean()
            recent_price_mom = price_momentum.tail(3).mean()
            
            if (recent_ma_mom > 0 and recent_price_mom < 0) or (recent_ma_mom < 0 and recent_price_mom > 0):
                reversal_factors.append(0.3)
            
            # 2. Overbought/oversold conditions relative to MA
            current_price = price_data['close'].iloc[-1]
            current_ma = ma_series.iloc[-1]
            ma_deviation = (current_price - current_ma) / current_ma
            
            # Historical deviation analysis
            historical_deviations = ((price_data['close'] - ma_series) / ma_series).dropna()
            deviation_std = historical_deviations.std()
            
            if abs(ma_deviation) > 2 * deviation_std:
                reversal_factors.append(0.4)
            
            # 3. Volume confirmation
            if 'volume' in price_data.columns:
                recent_volume = price_data['volume'].tail(5).mean()
                avg_volume = price_data['volume'].tail(20).mean()
                
                if recent_volume > 1.5 * avg_volume:  # High volume might confirm reversal:
                    reversal_factors.append(0.2)
            
            # 4. MA slope flattening
            ma_slope = ma_series.diff().tail(5)
            slope_change = ma_slope.std()
            avg_slope_change = ma_series.diff().tail(20).std()
            
            if slope_change < 0.5 * avg_slope_change:  # Slope flattening:
                reversal_factors.append(0.1)
            
            # Combine factors
            reversal_probability = min(1.0, sum(reversal_factors))
            
            return reversal_probability
            
        except Exception as e:
            logger.error(f"Error calculating reversal probability: {str(e)}")
            return 0.5
    
def _generate_crossover_signals(self, data: pd.DataFrame,:)
(                                   main_ma: pd.Series) -> List[Dict[str, Any]]:
        """Generate crossover signals with multiple timeframes"""
        try:
            signals = []
            crossover_periods = self.parameters['crossover_periods']
            
            for period in crossover_periods:
                if len(data) < period + 10:
                    continue
                
                # Calculate comparison MA
                comparison_ma = self._calculate_triangular_ma(data['close'], period)
                
                # Find crossovers
                if len(main_ma) >= 2 and len(comparison_ma) >= 2:
                    # Current and previous values
                    main_current = main_ma.iloc[-1]
                    main_previous = main_ma.iloc[-2]
                    comp_current = comparison_ma.iloc[-1]
                    comp_previous = comparison_ma.iloc[-2]
                    
                    # Detect crossover
                    bullish_cross = (main_previous <= comp_previous and main_current > comp_current)
                    bearish_cross = (main_previous >= comp_previous and main_current < comp_current)
                    
                    if bullish_cross or bearish_cross:
                        signal_type = 'bullish_crossover' if bullish_cross else 'bearish_crossover'
                        
                        # Calculate signal strength
                        ma_distance = abs(main_current - comp_current) / comp_current
                        signal_strength = min(1.0, ma_distance * 100)
                        
                        signals.append({)
                            'type': signal_type,
                            'period': period,
                            'strength': signal_strength,
                            'main_ma': main_current,
                            'comparison_ma': comp_current
(                        })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating crossover signals: {str(e)}")
            return []
    
def _calculate_momentum_alignment(self, data: pd.DataFrame,:)
(                                    ma_series: pd.Series) -> float:
        """Calculate momentum alignment with triangular MA"""
        try:
            if len(data) < 20 or len(ma_series) < 10:
                return 0.0
            
            momentum_periods = self.parameters['momentum_periods']
            alignments = []
            
            for period in momentum_periods:
                if len(data) < period:
                    continue
                
                # Price momentum
                price_momentum = data['close'].pct_change(periods=period).iloc[-1]
                
                # MA momentum
                ma_momentum = ma_series.pct_change(periods=period).iloc[-1]
                
                # Alignment (same direction = positive, opposite = negative)
                if not np.isnan(price_momentum) and not np.isnan(ma_momentum):
                    alignment = np.sign(price_momentum) * np.sign(ma_momentum)
                    alignments.append(alignment)
            
            # Average alignment across all periods
            return np.mean(alignments) if alignments else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating momentum alignment: {str(e)}")
            return 0.0
    
def _optimize_parameters(self, data: pd.DataFrame) -> AdaptiveParameters:
        """Optimize parameters based on recent performance"""
        try:
            if len(data) < self.parameters['optimization_window']:
                return AdaptiveParameters()
                    current_period=self.parameters['base_period'],
                    volatility_adjustment=1.0,
                    market_regime='unknown',
                    efficiency_score=0.0,
                    optimization_confidence=0.0
(                )
            
            window = self.parameters['optimization_window']
            optimization_data = data.tail(window)
            
            # Test different periods for best performance
            best_period = self.parameters['base_period']
            best_score = 0
            
            test_periods = range(self.parameters['min_period'], )
(                               self.parameters['max_period'] + 1, 2)
            
            for test_period in test_periods:
                ma = self._calculate_triangular_ma(optimization_data['close'], test_period)
                
                if ma.isna().all():
                    continue
                
                # Calculate performance score
                score = self._calculate_ma_performance_score(optimization_data, ma)
                
                if score > best_score:
                    best_score = score
                    best_period = test_period
            
            # Market regime detection
            volatility = optimization_data['close'].pct_change().std()
            regime = self._detect_market_regime(optimization_data)
            
            # Efficiency score
            efficiency = self._calculate_trend_efficiency(optimization_data)
            
            # Confidence in optimization
            confidence = min(1.0, best_score) if best_score > 0 else 0.0
            
            return AdaptiveParameters()
                current_period=best_period,
                volatility_adjustment=1.0 / (1.0 + volatility * 10),
                market_regime=regime,
                efficiency_score=efficiency,
                optimization_confidence=confidence
(            )
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}")
            return AdaptiveParameters()
                current_period=self.parameters['base_period'],
                volatility_adjustment=1.0,
                market_regime='unknown',
                efficiency_score=0.0,
                optimization_confidence=0.0
(            )
    
def _calculate_ma_performance_score(self, data: pd.DataFrame, ma: pd.Series) -> float:
        """Calculate performance score for MA configuration"""
        try:
            if len(ma) < 10:
                return 0.0
            
            valid_data = data[ma.notna()]
            valid_ma = ma.dropna()
            
            if len(valid_data) < 5:
                return 0.0
            
            # Trend following performance
            ma_signals = np.where(valid_data['close'] > valid_ma, 1, -1)
            price_returns = valid_data['close'].pct_change().fillna(0)
            
            # Calculate strategy returns
            strategy_returns = ma_signals[:-1] * price_returns.iloc[1:].values
            
            # Performance metrics
            total_return = np.sum(strategy_returns)
            sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10)
            
            # Smoothness penalty (prefer smoother MAs)
            ma_changes = abs(valid_ma.diff()).sum()
            smoothness_score = 1.0 / (1.0 + ma_changes / len(valid_ma))
            
            # Combined score
            performance_score = total_return * 10 + sharpe_ratio + smoothness_score
            
            return max(0, performance_score)
            
        except Exception as e:
            logger.error(f"Error calculating MA performance: {str(e)}")
            return 0.0
    
def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime"""
        try:
            if len(data) < 20:
                return 'unknown'
            
            # Volatility analysis
            returns = data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Trend strength
            price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
            
            # Volume analysis
            avg_volume = data['volume'].mean()
            recent_volume = data['volume'].tail(5).mean()
            volume_ratio = recent_volume / avg_volume
            
            # Regime classification
            if volatility < 0.01:  # Low volatility:
                if abs(price_change) < 0.02:
                    return 'consolidation'
                else:
                    return 'trending_low_vol'
            elif volatility < 0.03:  # Medium volatility:
                if volume_ratio > 1.5:
                    return 'breakout'
                else:
                    return 'trending_medium_vol'
            else:  # High volatility
                return 'high_volatility'
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return 'unknown'
    
def _calculate_trend_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate trend efficiency"""
        try:
            if len(data) < 10:
                return 0.0
            
            # Net price movement vs total price movement
            net_move = abs(data['close'].iloc[-1] - data['close'].iloc[0])
            total_move = abs(data['close'].diff()).sum()
            
            efficiency = net_move / total_move if total_move > 0 else 0
            
            return min(1.0, efficiency)
            
        except Exception as e:
            logger.error(f"Error calculating trend efficiency: {str(e)}")
            return 0.0    
def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive triangular moving average analysis
        """
        try:
            if len(data) < self.parameters['base_period']:
                return {
                    'triangular_ma': 0.0,
                    'trend_analysis': {},
                    'adaptive_parameters': {},
                    'crossover_signals': [],
                    'momentum_alignment': 0.0,
                    'ma_envelope': {},
                    'trend_strength': 0.0
                }
            
            # Determine period (adaptive or fixed)
            if self.parameters['adaptive_enabled']:
                period = self._calculate_adaptive_period(data)
            else:
                period = self.parameters['base_period']
            
            # Calculate main triangular MA
            triangular_ma = self._calculate_triangular_ma(data['close'], period)
            current_ma = triangular_ma.iloc[-1] if not triangular_ma.isna().all() else 0.0
            
            # Store in history
            if not np.isnan(current_ma):
                self.ma_history.append(current_ma)
                if len(self.ma_history) > 200:  # Limit history size:
                    self.ma_history = self.ma_history[-200:]
            
            # Comprehensive trend analysis
            trend_analysis = self._analyze_trend_characteristics(triangular_ma, data)
            
            # Parameter optimization
            adaptive_params = self._optimize_parameters(data)
            self.adaptive_params = adaptive_params
            
            # Crossover signal generation
            crossover_signals = self._generate_crossover_signals(data, triangular_ma)
            self.crossover_signals.extend(crossover_signals)
            
            # Keep only recent crossover signals
            if len(self.crossover_signals) > 50:
                self.crossover_signals = self.crossover_signals[-50:]
            
            # Momentum alignment
            momentum_alignment = self._calculate_momentum_alignment(data, triangular_ma)
            
            # MA envelope analysis
            ma_envelope = self._calculate_ma_envelope(data, triangular_ma)
            
            # Trend strength calculation
            trend_strength = self._calculate_trend_strength(data, triangular_ma)
            
            # Support/Resistance levels from MA
            support_resistance = self._identify_ma_support_resistance(triangular_ma)
            
            # Price distance analysis
            price_distance = self._analyze_price_distance(data['close'], triangular_ma)
            
            # MA smoothness analysis
            smoothness_metrics = self._analyze_ma_smoothness(triangular_ma)
            
            # Signal quality assessment
            signal_quality = self._assess_signal_quality(data, triangular_ma, trend_analysis)
            
            result = {
                'triangular_ma': current_ma,
                'adaptive_period': period,
                'trend_analysis': self._trend_analysis_to_dict(trend_analysis),
                'adaptive_parameters': self._adaptive_params_to_dict(adaptive_params),
                'crossover_signals': crossover_signals,
                'momentum_alignment': momentum_alignment,
                'ma_envelope': ma_envelope,
                'trend_strength': trend_strength,
                'support_resistance': support_resistance,
                'price_distance': price_distance,
                'smoothness_metrics': smoothness_metrics,
                'signal_quality': signal_quality,
                'ma_slope': triangular_ma.diff().iloc[-1] if len(triangular_ma) > 1 else 0.0,
                'ma_acceleration': triangular_ma.diff().diff().iloc[-1] if len(triangular_ma) > 2 else 0.0,
                'historical_performance': self._calculate_historical_performance(data, triangular_ma),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in triangular MA calculation: {str(e)}")
            raise IndicatorCalculationError()
                indicator_name=self.name,
                calculation_step="triangular_ma_calculation",
                message=str(e)
(            )
    
def _calculate_ma_envelope(self, data: pd.DataFrame, ma: pd.Series) -> Dict[str, Any]:
        """Calculate moving average envelope bands"""
        try:
            if len(ma) < 20:
                return {'upper_band': 0.0, 'lower_band': 0.0, 'bandwidth': 0.0, 'position': 0.5}
            
            # Calculate dynamic envelope based on volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.tail(20).std()
            
            # Envelope percentage based on volatility
            envelope_pct = max(0.01, min(0.05, volatility * 2))
            
            current_ma = ma.iloc[-1]
            upper_band = current_ma * (1 + envelope_pct)
            lower_band = current_ma * (1 - envelope_pct)
            
            # Current price position within envelope
            current_price = data['close'].iloc[-1]
            if upper_band != lower_band:
                position = (current_price - lower_band) / (upper_band - lower_band)
            else:
                position = 0.5
            
            position = max(0, min(1, position))
            
            return {
                'upper_band': upper_band,
                'lower_band': lower_band,
                'bandwidth': (upper_band - lower_band) / current_ma,
                'position': position,
                'envelope_percentage': envelope_pct
            }
            
        except Exception as e:
            logger.error(f"Error calculating MA envelope: {str(e)}")
            return {'upper_band': 0.0, 'lower_band': 0.0, 'bandwidth': 0.0, 'position': 0.5}
    
def _calculate_trend_strength(self, data: pd.DataFrame, ma: pd.Series) -> float:
        """Calculate overall trend strength"""
        try:
            if len(ma) < 10:
                return 0.0
            
            # Multiple factors for trend strength
            factors = []
            
            # 1. MA slope consistency
            ma_diff = ma.diff().dropna()
            if len(ma_diff) > 0:
                slope_consistency = abs(ma_diff.mean() / (ma_diff.std() + 1e-10))
                factors.append(min(1.0, slope_consistency))
            
            # 2. Price above/below MA consistency
            price_position = data['close'] - ma
            position_consistency = abs(price_position.tail(10).mean()) / (price_position.tail(10).std() + 1e-10)
            factors.append(min(1.0, position_consistency / 2))
            
            # 3. Volume confirmation
            if 'volume' in data.columns:
                volume_ma = data['volume'].rolling(20).mean()
                recent_volume = data['volume'].tail(5).mean()
                volume_factor = min(2.0, recent_volume / volume_ma.iloc[-1]) / 2
                factors.append(volume_factor)
            
            # 4. Directional movement
            price_direction = np.sign(data['close'].iloc[-1] - data['close'].iloc[-10])
            ma_direction = np.sign(ma.iloc[-1] - ma.iloc[-10])
            direction_alignment = 1 if price_direction == ma_direction else 0
            factors.append(direction_alignment)
            
            # Combined trend strength
            trend_strength = np.mean(factors) if factors else 0.0
            
            return min(1.0, max(0.0, trend_strength))
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
    
def _identify_ma_support_resistance(self, ma: pd.Series) -> Dict[str, Any]:
        """Identify support and resistance levels from MA"""
        try:
            if len(ma) < 50:
                return {'support_levels': [], 'resistance_levels': []}
            
            # Find local peaks and troughs in MA
            ma_values = ma.dropna().values
            
            # Simple peak/trough detection
            peaks = []
            troughs = []
            
            for i in range(2, len(ma_values) - 2):
                # Peak detection
                if (ma_values[i] > ma_values[i-1] and ma_values[i] > ma_values[i+1] and:)
(                    ma_values[i] > ma_values[i-2] and ma_values[i] > ma_values[i+2]):
                    peaks.append(ma_values[i])
                
                # Trough detection
                if (ma_values[i] < ma_values[i-1] and ma_values[i] < ma_values[i+1] and:)
(                    ma_values[i] < ma_values[i-2] and ma_values[i] < ma_values[i+2]):
                    troughs.append(ma_values[i])
            
            # Keep most recent and significant levels
            current_ma = ma.iloc[-1]
            
            # Filter levels close to current price
            relevant_resistance = [p for p in peaks if p > current_ma and p < current_ma * 1.1][-3:]
            relevant_support = [t for t in troughs if t < current_ma and t > current_ma * 0.9][-3:]
            
            return {
                'support_levels': sorted(relevant_support, reverse=True),
                'resistance_levels': sorted(relevant_resistance),
                'nearest_support': max(relevant_support) if relevant_support else current_ma * 0.98,
                'nearest_resistance': min(relevant_resistance) if relevant_resistance else current_ma * 1.02
            }
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {str(e)}")
            return {'support_levels': [], 'resistance_levels': []}
    
def _analyze_price_distance(self, price: pd.Series, ma: pd.Series) -> Dict[str, Any]:
        """Analyze price distance from MA"""
        try:
            if len(price) < 10 or len(ma) < 10:
                return {'current_distance': 0.0, 'distance_percentile': 0.5, 'mean_reversion_signal': 0.0}
            
            # Current distance
            current_distance = (price.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1]
            
            # Historical distances
            historical_distances = ((price - ma) / ma).dropna()
            
            # Distance percentile
            if len(historical_distances) > 0:
                distance_percentile = stats.percentileofscore(historical_distances, current_distance) / 100
            else:
                distance_percentile = 0.5
            
            # Mean reversion signal (extreme distances tend to revert)
            extreme_threshold = 2.0
            if len(historical_distances) > 20:
                distance_std = historical_distances.std()
                z_score = current_distance / (distance_std + 1e-10)
                
                if abs(z_score) > extreme_threshold:
                    mean_reversion_signal = -np.sign(z_score) * min(1.0, abs(z_score) / extreme_threshold)
                else:
                    mean_reversion_signal = 0.0
            else:
                mean_reversion_signal = 0.0
            
            return {
                'current_distance': current_distance,
                'distance_percentile': distance_percentile,
                'mean_reversion_signal': mean_reversion_signal,
                'distance_volatility': historical_distances.std() if len(historical_distances) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing price distance: {str(e)}")
            return {'current_distance': 0.0, 'distance_percentile': 0.5, 'mean_reversion_signal': 0.0}
    
def _analyze_ma_smoothness(self, ma: pd.Series) -> Dict[str, Any]:
        """Analyze moving average smoothness characteristics"""
        try:
            if len(ma) < 10:
                return {'smoothness_score': 0.0, 'noise_level': 1.0, 'directional_consistency': 0.0}
            
            ma_clean = ma.dropna()
            
            # Smoothness score (lower variation = smoother)
            ma_returns = ma_clean.pct_change().dropna()
            smoothness_score = 1.0 / (1.0 + ma_returns.std() * 100) if len(ma_returns) > 0 else 0.0
            
            # Noise level (frequency of direction changes)
            direction_changes = (ma_returns[:-1] * ma_returns[1:] < 0).sum() if len(ma_returns) > 1 else 0
            noise_level = direction_changes / max(1, len(ma_returns) - 1)
            
            # Directional consistency
            positive_moves = (ma_returns > 0).sum()
            negative_moves = (ma_returns < 0).sum()
            total_moves = positive_moves + negative_moves
            
            if total_moves > 0:
                directional_consistency = abs(positive_moves - negative_moves) / total_moves
            else:
                directional_consistency = 0.0
            
            return {
                'smoothness_score': smoothness_score,
                'noise_level': noise_level,
                'directional_consistency': directional_consistency,
                'average_move_size': abs(ma_returns).mean() if len(ma_returns) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing MA smoothness: {str(e)}")
            return {'smoothness_score': 0.0, 'noise_level': 1.0, 'directional_consistency': 0.0}
    
def _assess_signal_quality(self, data: pd.DataFrame, ma: pd.Series,:)
(                             trend_analysis: TrendAnalysis) -> Dict[str, Any]:
        """Assess the quality of current signals"""
        try:
            quality_factors = {}
            
            # Trend alignment quality
            trend_alignment = 0
            if trend_analysis.short_term_trend == trend_analysis.medium_term_trend:
                trend_alignment += 1
            if trend_analysis.medium_term_trend == trend_analysis.long_term_trend:
                trend_alignment += 1
            
            quality_factors['trend_alignment'] = trend_alignment / 2
            
            # Volume confirmation quality
            if 'volume' in data.columns and len(data) >= 20:
                recent_volume = data['volume'].tail(5).mean()
                avg_volume = data['volume'].tail(20).mean()
                volume_quality = min(2.0, recent_volume / avg_volume) / 2
            else:
                volume_quality = 0.5
            
            quality_factors['volume_confirmation'] = volume_quality
            
            # Price action quality (clean moves vs choppy)
            price_efficiency = self._calculate_trend_efficiency(data)
            quality_factors['price_action_quality'] = price_efficiency
            
            # MA responsiveness quality
            ma_responsiveness = 1.0 - min(1.0, abs(trend_analysis.trend_velocity) * 100)
            quality_factors['ma_responsiveness'] = ma_responsiveness
            
            # Overall signal quality
            overall_quality = np.mean(list(quality_factors.values()))
            
            return {
                'overall_quality': overall_quality,
                'quality_factors': quality_factors,
                'signal_reliability': min(1.0, overall_quality * trend_analysis.trend_consistency)
            }
            
        except Exception as e:
            logger.error(f"Error assessing signal quality: {str(e)}")
            return {'overall_quality': 0.5, 'quality_factors': {}, 'signal_reliability': 0.5}
    
def _calculate_historical_performance(self, data: pd.DataFrame, ma: pd.Series) -> Dict[str, Any]:
        """Calculate historical performance metrics"""
        try:
            if len(data) < 50 or len(ma) < 50:
                return {'accuracy': 0.5, 'avg_return': 0.0, 'max_drawdown': 0.0}
            
            # Align data
            valid_indices = ma.dropna().index
            aligned_data = data.loc[valid_indices]
            aligned_ma = ma.dropna()
            
            if len(aligned_data) < 30:
                return {'accuracy': 0.5, 'avg_return': 0.0, 'max_drawdown': 0.0}
            
            # Generate signals (1 = bullish, -1 = bearish)
            signals = np.where(aligned_data['close'] > aligned_ma, 1, -1)
            
            # Calculate returns
            returns = aligned_data['close'].pct_change().fillna(0)
            strategy_returns = signals[:-1] * returns.iloc[1:].values
            
            # Performance metrics
            accuracy = (strategy_returns > 0).mean()
            avg_return = strategy_returns.mean()
            
            # Max drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'accuracy': accuracy,
                'avg_return': avg_return,
                'max_drawdown': abs(max_drawdown),
                'total_return': cumulative_returns.iloc[-1] - 1,
                'sharpe_ratio': avg_return / (strategy_returns.std() + 1e-10)
            }
            
        except Exception as e:
            logger.error(f"Error calculating historical performance: {str(e)}")
            return {'accuracy': 0.5, 'avg_return': 0.0, 'max_drawdown': 0.0}
    
def _trend_analysis_to_dict(self, trend_analysis: TrendAnalysis) -> Dict[str, Any]:
        """Convert TrendAnalysis to dictionary"""
        return {
            'short_term_trend': trend_analysis.short_term_trend,
            'medium_term_trend': trend_analysis.medium_term_trend,
            'long_term_trend': trend_analysis.long_term_trend,
            'trend_consistency': trend_analysis.trend_consistency,
            'trend_velocity': trend_analysis.trend_velocity,
            'trend_acceleration': trend_analysis.trend_acceleration,
            'reversal_probability': trend_analysis.reversal_probability
        }
    
def _adaptive_params_to_dict(self, params: AdaptiveParameters) -> Dict[str, Any]:
        """Convert AdaptiveParameters to dictionary"""
        return {
            'current_period': params.current_period,
            'volatility_adjustment': params.volatility_adjustment,
            'market_regime': params.market_regime,
            'efficiency_score': params.efficiency_score,
            'optimization_confidence': params.optimization_confidence
        }
    
def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on triangular MA analysis
        """
        try:
            current_price = data['close'].iloc[-1]
            triangular_ma = value.get('triangular_ma', 0)
            trend_analysis = value.get('trend_analysis', {})
            momentum_alignment = value.get('momentum_alignment', 0.0)
            trend_strength = value.get('trend_strength', 0.0)
            signal_quality = value.get('signal_quality', {})
            
            if triangular_ma == 0:
                return SignalType.NEUTRAL, 0.0
            
            # Basic MA signal
            price_above_ma = current_price > triangular_ma
            
            # Trend analysis signals
            short_term_trend = trend_analysis.get('short_term_trend', 'neutral')
            medium_term_trend = trend_analysis.get('medium_term_trend', 'neutral')
            trend_consistency = trend_analysis.get('trend_consistency', 0.0)
            reversal_probability = trend_analysis.get('reversal_probability', 0.5)
            
            # Signal generation logic
            signal_strength = 0.0
            signal_type = SignalType.NEUTRAL
            
            # Primary signal from MA position
            if price_above_ma and short_term_trend == 'bullish':
                signal_type = SignalType.BUY
                signal_strength += 0.3
            elif not price_above_ma and short_term_trend == 'bearish':
                signal_type = SignalType.SELL
                signal_strength += 0.3
            
            # Trend alignment boost
            if short_term_trend == medium_term_trend:
                signal_strength += 0.2
            
            # Momentum alignment
            if momentum_alignment > 0.5 and signal_type == SignalType.BUY:
                signal_strength += 0.2
            elif momentum_alignment < -0.5 and signal_type == SignalType.SELL:
                signal_strength += 0.2
            
            # Trend strength factor
            signal_strength += trend_strength * 0.2
            
            # Trend consistency factor
            signal_strength *= trend_consistency
            
            # Signal quality adjustment
            overall_quality = signal_quality.get('overall_quality', 0.5)
            signal_strength *= overall_quality
            
            # Reversal probability penalty
            if reversal_probability > 0.7:
                signal_strength *= 0.5
            
            # Crossover signals boost
            crossover_signals = value.get('crossover_signals', [])
            for signal in crossover_signals[-3:]:  # Recent crossovers:
                if signal['type'] == 'bullish_crossover' and signal_type == SignalType.BUY:
                    signal_strength += signal['strength'] * 0.1
                elif signal['type'] == 'bearish_crossover' and signal_type == SignalType.SELL:
                    signal_strength += signal['strength'] * 0.1
            
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
        
        tma_metadata = {
            'adaptive_enabled': self.parameters['adaptive_enabled'],
            'base_period': self.parameters['base_period'],
            'current_period': self.adaptive_params.current_period if self.adaptive_params else self.parameters['base_period'],
            'smoothing_factor': self.parameters['smoothing_factor'],
            'ma_history_length': len(self.ma_history),
            'crossover_signals_count': len(self.crossover_signals),
            'optimization_window': self.parameters['optimization_window']
        }
        
        base_metadata.update(tma_metadata)
        return base_metadata


def create_triangular_moving_average_indicator(parameters: Optional[Dict[str, Any]] = None) -> TriangularMovingAverageIndicator:
    """
    Factory function to create a TriangularMovingAverageIndicator instance
    
    Args:
        parameters: Optional dictionary of parameters to customize the indicator
        
    Returns:
        Configured TriangularMovingAverageIndicator instance
    """
    return TriangularMovingAverageIndicator(parameters=parameters)
def get_data_requirements(self):
        """
        Get data requirements for TriangularMovingAverageIndicator.
        
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
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create realistic price data with trends
    base_price = 100
    trend = 0.0005  # Small upward trend
    volatility = 0.02
    
    # Generate price series with realistic patterns
    returns = np.random.normal(trend, volatility, len(dates))
    
    # Add some trending periods
    for i in range(50, 100):
        returns[i] += 0.002  # Strong uptrend
    for i in range(150, 200):
        returns[i] -= 0.002  # Downtrend
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate volume data
    base_volume = 100000
    volume_noise = np.random.lognormal(0, 0.3, len(dates))
    volumes = base_volume * volume_noise
    
    sample_data = pd.DataFrame({)
        'high': prices * np.random.uniform(1.001, 1.02, len(dates)),
        'low': prices * np.random.uniform(0.98, 0.999, len(dates)),
        'close': prices,
        'volume': volumes
(    }, index=dates)
    
    # Test the indicator
    indicator = create_triangular_moving_average_indicator({)
        'base_period': 20,
        'adaptive_enabled': True,
        'crossover_periods': [10, 30],
        'optimization_window': 50
(    })
    
    try:
        result = indicator.calculate(sample_data)
        print("Triangular Moving Average Analysis Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Current MA Value: {result.value.get('triangular_ma', 0):.2f}")
        print(f"Adaptive Period: {result.value.get('adaptive_period', 0)}")
        print(f"Trend Strength: {result.value.get('trend_strength', 0):.3f}")
        
        # Display trend analysis
        trend_analysis = result.value.get('trend_analysis', {})
        print(f"\nTrend Analysis:")
        print(f"Short-term: {trend_analysis.get('short_term_trend', 'unknown')}")
        print(f"Medium-term: {trend_analysis.get('medium_term_trend', 'unknown')}")
        print(f"Long-term: {trend_analysis.get('long_term_trend', 'unknown')}")
        print(f"Consistency: {trend_analysis.get('trend_consistency', 0):.3f}")
        print(f"Reversal Probability: {trend_analysis.get('reversal_probability', 0):.3f}")
        
        # Display adaptive parameters
        adaptive_params = result.value.get('adaptive_parameters', {})
        print(f"\nAdaptive Parameters:")
        print(f"Market Regime: {adaptive_params.get('market_regime', 'unknown')}")
        print(f"Efficiency Score: {adaptive_params.get('efficiency_score', 0):.3f}")
        print(f"Optimization Confidence: {adaptive_params.get('optimization_confidence', 0):.3f}")
        
        # Display recent crossover signals
        crossover_signals = result.value.get('crossover_signals', [])
        if crossover_signals:
            print(f"\nRecent Crossover Signals ({len(crossover_signals)}):")
            for signal in crossover_signals[-3:]:
                print(f"- {signal['type']} (Period: {signal['period']}, Strength: {signal['strength']:.3f})")
        
    except Exception as e:
        print(f"Error testing indicator: {str(e)}")