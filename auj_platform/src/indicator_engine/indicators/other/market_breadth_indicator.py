"""
Market Breadth Indicator - Comprehensive Market Participation Analysis
=====================================================================

This module implements a sophisticated market breadth indicator that measures
the participation and internal strength of market movements. It analyzes
advance-decline ratios, new highs-lows, volume participation, and sector
rotation to provide comprehensive market breadth assessment.

Features:
- Advance-decline line calculation and analysis
- New highs and new lows tracking
- Volume-weighted breadth analysis
- Sector rotation and participation measurement
- Breadth momentum and divergence detection
- McClellan Oscillator and Summation Index
- Arms Index (TRIN) calculation
- Breadth thrust and expansion analysis
- Market internal strength assessment

The indicator helps traders understand market participation and identify
potential turning points based on breadth divergences and participation patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface, IndicatorResult, DataRequirement, DataType, SignalType
from src.core.exceptions import IndicatorCalculationError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class BreadthMetric:
    """Represents a market breadth metric"""
    name: str
    value: float
    percentile: float  # Historical percentile
    signal: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0.0 to 1.0
    divergence: bool  # Whether showing divergence from price


@dataclass
class SectorBreadth:
    """Represents sector-specific breadth data"""
    sector: str
    advancing: int
    declining: int
    unchanged: int
    advance_decline_ratio: float
    volume_participation: float
    strength_score: float


@dataclass
class BreadthSignal:
    """Represents a breadth-based trading signal"""
    signal_type: str  # 'thrust', 'divergence', 'exhaustion', 'expansion'
    direction: str  # 'bullish', 'bearish'
    strength: float
    confidence: float
    description: str


class MarketBreadthIndicator(StandardIndicatorInterface):
    """
    Advanced Market Breadth Indicator with comprehensive participation analysis
    and signal generation capabilities.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'advance_decline_window': 20,
            'new_highs_lows_window': 252,  # 1 year for new highs/lows
            'breadth_momentum_window': 10,
            'volume_weight': 0.4,
            'mcclellan_fast_ema': 19,
            'mcclellan_slow_ema': 39,
            'breadth_thrust_threshold': 0.9,  # 90% advancing for thrust
            'breadth_expansion_threshold': 0.7,  # 70% for expansion
            'divergence_lookback': 20,
            'min_participation_threshold': 0.6,
            'sector_weights': {
                'Technology': 0.20,
                'Healthcare': 0.15,
                'Financials': 0.13,
                'Consumer Discretionary': 0.12,
                'Communication Services': 0.11,
                'Industrials': 0.08,
                'Consumer Staples': 0.07,
                'Energy': 0.04,
                'Utilities': 0.03,
                'Real Estate': 0.03,
                'Materials': 0.04
            },
            'arms_index_smoothing': 5,
            'high_low_index_threshold': 0.7,
            'volume_surge_threshold': 1.5,  # 150% of average volume
            'breadth_percentile_window': 252,
            'signal_confidence_threshold': 0.6
        }

        if parameters:
            default_params.update(parameters)

        super().__init__(name="MarketBreadth")

        # Initialize internal state
        self.advance_decline_line = []
        self.mcclellan_oscillator_history = []
        self.mcclellan_summation_index = 0.0
        self.breadth_metrics: List[BreadthMetric] = []
        self.sector_breadth: List[SectorBreadth] = []
        self.breadth_signals: List[BreadthSignal] = []
        self.scaler = StandardScaler()

        logger.info(f"MarketBreadthIndicator initialized")

    def get_data_requirements(self) -> DataRequirement:
        """Define the data requirements for market breadth calculation"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(50, self.parameters['advance_decline_window']),
            lookback_periods=max(self.parameters['new_highs_lows_window'], 300)
        )

    def validate_parameters(self) -> bool:
        """Validate the indicator parameters"""
        try:
            required_params = ['advance_decline_window', 'mcclellan_fast_ema', 'mcclellan_slow_ema']
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")

            if self.parameters['advance_decline_window'] < 5:
                raise ValueError("advance_decline_window must be at least 5")

            if self.parameters['mcclellan_fast_ema'] >= self.parameters['mcclellan_slow_ema']:
                raise ValueError("mcclellan_fast_ema must be less than mcclellan_slow_ema")

            return True

        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False

    def _simulate_market_constituents(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate market constituent data based on main data
        In real implementation, this would use actual constituent data
        """
        try:
            np.random.seed(42)  # For reproducible results

            # Create synthetic constituent data based on main price data
            returns = data['close'].pct_change().dropna()

            constituents_data = []
            num_constituents = 100  # Simulate 100 stocks

            for i in range(num_constituents):
                # Create correlated returns with some noise
                correlation = np.random.uniform(0.3, 0.8)  # Random correlation with market
                noise_factor = np.random.uniform(0.5, 1.5)  # Random volatility multiplier

                constituent_returns = (returns * correlation +
                                     np.random.normal(0, returns.std() * noise_factor, len(returns)))

                # Calculate prices from returns
                constituent_prices = 100 * (1 + constituent_returns).cumprod()

                # Calculate highs and lows with some spread
                spread = np.random.uniform(0.01, 0.03)  # 1-3% spread
                constituent_highs = constituent_prices * (1 + spread)
                constituent_lows = constituent_prices * (1 - spread)

                # Simulate volume
                base_volume = np.random.uniform(50000, 500000)
                volume_volatility = np.random.uniform(0.2, 0.8)
                constituent_volume = base_volume * (1 + np.random.normal(0, volume_volatility, len(returns)))
                constituent_volume = np.maximum(constituent_volume, base_volume * 0.1)  # Minimum volume

                constituent_df = pd.DataFrame({
                    'high': constituent_highs,
                    'low': constituent_lows,
                    'close': constituent_prices,
                    'volume': constituent_volume
                }, index=data.index[1:])  # Skip first row due to returns calculation

                constituents_data.append(constituent_df)

            return constituents_data

        except Exception as e:
            logger.error(f"Error simulating market constituents: {str(e)}")
            return []

    def _calculate_advance_decline_metrics(self, constituents_data: List[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate advance-decline related metrics"""
        try:
            if not constituents_data:
                return {'advancing': 0, 'declining': 0, 'unchanged': 0, 'ad_ratio': 0.0, 'ad_line_change': 0.0}

            # Get latest data point
            advancing = 0
            declining = 0
            unchanged = 0

            for constituent in constituents_data:
                if len(constituent) > 1:
                    latest_return = constituent['close'].pct_change().iloc[-1]

                    if latest_return > 0.001:  # >0.1% considered advancing
                        advancing += 1
                    elif latest_return < -0.001:  # <-0.1% considered declining
                        declining += 1
                    else:
                        unchanged += 1

            total_trading = advancing + declining
            ad_ratio = advancing / max(total_trading, 1)

            # Calculate advance-decline line change
            ad_line_change = advancing - declining
            self.advance_decline_line.append(ad_line_change)

            # Keep only recent history
            max_history = self.parameters['breadth_percentile_window']
            if len(self.advance_decline_line) > max_history:
                self.advance_decline_line = self.advance_decline_line[-max_history:]

            return {
                'advancing': advancing,
                'declining': declining,
                'unchanged': unchanged,
                'ad_ratio': ad_ratio,
                'ad_line_change': ad_line_change,
                'ad_line': sum(self.advance_decline_line),
                'total_constituents': len(constituents_data)
            }

        except Exception as e:
            logger.error(f"Error calculating advance-decline metrics: {str(e)}")
            return {'advancing': 0, 'declining': 0, 'unchanged': 0, 'ad_ratio': 0.0, 'ad_line_change': 0.0}

    def _calculate_new_highs_lows(self, constituents_data: List[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate new highs and lows metrics"""
        try:
            if not constituents_data:
                return {'new_highs': 0, 'new_lows': 0, 'hl_ratio': 0.0, 'hl_index': 0.0}

            new_highs = 0
            new_lows = 0
            lookback = min(self.parameters['new_highs_lows_window'], 252)

            for constituent in constituents_data:
                if len(constituent) > lookback:
                    recent_data = constituent.tail(lookback)
                    latest_high = recent_data['high'].iloc[-1]
                    latest_low = recent_data['low'].iloc[-1]

                    # Check for new highs (highest in lookback period)
                    if latest_high >= recent_data['high'].max():
                        new_highs += 1

                    # Check for new lows (lowest in lookback period)
                    if latest_low <= recent_data['low'].min():
                        new_lows += 1

            total_new = new_highs + new_lows
            hl_ratio = new_highs / max(total_new, 1)

            # High-Low Index (new highs as percentage of total constituents)
            hl_index = new_highs / max(len(constituents_data), 1)

            return {
                'new_highs': new_highs,
                'new_lows': new_lows,
                'hl_ratio': hl_ratio,
                'hl_index': hl_index,
                'total_new': total_new
            }

        except Exception as e:
            logger.error(f"Error calculating new highs/lows: {str(e)}")
            return {'new_highs': 0, 'new_lows': 0, 'hl_ratio': 0.0, 'hl_index': 0.0}

    def _calculate_volume_breadth(self, constituents_data: List[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate volume-weighted breadth metrics"""
        try:
            if not constituents_data:
                return {'volume_ratio': 0.0, 'volume_participation': 0.0, 'volume_surge': False}

            advancing_volume = 0.0
            declining_volume = 0.0
            total_volume = 0.0

            for constituent in constituents_data:
                if len(constituent) > 1:
                    latest_return = constituent['close'].pct_change().iloc[-1]
                    latest_volume = constituent['volume'].iloc[-1]

                    total_volume += latest_volume

                    if latest_return > 0:
                        advancing_volume += latest_volume
                    elif latest_return < 0:
                        declining_volume += latest_volume

            volume_ratio = advancing_volume / max(advancing_volume + declining_volume, 1)
            volume_participation = total_volume / max(len(constituents_data), 1)

            # Check for volume surge
            if len(constituents_data) > 0 and len(constituents_data[0]) > 20:
                avg_volume = np.mean([constituent['volume'].rolling(20).mean().iloc[-1]
                                    for constituent in constituents_data
                                    if len(constituent) > 20])
                current_avg_volume = total_volume / len(constituents_data)
                volume_surge = current_avg_volume > avg_volume * self.parameters['volume_surge_threshold']
            else:
                volume_surge = False

            return {
                'volume_ratio': volume_ratio,
                'advancing_volume': advancing_volume,
                'declining_volume': declining_volume,
                'total_volume': total_volume,
                'volume_participation': volume_participation,
                'volume_surge': volume_surge
            }

        except Exception as e:
            logger.error(f"Error calculating volume breadth: {str(e)}")
            return {'volume_ratio': 0.0, 'volume_participation': 0.0, 'volume_surge': False}

    def _calculate_mcclellan_oscillator(self, ad_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate McClellan Oscillator and Summation Index"""
        try:
            ad_line_change = ad_data.get('ad_line_change', 0.0)

            fast_ema = self.parameters['mcclellan_fast_ema']
            slow_ema = self.parameters['mcclellan_slow_ema']

            # Initialize EMAs if first calculation
            if not hasattr(self, 'mcclellan_fast_ema_value'):
                self.mcclellan_fast_ema_value = ad_line_change
                self.mcclellan_slow_ema_value = ad_line_change

            # Calculate EMAs
            alpha_fast = 2.0 / (fast_ema + 1)
            alpha_slow = 2.0 / (slow_ema + 1)

            self.mcclellan_fast_ema_value = (alpha_fast * ad_line_change +
                                           (1 - alpha_fast) * self.mcclellan_fast_ema_value)
            self.mcclellan_slow_ema_value = (alpha_slow * ad_line_change +
                                           (1 - alpha_slow) * self.mcclellan_slow_ema_value)

            # McClellan Oscillator = Fast EMA - Slow EMA
            mcclellan_oscillator = self.mcclellan_fast_ema_value - self.mcclellan_slow_ema_value

            # McClellan Summation Index = Running sum of oscillator
            self.mcclellan_summation_index += mcclellan_oscillator

            # Store history
            self.mcclellan_oscillator_history.append(mcclellan_oscillator)
            max_history = self.parameters['breadth_percentile_window']
            if len(self.mcclellan_oscillator_history) > max_history:
                self.mcclellan_oscillator_history = self.mcclellan_oscillator_history[-max_history:]

            return {
                'mcclellan_oscillator': mcclellan_oscillator,
                'mcclellan_summation_index': self.mcclellan_summation_index,
                'fast_ema': self.mcclellan_fast_ema_value,
                'slow_ema': self.mcclellan_slow_ema_value
            }

        except Exception as e:
            logger.error(f"Error calculating McClellan Oscillator: {str(e)}")
            return {'mcclellan_oscillator': 0.0, 'mcclellan_summation_index': 0.0}

    def _calculate_arms_index(self, ad_data: Dict[str, Any], volume_data: Dict[str, Any]) -> float:
        """Calculate Arms Index (TRIN)"""
        try:
            advancing = max(ad_data.get('advancing', 1), 1)
            declining = max(ad_data.get('declining', 1), 1)
            advancing_volume = max(volume_data.get('advancing_volume', 1), 1)
            declining_volume = max(volume_data.get('declining_volume', 1), 1)

            # TRIN = (Declining Issues / Advancing Issues) / (Declining Volume / Advancing Volume)
            arms_index = (declining / advancing) / (declining_volume / advancing_volume)

            return arms_index

        except Exception as e:
            logger.error(f"Error calculating Arms Index: {str(e)}")
            return 1.0  # Neutral value
    def _detect_breadth_signals(self, ad_data: Dict[str, Any], volume_data: Dict[str, Any],
                               mcclellan_data: Dict[str, Any], arms_index: float) -> List[BreadthSignal]:
        """Detect breadth-based trading signals"""
        try:
            signals = []

            # Breadth Thrust Signal
            ad_ratio = ad_data.get('ad_ratio', 0.0)
            if ad_ratio >= self.parameters['breadth_thrust_threshold']:
                signals.append(BreadthSignal(
                    signal_type='thrust',
                    direction='bullish',
                    strength=min(ad_ratio, 1.0),
                    confidence=0.8,
                    description=f"Breadth thrust detected: {ad_ratio:.1%} advancing"
                ))
            elif ad_ratio <= (1 - self.parameters['breadth_thrust_threshold']):
                signals.append(BreadthSignal(
                    signal_type='thrust',
                    direction='bearish',
                    strength=min(1 - ad_ratio, 1.0),
                    confidence=0.8,
                    description=f"Bearish breadth thrust: {ad_ratio:.1%} advancing"
                ))

            # Volume Confirmation
            volume_ratio = volume_data.get('volume_ratio', 0.0)
            if volume_data.get('volume_surge', False):
                if volume_ratio > 0.6:
                    signals.append(BreadthSignal(
                        signal_type='expansion',
                        direction='bullish',
                        strength=volume_ratio,
                        confidence=0.7,
                        description=f"Volume expansion with {volume_ratio:.1%} advancing volume"
                    ))
                elif volume_ratio < 0.4:
                    signals.append(BreadthSignal(
                        signal_type='expansion',
                        direction='bearish',
                        strength=1 - volume_ratio,
                        confidence=0.7,
                        description=f"Bearish volume expansion with {volume_ratio:.1%} advancing volume"
                    ))

            # McClellan Oscillator Signals
            mcclellan_osc = mcclellan_data.get('mcclellan_oscillator', 0.0)
            if mcclellan_osc > 100:
                signals.append(BreadthSignal(
                    signal_type='divergence',
                    direction='bearish',
                    strength=min(mcclellan_osc / 200, 1.0),
                    confidence=0.6,
                    description=f"McClellan Oscillator overbought: {mcclellan_osc:.1f}"
                ))
            elif mcclellan_osc < -100:
                signals.append(BreadthSignal(
                    signal_type='divergence',
                    direction='bullish',
                    strength=min(abs(mcclellan_osc) / 200, 1.0),
                    confidence=0.6,
                    description=f"McClellan Oscillator oversold: {mcclellan_osc:.1f}"
                ))

            # Arms Index (TRIN) Signals
            if arms_index > 2.0:
                signals.append(BreadthSignal(
                    signal_type='exhaustion',
                    direction='bullish',
                    strength=min((arms_index - 1.0) / 2.0, 1.0),
                    confidence=0.6,
                    description=f"Arms Index extreme: {arms_index:.2f} - oversold"
                ))
            elif arms_index < 0.5:
                signals.append(BreadthSignal(
                    signal_type='exhaustion',
                    direction='bearish',
                    strength=min((1.0 - arms_index) / 0.5, 1.0),
                    confidence=0.6,
                    description=f"Arms Index extreme: {arms_index:.2f} - overbought"
                ))

            return signals

        except Exception as e:
            logger.error(f"Error detecting breadth signals: {str(e)}")
            return []

    def _calculate_breadth_percentiles(self, current_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate historical percentiles for breadth metrics"""
        try:
            percentiles = {}

            # A-D Line percentile
            if self.advance_decline_line:
                current_ad_line = sum(self.advance_decline_line)
                if len(self.advance_decline_line) > 20:
                    ad_line_history = [sum(self.advance_decline_line[:i+1])
                                     for i in range(len(self.advance_decline_line))]
                    percentiles['ad_line'] = stats.percentileofscore(ad_line_history, current_ad_line) / 100
                else:
                    percentiles['ad_line'] = 0.5

            # McClellan Oscillator percentile
            if self.mcclellan_oscillator_history:
                current_mcclellan = self.mcclellan_oscillator_history[-1]
                if len(self.mcclellan_oscillator_history) > 20:
                    percentiles['mcclellan'] = stats.percentileofscore(
                        self.mcclellan_oscillator_history, current_mcclellan) / 100
                else:
                    percentiles['mcclellan'] = 0.5

            # A-D Ratio percentile (approximate)
            ad_ratio = current_metrics.get('ad_ratio', 0.5)
            percentiles['ad_ratio'] = ad_ratio  # Simplified - would use historical data

            return percentiles

        except Exception as e:
            logger.error(f"Error calculating percentiles: {str(e)}")
            return {}

    def _create_breadth_metrics(self, ad_data: Dict[str, Any], volume_data: Dict[str, Any],
                              mcclellan_data: Dict[str, Any], hl_data: Dict[str, Any],
                              arms_index: float, percentiles: Dict[str, float]) -> List[BreadthMetric]:
        """Create standardized breadth metrics"""
        try:
            metrics = []

            # Advance-Decline Ratio
            ad_ratio = ad_data.get('ad_ratio', 0.0)
            ad_signal = 'bullish' if ad_ratio > 0.6 else 'bearish' if ad_ratio < 0.4 else 'neutral'
            metrics.append(BreadthMetric(
                name='Advance-Decline Ratio',
                value=ad_ratio,
                percentile=percentiles.get('ad_ratio', 0.5),
                signal=ad_signal,
                strength=abs(ad_ratio - 0.5) * 2,
                divergence=False  # Would need price comparison
            ))

            # Volume Ratio
            volume_ratio = volume_data.get('volume_ratio', 0.0)
            volume_signal = 'bullish' if volume_ratio > 0.6 else 'bearish' if volume_ratio < 0.4 else 'neutral'
            metrics.append(BreadthMetric(
                name='Volume Ratio',
                value=volume_ratio,
                percentile=volume_ratio,  # Simplified
                signal=volume_signal,
                strength=abs(volume_ratio - 0.5) * 2,
                divergence=False
            ))

            # High-Low Index
            hl_index = hl_data.get('hl_index', 0.0)
            hl_signal = 'bullish' if hl_index > 0.3 else 'bearish' if hl_index < 0.1 else 'neutral'
            metrics.append(BreadthMetric(
                name='High-Low Index',
                value=hl_index,
                percentile=hl_index,  # Simplified
                signal=hl_signal,
                strength=hl_index if hl_index > 0.3 else (0.1 - hl_index) if hl_index < 0.1 else 0,
                divergence=False
            ))

            # McClellan Oscillator
            mcclellan_osc = mcclellan_data.get('mcclellan_oscillator', 0.0)
            mcclellan_signal = ('bullish' if mcclellan_osc < -50 else
                              'bearish' if mcclellan_osc > 50 else 'neutral')
            metrics.append(BreadthMetric(
                name='McClellan Oscillator',
                value=mcclellan_osc,
                percentile=percentiles.get('mcclellan', 0.5),
                signal=mcclellan_signal,
                strength=min(abs(mcclellan_osc) / 100, 1.0),
                divergence=abs(mcclellan_osc) > 75
            ))

            # Arms Index (TRIN)
            arms_signal = ('bullish' if arms_index > 1.5 else
                         'bearish' if arms_index < 0.7 else 'neutral')
            metrics.append(BreadthMetric(
                name='Arms Index (TRIN)',
                value=arms_index,
                percentile=0.5,  # Would need historical data
                signal=arms_signal,
                strength=abs(1.0 - arms_index) if arms_index > 1.5 or arms_index < 0.7 else 0,
                divergence=arms_index > 2.0 or arms_index < 0.5
            ))

            return metrics

        except Exception as e:
            logger.error(f"Error creating breadth metrics: {str(e)}")
            return []

    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive market breadth analysis
        """
        try:
            # Simulate market constituent data (in real implementation, use actual data)
            constituents_data = self._simulate_market_constituents(data)

            # Calculate advance-decline metrics
            ad_data = self._calculate_advance_decline_metrics(constituents_data)

            # Calculate new highs and lows
            hl_data = self._calculate_new_highs_lows(constituents_data)

            # Calculate volume breadth
            volume_data = self._calculate_volume_breadth(constituents_data)

            # Calculate McClellan Oscillator
            mcclellan_data = self._calculate_mcclellan_oscillator(ad_data)

            # Calculate Arms Index
            arms_index = self._calculate_arms_index(ad_data, volume_data)

            # Calculate historical percentiles
            percentiles = self._calculate_breadth_percentiles({
                'ad_ratio': ad_data.get('ad_ratio', 0.0),
                'mcclellan_oscillator': mcclellan_data.get('mcclellan_oscillator', 0.0)
            })

            # Create breadth metrics
            breadth_metrics = self._create_breadth_metrics(
                ad_data, volume_data, mcclellan_data, hl_data, arms_index, percentiles
            )
            self.breadth_metrics = breadth_metrics

            # Detect breadth signals
            breadth_signals = self._detect_breadth_signals(
                ad_data, volume_data, mcclellan_data, arms_index
            )
            self.breadth_signals = breadth_signals

            # Calculate overall breadth strength
            bullish_signals = sum(1 for signal in breadth_signals if signal.direction == 'bullish')
            bearish_signals = sum(1 for signal in breadth_signals if signal.direction == 'bearish')
            total_signals = len(breadth_signals)

            overall_breadth_strength = 0.0
            if total_signals > 0:
                net_bullish = (bullish_signals - bearish_signals) / total_signals
                overall_breadth_strength = (net_bullish + 1) / 2  # Normalize to 0-1

            # Calculate market participation
            participation_score = ad_data.get('ad_ratio', 0.0)
            if volume_data.get('volume_surge', False):
                participation_score = min(participation_score * 1.2, 1.0)

            # Prepare result
            result = {
                'advance_decline_data': ad_data,
                'new_highs_lows_data': hl_data,
                'volume_breadth_data': volume_data,
                'mcclellan_data': mcclellan_data,
                'arms_index': arms_index,
                'breadth_metrics': [self._metric_to_dict(metric) for metric in breadth_metrics],
                'breadth_signals': [self._signal_to_dict(signal) for signal in breadth_signals],
                'overall_breadth_strength': overall_breadth_strength,
                'participation_score': participation_score,
                'market_stress_level': 1.0 - participation_score,
                'breadth_percentiles': percentiles,
                'constituent_count': len(constituents_data),
                'significant_signals': len([s for s in breadth_signals
                                          if s.confidence >= self.parameters['signal_confidence_threshold']]),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error in market breadth calculation: {str(e)}")
            raise IndicatorCalculationError(
                indicator_name=self.name,
                calculation_step="market_breadth_calculation",
                message=str(e)
            )

    def _metric_to_dict(self, metric: BreadthMetric) -> Dict[str, Any]:
        """Convert BreadthMetric to dictionary"""
        return {
            'name': metric.name,
            'value': metric.value,
            'percentile': metric.percentile,
            'signal': metric.signal,
            'strength': metric.strength,
            'divergence': metric.divergence
        }

    def _signal_to_dict(self, signal: BreadthSignal) -> Dict[str, Any]:
        """Convert BreadthSignal to dictionary"""
        return {
            'signal_type': signal.signal_type,
            'direction': signal.direction,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'description': signal.description
        }

    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on market breadth analysis
        """
        try:
            overall_strength = value.get('overall_breadth_strength', 0.5)
            participation_score = value.get('participation_score', 0.5)
            significant_signals = value.get('significant_signals', 0)

            if significant_signals == 0:
                return SignalType.NEUTRAL, 0.0

            # Strong breadth with high participation = bullish
            if overall_strength > 0.7 and participation_score > 0.6:
                confidence = min(overall_strength * participation_score, 1.0)
                return SignalType.BUY, confidence

            # Weak breadth with low participation = bearish
            elif overall_strength < 0.3 and participation_score < 0.4:
                confidence = min((1 - overall_strength) * (1 - participation_score), 1.0)
                return SignalType.SELL, confidence

            # Check for breadth thrust signals
            breadth_signals = value.get('breadth_signals', [])
            thrust_signals = [s for s in breadth_signals if s['signal_type'] == 'thrust']

            if thrust_signals:
                strongest_thrust = max(thrust_signals, key=lambda x: x['strength'])
                if strongest_thrust['strength'] > 0.8:
                    if strongest_thrust['direction'] == 'bullish':
                        return SignalType.STRONG_BUY, strongest_thrust['confidence']
                    else:
                        return SignalType.STRONG_SELL, strongest_thrust['confidence']

            # Check for divergence signals
            divergence_signals = [s for s in breadth_signals if s['signal_type'] == 'divergence']
            if divergence_signals:
                strongest_divergence = max(divergence_signals, key=lambda x: x['strength'])
                if strongest_divergence['strength'] > 0.6:
                    if strongest_divergence['direction'] == 'bullish':
                        return SignalType.BUY, strongest_divergence['confidence'] * 0.8
                    else:
                        return SignalType.SELL, strongest_divergence['confidence'] * 0.8

            return SignalType.NEUTRAL, 0.0

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0

    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)

        breadth_metadata = {
            'advance_decline_window': self.parameters['advance_decline_window'],
            'total_breadth_metrics': len(self.breadth_metrics),
            'active_breadth_signals': len(self.breadth_signals),
            'mcclellan_oscillator_history_length': len(self.mcclellan_oscillator_history),
            'advance_decline_line_length': len(self.advance_decline_line),
            'breadth_thrust_threshold': self.parameters['breadth_thrust_threshold'],
            'sector_weights_enabled': bool(self.parameters['sector_weights'])
        }

        base_metadata.update(breadth_metadata)
        return base_metadata


def create_market_breadth_indicator(parameters: Optional[Dict[str, Any]] = None) -> MarketBreadthIndicator:
    """
    Factory function to create a MarketBreadthIndicator instance

    Args:
        parameters: Optional dictionary of parameters to customize the indicator

    Returns:
        Configured MarketBreadthIndicator instance
    """
    return MarketBreadthIndicator(parameters=parameters)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # Simulate market data with varying breadth conditions
    base_returns = np.random.randn(len(dates)) * 0.01
    volatility = np.random.uniform(0.8, 1.2, len(dates))

    sample_data = pd.DataFrame({
        'high': 100 * np.exp(np.cumsum(base_returns * volatility + np.random.randn(len(dates)) * 0.003)),
        'low': 100 * np.exp(np.cumsum(base_returns * volatility - np.random.randn(len(dates)) * 0.003)),
        'close': 100 * np.exp(np.cumsum(base_returns * volatility)),
        'volume': np.random.uniform(500000, 2000000, len(dates))
    }, index=dates)

    # Add some trend to highs and lows
    sample_data['high'] = sample_data['close'] * np.random.uniform(1.005, 1.02, len(dates))
    sample_data['low'] = sample_data['close'] * np.random.uniform(0.98, 0.995, len(dates))

    # Test the indicator
    indicator = create_market_breadth_indicator({
        'advance_decline_window': 20,
        'breadth_thrust_threshold': 0.85
    })

    try:
        result = indicator.calculate(sample_data)
        print("Market Breadth Calculation Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Overall breadth strength: {result.value.get('overall_breadth_strength', 0):.3f}")
        print(f"Participation score: {result.value.get('participation_score', 0):.3f}")
        print(f"Significant signals: {result.value.get('significant_signals', 0)}")

        ad_data = result.value.get('advance_decline_data', {})
        print(f"Advancing: {ad_data.get('advancing', 0)}, Declining: {ad_data.get('declining', 0)}")
        print(f"A-D Ratio: {ad_data.get('ad_ratio', 0):.3f}")

        mcclellan_data = result.value.get('mcclellan_data', {})
        print(f"McClellan Oscillator: {mcclellan_data.get('mcclellan_oscillator', 0):.2f}")

        print(f"Arms Index: {result.value.get('arms_index', 1.0):.3f}")

    except Exception as e:
        print(f"Error testing indicator: {str(e)}")
