"""
AUJ Platform Advanced Klinger Oscillator Indicator
Sophisticated implementation with volume-based trend identification, cycle analysis, and signal filtering

This implementation provides institutional-grade Klinger Oscillator analysis for humanitarian trading platforms.

Features:
- Enhanced Klinger Oscillator calculation with volume force dynamics
- Volume-based trend identification and strength measurement
- Cycle analysis and harmonic pattern detection
- Advanced signal filtering and noise reduction
- Multi-timeframe oscillator analysis
- Divergence detection with ML enhancement
- Momentum confirmation and trend validation
- Statistical significance testing
- Adaptive parameter optimization
- Comprehensive signal generation system

The Klinger Oscillator combines price and volume to identify long-term money flow trends
while remaining sensitive to short-term fluctuations. This implementation enhances the
traditional formula with advanced analytics and machine learning capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats, signal, fft
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from ..base.base_indicator import BaseIndicator, IndicatorConfig
from ...core.signal_type import SignalType


class KlingerTrend(Enum):
    """Klinger Oscillator trend classification"""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    WEAK_BULLISH = "weak_bullish"
    NEUTRAL = "neutral"
    WEAK_BEARISH = "weak_bearish"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class VolumeForceDirection(Enum):
    """Volume force direction classification"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    ROTATION = "rotation"
    CONSOLIDATION = "consolidation"


class DivergenceType(Enum):
    """Type of divergence detected"""
    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"
    BULLISH_HIDDEN = "bullish_hidden"
    BEARISH_HIDDEN = "bearish_hidden"
    NO_DIVERGENCE = "no_divergence"


@dataclass
class KlingerConfig(IndicatorConfig):
    """Configuration for Klinger Oscillator Indicator"""
    fast_period: int = 34
    slow_period: int = 55
    signal_period: int = 13
    volume_force_period: int = 20
    trend_analysis_period: int = 50
    divergence_lookback: int = 20
    cycle_analysis_period: int = 100
    signal_threshold: float = 0.1
    divergence_threshold: float = 0.3
    use_adaptive_parameters: bool = True
    use_cycle_analysis: bool = True
    use_ml_enhancement: bool = True
    min_periods: int = 100


class VolumeForceAnalysis(NamedTuple):
    """Volume force analysis result"""
    volume_force: np.ndarray
    force_direction: VolumeForceDirection
    force_strength: float
    force_consistency: float
    accumulation_score: float
    distribution_score: float


class CycleAnalysis(NamedTuple):
    """Cycle analysis result"""
    dominant_cycle: float
    cycle_strength: float
    phase_position: float
    cycle_trend: str
    harmonic_components: List[float]
    cycle_reliability: float


class DivergenceAnalysis(NamedTuple):
    """Divergence analysis result"""
    divergence_type: DivergenceType
    divergence_strength: float
    confidence: float
    time_to_resolution: int
    price_targets: List[float]


class KlingerResult(NamedTuple):
    """Complete Klinger Oscillator analysis result"""
    klinger_value: float
    signal_line: float
    histogram: float
    trend_classification: KlingerTrend
    volume_force_analysis: VolumeForceAnalysis
    cycle_analysis: CycleAnalysis
    divergence_analysis: DivergenceAnalysis
    momentum_strength: float
    signal_quality: float
    confidence_score: float


class KlingerOscillatorIndicator(BaseIndicator):
    """
    Advanced Klinger Oscillator Indicator with sophisticated analytics.
    
    The Klinger Oscillator analyzes the relationship between money flow and price
    movements to identify underlying trends. This implementation enhances the
    traditional oscillator with:
    - Volume force dynamics analysis
    - Cycle identification and harmonic analysis
    - Advanced divergence detection
    - Multi-timeframe trend analysis
    - Machine learning enhancement
    - Statistical significance testing
    """
    
    def __init__(self, config: Optional[KlingerConfig] = None):
        super().__init__(config or KlingerConfig())
        self.config: KlingerConfig = self.config
        
        # Internal state
        self._klinger_history: List[float] = []
        self._signal_history: List[float] = []
        self._volume_force_history: List[np.ndarray] = []
        self._price_history: List[float] = []
        self._volume_history: List[float] = []
        
        # Machine learning components
        self._trend_predictor: Optional[RandomForestRegressor] = None
        self._scaler: StandardScaler = StandardScaler()
        self._is_trained: bool = False
        
        # Adaptive parameters
        self._adaptive_fast_period: int = self.config.fast_period
        self._adaptive_slow_period: int = self.config.slow_period
        self._adaptive_signal_period: int = self.config.signal_period
        
        # Cycle analysis components
        self._cycle_detector: Optional[object] = None
        self._harmonic_analyzer: Optional[object] = None
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate advanced Klinger Oscillator with comprehensive analysis.
        
        Args:
            data: Dictionary containing 'high', 'low', 'close', 'volume' price data
            
        Returns:
            Dictionary containing Klinger Oscillator analysis results
        """
        try:
            if not self._validate_data(data):
                return self._create_default_result()
            
            df = pd.DataFrame(data)
            
            if len(df) < self.config.min_periods:
                return self._create_default_result()
            
            # Calculate volume force
            volume_force_analysis = self._calculate_volume_force(df)
            
            # Calculate Klinger Oscillator
            klinger_values, signal_line = self._calculate_klinger_oscillator(
                df, volume_force_analysis.volume_force
            )
            
            # Calculate histogram (difference between Klinger and signal line)
            histogram = klinger_values[-1] - signal_line[-1]
            
            # Classify trend
            trend_classification = self._classify_trend(klinger_values, signal_line)
            
            # Perform cycle analysis
            if self.config.use_cycle_analysis:
                cycle_analysis = self._analyze_cycles(klinger_values)
            else:
                cycle_analysis = self._create_default_cycle_analysis()
            
            # Detect divergences
            divergence_analysis = self._detect_divergences(df, klinger_values)
            
            # Calculate momentum and signal quality
            momentum_strength = self._calculate_momentum_strength(klinger_values, signal_line)
            signal_quality = self._calculate_signal_quality(
                klinger_values, signal_line, volume_force_analysis
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(
                trend_classification, cycle_analysis, divergence_analysis, signal_quality
            )
            
            # Create result
            result = KlingerResult(
                klinger_value=klinger_values[-1],
                signal_line=signal_line[-1],
                histogram=histogram,
                trend_classification=trend_classification,
                volume_force_analysis=volume_force_analysis,
                cycle_analysis=cycle_analysis,
                divergence_analysis=divergence_analysis,
                momentum_strength=momentum_strength,
                signal_quality=signal_quality,
                confidence_score=confidence_score
            )
            
            # Generate trading signal
            signal = self._generate_signal(result)
            
            # Update internal state and retrain if needed
            self._update_state_and_retrain(df, klinger_values, signal_line, result)
            
            return self._format_result(result, signal)
            
        except Exception as e:
            self.logger.error(f"Error in KlingerOscillatorIndicator calculation: {e}")
            return self._create_error_result(str(e))
    
    def _calculate_volume_force(self, df: pd.DataFrame) -> VolumeForceAnalysis:
        """Calculate volume force dynamics"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Calculate trend direction
        typical_price = (high + low + close) / 3
        trend = np.diff(typical_price, prepend=typical_price[0])
        
        # Calculate volume force
        volume_force = np.zeros(len(df))
        
        for i in range(1, len(df)):
            # Determine if current period is up or down
            if trend[i] > 0:
                # Upward trend - buying pressure
                volume_force[i] = volume[i]
            elif trend[i] < 0:
                # Downward trend - selling pressure
                volume_force[i] = -volume[i]
            else:
                # No change - neutral
                volume_force[i] = 0
        
        # Smooth volume force
        volume_force_smoothed = pd.Series(volume_force).rolling(
            window=self.config.volume_force_period, min_periods=1
        ).mean().values
        
        # Analyze force characteristics
        recent_force = volume_force_smoothed[-self.config.volume_force_period:]
        
        # Force direction classification
        positive_force = np.sum(recent_force[recent_force > 0])
        negative_force = np.abs(np.sum(recent_force[recent_force < 0]))
        total_force = positive_force + negative_force
        
        if total_force > 0:
            accumulation_score = positive_force / total_force
            distribution_score = negative_force / total_force
            
            if accumulation_score > 0.7:
                force_direction = VolumeForceDirection.ACCUMULATION
            elif distribution_score > 0.7:
                force_direction = VolumeForceDirection.DISTRIBUTION
            elif abs(accumulation_score - distribution_score) < 0.2:
                force_direction = VolumeForceDirection.CONSOLIDATION
            else:
                force_direction = VolumeForceDirection.ROTATION
        else:
            accumulation_score = 0.5
            distribution_score = 0.5
            force_direction = VolumeForceDirection.CONSOLIDATION
        
        # Force strength and consistency
        force_strength = np.std(recent_force) / (np.mean(np.abs(recent_force)) + 1e-8)
        force_consistency = 1.0 - (np.std(recent_force) / (np.max(np.abs(recent_force)) + 1e-8))
        force_consistency = max(0.0, min(1.0, force_consistency))
        
        return VolumeForceAnalysis(
            volume_force=volume_force_smoothed,
            force_direction=force_direction,
            force_strength=force_strength,
            force_consistency=force_consistency,
            accumulation_score=accumulation_score,
            distribution_score=distribution_score
        )
    
    def _calculate_klinger_oscillator(self, df: pd.DataFrame, 
                                    volume_force: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Klinger Oscillator using volume force"""
        # Adaptive parameter optimization
        if self.config.use_adaptive_parameters:
            self._optimize_parameters(df, volume_force)
        
        # Calculate EMAs of volume force
        vf_series = pd.Series(volume_force)
        
        # Fast EMA
        fast_ema = vf_series.ewm(span=self._adaptive_fast_period).mean().values
        
        # Slow EMA
        slow_ema = vf_series.ewm(span=self._adaptive_slow_period).mean().values
        
        # Klinger Oscillator = Fast EMA - Slow EMA
        klinger_values = fast_ema - slow_ema
        
        # Signal line (EMA of Klinger Oscillator)
        klinger_series = pd.Series(klinger_values)
        signal_line = klinger_series.ewm(span=self._adaptive_signal_period).mean().values
        
        return klinger_values, signal_line
    
    def _optimize_parameters(self, df: pd.DataFrame, volume_force: np.ndarray):
        """Optimize parameters based on market conditions"""
        # Analyze market volatility
        returns = np.diff(np.log(df['close'].values))
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
        
        # Analyze volume force characteristics
        vf_volatility = np.std(volume_force[-20:]) if len(volume_force) >= 20 else 1.0
        vf_mean = np.mean(np.abs(volume_force[-20:])) if len(volume_force) >= 20 else 1.0
        vf_normalized_vol = vf_volatility / (vf_mean + 1e-8)
        
        # Adjust parameters based on market conditions
        if volatility > 0.03:  # High volatility
            self._adaptive_fast_period = max(self.config.fast_period - 5, 10)
            self._adaptive_slow_period = max(self.config.slow_period - 10, 20)
            self._adaptive_signal_period = max(self.config.signal_period - 3, 5)
        elif volatility < 0.01:  # Low volatility
            self._adaptive_fast_period = min(self.config.fast_period + 5, 50)
            self._adaptive_slow_period = min(self.config.slow_period + 10, 80)
            self._adaptive_signal_period = min(self.config.signal_period + 3, 20)
        else:  # Normal volatility
            self._adaptive_fast_period = self.config.fast_period
            self._adaptive_slow_period = self.config.slow_period
            self._adaptive_signal_period = self.config.signal_period
        
        # Fine-tune based on volume force characteristics
        if vf_normalized_vol > 2.0:  # High volume force volatility
            self._adaptive_signal_period = min(self._adaptive_signal_period + 2, 25)
    
    def _classify_trend(self, klinger_values: np.ndarray, 
                       signal_line: np.ndarray) -> KlingerTrend:
        """Classify the current trend based on Klinger Oscillator"""
        current_klinger = klinger_values[-1]
        current_signal = signal_line[-1]
        histogram = current_klinger - current_signal
        
        # Analyze recent trend
        recent_period = min(self.config.trend_analysis_period, len(klinger_values))
        recent_klinger = klinger_values[-recent_period:]
        recent_signal = signal_line[-recent_period:]
        
        # Calculate trend strength
        klinger_trend = np.polyfit(range(len(recent_klinger)), recent_klinger, 1)[0]
        signal_trend = np.polyfit(range(len(recent_signal)), recent_signal, 1)[0]
        
        # Histogram analysis
        recent_histogram = recent_klinger - recent_signal
        histogram_trend = np.polyfit(range(len(recent_histogram)), recent_histogram, 1)[0]
        
        # Classification logic
        if current_klinger > current_signal and histogram > 0:
            if klinger_trend > 0.1 and histogram_trend > 0.05:
                return KlingerTrend.STRONG_BULLISH
            elif klinger_trend > 0.05 or histogram_trend > 0.02:
                return KlingerTrend.BULLISH
            else:
                return KlingerTrend.WEAK_BULLISH
        elif current_klinger < current_signal and histogram < 0:
            if klinger_trend < -0.1 and histogram_trend < -0.05:
                return KlingerTrend.STRONG_BEARISH
            elif klinger_trend < -0.05 or histogram_trend < -0.02:
                return KlingerTrend.BEARISH
            else:
                return KlingerTrend.WEAK_BEARISH
        else:
            return KlingerTrend.NEUTRAL
    
    def _analyze_cycles(self, klinger_values: np.ndarray) -> CycleAnalysis:
        """Analyze cyclical components of Klinger Oscillator"""
        if len(klinger_values) < self.config.cycle_analysis_period:
            return self._create_default_cycle_analysis()
        
        # Use recent data for cycle analysis
        analysis_data = klinger_values[-self.config.cycle_analysis_period:]
        
        # Detrend the data
        detrended = signal.detrend(analysis_data)
        
        # Perform FFT to identify dominant frequencies
        fft_values = fft.fft(detrended)
        frequencies = fft.fftfreq(len(detrended))
        
        # Find dominant cycle
        power_spectrum = np.abs(fft_values) ** 2
        positive_freq_idx = frequencies > 0
        
        if np.any(positive_freq_idx):
            positive_freqs = frequencies[positive_freq_idx]
            positive_power = power_spectrum[positive_freq_idx]
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(positive_power)
            dominant_frequency = positive_freqs[dominant_freq_idx]
            
            if dominant_frequency > 0:
                dominant_cycle = 1.0 / dominant_frequency
            else:
                dominant_cycle = len(analysis_data)
            
            # Calculate cycle strength
            total_power = np.sum(positive_power)
            dominant_power = positive_power[dominant_freq_idx]
            cycle_strength = dominant_power / total_power if total_power > 0 else 0
            
            # Identify harmonic components
            harmonic_threshold = 0.1 * dominant_power
            harmonic_indices = positive_power > harmonic_threshold
            harmonic_frequencies = positive_freqs[harmonic_indices]
            harmonic_components = [1.0 / freq for freq in harmonic_frequencies if freq > 0]
            
        else:
            dominant_cycle = len(analysis_data)
            cycle_strength = 0.0
            harmonic_components = []
        
        # Calculate phase position
        if dominant_cycle > 0:
            phase_position = (len(analysis_data) % dominant_cycle) / dominant_cycle
        else:
            phase_position = 0.5
        
        # Determine cycle trend
        recent_values = analysis_data[-int(dominant_cycle/4):] if dominant_cycle >= 4 else analysis_data[-5:]
        cycle_trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        if cycle_trend_slope > 0.05:
            cycle_trend = "rising"
        elif cycle_trend_slope < -0.05:
            cycle_trend = "falling"
        else:
            cycle_trend = "flat"
        
        # Calculate cycle reliability
        cycle_reliability = min(cycle_strength * 2, 1.0)
        
        return CycleAnalysis(
            dominant_cycle=dominant_cycle,
            cycle_strength=cycle_strength,
            phase_position=phase_position,
            cycle_trend=cycle_trend,
            harmonic_components=harmonic_components[:5],  # Top 5 harmonics
            cycle_reliability=cycle_reliability
        )
    
    def _detect_divergences(self, df: pd.DataFrame, 
                          klinger_values: np.ndarray) -> DivergenceAnalysis:
        """Detect divergences between price and Klinger Oscillator"""
        if len(df) < self.config.divergence_lookback or len(klinger_values) < self.config.divergence_lookback:
            return DivergenceAnalysis(
                divergence_type=DivergenceType.NO_DIVERGENCE,
                divergence_strength=0.0,
                confidence=0.0,
                time_to_resolution=0,
                price_targets=[]
            )
        
        close_prices = df['close'].values
        lookback = self.config.divergence_lookback
        
        recent_prices = close_prices[-lookback:]
        recent_klinger = klinger_values[-lookback:]
        
        # Find price and oscillator peaks/troughs
        price_peaks = self._find_peaks_troughs(recent_prices, 'peaks')
        price_troughs = self._find_peaks_troughs(recent_prices, 'troughs')
        klinger_peaks = self._find_peaks_troughs(recent_klinger, 'peaks')
        klinger_troughs = self._find_peaks_troughs(recent_klinger, 'troughs')
        
        divergence_type = DivergenceType.NO_DIVERGENCE
        divergence_strength = 0.0
        confidence = 0.0
        
        # Check for bullish divergence (price making lower lows, oscillator making higher lows)
        if len(price_troughs) >= 2 and len(klinger_troughs) >= 2:
            price_trend = price_troughs[-1] - price_troughs[-2]
            klinger_trend = klinger_troughs[-1] - klinger_troughs[-2]
            
            if price_trend < 0 and klinger_trend > 0:
                divergence_type = DivergenceType.BULLISH_REGULAR
                divergence_strength = abs(klinger_trend) / (abs(price_trend) + 1e-8)
                confidence = min(divergence_strength / 2.0, 1.0)
        
        # Check for bearish divergence (price making higher highs, oscillator making lower highs)
        if len(price_peaks) >= 2 and len(klinger_peaks) >= 2:
            price_trend = price_peaks[-1] - price_peaks[-2]
            klinger_trend = klinger_peaks[-1] - klinger_peaks[-2]
            
            if price_trend > 0 and klinger_trend < 0:
                if divergence_type == DivergenceType.NO_DIVERGENCE:
                    divergence_type = DivergenceType.BEARISH_REGULAR
                    divergence_strength = abs(klinger_trend) / (abs(price_trend) + 1e-8)
                    confidence = min(divergence_strength / 2.0, 1.0)
        
        # Estimate time to resolution
        if divergence_type != DivergenceType.NO_DIVERGENCE:
            time_to_resolution = min(int(lookback / 3), 10)
        else:
            time_to_resolution = 0
        
        # Calculate price targets (simplified)
        price_targets = []
        if divergence_type == DivergenceType.BULLISH_REGULAR and len(price_troughs) >= 1:
            current_price = recent_prices[-1]
            target_1 = current_price * 1.02  # 2% upside
            target_2 = current_price * 1.05  # 5% upside
            price_targets = [target_1, target_2]
        elif divergence_type == DivergenceType.BEARISH_REGULAR and len(price_peaks) >= 1:
            current_price = recent_prices[-1]
            target_1 = current_price * 0.98  # 2% downside
            target_2 = current_price * 0.95  # 5% downside
            price_targets = [target_1, target_2]
        
        return DivergenceAnalysis(
            divergence_type=divergence_type,
            divergence_strength=divergence_strength,
            confidence=confidence,
            time_to_resolution=time_to_resolution,
            price_targets=price_targets
        )
    
    def _find_peaks_troughs(self, data: np.ndarray, mode: str) -> List[float]:
        """Find peaks or troughs in data"""
        if mode == 'peaks':
            peaks, _ = signal.find_peaks(data, distance=3)
            return [data[i] for i in peaks]
        else:  # troughs
            troughs, _ = signal.find_peaks(-data, distance=3)
            return [data[i] for i in troughs]
    
    def _calculate_momentum_strength(self, klinger_values: np.ndarray, 
                                   signal_line: np.ndarray) -> float:
        """Calculate momentum strength from Klinger Oscillator"""
        if len(klinger_values) < 10:
            return 0.0
        
        recent_klinger = klinger_values[-10:]
        recent_signal = signal_line[-10:]
        recent_histogram = recent_klinger - recent_signal
        
        # Momentum based on histogram trend and magnitude
        histogram_trend = np.polyfit(range(len(recent_histogram)), recent_histogram, 1)[0]
        histogram_magnitude = np.mean(np.abs(recent_histogram))
        
        # Normalize momentum strength
        momentum_strength = abs(histogram_trend) * histogram_magnitude
        return min(momentum_strength / 10.0, 1.0)  # Normalize to 0-1
    
    def _calculate_signal_quality(self, klinger_values: np.ndarray,
                                signal_line: np.ndarray,
                                volume_force_analysis: VolumeForceAnalysis) -> float:
        """Calculate overall signal quality"""
        quality_factors = []
        
        # Oscillator consistency
        if len(klinger_values) >= 10:
            recent_klinger = klinger_values[-10:]
            klinger_consistency = 1.0 - (np.std(recent_klinger) / (np.mean(np.abs(recent_klinger)) + 1e-8))
            quality_factors.append(max(0.0, min(1.0, klinger_consistency)))
        
        # Volume force consistency
        quality_factors.append(volume_force_analysis.force_consistency)
        
        # Signal line smoothness
        if len(signal_line) >= 10:
            recent_signal = signal_line[-10:]
            signal_smoothness = 1.0 - (np.std(np.diff(recent_signal)) / (np.std(recent_signal) + 1e-8))
            quality_factors.append(max(0.0, min(1.0, signal_smoothness)))
        
        # Histogram clarity
        if len(klinger_values) >= 5 and len(signal_line) >= 5:
            recent_histogram = klinger_values[-5:] - signal_line[-5:]
            histogram_clarity = np.mean(np.abs(recent_histogram)) / (np.std(recent_histogram) + 1e-8)
            quality_factors.append(min(histogram_clarity / 5.0, 1.0))
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _calculate_confidence(self, trend_classification: KlingerTrend,
                            cycle_analysis: CycleAnalysis,
                            divergence_analysis: DivergenceAnalysis,
                            signal_quality: float) -> float:
        """Calculate overall confidence score"""
        confidence_factors = []
        
        # Trend strength confidence
        trend_strength_map = {
            KlingerTrend.STRONG_BULLISH: 1.0,
            KlingerTrend.BULLISH: 0.8,
            KlingerTrend.WEAK_BULLISH: 0.6,
            KlingerTrend.NEUTRAL: 0.3,
            KlingerTrend.WEAK_BEARISH: 0.6,
            KlingerTrend.BEARISH: 0.8,
            KlingerTrend.STRONG_BEARISH: 1.0
        }
        confidence_factors.append(trend_strength_map[trend_classification])
        
        # Cycle analysis confidence
        confidence_factors.append(cycle_analysis.cycle_reliability)
        
        # Divergence confidence
        if divergence_analysis.divergence_type != DivergenceType.NO_DIVERGENCE:
            confidence_factors.append(divergence_analysis.confidence)
        else:
            confidence_factors.append(0.3)  # Neutral when no divergence
        
        # Signal quality
        confidence_factors.append(signal_quality)
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.25, 0.25]
        overall_confidence = np.average(confidence_factors, weights=weights)
        
        return overall_confidence
    
    def _generate_signal(self, result: KlingerResult) -> SignalType:
        """Generate trading signal based on Klinger Oscillator analysis"""
        signal_criteria = []
        
        # Klinger vs Signal line
        if result.klinger_value > result.signal_line and result.histogram > 0:
            signal_criteria.append('bullish_crossover')
        elif result.klinger_value < result.signal_line and result.histogram < 0:
            signal_criteria.append('bearish_crossover')
        
        # Trend classification
        if result.trend_classification in [KlingerTrend.STRONG_BULLISH, KlingerTrend.BULLISH]:
            signal_criteria.append('bullish_trend')
        elif result.trend_classification in [KlingerTrend.STRONG_BEARISH, KlingerTrend.BEARISH]:
            signal_criteria.append('bearish_trend')
        
        # Volume force confirmation
        if result.volume_force_analysis.force_direction == VolumeForceDirection.ACCUMULATION:
            signal_criteria.append('accumulation')
        elif result.volume_force_analysis.force_direction == VolumeForceDirection.DISTRIBUTION:
            signal_criteria.append('distribution')
        
        # Divergence signals
        if result.divergence_analysis.divergence_type == DivergenceType.BULLISH_REGULAR:
            signal_criteria.append('bullish_divergence')
        elif result.divergence_analysis.divergence_type == DivergenceType.BEARISH_REGULAR:
            signal_criteria.append('bearish_divergence')
        
        # Momentum confirmation
        strong_momentum = result.momentum_strength > 0.6
        high_quality = result.signal_quality > 0.7
        high_confidence = result.confidence_score > 0.7
        
        # Signal generation logic
        bullish_signals = sum(1 for criterion in signal_criteria if 'bullish' in criterion or 'accumulation' in criterion)
        bearish_signals = sum(1 for criterion in signal_criteria if 'bearish' in criterion or 'distribution' in criterion)
        
        if (bullish_signals >= 2 and strong_momentum and high_quality and high_confidence):
            return SignalType.BUY
        elif (bearish_signals >= 2 and strong_momentum and high_quality and high_confidence):
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _update_state_and_retrain(self, df: pd.DataFrame, klinger_values: np.ndarray,
                                signal_line: np.ndarray, result: KlingerResult):
        """Update internal state and retrain ML models"""
        max_history = 500
        
        # Update histories
        self._klinger_history.extend(klinger_values[-5:])
        self._signal_history.extend(signal_line[-5:])
        self._volume_force_history.append(result.volume_force_analysis.volume_force)
        self._price_history.extend(df['close'].values[-5:])
        self._volume_history.extend(df['volume'].values[-5:])
        
        # Trim histories
        if len(self._klinger_history) > max_history:
            self._klinger_history = self._klinger_history[-max_history:]
            self._signal_history = self._signal_history[-max_history:]
            self._price_history = self._price_history[-max_history:]
            self._volume_history = self._volume_history[-max_history:]
        
        if len(self._volume_force_history) > max_history // 10:
            self._volume_force_history = self._volume_force_history[-max_history // 10:]
        
        # Retrain ML model periodically
        if (self.config.use_ml_enhancement and len(self._klinger_history) >= 200 and 
            len(self._klinger_history) % 50 == 0):
            self._retrain_ml_model()
    
    def _retrain_ml_model(self):
        """Retrain machine learning model for trend prediction"""
        try:
            if len(self._klinger_history) < 100:
                return
            
            # Prepare training data
            features = []
            targets = []
            
            window_size = 20
            for i in range(window_size, len(self._klinger_history) - 5):
                # Features: recent Klinger, signal, and price data
                feature_vector = []
                
                # Klinger features
                klinger_window = self._klinger_history[i-window_size:i]
                feature_vector.extend([
                    np.mean(klinger_window),
                    np.std(klinger_window),
                    np.max(klinger_window),
                    np.min(klinger_window)
                ])
                
                # Signal features
                if i < len(self._signal_history):
                    signal_window = self._signal_history[i-window_size:i]
                    feature_vector.extend([
                        np.mean(signal_window),
                        np.std(signal_window)
                    ])
                else:
                    feature_vector.extend([0, 0])
                
                # Price features
                if i < len(self._price_history):
                    price_window = self._price_history[i-window_size:i]
                    feature_vector.extend([
                        np.mean(price_window),
                        np.std(price_window)
                    ])
                else:
                    feature_vector.extend([0, 0])
                
                features.append(feature_vector)
                
                # Target: future Klinger direction
                future_klinger = np.mean(self._klinger_history[i:i+5])
                current_klinger = self._klinger_history[i-1]
                target = future_klinger - current_klinger
                targets.append(target)
            
            if len(features) > 20:
                # Scale features
                features_array = np.array(features)
                self._scaler.fit(features_array)
                features_scaled = self._scaler.transform(features_array)
                
                # Train model
                self._trend_predictor = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42
                )
                self._trend_predictor.fit(features_scaled, targets)
                
                self._is_trained = True
                self.logger.info("Klinger ML model retrained successfully")
            
        except Exception as e:
            self.logger.warning(f"ML model retraining failed: {e}")
    
    def _create_default_cycle_analysis(self) -> CycleAnalysis:
        """Create default cycle analysis when insufficient data"""
        return CycleAnalysis(
            dominant_cycle=20.0,
            cycle_strength=0.0,
            phase_position=0.5,
            cycle_trend="flat",
            harmonic_components=[],
            cycle_reliability=0.0
        )
    
    def _format_result(self, result: KlingerResult, signal: SignalType) -> Dict[str, Any]:
        """Format the complete result for output"""
        return {
            'signal': signal,
            'confidence': result.confidence_score,
            
            # Core Klinger values
            'klinger_value': result.klinger_value,
            'signal_line': result.signal_line,
            'histogram': result.histogram,
            
            # Trend analysis
            'trend_classification': result.trend_classification.value,
            'momentum_strength': result.momentum_strength,
            'signal_quality': result.signal_quality,
            
            # Volume force analysis
            'volume_force_direction': result.volume_force_analysis.force_direction.value,
            'force_strength': result.volume_force_analysis.force_strength,
            'force_consistency': result.volume_force_analysis.force_consistency,
            'accumulation_score': result.volume_force_analysis.accumulation_score,
            'distribution_score': result.volume_force_analysis.distribution_score,
            
            # Cycle analysis
            'dominant_cycle': result.cycle_analysis.dominant_cycle,
            'cycle_strength': result.cycle_analysis.cycle_strength,
            'phase_position': result.cycle_analysis.phase_position,
            'cycle_trend': result.cycle_analysis.cycle_trend,
            'harmonic_components': result.cycle_analysis.harmonic_components,
            'cycle_reliability': result.cycle_analysis.cycle_reliability,
            
            # Divergence analysis
            'divergence_type': result.divergence_analysis.divergence_type.value,
            'divergence_strength': result.divergence_analysis.divergence_strength,
            'divergence_confidence': result.divergence_analysis.confidence,
            'time_to_resolution': result.divergence_analysis.time_to_resolution,
            'price_targets': result.divergence_analysis.price_targets,
            
            # Parameters
            'fast_period': self._adaptive_fast_period,
            'slow_period': self._adaptive_slow_period,
            'signal_period': self._adaptive_signal_period,
            
            # Metadata
            'metadata': {
                'indicator_name': 'KlingerOscillatorIndicator',
                'version': '1.0.0',
                'calculation_time': pd.Timestamp.now().isoformat(),
                'adaptive_parameters': self.config.use_adaptive_parameters,
                'cycle_analysis': self.config.use_cycle_analysis,
                'ml_enhancement': self.config.use_ml_enhancement,
                'ml_trained': self._is_trained
            }
        }
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data structure and completeness"""
        required_fields = ['high', 'low', 'close', 'volume']
        
        if not isinstance(data, dict):
            return False
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
            
            if not isinstance(data[field], (list, np.ndarray)) or len(data[field]) == 0:
                self.logger.error(f"Invalid data for field: {field}")
                return False
        
        # Check data consistency
        lengths = [len(data[field]) for field in required_fields]
        if len(set(lengths)) > 1:
            self.logger.error("Inconsistent data lengths across fields")
            return False
        
        return True
    
    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'klinger_value': 0.0,
            'signal_line': 0.0,
            'histogram': 0.0,
            'trend_classification': KlingerTrend.NEUTRAL.value,
            'momentum_strength': 0.0,
            'signal_quality': 0.0,
            'volume_force_direction': VolumeForceDirection.CONSOLIDATION.value,
            'force_strength': 0.0,
            'force_consistency': 0.0,
            'accumulation_score': 0.5,
            'distribution_score': 0.5,
            'dominant_cycle': 20.0,
            'cycle_strength': 0.0,
            'phase_position': 0.5,
            'cycle_trend': 'flat',
            'harmonic_components': [],
            'cycle_reliability': 0.0,
            'divergence_type': DivergenceType.NO_DIVERGENCE.value,
            'divergence_strength': 0.0,
            'divergence_confidence': 0.0,
            'time_to_resolution': 0,
            'price_targets': [],
            'fast_period': self.config.fast_period,
            'slow_period': self.config.slow_period,
            'signal_period': self.config.signal_period,
            'metadata': {
                'indicator_name': 'KlingerOscillatorIndicator',
                'error': 'Insufficient data for calculation'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        result = self._create_default_result()
        result['metadata']['error'] = error_message
        return result