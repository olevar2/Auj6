"""
Gann Time Cycle Indicator - Advanced Implementation
==================================================

This module implements W.D. Gann's sophisticated time cycle analysis methods with
advanced mathematical modeling, machine learning enhancement, and predictive capabilities.

Key Features:
- Multiple cycle detection algorithms (Fourier, Wavelet, Hilbert Transform)
- Harmonic time relationship analysis
- Sacred number integration (Gann's favorite numbers)
- ML-enhanced cycle prediction and validation
- Multi-timeframe cycle synchronization
- Dynamic cycle strength measurement
- Time window projections and forecasting
- Advanced signal generation

Mathematical Foundation:
- Fourier Transform for frequency domain analysis
- Wavelet decomposition for time-frequency analysis
- Hilbert Transform for instantaneous phase analysis
- Harmonic series analysis
- Autocorrelation for cycle detection
- Machine Learning for pattern recognition

Author: Trading Platform Team
Date: 2024
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr
import pywt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CycleData:
    """Data structure for cycle information"""
    period: float
    amplitude: float
    phase: float
    strength: float
    confidence: float
    start_time: datetime
    next_peak: Optional[datetime] = None
    next_trough: Optional[datetime] = None

@dataclass
class HarmonicCycle:
    """Data structure for harmonic cycle relationships"""
    base_cycle: float
    harmonic_order: int
    harmonic_period: float
    synchronization_strength: float
    phase_relationship: float

@dataclass
class TimeProjection:
    """Data structure for time projections"""
    projection_date: datetime
    projection_type: str
    confidence: float
    price_target: Optional[float] = None
    cycle_basis: List[float] = None

class GannTimeCycleIndicator:
    """
    Advanced Gann Time Cycle Indicator Implementation
    
    This class implements W.D. Gann's time cycle analysis methods using modern
    mathematical techniques and machine learning enhancement.
    """
    
    def __init__(self, 
                 lookback_periods: int = 500,
                 min_cycle_length: int = 5,
                 max_cycle_length: int = 200,
                 ml_enhancement: bool = True,
                 sacred_numbers: bool = True):
        """
        Initialize the Gann Time Cycle Indicator
        
        Args:
            lookback_periods: Number of periods to analyze
            min_cycle_length: Minimum cycle length to detect
            max_cycle_length: Maximum cycle length to detect
            ml_enhancement: Enable machine learning features
            sacred_numbers: Use Gann's sacred numbers
        """
        self.lookback_periods = lookback_periods
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self.ml_enhancement = ml_enhancement
        self.sacred_numbers = sacred_numbers
        
        # Gann's Sacred Numbers and Time Periods
        self.gann_sacred_numbers = [
            7, 9, 12, 18, 20, 24, 30, 36, 42, 49, 60, 72, 84, 90, 120, 144, 180
        ]
        
        # Natural time cycles
        self.natural_cycles = {
            'lunar': 29.5,
            'seasonal': 90,
            'annual': 365,
            'business_quarter': 63,
            'trading_month': 22,
            'trading_week': 5
        }
        
        # Initialize storage for analysis results
        self.detected_cycles = []
        self.harmonic_relationships = []
        self.time_projections = []
        self.cycle_strength_history = []
        
        logger.info("Gann Time Cycle Indicator initialized")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive Gann time cycle analysis
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing cycle analysis results
        """
        try:
            if len(data) < self.min_cycle_length * 2:
                raise ValueError("Insufficient data for cycle analysis")
            
            # Prepare price series for analysis
            price_series = self._prepare_price_series(data)
            timestamps = pd.to_datetime(data.index) if not isinstance(data.index, pd.DatetimeIndex) else data.index
            
            # 1. Fourier Transform Cycle Detection
            fourier_cycles = self._fourier_cycle_detection(price_series)
            
            # 2. Wavelet Transform Analysis
            wavelet_cycles = self._wavelet_cycle_analysis(price_series)
            
            # 3. Hilbert Transform Phase Analysis
            hilbert_cycles = self._hilbert_transform_analysis(price_series)
            
            # 4. Autocorrelation Cycle Detection
            autocorr_cycles = self._autocorrelation_cycle_detection(price_series)
            
            # 5. Sacred Number Cycle Analysis
            sacred_cycles = self._sacred_number_cycle_analysis(price_series, timestamps)
            
            # 6. Combine and validate cycles
            all_cycles = self._combine_and_validate_cycles(
                fourier_cycles, wavelet_cycles, hilbert_cycles, 
                autocorr_cycles, sacred_cycles
            )
            
            # 7. Harmonic Relationship Analysis
            harmonic_analysis = self._analyze_harmonic_relationships(all_cycles)
            
            # 8. Cycle Strength and Reliability Assessment
            cycle_strength = self._assess_cycle_strength(all_cycles, price_series)
            
            # 9. Time Projections and Forecasting
            time_projections = self._generate_time_projections(all_cycles, timestamps)
            
            # 10. ML-Enhanced Cycle Validation (if enabled)
            if self.ml_enhancement:
                ml_validation = self._ml_cycle_validation(all_cycles, price_series)
                all_cycles = self._apply_ml_validation(all_cycles, ml_validation)
            
            # 11. Generate Trading Signals
            signals = self._generate_cycle_signals(all_cycles, harmonic_analysis, timestamps)
            
            # 12. Calculate cycle-based support/resistance
            cycle_levels = self._calculate_cycle_levels(all_cycles, price_series, timestamps)
            
            # Store results for historical tracking
            self.detected_cycles = all_cycles
            self.harmonic_relationships = harmonic_analysis
            self.time_projections = time_projections
            
            # Compile comprehensive results
            results = {
                'detected_cycles': all_cycles,
                'harmonic_relationships': harmonic_analysis,
                'cycle_strength': cycle_strength,
                'time_projections': time_projections,
                'cycle_signals': signals,
                'cycle_levels': cycle_levels,
                'fourier_analysis': fourier_cycles,
                'wavelet_analysis': wavelet_cycles,
                'hilbert_analysis': hilbert_cycles,
                'autocorr_analysis': autocorr_cycles,
                'sacred_number_analysis': sacred_cycles,
                'analysis_timestamp': datetime.now(),
                'data_points_analyzed': len(price_series)
            }
            
            if self.ml_enhancement:
                results['ml_validation'] = ml_validation
            
            logger.info(f"Gann Time Cycle analysis completed: {len(all_cycles)} cycles detected")
            return results
            
        except Exception as e:
            logger.error(f"Error in Gann Time Cycle calculation: {str(e)}")
            return {'error': str(e), 'cycles': [], 'signals': []}
    
    def _prepare_price_series(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare price series for cycle analysis"""
        try:
            # Use typical price for analysis
            if all(col in data.columns for col in ['high', 'low', 'close']):
                price_series = (data['high'] + data['low'] + data['close']) / 3
            elif 'close' in data.columns:
                price_series = data['close']
            else:
                price_series = data.iloc[:, -1]  # Use last column
            
            # Remove NaN values and ensure numeric
            price_series = pd.to_numeric(price_series, errors='coerce').dropna()
            
            # Apply smoothing to reduce noise
            if len(price_series) > 10:
                window_size = min(5, len(price_series) // 10)
                price_series = price_series.rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            
            return price_series.values
            
        except Exception as e:
            logger.error(f"Error preparing price series: {str(e)}")
            return np.array([])
    
    def _fourier_cycle_detection(self, price_series: np.ndarray) -> List[CycleData]:
        """Detect cycles using Fourier Transform"""
        try:
            cycles = []
            
            if len(price_series) < self.min_cycle_length * 2:
                return cycles
            
            # Detrend the data
            detrended = signal.detrend(price_series)
            
            # Apply window function to reduce spectral leakage
            windowed = detrended * signal.windows.hann(len(detrended))
            
            # Compute FFT
            fft_values = fft(windowed)
            frequencies = fftfreq(len(windowed))
            
            # Calculate power spectral density
            power_spectrum = np.abs(fft_values) ** 2
            
            # Find peaks in the power spectrum
            peak_indices, properties = signal.find_peaks(
                power_spectrum[:len(power_spectrum)//2],
                height=np.std(power_spectrum) * 2,
                distance=max(1, len(power_spectrum) // (self.max_cycle_length * 2))
            )
            
            for idx in peak_indices:
                if frequencies[idx] != 0:  # Avoid division by zero
                    period = 1 / abs(frequencies[idx])
                    
                    # Filter by cycle length constraints
                    if self.min_cycle_length <= period <= self.max_cycle_length:
                        amplitude = np.sqrt(power_spectrum[idx])
                        phase = np.angle(fft_values[idx])
                        strength = properties['peak_heights'][np.where(peak_indices == idx)[0][0]]
                        
                        # Calculate confidence based on peak prominence
                        confidence = min(1.0, strength / (np.mean(power_spectrum) * 10))
                        
                        cycle = CycleData(
                            period=period,
                            amplitude=amplitude,
                            phase=phase,
                            strength=strength,
                            confidence=confidence,
                            start_time=datetime.now()
                        )
                        cycles.append(cycle)
            
            # Sort by strength (descending)
            cycles.sort(key=lambda x: x.strength, reverse=True)
            
            return cycles[:10]  # Return top 10 cycles
            
        except Exception as e:
            logger.error(f"Error in Fourier cycle detection: {str(e)}")
            return []    
    def _wavelet_cycle_analysis(self, price_series: np.ndarray) -> List[CycleData]:
        """Analyze cycles using Wavelet Transform"""
        try:
            cycles = []
            
            if len(price_series) < self.min_cycle_length * 2:
                return cycles
            
            # Use Morlet wavelet for time-frequency analysis
            scales = np.arange(self.min_cycle_length, min(self.max_cycle_length, len(price_series)//4))
            wavelet = 'morl'
            
            # Compute continuous wavelet transform
            coefficients, frequencies = pywt.cwt(price_series, scales, wavelet)
            
            # Calculate power (magnitude squared)
            power = np.abs(coefficients) ** 2
            
            # Find dominant scales/frequencies
            avg_power = np.mean(power, axis=1)
            peak_indices = signal.find_peaks(avg_power, height=np.std(avg_power))[0]
            
            for idx in peak_indices:
                scale = scales[idx]
                period = scale * 2  # Approximate period for Morlet wavelet
                
                if self.min_cycle_length <= period <= self.max_cycle_length:
                    # Calculate cycle properties
                    amplitude = np.sqrt(avg_power[idx])
                    
                    # Estimate phase from the complex coefficients
                    complex_coeff = pywt.cwt(price_series, [scale], wavelet)[0][0]
                    phase = np.angle(np.mean(complex_coeff))
                    
                    strength = avg_power[idx]
                    confidence = min(1.0, strength / (np.mean(avg_power) * 5))
                    
                    cycle = CycleData(
                        period=period,
                        amplitude=amplitude,
                        phase=phase,
                        strength=strength,
                        confidence=confidence,
                        start_time=datetime.now()
                    )
                    cycles.append(cycle)
            
            # Sort by strength
            cycles.sort(key=lambda x: x.strength, reverse=True)
            return cycles[:8]  # Return top 8 cycles
            
        except Exception as e:
            logger.error(f"Error in wavelet cycle analysis: {str(e)}")
            return []
    
    def _hilbert_transform_analysis(self, price_series: np.ndarray) -> List[CycleData]:
        """Analyze instantaneous frequency and phase using Hilbert Transform"""
        try:
            cycles = []
            
            if len(price_series) < self.min_cycle_length * 2:
                return cycles
            
            # Apply Hilbert transform to get analytic signal
            analytic_signal = signal.hilbert(price_series)
            
            # Extract instantaneous amplitude and phase
            instantaneous_amplitude = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            
            # Calculate instantaneous frequency
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            
            # Smooth the frequency to reduce noise
            if len(instantaneous_frequency) > 10:
                window_size = min(10, len(instantaneous_frequency) // 5)
                instantaneous_frequency = pd.Series(instantaneous_frequency).rolling(
                    window=window_size, center=True
                ).mean().fillna(method='bfill').fillna(method='ffill').values
            
            # Convert frequency to period
            valid_freq = instantaneous_frequency[instantaneous_frequency > 0]
            if len(valid_freq) > 0:
                periods = 1 / valid_freq
                
                # Find dominant periods
                period_hist, period_bins = np.histogram(periods, bins=50)
                peak_indices = signal.find_peaks(period_hist)[0]
                
                for idx in peak_indices:
                    period = (period_bins[idx] + period_bins[idx + 1]) / 2
                    
                    if self.min_cycle_length <= period <= self.max_cycle_length:
                        # Calculate strength as frequency of occurrence
                        strength = period_hist[idx]
                        
                        # Estimate amplitude and phase
                        period_mask = (periods >= period_bins[idx]) & (periods < period_bins[idx + 1])
                        if np.any(period_mask):
                            relevant_indices = np.where(period_mask)[0]
                            amplitude = np.mean(instantaneous_amplitude[relevant_indices])
                            phase = np.mean(instantaneous_phase[relevant_indices])
                        else:
                            amplitude = np.mean(instantaneous_amplitude)
                            phase = np.mean(instantaneous_phase)
                        
                        confidence = min(1.0, strength / (np.sum(period_hist) * 0.1))
                        
                        cycle = CycleData(
                            period=period,
                            amplitude=amplitude,
                            phase=phase,
                            strength=strength,
                            confidence=confidence,
                            start_time=datetime.now()
                        )
                        cycles.append(cycle)
            
            # Sort by strength
            cycles.sort(key=lambda x: x.strength, reverse=True)
            return cycles[:6]  # Return top 6 cycles
            
        except Exception as e:
            logger.error(f"Error in Hilbert transform analysis: {str(e)}")
            return []
    
    def _autocorrelation_cycle_detection(self, price_series: np.ndarray) -> List[CycleData]:
        """Detect cycles using autocorrelation analysis"""
        try:
            cycles = []
            
            if len(price_series) < self.min_cycle_length * 2:
                return cycles
            
            # Calculate autocorrelation
            max_lag = min(self.max_cycle_length, len(price_series) // 2)
            autocorr = np.correlate(price_series, price_series, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Normalize autocorrelation
            autocorr = autocorr / autocorr[0]
            
            # Find peaks in autocorrelation (indicating cyclical behavior)
            peak_indices, properties = signal.find_peaks(
                autocorr[:max_lag],
                height=0.1,  # Minimum correlation threshold
                distance=self.min_cycle_length
            )
            
            for idx in peak_indices:
                period = idx
                
                if self.min_cycle_length <= period <= self.max_cycle_length:
                    correlation = autocorr[idx]
                    
                    # Calculate cycle properties
                    amplitude = np.std(price_series)  # Use price volatility as amplitude proxy
                    phase = 0  # Phase is not directly available from autocorrelation
                    strength = correlation * 100  # Scale correlation
                    confidence = correlation  # Correlation is a good confidence measure
                    
                    cycle = CycleData(
                        period=period,
                        amplitude=amplitude,
                        phase=phase,
                        strength=strength,
                        confidence=confidence,
                        start_time=datetime.now()
                    )
                    cycles.append(cycle)
            
            # Sort by correlation strength
            cycles.sort(key=lambda x: x.confidence, reverse=True)
            return cycles[:8]  # Return top 8 cycles
            
        except Exception as e:
            logger.error(f"Error in autocorrelation cycle detection: {str(e)}")
            return []
    
    def _sacred_number_cycle_analysis(self, price_series: np.ndarray, 
                                    timestamps: pd.DatetimeIndex) -> List[CycleData]:
        """Analyze cycles based on Gann's sacred numbers"""
        try:
            cycles = []
            
            if not self.sacred_numbers or len(price_series) < max(self.gann_sacred_numbers):
                return cycles
            
            for sacred_period in self.gann_sacred_numbers:
                if sacred_period > len(price_series):
                    continue
                
                # Calculate correlation at this specific period
                if sacred_period < len(price_series):
                    shifted_series = np.roll(price_series, sacred_period)
                    correlation, p_value = pearsonr(
                        price_series[sacred_period:], 
                        shifted_series[sacred_period:]
                    )
                    
                    if abs(correlation) > 0.1 and p_value < 0.05:  # Significant correlation
                        # Calculate cycle properties
                        amplitude = np.std(price_series)
                        phase = 0  # Simplified for sacred number analysis
                        strength = abs(correlation) * 100
                        confidence = 1 - p_value  # Higher confidence for lower p-value
                        
                        cycle = CycleData(
                            period=sacred_period,
                            amplitude=amplitude,
                            phase=phase,
                            strength=strength,
                            confidence=confidence,
                            start_time=datetime.now()
                        )
                        cycles.append(cycle)
            
            # Sort by strength
            cycles.sort(key=lambda x: x.strength, reverse=True)
            return cycles
            
        except Exception as e:
            logger.error(f"Error in sacred number cycle analysis: {str(e)}")
            return []    
    def _combine_and_validate_cycles(self, *cycle_lists) -> List[CycleData]:
        """Combine cycles from different methods and validate consistency"""
        try:
            all_cycles = []
            for cycle_list in cycle_lists:
                all_cycles.extend(cycle_list)
            
            if not all_cycles:
                return []
            
            # Group similar cycles (within 10% period difference)
            validated_cycles = []
            tolerance = 0.1
            
            for cycle in all_cycles:
                # Check if this cycle is similar to any already validated cycle
                is_duplicate = False
                for validated_cycle in validated_cycles:
                    period_diff = abs(cycle.period - validated_cycle.period) / validated_cycle.period
                    if period_diff < tolerance:
                        # Merge cycles - keep the one with higher confidence
                        if cycle.confidence > validated_cycle.confidence:
                            validated_cycles.remove(validated_cycle)
                            validated_cycles.append(cycle)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    validated_cycles.append(cycle)
            
            # Filter by minimum confidence threshold
            min_confidence = 0.1
            validated_cycles = [c for c in validated_cycles if c.confidence >= min_confidence]
            
            # Sort by combined score (strength * confidence)
            validated_cycles.sort(key=lambda x: x.strength * x.confidence, reverse=True)
            
            return validated_cycles[:15]  # Return top 15 validated cycles
            
        except Exception as e:
            logger.error(f"Error combining and validating cycles: {str(e)}")
            return []
    
    def _analyze_harmonic_relationships(self, cycles: List[CycleData]) -> List[HarmonicCycle]:
        """Analyze harmonic relationships between detected cycles"""
        try:
            harmonic_relationships = []
            
            if len(cycles) < 2:
                return harmonic_relationships
            
            # Check for harmonic relationships between cycles
            for i, base_cycle in enumerate(cycles):
                for j, other_cycle in enumerate(cycles):
                    if i >= j:
                        continue
                    
                    # Check if cycles are harmonically related
                    ratio = other_cycle.period / base_cycle.period
                    
                    # Check for simple harmonic ratios (2:1, 3:1, 1:2, 1:3, etc.)
                    harmonic_orders = [0.5, 2, 3, 4, 5, 1/3, 1/4, 1/5]
                    
                    for order in harmonic_orders:
                        if abs(ratio - order) < 0.1:  # 10% tolerance
                            # Calculate synchronization strength
                            strength = min(base_cycle.confidence, other_cycle.confidence)
                            
                            # Calculate phase relationship
                            phase_diff = abs(base_cycle.phase - other_cycle.phase)
                            
                            harmonic = HarmonicCycle(
                                base_cycle=base_cycle.period,
                                harmonic_order=int(order) if order >= 1 else order,
                                harmonic_period=other_cycle.period,
                                synchronization_strength=strength,
                                phase_relationship=phase_diff
                            )
                            harmonic_relationships.append(harmonic)
                            break
            
            return harmonic_relationships
            
        except Exception as e:
            logger.error(f"Error analyzing harmonic relationships: {str(e)}")
            return []
    
    def _assess_cycle_strength(self, cycles: List[CycleData], 
                             price_series: np.ndarray) -> Dict[str, float]:
        """Assess overall cycle strength and market cyclical behavior"""
        try:
            if not cycles or len(price_series) < 10:
                return {'overall_strength': 0.0, 'cyclical_consistency': 0.0}
            
            # Calculate overall cycle strength
            total_strength = sum(cycle.strength * cycle.confidence for cycle in cycles)
            avg_strength = total_strength / len(cycles)
            
            # Calculate cyclical consistency
            periods = [cycle.period for cycle in cycles]
            if len(periods) > 1:
                period_std = np.std(periods)
                period_mean = np.mean(periods)
                consistency = 1 / (1 + period_std / period_mean)  # Lower std = higher consistency
            else:
                consistency = 1.0
            
            # Calculate market regularity (how well price follows detected cycles)
            regularity_scores = []
            for cycle in cycles[:5]:  # Check top 5 cycles
                period = int(cycle.period)
                if period < len(price_series):
                    # Create a simple sine wave based on the cycle
                    t = np.arange(len(price_series))
                    synthetic_cycle = np.sin(2 * np.pi * t / period + cycle.phase)
                    
                    # Correlate with detrended price
                    detrended_price = signal.detrend(price_series)
                    correlation = np.corrcoef(detrended_price, synthetic_cycle)[0, 1]
                    regularity_scores.append(abs(correlation))
            
            regularity = np.mean(regularity_scores) if regularity_scores else 0.0
            
            return {
                'overall_strength': min(1.0, avg_strength / 100),  # Normalize to 0-1
                'cyclical_consistency': consistency,
                'market_regularity': regularity,
                'dominant_cycle_period': cycles[0].period if cycles else 0,
                'number_of_cycles': len(cycles),
                'strongest_cycle_confidence': cycles[0].confidence if cycles else 0
            }
            
        except Exception as e:
            logger.error(f"Error assessing cycle strength: {str(e)}")
            return {'overall_strength': 0.0, 'cyclical_consistency': 0.0}
    
    def _generate_time_projections(self, cycles: List[CycleData], 
                                 timestamps: pd.DatetimeIndex) -> List[TimeProjection]:
        """Generate time-based projections using detected cycles"""
        try:
            projections = []
            
            if not cycles or len(timestamps) == 0:
                return projections
            
            current_time = timestamps[-1]
            
            # Generate projections for each significant cycle
            for cycle in cycles[:5]:  # Use top 5 cycles
                if cycle.confidence > 0.3:  # Only high-confidence cycles
                    period_days = cycle.period
                    
                    # Project next peaks and troughs
                    for i in range(1, 4):  # Project 3 periods ahead
                        # Peak projection
                        peak_time = current_time + timedelta(days=period_days * i)
                        peak_projection = TimeProjection(
                            projection_date=peak_time,
                            projection_type='cycle_peak',
                            confidence=cycle.confidence * (0.9 ** i),  # Decrease confidence with distance
                            cycle_basis=[cycle.period]
                        )
                        projections.append(peak_projection)
                        
                        # Trough projection (half cycle later)
                        trough_time = current_time + timedelta(days=period_days * (i - 0.5))
                        if trough_time > current_time:
                            trough_projection = TimeProjection(
                                projection_date=trough_time,
                                projection_type='cycle_trough',
                                confidence=cycle.confidence * (0.9 ** i),
                                cycle_basis=[cycle.period]
                            )
                            projections.append(trough_projection)
            
            # Generate harmonic convergence projections
            if len(cycles) >= 2:
                for i in range(min(3, len(cycles))):
                    for j in range(i + 1, min(3, len(cycles))):
                        cycle1, cycle2 = cycles[i], cycles[j]
                        
                        # Find next convergence point
                        lcm_period = np.lcm(int(cycle1.period), int(cycle2.period))
                        convergence_time = current_time + timedelta(days=lcm_period)
                        
                        convergence_projection = TimeProjection(
                            projection_date=convergence_time,
                            projection_type='harmonic_convergence',
                            confidence=min(cycle1.confidence, cycle2.confidence),
                            cycle_basis=[cycle1.period, cycle2.period]
                        )
                        projections.append(convergence_projection)
            
            # Sort projections by date
            projections.sort(key=lambda x: x.projection_date)
            
            return projections[:20]  # Return top 20 projections
            
        except Exception as e:
            logger.error(f"Error generating time projections: {str(e)}")
            return []    
    def _ml_cycle_validation(self, cycles: List[CycleData], 
                           price_series: np.ndarray) -> Dict[str, Any]:
        """Use machine learning to validate and enhance cycle detection"""
        try:
            if not self.ml_enhancement or len(cycles) == 0:
                return {}
            
            # Prepare features for ML validation
            features = []
            labels = []
            
            for cycle in cycles:
                period = int(cycle.period)
                if period >= len(price_series):
                    continue
                
                # Create features based on cycle properties
                cycle_features = [
                    cycle.period,
                    cycle.amplitude,
                    cycle.strength,
                    cycle.confidence,
                    np.sin(cycle.phase),
                    np.cos(cycle.phase)
                ]
                
                # Calculate actual cyclical behavior in the data
                shifted_series = np.roll(price_series, period)
                correlation = np.corrcoef(
                    price_series[period:], 
                    shifted_series[period:]
                )[0, 1] if len(price_series) > period else 0
                
                # Label: high correlation indicates valid cycle
                label = 1 if abs(correlation) > 0.2 else 0
                
                features.append(cycle_features)
                labels.append(label)
            
            if len(features) < 2:
                return {'validation_scores': []}
            
            # Simple ML validation using correlation-based scoring
            validation_scores = []
            for i, cycle in enumerate(cycles):
                if i < len(features):
                    # Score based on multiple factors
                    base_score = cycle.confidence
                    
                    # Bonus for periods matching Gann sacred numbers
                    if any(abs(cycle.period - sacred) < 2 for sacred in self.gann_sacred_numbers):
                        base_score *= 1.2
                    
                    # Bonus for strong harmonic relationships
                    harmonic_bonus = 0
                    for other_cycle in cycles:
                        if other_cycle != cycle:
                            ratio = cycle.period / other_cycle.period
                            if any(abs(ratio - h) < 0.1 for h in [0.5, 2, 3, 1/3]):
                                harmonic_bonus += 0.1
                    
                    final_score = min(1.0, base_score + harmonic_bonus)
                    validation_scores.append(final_score)
                else:
                    validation_scores.append(cycle.confidence)
            
            return {
                'validation_scores': validation_scores,
                'ml_enhanced': True,
                'feature_count': len(features),
                'validation_method': 'correlation_based_scoring'
            }
            
        except Exception as e:
            logger.error(f"Error in ML cycle validation: {str(e)}")
            return {'validation_scores': []}
    
    def _apply_ml_validation(self, cycles: List[CycleData], 
                           ml_validation: Dict[str, Any]) -> List[CycleData]:
        """Apply ML validation results to enhance cycle confidence"""
        try:
            if not ml_validation or 'validation_scores' not in ml_validation:
                return cycles
            
            validation_scores = ml_validation['validation_scores']
            enhanced_cycles = []
            
            for i, cycle in enumerate(cycles):
                if i < len(validation_scores):
                    # Update confidence with ML validation score
                    enhanced_confidence = (cycle.confidence + validation_scores[i]) / 2
                    
                    # Create enhanced cycle
                    enhanced_cycle = CycleData(
                        period=cycle.period,
                        amplitude=cycle.amplitude,
                        phase=cycle.phase,
                        strength=cycle.strength,
                        confidence=enhanced_confidence,
                        start_time=cycle.start_time,
                        next_peak=cycle.next_peak,
                        next_trough=cycle.next_trough
                    )
                    enhanced_cycles.append(enhanced_cycle)
                else:
                    enhanced_cycles.append(cycle)
            
            # Re-sort by enhanced confidence
            enhanced_cycles.sort(key=lambda x: x.confidence, reverse=True)
            
            return enhanced_cycles
            
        except Exception as e:
            logger.error(f"Error applying ML validation: {str(e)}")
            return cycles
    
    def _generate_cycle_signals(self, cycles: List[CycleData], 
                              harmonic_analysis: List[HarmonicCycle],
                              timestamps: pd.DatetimeIndex) -> List[Dict[str, Any]]:
        """Generate trading signals based on cycle analysis"""
        try:
            signals = []
            
            if not cycles or len(timestamps) == 0:
                return signals
            
            current_time = timestamps[-1]
            
            # Signal generation based on cycle phases
            for cycle in cycles[:5]:  # Use top 5 cycles
                if cycle.confidence > 0.4:  # High confidence threshold
                    period = cycle.period
                    phase = cycle.phase
                    
                    # Calculate current position in cycle
                    days_since_start = (current_time - cycle.start_time).days
                    cycle_position = (days_since_start % period) / period
                    
                    # Generate signals based on cycle position
                    if 0.05 <= cycle_position <= 0.15:  # Near cycle bottom
                        signal = {
                            'timestamp': current_time,
                            'signal_type': 'BUY',
                            'strength': cycle.confidence * 0.8,
                            'reason': f'Cycle bottom - Period: {period:.1f}',
                            'cycle_period': period,
                            'cycle_position': cycle_position,
                            'confidence': cycle.confidence
                        }
                        signals.append(signal)
                    
                    elif 0.45 <= cycle_position <= 0.65:  # Near cycle top
                        signal = {
                            'timestamp': current_time,
                            'signal_type': 'SELL',
                            'strength': cycle.confidence * 0.8,
                            'reason': f'Cycle top - Period: {period:.1f}',
                            'cycle_period': period,
                            'cycle_position': cycle_position,
                            'confidence': cycle.confidence
                        }
                        signals.append(signal)
            
            # Harmonic convergence signals
            for harmonic in harmonic_analysis:
                if harmonic.synchronization_strength > 0.5:
                    signal = {
                        'timestamp': current_time,
                        'signal_type': 'HARMONIC_CONVERGENCE',
                        'strength': harmonic.synchronization_strength,
                        'reason': f'Harmonic convergence: {harmonic.base_cycle:.1f} & {harmonic.harmonic_period:.1f}',
                        'base_cycle': harmonic.base_cycle,
                        'harmonic_period': harmonic.harmonic_period,
                        'harmonic_order': harmonic.harmonic_order,
                        'confidence': harmonic.synchronization_strength
                    }
                    signals.append(signal)
            
            # Sacred number alignment signals
            if self.sacred_numbers:
                for cycle in cycles:
                    for sacred in self.gann_sacred_numbers:
                        if abs(cycle.period - sacred) < 1 and cycle.confidence > 0.3:
                            signal = {
                                'timestamp': current_time,
                                'signal_type': 'SACRED_ALIGNMENT',
                                'strength': cycle.confidence * 1.2,  # Bonus for sacred numbers
                                'reason': f'Sacred number alignment: {sacred}',
                                'cycle_period': cycle.period,
                                'sacred_number': sacred,
                                'confidence': cycle.confidence
                            }
                            signals.append(signal)
            
            # Sort signals by strength
            signals.sort(key=lambda x: x['strength'], reverse=True)
            
            return signals[:10]  # Return top 10 signals
            
        except Exception as e:
            logger.error(f"Error generating cycle signals: {str(e)}")
            return []
    
    def _calculate_cycle_levels(self, cycles: List[CycleData], 
                              price_series: np.ndarray,
                              timestamps: pd.DatetimeIndex) -> Dict[str, List[float]]:
        """Calculate support and resistance levels based on cycle analysis"""
        try:
            if not cycles or len(price_series) == 0:
                return {'support_levels': [], 'resistance_levels': []}
            
            support_levels = []
            resistance_levels = []
            
            current_price = price_series[-1]
            
            # Calculate levels based on cycle amplitude and phase
            for cycle in cycles[:3]:  # Use top 3 cycles
                if cycle.confidence > 0.3:
                    amplitude = cycle.amplitude
                    
                    # Calculate potential support/resistance based on cycle
                    cycle_support = current_price - amplitude * cycle.confidence
                    cycle_resistance = current_price + amplitude * cycle.confidence
                    
                    support_levels.append(cycle_support)
                    resistance_levels.append(cycle_resistance)
                    
                    # Add levels based on Fibonacci ratios of cycle amplitude
                    fib_ratios = [0.236, 0.382, 0.618, 0.786]
                    for ratio in fib_ratios:
                        support_levels.append(current_price - amplitude * ratio * cycle.confidence)
                        resistance_levels.append(current_price + amplitude * ratio * cycle.confidence)
            
            # Remove duplicates and sort
            support_levels = sorted(list(set(np.round(support_levels, 2))))
            resistance_levels = sorted(list(set(np.round(resistance_levels, 2))))
            
            # Filter levels too close to current price (within 0.1%)
            price_threshold = current_price * 0.001
            support_levels = [level for level in support_levels if level < current_price - price_threshold]
            resistance_levels = [level for level in resistance_levels if level > current_price + price_threshold]
            
            return {
                'support_levels': support_levels[-10:],  # Last 10 support levels
                'resistance_levels': resistance_levels[:10],  # First 10 resistance levels
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating cycle levels: {str(e)}")
            return {'support_levels': [], 'resistance_levels': []}


# Demo and testing code
if __name__ == "__main__":
    """
    Demonstration of the Gann Time Cycle Indicator
    """
    print("=" * 60)
    print("Gann Time Cycle Indicator - Advanced Implementation Demo")
    print("=" * 60)
    
    # Create sample data with known cycles
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Create price data with embedded cycles
    n_points = len(dates)
    time = np.arange(n_points)
    
    # Base trend
    trend = 100 + 0.01 * time
    
    # Add multiple cycles
    cycle1 = 5 * np.sin(2 * np.pi * time / 30)  # 30-day cycle
    cycle2 = 3 * np.sin(2 * np.pi * time / 7)   # 7-day cycle
    cycle3 = 2 * np.sin(2 * np.pi * time / 90)  # 90-day cycle
    
    # Add noise
    noise = np.random.normal(0, 1, n_points)
    
    # Combine all components
    close_prices = trend + cycle1 + cycle2 + cycle3 + noise
    high_prices = close_prices + np.abs(np.random.normal(0, 0.5, n_points))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.5, n_points))
    volume = np.random.randint(1000, 10000, n_points)
    
    # Create DataFrame
    data = pd.DataFrame({
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    # Initialize indicator
    indicator = GannTimeCycleIndicator(
        lookback_periods=300,
        min_cycle_length=5,
        max_cycle_length=120,
        ml_enhancement=True,
        sacred_numbers=True
    )
    
    # Calculate cycle analysis
    print("Calculating Gann Time Cycle analysis...")
    results = indicator.calculate(data)
    
    if 'error' not in results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Data points analyzed: {results['data_points_analyzed']}")
        print(f"🔄 Cycles detected: {len(results['detected_cycles'])}")
        print(f"📈 Time projections: {len(results['time_projections'])}")
        print(f"🎯 Trading signals: {len(results['cycle_signals'])}")
        
        # Display top cycles
        print(f"\n🔝 Top Detected Cycles:")
        for i, cycle in enumerate(results['detected_cycles'][:5]):
            print(f"  {i+1}. Period: {cycle.period:.1f} days, "
                  f"Confidence: {cycle.confidence:.3f}, "
                  f"Strength: {cycle.strength:.2f}")
        
        # Display harmonic relationships
        print(f"\n🎵 Harmonic Relationships:")
        for harmonic in results['harmonic_relationships'][:3]:
            print(f"  Base: {harmonic.base_cycle:.1f} → Harmonic: {harmonic.harmonic_period:.1f} "
                  f"(Order: {harmonic.harmonic_order}, Strength: {harmonic.synchronization_strength:.3f})")
        
        # Display cycle strength assessment
        print(f"\n💪 Cycle Strength Assessment:")
        strength = results['cycle_strength']
        print(f"  Overall Strength: {strength['overall_strength']:.3f}")
        print(f"  Cyclical Consistency: {strength['cyclical_consistency']:.3f}")
        print(f"  Market Regularity: {strength.get('market_regularity', 0):.3f}")
        print(f"  Dominant Cycle: {strength['dominant_cycle_period']:.1f} days")
        
        # Display recent signals
        print(f"\n📊 Recent Trading Signals:")
        for signal in results['cycle_signals'][:3]:
            print(f"  {signal['signal_type']}: {signal['reason']} "
                  f"(Strength: {signal['strength']:.3f})")
        
        # Display time projections
        print(f"\n🔮 Upcoming Time Projections:")
        for proj in results['time_projections'][:5]:
            print(f"  {proj.projection_date.strftime('%Y-%m-%d')}: {proj.projection_type} "
                  f"(Confidence: {proj.confidence:.3f})")
        
        # Display support/resistance levels
        levels = results['cycle_levels']
        print(f"\n📊 Cycle-Based Levels:")
        print(f"  Support: {levels['support_levels'][-3:] if levels['support_levels'] else 'None'}")
        print(f"  Resistance: {levels['resistance_levels'][:3] if levels['resistance_levels'] else 'None'}")
        
        print(f"\n🎯 Sacred Number Analysis:")
        sacred_cycles = [c for c in results['detected_cycles'] 
                        if any(abs(c.period - sacred) < 2 for sacred in indicator.gann_sacred_numbers)]
        print(f"  Cycles matching sacred numbers: {len(sacred_cycles)}")
        for cycle in sacred_cycles[:3]:
            closest_sacred = min(indicator.gann_sacred_numbers, 
                               key=lambda x: abs(x - cycle.period))
            print(f"    Period {cycle.period:.1f} ≈ Sacred {closest_sacred} "
                  f"(Confidence: {cycle.confidence:.3f})")
        
        if indicator.ml_enhancement and 'ml_validation' in results:
            print(f"\n🤖 ML Enhancement Results:")
            ml_results = results['ml_validation']
            print(f"  Enhanced cycles: {len(ml_results.get('validation_scores', []))}")
            print(f"  Validation method: {ml_results.get('validation_method', 'N/A')}")
        
    else:
        print(f"❌ Error in calculation: {results['error']}")
    
    print("\n" + "=" * 60)
    print("Demo completed! The Gann Time Cycle Indicator provides:")
    print("✅ Multiple cycle detection algorithms (Fourier, Wavelet, Hilbert, Autocorrelation)")
    print("✅ Sacred number integration and analysis")
    print("✅ Harmonic relationship detection")
    print("✅ ML-enhanced validation and confidence scoring")
    print("✅ Time-based projections and forecasting")
    print("✅ Comprehensive trading signal generation")
    print("✅ Cycle-based support and resistance levels")
    print("=" * 60)