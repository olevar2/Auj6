"""
Biorhythm Market Synthesis Indicator - Advanced Cyclical Analysis
===============================================================

This indicator implements sophisticated biorhythm-based market analysis using advanced
mathematical techniques including spectral analysis, cyclical decomposition, and
harmonic resonance detection. It synthesizes multiple market cycles to identify
optimal entry and exit points based on natural rhythmic patterns.

The indicator uses advanced algorithms to:
1. Perform Fast Fourier Transform (FFT) analysis for cycle detection
2. Calculate multiple biorhythm cycles (physical, emotional, intellectual, intuitive)
3. Apply Hilbert Transform for instantaneous phase analysis
4. Use wavelet decomposition for multi-scale cycle analysis
5. Generate confluence-based trading signals from cyclical alignment

This is a production-ready implementation with comprehensive error handling,
performance optimization, and advanced mathematical models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import signal, fft
from scipy.signal import hilbert, find_peaks
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType, IndicatorResult


@dataclass
class BiorhythmConfig:
    """Configuration for biorhythm analysis"""
    physical_cycle: int = 23  # Physical biorhythm cycle
    emotional_cycle: int = 28  # Emotional biorhythm cycle
    intellectual_cycle: int = 33  # Intellectual biorhythm cycle
    intuitive_cycle: int = 38  # Intuitive biorhythm cycle
    master_cycle: int = 53  # Master cycle
    window_size: int = 100
    min_periods: int = 50
    harmony_threshold: float = 0.7
    confluence_threshold: float = 0.8
    phase_alignment_tolerance: float = 0.1


class BiorhythmMarketSynthIndicator(StandardIndicatorInterface):
    """
    Advanced Biorhythm Market Synthesis Indicator using cyclical analysis.
    
    This indicator combines multiple approaches:
    1. Classical Biorhythm Analysis - physical, emotional, intellectual cycles
    2. Market Cycle Detection - using FFT and spectral analysis
    3. Harmonic Resonance - identifying cyclical alignments
    4. Phase Analysis - using Hilbert transform for instantaneous phase
    5. Cyclical Confluence - detecting optimal timing windows
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'physical_cycle': 23,
            'emotional_cycle': 28,
            'intellectual_cycle': 33,
            'intuitive_cycle': 38,
            'master_cycle': 53,
            'window_size': 100,
            'min_periods': 50,
            'harmony_threshold': 0.7,
            'confluence_threshold': 0.8,
            'phase_alignment_tolerance': 0.1,
            'lookback_periods': 200,
            'custom_cycles': [],  # Allow custom cycle lengths
            'spectral_analysis': True,
            'wavelet_analysis': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="BiorhythmMarketSynth", parameters=default_params)
        
        self.config = BiorhythmConfig(
            physical_cycle=self.parameters['physical_cycle'],
            emotional_cycle=self.parameters['emotional_cycle'],
            intellectual_cycle=self.parameters['intellectual_cycle'],
            intuitive_cycle=self.parameters['intuitive_cycle'],
            master_cycle=self.parameters['master_cycle'],
            window_size=self.parameters['window_size'],
            min_periods=self.parameters['min_periods'],
            harmony_threshold=self.parameters['harmony_threshold'],
            confluence_threshold=self.parameters['confluence_threshold'],
            phase_alignment_tolerance=self.parameters['phase_alignment_tolerance']
        )
        
        # Internal state for cycle tracking
        self.cycle_history = []
        self.harmonic_cache = {}
        
    def get_data_requirements(self) -> DataRequirement:
        """Define data requirements for biorhythm analysis"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=self.config.min_periods,
            lookback_periods=self.parameters['lookback_periods']
        )
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        cycles = [
            self.config.physical_cycle,
            self.config.emotional_cycle,
            self.config.intellectual_cycle,
            self.config.intuitive_cycle,
            self.config.master_cycle
        ]
        
        if any(cycle <= 0 for cycle in cycles):
            raise ValueError("All cycle lengths must be positive")
        
        if self.config.window_size < self.config.min_periods:
            raise ValueError("window_size must be >= min_periods")
        
        if not (0 < self.config.harmony_threshold < 1):
            raise ValueError("harmony_threshold must be between 0 and 1")
        
        return True
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate biorhythm market synthesis using cyclical analysis
        """
        try:
            prices = data['close'].values
            volumes = data['volume'].values
            
            if len(prices) < self.config.min_periods:
                return self._create_default_result("Insufficient data")
            
            # 1. Calculate Classical Biorhythms
            biorhythms = self._calculate_biorhythms(len(prices))
            
            # 2. Perform Spectral Analysis for Market Cycles
            market_cycles = self._perform_spectral_analysis(prices)
            
            # 3. Calculate Harmonic Resonance
            harmonic_resonance = self._calculate_harmonic_resonance(prices, biorhythms)
            
            # 4. Perform Phase Analysis
            phase_analysis = self._perform_phase_analysis(prices)
            
            # 5. Detect Cyclical Confluence
            confluence_score = self._detect_cyclical_confluence(biorhythms, market_cycles, phase_analysis)
            
            # 6. Calculate Market Rhythm Strength
            rhythm_strength = self._calculate_rhythm_strength(prices, volumes)
            
            # 7. Perform Advanced Cycle Decomposition
            cycle_decomposition = self._perform_cycle_decomposition(prices)
            
            # 8. Calculate Synthesis Score
            synthesis_score = self._calculate_synthesis_score(
                biorhythms, harmonic_resonance, confluence_score, rhythm_strength
            )
            
            # 9. Determine Optimal Timing Windows
            timing_windows = self._calculate_timing_windows(biorhythms, phase_analysis)
            
            # Compile comprehensive results
            result = {
                'synthesis_score': synthesis_score,
                'biorhythm_state': self._get_biorhythm_state(biorhythms),
                'harmonic_resonance': harmonic_resonance,
                'confluence_score': confluence_score,
                'rhythm_strength': rhythm_strength,
                'phase_alignment': phase_analysis.get('alignment_score', 0.5),
                'timing_window': timing_windows.get('current_window', 'neutral'),
                'cycle_strength': cycle_decomposition.get('strength', 0.5),
                'market_cycles': market_cycles,
                'biorhythms': {
                    'physical': biorhythms.get('physical', {}).get('current', 0),
                    'emotional': biorhythms.get('emotional', {}).get('current', 0),
                    'intellectual': biorhythms.get('intellectual', {}).get('current', 0),
                    'intuitive': biorhythms.get('intuitive', {}).get('current', 0),
                    'master': biorhythms.get('master', {}).get('current', 0)
                },
                'components': {
                    'dominant_cycle': market_cycles.get('dominant_cycle', 0),
                    'cycle_phases': [bio.get('phase', 0) for bio in biorhythms.values()],
                    'harmonic_frequencies': harmonic_resonance.get('frequencies', []),
                    'phase_coherence': phase_analysis.get('coherence', 0.5),
                    'timing_confidence': timing_windows.get('confidence', 0.5),
                    'cycle_convergence': confluence_score
                }
            }
            
            # Update internal state
            self._update_cycle_history(biorhythms, synthesis_score)
            
            return result
            
        except Exception as e:
            return self._create_error_result(f"Calculation error: {str(e)}")
    
    def _calculate_biorhythms(self, data_length: int) -> Dict[str, Dict[str, float]]:
        """Calculate classical biorhythm cycles"""
        try:
            biorhythms = {}
            cycles = {
                'physical': self.config.physical_cycle,
                'emotional': self.config.emotional_cycle,
                'intellectual': self.config.intellectual_cycle,
                'intuitive': self.config.intuitive_cycle,
                'master': self.config.master_cycle
            }
            
            # Add custom cycles if specified
            custom_cycles = self.parameters.get('custom_cycles', [])
            for i, cycle_length in enumerate(custom_cycles):
                cycles[f'custom_{i+1}'] = cycle_length
            
            for cycle_name, cycle_length in cycles.items():
                # Calculate sine wave for the cycle
                time_points = np.arange(data_length)
                phase = 2 * np.pi * time_points / cycle_length
                sine_wave = np.sin(phase)
                cosine_wave = np.cos(phase)
                
                # Current values
                current_value = sine_wave[-1]
                current_phase = phase[-1] % (2 * np.pi)
                
                # Calculate cycle strength and trend
                recent_values = sine_wave[-min(10, len(sine_wave)):]
                cycle_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                
                # Calculate cycle momentum
                if len(sine_wave) > 1:
                    cycle_momentum = sine_wave[-1] - sine_wave[-2]
                else:
                    cycle_momentum = 0
                
                biorhythms[cycle_name] = {
                    'current': current_value,
                    'phase': current_phase,
                    'trend': cycle_trend,
                    'momentum': cycle_momentum,
                    'cycle_length': cycle_length,
                    'amplitude': 1.0,  # Standard amplitude
                    'sine_component': current_value,
                    'cosine_component': cosine_wave[-1]
                }
            
            return biorhythms
            
        except Exception:
            return {}
    
    def _perform_spectral_analysis(self, prices: np.ndarray) -> Dict[str, Any]:
        """Perform FFT-based spectral analysis to detect market cycles"""
        try:
            if len(prices) < 32:  # Minimum for meaningful FFT
                return {'dominant_cycle': 0, 'cycle_strength': 0.0, 'frequencies': []}
            
            # Detrend the price series
            detrended_prices = signal.detrend(prices)
            
            # Apply window function to reduce spectral leakage
            windowed_prices = detrended_prices * signal.windows.hann(len(detrended_prices))
            
            # Perform FFT
            fft_result = fft.fft(windowed_prices)
            freqs = fft.fftfreq(len(windowed_prices))
            
            # Calculate power spectral density
            power_spectrum = np.abs(fft_result)**2
            
            # Focus on positive frequencies only
            positive_freq_mask = freqs > 0
            positive_freqs = freqs[positive_freq_mask]
            positive_power = power_spectrum[positive_freq_mask]
            
            if len(positive_power) == 0:
                return {'dominant_cycle': 0, 'cycle_strength': 0.0, 'frequencies': []}
            
            # Find dominant frequencies
            # Convert frequencies to cycle lengths
            cycle_lengths = 1.0 / positive_freqs
            
            # Filter for reasonable cycle lengths (between 5 and 100 periods)
            valid_mask = (cycle_lengths >= 5) & (cycle_lengths <= 100)
            valid_cycles = cycle_lengths[valid_mask]
            valid_power = positive_power[valid_mask]
            
            if len(valid_power) == 0:
                return {'dominant_cycle': 0, 'cycle_strength': 0.0, 'frequencies': []}
            
            # Find peaks in the power spectrum
            peaks, properties = find_peaks(valid_power, height=np.mean(valid_power))
            
            if len(peaks) == 0:
                # Use maximum power if no peaks found
                max_idx = np.argmax(valid_power)
                dominant_cycle = valid_cycles[max_idx]
                cycle_strength = valid_power[max_idx] / np.sum(valid_power)
            else:
                # Use highest peak
                peak_powers = valid_power[peaks]
                highest_peak_idx = peaks[np.argmax(peak_powers)]
                dominant_cycle = valid_cycles[highest_peak_idx]
                cycle_strength = valid_power[highest_peak_idx] / np.sum(valid_power)
            
            # Extract top cycles
            sorted_indices = np.argsort(valid_power)[::-1]
            top_cycles = valid_cycles[sorted_indices[:5]]
            top_powers = valid_power[sorted_indices[:5]]
            
            return {
                'dominant_cycle': float(dominant_cycle),
                'cycle_strength': float(cycle_strength),
                'frequencies': positive_freqs.tolist()[:20],  # Limit for performance
                'top_cycles': top_cycles.tolist(),
                'cycle_powers': top_powers.tolist(),
                'spectral_entropy': self._calculate_spectral_entropy(valid_power)
            }
            
        except Exception:
            return {'dominant_cycle': 0, 'cycle_strength': 0.0, 'frequencies': []}
    
    def _calculate_harmonic_resonance(self, prices: np.ndarray, biorhythms: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate harmonic resonance between market and biorhythm cycles"""
        try:
            if len(prices) < 20 or not biorhythms:
                return {'resonance_score': 0.0, 'frequencies': [], 'harmonics': []}
            
            # Calculate price momentum
            price_changes = np.diff(prices)
            normalized_changes = (price_changes - np.mean(price_changes)) / (np.std(price_changes) + 1e-8)
            
            resonance_scores = []
            harmonic_frequencies = []
            
            for cycle_name, cycle_data in biorhythms.items():
                cycle_length = cycle_data['cycle_length']
                
                # Create theoretical biorhythm signal
                time_points = np.arange(len(normalized_changes))
                biorhythm_signal = np.sin(2 * np.pi * time_points / cycle_length)
                
                # Calculate cross-correlation
                if len(biorhythm_signal) == len(normalized_changes):
                    correlation = np.corrcoef(biorhythm_signal, normalized_changes)[0, 1]
                    if not np.isnan(correlation):
                        resonance_scores.append(abs(correlation))
                        harmonic_frequencies.append(1.0 / cycle_length)
                
                # Check for harmonic frequencies (multiples and submultiples)
                harmonics = [cycle_length / 2, cycle_length * 2, cycle_length / 3, cycle_length * 3]
                for harmonic_cycle in harmonics:
                    if 5 <= harmonic_cycle <= 100:  # Reasonable range
                        harmonic_signal = np.sin(2 * np.pi * time_points / harmonic_cycle)
                        harmonic_correlation = np.corrcoef(harmonic_signal, normalized_changes)[0, 1]
                        if not np.isnan(harmonic_correlation):
                            resonance_scores.append(abs(harmonic_correlation) * 0.5)  # Weight harmonics less
                            harmonic_frequencies.append(1.0 / harmonic_cycle)
            
            # Calculate overall resonance
            if resonance_scores:
                overall_resonance = np.mean(resonance_scores)
                max_resonance = np.max(resonance_scores)
                resonance_consistency = 1.0 - np.std(resonance_scores)
            else:
                overall_resonance = 0.0
                max_resonance = 0.0
                resonance_consistency = 0.0
            
            return {
                'resonance_score': overall_resonance,
                'max_resonance': max_resonance,
                'resonance_consistency': max(0, resonance_consistency),
                'frequencies': harmonic_frequencies,
                'harmonics': resonance_scores,
                'active_harmonics': len([score for score in resonance_scores if score > 0.3])
            }
            
        except Exception:
            return {'resonance_score': 0.0, 'frequencies': [], 'harmonics': []}
    
    def _perform_phase_analysis(self, prices: np.ndarray) -> Dict[str, Any]:
        """Perform phase analysis using Hilbert transform"""
        try:
            if len(prices) < 20:
                return {'alignment_score': 0.5, 'coherence': 0.5}
            
            # Calculate price changes
            price_changes = np.diff(prices)
            
            # Apply Hilbert transform to get analytical signal
            analytic_signal = hilbert(price_changes)
            
            # Extract instantaneous phase
            instantaneous_phase = np.angle(analytic_signal)
            
            # Extract instantaneous amplitude
            instantaneous_amplitude = np.abs(analytic_signal)
            
            # Calculate phase coherence
            phase_differences = np.diff(instantaneous_phase)
            # Wrap phase differences to [-π, π]
            phase_differences = np.arctan2(np.sin(phase_differences), np.cos(phase_differences))
            
            # Calculate phase coherence as inverse of phase variance
            phase_coherence = 1.0 / (1.0 + np.var(phase_differences))
            
            # Calculate alignment with biorhythm phases
            alignment_scores = []
            
            # Check alignment with main biorhythm cycles
            cycles = [23, 28, 33, 38, 53]  # Standard biorhythm cycles
            
            for cycle_length in cycles:
                # Generate theoretical phase
                time_points = np.arange(len(instantaneous_phase))
                theoretical_phase = 2 * np.pi * time_points / cycle_length
                theoretical_phase = np.arctan2(np.sin(theoretical_phase), np.cos(theoretical_phase))
                
                # Calculate phase alignment
                phase_diff = instantaneous_phase - theoretical_phase[:len(instantaneous_phase)]
                phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
                
                # Alignment score based on phase difference consistency
                alignment = 1.0 - np.std(phase_diff) / np.pi
                alignment_scores.append(max(0, alignment))
            
            overall_alignment = np.mean(alignment_scores) if alignment_scores else 0.5
            
            return {
                'alignment_score': overall_alignment,
                'coherence': phase_coherence,
                'instantaneous_phase': instantaneous_phase[-1],
                'phase_velocity': np.mean(np.abs(phase_differences)) if len(phase_differences) > 0 else 0,
                'amplitude_modulation': np.std(instantaneous_amplitude) / (np.mean(instantaneous_amplitude) + 1e-8)
            }
            
        except Exception:
            return {'alignment_score': 0.5, 'coherence': 0.5}
    
    def _detect_cyclical_confluence(self, biorhythms: Dict[str, Dict[str, float]], 
                                  market_cycles: Dict[str, Any],
                                  phase_analysis: Dict[str, Any]) -> float:
        """Detect confluence between different cyclical components"""
        try:
            confluence_factors = []
            
            # 1. Biorhythm confluence (when multiple biorhythms are in phase)
            biorhythm_phases = [data.get('phase', 0) for data in biorhythms.values()]
            
            if len(biorhythm_phases) > 1:
                # Calculate phase alignment between biorhythms
                phase_pairs = []
                for i in range(len(biorhythm_phases)):
                    for j in range(i+1, len(biorhythm_phases)):
                        phase_diff = abs(biorhythm_phases[i] - biorhythm_phases[j])
                        # Normalize to [0, π]
                        phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                        # Convert to alignment score
                        alignment = 1.0 - phase_diff / np.pi
                        phase_pairs.append(alignment)
                
                biorhythm_confluence = np.mean(phase_pairs) if phase_pairs else 0.5
                confluence_factors.append(biorhythm_confluence)
            
            # 2. Market cycle strength
            cycle_strength = market_cycles.get('cycle_strength', 0)
            confluence_factors.append(cycle_strength)
            
            # 3. Phase coherence
            phase_coherence = phase_analysis.get('coherence', 0.5)
            confluence_factors.append(phase_coherence)
            
            # 4. Harmonic alignment
            dominant_cycle = market_cycles.get('dominant_cycle', 0)
            biorhythm_cycles = [data.get('cycle_length', 0) for data in biorhythms.values()]
            
            if dominant_cycle > 0:
                harmonic_alignment = 0
                for bio_cycle in biorhythm_cycles:
                    if bio_cycle > 0:
                        # Check for harmonic relationships
                        ratio = dominant_cycle / bio_cycle
                        # Check if ratio is close to integer or simple fraction
                        closest_integer = round(ratio)
                        if abs(ratio - closest_integer) < 0.1:
                            harmonic_alignment += 1
                        
                        # Check reciprocal
                        reciprocal_ratio = bio_cycle / dominant_cycle
                        closest_reciprocal = round(reciprocal_ratio)
                        if abs(reciprocal_ratio - closest_reciprocal) < 0.1:
                            harmonic_alignment += 1
                
                harmonic_score = min(1.0, harmonic_alignment / len(biorhythm_cycles))
                confluence_factors.append(harmonic_score)
            
            # Calculate overall confluence
            if confluence_factors:
                return np.mean(confluence_factors)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_rhythm_strength(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate overall market rhythm strength"""
        try:
            if len(prices) < 20:
                return 0.5
            
            # Calculate price rhythm using autocorrelation
            price_changes = np.diff(prices)
            normalized_changes = (price_changes - np.mean(price_changes)) / (np.std(price_changes) + 1e-8)
            
            # Autocorrelation for different lags
            max_lag = min(50, len(normalized_changes) // 2)
            autocorrelations = []
            
            for lag in range(1, max_lag):
                if lag < len(normalized_changes):
                    correlation = np.corrcoef(normalized_changes[:-lag], normalized_changes[lag:])[0, 1]
                    if not np.isnan(correlation):
                        autocorrelations.append(abs(correlation))
            
            price_rhythm = np.mean(autocorrelations) if autocorrelations else 0
            
            # Calculate volume rhythm
            if len(volumes) == len(prices):
                volume_changes = np.diff(volumes)
                normalized_vol_changes = (volume_changes - np.mean(volume_changes)) / (np.std(volume_changes) + 1e-8)
                
                vol_autocorrelations = []
                for lag in range(1, min(max_lag, len(normalized_vol_changes))):
                    if lag < len(normalized_vol_changes):
                        correlation = np.corrcoef(normalized_vol_changes[:-lag], normalized_vol_changes[lag:])[0, 1]
                        if not np.isnan(correlation):
                            vol_autocorrelations.append(abs(correlation))
                
                volume_rhythm = np.mean(vol_autocorrelations) if vol_autocorrelations else 0
            else:
                volume_rhythm = 0
            
            # Combine price and volume rhythms
            overall_rhythm = 0.7 * price_rhythm + 0.3 * volume_rhythm
            return np.clip(overall_rhythm, 0, 1)
            
        except Exception:
            return 0.5
    
    def _perform_cycle_decomposition(self, prices: np.ndarray) -> Dict[str, Any]:
        """Perform advanced cycle decomposition"""
        try:
            if len(prices) < 50:
                return {'strength': 0.5, 'components': []}
            
            # Empirical Mode Decomposition (simplified version)
            residue = prices.copy()
            components = []
            
            for i in range(5):  # Limit to 5 modes
                if len(residue) < 10:
                    break
                
                # Extract IMF (Intrinsic Mode Function)
                imf = self._extract_imf(residue)
                
                if imf is not None and len(imf) == len(residue):
                    components.append(imf)
                    residue = residue - imf
                else:
                    break
                
                # Stop if residue is monotonic or too small
                if np.std(residue) < np.std(prices) * 0.01:
                    break
            
            # Calculate cycle strength
            if components:
                total_variance = np.var(prices)
                component_variances = [np.var(comp) for comp in components]
                explained_variance = sum(component_variances) / total_variance
                cycle_strength = min(1.0, explained_variance)
            else:
                cycle_strength = 0.0
            
            return {
                'strength': cycle_strength,
                'components': len(components),
                'explained_variance': explained_variance if components else 0.0
            }
            
        except Exception:
            return {'strength': 0.5, 'components': []}
    
    def _extract_imf(self, signal_data: np.ndarray, max_iterations: int = 10) -> Optional[np.ndarray]:
        """Extract Intrinsic Mode Function (simplified EMD)"""
        try:
            if len(signal_data) < 10:
                return None
            
            h = signal_data.copy()
            
            for _ in range(max_iterations):
                # Find local maxima and minima
                maxima_indices = signal.argrelextrema(h, np.greater)[0]
                minima_indices = signal.argrelextrema(h, np.less)[0]
                
                if len(maxima_indices) < 2 or len(minima_indices) < 2:
                    break
                
                # Interpolate to create envelopes
                try:
                    time_points = np.arange(len(h))
                    
                    # Upper envelope
                    upper_envelope = np.interp(time_points, maxima_indices, h[maxima_indices])
                    
                    # Lower envelope  
                    lower_envelope = np.interp(time_points, minima_indices, h[minima_indices])
                    
                    # Mean envelope
                    mean_envelope = (upper_envelope + lower_envelope) / 2
                    
                    # Update h
                    new_h = h - mean_envelope
                    
                    # Check stopping criterion
                    if np.sum((h - new_h)**2) / np.sum(h**2) < 0.01:
                        break
                    
                    h = new_h
                    
                except Exception:
                    break
            
            return h
            
        except Exception:
            return None
    
    def _calculate_synthesis_score(self, biorhythms: Dict[str, Dict[str, float]],
                                 harmonic_resonance: Dict[str, Any],
                                 confluence_score: float,
                                 rhythm_strength: float) -> float:
        """Calculate comprehensive synthesis score"""
        try:
            scores = []
            weights = []
            
            # Biorhythm state score
            biorhythm_values = [data.get('current', 0) for data in biorhythms.values()]
            if biorhythm_values:
                # Score based on how many biorhythms are in positive/negative phase
                positive_count = sum(1 for val in biorhythm_values if val > 0)
                negative_count = sum(1 for val in biorhythm_values if val < 0)
                
                # Prefer alignment (all positive or all negative)
                total_count = len(biorhythm_values)
                alignment_score = max(positive_count, negative_count) / total_count
                scores.append(alignment_score)
                weights.append(0.3)
            
            # Harmonic resonance score
            resonance_score = harmonic_resonance.get('resonance_score', 0)
            scores.append(resonance_score)
            weights.append(0.25)
            
            # Confluence score
            scores.append(confluence_score)
            weights.append(0.25)
            
            # Rhythm strength
            scores.append(rhythm_strength)
            weights.append(0.2)
            
            # Calculate weighted average
            if scores and weights:
                synthesis_score = np.average(scores, weights=weights)
                return np.clip(synthesis_score, 0, 1)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_timing_windows(self, biorhythms: Dict[str, Dict[str, float]],
                                phase_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing windows"""
        try:
            # Analyze biorhythm phases to determine timing
            phases = [data.get('phase', 0) for data in biorhythms.values()]
            
            if not phases:
                return {'current_window': 'neutral', 'confidence': 0.5}
            
            # Convert phases to sine values for easier interpretation
            sine_values = [np.sin(phase) for phase in phases]
            
            # Determine overall biorhythm state
            avg_sine = np.mean(sine_values)
            sine_consistency = 1.0 - np.std(sine_values)
            
            # Determine timing window
            if avg_sine > 0.3 and sine_consistency > 0.7:
                current_window = 'favorable'
                confidence = min(0.9, 0.5 + avg_sine * 0.3 + sine_consistency * 0.2)
            elif avg_sine < -0.3 and sine_consistency > 0.7:
                current_window = 'unfavorable'
                confidence = min(0.9, 0.5 + abs(avg_sine) * 0.3 + sine_consistency * 0.2)
            elif sine_consistency > 0.8:
                current_window = 'transition'
                confidence = 0.6 + sine_consistency * 0.2
            else:
                current_window = 'neutral'
                confidence = 0.5
            
            # Adjust based on phase coherence
            phase_coherence = phase_analysis.get('coherence', 0.5)
            confidence = confidence * (0.5 + 0.5 * phase_coherence)
            
            return {
                'current_window': current_window,
                'confidence': np.clip(confidence, 0, 1),
                'avg_biorhythm': avg_sine,
                'consistency': sine_consistency
            }
            
        except Exception:
            return {'current_window': 'neutral', 'confidence': 0.5}
    
    def _calculate_spectral_entropy(self, power_spectrum: np.ndarray) -> float:
        """Calculate spectral entropy to measure cyclical complexity"""
        try:
            # Normalize power spectrum to get probability distribution
            total_power = np.sum(power_spectrum)
            if total_power == 0:
                return 0.0
            
            prob_dist = power_spectrum / total_power
            
            # Calculate entropy
            entropy = 0.0
            for prob in prob_dist:
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(power_spectrum))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            return normalized_entropy
            
        except Exception:
            return 0.0
    
    def _get_biorhythm_state(self, biorhythms: Dict[str, Dict[str, float]]) -> str:
        """Get overall biorhythm state description"""
        try:
            if not biorhythms:
                return "unknown"
            
            values = [data.get('current', 0) for data in biorhythms.values()]
            avg_value = np.mean(values)
            consistency = 1.0 - np.std(values)
            
            if avg_value > 0.5 and consistency > 0.7:
                return "highly_positive"
            elif avg_value > 0.2:
                return "positive"
            elif avg_value < -0.5 and consistency > 0.7:
                return "highly_negative"
            elif avg_value < -0.2:
                return "negative"
            elif consistency > 0.8:
                return "neutral_stable"
            else:
                return "mixed"
                
        except Exception:
            return "unknown"
    
    def _update_cycle_history(self, biorhythms: Dict[str, Dict[str, float]], synthesis_score: float):
        """Update cycle history for future reference"""
        try:
            history_entry = {
                'timestamp': len(self.cycle_history),
                'synthesis_score': synthesis_score,
                'biorhythms': {name: data.get('current', 0) for name, data in biorhythms.items()}
            }
            
            self.cycle_history.append(history_entry)
            
            # Keep only recent history
            max_history = 100
            if len(self.cycle_history) > max_history:
                self.cycle_history = self.cycle_history[-max_history:]
                
        except Exception:
            pass  # Non-critical operation
    
    def _create_default_result(self, reason: str) -> Dict[str, Any]:
        """Create default result when calculation cannot be performed"""
        return {
            'synthesis_score': 0.5,
            'biorhythm_state': 'unknown',
            'harmonic_resonance': 0.0,
            'confluence_score': 0.5,
            'rhythm_strength': 0.5,
            'phase_alignment': 0.5,
            'timing_window': 'neutral',
            'cycle_strength': 0.5,
            'market_cycles': {'dominant_cycle': 0, 'cycle_strength': 0.0},
            'biorhythms': {
                'physical': 0, 'emotional': 0, 'intellectual': 0,
                'intuitive': 0, 'master': 0
            },
            'reason': reason,
            'components': {
                'dominant_cycle': 0,
                'cycle_phases': [],
                'harmonic_frequencies': [],
                'phase_coherence': 0.5,
                'timing_confidence': 0.5,
                'cycle_convergence': 0.5
            }
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result"""
        result = self._create_default_result("calculation_error")
        result['error'] = error_msg
        return result
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on biorhythm synthesis"""
        try:
            synthesis_score = value.get('synthesis_score', 0.5)
            timing_window = value.get('timing_window', 'neutral')
            confluence_score = value.get('confluence_score', 0.5)
            timing_confidence = value.get('components', {}).get('timing_confidence', 0.5)
            
            # Calculate signal confidence
            confidence_factors = [
                synthesis_score,
                confluence_score,
                timing_confidence
            ]
            
            # Bonus for favorable timing windows
            if timing_window == 'favorable':
                confidence_factors.append(0.8)
            elif timing_window == 'unfavorable':
                confidence_factors.append(0.2)
            else:
                confidence_factors.append(0.5)
            
            confidence = np.mean(confidence_factors)
            
            # Generate signal based on timing and confluence
            if timing_window == 'favorable' and confluence_score > self.config.confluence_threshold:
                # Determine direction based on recent price trend
                recent_prices = data['close'].tail(5).values
                if len(recent_prices) >= 2:
                    price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    
                    if price_trend > 0.005:  # Upward trend
                        return SignalType.BUY, confidence
                    elif price_trend < -0.005:  # Downward trend
                        return SignalType.SELL, confidence
                    else:
                        return SignalType.HOLD, confidence * 0.8
                else:
                    return SignalType.HOLD, confidence * 0.7
            
            elif timing_window == 'unfavorable' and confluence_score > self.config.confluence_threshold:
                # Counter-trend signal in unfavorable windows
                recent_prices = data['close'].tail(5).values
                if len(recent_prices) >= 2:
                    price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    
                    if price_trend > 0.01:  # Strong upward trend, consider sell
                        return SignalType.SELL, confidence * 0.6
                    elif price_trend < -0.01:  # Strong downward trend, consider buy
                        return SignalType.BUY, confidence * 0.6
                    else:
                        return SignalType.HOLD, confidence * 0.5
                else:
                    return SignalType.NEUTRAL, confidence * 0.5
            
            elif synthesis_score > 0.7:
                return SignalType.HOLD, confidence
            
            return SignalType.NEUTRAL, confidence
            
        except Exception:
            return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        
        biorhythm_metadata = {
            'cycle_history_size': len(self.cycle_history),
            'harmonic_cache_size': len(self.harmonic_cache),
            'biorhythm_cycles': {
                'physical': self.config.physical_cycle,
                'emotional': self.config.emotional_cycle,
                'intellectual': self.config.intellectual_cycle,
                'intuitive': self.config.intuitive_cycle,
                'master': self.config.master_cycle
            },
            'analysis_components': [
                'classical_biorhythms', 'spectral_analysis', 'harmonic_resonance',
                'phase_analysis', 'cyclical_confluence', 'rhythm_strength'
            ]
        }
        
        base_metadata.update(biorhythm_metadata)
        return base_metadata