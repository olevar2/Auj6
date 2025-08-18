"""
Photonic Wavelength Analyzer

Quantum-inspired implementation to analyze market wavelengths and wave interference 
patterns using advanced quantum mechanics principles for the humanitarian trading platform.

This indicator applies quantum field theory concepts, wave-particle duality, and 
photonic analysis to understand market dynamics at the fundamental level.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Complex
import logging
from dataclasses import dataclass
from scipy import signal, fft
from scipy.signal import hilbert, morlet, find_peaks, spectrogram
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class PhotonicConfig:
    """Configuration for Photonic Wavelength Analyzer."""
    quantum_window: int = 128
    wavelength_range: Tuple[int, int] = (5, 100)
    interference_threshold: float = 0.1
    coherence_length: int = 50
    planck_resolution: float = 0.01
    photon_energy_levels: int = 16
    wave_packet_width: float = 2.0
    entanglement_distance: int = 20


@dataclass
class QuantumState:
    """Quantum state representation of market."""
    amplitude: Complex
    phase: float
    energy: float
    momentum: float
    uncertainty: float
    coherence: float
    entanglement: float


@dataclass
class PhotonicWave:
    """Photonic wave properties."""
    wavelength: float
    frequency: float
    amplitude: float
    phase: float
    energy: float
    interference_pattern: np.ndarray
    coherence_length: float
    polarization: float


@dataclass
class InterferencePattern:
    """Wave interference analysis result."""
    constructive_points: List[int]
    destructive_points: List[int]
    pattern_strength: float
    coherence_measure: float
    phase_correlation: float


class PhotonicWavelengthAnalyzer(StandardIndicatorInterface):
    """
    Quantum-inspired market wavelength analyzer using photonic principles.
    
    This indicator applies quantum mechanics concepts to analyze market waves,
    including wave-particle duality, superposition, and interference patterns.
    """
    
    def __init__(self, config: Optional[PhotonicConfig] = None):
        super().__init__()
        self.config = config or PhotonicConfig()
        self.logger = logging.getLogger(__name__)
        
        # Quantum constants (adapted for market analysis)
        self.planck_constant = 6.62607015e-34  # Scaled for market context
        self.light_speed = 299792458  # Metaphorical for information speed
        self.quantum_resolution = self.config.planck_resolution
        
        # Wave analysis components
        self.detected_wavelengths: List[PhotonicWave] = []
        self.quantum_states: List[QuantumState] = []
        self.interference_history: List[InterferencePattern] = []
        
        # Fourier transform cache
        self.fft_cache: Dict[str, np.ndarray] = {}
        
        # Energy level quantization
        self.energy_levels = np.linspace(0.1, 2.0, self.config.photon_energy_levels)
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market wavelengths using quantum-inspired algorithms.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing photonic analysis results
        """
        try:
            if len(data) < self.config.quantum_window:
                raise IndicatorCalculationError(
                    f"Insufficient data: need at least {self.config.quantum_window} periods"
                )
            
            # Prepare quantum field representation
            quantum_field = self._create_quantum_field(data)
            
            # Perform wavelength decomposition
            wavelength_spectrum = self._decompose_wavelengths(quantum_field)
            
            # Analyze wave interference patterns
            interference_analysis = self._analyze_interference_patterns(wavelength_spectrum)
            
            # Calculate quantum coherence
            coherence_analysis = self._calculate_quantum_coherence(quantum_field)
            
            # Perform photonic energy analysis
            energy_analysis = self._analyze_photonic_energy(quantum_field, data)
            
            # Calculate wave-particle duality metrics
            duality_analysis = self._analyze_wave_particle_duality(quantum_field)
            
            # Detect quantum entanglement in price movements
            entanglement_analysis = self._detect_quantum_entanglement(data)
            
            # Generate predictions using quantum superposition
            quantum_predictions = self._generate_quantum_predictions(wavelength_spectrum, coherence_analysis)
            
            result = {
                'dominant_wavelength': wavelength_spectrum['dominant']['wavelength'],
                'dominant_frequency': wavelength_spectrum['dominant']['frequency'],
                'dominant_amplitude': wavelength_spectrum['dominant']['amplitude'],
                'wavelength_spectrum': wavelength_spectrum['spectrum'],
                'interference_strength': interference_analysis['pattern_strength'],
                'constructive_interference': interference_analysis['constructive_points'],
                'destructive_interference': interference_analysis['destructive_points'],
                'quantum_coherence': coherence_analysis['coherence'],
                'coherence_length': coherence_analysis['coherence_length'],
                'decoherence_rate': coherence_analysis['decoherence_rate'],
                'photonic_energy': energy_analysis['total_energy'],
                'energy_distribution': energy_analysis['energy_levels'],
                'photon_count': energy_analysis['photon_count'],
                'wave_particle_ratio': duality_analysis['wave_particle_ratio'],
                'uncertainty_principle': duality_analysis['uncertainty'],
                'quantum_momentum': duality_analysis['momentum'],
                'entanglement_strength': entanglement_analysis['strength'],
                'entangled_periods': entanglement_analysis['entangled_periods'],
                'quantum_correlation': entanglement_analysis['correlation'],
                'superposition_states': quantum_predictions['superposition_states'],
                'probability_amplitudes': quantum_predictions['probability_amplitudes'],
                'quantum_forecast': quantum_predictions['forecast'],
                'phase_transitions': self._detect_phase_transitions(wavelength_spectrum),
                'quantum_tunneling': self._analyze_quantum_tunneling(data, quantum_field),
                'signal_type': self._determine_signal_type(wavelength_spectrum, interference_analysis),
                'raw_data': {
                    'quantum_field': quantum_field.tolist(),
                    'detected_wavelengths': [w.__dict__ for w in self.detected_wavelengths],
                    'quantum_states': [s.__dict__ for s in self.quantum_states]
                }
            }
            
            self.logger.info(f"Photonic analysis completed - Dominant wavelength: {wavelength_spectrum['dominant']['wavelength']:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Photonic Wavelength Analyzer: {str(e)}")
            raise IndicatorCalculationError(f"Photonic analysis failed: {str(e)}")
    
    def _create_quantum_field(self, data: pd.DataFrame) -> np.ndarray:
        """Create quantum field representation of market data."""
        try:
            # Use recent data window
            recent_data = data.tail(self.config.quantum_window)
            
            # Normalize price data to create wave function
            prices = recent_data['close'].values
            normalized_prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
            
            # Create complex wave function (quantum field)
            # Real part: normalized price
            # Imaginary part: Hilbert transform (phase information)
            analytic_signal = hilbert(normalized_prices)
            
            # Add quantum uncertainty
            uncertainty = np.random.normal(0, self.quantum_resolution, len(analytic_signal))
            quantum_field = analytic_signal + uncertainty
            
            return quantum_field
            
        except Exception as e:
            self.logger.warning(f"Quantum field creation failed: {str(e)}")
            return np.zeros(self.config.quantum_window, dtype=complex)
    
    def _decompose_wavelengths(self, quantum_field: np.ndarray) -> Dict[str, Any]:
        """Decompose quantum field into constituent wavelengths."""
        try:
            # Perform FFT to get frequency domain
            fft_result = fft.fft(quantum_field)
            frequencies = fft.fftfreq(len(quantum_field))
            
            # Convert frequencies to wavelengths
            wavelengths = []
            amplitudes = []
            phases = []
            
            for i, freq in enumerate(frequencies):
                if freq > 0:  # Only positive frequencies
                    wavelength = 1.0 / freq if freq != 0 else float('inf')
                    if self.config.wavelength_range[0] <= wavelength <= self.config.wavelength_range[1]:
                        wavelengths.append(wavelength)
                        amplitudes.append(abs(fft_result[i]))
                        phases.append(np.angle(fft_result[i]))
            
            if not wavelengths:
                # Default values if no valid wavelengths found
                wavelengths = [20.0]
                amplitudes = [1.0]
                phases = [0.0]
            
            # Find dominant wavelength
            max_amplitude_idx = np.argmax(amplitudes)
            dominant_wavelength = wavelengths[max_amplitude_idx]
            dominant_frequency = 1.0 / dominant_wavelength
            dominant_amplitude = amplitudes[max_amplitude_idx]
            
            # Create photonic waves
            self.detected_wavelengths = []
            for i, (wl, amp, phase) in enumerate(zip(wavelengths, amplitudes, phases)):
                if amp > np.max(amplitudes) * 0.1:  # Only significant wavelengths
                    photonic_wave = PhotonicWave(
                        wavelength=wl,
                        frequency=1.0/wl,
                        amplitude=amp,
                        phase=phase,
                        energy=self._calculate_photon_energy(1.0/wl),
                        interference_pattern=self._calculate_interference_pattern(wl, quantum_field),
                        coherence_length=self._calculate_coherence_length(wl),
                        polarization=self._calculate_polarization(quantum_field, i)
                    )
                    self.detected_wavelengths.append(photonic_wave)
            
            return {
                'dominant': {
                    'wavelength': float(dominant_wavelength),
                    'frequency': float(dominant_frequency),
                    'amplitude': float(dominant_amplitude)
                },
                'spectrum': {
                    'wavelengths': [float(w) for w in wavelengths],
                    'amplitudes': [float(a) for a in amplitudes],
                    'phases': [float(p) for p in phases]
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Wavelength decomposition failed: {str(e)}")
            return {
                'dominant': {'wavelength': 20.0, 'frequency': 0.05, 'amplitude': 1.0},
                'spectrum': {'wavelengths': [20.0], 'amplitudes': [1.0], 'phases': [0.0]}
            }
    
    def _analyze_interference_patterns(self, wavelength_spectrum: Dict[str, Any]) -> InterferencePattern:
        """Analyze wave interference patterns."""
        try:
            wavelengths = wavelength_spectrum['spectrum']['wavelengths']
            amplitudes = wavelength_spectrum['spectrum']['amplitudes']
            phases = wavelength_spectrum['spectrum']['phases']
            
            if len(wavelengths) < 2:
                return InterferencePattern([], [], 0.0, 0.0, 0.0)
            
            # Calculate interference pattern
            pattern_length = self.config.quantum_window
            interference_pattern = np.zeros(pattern_length)
            
            # Superpose all waves
            for wl, amp, phase in zip(wavelengths, amplitudes, phases):
                x = np.arange(pattern_length)
                wave = amp * np.sin(2 * np.pi * x / wl + phase)
                interference_pattern += wave
            
            # Find constructive and destructive interference points
            peaks, _ = find_peaks(interference_pattern, distance=5)
            troughs, _ = find_peaks(-interference_pattern, distance=5)
            
            # Calculate pattern strength
            pattern_strength = np.std(interference_pattern) / (np.mean(np.abs(interference_pattern)) + 1e-8)
            
            # Calculate coherence measure
            coherence_measure = self._calculate_interference_coherence(wavelengths, phases)
            
            # Calculate phase correlation
            phase_correlation = self._calculate_phase_correlation(phases)
            
            pattern = InterferencePattern(
                constructive_points=peaks.tolist(),
                destructive_points=troughs.tolist(),
                pattern_strength=float(pattern_strength),
                coherence_measure=float(coherence_measure),
                phase_correlation=float(phase_correlation)
            )
            
            self.interference_history.append(pattern)
            return pattern
            
        except Exception as e:
            self.logger.warning(f"Interference analysis failed: {str(e)}")
            return InterferencePattern([], [], 0.0, 0.0, 0.0)
    
    def _calculate_quantum_coherence(self, quantum_field: np.ndarray) -> Dict[str, float]:
        """Calculate quantum coherence properties."""
        try:
            # Coherence function using autocorrelation
            autocorr = np.correlate(quantum_field, quantum_field, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Normalize autocorrelation
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
            
            # Calculate coherence length (where correlation drops to 1/e)
            coherence_threshold = 1.0 / np.e
            coherence_length = 0
            for i, corr in enumerate(np.abs(autocorr)):
                if corr < coherence_threshold:
                    coherence_length = i
                    break
            
            if coherence_length == 0:
                coherence_length = len(autocorr) // 2
            
            # Overall coherence (average correlation in coherence length)
            coherence = float(np.mean(np.abs(autocorr[:coherence_length])))
            
            # Decoherence rate
            if coherence_length > 1:
                decoherence_rate = float(-np.log(coherence) / coherence_length)
            else:
                decoherence_rate = 1.0
            
            return {
                'coherence': coherence,
                'coherence_length': float(coherence_length),
                'decoherence_rate': decoherence_rate
            }
            
        except Exception as e:
            self.logger.warning(f"Coherence calculation failed: {str(e)}")
            return {'coherence': 0.5, 'coherence_length': 10.0, 'decoherence_rate': 0.1}
    
    def _analyze_photonic_energy(self, quantum_field: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze photonic energy distribution."""
        try:
            # Calculate energy using quantum mechanics principles
            # E = hf (Planck's equation adapted for market analysis)
            
            # Get frequency spectrum
            fft_result = fft.fft(quantum_field)
            frequencies = fft.fftfreq(len(quantum_field))
            power_spectrum = np.abs(fft_result) ** 2
            
            # Calculate photonic energy for each frequency
            photon_energies = []
            for freq, power in zip(frequencies, power_spectrum):
                if freq > 0:  # Only positive frequencies
                    energy = self._calculate_photon_energy(freq) * power
                    photon_energies.append(energy)
            
            total_energy = float(np.sum(photon_energies))
            
            # Quantize energy levels
            energy_distribution = np.histogram(photon_energies, bins=self.energy_levels)[0]
            energy_distribution = energy_distribution / np.sum(energy_distribution) if np.sum(energy_distribution) > 0 else energy_distribution
            
            # Count photons (discrete energy packets)
            photon_count = len([e for e in photon_energies if e > total_energy * 0.01])
            
            return {
                'total_energy': total_energy,
                'energy_levels': energy_distribution.tolist(),
                'photon_count': photon_count
            }
            
        except Exception as e:
            self.logger.warning(f"Energy analysis failed: {str(e)}")
            return {'total_energy': 1.0, 'energy_levels': [0.1] * len(self.energy_levels), 'photon_count': 10}
    
    def _analyze_wave_particle_duality(self, quantum_field: np.ndarray) -> Dict[str, float]:
        """Analyze wave-particle duality characteristics."""
        try:
            # Wave characteristics (continuous)
            wave_amplitude = float(np.std(np.real(quantum_field)))
            wave_phase_coherence = float(np.std(np.angle(quantum_field)))
            
            # Particle characteristics (discrete)
            # Use threshold to identify "particles" (significant price movements)
            threshold = np.std(np.abs(quantum_field))
            particles = np.abs(quantum_field) > threshold
            particle_count = float(np.sum(particles))
            
            # Wave-particle ratio
            wave_score = wave_amplitude / (wave_phase_coherence + 1e-8)
            particle_score = particle_count / len(quantum_field)
            wave_particle_ratio = wave_score / (particle_score + 1e-8)
            
            # Heisenberg uncertainty principle (position-momentum uncertainty)
            position_uncertainty = float(np.std(np.real(quantum_field)))
            momentum_uncertainty = float(np.std(np.diff(np.real(quantum_field))))
            uncertainty = position_uncertainty * momentum_uncertainty
            
            # Quantum momentum
            momentum = float(np.mean(np.diff(np.real(quantum_field))))
            
            return {
                'wave_particle_ratio': float(wave_particle_ratio),
                'uncertainty': float(uncertainty),
                'momentum': momentum
            }
            
        except Exception as e:
            self.logger.warning(f"Wave-particle duality analysis failed: {str(e)}")
            return {'wave_particle_ratio': 1.0, 'uncertainty': 0.1, 'momentum': 0.0}
    
    def _detect_quantum_entanglement(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect quantum entanglement in price movements."""
        try:
            # Analyze correlation between different time periods (quantum entanglement analogy)
            recent_data = data.tail(self.config.quantum_window)
            
            entangled_periods = []
            correlations = []
            
            # Check for non-local correlations (entanglement)
            for lag in range(1, self.config.entanglement_distance):
                if lag < len(recent_data) - 1:
                    series1 = recent_data['close'].values[:-lag]
                    series2 = recent_data['close'].values[lag:]
                    
                    correlation = np.corrcoef(series1, series2)[0, 1]
                    if not np.isnan(correlation) and abs(correlation) > 0.7:  # Strong correlation threshold
                        entangled_periods.append(lag)
                        correlations.append(abs(correlation))
            
            # Entanglement strength
            entanglement_strength = float(np.mean(correlations)) if correlations else 0.0
            
            # Quantum correlation (Bell's theorem inspired)
            quantum_correlation = self._calculate_quantum_correlation(recent_data)
            
            return {
                'strength': entanglement_strength,
                'entangled_periods': entangled_periods,
                'correlation': float(quantum_correlation)
            }
            
        except Exception as e:
            self.logger.warning(f"Entanglement detection failed: {str(e)}")
            return {'strength': 0.0, 'entangled_periods': [], 'correlation': 0.0}
    
    def _generate_quantum_predictions(self, wavelength_spectrum: Dict[str, Any], 
                                    coherence_analysis: Dict[str, float]) -> Dict[str, Any]:
        """Generate predictions using quantum superposition principles."""
        try:
            wavelengths = wavelength_spectrum['spectrum']['wavelengths']
            amplitudes = wavelength_spectrum['spectrum']['amplitudes']
            phases = wavelength_spectrum['spectrum']['phases']
            
            # Create superposition of quantum states
            superposition_states = []
            probability_amplitudes = []
            
            # Each wavelength represents a possible future state
            for wl, amp, phase in zip(wavelengths, amplitudes, phases):
                # Probability amplitude (complex number)
                prob_amplitude = amp * np.exp(1j * phase)
                probability_amplitudes.append(complex(prob_amplitude))
                
                # Quantum state description
                state = {
                    'wavelength': float(wl),
                    'probability': float(abs(prob_amplitude) ** 2),
                    'phase': float(phase),
                    'energy': float(self._calculate_photon_energy(1.0/wl))
                }
                superposition_states.append(state)
            
            # Quantum forecast using Born rule (probability = |amplitude|^2)
            probabilities = [abs(amp) ** 2 for amp in probability_amplitudes]
            total_prob = sum(probabilities)
            normalized_probs = [p / total_prob for p in probabilities] if total_prob > 0 else probabilities
            
            # Expected value forecast
            forecast_components = []
            for i, (wl, prob) in enumerate(zip(wavelengths, normalized_probs)):
                # Future wave component
                future_steps = 10
                x = np.arange(future_steps)
                wave_component = prob * np.sin(2 * np.pi * x / wl + phases[i])
                forecast_components.append(wave_component)
            
            # Superpose all components
            quantum_forecast = np.sum(forecast_components, axis=0) if forecast_components else np.zeros(10)
            
            return {
                'superposition_states': superposition_states,
                'probability_amplitudes': [{'real': float(amp.real), 'imag': float(amp.imag)} 
                                          for amp in probability_amplitudes],
                'forecast': quantum_forecast.tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"Quantum prediction failed: {str(e)}")
            return {
                'superposition_states': [],
                'probability_amplitudes': [],
                'forecast': [0.0] * 10
            }    
    def _calculate_photon_energy(self, frequency: float) -> float:
        """Calculate photon energy using Planck's equation (adapted for markets)."""
        # E = hf (adapted for market context)
        energy = self.planck_constant * frequency * 1e34  # Scale for market context
        return float(energy)
    
    def _calculate_interference_pattern(self, wavelength: float, quantum_field: np.ndarray) -> np.ndarray:
        """Calculate interference pattern for a specific wavelength."""
        try:
            pattern_length = len(quantum_field)
            x = np.arange(pattern_length)
            
            # Create reference wave
            reference_wave = np.sin(2 * np.pi * x / wavelength)
            
            # Calculate interference with quantum field
            field_real = np.real(quantum_field)
            interference = reference_wave * field_real
            
            return interference
            
        except Exception:
            return np.zeros(len(quantum_field))
    
    def _calculate_coherence_length(self, wavelength: float) -> float:
        """Calculate coherence length for a wavelength."""
        # Coherence length is related to wavelength and spectral purity
        coherence_length = wavelength * self.config.wave_packet_width
        return float(coherence_length)
    
    def _calculate_polarization(self, quantum_field: np.ndarray, index: int) -> float:
        """Calculate polarization state of the wave."""
        try:
            # Use phase relationship to determine polarization
            real_part = np.real(quantum_field)
            imag_part = np.imag(quantum_field)
            
            # Polarization angle
            polarization = np.arctan2(np.mean(imag_part), np.mean(real_part))
            return float(polarization)
            
        except Exception:
            return 0.0
    
    def _calculate_interference_coherence(self, wavelengths: List[float], phases: List[float]) -> float:
        """Calculate coherence of interference pattern."""
        try:
            if len(wavelengths) < 2:
                return 1.0
            
            # Phase differences between waves
            phase_diffs = []
            for i in range(len(phases) - 1):
                phase_diff = abs(phases[i+1] - phases[i])
                phase_diffs.append(min(phase_diff, 2*np.pi - phase_diff))  # Wrap to [0, π]
            
            # Coherence is inversely related to phase spread
            phase_spread = np.std(phase_diffs) if phase_diffs else 0.0
            coherence = np.exp(-phase_spread)  # Exponential decay with phase spread
            
            return float(coherence)
            
        except Exception:
            return 0.5
    
    def _calculate_phase_correlation(self, phases: List[float]) -> float:
        """Calculate phase correlation between different wavelengths."""
        try:
            if len(phases) < 2:
                return 1.0
            
            # Convert phases to unit circle coordinates
            x_coords = [np.cos(phase) for phase in phases]
            y_coords = [np.sin(phase) for phase in phases]
            
            # Calculate correlation in phase space
            if len(x_coords) > 1:
                x_corr = np.corrcoef(x_coords[:-1], x_coords[1:])[0, 1]
                y_corr = np.corrcoef(y_coords[:-1], y_coords[1:])[0, 1]
                
                phase_correlation = (abs(x_corr) + abs(y_corr)) / 2
                return float(phase_correlation) if not np.isnan(phase_correlation) else 0.5
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_quantum_correlation(self, data: pd.DataFrame) -> float:
        """Calculate quantum correlation using Bell's theorem inspired approach."""
        try:
            prices = data['close'].values
            
            # Create two entangled observables (price changes and volume changes)
            price_changes = np.diff(prices)
            volume_changes = np.diff(data['volume'].values)
            
            if len(price_changes) != len(volume_changes):
                min_len = min(len(price_changes), len(volume_changes))
                price_changes = price_changes[:min_len]
                volume_changes = volume_changes[:min_len]
            
            # Quantum correlation (violation of Bell's inequality analogy)
            # Measure correlation at different "measurement angles"
            correlations = []
            for shift in range(1, min(5, len(price_changes))):
                if shift < len(price_changes):
                    shifted_prices = price_changes[:-shift]
                    shifted_volumes = volume_changes[shift:]
                    
                    min_len = min(len(shifted_prices), len(shifted_volumes))
                    correlation = np.corrcoef(shifted_prices[:min_len], shifted_volumes[:min_len])[0, 1]
                    
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
            
            # Bell parameter (simplified)
            quantum_correlation = np.max(correlations) if correlations else 0.0
            return float(quantum_correlation)
            
        except Exception:
            return 0.0
    
    def _detect_phase_transitions(self, wavelength_spectrum: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect quantum phase transitions in market state."""
        try:
            amplitudes = wavelength_spectrum['spectrum']['amplitudes']
            wavelengths = wavelength_spectrum['spectrum']['wavelengths']
            
            phase_transitions = []
            
            # Look for sudden changes in dominant wavelength (phase transition)
            if len(self.detected_wavelengths) > 5:
                recent_wavelengths = [w.wavelength for w in self.detected_wavelengths[-5:]]
                
                # Detect discontinuous changes
                wavelength_changes = np.diff(recent_wavelengths)
                large_changes = np.where(np.abs(wavelength_changes) > np.std(wavelength_changes) * 2)[0]
                
                for change_idx in large_changes:
                    transition = {
                        'position': int(change_idx),
                        'from_wavelength': float(recent_wavelengths[change_idx]),
                        'to_wavelength': float(recent_wavelengths[change_idx + 1]),
                        'magnitude': float(abs(wavelength_changes[change_idx])),
                        'type': 'quantum_phase_transition'
                    }
                    phase_transitions.append(transition)
            
            return phase_transitions
            
        except Exception as e:
            self.logger.warning(f"Phase transition detection failed: {str(e)}")
            return []
    
    def _analyze_quantum_tunneling(self, data: pd.DataFrame, quantum_field: np.ndarray) -> Dict[str, Any]:
        """Analyze quantum tunneling effects in price movements."""
        try:
            prices = data['close'].values[-len(quantum_field):]
            
            # Identify potential energy barriers (resistance/support levels)
            price_peaks, _ = find_peaks(prices, distance=5)
            price_troughs, _ = find_peaks(-prices, distance=5)
            
            tunneling_events = []
            
            # Look for price movements that "tunnel" through barriers
            for i in range(1, len(prices) - 1):
                current_price = prices[i]
                
                # Check if price is between resistance and support
                nearby_peaks = [prices[p] for p in price_peaks if abs(p - i) < 10]
                nearby_troughs = [prices[t] for t in price_troughs if abs(t - i) < 10]
                
                if nearby_peaks and nearby_troughs:
                    max_resistance = max(nearby_peaks)
                    min_support = min(nearby_troughs)
                    
                    # Check for tunneling (rapid movement through barrier)
                    if min_support < current_price < max_resistance:
                        price_velocity = abs(prices[i+1] - prices[i-1]) / 2
                        barrier_width = max_resistance - min_support
                        
                        # Tunneling probability (simplified quantum mechanics)
                        tunneling_prob = np.exp(-barrier_width / (price_velocity + 1e-8))
                        
                        if tunneling_prob > 0.1:  # Significant tunneling probability
                            tunneling_events.append({
                                'position': int(i),
                                'probability': float(tunneling_prob),
                                'barrier_width': float(barrier_width),
                                'velocity': float(price_velocity)
                            })
            
            # Overall tunneling analysis
            tunneling_strength = np.mean([e['probability'] for e in tunneling_events]) if tunneling_events else 0.0
            
            return {
                'tunneling_events': tunneling_events,
                'tunneling_strength': float(tunneling_strength),
                'event_count': len(tunneling_events)
            }
            
        except Exception as e:
            self.logger.warning(f"Quantum tunneling analysis failed: {str(e)}")
            return {'tunneling_events': [], 'tunneling_strength': 0.0, 'event_count': 0}
    
    def _determine_signal_type(self, wavelength_spectrum: Dict[str, Any], 
                              interference_analysis: InterferencePattern) -> SignalType:
        """Determine signal type based on photonic analysis."""
        try:
            dominant_amplitude = wavelength_spectrum['dominant']['amplitude']
            dominant_wavelength = wavelength_spectrum['dominant']['wavelength']
            interference_strength = interference_analysis.pattern_strength
            coherence_measure = interference_analysis.coherence_measure
            
            # Strong signals from high amplitude, coherent waves
            if dominant_amplitude > 2.0 and coherence_measure > 0.8:
                # Short wavelengths indicate rapid changes
                if dominant_wavelength < 20:
                    return SignalType.STRONG_BUY if len(interference_analysis.constructive_points) > len(interference_analysis.destructive_points) else SignalType.STRONG_SELL
                else:
                    return SignalType.BUY if len(interference_analysis.constructive_points) > len(interference_analysis.destructive_points) else SignalType.SELL
            
            # Moderate signals from moderate coherence
            elif coherence_measure > 0.5 and interference_strength > 1.0:
                return SignalType.BUY if len(interference_analysis.constructive_points) > len(interference_analysis.destructive_points) else SignalType.SELL
            
            # Default to neutral for low coherence systems
            else:
                return SignalType.NEUTRAL
                
        except Exception:
            return SignalType.NEUTRAL
    
    def get_signal_type(self) -> SignalType:
        """Get the current signal type."""
        return getattr(self, '_last_signal_type', SignalType.NEUTRAL)
    
    def get_signal_strength(self) -> float:
        """Get the current signal strength."""
        if self.detected_wavelengths:
            return self.detected_wavelengths[-1].amplitude
        return 0.0
    
    def get_quantum_state(self) -> Optional[QuantumState]:
        """Get the current quantum state of the market."""
        if self.quantum_states:
            return self.quantum_states[-1]
        return None
    
    def get_dominant_wavelength(self) -> float:
        """Get the dominant wavelength in the current analysis."""
        if self.detected_wavelengths:
            # Find wavelength with maximum amplitude
            max_amplitude_wave = max(self.detected_wavelengths, key=lambda w: w.amplitude)
            return max_amplitude_wave.wavelength
        return 20.0  # Default wavelength
    
    def get_interference_strength(self) -> float:
        """Get the current interference pattern strength."""
        if self.interference_history:
            return self.interference_history[-1].pattern_strength
        return 0.0
    
    def get_coherence_metrics(self) -> Dict[str, float]:
        """Get detailed coherence analysis metrics."""
        if not self.detected_wavelengths:
            return {'coherence': 0.5, 'coherence_length': 10.0, 'phase_stability': 0.5}
        
        # Calculate ensemble coherence metrics
        coherence_lengths = [w.coherence_length for w in self.detected_wavelengths]
        phases = [w.phase for w in self.detected_wavelengths]
        
        avg_coherence_length = np.mean(coherence_lengths)
        phase_stability = 1.0 - np.std(phases) / (2 * np.pi)  # Normalized phase stability
        
        overall_coherence = np.mean([
            self._calculate_interference_coherence([w.wavelength for w in self.detected_wavelengths],
                                                 [w.phase for w in self.detected_wavelengths])
        ])
        
        return {
            'coherence': float(overall_coherence),
            'coherence_length': float(avg_coherence_length),
            'phase_stability': float(max(0.0, phase_stability))
        }