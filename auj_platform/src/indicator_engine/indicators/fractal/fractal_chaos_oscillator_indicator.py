"""
Fractal Chaos Oscillator Indicator - Advanced Implementation

This indicator implements Williams' Chaos Theory with sophisticated mathematical models
for fractal energy calculation and phase space analysis. Features include:
- Williams Fractal analysis with energy calculation
- Phase space reconstruction and attractor analysis
- Chaos oscillator with momentum integration
- Fractal energy bands and signal generation
- Advanced chaos theory metrics for market analysis

Mission: Supporting humanitarian trading platform for poor and sick children through
maximum profitability via advanced fractal chaos analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import hilbert, find_peaks
from sklearn.preprocessing import StandardScaler
import warnings

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChaosOscillatorResult:
    """Results container for Fractal Chaos Oscillator"""
    oscillator_value: float
    fractal_energy: float
    chaos_level: float
    phase_angle: float
    momentum_signal: str
    energy_band: str
    chaos_regime: str
    signal_strength: float
    trend_chaos_alignment: float

class FractalChaosOscillatorIndicator(StandardIndicatorInterface):
    """
    Advanced Fractal Chaos Oscillator Indicator
    
    Implements Williams' Chaos Theory with sophisticated mathematical models
    for energy calculation and phase space analysis.
    """
    
    def __init__(self, 
                 period: int = 5,
                 energy_period: int = 34,
                 chaos_window: int = 100,
                 energy_threshold: float = 0.3):
        super().__init__()
        self.period = period
        self.energy_period = energy_period
        self.chaos_window = chaos_window
        self.energy_threshold = energy_threshold
        
        logger.info(f"Initialized FractalChaosOscillatorIndicator with period={period}")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            self._validate_data(data)
            
            if len(data) < self.chaos_window:
                return self._create_default_result()
            
            recent_data = data[-self.chaos_window:].copy()
            
            # Calculate Williams fractals
            fractals = self._calculate_williams_fractals(recent_data)
            
            # Calculate fractal energy
            fractal_energy = self._calculate_fractal_energy(recent_data, fractals)
            
            # Calculate chaos oscillator
            oscillator_value = self._calculate_chaos_oscillator(recent_data, fractal_energy)
            
            # Phase space analysis
            phase_analysis = self._analyze_phase_space(recent_data)
            
            # Generate signals
            signals = self._generate_chaos_signals(oscillator_value, fractal_energy, phase_analysis)
            
            result = ChaosOscillatorResult(
                oscillator_value=oscillator_value,
                fractal_energy=fractal_energy,
                chaos_level=phase_analysis['chaos_level'],
                phase_angle=phase_analysis['phase_angle'],
                momentum_signal=signals['momentum_signal'],
                energy_band=signals['energy_band'],
                chaos_regime=signals['chaos_regime'],
                signal_strength=signals['signal_strength'],
                trend_chaos_alignment=signals['trend_chaos_alignment']
            )
            
            return self._format_output(result, data.index[-1])
            
        except Exception as e:
            logger.error(f"Error in fractal chaos oscillator calculation: {e}")
            raise IndicatorCalculationError(f"FractalChaosOscillatorIndicator calculation failed: {e}")

    def _calculate_williams_fractals(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """Calculate Williams fractal points"""
        highs = data['high'].values
        lows = data['low'].values
        fractal_highs = []
        fractal_lows = []
        
        half_period = self.period // 2
        
        for i in range(half_period, len(data) - half_period):
            # Fractal high
            if all(highs[i] >= highs[i-j] for j in range(1, half_period + 1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, half_period + 1)):
                fractal_highs.append(i)
            
            # Fractal low
            if all(lows[i] <= lows[i-j] for j in range(1, half_period + 1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, half_period + 1)):
                fractal_lows.append(i)
        
        return {'highs': fractal_highs, 'lows': fractal_lows}

    def _calculate_fractal_energy(self, data: pd.DataFrame, fractals: Dict[str, List[int]]) -> float:
        """Calculate fractal energy using advanced mathematical models"""
        try:
            if not fractals['highs'] and not fractals['lows']:
                return 0.0
            
            # Energy from fractal density
            total_fractals = len(fractals['highs']) + len(fractals['lows'])
            fractal_density = total_fractals / len(data)
            
            # Energy from price volatility
            returns = data['close'].pct_change().dropna()
            volatility_energy = returns.std()
            
            # Energy from fractal spacing
            all_fractals = sorted(fractals['highs'] + fractals['lows'])
            if len(all_fractals) > 1:
                spacings = np.diff(all_fractals)
                spacing_energy = 1.0 / (np.mean(spacings) + 1)
            else:
                spacing_energy = 0.0
            
            # Combine energies
            fractal_energy = (fractal_density * 0.4 + 
                            volatility_energy * 0.4 + 
                            spacing_energy * 0.2)
            
            return np.clip(fractal_energy, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Fractal energy calculation error: {e}")
            return 0.0

    def _calculate_chaos_oscillator(self, data: pd.DataFrame, fractal_energy: float) -> float:
        """Calculate chaos oscillator value"""
        try:
            closes = data['close'].values
            
            # Momentum component
            if len(closes) >= self.energy_period:
                momentum = (closes[-1] / closes[-self.energy_period] - 1) * 100
            else:
                momentum = 0.0
            
            # Energy adjustment
            energy_factor = 1.0 + fractal_energy
            
            # Chaos oscillator
            oscillator = momentum * energy_factor
            
            return np.clip(oscillator, -100, 100)
            
        except Exception as e:
            logger.warning(f"Chaos oscillator calculation error: {e}")
            return 0.0

    def _analyze_phase_space(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze phase space characteristics"""
        try:
            prices = data['close'].values
            returns = np.diff(np.log(prices))
            
            if len(returns) < 10:
                return {'chaos_level': 0.5, 'phase_angle': 0.0}
            
            # Phase space embedding
            embedded = self._embed_time_series(returns, dim=3, delay=1)
            
            # Calculate chaos metrics
            chaos_level = self._estimate_lyapunov_exponent(embedded)
            phase_angle = self._calculate_phase_angle(returns)
            
            return {
                'chaos_level': np.clip(chaos_level, 0.0, 1.0),
                'phase_angle': phase_angle
            }
            
        except Exception as e:
            logger.warning(f"Phase space analysis error: {e}")
            return {'chaos_level': 0.5, 'phase_angle': 0.0}

    def _embed_time_series(self, data: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """Create phase space embedding"""
        n = len(data)
        m = n - (dim - 1) * delay
        
        if m <= 0:
            return np.array([[]])
        
        embedded = np.zeros((m, dim))
        for i in range(dim):
            embedded[:, i] = data[i * delay:i * delay + m]
            
        return embedded

    def _estimate_lyapunov_exponent(self, embedded: np.ndarray) -> float:
        """Estimate Lyapunov exponent"""
        try:
            if embedded.size == 0 or embedded.shape[0] < 10:
                return 0.0
            
            divergences = []
            
            for i in range(min(embedded.shape[0] - 5, 20)):
                distances = np.linalg.norm(embedded[i+1:] - embedded[i], axis=1)
                if len(distances) == 0:
                    continue
                    
                nearest_idx = np.argmin(distances) + i + 1
                if nearest_idx >= embedded.shape[0] - 3:
                    continue
                
                initial_distance = distances[nearest_idx - i - 1]
                if initial_distance == 0:
                    continue
                
                for t in range(1, min(5, embedded.shape[0] - nearest_idx)):
                    current_distance = np.linalg.norm(embedded[i + t] - embedded[nearest_idx + t])
                    if current_distance > 0:
                        divergence = np.log(current_distance / initial_distance) / t
                        divergences.append(divergence)
            
            if divergences:
                return np.median(divergences)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Lyapunov estimation error: {e}")
            return 0.0

    def _calculate_phase_angle(self, returns: np.ndarray) -> float:
        """Calculate phase angle using Hilbert transform"""
        try:
            if len(returns) < 10:
                return 0.0
            
            # Apply Hilbert transform
            analytic_signal = hilbert(returns)
            phase = np.angle(analytic_signal)
            
            # Return current phase angle in degrees
            return np.degrees(phase[-1])
            
        except Exception as e:
            logger.warning(f"Phase angle calculation error: {e}")
            return 0.0

    def _generate_chaos_signals(self, oscillator: float, energy: float, 
                              phase_analysis: Dict[str, float]) -> Dict[str, Any]:
        """Generate trading signals from chaos analysis"""
        try:
            # Momentum signal
            if oscillator > 20:
                momentum_signal = "BULLISH"
            elif oscillator < -20:
                momentum_signal = "BEARISH"
            else:
                momentum_signal = "NEUTRAL"
            
            # Energy band
            if energy > 0.7:
                energy_band = "HIGH_ENERGY"
            elif energy > 0.3:
                energy_band = "MEDIUM_ENERGY"
            else:
                energy_band = "LOW_ENERGY"
            
            # Chaos regime
            chaos_level = phase_analysis['chaos_level']
            if chaos_level > 0.7:
                chaos_regime = "CHAOTIC"
            elif chaos_level > 0.3:
                chaos_regime = "TRANSITIONAL"
            else:
                chaos_regime = "ORDERED"
            
            # Signal strength
            signal_strength = min(abs(oscillator) / 50, 1.0) * energy
            
            # Trend-chaos alignment
            trend_factor = abs(oscillator) / 100
            chaos_factor = chaos_level
            alignment = (trend_factor + chaos_factor) / 2
            
            return {
                'momentum_signal': momentum_signal,
                'energy_band': energy_band,
                'chaos_regime': chaos_regime,
                'signal_strength': signal_strength,
                'trend_chaos_alignment': alignment
            }
            
        except Exception as e:
            logger.warning(f"Signal generation error: {e}")
            return {
                'momentum_signal': 'NEUTRAL',
                'energy_band': 'MEDIUM_ENERGY',
                'chaos_regime': 'UNKNOWN',
                'signal_strength': 0.0,
                'trend_chaos_alignment': 0.0
            }

    def _format_output(self, result: ChaosOscillatorResult, timestamp) -> Dict[str, Any]:
        """Format the calculation results for output"""
        return {
            'timestamp': timestamp,
            'indicator_name': 'FractalChaosOscillator',
            'oscillator_value': round(result.oscillator_value, 4),
            'fractal_energy': round(result.fractal_energy, 4),
            'chaos_level': round(result.chaos_level, 4),
            'phase_angle': round(result.phase_angle, 2),
            'momentum_signal': result.momentum_signal,
            'energy_band': result.energy_band,
            'chaos_regime': result.chaos_regime,
            'signal_strength': round(result.signal_strength, 4),
            'trend_chaos_alignment': round(result.trend_chaos_alignment, 4)
        }

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
        if len(data) < self.period * 4:
            raise IndicatorCalculationError(f"Insufficient data: minimum {self.period * 4} periods required")

    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data cases"""
        return {
            'timestamp': pd.Timestamp.now(),
            'indicator_name': 'FractalChaosOscillator',
            'oscillator_value': 0.0,
            'fractal_energy': 0.0,
            'chaos_level': 0.5,
            'phase_angle': 0.0,
            'momentum_signal': 'INSUFFICIENT_DATA',
            'energy_band': 'UNKNOWN',
            'chaos_regime': 'UNKNOWN',
            'signal_strength': 0.0,
            'trend_chaos_alignment': 0.0
        }

    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']

    def get_indicator_name(self) -> str:
        return "FractalChaosOscillator"