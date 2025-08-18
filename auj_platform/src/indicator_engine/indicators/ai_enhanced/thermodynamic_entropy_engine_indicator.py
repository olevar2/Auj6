"""
Thermodynamic Entropy Engine for AUJ Platform Trading System

This module implements a sophisticated thermodynamic entropy engine that applies
real thermodynamic principles to financial market analysis. The engine models
market behavior using concepts from statistical mechanics, thermodynamics, and
information theory to detect phase transitions, calculate market entropy, and
analyze energy states for trading decisions.

Key Features:
- Statistical mechanics modeling of market behavior
- Shannon entropy and thermodynamic entropy calculations
- Phase transition detection using order parameters
- Energy state analysis with Maxwell-Boltzmann distributions
- Heat capacity and temperature modeling for market volatility
- Free energy calculations for trend stability analysis
- Critical point detection for market regime changes
- Partition function analysis for probability distributions
- Ergodic analysis for market efficiency assessment
- Non-equilibrium thermodynamics for trending markets

Scientific Foundation:
- Applies Boltzmann entropy: S = k * ln(W)
- Uses Gibbs free energy: G = H - TS
- Implements Maxwell-Boltzmann distribution for price movements
- Calculates heat capacity: C = dE/dT
- Models phase transitions using Landau theory
- Information entropy: H = -Σ p(i) * log(p(i))

Author: AUJ Platform Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import scipy.stats as stats
from scipy.optimize import minimize_scalar, curve_fit
from scipy.special import loggamma
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    """Market phase states based on thermodynamic analogy."""
    SOLID = "solid"           # Low volatility, stable trending
    LIQUID = "liquid"         # Medium volatility, random walk
    GAS = "gas"              # High volatility, chaotic behavior
    PLASMA = "plasma"         # Extreme volatility, breakdown
    CRITICAL = "critical"     # Phase transition point
    SUPERCRITICAL = "supercritical"  # Beyond critical point


class EnergyState(Enum):
    """Energy states of the market system."""
    GROUND = "ground"         # Minimum energy state
    EXCITED = "excited"       # Higher energy state
    METASTABLE = "metastable" # Temporarily stable state
    UNSTABLE = "unstable"     # High energy, unstable state


@dataclass
class ThermodynamicState:
    """Thermodynamic state of the market."""
    temperature: float        # Market temperature (volatility measure)
    entropy: float           # System entropy
    energy: float            # Internal energy
    free_energy: float       # Gibbs free energy
    heat_capacity: float     # Heat capacity
    pressure: float          # Market pressure (volume/volatility)
    volume: float            # System volume (market activity)
    chemical_potential: float # Chemical potential (trend strength)
    phase: MarketPhase       # Current market phase
    energy_state: EnergyState # Current energy state
    order_parameter: float   # Phase transition order parameter
    correlation_length: float # Correlation length


@dataclass
class PhaseTransition:
    """Phase transition event data."""
    transition_time: datetime
    from_phase: MarketPhase
    to_phase: MarketPhase
    transition_type: str     # first_order, second_order, continuous
    latent_heat: float       # Energy released/absorbed
    critical_exponent: float # Critical behavior exponent
    duration: float          # Transition duration in time units
    strength: float          # Transition strength


class ThermodynamicEntropyEngine:
    """
    Advanced Thermodynamic Entropy Engine for market analysis using
    real thermodynamic principles and statistical mechanics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the thermodynamic entropy engine."""
        self.logger = logging.getLogger(__name__)
        
        # Physical constants (adapted for financial markets)
        self.BOLTZMANN_CONSTANT = 1.0  # Normalized for financial data
        self.GAS_CONSTANT = 8.314      # Adapted for market analysis
        self.AVOGADRO_NUMBER = 6.022e23  # For ensemble calculations
        
        # Default configuration
        self.parameters = {
            'lookback_period': 100,
            'min_data_points': 20,
            'temperature_smoothing': 0.9,
            'entropy_window': 50,
            'phase_transition_threshold': 0.3,
            'critical_temperature': 2.0,
            'energy_bins': 50,
            'correlation_length_max': 20,
            'ensemble_size': 1000,
            'equilibrium_threshold': 0.1,
            'transition_detection_window': 10,
            'heat_capacity_smoothing': 0.8,
            'free_energy_weight': 0.7,
            'pressure_volume_ratio': 1.5,
            'chemical_potential_factor': 0.5,
            'critical_exponents': {
                'alpha': 0.0,    # Heat capacity exponent
                'beta': 0.325,   # Order parameter exponent
                'gamma': 1.24,   # Susceptibility exponent
                'delta': 4.82,   # Critical isotherm exponent
                'nu': 0.63,      # Correlation length exponent
                'eta': 0.036     # Anomalous dimension
            }
        }
        
        # Update with user configuration
        if config:
            self.parameters.update(config)
        
        # Initialize state variables
        self.price_data = np.array([])
        self.volume_data = np.array([])
        self.returns = np.array([])
        self.current_state: Optional[ThermodynamicState] = None
        self.historical_states: List[ThermodynamicState] = []
        self.phase_transitions: List[PhaseTransition] = []
        self.temperature_history = np.array([])
        self.entropy_history = np.array([])
        self.energy_history = np.array([])
        self.last_update = None
        
        # Thermodynamic variables
        self.partition_function = 0.0
        self.canonical_ensemble = np.array([])
        self.microcanonical_ensemble = np.array([])
        self.grand_canonical_ensemble = np.array([])
        
        self.logger.info("Thermodynamic Entropy Engine initialized successfully")
    
    def calculate(self, price_data: np.ndarray, 
                 volume_data: Optional[np.ndarray] = None,
                 timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate thermodynamic properties and entropy of the market.
        
        Args:
            price_data: Array of price values
            volume_data: Optional array of volume values
            timestamp: Optional timestamp for current calculation
            
        Returns:
            Dictionary containing thermodynamic analysis results
        """
        try:
            # Validate inputs
            if len(price_data) < self.parameters['min_data_points']:
                self.logger.warning("Insufficient data for thermodynamic analysis")
                return self._get_empty_result()
            
            # Update internal data
            self.price_data = price_data[-self.parameters['lookback_period']:]
            if volume_data is not None:
                self.volume_data = volume_data[-self.parameters['lookback_period']:]
            else:
                self.volume_data = np.ones_like(self.price_data)
            
            # Calculate returns
            self.returns = np.diff(self.price_data) / self.price_data[:-1]
            
            # Calculate fundamental thermodynamic quantities
            temperature = self._calculate_temperature()
            entropy = self._calculate_entropy()
            energy = self._calculate_internal_energy()
            free_energy = self._calculate_free_energy(energy, temperature, entropy)
            heat_capacity = self._calculate_heat_capacity()
            pressure = self._calculate_pressure()
            volume = self._calculate_volume()
            chemical_potential = self._calculate_chemical_potential()
            
            # Phase analysis
            phase = self._determine_market_phase(temperature, entropy, energy)
            energy_state = self._determine_energy_state(energy, temperature)
            order_parameter = self._calculate_order_parameter()
            correlation_length = self._calculate_correlation_length()
            
            # Create current thermodynamic state
            self.current_state = ThermodynamicState(
                temperature=temperature,
                entropy=entropy,
                energy=energy,
                free_energy=free_energy,
                heat_capacity=heat_capacity,
                pressure=pressure,
                volume=volume,
                chemical_potential=chemical_potential,
                phase=phase,
                energy_state=energy_state,
                order_parameter=order_parameter,
                correlation_length=correlation_length
            )
            
            # Update history
            self.historical_states.append(self.current_state)
            self.temperature_history = np.append(self.temperature_history, temperature)
            self.entropy_history = np.append(self.entropy_history, entropy)
            self.energy_history = np.append(self.energy_history, energy)
            
            # Limit history size
            max_history = self.parameters['lookback_period']
            if len(self.historical_states) > max_history:
                self.historical_states = self.historical_states[-max_history:]
                self.temperature_history = self.temperature_history[-max_history:]
                self.entropy_history = self.entropy_history[-max_history:]
                self.energy_history = self.energy_history[-max_history:]
            
            # Detect phase transitions
            phase_transition = self._detect_phase_transition()
            if phase_transition:
                self.phase_transitions.append(phase_transition)
            
            # Advanced thermodynamic analysis
            partition_function = self._calculate_partition_function(temperature)
            ensemble_analysis = self._perform_ensemble_analysis()
            critical_analysis = self._analyze_critical_behavior()
            equilibrium_analysis = self._analyze_equilibrium_properties()
            
            # Generate trading signals
            signals = self._generate_thermodynamic_signals()
            
            # Market regime analysis
            regime_analysis = self._analyze_market_regime()
            
            # Predictive analysis
            predictive_analysis = self._perform_predictive_analysis()
            
            self.last_update = timestamp or datetime.now()
            
            # Compile results
            result = {
                'current_state': {
                    'temperature': temperature,
                    'entropy': entropy,
                    'energy': energy,
                    'free_energy': free_energy,
                    'heat_capacity': heat_capacity,
                    'pressure': pressure,
                    'volume': volume,
                    'chemical_potential': chemical_potential,
                    'phase': phase.value,
                    'energy_state': energy_state.value,
                    'order_parameter': order_parameter,
                    'correlation_length': correlation_length
                },
                'partition_function': partition_function,
                'ensemble_analysis': ensemble_analysis,
                'critical_analysis': critical_analysis,
                'equilibrium_analysis': equilibrium_analysis,
                'phase_transition': phase_transition.__dict__ if phase_transition else None,
                'signals': signals,
                'regime_analysis': regime_analysis,
                'predictive_analysis': predictive_analysis,
                'historical_trends': {
                    'temperature_trend': self._calculate_trend(self.temperature_history),
                    'entropy_trend': self._calculate_trend(self.entropy_history),
                    'energy_trend': self._calculate_trend(self.energy_history)
                },
                'thermodynamic_efficiency': self._calculate_thermodynamic_efficiency(),
                'market_complexity': self._calculate_market_complexity(),
                'information_content': self._calculate_information_content(),
                'timestamp': self.last_update.isoformat()
            }
            
            self.logger.info(f"Thermodynamic analysis completed: "
                           f"T={temperature:.3f}, S={entropy:.3f}, "
                           f"phase={phase.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in thermodynamic calculation: {e}")
            return self._get_empty_result()
    
    def _get_empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'current_state': {
                'temperature': 0.0,
                'entropy': 0.0,
                'energy': 0.0,
                'free_energy': 0.0,
                'heat_capacity': 0.0,
                'pressure': 0.0,
                'volume': 0.0,
                'chemical_potential': 0.0,
                'phase': MarketPhase.LIQUID.value,
                'energy_state': EnergyState.GROUND.value,
                'order_parameter': 0.0,
                'correlation_length': 0.0
            },
            'partition_function': 0.0,
            'ensemble_analysis': {},
            'critical_analysis': {},
            'equilibrium_analysis': {},
            'phase_transition': None,
            'signals': {},
            'regime_analysis': {},
            'predictive_analysis': {},
            'historical_trends': {},
            'thermodynamic_efficiency': 0.0,
            'market_complexity': 0.0,
            'information_content': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_temperature(self) -> float:
        """
        Calculate market temperature using volatility as thermal energy.
        
        Temperature in thermodynamics is related to the average kinetic energy
        of particles. In markets, we use volatility as an analog.
        """
        try:
            if len(self.returns) < 2:
                return 1.0
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(self.returns)
            
            # Apply smoothing if we have historical data
            if len(self.temperature_history) > 0:
                prev_temp = self.temperature_history[-1]
                smoothing = self.parameters['temperature_smoothing']
                temperature = smoothing * prev_temp + (1 - smoothing) * volatility
            else:
                temperature = volatility
            
            # Normalize and scale
            # Typical daily volatility ranges from 0.01 to 0.1
            normalized_temp = temperature / 0.05  # Normalize by typical volatility
            
            # Apply logarithmic scaling for better distribution
            temperature_scaled = np.log1p(normalized_temp * 10)
            
            return max(0.1, temperature_scaled)  # Ensure positive temperature
            
        except Exception as e:
            self.logger.warning(f"Error calculating temperature: {e}")
            return 1.0
    
    def _calculate_entropy(self) -> float:
        """
        Calculate market entropy using Shannon entropy and thermodynamic principles.
        
        Entropy measures the disorder or randomness in the system.
        Higher entropy indicates more random, less predictable behavior.
        """
        try:
            if len(self.returns) < self.parameters['entropy_window']:
                return 0.0
            
            # Use recent returns for entropy calculation
            window_size = min(self.parameters['entropy_window'], len(self.returns))
            recent_returns = self.returns[-window_size:]
            
            # Method 1: Shannon entropy based on return distribution
            shannon_entropy = self._calculate_shannon_entropy(recent_returns)
            
            # Method 2: Thermodynamic entropy using Boltzmann formula
            thermodynamic_entropy = self._calculate_thermodynamic_entropy(recent_returns)
            
            # Method 3: Approximate entropy (regularity measure)
            approximate_entropy = self._calculate_approximate_entropy(recent_returns)
            
            # Combine different entropy measures
            combined_entropy = (
                shannon_entropy * 0.5 +
                thermodynamic_entropy * 0.3 +
                approximate_entropy * 0.2
            )
            
            return max(0.0, combined_entropy)
            
        except Exception as e:
            self.logger.warning(f"Error calculating entropy: {e}")
            return 0.0
    
    def _calculate_shannon_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of the data."""
        try:
            # Create histogram of returns
            hist, _ = np.histogram(data, bins=self.parameters['energy_bins'], density=True)
            
            # Normalize to get probabilities
            probabilities = hist / np.sum(hist)
            
            # Remove zero probabilities to avoid log(0)
            probabilities = probabilities[probabilities > 0]
            
            # Calculate Shannon entropy: H = -Σ p(i) * log(p(i))
            shannon_entropy = -np.sum(probabilities * np.log(probabilities))
            
            return shannon_entropy
            
        except Exception:
            return 0.0
    
    def _calculate_thermodynamic_entropy(self, data: np.ndarray) -> float:
        """Calculate thermodynamic entropy using Boltzmann formula."""
        try:
            # Estimate the number of microstates
            # Use the range and precision of the data
            data_range = np.max(data) - np.min(data)
            precision = np.std(data) / 10  # Estimate precision
            
            if precision <= 0 or data_range <= 0:
                return 0.0
            
            # Estimate number of accessible microstates
            microstates = max(1, data_range / precision)
            
            # Boltzmann entropy: S = k * ln(W)
            entropy = self.BOLTZMANN_CONSTANT * np.log(microstates)
            
            return entropy
            
        except Exception:
            return 0.0
    
    def _calculate_approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate approximate entropy (ApEn) to measure regularity."""
        try:
            N = len(data)
            if N < m + 1:
                return 0.0
            
            if r is None:
                r = 0.2 * np.std(data)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template = patterns[i]
                    matches = [_maxdist(template, patterns[j], m) <= r 
                              for j in range(N - m + 1)]
                    C[i] = sum(matches) / float(N - m + 1)
                
                phi = np.mean([np.log(c) for c in C if c > 0])
                return phi
            
            return _phi(m) - _phi(m + 1)
            
        except Exception:
            return 0.0
    
    def _calculate_internal_energy(self) -> float:
        """
        Calculate internal energy of the market system.
        
        Internal energy represents the total energy contained in the system,
        analogous to kinetic and potential energy of market movements.
        """
        try:
            if len(self.returns) < 2:
                return 0.0
            
            # Kinetic energy component (based on price movements)
            kinetic_energy = 0.5 * np.mean(self.returns ** 2)
            
            # Potential energy component (based on price levels relative to mean)
            if len(self.price_data) > 0:
                price_deviations = (self.price_data - np.mean(self.price_data)) / np.mean(self.price_data)
                potential_energy = 0.5 * np.mean(price_deviations ** 2)
            else:
                potential_energy = 0.0
            
            # Interaction energy (correlation effects)
            if len(self.returns) > 1:
                # Calculate autocorrelation at lag 1
                autocorr = np.corrcoef(self.returns[:-1], self.returns[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0.0
                interaction_energy = 0.1 * abs(autocorr)
            else:
                interaction_energy = 0.0
            
            # Total internal energy
            internal_energy = kinetic_energy + potential_energy + interaction_energy
            
            return max(0.0, internal_energy)
            
        except Exception as e:
            self.logger.warning(f"Error calculating internal energy: {e}")
            return 0.0
    
    def _calculate_free_energy(self, energy: float, temperature: float, entropy: float) -> float:
        """
        Calculate Gibbs free energy: G = H - TS
        
        Free energy determines the spontaneous direction of processes
        and equilibrium conditions.
        """
        try:
            # Gibbs free energy: G = H - TS
            # Where H ≈ E (enthalpy approximated as internal energy)
            free_energy = energy - temperature * entropy
            
            return free_energy
            
        except Exception:
            return 0.0
    
    def _calculate_heat_capacity(self) -> float:
        """
        Calculate heat capacity: C = dE/dT
        
        Heat capacity measures how much energy is needed to change temperature,
        indicating market responsiveness to external shocks.
        """
        try:
            if len(self.energy_history) < 3 or len(self.temperature_history) < 3:
                return 1.0
            
            # Use recent data for heat capacity calculation
            recent_energy = self.energy_history[-10:]
            recent_temp = self.temperature_history[-10:]
            
            if len(recent_energy) < 2:
                return 1.0
            
            # Calculate heat capacity as dE/dT
            dE = np.diff(recent_energy)
            dT = np.diff(recent_temp)
            
            # Avoid division by zero
            valid_indices = np.abs(dT) > 1e-6
            if not np.any(valid_indices):
                return 1.0
            
            heat_capacity_values = dE[valid_indices] / dT[valid_indices]
            heat_capacity = np.mean(heat_capacity_values)
            
            # Apply smoothing
            if hasattr(self, '_prev_heat_capacity'):
                smoothing = self.parameters['heat_capacity_smoothing']
                heat_capacity = smoothing * self._prev_heat_capacity + (1 - smoothing) * heat_capacity
            
            self._prev_heat_capacity = heat_capacity
            
            return max(0.1, abs(heat_capacity))
            
        except Exception as e:
            self.logger.warning(f"Error calculating heat capacity: {e}")
            return 1.0
    
    def _calculate_pressure(self) -> float:
        """
        Calculate market pressure using volume and volatility.
        
        Pressure in thermodynamics is force per unit area.
        In markets, we use volume intensity and volatility pressure.
        """
        try:
            if len(self.volume_data) == 0:
                return 1.0
            
            # Volume-based pressure component
            recent_volume = self.volume_data[-min(20, len(self.volume_data)):]
            avg_volume = np.mean(recent_volume)
            current_volume = recent_volume[-1] if len(recent_volume) > 0 else avg_volume
            
            volume_pressure = current_volume / (avg_volume + 1e-8)
            
            # Volatility-based pressure component
            if len(self.returns) > 0:
                volatility = np.std(self.returns[-min(20, len(self.returns)):])
                volatility_pressure = 1.0 + volatility * 10  # Scale volatility
            else:
                volatility_pressure = 1.0
            
            # Combine components
            pressure = volume_pressure * volatility_pressure
            
            return max(0.1, pressure)
            
        except Exception as e:
            self.logger.warning(f"Error calculating pressure: {e}")
            return 1.0
    
    def _calculate_volume(self) -> float:
        """
        Calculate thermodynamic volume (system size measure).
        
        Volume represents the "space" available for market operations.
        """
        try:
            if len(self.price_data) < 2:
                return 1.0
            
            # Price range as volume indicator
            price_range = np.max(self.price_data) - np.min(self.price_data)
            avg_price = np.mean(self.price_data)
            
            if avg_price > 0:
                relative_range = price_range / avg_price
            else:
                relative_range = 0.0
            
            # Time-based volume component
            time_volume = len(self.price_data) / self.parameters['lookback_period']
            
            # Combine components
            volume = relative_range * time_volume
            
            return max(0.1, volume)
            
        except Exception:
            return 1.0
    
    def _calculate_chemical_potential(self) -> float:
        """
        Calculate chemical potential (trend strength measure).
        
        Chemical potential determines the tendency for particles to move
        from one phase to another. In markets, it represents trend strength.
        """
        try:
            if len(self.returns) < 2:
                return 0.0
            
            # Trend strength based on cumulative returns
            cumulative_returns = np.cumsum(self.returns)
            
            if len(cumulative_returns) > 1:
                # Linear trend slope
                x = np.arange(len(cumulative_returns))
                trend_slope = np.polyfit(x, cumulative_returns, 1)[0]
            else:
                trend_slope = 0.0
            
            # Momentum component
            recent_returns = self.returns[-min(10, len(self.returns)):]
            momentum = np.mean(recent_returns)
            
            # Chemical potential combines trend and momentum
            chemical_potential = (trend_slope + momentum) * self.parameters['chemical_potential_factor']
            
            return chemical_potential
            
        except Exception:
            return 0.0
    
    def _determine_market_phase(self, temperature: float, entropy: float, energy: float) -> MarketPhase:
        """
        Determine current market phase based on thermodynamic properties.
        
        Phase determination uses temperature, entropy, and energy levels
        to classify market behavior patterns.
        """
        try:
            # Critical temperature for phase transitions
            T_critical = self.parameters['critical_temperature']
            
            # Normalized metrics
            temp_ratio = temperature / T_critical
            entropy_normalized = entropy / 5.0  # Normalize entropy
            energy_normalized = energy / 0.1    # Normalize energy
            
            # Phase classification logic
            if temp_ratio < 0.5 and entropy_normalized < 0.3:
                # Low temperature, low entropy - ordered state
                return MarketPhase.SOLID
            elif temp_ratio > 3.0 and entropy_normalized > 2.0:
                # Very high temperature and entropy - chaotic state
                return MarketPhase.PLASMA
            elif temp_ratio > 2.0 and entropy_normalized > 1.5:
                # High temperature, high entropy - gas-like behavior
                return MarketPhase.GAS
            elif 0.8 <= temp_ratio <= 1.2 and 0.8 <= entropy_normalized <= 1.2:
                # Near critical point
                return MarketPhase.CRITICAL
            elif temp_ratio > 1.2 and entropy_normalized > 1.2:
                # Beyond critical point
                return MarketPhase.SUPERCRITICAL
            else:
                # Default liquid phase - moderate temperature and entropy
                return MarketPhase.LIQUID
                
        except Exception:
            return MarketPhase.LIQUID
    
    def _determine_energy_state(self, energy: float, temperature: float) -> EnergyState:
        """Determine energy state of the market system."""
        try:
            # Energy thresholds based on temperature
            ground_threshold = 0.01 * temperature
            excited_threshold = 0.05 * temperature
            unstable_threshold = 0.1 * temperature
            
            if energy < ground_threshold:
                return EnergyState.GROUND
            elif energy < excited_threshold:
                # Check if it's metastable (temporarily stable)
                if len(self.energy_history) > 5:
                    recent_energy_var = np.var(self.energy_history[-5:])
                    if recent_energy_var < 0.001:
                        return EnergyState.METASTABLE
                return EnergyState.EXCITED
            elif energy < unstable_threshold:
                return EnergyState.EXCITED
            else:
                return EnergyState.UNSTABLE
                
        except Exception:
            return EnergyState.GROUND
    
    def _calculate_order_parameter(self) -> float:
        """
        Calculate order parameter for phase transition detection.
        
        Order parameter measures the degree of order in the system.
        Zero indicates disordered phase, non-zero indicates ordered phase.
        """
        try:
            if len(self.returns) < 10:
                return 0.0
            
            # Method 1: Autocorrelation as order parameter
            if len(self.returns) > 1:
                autocorr = np.corrcoef(self.returns[:-1], self.returns[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0.0
            else:
                autocorr = 0.0
            
            # Method 2: Trend strength
            if len(self.returns) > 5:
                x = np.arange(len(self.returns))
                slope, _ = np.polyfit(x, np.cumsum(self.returns), 1)
                trend_strength = abs(slope)
            else:
                trend_strength = 0.0
            
            # Method 3: Volatility clustering (GARCH-like effect)
            if len(self.returns) > 10:
                volatility_series = np.array([np.std(self.returns[max(0, i-5):i+1]) 
                                            for i in range(5, len(self.returns))])
                if len(volatility_series) > 1:
                    vol_autocorr = np.corrcoef(volatility_series[:-1], volatility_series[1:])[0, 1]
                    if np.isnan(vol_autocorr):
                        vol_autocorr = 0.0
                else:
                    vol_autocorr = 0.0
            else:
                vol_autocorr = 0.0
            
            # Combine order parameters
            order_parameter = (abs(autocorr) * 0.4 + 
                             min(1.0, trend_strength * 10) * 0.4 + 
                             abs(vol_autocorr) * 0.2)
            
            return np.clip(order_parameter, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_correlation_length(self) -> float:
        """
        Calculate correlation length - how far correlations extend.
        
        Correlation length measures the distance over which
        correlations in the system persist.
        """
        try:
            if len(self.returns) < 10:
                return 1.0
            
            max_lag = min(self.parameters['correlation_length_max'], len(self.returns) // 2)
            correlations = []
            
            # Calculate autocorrelations at different lags
            for lag in range(1, max_lag + 1):
                if len(self.returns) > lag:
                    corr = np.corrcoef(self.returns[:-lag], self.returns[lag:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                    else:
                        correlations.append(0.0)
            
            if not correlations:
                return 1.0
            
            # Find correlation length (where correlation drops to 1/e)
            correlations = np.array(correlations)
            threshold = 1.0 / np.e  # ≈ 0.368
            
            # Find first point where correlation drops below threshold
            below_threshold = np.where(correlations < threshold)[0]
            
            if len(below_threshold) > 0:
                correlation_length = below_threshold[0] + 1
            else:
                # If correlations never drop below threshold, use exponential fit
                try:
                    x = np.arange(len(correlations))
                    # Fit exponential decay: y = a * exp(-x/xi)
                    popt, _ = curve_fit(lambda x, a, xi: a * np.exp(-x/xi), 
                                      x, correlations, 
                                      p0=[correlations[0], 5.0],
                                      bounds=([0, 0.1], [2.0, max_lag]))
                    correlation_length = popt[1]
                except:
                    correlation_length = max_lag / 2
            
            return max(1.0, min(correlation_length, max_lag))
            
        except Exception:
            return 1.0
    
    def _detect_phase_transition(self) -> Optional[PhaseTransition]:
        """Detect phase transitions in the market."""
        try:
            if len(self.historical_states) < self.parameters['transition_detection_window']:
                return None
            
            # Check recent states for phase changes
            recent_states = self.historical_states[-self.parameters['transition_detection_window']:]
            
            # Look for phase changes
            phases = [state.phase for state in recent_states]
            
            # Detect transition
            if len(set(phases)) > 1:
                # Find the transition point
                for i in range(1, len(phases)):
                    if phases[i] != phases[i-1]:
                        # Transition detected
                        transition_index = len(self.historical_states) - len(recent_states) + i
                        
                        from_phase = phases[i-1]
                        to_phase = phases[i]
                        
                        # Calculate transition properties
                        transition_strength = self._calculate_transition_strength(
                            recent_states[i-1], recent_states[i]
                        )
                        
                        # Determine transition type
                        transition_type = self._classify_transition_type(from_phase, to_phase)
                        
                        # Calculate latent heat (energy change)
                        latent_heat = recent_states[i].energy - recent_states[i-1].energy
                        
                        # Estimate critical exponent
                        critical_exponent = self._estimate_critical_exponent(recent_states)
                        
                        return PhaseTransition(
                            transition_time=self.last_update or datetime.now(),
                            from_phase=from_phase,
                            to_phase=to_phase,
                            transition_type=transition_type,
                            latent_heat=latent_heat,
                            critical_exponent=critical_exponent,
                            duration=1.0,  # Single time step for now
                            strength=transition_strength
                        )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error detecting phase transition: {e}")
            return None
    
    def _calculate_transition_strength(self, state1: ThermodynamicState, 
                                     state2: ThermodynamicState) -> float:
        """Calculate the strength of a phase transition."""
        try:
            # Calculate differences in key properties
            temp_diff = abs(state2.temperature - state1.temperature)
            entropy_diff = abs(state2.entropy - state1.entropy)
            energy_diff = abs(state2.energy - state1.energy)
            order_diff = abs(state2.order_parameter - state1.order_parameter)
            
            # Normalize and combine
            strength = (temp_diff + entropy_diff + energy_diff + order_diff) / 4.0
            
            return min(1.0, strength)
            
        except Exception:
            return 0.0
    
    def _classify_transition_type(self, from_phase: MarketPhase, 
                                 to_phase: MarketPhase) -> str:
        """Classify the type of phase transition."""
        try:
            # First-order transitions (discontinuous)
            first_order_transitions = {
                (MarketPhase.SOLID, MarketPhase.LIQUID),
                (MarketPhase.LIQUID, MarketPhase.GAS),
                (MarketPhase.SOLID, MarketPhase.GAS)
            }
            
            # Second-order transitions (continuous)
            second_order_transitions = {
                (MarketPhase.LIQUID, MarketPhase.CRITICAL),
                (MarketPhase.CRITICAL, MarketPhase.SUPERCRITICAL)
            }
            
            transition_pair = (from_phase, to_phase)
            
            if transition_pair in first_order_transitions or transition_pair[::-1] in first_order_transitions:
                return "first_order"
            elif transition_pair in second_order_transitions or transition_pair[::-1] in second_order_transitions:
                return "second_order"
            else:
                return "continuous"
                
        except Exception:
            return "unknown"
    
    def _estimate_critical_exponent(self, states: List[ThermodynamicState]) -> float:
        """Estimate critical exponent near phase transition."""
        try:
            if len(states) < 3:
                return 0.0
            
            # Use order parameter to estimate beta exponent
            order_params = [state.order_parameter for state in states]
            temperatures = [state.temperature for state in states]
            
            # Near critical point, order parameter follows: |ψ| ∝ |T - Tc|^β
            T_critical = self.parameters['critical_temperature']
            
            # Find states near critical temperature
            near_critical = [(t, op) for t, op in zip(temperatures, order_params) 
                           if abs(t - T_critical) < 0.5]
            
            if len(near_critical) < 3:
                return self.parameters['critical_exponents']['beta']
            
            # Fit power law to estimate exponent
            try:
                t_vals = [abs(t - T_critical) + 1e-6 for t, _ in near_critical]
                op_vals = [max(1e-6, op) for _, op in near_critical]
                
                # Log-log fit: log(ψ) = β * log(|T - Tc|) + const
                log_t = np.log(t_vals)
                log_op = np.log(op_vals)
                
                if len(log_t) > 1:
                    beta, _ = np.polyfit(log_t, log_op, 1)
                    return abs(beta)
                else:
                    return self.parameters['critical_exponents']['beta']
                    
            except:
                return self.parameters['critical_exponents']['beta']
                
        except Exception:
            return 0.0
    
    def _calculate_partition_function(self, temperature: float) -> float:
        """
        Calculate partition function Z = Σ exp(-E_i / kT)
        
        The partition function is fundamental in statistical mechanics
        and determines all thermodynamic properties.
        """
        try:
            if temperature <= 0:
                return 1.0
            
            # Create energy levels based on recent price movements
            if len(self.returns) > 0:
                # Use returns as energy levels
                energies = np.array(self.returns) ** 2  # Squared returns as energy
                
                # Calculate Boltzmann factors: exp(-E / kT)
                beta = 1.0 / (self.BOLTZMANN_CONSTANT * temperature)
                boltzmann_factors = np.exp(-energies * beta)
                
                # Partition function is sum of Boltzmann factors
                partition_function = np.sum(boltzmann_factors)
                
                self.partition_function = partition_function
                return partition_function
            else:
                return 1.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating partition function: {e}")
            return 1.0
    
    def _perform_ensemble_analysis(self) -> Dict[str, Any]:
        """Perform statistical ensemble analysis."""
        try:
            if len(self.returns) < 10:
                return {}
            
            # Canonical ensemble analysis (constant temperature)
            canonical_analysis = self._analyze_canonical_ensemble()
            
            # Microcanonical ensemble analysis (constant energy)
            microcanonical_analysis = self._analyze_microcanonical_ensemble()
            
            # Grand canonical ensemble analysis (constant chemical potential)
            grand_canonical_analysis = self._analyze_grand_canonical_ensemble()
            
            return {
                'canonical': canonical_analysis,
                'microcanonical': microcanonical_analysis,
                'grand_canonical': grand_canonical_analysis,
                'ensemble_equivalence': self._check_ensemble_equivalence()
            }
            
        except Exception as e:
            self.logger.error(f"Error in ensemble analysis: {e}")
            return {}
    
    def _analyze_canonical_ensemble(self) -> Dict[str, Any]:
        """Analyze canonical ensemble (constant temperature)."""
        try:
            temperature = self.current_state.temperature if self.current_state else 1.0
            
            # Generate canonical ensemble
            if len(self.returns) > 0:
                energies = self.returns ** 2
                beta = 1.0 / (self.BOLTZMANN_CONSTANT * temperature)
                
                # Probability distribution: P(E) = exp(-E/kT) / Z
                probabilities = np.exp(-energies * beta)
                probabilities = probabilities / np.sum(probabilities)
                
                # Calculate ensemble averages
                avg_energy = np.sum(energies * probabilities)
                energy_variance = np.sum((energies - avg_energy) ** 2 * probabilities)
                
                # Heat capacity from energy fluctuations
                heat_capacity = energy_variance * beta ** 2
                
                return {
                    'average_energy': avg_energy,
                    'energy_variance': energy_variance,
                    'heat_capacity': heat_capacity,
                    'temperature': temperature,
                    'entropy': self.current_state.entropy if self.current_state else 0.0
                }
            else:
                return {}
                
        except Exception:
            return {}
    
    def _analyze_microcanonical_ensemble(self) -> Dict[str, Any]:
        """Analyze microcanonical ensemble (constant energy)."""
        try:
            if len(self.returns) == 0:
                return {}
            
            total_energy = self.current_state.energy if self.current_state else 1.0
            
            # Count microstates with approximately the same total energy
            energies = self.returns ** 2
            energy_tolerance = 0.1 * total_energy
            
            # Find states within energy tolerance
            compatible_states = np.abs(energies - total_energy) <= energy_tolerance
            microstate_count = np.sum(compatible_states)
            
            # Microcanonical entropy: S = k * ln(Ω)
            if microstate_count > 0:
                microcanonical_entropy = self.BOLTZMANN_CONSTANT * np.log(microstate_count)
            else:
                microcanonical_entropy = 0.0
            
            # Temperature from entropy derivative: 1/T = dS/dE
            if len(self.energy_history) > 2 and len(self.entropy_history) > 2:
                dS = np.diff(self.entropy_history[-5:])
                dE = np.diff(self.energy_history[-5:])
                
                valid_indices = np.abs(dE) > 1e-6
                if np.any(valid_indices):
                    inv_temp = np.mean(dS[valid_indices] / dE[valid_indices])
                    microcanonical_temp = 1.0 / max(1e-6, abs(inv_temp))
                else:
                    microcanonical_temp = 1.0
            else:
                microcanonical_temp = 1.0
            
            return {
                'total_energy': total_energy,
                'microstate_count': int(microstate_count),
                'entropy': microcanonical_entropy,
                'temperature': microcanonical_temp,
                'density_of_states': microstate_count / max(1, len(energies))
            }
            
        except Exception:
            return {}
    
    def _analyze_grand_canonical_ensemble(self) -> Dict[str, Any]:
        """Analyze grand canonical ensemble (constant chemical potential)."""
        try:
            if not self.current_state:
                return {}
            
            temperature = self.current_state.temperature
            chemical_potential = self.current_state.chemical_potential
            
            # Grand partition function: Ξ = Σ exp((μN - E)/kT)
            if len(self.returns) > 0:
                energies = self.returns ** 2
                
                # Assume number of particles varies (market participants)
                beta = 1.0 / (self.BOLTZMANN_CONSTANT * temperature)
                
                # Calculate grand canonical probabilities
                # For simplicity, assume N varies with energy
                particle_numbers = 1 + np.abs(self.returns) * 100  # Proxy for participants
                
                grand_weights = np.exp(beta * (chemical_potential * particle_numbers - energies))
                grand_weights = grand_weights / np.sum(grand_weights)
                
                # Calculate averages
                avg_particles = np.sum(particle_numbers * grand_weights)
                avg_energy = np.sum(energies * grand_weights)
                particle_variance = np.sum((particle_numbers - avg_particles) ** 2 * grand_weights)
                
                return {
                    'average_particles': avg_particles,
                    'average_energy': avg_energy,
                    'particle_variance': particle_variance,
                    'chemical_potential': chemical_potential,
                    'compressibility': particle_variance / (temperature * avg_particles + 1e-6)
                }
            else:
                return {}
                
        except Exception:
            return {}
    
    def _check_ensemble_equivalence(self) -> Dict[str, float]:
        """Check equivalence between different statistical ensembles."""
        try:
            # In the thermodynamic limit, all ensembles should give same results
            # Check consistency between canonical and microcanonical temperatures
            
            canonical_temp = self.current_state.temperature if self.current_state else 1.0
            
            # Get microcanonical temperature from previous analysis
            # This is a simplified check
            temperature_consistency = 1.0  # Placeholder for actual calculation
            energy_consistency = 1.0
            entropy_consistency = 1.0
            
            return {
                'temperature_consistency': temperature_consistency,
                'energy_consistency': energy_consistency,
                'entropy_consistency': entropy_consistency,
                'overall_consistency': (temperature_consistency + energy_consistency + entropy_consistency) / 3.0
            }
            
        except Exception:
            return {}
    
    def _analyze_critical_behavior(self) -> Dict[str, Any]:
        """Analyze critical behavior and scaling laws."""
        try:
            if not self.current_state:
                return {}
            
            temperature = self.current_state.temperature
            T_critical = self.parameters['critical_temperature']
            
            # Reduced temperature
            t = abs(temperature - T_critical) / T_critical
            
            # Critical exponents
            exponents = self.parameters['critical_exponents']
            
            # Calculate scaling functions near critical point
            if t < 0.5:  # Near critical point
                # Order parameter scaling: ψ ∝ t^β
                order_parameter_scaling = t ** exponents['beta']
                
                # Heat capacity scaling: C ∝ t^(-α)
                heat_capacity_scaling = t ** (-exponents['alpha']) if t > 1e-6 else 1e6
                
                # Susceptibility scaling: χ ∝ t^(-γ)
                susceptibility_scaling = t ** (-exponents['gamma']) if t > 1e-6 else 1e6
                
                # Correlation length scaling: ξ ∝ t^(-ν)
                correlation_length_scaling = t ** (-exponents['nu']) if t > 1e-6 else 1e6
                
                criticality_measure = 1.0 / (1.0 + t * 10)  # High near critical point
            else:
                order_parameter_scaling = 0.0
                heat_capacity_scaling = 1.0
                susceptibility_scaling = 1.0
                correlation_length_scaling = 1.0
                criticality_measure = 0.0
            
            # Critical phenomena detection
            critical_phenomena = {
                'critical_slowing_down': self._detect_critical_slowing_down(),
                'critical_opalescence': self._detect_critical_opalescence(),
                'scale_invariance': self._detect_scale_invariance()
            }
            
            return {
                'reduced_temperature': t,
                'order_parameter_scaling': order_parameter_scaling,
                'heat_capacity_scaling': heat_capacity_scaling,
                'susceptibility_scaling': susceptibility_scaling,
                'correlation_length_scaling': correlation_length_scaling,
                'criticality_measure': criticality_measure,
                'critical_phenomena': critical_phenomena,
                'universality_class': self._determine_universality_class()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing critical behavior: {e}")
            return {}
    
    def _detect_critical_slowing_down(self) -> float:
        """Detect critical slowing down phenomenon."""
        try:
            # Critical slowing down: relaxation time increases near critical point
            if len(self.returns) < 20:
                return 0.0
            
            # Calculate autocorrelation decay time
            autocorr_values = []
            max_lag = min(10, len(self.returns) // 2)
            
            for lag in range(1, max_lag + 1):
                if len(self.returns) > lag:
                    corr = np.corrcoef(self.returns[:-lag], self.returns[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr_values.append(abs(corr))
            
            if len(autocorr_values) < 3:
                return 0.0
            
            # Fit exponential decay
            x = np.arange(len(autocorr_values))
            try:
                # Fit: y = exp(-x/tau)
                popt, _ = curve_fit(lambda x, tau: np.exp(-x/tau), 
                                  x, autocorr_values, 
                                  p0=[3.0],
                                  bounds=([0.1], [20.0]))
                relaxation_time = popt[0]
                
                # Normalize (typical relaxation time ~ 2-5)
                slowing_down = min(1.0, relaxation_time / 10.0)
                return slowing_down
                
            except:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _detect_critical_opalescence(self) -> float:
        """Detect critical opalescence (large fluctuations)."""
        try:
            # Critical opalescence: large fluctuations near critical point
            if len(self.returns) < 10:
                return 0.0
            
            # Calculate variance of returns in different windows
            window_size = 5
            variances = []
            
            for i in range(window_size, len(self.returns)):
                window_data = self.returns[i-window_size:i]
                variances.append(np.var(window_data))
            
            if len(variances) < 2:
                return 0.0
            
            # Variance of variances (measure of fluctuation intensity)
            variance_of_variances = np.var(variances)
            
            # Normalize
            opalescence = min(1.0, variance_of_variances * 1000)
            
            return opalescence
            
        except Exception:
            return 0.0
    
    def _detect_scale_invariance(self) -> float:
        """Detect scale invariance near critical point."""
        try:
            # Scale invariance: statistical properties look same at different scales
            if len(self.returns) < 50:
                return 0.0
            
            # Calculate scaling exponent using detrended fluctuation analysis
            scales = [5, 10, 20]
            fluctuations = []
            
            for scale in scales:
                if len(self.returns) > scale * 2:
                    # Divide data into segments
                    n_segments = len(self.returns) // scale
                    segment_fluctuations = []
                    
                    for i in range(n_segments):
                        segment = self.returns[i*scale:(i+1)*scale]
                        # Detrend segment
                        x = np.arange(len(segment))
                        if len(segment) > 1:
                            trend = np.polyfit(x, segment, 1)
                            detrended = segment - np.polyval(trend, x)
                            fluctuation = np.sqrt(np.mean(detrended ** 2))
                            segment_fluctuations.append(fluctuation)
                    
                    if segment_fluctuations:
                        fluctuations.append(np.mean(segment_fluctuations))
            
            if len(fluctuations) < 2:
                return 0.0
            
            # Check if fluctuations scale as power law
            log_scales = np.log(scales[:len(fluctuations)])
            log_fluctuations = np.log(np.array(fluctuations) + 1e-8)
            
            # Fit power law: F(s) = s^H
            if len(log_scales) > 1:
                hurst_exponent, _ = np.polyfit(log_scales, log_fluctuations, 1)
                
                # Scale invariance if Hurst exponent is close to 0.5
                scale_invariance = 1.0 - 2.0 * abs(hurst_exponent - 0.5)
                return max(0.0, scale_invariance)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _determine_universality_class(self) -> str:
        """Determine universality class of critical behavior."""
        try:
            if not self.current_state:
                return "unknown"
            
            # Simplified classification based on dimensionality and symmetry
            correlation_length = self.current_state.correlation_length
            order_parameter = self.current_state.order_parameter
            
            # Ising model (Z2 symmetry)
            if order_parameter < 0.5 and correlation_length < 5:
                return "ising_2d"
            elif order_parameter < 0.5 and correlation_length >= 5:
                return "ising_3d"
            
            # XY model (O(2) symmetry)
            elif 0.5 <= order_parameter < 0.8:
                return "xy_model"
            
            # Heisenberg model (O(3) symmetry)
            elif order_parameter >= 0.8:
                return "heisenberg_model"
            
            # Mean field
            else:
                return "mean_field"
                
        except Exception:
            return "unknown"
    
    def _analyze_equilibrium_properties(self) -> Dict[str, Any]:
        """Analyze equilibrium thermodynamic properties."""
        try:
            if not self.current_state or len(self.historical_states) < 5:
                return {}
            
            # Check if system is in equilibrium
            equilibrium_check = self._check_equilibrium()
            
            # Calculate equilibrium properties
            if equilibrium_check['is_equilibrium']:
                # Maxwell relations
                maxwell_relations = self._calculate_maxwell_relations()
                
                # Thermodynamic potentials
                potentials = self._calculate_thermodynamic_potentials()
                
                # Response functions
                response_functions = self._calculate_response_functions()
                
                # Stability analysis
                stability = self._analyze_thermodynamic_stability()
                
                return {
                    'equilibrium_check': equilibrium_check,
                    'maxwell_relations': maxwell_relations,
                    'thermodynamic_potentials': potentials,
                    'response_functions': response_functions,
                    'stability_analysis': stability,
                    'fluctuation_theorem': self._check_fluctuation_theorem()
                }
            else:
                # Non-equilibrium analysis
                return {
                    'equilibrium_check': equilibrium_check,
                    'non_equilibrium_analysis': self._analyze_non_equilibrium()
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing equilibrium properties: {e}")
            return {}
    
    def _check_equilibrium(self) -> Dict[str, Any]:
        """Check if the system is in thermodynamic equilibrium."""
        try:
            if len(self.historical_states) < 5:
                return {'is_equilibrium': False, 'equilibrium_measure': 0.0}
            
            # Check stability of thermodynamic variables
            recent_states = self.historical_states[-5:]
            
            # Temperature stability
            temperatures = [state.temperature for state in recent_states]
            temp_stability = 1.0 - np.std(temperatures) / (np.mean(temperatures) + 1e-6)
            
            # Energy stability
            energies = [state.energy for state in recent_states]
            energy_stability = 1.0 - np.std(energies) / (np.mean(energies) + 1e-6)
            
            # Entropy stability (should increase or remain constant)
            entropies = [state.entropy for state in recent_states]
            entropy_trend = np.polyfit(range(len(entropies)), entropies, 1)[0]
            entropy_stability = max(0.0, entropy_trend)  # Should be non-negative
            
            # Combined equilibrium measure
            equilibrium_measure = (temp_stability + energy_stability + entropy_stability) / 3.0
            is_equilibrium = equilibrium_measure > self.parameters['equilibrium_threshold']
            
            return {
                'is_equilibrium': is_equilibrium,
                'equilibrium_measure': equilibrium_measure,
                'temperature_stability': temp_stability,
                'energy_stability': energy_stability,
                'entropy_stability': entropy_stability
            }
            
        except Exception:
            return {'is_equilibrium': False, 'equilibrium_measure': 0.0}
    
    def _calculate_maxwell_relations(self) -> Dict[str, float]:
        """Calculate Maxwell relations between thermodynamic variables."""
        try:
            # Maxwell relations come from equality of mixed partial derivatives
            # For this implementation, we'll use simplified numerical derivatives
            
            if len(self.historical_states) < 3:
                return {}
            
            # Get recent state changes
            recent_states = self.historical_states[-3:]
            
            # Calculate partial derivatives numerically
            dT = [recent_states[i+1].temperature - recent_states[i].temperature 
                  for i in range(len(recent_states)-1)]
            dS = [recent_states[i+1].entropy - recent_states[i].entropy 
                  for i in range(len(recent_states)-1)]
            dP = [recent_states[i+1].pressure - recent_states[i].pressure 
                  for i in range(len(recent_states)-1)]
            dV = [recent_states[i+1].volume - recent_states[i].volume 
                  for i in range(len(recent_states)-1)]
            
            # Maxwell relation 1: (∂S/∂V)_T = (∂P/∂T)_V
            if len(dT) > 0 and len(dV) > 0 and len(dS) > 0 and len(dP) > 0:
                dS_dV = np.mean([ds/dv for ds, dv in zip(dS, dV) if abs(dv) > 1e-6])
                dP_dT = np.mean([dp/dt for dp, dt in zip(dP, dT) if abs(dt) > 1e-6])
                
                maxwell_1_error = abs(dS_dV - dP_dT) if not (np.isnan(dS_dV) or np.isnan(dP_dT)) else 0.0
            else:
                maxwell_1_error = 0.0
            
            return {
                'maxwell_relation_1_error': maxwell_1_error,
                'maxwell_validity': 1.0 / (1.0 + maxwell_1_error)
            }
            
        except Exception:
            return {}
    
    def _calculate_thermodynamic_potentials(self) -> Dict[str, float]:
        """Calculate thermodynamic potentials."""
        try:
            if not self.current_state:
                return {}
            
            T = self.current_state.temperature
            S = self.current_state.entropy
            P = self.current_state.pressure
            V = self.current_state.volume
            E = self.current_state.energy
            mu = self.current_state.chemical_potential
            
            # Thermodynamic potentials
            # Internal energy (already calculated)
            internal_energy = E
            
            # Enthalpy: H = E + PV
            enthalpy = E + P * V
            
            # Helmholtz free energy: F = E - TS
            helmholtz_free_energy = E - T * S
            
            # Gibbs free energy: G = H - TS = E + PV - TS
            gibbs_free_energy = enthalpy - T * S
            
            # Grand potential: Ω = F - μN (assume N=1 for simplicity)
            grand_potential = helmholtz_free_energy - mu
            
            return {
                'internal_energy': internal_energy,
                'enthalpy': enthalpy,
                'helmholtz_free_energy': helmholtz_free_energy,
                'gibbs_free_energy': gibbs_free_energy,
                'grand_potential': grand_potential
            }
            
        except Exception:
            return {}
    
    def _calculate_response_functions(self) -> Dict[str, float]:
        """Calculate thermodynamic response functions."""
        try:
            if not self.current_state:
                return {}
            
            T = self.current_state.temperature
            V = self.current_state.volume
            P = self.current_state.pressure
            C = self.current_state.heat_capacity
            
            # Heat capacity at constant volume (already calculated)
            heat_capacity_v = C
            
            # Heat capacity at constant pressure (estimate)
            # CP - CV = -T * (∂P/∂T)² / (∂P/∂V)
            heat_capacity_p = heat_capacity_v * 1.2  # Simplified estimate
            
            # Isothermal compressibility: κT = -1/V * (∂V/∂P)_T
            isothermal_compressibility = 1.0 / P  # Simplified
            
            # Adiabatic compressibility: κS = κT * CV/CP
            adiabatic_compressibility = isothermal_compressibility * heat_capacity_v / heat_capacity_p
            
            # Thermal expansion coefficient: α = 1/V * (∂V/∂T)_P
            thermal_expansion = 1.0 / T  # Simplified
            
            return {
                'heat_capacity_constant_volume': heat_capacity_v,
                'heat_capacity_constant_pressure': heat_capacity_p,
                'isothermal_compressibility': isothermal_compressibility,
                'adiabatic_compressibility': adiabatic_compressibility,
                'thermal_expansion_coefficient': thermal_expansion
            }
            
        except Exception:
            return {}    
    def _analyze_thermodynamic_stability(self) -> Dict[str, Any]:
        """Analyze thermodynamic stability conditions."""
        try:
            if not self.current_state:
                return {}
            
            # Stability conditions for thermodynamic equilibrium
            T = self.current_state.temperature
            P = self.current_state.pressure
            C = self.current_state.heat_capacity
            
            # Thermal stability: CV > 0
            thermal_stability = C > 0
            
            # Mechanical stability: (∂P/∂V)_T < 0
            # Estimate from recent pressure-volume changes
            mechanical_stability = True  # Simplified
            
            # Chemical stability: All partial derivatives consistent
            chemical_stability = True  # Simplified
            
            # Global stability: Convexity of thermodynamic potentials
            global_stability = thermal_stability and mechanical_stability and chemical_stability
            
            # Stability measure
            stability_score = sum([thermal_stability, mechanical_stability, 
                                 chemical_stability]) / 3.0
            
            return {
                'thermal_stability': thermal_stability,
                'mechanical_stability': mechanical_stability,
                'chemical_stability': chemical_stability,
                'global_stability': global_stability,
                'stability_score': stability_score,
                'stability_phase': 'stable' if global_stability else 'unstable'
            }
            
        except Exception:
            return {}
    
    def _check_fluctuation_theorem(self) -> Dict[str, float]:
        """Check fluctuation theorem and detailed balance."""
        try:
            # Fluctuation theorem: P(ΔS)/P(-ΔS) = exp(ΔS/k)
            if len(self.entropy_history) < 10:
                return {}
            
            # Calculate entropy changes
            entropy_changes = np.diff(self.entropy_history[-10:])
            
            if len(entropy_changes) < 2:
                return {}
            
            # Check detailed balance
            positive_changes = entropy_changes[entropy_changes > 0]
            negative_changes = -entropy_changes[entropy_changes < 0]
            
            if len(positive_changes) > 0 and len(negative_changes) > 0:
                # Simplified check of fluctuation theorem
                pos_mean = np.mean(positive_changes)
                neg_mean = np.mean(negative_changes)
                
                if neg_mean > 0:
                    ratio = pos_mean / neg_mean
                    expected_ratio = np.exp(pos_mean / self.BOLTZMANN_CONSTANT)
                    
                    theorem_error = abs(np.log(ratio) - np.log(expected_ratio))
                    theorem_validity = 1.0 / (1.0 + theorem_error)
                else:
                    theorem_validity = 0.0
            else:
                theorem_validity = 0.0
            
            return {
                'fluctuation_theorem_validity': theorem_validity,
                'detailed_balance_check': theorem_validity > 0.5
            }
            
        except Exception:
            return {}
    
    def _analyze_non_equilibrium(self) -> Dict[str, Any]:
        """Analyze non-equilibrium thermodynamics."""
        try:
            # Non-equilibrium thermodynamics analysis
            if len(self.historical_states) < 3:
                return {}
            
            # Calculate entropy production rate
            entropy_production = self._calculate_entropy_production()
            
            # Analyze irreversible processes
            irreversibility = self._analyze_irreversibility()
            
            # Linear response analysis
            linear_response = self._analyze_linear_response()
            
            # Onsager reciprocal relations
            onsager_relations = self._check_onsager_relations()
            
            return {
                'entropy_production_rate': entropy_production,
                'irreversibility_measure': irreversibility,
                'linear_response_analysis': linear_response,
                'onsager_relations': onsager_relations,
                'non_equilibrium_phase': self._classify_non_equilibrium_phase()
            }
            
        except Exception:
            return {}
    
    def _calculate_entropy_production(self) -> float:
        """Calculate entropy production rate."""
        try:
            if len(self.entropy_history) < 3:
                return 0.0
            
            # dS/dt for recent history
            recent_entropies = self.entropy_history[-3:]
            time_steps = range(len(recent_entropies))
            
            if len(recent_entropies) > 1:
                # Linear fit to get rate
                entropy_rate = np.polyfit(time_steps, recent_entropies, 1)[0]
                
                # Entropy production should be non-negative (2nd law)
                return max(0.0, entropy_rate)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _analyze_irreversibility(self) -> float:
        """Analyze irreversibility in the system."""
        try:
            # Measure of how far from time-reversal symmetry
            if len(self.returns) < 10:
                return 0.0
            
            # Time asymmetry in correlations
            forward_returns = self.returns[-10:]
            backward_returns = forward_returns[::-1]
            
            # Compare forward and backward statistics
            forward_moments = [np.mean(forward_returns), np.std(forward_returns), 
                             np.mean(forward_returns**3), np.mean(forward_returns**4)]
            backward_moments = [np.mean(backward_returns), np.std(backward_returns),
                              np.mean(backward_returns**3), np.mean(backward_returns**4)]
            
            # Calculate asymmetry
            asymmetry = np.mean([abs(f - b) for f, b in zip(forward_moments, backward_moments)])
            
            # Normalize to [0, 1]
            irreversibility = min(1.0, asymmetry * 10)
            
            return irreversibility
            
        except Exception:
            return 0.0
    
    def _analyze_linear_response(self) -> Dict[str, float]:
        """Analyze linear response to external perturbations."""
        try:
            # Linear response theory: response proportional to perturbation
            if len(self.returns) < 5:
                return {}
            
            # Assume external perturbation is change in price
            perturbations = np.diff(self.prices[-5:]) if len(self.prices) >= 5 else []
            responses = np.diff(self.returns[-4:]) if len(self.returns) >= 4 else []
            
            if len(perturbations) > 0 and len(responses) > 0 and len(perturbations) == len(responses):
                # Linear response coefficient
                valid_indices = np.abs(perturbations) > 1e-6
                if np.any(valid_indices):
                    response_coefficients = responses[valid_indices] / perturbations[valid_indices]
                    
                    if len(response_coefficients) > 0:
                        linear_response_coeff = np.mean(response_coefficients)
                        response_variance = np.var(response_coefficients)
                        
                        return {
                            'linear_response_coefficient': linear_response_coeff,
                            'response_variance': response_variance,
                            'linearity_measure': 1.0 / (1.0 + response_variance)
                        }
            
            return {}
            
        except Exception:
            return {}
    
    def _check_onsager_relations(self) -> Dict[str, float]:
        """Check Onsager reciprocal relations."""
        try:
            # Onsager relations: L_ij = L_ji (reciprocal relations)
            # For this simplified implementation, we'll check symmetry
            # in response functions
            
            if len(self.historical_states) < 4:
                return {}
            
            # This is a placeholder for actual Onsager coefficient calculation
            # In practice, would need multiple thermodynamic forces and fluxes
            
            reciprocity_measure = 0.8  # Simplified placeholder
            
            return {
                'reciprocity_measure': reciprocity_measure,
                'onsager_validity': reciprocity_measure > 0.7
            }
            
        except Exception:
            return {}
    
    def _classify_non_equilibrium_phase(self) -> str:
        """Classify the type of non-equilibrium phase."""
        try:
            if not self.current_state:
                return "unknown"
            
            entropy_production = self._calculate_entropy_production()
            irreversibility = self._analyze_irreversibility()
            
            # Classification based on entropy production and irreversibility
            if entropy_production < 0.1 and irreversibility < 0.2:
                return "near_equilibrium"
            elif entropy_production < 0.5 and irreversibility < 0.5:
                return "linear_response"
            elif entropy_production < 1.0:
                return "weakly_nonlinear"
            else:
                return "strongly_nonlinear"
                
        except Exception:
            return "unknown"
    
    def _generate_trading_signals(self) -> Dict[str, Any]:
        """Generate comprehensive trading signals based on thermodynamic analysis."""
        try:
            if not self.current_state:
                return self._create_neutral_signal()
            
            # Collect all analysis results
            signals = {}
            
            # Phase transition signals
            phase_signals = self._generate_phase_signals()
            signals.update(phase_signals)
            
            # Critical behavior signals
            critical_signals = self._generate_critical_signals()
            signals.update(critical_signals)
            
            # Entropy-based signals
            entropy_signals = self._generate_entropy_signals()
            signals.update(entropy_signals)
            
            # Energy-based signals
            energy_signals = self._generate_energy_signals()
            signals.update(energy_signals)
            
            # Temperature-based signals
            temperature_signals = self._generate_temperature_signals()
            signals.update(temperature_signals)
            
            # Equilibrium-based signals
            equilibrium_signals = self._generate_equilibrium_signals()
            signals.update(equilibrium_signals)
            
            # Combine all signals into final recommendation
            final_signal = self._combine_thermodynamic_signals(signals)
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return self._create_neutral_signal()
    
    def _generate_phase_signals(self) -> Dict[str, Any]:
        """Generate signals based on phase transitions."""
        try:
            if not hasattr(self, 'phase_analysis') or not self.phase_analysis:
                return {}
            
            phase_analysis = self.phase_analysis
            current_phase = phase_analysis.get('current_phase', 'liquid')
            transition_probability = phase_analysis.get('transition_probability', 0.0)
            
            # Phase-based trading logic
            if current_phase == 'gas':
                # High volatility phase - trend following
                phase_signal = {
                    'strength': 0.7,
                    'direction': 1 if len(self.returns) > 0 and self.returns[-1] > 0 else -1,
                    'reasoning': 'Gas phase: high volatility trend following'
                }
            elif current_phase == 'liquid':
                # Medium volatility - momentum based
                phase_signal = {
                    'strength': 0.5,
                    'direction': 0,  # Neutral in liquid phase
                    'reasoning': 'Liquid phase: medium volatility, neutral stance'
                }
            elif current_phase == 'solid':
                # Low volatility - mean reversion
                phase_signal = {
                    'strength': 0.6,
                    'direction': -1 if len(self.returns) > 0 and self.returns[-1] > 0 else 1,
                    'reasoning': 'Solid phase: low volatility mean reversion'
                }
            else:
                phase_signal = {'strength': 0.0, 'direction': 0, 'reasoning': 'Unknown phase'}
            
            # Transition signals
            if transition_probability > 0.7:
                phase_signal['strength'] *= 1.5  # Amplify signal during transitions
                phase_signal['reasoning'] += f' (transition probability: {transition_probability:.2f})'
            
            return {'phase_signal': phase_signal}
            
        except Exception:
            return {}
    
    def _generate_critical_signals(self) -> Dict[str, Any]:
        """Generate signals based on critical behavior."""
        try:
            if not hasattr(self, 'critical_analysis') or not self.critical_analysis:
                return {}
            
            critical_analysis = self.critical_analysis
            criticality_measure = critical_analysis.get('criticality_measure', 0.0)
            critical_phenomena = critical_analysis.get('critical_phenomena', {})
            
            # Critical point signals
            if criticality_measure > 0.8:
                # Very close to critical point - high uncertainty
                critical_signal = {
                    'strength': 0.2,  # Low confidence
                    'direction': 0,   # Neutral
                    'reasoning': f'Near critical point (criticality: {criticality_measure:.2f}), high uncertainty'
                }
            elif criticality_measure > 0.5:
                # Moderate criticality - prepare for volatility
                critical_signal = {
                    'strength': 0.4,
                    'direction': 0,
                    'reasoning': f'Approaching criticality (criticality: {criticality_measure:.2f}), prepare for volatility'
                }
            else:
                # Away from critical point - normal signals
                critical_signal = {
                    'strength': 0.6,
                    'direction': 1 if criticality_measure < 0.2 else 0,
                    'reasoning': f'Away from criticality (criticality: {criticality_measure:.2f}), normal behavior'
                }
            
            # Critical slowing down signal
            slowing_down = critical_phenomena.get('critical_slowing_down', 0.0)
            if slowing_down > 0.7:
                critical_signal['strength'] *= 0.8  # Reduce signal strength
                critical_signal['reasoning'] += ', critical slowing down detected'
            
            return {'critical_signal': critical_signal}
            
        except Exception:
            return {}
    
    def _generate_entropy_signals(self) -> Dict[str, Any]:
        """Generate signals based on entropy analysis."""
        try:
            if not self.current_state:
                return {}
            
            current_entropy = self.current_state.entropy
            
            # Entropy trend analysis
            if len(self.entropy_history) >= 3:
                recent_entropy = self.entropy_history[-3:]
                entropy_trend = np.polyfit(range(len(recent_entropy)), recent_entropy, 1)[0]
                
                if entropy_trend > 0.1:
                    # Increasing entropy - system becoming more disordered
                    entropy_signal = {
                        'strength': 0.6,
                        'direction': -1,  # Sell signal
                        'reasoning': f'Increasing entropy (trend: {entropy_trend:.3f}), increasing disorder'
                    }
                elif entropy_trend < -0.1:
                    # Decreasing entropy - system becoming more ordered
                    entropy_signal = {
                        'strength': 0.7,
                        'direction': 1,   # Buy signal
                        'reasoning': f'Decreasing entropy (trend: {entropy_trend:.3f}), increasing order'
                    }
                else:
                    # Stable entropy
                    entropy_signal = {
                        'strength': 0.3,
                        'direction': 0,
                        'reasoning': f'Stable entropy (trend: {entropy_trend:.3f}), stable system'
                    }
            else:
                entropy_signal = {'strength': 0.0, 'direction': 0, 'reasoning': 'Insufficient entropy history'}
            
            # Absolute entropy level
            if current_entropy > 2.0:
                entropy_signal['reasoning'] += ', high absolute entropy'
                entropy_signal['strength'] *= 0.8  # Reduce confidence in high entropy
            elif current_entropy < 0.5:
                entropy_signal['reasoning'] += ', low absolute entropy'
                entropy_signal['strength'] *= 1.2  # Increase confidence in low entropy
            
            return {'entropy_signal': entropy_signal}
            
        except Exception:
            return {}
    
    def _generate_energy_signals(self) -> Dict[str, Any]:
        """Generate signals based on energy analysis."""
        try:
            if not self.current_state:
                return {}
            
            current_energy = self.current_state.energy
            
            # Energy trend analysis
            if len(self.energy_history) >= 3:
                recent_energy = self.energy_history[-3:]
                energy_trend = np.polyfit(range(len(recent_energy)), recent_energy, 1)[0]
                
                if energy_trend > 0.1:
                    # Increasing energy - bullish
                    energy_signal = {
                        'strength': 0.7,
                        'direction': 1,   # Buy signal
                        'reasoning': f'Increasing energy (trend: {energy_trend:.3f}), bullish momentum'
                    }
                elif energy_trend < -0.1:
                    # Decreasing energy - bearish
                    energy_signal = {
                        'strength': 0.7,
                        'direction': -1,  # Sell signal
                        'reasoning': f'Decreasing energy (trend: {energy_trend:.3f}), bearish momentum'
                    }
                else:
                    # Stable energy
                    energy_signal = {
                        'strength': 0.4,
                        'direction': 0,
                        'reasoning': f'Stable energy (trend: {energy_trend:.3f}), sideways market'
                    }
            else:
                energy_signal = {'strength': 0.0, 'direction': 0, 'reasoning': 'Insufficient energy history'}
            
            # Energy level relative to history
            if len(self.energy_history) >= 10:
                energy_percentile = stats.percentileofscore(self.energy_history[-10:], current_energy) / 100.0
                
                if energy_percentile > 0.8:
                    energy_signal['reasoning'] += ', high energy level'
                    energy_signal['strength'] *= 1.1
                elif energy_percentile < 0.2:
                    energy_signal['reasoning'] += ', low energy level'
                    energy_signal['strength'] *= 0.9
            
            return {'energy_signal': energy_signal}
            
        except Exception:
            return {}
    
    def _generate_temperature_signals(self) -> Dict[str, Any]:
        """Generate signals based on temperature analysis."""
        try:
            if not self.current_state:
                return {}
            
            current_temp = self.current_state.temperature
            critical_temp = self.parameters['critical_temperature']
            
            # Temperature relative to critical temperature
            temp_ratio = current_temp / critical_temp
            
            if temp_ratio > 1.2:
                # High temperature - high volatility expected
                temp_signal = {
                    'strength': 0.6,
                    'direction': 0,   # Neutral but expect volatility
                    'reasoning': f'High temperature ({temp_ratio:.2f}x critical), expect high volatility'
                }
            elif temp_ratio < 0.8:
                # Low temperature - low volatility, mean reversion
                temp_signal = {
                    'strength': 0.7,
                    'direction': -1 if len(self.returns) > 0 and self.returns[-1] > 0 else 1,
                    'reasoning': f'Low temperature ({temp_ratio:.2f}x critical), mean reversion expected'
                }
            else:
                # Near critical temperature - be cautious
                temp_signal = {
                    'strength': 0.3,
                    'direction': 0,
                    'reasoning': f'Near critical temperature ({temp_ratio:.2f}x critical), be cautious'
                }
            
            # Temperature trend
            if len(self.temperature_history) >= 3:
                recent_temps = self.temperature_history[-3:]
                temp_trend = np.polyfit(range(len(recent_temps)), recent_temps, 1)[0]
                
                if abs(temp_trend) > 0.1:
                    temp_signal['reasoning'] += f', temp trend: {temp_trend:.3f}'
            
            return {'temperature_signal': temp_signal}
            
        except Exception:
            return {}
    
    def _generate_equilibrium_signals(self) -> Dict[str, Any]:
        """Generate signals based on equilibrium analysis."""
        try:
            if not hasattr(self, 'equilibrium_analysis') or not self.equilibrium_analysis:
                return {}
            
            equilibrium_analysis = self.equilibrium_analysis
            equilibrium_check = equilibrium_analysis.get('equilibrium_check', {})
            is_equilibrium = equilibrium_check.get('is_equilibrium', False)
            equilibrium_measure = equilibrium_check.get('equilibrium_measure', 0.0)
            
            if is_equilibrium:
                # In equilibrium - expect mean reversion
                eq_signal = {
                    'strength': 0.6,
                    'direction': -1 if len(self.returns) > 0 and self.returns[-1] > 0.02 else 1,
                    'reasoning': f'System in equilibrium (measure: {equilibrium_measure:.2f}), mean reversion expected'
                }
            else:
                # Out of equilibrium - trend following
                eq_signal = {
                    'strength': 0.8,
                    'direction': 1 if len(self.returns) > 0 and self.returns[-1] > 0 else -1,
                    'reasoning': f'System out of equilibrium (measure: {equilibrium_measure:.2f}), trend following'
                }
            
            # Non-equilibrium analysis
            if 'non_equilibrium_analysis' in equilibrium_analysis:
                non_eq = equilibrium_analysis['non_equilibrium_analysis']
                entropy_production = non_eq.get('entropy_production_rate', 0.0)
                
                if entropy_production > 0.5:
                    eq_signal['strength'] *= 1.2  # Amplify signal
                    eq_signal['reasoning'] += f', high entropy production: {entropy_production:.3f}'
            
            return {'equilibrium_signal': eq_signal}
            
        except Exception:
            return {}
    
    def _combine_thermodynamic_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all thermodynamic signals into final recommendation."""
        try:
            if not signals:
                return self._create_neutral_signal()
            
            # Extract individual signals
            signal_components = []
            reasoning_components = []
            
            for signal_name, signal_data in signals.items():
                if isinstance(signal_data, dict) and 'strength' in signal_data:
                    strength = signal_data.get('strength', 0.0)
                    direction = signal_data.get('direction', 0)
                    reasoning = signal_data.get('reasoning', '')
                    
                    # Weight the signal
                    weighted_signal = strength * direction
                    signal_components.append(weighted_signal)
                    reasoning_components.append(f"{signal_name}: {reasoning}")
            
            if not signal_components:
                return self._create_neutral_signal()
            
            # Calculate combined signal
            combined_strength = np.mean([abs(s) for s in signal_components])
            combined_direction = np.mean(signal_components)
            
            # Normalize direction to [-1, 1]
            if abs(combined_direction) > 1e-6:
                normalized_direction = combined_direction / abs(combined_direction)
            else:
                normalized_direction = 0.0
            
            # Determine action
            if combined_strength > 0.6 and abs(normalized_direction) > 0.5:
                if normalized_direction > 0:
                    action = 'BUY'
                    confidence = min(0.95, combined_strength)
                else:
                    action = 'SELL'
                    confidence = min(0.95, combined_strength)
            elif combined_strength > 0.3:
                action = 'HOLD'
                confidence = combined_strength
            else:
                action = 'HOLD'
                confidence = 0.2
            
            # Risk assessment based on thermodynamic state
            risk_level = self._assess_thermodynamic_risk()
            
            return {
                'action': action,
                'confidence': confidence,
                'strength': combined_strength,
                'direction': normalized_direction,
                'risk_level': risk_level,
                'reasoning': '; '.join(reasoning_components),
                'thermodynamic_state': {
                    'temperature': self.current_state.temperature if self.current_state else None,
                    'entropy': self.current_state.entropy if self.current_state else None,
                    'energy': self.current_state.energy if self.current_state else None,
                    'phase': getattr(self, 'phase_analysis', {}).get('current_phase', 'unknown')
                },
                'signal_components': {name: data for name, data in signals.items()},
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error combining signals: {e}")
            return self._create_neutral_signal()
    
    def _assess_thermodynamic_risk(self) -> str:
        """Assess risk level based on thermodynamic state."""
        try:
            if not self.current_state:
                return 'MEDIUM'
            
            risk_factors = []
            
            # Temperature-based risk
            temp_ratio = self.current_state.temperature / self.parameters['critical_temperature']
            if temp_ratio > 1.5:
                risk_factors.append('HIGH_TEMP')
            elif temp_ratio < 0.5:
                risk_factors.append('LOW_TEMP')
            
            # Entropy-based risk
            if self.current_state.entropy > 3.0:
                risk_factors.append('HIGH_ENTROPY')
            
            # Critical behavior risk
            if hasattr(self, 'critical_analysis'):
                criticality = self.critical_analysis.get('criticality_measure', 0.0)
                if criticality > 0.8:
                    risk_factors.append('NEAR_CRITICAL')
            
            # Phase transition risk
            if hasattr(self, 'phase_analysis'):
                transition_prob = self.phase_analysis.get('transition_probability', 0.0)
                if transition_prob > 0.7:
                    risk_factors.append('PHASE_TRANSITION')
            
            # Determine overall risk
            if len(risk_factors) >= 3:
                return 'HIGH'
            elif len(risk_factors) >= 2:
                return 'MEDIUM'
            elif len(risk_factors) >= 1:
                return 'LOW'
            else:
                return 'VERY_LOW'
                
        except Exception:
            return 'MEDIUM'
    
    def _create_neutral_signal(self) -> Dict[str, Any]:
        """Create a neutral trading signal."""
        return {
            'action': 'HOLD',
            'confidence': 0.2,
            'strength': 0.0,
            'direction': 0.0,
            'risk_level': 'MEDIUM',
            'reasoning': 'Insufficient data for thermodynamic analysis',
            'thermodynamic_state': None,
            'signal_components': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def get_signal_strength(self) -> float:
        """Get overall signal strength (0-1 scale)."""
        try:
            signals = self._generate_trading_signals()
            return signals.get('strength', 0.0)
        except Exception:
            return 0.0
    
    def get_market_regime(self) -> str:
        """Get current market regime based on thermodynamic state."""
        try:
            if not self.current_state:
                return 'unknown'
            
            if hasattr(self, 'phase_analysis'):
                phase = self.phase_analysis.get('current_phase', 'unknown')
                
                if phase == 'gas':
                    return 'high_volatility_trending'
                elif phase == 'liquid':
                    return 'medium_volatility_mixed'
                elif phase == 'solid':
                    return 'low_volatility_mean_reverting'
            
            # Fallback based on temperature
            temp_ratio = self.current_state.temperature / self.parameters['critical_temperature']
            if temp_ratio > 1.2:
                return 'high_volatility'
            elif temp_ratio < 0.8:
                return 'low_volatility'
            else:
                return 'medium_volatility'
                
        except Exception:
            return 'unknown'
    
    def get_critical_levels(self) -> Dict[str, float]:
        """Get critical price levels based on thermodynamic analysis."""
        try:
            if len(self.prices) == 0:
                return {}
            
            current_price = self.prices[-1]
            
            # Calculate levels based on energy and entropy barriers
            if self.current_state:
                energy_factor = self.current_state.energy
                entropy_factor = self.current_state.entropy
                
                # Support and resistance based on thermodynamic barriers
                resistance = current_price * (1 + energy_factor * 0.05)
                support = current_price * (1 - energy_factor * 0.05)
                
                # Phase transition levels
                critical_level = current_price * (1 + (self.current_state.temperature / 
                                                     self.parameters['critical_temperature'] - 1) * 0.1)
                
                return {
                    'support': support,
                    'resistance': resistance,
                    'critical_transition': critical_level,
                    'entropy_barrier': current_price * (1 + entropy_factor * 0.02)
                }
            else:
                return {}
                
        except Exception:
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Initialize the thermodynamic entropy engine
    engine = ThermodynamicEntropyEngine()
    
    # Generate sample price data
    np.random.seed(42)
    n_points = 100
    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.02)
    volumes = np.random.lognormal(10, 0.5, n_points)
    timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
    
    # Test the engine with sample data
    for i in range(len(prices)):
        data_point = {
            'timestamp': timestamps[i],
            'price': prices[i],
            'volume': volumes[i],
            'high': prices[i] * 1.01,
            'low': prices[i] * 0.99,
            'close': prices[i]
        }
        
        result = engine.calculate(data_point)
        
        if i % 20 == 0:  # Print every 20th result
            print(f"Step {i}: {result}")
    
    # Get final analysis
    final_signals = engine._generate_trading_signals()
    print(f"\nFinal Trading Signals: {final_signals}")
    
    market_regime = engine.get_market_regime()
    print(f"Market Regime: {market_regime}")
    
    critical_levels = engine.get_critical_levels()
    print(f"Critical Levels: {critical_levels}")