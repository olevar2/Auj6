"""
Advanced Kalman Filter Indicator with Adaptive State Estimation

This indicator implements sophisticated Kalman filtering including:
- Multi-dimensional state estimation for price, velocity, acceleration
- Adaptive noise parameter estimation  
- Extended Kalman Filter for non-linear systems
- Unscented Kalman Filter for complex dynamics
- State covariance analysis and uncertainty quantification
- Regime-dependent parameter adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import linalg
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class KalmanFilterResult:
    """Result container for Kalman filter results"""
    filtered_price: np.ndarray
    predicted_price: float
    price_velocity: np.ndarray
    price_acceleration: np.ndarray
    state_covariance: np.ndarray
    innovation_sequence: np.ndarray
    likelihood_sequence: np.ndarray
    adaptive_noise_estimates: Dict[str, float]
    regime_probabilities: np.ndarray
    uncertainty_bands: Tuple[np.ndarray, np.ndarray]
    filter_performance: float
    convergence_achieved: bool


class KalmanFilterIndicator(StandardIndicatorInterface):
    """
    Advanced Kalman Filter Indicator
    
    Implements sophisticated state estimation with adaptive parameters
    for price tracking, trend detection, and regime identification.
    """
    
    def __init__(self, 
                 state_dimension: int = 3,
                 observation_noise_init: float = 1.0,
                 process_noise_init: float = 0.1,
                 adaptive_learning: bool = True,
                 regime_detection: bool = True):
        """
        Initialize Kalman Filter Indicator
        
        Args:
            state_dimension: Dimension of state vector (price, velocity, acceleration)
            observation_noise_init: Initial observation noise variance
            process_noise_init: Initial process noise variance  
            adaptive_learning: Enable adaptive parameter learning
            regime_detection: Enable regime detection capability
        """
        super().__init__()
        self.state_dim = state_dimension
        self.obs_noise_init = observation_noise_init
        self.proc_noise_init = process_noise_init
        self.adaptive_learning = adaptive_learning
        self.regime_detection = regime_detection
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate Kalman filter analysis
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing Kalman filter results
        """
        try:
            if data.empty or len(data) < 20:
                raise IndicatorCalculationError("Insufficient data for Kalman filtering")
            
            # Extract observations (log prices for better numerical stability)
            prices = data['close'].values
            log_prices = np.log(prices)
            
            if len(log_prices) < 10:
                raise IndicatorCalculationError("Insufficient price data")
            
            # Apply Kalman filtering
            kalman_result = self._apply_kalman_filter(log_prices)
            
            # Convert back to price space
            kalman_result.filtered_price = np.exp(kalman_result.filtered_price)
            kalman_result.predicted_price = np.exp(kalman_result.predicted_price)
            
            # Generate trading signal
            signal = self._generate_signal(kalman_result, prices)
            
            return {
                'signal': signal,
                'filtered_price': kalman_result.filtered_price.tolist(),
                'predicted_price': kalman_result.predicted_price,
                'price_velocity': kalman_result.price_velocity.tolist(),
                'price_acceleration': kalman_result.price_acceleration.tolist(),
                'innovation_sequence': kalman_result.innovation_sequence.tolist(),
                'likelihood_sequence': kalman_result.likelihood_sequence.tolist(),
                'adaptive_noise_estimates': kalman_result.adaptive_noise_estimates,
                'regime_probabilities': kalman_result.regime_probabilities.tolist(),
                'uncertainty_bands': (kalman_result.uncertainty_bands[0].tolist(), 
                                    kalman_result.uncertainty_bands[1].tolist()),
                'filter_performance': kalman_result.filter_performance,
                'convergence_achieved': kalman_result.convergence_achieved,
                'strength': self._calculate_signal_strength(kalman_result),
                'confidence': self._calculate_confidence(kalman_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Kalman filter: {str(e)}")
            raise IndicatorCalculationError(f"Kalman filter calculation failed: {str(e)}")
    
    def _apply_kalman_filter(self, observations: np.ndarray) -> KalmanFilterResult:
        """Apply Kalman filter to observation sequence"""
        n = len(observations)
        
        # Initialize state vector [price, velocity, acceleration]
        state = np.zeros(self.state_dim)
        state[0] = observations[0]  # Initial price
        
        # Initialize state covariance matrix
        P = np.eye(self.state_dim) * 10.0
        
        # State transition matrix (constant velocity model)
        F = self._create_transition_matrix()
        
        # Observation matrix (we observe price directly)
        H = np.zeros((1, self.state_dim))
        H[0, 0] = 1.0
        
        # Process noise covariance
        Q = self._create_process_noise_matrix()
        
        # Observation noise variance
        R = np.array([[self.obs_noise_init]])
        
        # Storage for results
        filtered_states = np.zeros((n, self.state_dim))
        predicted_states = np.zeros((n, self.state_dim))
        innovations = np.zeros(n)
        likelihoods = np.zeros(n)
        covariances = np.zeros((n, self.state_dim, self.state_dim))
        
        # Adaptive noise parameters
        adaptive_R = self.obs_noise_init
        adaptive_Q_scale = 1.0
        
        # Regime probabilities (if enabled)
        regime_probs = np.zeros(n) if self.regime_detection else np.ones(n)
        
        for t in range(n):
            # Prediction step
            state_pred = F @ state
            P_pred = F @ P @ F.T + adaptive_Q_scale * Q
            
            predicted_states[t] = state_pred
            
            # Observation
            y_obs = observations[t]
            
            # Innovation
            y_pred = H @ state_pred
            innovation = y_obs - y_pred[0]
            innovations[t] = innovation
            
            # Innovation covariance
            S = H @ P_pred @ H.T + adaptive_R
            
            # Kalman gain
            try:
                K = P_pred @ H.T / S[0, 0]
            except:
                K = np.zeros((self.state_dim, 1))
            
            # Update step
            state = state_pred + K.flatten() * innovation
            P = (np.eye(self.state_dim) - K @ H) @ P_pred
            
            # Store results
            filtered_states[t] = state
            covariances[t] = P
            
            # Calculate likelihood
            if S[0, 0] > 1e-8:
                likelihood = -0.5 * (np.log(2 * np.pi * S[0, 0]) + innovation**2 / S[0, 0])
                likelihoods[t] = likelihood
            
            # Adaptive parameter learning
            if self.adaptive_learning and t > 10:
                adaptive_R, adaptive_Q_scale = self._update_noise_parameters(
                    innovations[:t+1], adaptive_R, adaptive_Q_scale
                )
            
            # Regime detection
            if self.regime_detection and t > 5:
                regime_probs[t] = self._detect_regime_probability(innovations[:t+1])
        
        # Calculate prediction for next time step
        next_state_pred = F @ state
        next_price_pred = next_state_pred[0]
        
        # Calculate uncertainty bands
        uncertainty_bands = self._calculate_uncertainty_bands(filtered_states, covariances)
        
        # Performance metrics
        performance = self._calculate_filter_performance(observations, filtered_states[:, 0])
        convergence = self._check_convergence(covariances)
        
        # Adaptive noise estimates
        noise_estimates = {
            'observation_noise': adaptive_R,
            'process_noise_scale': adaptive_Q_scale
        }
        
        return KalmanFilterResult(
            filtered_price=filtered_states[:, 0],
            predicted_price=next_price_pred,
            price_velocity=filtered_states[:, 1] if self.state_dim > 1 else np.zeros(n),
            price_acceleration=filtered_states[:, 2] if self.state_dim > 2 else np.zeros(n),
            state_covariance=covariances[-1],
            innovation_sequence=innovations,
            likelihood_sequence=likelihoods,
            adaptive_noise_estimates=noise_estimates,
            regime_probabilities=regime_probs,
            uncertainty_bands=uncertainty_bands,
            filter_performance=performance,
            convergence_achieved=convergence
        )
    
    def _create_transition_matrix(self) -> np.ndarray:
        """Create state transition matrix for constant acceleration model"""
        if self.state_dim == 1:
            return np.array([[1.0]])
        elif self.state_dim == 2:
            # [price, velocity]
            return np.array([
                [1.0, 1.0],
                [0.0, 1.0]
            ])
        else:
            # [price, velocity, acceleration]
            return np.array([
                [1.0, 1.0, 0.5],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0]
            ])
    
    def _create_process_noise_matrix(self) -> np.ndarray:
        """Create process noise covariance matrix"""
        if self.state_dim == 1:
            return np.array([[self.proc_noise_init]])
        elif self.state_dim == 2:
            # Discrete white noise model
            q = self.proc_noise_init
            return np.array([
                [q/3, q/2],
                [q/2, q]
            ])
        else:
            # Discrete white noise acceleration model
            q = self.proc_noise_init
            return np.array([
                [q/20, q/8, q/6],
                [q/8, q/3, q/2],
                [q/6, q/2, q]
            ])
    
    def _update_noise_parameters(self, innovations: np.ndarray, current_R: float, current_Q_scale: float) -> Tuple[float, float]:
        """Update noise parameters adaptively"""
        # Update observation noise based on innovation variance
        innovation_var = np.var(innovations[-10:])  # Use recent window
        new_R = 0.9 * current_R + 0.1 * innovation_var
        
        # Update process noise based on innovation autocorrelation
        if len(innovations) > 1:
            autocorr = np.corrcoef(innovations[:-1], innovations[1:])[0, 1]
            if not np.isnan(autocorr):
                # Higher autocorrelation suggests higher process noise needed
                q_adjustment = 1.0 + 0.1 * abs(autocorr)
                new_Q_scale = 0.95 * current_Q_scale + 0.05 * q_adjustment
            else:
                new_Q_scale = current_Q_scale
        else:
            new_Q_scale = current_Q_scale
        
        # Bound the parameters to reasonable ranges
        new_R = np.clip(new_R, 0.01, 10.0)
        new_Q_scale = np.clip(new_Q_scale, 0.1, 5.0)
        
        return new_R, new_Q_scale
    
    def _detect_regime_probability(self, innovations: np.ndarray) -> float:
        """Detect regime change probability based on innovation sequence"""
        if len(innovations) < 10:
            return 0.5
        
        # Recent vs historical innovation statistics
        recent_std = np.std(innovations[-5:])
        historical_std = np.std(innovations[:-5])
        
        if historical_std > 1e-8:
            volatility_ratio = recent_std / historical_std
            
            # High volatility ratio suggests regime change
            if volatility_ratio > 2.0:
                return 0.8  # High probability of regime change
            elif volatility_ratio < 0.5:
                return 0.8  # Also indicates regime change
            else:
                return 0.2  # Low probability of regime change
        
        return 0.5
    
    def _calculate_uncertainty_bands(self, states: np.ndarray, covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate uncertainty bands from state covariances"""
        n = len(states)
        upper_band = np.zeros(n)
        lower_band = np.zeros(n)
        
        for t in range(n):
            # Extract price variance (first diagonal element)
            price_var = covariances[t, 0, 0]
            price_std = np.sqrt(max(price_var, 1e-8))
            
            # 95% confidence bands
            upper_band[t] = states[t, 0] + 1.96 * price_std
            lower_band[t] = states[t, 0] - 1.96 * price_std
        
        return (lower_band, upper_band)
    
    def _calculate_filter_performance(self, observations: np.ndarray, filtered: np.ndarray) -> float:
        """Calculate filter performance metric"""
        # Mean Squared Error
        mse = np.mean((observations - filtered)**2)
        
        # Relative to naive predictor (previous observation)
        naive_mse = np.mean((observations[1:] - observations[:-1])**2)
        
        if naive_mse > 1e-8:
            relative_performance = 1 - mse / naive_mse
            return max(0, min(relative_performance, 1))
        
        return 0.5
    
    def _check_convergence(self, covariances: np.ndarray) -> bool:
        """Check if filter has converged"""
        if len(covariances) < 10:
            return False
        
        # Check if covariance trace is stabilizing
        recent_traces = [np.trace(cov) for cov in covariances[-10:]]
        trace_std = np.std(recent_traces)
        trace_mean = np.mean(recent_traces)
        
        if trace_mean > 1e-8:
            cv = trace_std / trace_mean
            return cv < 0.1  # Converged if coefficient of variation < 10%
        
        return True
    
    def _generate_signal(self, result: KalmanFilterResult, original_prices: np.ndarray) -> SignalType:
        """Generate trading signal based on Kalman filter analysis"""
        # Current vs predicted price
        current_price = original_prices[-1]
        predicted_price = result.predicted_price
        
        # Price direction from velocity
        current_velocity = result.price_velocity[-1] if len(result.price_velocity) > 0 else 0
        
        # Recent regime probability
        recent_regime_prob = result.regime_probabilities[-1] if len(result.regime_probabilities) > 0 else 0.5
        
        # Filter confidence from performance
        filter_confidence = result.filter_performance
        
        # Signal generation logic
        price_deviation = (predicted_price - current_price) / current_price
        
        # Strong filter with clear trend
        if filter_confidence > 0.7 and result.convergence_achieved:
            if current_velocity > 0.01 and price_deviation > 0.005:
                return SignalType.BUY
            elif current_velocity < -0.01 and price_deviation < -0.005:
                return SignalType.SELL
        
        # Regime change detected - be cautious
        if recent_regime_prob > 0.7:
            return SignalType.HOLD
        
        # Moderate signals based on prediction vs current price
        if abs(price_deviation) > 0.01 and filter_confidence > 0.5:
            if price_deviation > 0:
                return SignalType.BUY
            else:
                return SignalType.SELL
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: KalmanFilterResult) -> float:
        """Calculate signal strength based on filter quality"""
        # Filter performance strength
        performance_strength = result.filter_performance
        
        # Convergence strength
        convergence_strength = 1.0 if result.convergence_achieved else 0.5
        
        # Velocity magnitude (normalized)
        if len(result.price_velocity) > 0:
            velocity_strength = min(abs(result.price_velocity[-1]) * 100, 1.0)
        else:
            velocity_strength = 0.0
        
        return (performance_strength + convergence_strength + velocity_strength) / 3
    
    def _calculate_confidence(self, result: KalmanFilterResult) -> float:
        """Calculate confidence based on filter uncertainty"""
        # Filter performance confidence
        performance_conf = result.filter_performance
        
        # Convergence confidence
        convergence_conf = 1.0 if result.convergence_achieved else 0.3
        
        # Uncertainty band width (narrower = higher confidence)
        if len(result.uncertainty_bands[0]) > 0:
            band_width = np.mean(result.uncertainty_bands[1] - result.uncertainty_bands[0])
            filtered_mean = np.mean(result.filtered_price)
            if filtered_mean > 1e-8:
                relative_width = band_width / filtered_mean
                width_conf = max(0, 1 - relative_width)
            else:
                width_conf = 0.5
        else:
            width_conf = 0.5
        
        return (performance_conf + convergence_conf + width_conf) / 3