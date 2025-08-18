"""
Advanced Market Regime Detection Indicator with Hidden Markov Models

This indicator implements sophisticated regime detection including:
- Hidden Markov Models (HMM) with multiple states
- Regime switching models with volatility clustering
- State transition probability analysis
- Regime persistence and stability metrics
- Economic regime classification (bull, bear, sideways)
- Volatility regime detection (low, medium, high)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from scipy.special import logsumexp
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class MarketRegimeResult:
    """Result container for market regime detection results"""
    current_regime: int
    regime_probabilities: np.ndarray
    regime_labels: List[str]
    transition_matrix: np.ndarray
    regime_persistence: np.ndarray
    volatility_regimes: np.ndarray
    trend_regimes: np.ndarray
    regime_change_probability: float
    expected_regime_duration: float
    regime_stability_score: float
    model_likelihood: float


class MarketRegimeDetectionIndicator(StandardIndicatorInterface):
    """
    Advanced Market Regime Detection Indicator
    
    Implements HMM-based regime detection with comprehensive
    market state analysis and transition modeling.
    """
    
    def __init__(self, 
                 n_regimes: int = 3,
                 lookback_period: int = 252,
                 regime_types: List[str] = ['bear', 'sideways', 'bull']):
        """
        Initialize Market Regime Detection Indicator
        
        Args:
            n_regimes: Number of market regimes to detect
            lookback_period: Historical data window for analysis
            regime_types: Labels for different regimes
        """
        super().__init__()
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.regime_types = regime_types[:n_regimes] if len(regime_types) >= n_regimes else \
                           [f'regime_{i}' for i in range(n_regimes)]
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate market regime detection analysis
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing regime detection results
        """
        try:
            if data.empty or len(data) < 60:
                raise IndicatorCalculationError("Insufficient data for regime detection")
            
            # Extract returns and volatility measures
            prices = data['close'].values
            returns = np.diff(np.log(prices)) * 100  # Percentage returns
            
            if len(returns) < 50:
                raise IndicatorCalculationError("Insufficient return data")
            
            # Perform regime detection
            regime_result = self._detect_market_regimes(returns)
            
            # Generate trading signal
            signal = self._generate_signal(regime_result, returns)
            
            return {
                'signal': signal,
                'current_regime': regime_result.current_regime,
                'regime_probabilities': regime_result.regime_probabilities.tolist(),
                'regime_labels': regime_result.regime_labels,
                'transition_matrix': regime_result.transition_matrix.tolist(),
                'regime_persistence': regime_result.regime_persistence.tolist(),
                'volatility_regimes': regime_result.volatility_regimes.tolist(),
                'trend_regimes': regime_result.trend_regimes.tolist(),
                'regime_change_probability': regime_result.regime_change_probability,
                'expected_regime_duration': regime_result.expected_regime_duration,
                'regime_stability_score': regime_result.regime_stability_score,
                'model_likelihood': regime_result.model_likelihood,
                'strength': self._calculate_signal_strength(regime_result),
                'confidence': self._calculate_confidence(regime_result)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating market regime detection: {str(e)}")
            raise IndicatorCalculationError(f"Market regime detection failed: {str(e)}")
    
    def _detect_market_regimes(self, returns: np.ndarray) -> MarketRegimeResult:
        """Detect market regimes using Hidden Markov Model"""
        
        # Prepare observations (returns and volatility)
        observations = self._prepare_observations(returns)
        
        # Fit HMM model
        hmm_result = self._fit_hmm_model(observations)
        
        # Extract regime information
        regime_probs, transition_matrix, emission_params = hmm_result
        
        # Current regime (most likely state)
        current_regime = np.argmax(regime_probs[-1])
        
        # Calculate regime characteristics
        persistence = self._calculate_regime_persistence(transition_matrix)
        volatility_regimes = self._classify_volatility_regimes(returns, regime_probs)
        trend_regimes = self._classify_trend_regimes(returns, regime_probs)
        
        # Regime change probability
        change_prob = 1 - transition_matrix[current_regime, current_regime]
        
        # Expected duration in current regime
        expected_duration = 1 / change_prob if change_prob > 0 else np.inf
        
        # Regime stability score
        stability_score = self._calculate_stability_score(regime_probs, transition_matrix)
        
        # Model likelihood
        model_likelihood = self._calculate_model_likelihood(observations, regime_probs, emission_params)
        
        return MarketRegimeResult(
            current_regime=current_regime,
            regime_probabilities=regime_probs,
            regime_labels=self.regime_types,
            transition_matrix=transition_matrix,
            regime_persistence=persistence,
            volatility_regimes=volatility_regimes,
            trend_regimes=trend_regimes,
            regime_change_probability=change_prob,
            expected_regime_duration=expected_duration,
            regime_stability_score=stability_score,
            model_likelihood=model_likelihood
        )
    
    def _prepare_observations(self, returns: np.ndarray) -> np.ndarray:
        """Prepare observation matrix for HMM"""
        n = len(returns)
        
        # Feature 1: Returns
        feature_returns = returns
        
        # Feature 2: Absolute returns (volatility proxy)
        feature_volatility = np.abs(returns)
        
        # Feature 3: Rolling volatility
        rolling_vol = np.zeros(n)
        window = 10
        for i in range(n):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            rolling_vol[i] = np.std(returns[start_idx:end_idx])
        
        # Standardize features
        features = np.column_stack([feature_returns, feature_volatility, rolling_vol])
        
        # Standardize each feature
        for j in range(features.shape[1]):
            feature_col = features[:, j]
            mean_feat = np.mean(feature_col)
            std_feat = np.std(feature_col)
            if std_feat > 1e-8:
                features[:, j] = (feature_col - mean_feat) / std_feat
        
        return features
    
    def _fit_hmm_model(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Fit Hidden Markov Model using simplified EM algorithm"""
        n_obs, n_features = observations.shape
        n_states = self.n_regimes
        
        # Initialize parameters
        transition_matrix = np.random.rand(n_states, n_states)
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        
        # Initial state probabilities
        initial_probs = np.ones(n_states) / n_states
        
        # Emission parameters (Gaussian for each state)
        means = np.random.randn(n_states, n_features)
        covariances = np.array([np.eye(n_features) for _ in range(n_states)])
        
        # EM algorithm (simplified)
        max_iterations = 50
        tolerance = 1e-4
        
        for iteration in range(max_iterations):
            # E-step: Forward-backward algorithm
            state_probs = self._forward_backward(observations, transition_matrix, 
                                               initial_probs, means, covariances)
            
            # M-step: Update parameters
            new_transition_matrix = self._update_transition_matrix(state_probs)
            new_means, new_covariances = self._update_emission_parameters(observations, state_probs)
            
            # Check convergence
            trans_diff = np.mean(np.abs(new_transition_matrix - transition_matrix))
            if trans_diff < tolerance:
                break
            
            transition_matrix = new_transition_matrix
            means = new_means
            covariances = new_covariances
        
        emission_params = {'means': means, 'covariances': covariances}
        
        return state_probs, transition_matrix, emission_params
    
    def _forward_backward(self, observations: np.ndarray, transition_matrix: np.ndarray,
                         initial_probs: np.ndarray, means: np.ndarray, 
                         covariances: np.ndarray) -> np.ndarray:
        """Simplified forward-backward algorithm"""
        n_obs, n_features = observations.shape
        n_states = len(initial_probs)
        
        # Emission probabilities
        emission_probs = np.zeros((n_obs, n_states))
        for t in range(n_obs):
            for s in range(n_states):
                try:
                    # Multivariate normal probability
                    diff = observations[t] - means[s]
                    cov_inv = np.linalg.inv(covariances[s] + np.eye(n_features) * 1e-6)
                    mahalanobis = diff @ cov_inv @ diff.T
                    emission_probs[t, s] = np.exp(-0.5 * mahalanobis)
                except:
                    emission_probs[t, s] = 1e-10
        
        # Normalize emission probabilities
        emission_probs = emission_probs / (emission_probs.sum(axis=1, keepdims=True) + 1e-10)
        
        # Forward pass
        alpha = np.zeros((n_obs, n_states))
        alpha[0] = initial_probs * emission_probs[0]
        alpha[0] = alpha[0] / (alpha[0].sum() + 1e-10)
        
        for t in range(1, n_obs):
            for s in range(n_states):
                alpha[t, s] = emission_probs[t, s] * np.sum(alpha[t-1] * transition_matrix[:, s])
            alpha[t] = alpha[t] / (alpha[t].sum() + 1e-10)
        
        # Backward pass (simplified - just use forward probabilities)
        return alpha
    
    def _update_transition_matrix(self, state_probs: np.ndarray) -> np.ndarray:
        """Update transition matrix"""
        n_obs, n_states = state_probs.shape
        new_trans = np.zeros((n_states, n_states))
        
        for i in range(n_states):
            for j in range(n_states):
                numerator = 0
                denominator = 0
                for t in range(n_obs - 1):
                    numerator += state_probs[t, i] * state_probs[t+1, j]
                    denominator += state_probs[t, i]
                
                if denominator > 1e-10:
                    new_trans[i, j] = numerator / denominator
        
        # Normalize rows
        row_sums = new_trans.sum(axis=1, keepdims=True)
        new_trans = new_trans / (row_sums + 1e-10)
        
        return new_trans
    
    def _update_emission_parameters(self, observations: np.ndarray, 
                                  state_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update emission parameters (means and covariances)"""
        n_obs, n_features = observations.shape
        n_states = state_probs.shape[1]
        
        new_means = np.zeros((n_states, n_features))
        new_covariances = np.zeros((n_states, n_features, n_features))
        
        for s in range(n_states):
            # Weighted mean
            weights = state_probs[:, s]
            weight_sum = weights.sum()
            
            if weight_sum > 1e-10:
                new_means[s] = np.sum(observations * weights.reshape(-1, 1), axis=0) / weight_sum
                
                # Weighted covariance
                diff = observations - new_means[s]
                weighted_diff = diff * weights.reshape(-1, 1)
                new_covariances[s] = (weighted_diff.T @ diff) / weight_sum
            else:
                new_means[s] = np.mean(observations, axis=0)
                new_covariances[s] = np.eye(n_features)
        
        return new_means, new_covariances
    
    def _calculate_regime_persistence(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Calculate persistence of each regime"""
        return np.diag(transition_matrix)
    
    def _classify_volatility_regimes(self, returns: np.ndarray, regime_probs: np.ndarray) -> np.ndarray:
        """Classify volatility regimes"""
        n = len(returns)
        vol_regimes = np.zeros(n)
        
        # Calculate rolling volatility for each regime
        window = 20
        for i in range(n):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            window_returns = returns[start_idx:end_idx]
            vol = np.std(window_returns)
            
            # Classify volatility level
            if vol < 1.0:
                vol_regimes[i] = 0  # Low volatility
            elif vol < 2.0:
                vol_regimes[i] = 1  # Medium volatility
            else:
                vol_regimes[i] = 2  # High volatility
        
        return vol_regimes
    
    def _classify_trend_regimes(self, returns: np.ndarray, regime_probs: np.ndarray) -> np.ndarray:
        """Classify trend regimes"""
        n = len(returns)
        trend_regimes = np.zeros(n)
        
        # Calculate rolling mean for trend classification
        window = 20
        for i in range(n):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            window_returns = returns[start_idx:end_idx]
            mean_return = np.mean(window_returns)
            
            # Classify trend direction
            if mean_return > 0.1:
                trend_regimes[i] = 2  # Bull trend
            elif mean_return < -0.1:
                trend_regimes[i] = 0  # Bear trend
            else:
                trend_regimes[i] = 1  # Sideways
        
        return trend_regimes
    
    def _calculate_stability_score(self, regime_probs: np.ndarray, transition_matrix: np.ndarray) -> float:
        """Calculate overall regime stability score"""
        # High diagonal values in transition matrix indicate stability
        stability_from_transitions = np.mean(np.diag(transition_matrix))
        
        # Low variance in regime probabilities indicates stability
        prob_variance = np.mean(np.var(regime_probs, axis=1))
        stability_from_probs = 1 / (1 + prob_variance)
        
        return (stability_from_transitions + stability_from_probs) / 2
    
    def _calculate_model_likelihood(self, observations: np.ndarray, regime_probs: np.ndarray,
                                   emission_params: Dict) -> float:
        """Calculate model likelihood"""
        # Simplified likelihood calculation
        n_obs = len(observations)
        log_likelihood = 0
        
        for t in range(n_obs):
            state_likelihood = np.sum(regime_probs[t])
            log_likelihood += np.log(max(state_likelihood, 1e-10))
        
        return log_likelihood / n_obs  # Average log-likelihood
    
    def _generate_signal(self, result: MarketRegimeResult, returns: np.ndarray) -> SignalType:
        """Generate trading signal based on regime detection"""
        current_regime = result.current_regime
        regime_labels = result.regime_labels
        change_prob = result.regime_change_probability
        
        # Recent return momentum
        recent_momentum = np.mean(returns[-10:]) if len(returns) >= 10 else 0
        
        # Signal based on current regime and stability
        if current_regime < len(regime_labels):
            regime_label = regime_labels[current_regime]
            
            # Bull regime with low change probability
            if 'bull' in regime_label.lower() and change_prob < 0.3:
                return SignalType.BUY
            
            # Bear regime with low change probability
            elif 'bear' in regime_label.lower() and change_prob < 0.3:
                return SignalType.SELL
            
            # Regime change likely - be cautious
            elif change_prob > 0.7:
                return SignalType.HOLD
            
            # Sideways regime - use momentum
            elif 'sideways' in regime_label.lower():
                if recent_momentum > 0.5:
                    return SignalType.BUY
                elif recent_momentum < -0.5:
                    return SignalType.SELL
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: MarketRegimeResult) -> float:
        """Calculate signal strength based on regime characteristics"""
        # Regime stability strength
        stability_strength = result.regime_stability_score
        
        # Current regime confidence
        current_regime_prob = result.regime_probabilities[-1, result.current_regime]
        confidence_strength = current_regime_prob
        
        # Regime persistence
        persistence_strength = result.regime_persistence[result.current_regime]
        
        return (stability_strength + confidence_strength + persistence_strength) / 3
    
    def _calculate_confidence(self, result: MarketRegimeResult) -> float:
        """Calculate confidence based on model quality"""
        # Model likelihood confidence
        likelihood_conf = min(abs(result.model_likelihood), 1.0)
        
        # Regime probability confidence
        prob_conf = result.regime_probabilities[-1, result.current_regime]
        
        # Stability confidence
        stability_conf = result.regime_stability_score
        
        return (likelihood_conf + prob_conf + stability_conf) / 3