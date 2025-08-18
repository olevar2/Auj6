"""
Attractor Point - Quantum Market Dynamics Analysis
================================================

Advanced quantum-inspired market attractor analysis using chaos theory and fractal geometry.
Identifies market equilibrium points and phase transitions using sophisticated mathematical models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from scipy import signal
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class AttractorPoint(StandardIndicatorInterface):
    """
    Quantum-inspired attractor point analysis for market dynamics.
    
    Features:
    - Strange attractor identification using phase space reconstruction
    - Lyapunov exponent calculation for chaos detection
    - Fractal dimension analysis of price trajectories
    - Market phase transition detection
    - Equilibrium point prediction using Poincaré maps
    - Quantum coherence measurement in price movements
    - Multi-dimensional attractor basin analysis
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'embedding_dimension': 3,    # Phase space embedding dimension
            'time_delay': 1,            # Time delay for embedding
            'min_trajectory_length': 100, # Minimum points for analysis
            'attractor_sensitivity': 0.1, # Sensitivity for attractor detection
            'chaos_threshold': 0.0,     # Positive Lyapunov for chaos
            'fractal_window': 50,       # Window for fractal analysis
            'coherence_window': 20,     # Quantum coherence measurement window
            'phase_clusters': 5,        # Number of phase space clusters
            'stability_threshold': 0.7,  # Stability threshold for attractors
            'prediction_horizon': 10,   # Periods to predict ahead
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("AttractorPoint", default_params)
        
        # Analysis components
        self.phase_space_history = []
        self.lyapunov_history = []
        self.attractor_points = []
        self.last_phase_state = None
        
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=self.parameters['min_trajectory_length'] + self.parameters['embedding_dimension'] * self.parameters['time_delay'],
            lookback_periods=500
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate quantum attractor point analysis.
        """
        try:
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            # Phase space reconstruction
            phase_space = self._reconstruct_phase_space(close)
            
            # Identify strange attractors
            attractors = self._identify_attractors(phase_space)
            
            # Calculate Lyapunov exponents
            lyapunov_exponent = self._calculate_lyapunov_exponent(close)
            
            # Fractal dimension analysis
            fractal_dimension = self._calculate_fractal_dimension(close)
            
            # Quantum coherence measurement
            coherence = self._measure_quantum_coherence(close)
            
            # Phase transition detection
            phase_transition = self._detect_phase_transition(phase_space)
            
            # Equilibrium point prediction
            equilibrium_prediction = self._predict_equilibrium_points(phase_space, close)
            
            # Attractor basin analysis
            basin_analysis = self._analyze_attractor_basins(phase_space)
            
            # Market dynamics classification
            dynamics_class = self._classify_market_dynamics(lyapunov_exponent, fractal_dimension, coherence)
            
            # Stability analysis
            stability_metrics = self._calculate_stability_metrics(attractors, phase_space)
            
            # Signal generation based on attractor dynamics
            signal_strength = self._calculate_attractor_signal_strength(
                attractors, equilibrium_prediction, phase_transition, stability_metrics
            )
            
            return {
                'attractors': attractors,
                'lyapunov_exponent': lyapunov_exponent,
                'fractal_dimension': fractal_dimension,
                'quantum_coherence': coherence,
                'phase_transition': phase_transition,
                'equilibrium_prediction': equilibrium_prediction,
                'basin_analysis': basin_analysis,
                'dynamics_classification': dynamics_class,
                'stability_metrics': stability_metrics,
                'signal_strength': signal_strength,
                'chaos_detected': lyapunov_exponent > self.parameters['chaos_threshold'],
                'phase_space_dimension': phase_space.shape[1] if len(phase_space) > 0 else 0,
                'current_phase_state': phase_space[-1].tolist() if len(phase_space) > 0 else []
            }
            
        except Exception as e:
            raise Exception(f"AttractorPoint calculation failed: {str(e)}")
    
    def _reconstruct_phase_space(self, time_series: np.ndarray) -> np.ndarray:
        """Reconstruct phase space using Takens' embedding theorem."""
        if len(time_series) < self.parameters['min_trajectory_length']:
            return np.array([])
        
        m = self.parameters['embedding_dimension']
        tau = self.parameters['time_delay']
        
        # Calculate embedding parameters
        N = len(time_series)
        M = N - (m - 1) * tau
        
        if M <= 0:
            return np.array([])
        
        # Reconstruct phase space
        phase_space = np.zeros((M, m))
        
        for i in range(m):
            phase_space[:, i] = time_series[i * tau:i * tau + M]
        
        return phase_space
    
    def _identify_attractors(self, phase_space: np.ndarray) -> Dict[str, Any]:
        """Identify strange attractors in phase space."""
        if len(phase_space) == 0:
            return {}
        
        try:
            # Use K-means clustering to identify attractor regions
            n_clusters = min(self.parameters['phase_clusters'], len(phase_space) // 10)
            if n_clusters < 2:
                return {}
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(phase_space)
            cluster_centers = kmeans.cluster_centers_
            
            # Analyze each cluster as potential attractor
            attractors = []
            for i, center in enumerate(cluster_centers):
                cluster_points = phase_space[cluster_labels == i]
                
                if len(cluster_points) > 5:
                    # Calculate attractor properties
                    attractor_radius = np.mean(np.linalg.norm(cluster_points - center, axis=1))
                    stability = self._calculate_local_stability(cluster_points, center)
                    density = len(cluster_points) / len(phase_space)
                    
                    attractors.append({
                        'center': center.tolist(),
                        'radius': float(attractor_radius),
                        'stability': float(stability),
                        'density': float(density),
                        'points_count': len(cluster_points)
                    })
            
            # Find dominant attractor
            dominant_attractor = None
            if attractors:
                # Sort by combination of stability and density
                attractors.sort(key=lambda x: x['stability'] * x['density'], reverse=True)
                dominant_attractor = attractors[0]
            
            return {
                'attractors': attractors,
                'dominant_attractor': dominant_attractor,
                'attractor_count': len(attractors),
                'phase_space_coverage': self._calculate_phase_space_coverage(phase_space, attractors)
            }
            
        except Exception:
            return {}
    
    def _calculate_local_stability(self, points: np.ndarray, center: np.ndarray) -> float:
        """Calculate local stability around an attractor point."""
        if len(points) < 3:
            return 0.0
        
        # Calculate variance in each dimension
        variances = np.var(points, axis=0)
        
        # Stability is inverse of average variance (normalized)
        avg_variance = np.mean(variances)
        stability = 1.0 / (1.0 + avg_variance)
        
        return stability
    
    def _calculate_lyapunov_exponent(self, time_series: np.ndarray) -> float:
        """Calculate largest Lyapunov exponent for chaos detection."""
        if len(time_series) < 50:
            return 0.0
        
        try:
            # Use Rosenstein's algorithm (simplified version)
            N = len(time_series)
            m = 3  # Embedding dimension
            tau = 1  # Time delay
            
            # Reconstruct trajectory
            M = N - (m - 1) * tau
            if M <= 10:
                return 0.0
            
            Y = np.zeros((M, m))
            for i in range(m):
                Y[:, i] = time_series[i * tau:i * tau + M]
            
            # Find nearest neighbors
            max_separation = min(10, M // 4)
            divergences = []
            
            for i in range(M - max_separation):
                # Calculate distances to all other points
                distances = np.linalg.norm(Y[i+1:] - Y[i], axis=1)
                
                # Find nearest neighbor (avoiding temporally close points)
                valid_indices = np.where(np.arange(len(distances)) > 10)[0]
                if len(valid_indices) == 0:
                    continue
                
                nearest_idx = valid_indices[np.argmin(distances[valid_indices])]
                j = i + 1 + nearest_idx
                
                # Track divergence over time
                for k in range(1, min(max_separation, M - j)):
                    if i + k < M and j + k < M:
                        divergence = np.linalg.norm(Y[i + k] - Y[j + k])
                        if divergence > 0:
                            divergences.append(np.log(divergence))
            
            if len(divergences) > 10:
                # Linear regression to estimate Lyapunov exponent
                x = np.arange(len(divergences))
                lyapunov = np.polyfit(x, divergences, 1)[0]
                return float(lyapunov)
            
        except Exception:
            pass
        
        return 0.0
    
    def _calculate_fractal_dimension(self, time_series: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method."""
        if len(time_series) < self.parameters['fractal_window']:
            return 1.0
        
        try:
            # Use recent data for fractal analysis
            data = time_series[-self.parameters['fractal_window']:]
            
            # Normalize data
            data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            
            # Box-counting algorithm
            scales = np.logspace(-2, 0, 20)  # Different box sizes
            counts = []
            
            for scale in scales:
                # Discretize into boxes
                box_size = scale
                if box_size <= 0:
                    continue
                
                # Count boxes containing data points
                boxes_x = np.floor(np.arange(len(data)) / (len(data) * box_size)).astype(int)
                boxes_y = np.floor(data / box_size).astype(int)
                
                # Count unique boxes
                unique_boxes = len(set(zip(boxes_x, boxes_y)))
                counts.append(unique_boxes)
            
            if len(counts) > 5:
                # Calculate fractal dimension from slope
                log_scales = np.log(1.0 / scales[:len(counts)])
                log_counts = np.log(counts)
                
                # Linear regression
                fractal_dim = np.polyfit(log_scales, log_counts, 1)[0]
                return float(np.clip(fractal_dim, 1.0, 3.0))
            
        except Exception:
            pass
        
        return 1.5  # Default fractal dimension
    
    def _measure_quantum_coherence(self, time_series: np.ndarray) -> float:
        """Measure quantum-like coherence in price movements."""
        if len(time_series) < self.parameters['coherence_window']:
            return 0.0
        
        try:
            # Use recent data
            data = time_series[-self.parameters['coherence_window']:]
            
            # Calculate price changes
            changes = np.diff(data) / data[:-1]
            
            # Quantum coherence inspired by wave function collapse
            # Measure phase coherence using Fourier analysis
            fft = np.fft.fft(changes)
            power_spectrum = np.abs(fft) ** 2
            
            # Coherence as concentration of power in specific frequencies
            total_power = np.sum(power_spectrum)
            if total_power == 0:
                return 0.0
            
            # Calculate spectral entropy (inverse of coherence)
            normalized_spectrum = power_spectrum / total_power
            spectral_entropy = -np.sum(normalized_spectrum * np.log(normalized_spectrum + 1e-8))
            
            # Coherence as normalized inverse entropy
            max_entropy = np.log(len(normalized_spectrum))
            coherence = 1.0 - (spectral_entropy / max_entropy)
            
            return float(np.clip(coherence, 0, 1))
            
        except Exception:
            return 0.0
    
    def _detect_phase_transition(self, phase_space: np.ndarray) -> Dict[str, Any]:
        """Detect phase transitions in market dynamics."""
        if len(phase_space) < 20:
            return {}
        
        try:
            # Calculate local variance in phase space
            window_size = 10
            local_variances = []
            
            for i in range(window_size, len(phase_space)):
                window = phase_space[i-window_size:i]
                variance = np.mean(np.var(window, axis=0))
                local_variances.append(variance)
            
            local_variances = np.array(local_variances)
            
            # Detect sudden changes in variance (phase transitions)
            variance_changes = np.abs(np.diff(local_variances))
            
            # Threshold for detecting transitions
            threshold = np.mean(variance_changes) + 2 * np.std(variance_changes)
            
            # Find recent transitions
            transition_indices = np.where(variance_changes > threshold)[0]
            
            recent_transition = None
            if len(transition_indices) > 0:
                recent_idx = transition_indices[-1]
                periods_ago = len(variance_changes) - recent_idx
                
                if periods_ago <= 5:  # Recent transition
                    transition_strength = variance_changes[recent_idx] / threshold
                    recent_transition = {
                        'periods_ago': int(periods_ago),
                        'strength': float(transition_strength),
                        'type': 'volatility_increase' if local_variances[recent_idx+1] > local_variances[recent_idx] else 'volatility_decrease'
                    }
            
            return {
                'recent_transition': recent_transition,
                'current_volatility': float(local_variances[-1]),
                'volatility_trend': float(np.polyfit(range(len(local_variances[-10:])), local_variances[-10:], 1)[0])
            }
            
        except Exception:
            return {}
    
    def _predict_equilibrium_points(self, phase_space: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        """Predict future equilibrium points using Poincaré maps."""
        if len(phase_space) < 30 or len(prices) < 30:
            return {}
        
        try:
            # Create Poincaré section (simplified 2D projection)
            if phase_space.shape[1] >= 2:
                section_points = phase_space[:, :2]
            else:
                return {}
            
            # Find crossings of a hyperplane (price level)
            current_price = prices[-1]
            crossings = []
            
            for i in range(1, len(section_points)):
                prev_point = section_points[i-1]
                curr_point = section_points[i]
                
                # Check if trajectory crosses current price level
                if (prev_point[0] <= current_price <= curr_point[0]) or (curr_point[0] <= current_price <= prev_point[0]):
                    crossing = {
                        'index': i,
                        'point': curr_point.tolist(),
                        'price': prices[i] if i < len(prices) else current_price
                    }
                    crossings.append(crossing)
            
            # Predict next equilibrium based on crossing patterns
            equilibrium_price = current_price
            confidence = 0.5
            
            if len(crossings) >= 3:
                # Analyze crossing pattern
                recent_crossings = crossings[-3:]
                price_levels = [c['price'] for c in recent_crossings]
                
                # Simple prediction: extrapolate trend
                if len(price_levels) >= 2:
                    trend = np.polyfit(range(len(price_levels)), price_levels, 1)[0]
                    equilibrium_price = price_levels[-1] + trend * self.parameters['prediction_horizon']
                    
                    # Confidence based on trend consistency
                    price_changes = np.diff(price_levels)
                    trend_consistency = 1.0 - np.std(price_changes) / (np.mean(np.abs(price_changes)) + 1e-8)
                    confidence = np.clip(trend_consistency, 0, 1)
            
            return {
                'predicted_equilibrium': float(equilibrium_price),
                'confidence': float(confidence),
                'crossings_count': len(crossings),
                'prediction_horizon': self.parameters['prediction_horizon']
            }
            
        except Exception:
            return {}
    
    def _analyze_attractor_basins(self, phase_space: np.ndarray) -> Dict[str, Any]:
        """Analyze attractor basins and their properties."""
        if len(phase_space) < 20:
            return {}
        
        try:
            # Calculate trajectory convergence properties
            # Measure how trajectories converge or diverge
            
            convergence_rates = []
            for i in range(10, len(phase_space)):
                # Look at recent trajectory
                recent_traj = phase_space[i-10:i]
                
                # Calculate convergence to center of mass
                center = np.mean(recent_traj, axis=0)
                distances = np.linalg.norm(recent_traj - center, axis=1)
                
                # Rate of convergence (negative slope means convergence)
                if len(distances) > 2:
                    rate = np.polyfit(range(len(distances)), distances, 1)[0]
                    convergence_rates.append(rate)
            
            avg_convergence = np.mean(convergence_rates) if convergence_rates else 0
            
            # Basin stability (how stable is current position)
            stability = 1.0 / (1.0 + abs(avg_convergence))
            
            # Basin depth (how deep in the attractor basin)
            current_position = phase_space[-1]
            phase_center = np.mean(phase_space[-20:], axis=0)
            basin_depth = 1.0 - np.linalg.norm(current_position - phase_center) / (np.std(phase_space[-20:], axis=0).sum() + 1e-8)
            
            return {
                'convergence_rate': float(avg_convergence),
                'basin_stability': float(stability),
                'basin_depth': float(np.clip(basin_depth, 0, 1)),
                'trajectory_coherence': self._calculate_trajectory_coherence(phase_space)
            }
            
        except Exception:
            return {}
    
    def _calculate_trajectory_coherence(self, phase_space: np.ndarray) -> float:
        """Calculate coherence of trajectory in phase space."""
        if len(phase_space) < 10:
            return 0.5
        
        # Calculate direction changes in trajectory
        if phase_space.shape[1] >= 2:
            trajectory_vectors = np.diff(phase_space[:, :2], axis=0)
            
            # Calculate angle changes
            angles = []
            for i in range(1, len(trajectory_vectors)):
                v1 = trajectory_vectors[i-1]
                v2 = trajectory_vectors[i]
                
                dot_product = np.dot(v1, v2)
                norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                
                if norms > 1e-8:
                    angle = np.arccos(np.clip(dot_product / norms, -1, 1))
                    angles.append(angle)
            
            if angles:
                # Coherence as inverse of angle variance
                angle_variance = np.var(angles)
                coherence = 1.0 / (1.0 + angle_variance)
                return float(coherence)
        
        return 0.5
    
    def _classify_market_dynamics(self, lyapunov: float, fractal_dim: float, coherence: float) -> str:
        """Classify market dynamics based on chaos metrics."""
        if lyapunov > 0.1:
            if fractal_dim > 2.0:
                return 'chaotic_complex'
            else:
                return 'chaotic_simple'
        elif lyapunov > 0:
            if coherence > 0.7:
                return 'quasi_periodic'
            else:
                return 'weakly_chaotic'
        else:
            if coherence > 0.8:
                return 'periodic'
            elif fractal_dim < 1.5:
                return 'linear_trend'
            else:
                return 'random_walk'
    
    def _calculate_stability_metrics(self, attractors: Dict, phase_space: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive stability metrics."""
        if not attractors or len(phase_space) == 0:
            return {'overall_stability': 0.5, 'attractor_strength': 0.5}
        
        # Overall system stability
        if 'dominant_attractor' in attractors and attractors['dominant_attractor']:
            attractor_stability = attractors['dominant_attractor']['stability']
            attractor_density = attractors['dominant_attractor']['density']
            
            overall_stability = (attractor_stability * 0.7 + attractor_density * 0.3)
        else:
            overall_stability = 0.5
        
        # Attractor strength (how well-defined are the attractors)
        attractor_count = attractors.get('attractor_count', 0)
        if attractor_count > 0:
            avg_stability = np.mean([a['stability'] for a in attractors.get('attractors', [])])
            attractor_strength = avg_stability / attractor_count if attractor_count > 0 else 0
        else:
            attractor_strength = 0.5
        
        return {
            'overall_stability': float(np.clip(overall_stability, 0, 1)),
            'attractor_strength': float(np.clip(attractor_strength, 0, 1))
        }
    
    def _calculate_phase_space_coverage(self, phase_space: np.ndarray, attractors: List[Dict]) -> float:
        """Calculate how much of phase space is covered by attractors."""
        if len(phase_space) == 0 or not attractors:
            return 0.0
        
        try:
            total_volume = 1.0
            attractor_volume = 0.0
            
            for attractor in attractors:
                # Approximate volume as sphere with given radius
                radius = attractor['radius']
                dimension = phase_space.shape[1]
                
                # Volume of n-dimensional sphere
                volume = (np.pi ** (dimension / 2)) * (radius ** dimension) / np.math.gamma(dimension / 2 + 1)
                attractor_volume += volume
            
            # Calculate phase space volume (approximate)
            ranges = np.ptp(phase_space, axis=0)
            total_volume = np.prod(ranges)
            
            if total_volume > 0:
                coverage = min(attractor_volume / total_volume, 1.0)
                return float(coverage)
            
        except Exception:
            pass
        
        return 0.0
    
    def _calculate_attractor_signal_strength(self, attractors: Dict, equilibrium: Dict,
                                           phase_transition: Dict, stability: Dict) -> float:
        """Calculate signal strength based on attractor analysis."""
        signal_components = []
        
        # Equilibrium prediction signal
        if equilibrium and 'predicted_equilibrium' in equilibrium:
            eq_confidence = equilibrium.get('confidence', 0.5)
            signal_components.append(eq_confidence - 0.5)
        
        # Phase transition signal
        if phase_transition and 'recent_transition' in phase_transition:
            transition = phase_transition['recent_transition']
            if transition:
                transition_signal = transition['strength'] * (1 if transition['type'] == 'volatility_increase' else -1)
                signal_components.append(transition_signal * 0.3)
        
        # Stability signal
        overall_stability = stability.get('overall_stability', 0.5)
        stability_signal = (overall_stability - 0.5) * 0.5
        signal_components.append(stability_signal)
        
        # Attractor strength signal
        attractor_strength = stability.get('attractor_strength', 0.5)
        attractor_signal = (attractor_strength - 0.5) * 0.3
        signal_components.append(attractor_signal)
        
        # Combine signals
        if signal_components:
            total_signal = np.sum(signal_components)
            return float(np.clip(total_signal, -1, 1))
        
        return 0.0
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on attractor analysis."""
        signal_strength = value['signal_strength']
        stability = value['stability_metrics']['overall_stability']
        chaos_detected = value['chaos_detected']
        
        # Reduce confidence during chaotic periods
        confidence_factor = 0.7 if chaos_detected else 1.0
        confidence = min(abs(signal_strength) * stability * confidence_factor, 1.0)
        
        # Require minimum stability for strong signals
        if stability < self.parameters['stability_threshold']:
            return SignalType.NEUTRAL, confidence
        
        if signal_strength > 0.6:
            return SignalType.STRONG_BUY, confidence
        elif signal_strength > 0.3:
            return SignalType.BUY, confidence
        elif signal_strength < -0.6:
            return SignalType.STRONG_SELL, confidence
        elif signal_strength < -0.3:
            return SignalType.SELL, confidence
        else:
            return SignalType.NEUTRAL, confidence
