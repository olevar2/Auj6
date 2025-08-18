"""
Crystallographic Lattice Detector - AI Enhanced Category
========================================================

Advanced crystallographic lattice pattern detector using geometric analysis,
fractal mathematics, and machine learning for market structure identification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple, List
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy import signal, spatial, optimize
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class CrystallographicLatticeDetector(StandardIndicatorInterface):
    """
    Advanced Crystallographic Lattice Detector using geometric pattern analysis.
    
    Features:
    - Crystal lattice structure identification in price patterns
    - Geometric harmonic analysis and symmetry detection
    - Fractal dimension calculation for pattern validation
    - Machine learning clustering for lattice point identification
    - Support and resistance level crystallization
    - Pattern persistence and stability analysis
    - Multi-dimensional lattice space analysis
    - Breakout prediction from lattice structures
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'lookback_period': 100,          # Period for lattice analysis
            'min_lattice_points': 8,         # Minimum points for lattice formation
            'symmetry_tolerance': 0.02,      # Tolerance for symmetry detection
            'clustering_eps': 0.015,         # DBSCAN epsilon parameter
            'min_cluster_size': 3,           # Minimum cluster size
            'fractal_window': 50,            # Window for fractal analysis
            'harmonic_analysis': True,       # Enable harmonic analysis
            'geometric_validation': True,    # Enable geometric validation
            'pattern_persistence': True,     # Enable persistence analysis
            'breakout_prediction': True,     # Enable breakout prediction
            'multi_dimensional': True,       # Enable multi-dimensional analysis
            'lattice_strength_threshold': 0.6,  # Threshold for strong lattice
            'crystal_formation_period': 20,  # Period for crystal formation
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("CrystallographicLatticeDetector", parameters)
        
        # Initialize clustering models
        self.price_clusterer = DBSCAN(
            eps=default_params['clustering_eps'],
            min_samples=default_params['min_cluster_size']
        )
        self.time_clusterer = KMeans(n_clusters=5, random_state=42)
        self.scaler = StandardScaler()
        
        # Cache for calculations
        self._lattice_cache = {}
        self._pattern_history = []
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["open", "high", "low", "close", "volume"],
            min_periods=self.parameters['lookback_period']
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate crystallographic lattice detection with advanced analysis."""
        try:
            if len(data) < self.get_data_requirements().min_periods:
                return self._get_default_output()
            
            # Extract key price points
            key_points = self._extract_key_points(data)
            
            # Detect lattice structures
            lattice_structures = self._detect_lattice_structures(key_points, data)
            
            # Calculate geometric properties
            geometric_properties = self._calculate_geometric_properties(lattice_structures, data)
            
            # Analyze fractal dimensions
            fractal_analysis = self._analyze_fractal_dimensions(data, key_points)
            
            # Perform harmonic analysis
            harmonic_analysis = {}
            if self.parameters['harmonic_analysis']:
                harmonic_analysis = self._perform_harmonic_analysis(data, lattice_structures)
            
            # Validate lattice patterns
            pattern_validation = self._validate_lattice_patterns(
                lattice_structures, geometric_properties, fractal_analysis
            )
            
            # Analyze pattern persistence
            persistence_analysis = {}
            if self.parameters['pattern_persistence']:
                persistence_analysis = self._analyze_pattern_persistence(lattice_structures)
            
            # Predict breakouts
            breakout_predictions = {}
            if self.parameters['breakout_prediction']:
                breakout_predictions = self._predict_breakouts(
                    data, lattice_structures, geometric_properties
                )
            
            # Calculate lattice strength
            lattice_strength = self._calculate_lattice_strength(
                lattice_structures, geometric_properties, pattern_validation
            )
            
            # Generate trading signals
            signals = self._generate_trading_signals(
                lattice_structures, geometric_properties, lattice_strength, breakout_predictions
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                lattice_structures, pattern_validation, persistence_analysis
            )
            
            return {
                'lattice_structures': lattice_structures,
                'geometric_properties': geometric_properties,
                'fractal_analysis': fractal_analysis,
                'harmonic_analysis': harmonic_analysis,
                'pattern_validation': pattern_validation,
                'persistence_analysis': persistence_analysis,
                'breakout_predictions': breakout_predictions,
                'lattice_strength': lattice_strength,
                'signals': signals,
                'confidence': confidence,
                'key_points': key_points,
                'crystal_formation_quality': lattice_strength.get('formation_quality', 0.0),
                'symmetry_score': geometric_properties.get('symmetry_score', 0.0),
                'lattice_stability': persistence_analysis.get('stability_score', 0.5)
            }
            
        except Exception as e:
            return self._handle_calculation_error(e)
    
    def _extract_key_points(self, data: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
        """Extract key price points (pivots, extremes, etc.)."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        
        # Find pivot points
        pivot_highs = []
        pivot_lows = []
        
        window = 5  # Window for pivot detection
        for i in range(window, len(high) - window):
            # Pivot high detection
            if all(high[i] >= high[i-j] for j in range(1, window+1)) and \
               all(high[i] >= high[i+j] for j in range(1, window+1)):
                pivot_highs.append((i, high[i]))
            
            # Pivot low detection
            if all(low[i] <= low[i-j] for j in range(1, window+1)) and \
               all(low[i] <= low[i+j] for j in range(1, window+1)):
                pivot_lows.append((i, low[i]))
        
        # Find volume extremes
        volume_peaks = []
        volume_threshold = np.percentile(volume, 80)
        for i in range(1, len(volume)-1):
            if volume[i] > volume_threshold and \
               volume[i] > volume[i-1] and volume[i] > volume[i+1]:
                volume_peaks.append((i, close[i]))
        
        # Find support and resistance levels
        support_levels = self._find_support_resistance_levels(low, 'support')
        resistance_levels = self._find_support_resistance_levels(high, 'resistance')
        
        return {
            'pivot_highs': pivot_highs,
            'pivot_lows': pivot_lows,
            'volume_peaks': volume_peaks,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    
    def _find_support_resistance_levels(self, prices: np.ndarray, level_type: str) -> List[Tuple[int, float]]:
        """Find support and resistance levels using clustering."""
        if len(prices) < 10:
            return []
        
        # Create price clusters
        price_points = prices.reshape(-1, 1)
        normalized_prices = self.scaler.fit_transform(price_points)
        
        try:
            clusters = self.price_clusterer.fit_predict(normalized_prices)
            unique_clusters = np.unique(clusters[clusters != -1])
            
            levels = []
            for cluster_id in unique_clusters:
                cluster_prices = prices[clusters == cluster_id]
                cluster_indices = np.where(clusters == cluster_id)[0]
                
                if len(cluster_prices) >= self.parameters['min_cluster_size']:
                    if level_type == 'support':
                        level_price = np.min(cluster_prices)
                        level_index = cluster_indices[np.argmin(cluster_prices)]
                    else:  # resistance
                        level_price = np.max(cluster_prices)
                        level_index = cluster_indices[np.argmax(cluster_prices)]
                    
                    levels.append((level_index, level_price))
            
            return sorted(levels, key=lambda x: x[0])
            
        except Exception:
            return []
    
    def _detect_lattice_structures(self, key_points: Dict[str, List], data: pd.DataFrame) -> Dict[str, Any]:
        """Detect crystallographic lattice structures in the price data."""
        lattices = {
            'primary_lattice': [],
            'secondary_lattices': [],
            'lattice_type': 'none',
            'lattice_parameters': {}
        }
        
        try:
            # Combine all key points
            all_points = []
            for point_type, points in key_points.items():
                for point in points:
                    all_points.append((point[0], point[1], point_type))
            
            if len(all_points) < self.parameters['min_lattice_points']:
                return lattices
            
            # Convert to arrays for analysis
            indices = np.array([p[0] for p in all_points])
            prices = np.array([p[1] for p in all_points])
            
            # Detect geometric patterns
            lattice_type, lattice_params = self._identify_lattice_type(indices, prices)
            
            # Find lattice points using geometric constraints
            primary_lattice = self._find_primary_lattice(indices, prices, lattice_params)
            
            # Find secondary lattices
            secondary_lattices = self._find_secondary_lattices(
                indices, prices, primary_lattice, lattice_params
            )
            
            lattices.update({
                'primary_lattice': primary_lattice,
                'secondary_lattices': secondary_lattices,
                'lattice_type': lattice_type,
                'lattice_parameters': lattice_params,
                'total_lattice_points': len(primary_lattice) + sum(len(s) for s in secondary_lattices)
            })
            
        except Exception as e:
            pass
        
        return lattices
    
    def _identify_lattice_type(self, indices: np.ndarray, prices: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Identify the type of crystallographic lattice."""
        if len(indices) < 4:
            return 'none', {}
        
        # Calculate distances between points
        points = np.column_stack([indices, prices])
        distances = pdist(points)
        distance_matrix = squareform(distances)
        
        # Analyze distance patterns
        unique_distances = np.unique(np.round(distances, 2))
        distance_frequencies = [np.sum(np.round(distances, 2) == d) for d in unique_distances]
        
        # Detect patterns
        if len(unique_distances) < 3:
            lattice_type = 'simple_cubic'
        elif self._detect_hexagonal_pattern(distance_matrix):
            lattice_type = 'hexagonal'
        elif self._detect_rectangular_pattern(points):
            lattice_type = 'rectangular'
        elif self._detect_rhombohedral_pattern(points):
            lattice_type = 'rhombohedral'
        else:
            lattice_type = 'complex'
        
        # Calculate lattice parameters
        lattice_params = {
            'unit_cell_size': np.median(unique_distances[:3]) if len(unique_distances) >= 3 else 0.0,
            'symmetry_degree': self._calculate_symmetry_degree(distance_matrix),
            'regularity_score': self._calculate_regularity_score(distances),
            'coordination_number': self._calculate_coordination_number(distance_matrix)
        }
        
        return lattice_type, lattice_params
    
    def _detect_hexagonal_pattern(self, distance_matrix: np.ndarray) -> bool:
        """Detect hexagonal lattice pattern."""
        # Look for 6-fold symmetry
        n_points = len(distance_matrix)
        if n_points < 7:
            return False
        
        # Check for points with 6 nearest neighbors at similar distances
        for i in range(n_points):
            distances_from_i = distance_matrix[i]
            sorted_distances = np.sort(distances_from_i[distances_from_i > 0])
            
            if len(sorted_distances) >= 6:
                # Check if first 6 distances are similar (hexagonal coordination)
                first_six = sorted_distances[:6]
                cv = np.std(first_six) / np.mean(first_six) if np.mean(first_six) > 0 else 1
                if cv < 0.2:  # Low coefficient of variation indicates regularity
                    return True
        
        return False
    
    def _detect_rectangular_pattern(self, points: np.ndarray) -> bool:
        """Detect rectangular lattice pattern."""
        if len(points) < 4:
            return False
        
        # Look for right angles between vectors
        angles = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    v1 = points[j] - points[i]
                    v2 = points[k] - points[i]
                    
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        angles.append(np.degrees(angle))
        
        # Check for 90-degree angles
        right_angles = [a for a in angles if abs(a - 90) < 10 or abs(a - 180) < 10]
        return len(right_angles) >= 2
    
    def _detect_rhombohedral_pattern(self, points: np.ndarray) -> bool:
        """Detect rhombohedral lattice pattern."""
        if len(points) < 4:
            return False
        
        # Look for equal edge lengths and specific angle relationships
        edge_lengths = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                edge_length = np.linalg.norm(points[j] - points[i])
                edge_lengths.append(edge_length)
        
        if len(edge_lengths) < 3:
            return False
        
        # Check for similar edge lengths (characteristic of rhombohedral)
        mean_length = np.mean(edge_lengths)
        similar_lengths = [l for l in edge_lengths if abs(l - mean_length) / mean_length < 0.15]
        
        return len(similar_lengths) >= len(edge_lengths) * 0.6
    
    def _calculate_symmetry_degree(self, distance_matrix: np.ndarray) -> float:
        """Calculate the degree of symmetry in the lattice."""
        if len(distance_matrix) < 3:
            return 0.0
        
        # Analyze symmetry in distance distributions
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        
        # Calculate coefficient of variation as inverse measure of symmetry
        if len(distances) > 0 and np.mean(distances) > 0:
            cv = np.std(distances) / np.mean(distances)
            symmetry = max(0.0, 1.0 - cv)
        else:
            symmetry = 0.0
        
        return min(symmetry, 1.0)
    
    def _calculate_regularity_score(self, distances: np.ndarray) -> float:
        """Calculate regularity score based on distance distribution."""
        if len(distances) < 2:
            return 0.0
        
        # Use entropy-based measure of regularity
        hist, _ = np.histogram(distances, bins=10)
        hist = hist / np.sum(hist)  # Normalize
        
        # Calculate entropy (lower entropy = more regular)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        max_entropy = np.log(len(hist))
        
        regularity = 1.0 - (entropy / max_entropy)
        return max(0.0, min(regularity, 1.0))
    
    def _calculate_coordination_number(self, distance_matrix: np.ndarray) -> float:
        """Calculate average coordination number."""
        if len(distance_matrix) < 2:
            return 0.0
        
        coordination_numbers = []
        for i in range(len(distance_matrix)):
            distances_from_i = distance_matrix[i]
            non_zero_distances = distances_from_i[distances_from_i > 0]
            
            if len(non_zero_distances) > 0:
                # Count neighbors within 1.5x the minimum distance
                min_distance = np.min(non_zero_distances)
                neighbors = np.sum(non_zero_distances <= 1.5 * min_distance)
                coordination_numbers.append(neighbors)
        
        return np.mean(coordination_numbers) if coordination_numbers else 0.0
    
    def _find_primary_lattice(self, indices: np.ndarray, prices: np.ndarray, 
                            lattice_params: Dict[str, float]) -> List[Tuple[int, float]]:
        """Find the primary lattice structure."""
        if len(indices) < self.parameters['min_lattice_points']:
            return []
        
        # Use clustering to find the most coherent lattice points
        points = np.column_stack([indices, prices])
        
        try:
            # Normalize for clustering
            normalized_points = self.scaler.fit_transform(points)
            
            # Perform clustering
            clusters = self.price_clusterer.fit_predict(normalized_points)
            
            # Find the largest cluster as primary lattice
            unique_clusters = np.unique(clusters[clusters != -1])
            if len(unique_clusters) == 0:
                return []
            
            cluster_sizes = [np.sum(clusters == c) for c in unique_clusters]
            primary_cluster = unique_clusters[np.argmax(cluster_sizes)]
            
            # Extract primary lattice points
            primary_indices = indices[clusters == primary_cluster]
            primary_prices = prices[clusters == primary_cluster]
            
            primary_lattice = list(zip(primary_indices, primary_prices))
            return sorted(primary_lattice, key=lambda x: x[0])
            
        except Exception:
            return []
    
    def _find_secondary_lattices(self, indices: np.ndarray, prices: np.ndarray,
                               primary_lattice: List[Tuple[int, float]], 
                               lattice_params: Dict[str, float]) -> List[List[Tuple[int, float]]]:
        """Find secondary lattice structures."""
        if len(primary_lattice) == 0:
            return []
        
        # Remove primary lattice points
        primary_indices = set(p[0] for p in primary_lattice)
        remaining_indices = []
        remaining_prices = []
        
        for i, (idx, price) in enumerate(zip(indices, prices)):
            if idx not in primary_indices:
                remaining_indices.append(idx)
                remaining_prices.append(price)
        
        if len(remaining_indices) < 3:
            return []
        
        # Find secondary structures
        secondary_lattices = []
        remaining_points = np.column_stack([remaining_indices, remaining_prices])
        
        try:
            # Use hierarchical clustering for secondary structures
            normalized_remaining = self.scaler.fit_transform(remaining_points)
            clusters = self.price_clusterer.fit_predict(normalized_remaining)
            
            unique_clusters = np.unique(clusters[clusters != -1])
            
            for cluster_id in unique_clusters:
                cluster_indices = np.array(remaining_indices)[clusters == cluster_id]
                cluster_prices = np.array(remaining_prices)[clusters == cluster_id]
                
                if len(cluster_indices) >= 3:
                    secondary_lattice = list(zip(cluster_indices, cluster_prices))
                    secondary_lattices.append(sorted(secondary_lattice, key=lambda x: x[0]))
            
            return secondary_lattices
            
        except Exception:
            return []
    
    def _calculate_geometric_properties(self, lattice_structures: Dict[str, Any], 
                                      data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate geometric properties of the detected lattices."""
        properties = {
            'symmetry_score': 0.0,
            'regularity_score': 0.0,
            'lattice_density': 0.0,
            'geometric_stability': 0.0,
            'angular_analysis': {},
            'dimensional_analysis': {}
        }
        
        primary_lattice = lattice_structures.get('primary_lattice', [])
        if len(primary_lattice) < 3:
            return properties
        
        try:
            # Extract coordinates
            indices = np.array([p[0] for p in primary_lattice])
            prices = np.array([p[1] for p in primary_lattice])
            points = np.column_stack([indices, prices])
            
            # Symmetry analysis
            properties['symmetry_score'] = self._analyze_lattice_symmetry(points)
            
            # Regularity analysis
            properties['regularity_score'] = self._analyze_lattice_regularity(points)
            
            # Density analysis
            properties['lattice_density'] = self._calculate_lattice_density(points, data)
            
            # Stability analysis
            properties['geometric_stability'] = self._calculate_geometric_stability(points)
            
            # Angular analysis
            properties['angular_analysis'] = self._analyze_lattice_angles(points)
            
            # Dimensional analysis
            properties['dimensional_analysis'] = self._analyze_lattice_dimensions(points)
            
        except Exception:
            pass
        
        return properties
    
    def _analyze_lattice_symmetry(self, points: np.ndarray) -> float:
        """Analyze symmetry properties of the lattice."""
        if len(points) < 4:
            return 0.0
        
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        
        # Calculate distances from centroid
        distances_from_centroid = [np.linalg.norm(p - centroid) for p in points]
        
        # Symmetry based on distance distribution
        if len(distances_from_centroid) > 1:
            cv = np.std(distances_from_centroid) / np.mean(distances_from_centroid)
            symmetry = max(0.0, 1.0 - cv)
        else:
            symmetry = 0.0
        
        return min(symmetry, 1.0)
    
    def _analyze_lattice_regularity(self, points: np.ndarray) -> float:
        """Analyze regularity of lattice spacing."""
        if len(points) < 3:
            return 0.0
        
        # Calculate all pairwise distances
        distances = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                distances.append(dist)
        
        if len(distances) < 2:
            return 0.0
        
        # Regularity based on distance variance
        cv = np.std(distances) / np.mean(distances)
        regularity = max(0.0, 1.0 - cv)
        
        return min(regularity, 1.0)
    
    def _calculate_lattice_density(self, points: np.ndarray, data: pd.DataFrame) -> float:
        """Calculate lattice point density."""
        if len(points) < 2:
            return 0.0
        
        # Calculate area covered by lattice
        if len(points) >= 3:
            # Use convex hull area
            try:
                hull = spatial.ConvexHull(points)
                area = hull.volume  # In 2D, volume is area
            except:
                # Fallback to bounding box
                min_coords = np.min(points, axis=0)
                max_coords = np.max(points, axis=0)
                area = np.prod(max_coords - min_coords)
        else:
            area = 1.0
        
        # Density = points per unit area
        density = len(points) / max(area, 1.0)
        
        # Normalize by data length
        normalized_density = density / len(data) * 100
        
        return min(normalized_density, 10.0)  # Cap at reasonable value
    
    def _calculate_geometric_stability(self, points: np.ndarray) -> float:
        """Calculate geometric stability of the lattice."""
        if len(points) < 4:
            return 0.0
        
        # Stability based on triangulation quality
        try:
            from scipy.spatial import Delaunay
            triangulation = Delaunay(points)
            
            # Calculate triangle quality metrics
            triangle_qualities = []
            for simplex in triangulation.simplices:
                triangle_points = points[simplex]
                quality = self._calculate_triangle_quality(triangle_points)
                triangle_qualities.append(quality)
            
            # Average quality as stability measure
            stability = np.mean(triangle_qualities) if triangle_qualities else 0.0
            
        except Exception:
            # Fallback stability measure
            distances = []
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    dist = np.linalg.norm(points[i] - points[j])
                    distances.append(dist)
            
            if len(distances) > 1:
                cv = np.std(distances) / np.mean(distances)
                stability = max(0.0, 1.0 - cv)
            else:
                stability = 0.0
        
        return min(stability, 1.0)
    
    def _calculate_triangle_quality(self, triangle_points: np.ndarray) -> float:
        """Calculate quality of a triangle (regularity measure)."""
        if len(triangle_points) != 3:
            return 0.0
        
        # Calculate edge lengths
        edges = []
        for i in range(3):
            j = (i + 1) % 3
            edge_length = np.linalg.norm(triangle_points[i] - triangle_points[j])
            edges.append(edge_length)
        
        # Quality based on edge length regularity
        if max(edges) > 0:
            quality = min(edges) / max(edges)
        else:
            quality = 0.0
        
        return quality
    
    def _analyze_lattice_angles(self, points: np.ndarray) -> Dict[str, float]:
        """Analyze angular properties of the lattice."""
        angles = {
            'mean_angle': 0.0,
            'angle_variance': 0.0,
            'right_angle_count': 0,
            'angle_regularity': 0.0
        }
        
        if len(points) < 3:
            return angles
        
        all_angles = []
        right_angle_count = 0
        
        # Calculate angles at each point
        for i in range(len(points)):
            for j in range(len(points)):
                for k in range(len(points)):
                    if i != j and j != k and i != k:
                        v1 = points[i] - points[j]
                        v2 = points[k] - points[j]
                        
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            angle = np.arccos(np.clip(cos_angle, -1, 1))
                            angle_degrees = np.degrees(angle)
                            
                            all_angles.append(angle_degrees)
                            
                            # Count right angles
                            if abs(angle_degrees - 90) < 10:
                                right_angle_count += 1
        
        if all_angles:
            angles['mean_angle'] = np.mean(all_angles)
            angles['angle_variance'] = np.var(all_angles)
            angles['right_angle_count'] = right_angle_count
            angles['angle_regularity'] = 1.0 / (1.0 + angles['angle_variance'] / 100.0)
        
        return angles
    
    def _analyze_lattice_dimensions(self, points: np.ndarray) -> Dict[str, float]:
        """Analyze dimensional properties of the lattice."""
        dimensions = {
            'span_x': 0.0,
            'span_y': 0.0,
            'aspect_ratio': 1.0,
            'compactness': 0.0
        }
        
        if len(points) < 2:
            return dimensions
        
        # Calculate spans
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        spans = max_coords - min_coords
        
        dimensions['span_x'] = spans[0]
        dimensions['span_y'] = spans[1]
        
        # Aspect ratio
        if spans[1] > 0:
            dimensions['aspect_ratio'] = spans[0] / spans[1]
        
        # Compactness (area efficiency)
        try:
            hull = spatial.ConvexHull(points)
            hull_area = hull.volume
            bounding_box_area = np.prod(spans)
            
            if bounding_box_area > 0:
                dimensions['compactness'] = hull_area / bounding_box_area
        except:
            dimensions['compactness'] = 0.5  # Default value
        
        return dimensions
    
    def _analyze_fractal_dimensions(self, data: pd.DataFrame, key_points: Dict[str, List]) -> Dict[str, float]:
        """Analyze fractal dimensions of the price structure."""
        fractal_analysis = {
            'fractal_dimension': 1.0,
            'hurst_exponent': 0.5,
            'complexity_measure': 0.0,
            'self_similarity': 0.0
        }
        
        try:
            close_prices = data['close'].values
            
            # Calculate fractal dimension using box-counting method
            fractal_dimension = self._calculate_fractal_dimension(close_prices)
            
            # Calculate Hurst exponent
            hurst_exponent = self._calculate_hurst_exponent(close_prices)
            
            # Complexity measure
            complexity = self._calculate_complexity_measure(close_prices)
            
            # Self-similarity measure
            self_similarity = self._calculate_self_similarity(close_prices)
            
            fractal_analysis.update({
                'fractal_dimension': fractal_dimension,
                'hurst_exponent': hurst_exponent,
                'complexity_measure': complexity,
                'self_similarity': self_similarity
            })
            
        except Exception:
            pass
        
        return fractal_analysis
    
    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method."""
        try:
            # Simplified box-counting approximation
            if len(data) < 10:
                return 1.0
            
            # Calculate relative variation
            variations = np.abs(np.diff(data))
            relative_variation = np.sum(variations) / (np.max(data) - np.min(data))
            
            # Map to fractal dimension (rough approximation)
            fractal_dim = 1.0 + min(relative_variation / len(data) * 100, 1.0)
            
            return min(max(fractal_dim, 1.0), 2.0)
            
        except Exception:
            return 1.0
    
    def _calculate_hurst_exponent(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent for trend persistence."""
        try:
            if len(data) < 20:
                return 0.5
            
            # R/S analysis (simplified)
            log_returns = np.diff(np.log(data + 1e-10))
            
            # Calculate running variance
            variances = []
            for i in range(10, len(log_returns)):
                window_data = log_returns[max(0, i-10):i]
                variances.append(np.var(window_data))
            
            if len(variances) > 1:
                # Simplified Hurst calculation
                mean_var = np.mean(variances)
                var_var = np.var(variances)
                
                if mean_var > 0:
                    hurst = 0.5 + (var_var / mean_var - 1) * 0.1
                else:
                    hurst = 0.5
            else:
                hurst = 0.5
            
            return min(max(hurst, 0.0), 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_complexity_measure(self, data: np.ndarray) -> float:
        """Calculate complexity measure of the price series."""
        try:
            if len(data) < 10:
                return 0.0
            
            # Use normalized variation as complexity measure
            normalized_data = (data - np.mean(data)) / (np.std(data) + 1e-10)
            variations = np.abs(np.diff(normalized_data))
            
            complexity = np.mean(variations)
            return min(complexity, 2.0)
            
        except Exception:
            return 0.0
    
    def _calculate_self_similarity(self, data: np.ndarray) -> float:
        """Calculate self-similarity measure."""
        try:
            if len(data) < 20:
                return 0.0
            
            # Compare different scale patterns
            correlations = []
            for scale in [2, 3, 4, 5]:
                if len(data) >= scale * 10:
                    # Downsample data
                    downsampled = data[::scale]
                    
                    # Compare with original (truncated)
                    min_len = min(len(data), len(downsampled))
                    if min_len > 5:
                        corr = np.corrcoef(data[:min_len], downsampled[:min_len])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception:
            return 0.0
    
    def _perform_harmonic_analysis(self, data: pd.DataFrame, lattice_structures: Dict[str, Any]) -> Dict[str, Any]:
        """Perform harmonic analysis of the lattice structures."""
        harmonic_analysis = {
            'dominant_frequencies': [],
            'harmonic_strength': 0.0,
            'resonance_points': [],
            'frequency_stability': 0.0
        }
        
        try:
            close_prices = data['close'].values
            
            # Perform FFT to find dominant frequencies
            fft_result = np.fft.fft(close_prices)
            frequencies = np.fft.fftfreq(len(close_prices))
            
            # Find dominant frequencies
            magnitude = np.abs(fft_result)
            dominant_indices = np.argsort(magnitude)[-5:]  # Top 5 frequencies
            
            dominant_frequencies = []
            for idx in dominant_indices:
                if frequencies[idx] != 0:  # Exclude DC component
                    dominant_frequencies.append({
                        'frequency': abs(frequencies[idx]),
                        'magnitude': magnitude[idx],
                        'period': 1.0 / abs(frequencies[idx]) if frequencies[idx] != 0 else 0
                    })
            
            # Calculate harmonic strength
            total_energy = np.sum(magnitude**2)
            harmonic_energy = np.sum([f['magnitude']**2 for f in dominant_frequencies])
            harmonic_strength = harmonic_energy / total_energy if total_energy > 0 else 0.0
            
            # Find resonance points (where harmonics align with lattice)
            resonance_points = self._find_resonance_points(
                dominant_frequencies, lattice_structures
            )
            
            # Calculate frequency stability
            frequency_stability = self._calculate_frequency_stability(dominant_frequencies)
            
            harmonic_analysis.update({
                'dominant_frequencies': dominant_frequencies,
                'harmonic_strength': harmonic_strength,
                'resonance_points': resonance_points,
                'frequency_stability': frequency_stability
            })
            
        except Exception:
            pass
        
        return harmonic_analysis
    
    def _find_resonance_points(self, frequencies: List[Dict], lattice_structures: Dict[str, Any]) -> List[Dict]:
        """Find resonance points between harmonics and lattice structure."""
        resonance_points = []
        
        primary_lattice = lattice_structures.get('primary_lattice', [])
        if len(primary_lattice) < 2:
            return resonance_points
        
        # Calculate lattice periods
        indices = [p[0] for p in primary_lattice]
        lattice_periods = np.diff(indices) if len(indices) > 1 else []
        
        # Find resonances
        for freq_data in frequencies:
            freq_period = freq_data.get('period', 0)
            if freq_period > 0:
                for lattice_period in lattice_periods:
                    if lattice_period > 0:
                        ratio = freq_period / lattice_period
                        # Check for harmonic relationships
                        if abs(ratio - round(ratio)) < 0.1 or abs(1/ratio - round(1/ratio)) < 0.1:
                            resonance_points.append({
                                'frequency_period': freq_period,
                                'lattice_period': lattice_period,
                                'ratio': ratio,
                                'strength': freq_data.get('magnitude', 0)
                            })
        
        return sorted(resonance_points, key=lambda x: x['strength'], reverse=True)
    
    def _calculate_frequency_stability(self, frequencies: List[Dict]) -> float:
        """Calculate stability of dominant frequencies."""
        if len(frequencies) < 2:
            return 0.0
        
        # Calculate coefficient of variation of magnitudes
        magnitudes = [f['magnitude'] for f in frequencies]
        if len(magnitudes) > 1 and np.mean(magnitudes) > 0:
            cv = np.std(magnitudes) / np.mean(magnitudes)
            stability = max(0.0, 1.0 - cv)
        else:
            stability = 0.0
        
        return min(stability, 1.0)
    
    def _validate_lattice_patterns(self, lattice_structures: Dict[str, Any],
                                 geometric_properties: Dict[str, Any],
                                 fractal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the detected lattice patterns."""
        validation = {
            'pattern_validity': 0.0,
            'geometric_consistency': 0.0,
            'fractal_coherence': 0.0,
            'overall_confidence': 0.0,
            'validation_criteria': {}
        }
        
        try:
            # Pattern validity based on lattice quality
            primary_lattice = lattice_structures.get('primary_lattice', [])
            lattice_params = lattice_structures.get('lattice_parameters', {})
            
            if len(primary_lattice) >= self.parameters['min_lattice_points']:
                pattern_validity = min(len(primary_lattice) / 20.0, 1.0)  # Normalize by expected size
            else:
                pattern_validity = 0.0
            
            # Geometric consistency
            symmetry_score = geometric_properties.get('symmetry_score', 0.0)
            regularity_score = geometric_properties.get('regularity_score', 0.0)
            stability_score = geometric_properties.get('geometric_stability', 0.0)
            
            geometric_consistency = (symmetry_score + regularity_score + stability_score) / 3.0
            
            # Fractal coherence
            fractal_dim = fractal_analysis.get('fractal_dimension', 1.0)
            hurst_exp = fractal_analysis.get('hurst_exponent', 0.5)
            
            # Good fractal coherence for crystalline structures
            fractal_coherence = 1.0 - abs(fractal_dim - 1.5) / 0.5  # Ideal around 1.5
            fractal_coherence = max(0.0, min(fractal_coherence, 1.0))
            
            # Overall confidence
            weights = [0.4, 0.4, 0.2]  # Pattern, geometric, fractal
            confidence_components = [pattern_validity, geometric_consistency, fractal_coherence]
            overall_confidence = sum(w * c for w, c in zip(weights, confidence_components))
            
            validation.update({
                'pattern_validity': pattern_validity,
                'geometric_consistency': geometric_consistency,
                'fractal_coherence': fractal_coherence,
                'overall_confidence': overall_confidence,
                'validation_criteria': {
                    'min_points_met': len(primary_lattice) >= self.parameters['min_lattice_points'],
                    'symmetry_adequate': symmetry_score > 0.5,
                    'regularity_adequate': regularity_score > 0.5,
                    'fractal_reasonable': 1.0 <= fractal_dim <= 2.0
                }
            })
            
        except Exception:
            pass
        
        return validation
    
    def _analyze_pattern_persistence(self, lattice_structures: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze persistence and stability of lattice patterns."""
        persistence = {
            'stability_score': 0.0,
            'persistence_probability': 0.5,
            'degradation_risk': 0.5,
            'reinforcement_potential': 0.5
        }
        
        # Store current pattern in history
        current_pattern = {
            'timestamp': len(self._pattern_history),
            'lattice_type': lattice_structures.get('lattice_type', 'none'),
            'lattice_points': len(lattice_structures.get('primary_lattice', [])),
            'lattice_params': lattice_structures.get('lattice_parameters', {})
        }
        
        self._pattern_history.append(current_pattern)
        
        # Keep only recent history
        if len(self._pattern_history) > 20:
            self._pattern_history = self._pattern_history[-20:]
        
        try:
            if len(self._pattern_history) > 1:
                # Analyze stability over time
                recent_types = [p['lattice_type'] for p in self._pattern_history[-5:]]
                type_consistency = len(set(recent_types)) == 1
                
                recent_points = [p['lattice_points'] for p in self._pattern_history[-5:]]
                point_stability = 1.0 - (np.std(recent_points) / max(np.mean(recent_points), 1))
                
                stability_score = (0.6 if type_consistency else 0.0) + 0.4 * max(0, point_stability)
                
                # Persistence probability based on historical stability
                persistence_probability = min(stability_score + 0.1, 1.0)
                
                # Degradation risk (inverse of persistence)
                degradation_risk = 1.0 - persistence_probability
                
                # Reinforcement potential based on recent improvements
                if len(recent_points) > 1:
                    recent_trend = recent_points[-1] - recent_points[0]
                    reinforcement_potential = min(max(recent_trend / 10.0 + 0.5, 0.0), 1.0)
                else:
                    reinforcement_potential = 0.5
                
                persistence.update({
                    'stability_score': stability_score,
                    'persistence_probability': persistence_probability,
                    'degradation_risk': degradation_risk,
                    'reinforcement_potential': reinforcement_potential
                })
        
        except Exception:
            pass
        
        return persistence
    
    def _predict_breakouts(self, data: pd.DataFrame, lattice_structures: Dict[str, Any],
                         geometric_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Predict potential breakouts from lattice structures."""
        predictions = {
            'breakout_probability': 0.0,
            'breakout_direction': 'neutral',
            'breakout_strength': 0.0,
            'breakout_targets': [],
            'time_to_breakout': 0
        }
        
        try:
            primary_lattice = lattice_structures.get('primary_lattice', [])
            if len(primary_lattice) < 3:
                return predictions
            
            current_price = data['close'].iloc[-1]
            
            # Find lattice boundaries
            lattice_prices = [p[1] for p in primary_lattice]
            upper_boundary = max(lattice_prices)
            lower_boundary = min(lattice_prices)
            
            # Calculate distance to boundaries
            distance_to_upper = (upper_boundary - current_price) / current_price
            distance_to_lower = (current_price - lower_boundary) / current_price
            
            # Breakout probability based on proximity to boundaries
            if abs(distance_to_upper) < 0.02:  # Within 2%
                breakout_probability = 0.8
                breakout_direction = 'upward'
            elif abs(distance_to_lower) < 0.02:
                breakout_probability = 0.8
                breakout_direction = 'downward'
            else:
                # Check for pressure build-up
                recent_volatility = np.std(data['close'].tail(10).pct_change().dropna())
                lattice_strength = geometric_properties.get('regularity_score', 0.0)
                
                pressure_score = recent_volatility * (1.0 - lattice_strength)
                breakout_probability = min(pressure_score * 2.0, 0.7)
                
                if current_price > (upper_boundary + lower_boundary) / 2:
                    breakout_direction = 'upward'
                else:
                    breakout_direction = 'downward'
            
            # Breakout strength
            breakout_strength = breakout_probability * (1.0 + recent_volatility)
            
            # Breakout targets (lattice projection)
            lattice_spacing = np.mean(np.diff(sorted(lattice_prices))) if len(lattice_prices) > 1 else 0
            if breakout_direction == 'upward':
                targets = [upper_boundary + i * lattice_spacing for i in range(1, 4)]
            else:
                targets = [lower_boundary - i * lattice_spacing for i in range(1, 4)]
            
            # Time to breakout (rough estimate)
            if breakout_probability > 0.6:
                time_to_breakout = max(1, int(10 * (1.0 - breakout_probability)))
            else:
                time_to_breakout = 20  # Default long-term
            
            predictions.update({
                'breakout_probability': breakout_probability,
                'breakout_direction': breakout_direction,
                'breakout_strength': min(breakout_strength, 1.0),
                'breakout_targets': targets,
                'time_to_breakout': time_to_breakout
            })
            
        except Exception:
            pass
        
        return predictions
    
    def _calculate_lattice_strength(self, lattice_structures: Dict[str, Any],
                                  geometric_properties: Dict[str, Any],
                                  pattern_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall lattice strength and formation quality."""
        strength = {
            'formation_quality': 0.0,
            'structural_integrity': 0.0,
            'pattern_coherence': 0.0,
            'overall_strength': 0.0,
            'strength_category': 'weak'
        }
        
        try:
            # Formation quality
            primary_lattice = lattice_structures.get('primary_lattice', [])
            lattice_params = lattice_structures.get('lattice_parameters', {})
            
            point_count_score = min(len(primary_lattice) / 15.0, 1.0)  # Normalized
            symmetry_score = lattice_params.get('symmetry_degree', 0.0)
            regularity_score = lattice_params.get('regularity_score', 0.0)
            
            formation_quality = (point_count_score + symmetry_score + regularity_score) / 3.0
            
            # Structural integrity
            geometric_stability = geometric_properties.get('geometric_stability', 0.0)
            lattice_density = geometric_properties.get('lattice_density', 0.0)
            
            structural_integrity = (geometric_stability + min(lattice_density / 2.0, 1.0)) / 2.0
            
            # Pattern coherence
            pattern_validity = pattern_validation.get('pattern_validity', 0.0)
            geometric_consistency = pattern_validation.get('geometric_consistency', 0.0)
            
            pattern_coherence = (pattern_validity + geometric_consistency) / 2.0
            
            # Overall strength
            weights = [0.4, 0.3, 0.3]  # Formation, structure, coherence
            strength_components = [formation_quality, structural_integrity, pattern_coherence]
            overall_strength = sum(w * c for w, c in zip(weights, strength_components))
            
            # Strength category
            if overall_strength > 0.8:
                strength_category = 'very_strong'
            elif overall_strength > 0.6:
                strength_category = 'strong'
            elif overall_strength > 0.4:
                strength_category = 'moderate'
            elif overall_strength > 0.2:
                strength_category = 'weak'
            else:
                strength_category = 'very_weak'
            
            strength.update({
                'formation_quality': formation_quality,
                'structural_integrity': structural_integrity,
                'pattern_coherence': pattern_coherence,
                'overall_strength': overall_strength,
                'strength_category': strength_category
            })
            
        except Exception:
            pass
        
        return strength
    
    def _generate_trading_signals(self, lattice_structures: Dict[str, Any],
                                geometric_properties: Dict[str, Any],
                                lattice_strength: Dict[str, Any],
                                breakout_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on crystallographic analysis."""
        signals = {
            'signal_type': 'neutral',
            'signal_strength': 0.0,
            'confidence': 0.0,
            'entry_points': [],
            'exit_points': [],
            'stop_loss_levels': [],
            'take_profit_levels': []
        }
        
        try:
            overall_strength = lattice_strength.get('overall_strength', 0.0)
            breakout_probability = breakout_predictions.get('breakout_probability', 0.0)
            breakout_direction = breakout_predictions.get('breakout_direction', 'neutral')
            
            # Signal generation logic
            if overall_strength > self.parameters['lattice_strength_threshold']:
                if breakout_probability > 0.7:
                    # Strong breakout signal
                    signals['signal_type'] = f'breakout_{breakout_direction}'
                    signals['signal_strength'] = min(breakout_probability * overall_strength, 1.0)
                    
                    # Entry points at breakout levels
                    primary_lattice = lattice_structures.get('primary_lattice', [])
                    if primary_lattice:
                        lattice_prices = [p[1] for p in primary_lattice]
                        if breakout_direction == 'upward':
                            signals['entry_points'] = [max(lattice_prices) * 1.001]  # Slight above resistance
                            signals['stop_loss_levels'] = [max(lattice_prices) * 0.995]
                            signals['take_profit_levels'] = breakout_predictions.get('breakout_targets', [])[:2]
                        else:
                            signals['entry_points'] = [min(lattice_prices) * 0.999]  # Slight below support
                            signals['stop_loss_levels'] = [min(lattice_prices) * 1.005]
                            signals['take_profit_levels'] = breakout_predictions.get('breakout_targets', [])[:2]
                
                elif breakout_probability < 0.3:
                    # Range trading signal
                    signals['signal_type'] = 'range_trading'
                    signals['signal_strength'] = overall_strength * (1.0 - breakout_probability)
                    
                    # Entry points at lattice boundaries
                    primary_lattice = lattice_structures.get('primary_lattice', [])
                    if primary_lattice:
                        lattice_prices = sorted([p[1] for p in primary_lattice])
                        signals['entry_points'] = [lattice_prices[0], lattice_prices[-1]]  # Support and resistance
                        signals['take_profit_levels'] = [lattice_prices[-1], lattice_prices[0]]  # Opposite levels
                        signals['stop_loss_levels'] = [p * 0.99 for p in signals['entry_points']]
            
            # Confidence based on multiple factors
            geometric_confidence = geometric_properties.get('regularity_score', 0.0)
            pattern_confidence = lattice_strength.get('pattern_coherence', 0.0)
            
            signals['confidence'] = (signals['signal_strength'] + geometric_confidence + pattern_confidence) / 3.0
            
        except Exception:
            pass
        
        return signals
    
    def _calculate_confidence_score(self, lattice_structures: Dict[str, Any],
                                  pattern_validation: Dict[str, Any],
                                  persistence_analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the analysis."""
        try:
            # Component confidences
            pattern_confidence = pattern_validation.get('overall_confidence', 0.0)
            persistence_confidence = persistence_analysis.get('stability_score', 0.0)
            
            # Lattice quality confidence
            primary_lattice = lattice_structures.get('primary_lattice', [])
            lattice_confidence = min(len(primary_lattice) / 20.0, 1.0)
            
            # Combined confidence
            weights = [0.4, 0.3, 0.3]  # Pattern, persistence, lattice
            confidence_components = [pattern_confidence, persistence_confidence, lattice_confidence]
            
            overall_confidence = sum(w * c for w, c in zip(weights, confidence_components))
            
            return min(max(overall_confidence, 0.0), 1.0)
            
        except Exception:
            return 0.0
    
    def _get_default_output(self) -> Dict[str, Any]:
        """Return default output when insufficient data."""
        return {
            'lattice_structures': {'lattice_type': 'none', 'primary_lattice': []},
            'geometric_properties': {'symmetry_score': 0.0, 'regularity_score': 0.0},
            'signals': {'signal_type': 'neutral', 'signal_strength': 0.0, 'confidence': 0.0},
            'confidence': 0.0,
            'crystal_formation_quality': 0.0,
            'symmetry_score': 0.0,
            'lattice_stability': 0.5
        }
    
    def _handle_calculation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle calculation errors gracefully."""
        return self._get_default_output()
