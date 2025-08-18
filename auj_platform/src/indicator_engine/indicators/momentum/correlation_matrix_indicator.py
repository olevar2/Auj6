"""
Correlation Matrix Indicator - Advanced Implementation
=====================================================

Advanced correlation matrix analysis for momentum and trend detection,
with machine learning integration for market regime classification.

Author: AUJ Platform Development Team
Mission: Building advanced trading indicators for humanitarian impact
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from scipy import stats
from scipy.linalg import eigh
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import (
    StandardIndicatorInterface, 
    DataRequirement, 
    DataType, 
    SignalType
)
from ....core.exceptions import IndicatorCalculationException


class CorrelationMatrixIndicator(StandardIndicatorInterface):
    """
    Advanced Correlation Matrix Analysis Implementation
    
    Features:
    - Multi-asset correlation analysis with adaptive windows
    - Principal Component Analysis for factor extraction
    - ML-enhanced regime classification based on correlation patterns
    - Dynamic correlation clustering for market structure detection
    - Statistical significance testing for correlation stability
    - Advanced signal generation from correlation breakdown/formation
    - Risk-adjusted correlation analysis with volatility weighting
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'correlation_window': 30,
            'rolling_window': 20,
            'assets': ['price', 'volume', 'volatility'],
            'pca_components': 3,
            'clustering_enabled': True,
            'regime_classification': True,
            'significance_test': True,
            'adaptive_window': True,
            'volatility_weighted': True,
            'min_correlation_threshold': 0.3,
            'correlation_breakout_threshold': 0.7,
            'stability_lookback': 10,
            'ml_lookback': 50,
            'risk_adjustment': True
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="CorrelationMatrixIndicator", parameters=default_params)
        
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.regime_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.correlation_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.pca = PCA(n_components=self.parameters['pca_components'])
        self.ica = FastICA(n_components=2, random_state=42)
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)
        self.models_trained = False
        
        self.history = {
            'correlation_matrices': [],
            'eigenvalues': [],
            'pca_components': [],
            'cluster_labels': [],
            'regime_predictions': [],
            'correlation_signals': []
        }
    
    def get_data_requirements(self) -> DataRequirement:
        max_window = max(self.parameters['correlation_window'], 
                        self.parameters['rolling_window'],
                        self.parameters['ml_lookback'])
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close', 'volume'],
            min_periods=max_window * 2 + 50,
            lookback_periods=200
        )
    
    def _create_asset_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create multi-asset data matrix for correlation analysis"""
        asset_data = pd.DataFrame(index=data.index)
        
        # Price-based assets
        asset_data['close_price'] = data['close']
        asset_data['high_low_range'] = (data['high'] - data['low']) / data['close']
        asset_data['open_close_range'] = (data['close'] - data['open']) / data['open']
        
        # Returns-based assets
        asset_data['returns'] = data['close'].pct_change()
        asset_data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Volume-based assets
        if 'volume' in data.columns:
            asset_data['volume'] = data['volume']
            asset_data['volume_price'] = data['volume'] * data['close']
            asset_data['volume_returns'] = data['volume'].pct_change()
        else:
            asset_data['volume'] = 1
            asset_data['volume_price'] = data['close']
            asset_data['volume_returns'] = 0
        
        # Volatility measures
        asset_data['realized_volatility'] = asset_data['returns'].rolling(window=10).std()
        asset_data['close_to_close_vol'] = asset_data['log_returns'].rolling(window=10).std()
        asset_data['high_low_vol'] = asset_data['high_low_range'].rolling(window=10).std()
        
        # Technical indicators for correlation
        asset_data['sma_ratio'] = data['close'] / data['close'].rolling(window=20).mean()
        asset_data['ema_ratio'] = data['close'] / data['close'].ewm(span=12).mean()
        asset_data['rsi_proxy'] = self._calculate_rsi_proxy(data['close'])
        
        # Momentum indicators
        asset_data['momentum_5'] = data['close'] / data['close'].shift(5)
        asset_data['momentum_10'] = data['close'] / data['close'].shift(10)
        asset_data['momentum_20'] = data['close'] / data['close'].shift(20)
        
        # Advanced price relationships
        asset_data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
        asset_data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']
        asset_data['body_size'] = abs(data['close'] - data['open']) / data['close']
        
        return asset_data.dropna()
    
    def _calculate_rsi_proxy(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate a simple RSI proxy for correlation analysis"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _adapt_correlation_window(self, asset_data: pd.DataFrame) -> int:
        """Adapt correlation window based on market conditions"""
        if not self.parameters['adaptive_window'] or len(asset_data) < 60:
            return self.parameters['correlation_window']
        
        # Calculate market volatility
        returns = asset_data['returns'].tail(60)
        current_vol = returns.std()
        avg_vol = returns.rolling(window=20).std().mean()
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        # Calculate correlation stability
        recent_data = asset_data.tail(60)
        if len(recent_data) >= 30:
            corr_matrix_1 = recent_data.tail(30).corr()
            corr_matrix_2 = recent_data.head(30).corr()
            
            # Calculate Frobenius norm of difference
            correlation_stability = np.linalg.norm(corr_matrix_1.values - corr_matrix_2.values, 'fro')
        else:
            correlation_stability = 1.0
        
        base_window = self.parameters['correlation_window']
        
        # Adjust window based on volatility and stability
        if vol_ratio > 1.5 or correlation_stability > 2.0:  # High volatility or unstable correlations
            adjusted_window = max(15, int(base_window * 0.7))
        elif vol_ratio < 0.6 and correlation_stability < 0.5:  # Low volatility and stable correlations
            adjusted_window = min(60, int(base_window * 1.4))
        else:
            adjusted_window = base_window
        
        return adjusted_window
    
    def _calculate_correlation_matrix(self, asset_data: pd.DataFrame, window: int) -> np.ndarray:
        """Calculate correlation matrix with optional volatility weighting"""
        if len(asset_data) < window:
            return np.eye(len(asset_data.columns))
        
        recent_data = asset_data.tail(window)
        
        if self.parameters['volatility_weighted']:
            # Calculate volatility weights
            volatilities = recent_data.std()
            weights = 1.0 / (volatilities + 1e-8)
            weights = weights / weights.sum()
            
            # Apply weights to data
            weighted_data = recent_data * np.sqrt(weights)
            correlation_matrix = weighted_data.corr().values
        else:
            correlation_matrix = recent_data.corr().values
        
        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        # Ensure positive semi-definite
        eigenvals, eigenvecs = eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return correlation_matrix
    
    def _perform_pca_analysis(self, correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """Perform Principal Component Analysis on correlation matrix"""
        try:
            eigenvals, eigenvecs = eigh(correlation_matrix)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Calculate explained variance ratio
            total_variance = np.sum(eigenvals)
            explained_variance_ratio = eigenvals / total_variance
            
            # Extract principal components
            n_components = min(self.parameters['pca_components'], len(eigenvals))
            principal_components = eigenvecs[:, :n_components]
            
            # Calculate component loadings
            loadings = principal_components * np.sqrt(eigenvals[:n_components])
            
            return {
                'eigenvalues': eigenvals.tolist(),
                'eigenvectors': eigenvecs.tolist(),
                'explained_variance_ratio': explained_variance_ratio.tolist(),
                'principal_components': principal_components.tolist(),
                'loadings': loadings.tolist(),
                'cumulative_variance': np.cumsum(explained_variance_ratio).tolist(),
                'dominant_eigenvalue': float(eigenvals[0]),
                'spectral_radius': float(np.max(np.abs(eigenvals))),
                'condition_number': float(eigenvals[0] / (eigenvals[-1] + 1e-8))
            }
        except Exception:
            # Return empty result if PCA fails
            n_assets = correlation_matrix.shape[0]
            return {
                'eigenvalues': [1.0] * n_assets,
                'eigenvectors': np.eye(n_assets).tolist(),
                'explained_variance_ratio': [1.0/n_assets] * n_assets,
                'principal_components': np.eye(n_assets).tolist(),
                'loadings': np.eye(n_assets).tolist(),
                'cumulative_variance': list(np.cumsum([1.0/n_assets] * n_assets)),
                'dominant_eigenvalue': 1.0,
                'spectral_radius': 1.0,
                'condition_number': 1.0
            }
    
    def _detect_correlation_clusters(self, correlation_matrix: np.ndarray, asset_names: List[str]) -> Dict[str, Any]:
        """Detect clusters in the correlation structure"""
        if not self.parameters['clustering_enabled'] or correlation_matrix.shape[0] < 4:
            return {'clusters': {}, 'n_clusters': 0, 'silhouette_score': 0.0}
        
        try:
            # Convert correlation to distance matrix
            distance_matrix = 1 - np.abs(correlation_matrix)
            
            # Ensure diagonal is zero
            np.fill_diagonal(distance_matrix, 0)
            
            # Apply K-means clustering
            if correlation_matrix.shape[0] >= 4:
                n_clusters = min(4, correlation_matrix.shape[0] // 2)
                kmeans_labels = self.kmeans.fit_predict(distance_matrix)
                
                # Apply DBSCAN clustering
                dbscan_labels = self.dbscan.fit_predict(distance_matrix)
                
                # Choose better clustering result
                if len(set(dbscan_labels)) > 1 and -1 not in dbscan_labels:
                    cluster_labels = dbscan_labels
                    clustering_method = 'dbscan'
                else:
                    cluster_labels = kmeans_labels
                    clustering_method = 'kmeans'
                
                # Create cluster dictionary
                clusters = {}
                for i, asset in enumerate(asset_names):
                    cluster_id = int(cluster_labels[i])
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(asset)
                
                # Calculate silhouette score approximation
                if len(set(cluster_labels)) > 1:
                    silhouette_score = self._calculate_silhouette_score(distance_matrix, cluster_labels)
                else:
                    silhouette_score = 0.0
            else:
                clusters = {0: asset_names}
                silhouette_score = 0.0
                clustering_method = 'single_cluster'
            
            return {
                'clusters': clusters,
                'n_clusters': len(clusters),
                'silhouette_score': float(silhouette_score),
                'clustering_method': clustering_method,
                'cluster_labels': cluster_labels.tolist() if 'cluster_labels' in locals() else [0] * len(asset_names)
            }
        except Exception:
            return {
                'clusters': {0: asset_names},
                'n_clusters': 1,
                'silhouette_score': 0.0,
                'clustering_method': 'fallback',
                'cluster_labels': [0] * len(asset_names)
            }
    
    def _calculate_silhouette_score(self, distance_matrix: np.ndarray, labels: np.ndarray) -> float:
        """Calculate simplified silhouette score"""
        try:
            n_samples = len(labels)
            silhouette_scores = []
            
            for i in range(n_samples):
                # Mean distance to points in same cluster
                same_cluster_distances = [distance_matrix[i, j] for j in range(n_samples) if labels[j] == labels[i] and i != j]
                a = np.mean(same_cluster_distances) if same_cluster_distances else 0
                
                # Mean distance to points in nearest different cluster
                b = float('inf')
                for cluster_label in set(labels):
                    if cluster_label != labels[i]:
                        cluster_distances = [distance_matrix[i, j] for j in range(n_samples) if labels[j] == cluster_label]
                        if cluster_distances:
                            mean_dist = np.mean(cluster_distances)
                            b = min(b, mean_dist)
                
                if b == float('inf'):
                    b = 0
                
                # Silhouette score for point i
                if max(a, b) > 0:
                    s = (b - a) / max(a, b)
                else:
                    s = 0
                
                silhouette_scores.append(s)
            
            return np.mean(silhouette_scores)
        except:
            return 0.0
    
    def _test_correlation_significance(self, correlation_matrix: np.ndarray, sample_size: int) -> Dict[str, Any]:
        """Test statistical significance of correlations"""
        if not self.parameters['significance_test'] or sample_size < 10:
            return {'significant_correlations': 0, 'avg_p_value': 1.0, 'critical_value': 0.0}
        
        # Calculate critical value for given sample size (95% confidence)
        critical_value = 1.96 / np.sqrt(sample_size - 3)
        
        significant_count = 0
        p_values = []
        n_correlations = 0
        
        for i in range(correlation_matrix.shape[0]):
            for j in range(i+1, correlation_matrix.shape[1]):
                correlation = correlation_matrix[i, j]
                n_correlations += 1
                
                # Fisher's z-transformation
                if abs(correlation) >= 0.99:
                    correlation = 0.99 * np.sign(correlation)
                
                z = 0.5 * np.log((1 + correlation) / (1 - correlation))
                z_stat = z * np.sqrt(sample_size - 3)
                
                # Two-tailed p-value
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                p_values.append(p_value)
                
                if p_value < 0.05:  # 95% confidence level
                    significant_count += 1
        
        avg_p_value = np.mean(p_values) if p_values else 1.0
        
        return {
            'significant_correlations': significant_count,
            'total_correlations': n_correlations,
            'significance_ratio': significant_count / (n_correlations + 1e-8),
            'avg_p_value': float(avg_p_value),
            'critical_value': float(critical_value)
        }
    
    def _classify_market_regime(self, pca_results: Dict, correlation_matrix: np.ndarray, 
                              asset_data: pd.DataFrame) -> Dict[str, Any]:
        """Classify market regime based on correlation patterns"""
        if not self.parameters['regime_classification']:
            return {'regime': 'unknown', 'confidence': 0.0}
        
        # Market regime features
        dominant_eigenvalue = pca_results['dominant_eigenvalue']
        condition_number = pca_results['condition_number']
        first_pc_variance = pca_results['explained_variance_ratio'][0] if pca_results['explained_variance_ratio'] else 0
        
        # Correlation characteristics
        avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        max_correlation = np.max(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        min_correlation = np.min(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        
        # Volatility characteristics
        if len(asset_data) >= 20:
            recent_vol = asset_data['realized_volatility'].tail(20).mean()
            vol_of_vol = asset_data['realized_volatility'].tail(20).std()
        else:
            recent_vol = 0.1
            vol_of_vol = 0.05
        
        # Regime classification logic
        if first_pc_variance > 0.7 and avg_correlation > 0.6:
            regime = 'high_correlation'  # Crisis or strong trend
        elif avg_correlation < 0.2 and condition_number < 3:
            regime = 'low_correlation'   # Market differentiation
        elif recent_vol > 0.03 and vol_of_vol > 0.01:
            regime = 'high_volatility'   # Uncertain market
        elif recent_vol < 0.01 and avg_correlation < 0.3:
            regime = 'low_volatility'    # Calm market
        elif condition_number > 10:
            regime = 'unstable'          # Structural instability
        else:
            regime = 'normal'            # Normal market conditions
        
        # Calculate confidence based on feature clarity
        confidence = min(1.0, abs(first_pc_variance - 0.5) + abs(avg_correlation - 0.3) + 
                      abs(recent_vol - 0.02) * 10)
        
        return {
            'regime': regime,
            'confidence': float(confidence),
            'features': {
                'dominant_eigenvalue': float(dominant_eigenvalue),
                'condition_number': float(condition_number),
                'first_pc_variance': float(first_pc_variance),
                'avg_correlation': float(avg_correlation),
                'max_correlation': float(max_correlation),
                'min_correlation': float(min_correlation),
                'recent_volatility': float(recent_vol),
                'volatility_of_volatility': float(vol_of_vol)
            }
        }
    
    def _detect_correlation_signals(self, current_matrix: np.ndarray, asset_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect trading signals from correlation analysis"""
        signals = []
        signal_strength = 0.0
        
        if len(self.history['correlation_matrices']) == 0:
            return {'signals': signals, 'overall_strength': 0.0, 'signal_type': 'neutral'}
        
        # Compare with previous correlation matrix
        prev_matrix = np.array(self.history['correlation_matrices'][-1])
        
        if prev_matrix.shape == current_matrix.shape:
            # Calculate correlation change
            correlation_change = current_matrix - prev_matrix
            max_change = np.max(np.abs(correlation_change))
            
            # Significant correlation breakdown
            if max_change > 0.3:
                breakdown_locations = np.where(np.abs(correlation_change) > 0.3)
                for i, j in zip(breakdown_locations[0], breakdown_locations[1]):
                    if i != j:  # Ignore diagonal
                        change_val = correlation_change[i, j]
                        if change_val < -0.3:
                            signals.append({
                                'type': 'correlation_breakdown',
                                'assets': [i, j],
                                'magnitude': float(abs(change_val)),
                                'direction': 'negative'
                            })
                            signal_strength += abs(change_val)
                        elif change_val > 0.3:
                            signals.append({
                                'type': 'correlation_formation',
                                'assets': [i, j],
                                'magnitude': float(abs(change_val)),
                                'direction': 'positive'
                            })
                            signal_strength += abs(change_val) * 0.8  # Formation signals slightly weaker
            
            # Overall correlation regime shift
            current_avg_corr = np.mean(np.abs(current_matrix[np.triu_indices_from(current_matrix, k=1)]))
            prev_avg_corr = np.mean(np.abs(prev_matrix[np.triu_indices_from(prev_matrix, k=1)]))
            corr_shift = current_avg_corr - prev_avg_corr
            
            if abs(corr_shift) > 0.2:
                signals.append({
                    'type': 'regime_shift',
                    'magnitude': float(abs(corr_shift)),
                    'direction': 'increasing' if corr_shift > 0 else 'decreasing',
                    'from_correlation': float(prev_avg_corr),
                    'to_correlation': float(current_avg_corr)
                })
                signal_strength += abs(corr_shift)
        
        # Determine overall signal type
        breakdown_count = len([s for s in signals if s['type'] == 'correlation_breakdown'])
        formation_count = len([s for s in signals if s['type'] == 'correlation_formation'])
        
        if breakdown_count > formation_count:
            signal_type = 'bearish'  # Correlation breakdown often indicates stress
        elif formation_count > breakdown_count:
            signal_type = 'bullish'  # Correlation formation can indicate stability
        else:
            signal_type = 'neutral'
        
        return {
            'signals': signals,
            'overall_strength': min(1.0, signal_strength),
            'signal_type': signal_type,
            'signal_count': len(signals)
        }
    
    def _train_ml_models(self, asset_data: pd.DataFrame) -> bool:
        """Train ML models for correlation prediction"""
        if len(asset_data) < self.parameters['ml_lookback'] * 2:
            return False
        
        try:
            features, regime_targets, correlation_targets = self._prepare_ml_data(asset_data)
            if len(features) > 30:
                self.scaler.fit(features)
                scaled_features = self.scaler.transform(features)
                
                # Train regime classifier
                self.regime_classifier.fit(scaled_features, regime_targets)
                
                # Train correlation predictor
                self.correlation_predictor.fit(scaled_features, correlation_targets)
                
                self.models_trained = True
                return True
        except Exception:
            pass
        return False
    
    def _prepare_ml_data(self, asset_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare ML training data from historical correlation patterns"""
        features, regime_targets, correlation_targets = [], [], []
        lookback = 20
        
        for i in range(lookback, len(asset_data) - 10):
            window_data = asset_data.iloc[i-lookback:i]
            
            if len(window_data) < lookback:
                continue
            
            # Calculate correlation matrix for window
            corr_matrix = window_data.corr().values
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            # Extract features from correlation matrix
            eigenvals, _ = eigh(corr_matrix)
            eigenvals = np.sort(eigenvals)[::-1]
            
            # Correlation features
            avg_corr = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            max_corr = np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            min_corr = np.min(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
            
            # Eigenvalue features
            dominant_eigenval = eigenvals[0] if len(eigenvals) > 0 else 1.0
            condition_number = dominant_eigenval / (eigenvals[-1] + 1e-8) if len(eigenvals) > 0 else 1.0
            
            # Market features
            volatility = window_data['realized_volatility'].mean()
            volume_trend = np.polyfit(range(len(window_data)), window_data['volume'].values, 1)[0]
            price_trend = np.polyfit(range(len(window_data)), window_data['close_price'].values, 1)[0]
            
            feature_vector = [
                avg_corr, max_corr, min_corr,
                dominant_eigenval, condition_number,
                volatility, volume_trend, price_trend
            ]
            
            # Add top eigenvalues as features
            n_eigen_features = min(3, len(eigenvals))
            for j in range(n_eigen_features):
                feature_vector.append(eigenvals[j])
            
            # Pad with zeros if needed
            while len(feature_vector) < 11:
                feature_vector.append(0.0)
            
            # Future targets
            future_data = asset_data.iloc[i+1:i+11]
            if len(future_data) >= 5:
                future_corr_matrix = future_data.corr().values
                future_corr_matrix = np.nan_to_num(future_corr_matrix, nan=0.0)
                
                future_avg_corr = np.mean(np.abs(future_corr_matrix[np.triu_indices_from(future_corr_matrix, k=1)]))
                future_volatility = future_data['realized_volatility'].mean()
                
                # Regime target
                if future_avg_corr > 0.6 and future_volatility > 0.03:
                    regime_target = 0  # Crisis
                elif future_avg_corr < 0.2:
                    regime_target = 1  # Differentiation
                elif future_volatility > 0.03:
                    regime_target = 2  # High volatility
                else:
                    regime_target = 3  # Normal
                
                # Correlation target
                correlation_target = future_avg_corr - avg_corr
            else:
                regime_target = 3
                correlation_target = 0.0
            
            features.append(feature_vector)
            regime_targets.append(regime_target)
            correlation_targets.append(correlation_target)
        
        return np.array(features), np.array(regime_targets), np.array(correlation_targets)
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation matrix analysis with comprehensive insights"""
        try:
            # Create multi-asset data matrix
            asset_data = self._create_asset_matrix(data)
            asset_names = asset_data.columns.tolist()
            
            # Adapt correlation window
            correlation_window = self._adapt_correlation_window(asset_data)
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(asset_data, correlation_window)
            
            # PCA analysis
            pca_results = self._perform_pca_analysis(correlation_matrix)
            
            # Clustering analysis
            cluster_results = self._detect_correlation_clusters(correlation_matrix, asset_names)
            
            # Statistical significance testing
            significance_results = self._test_correlation_significance(correlation_matrix, correlation_window)
            
            # Market regime classification
            regime_results = self._classify_market_regime(pca_results, correlation_matrix, asset_data)
            
            # Signal detection
            signal_results = self._detect_correlation_signals(correlation_matrix, asset_data)
            
            # Train ML models
            if not self.models_trained:
                self._train_ml_models(asset_data)
            
            # Generate overall signal
            signal, confidence = self._generate_correlation_signal(
                correlation_matrix, pca_results, regime_results, signal_results
            )
            
            # Update history
            self.history['correlation_matrices'].append(correlation_matrix.tolist())
            self.history['eigenvalues'].append(pca_results['eigenvalues'])
            self.history['pca_components'].append(pca_results['principal_components'])
            self.history['cluster_labels'].append(cluster_results.get('cluster_labels', []))
            self.history['regime_predictions'].append(regime_results)
            
            # Keep history limited
            for key in self.history:
                if len(self.history[key]) > 50:
                    self.history[key] = self.history[key][-50:]
            
            result = {
                'correlation_matrix': correlation_matrix.tolist(),
                'pca_analysis': pca_results,
                'cluster_analysis': cluster_results,
                'significance_analysis': significance_results,
                'regime_analysis': regime_results,
                'signal_analysis': signal_results,
                'signal': signal,
                'confidence': confidence,
                'asset_names': asset_names,
                'correlation_window': correlation_window,
                'market_structure': self._analyze_market_structure(correlation_matrix, pca_results),
                'risk_metrics': self._calculate_risk_metrics(correlation_matrix, asset_data)
            }
            
            return result
            
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="raw_calculation",
                message=f"Failed to calculate Correlation Matrix: {str(e)}",
                cause=e
            )
    
    def _generate_correlation_signal(self, correlation_matrix: np.ndarray, pca_results: Dict,
                                   regime_results: Dict, signal_results: Dict) -> Tuple[SignalType, float]:
        """Generate comprehensive correlation-based signal"""
        signal_components = []
        confidence_components = []
        
        # Regime-based signals
        regime = regime_results['regime']
        regime_confidence = regime_results['confidence']
        
        if regime == 'high_correlation' and regime_confidence > 0.7:
            signal_components.append(-0.7)  # High correlation often bearish (crisis)
            confidence_components.append(regime_confidence)
        elif regime == 'low_correlation' and regime_confidence > 0.7:
            signal_components.append(0.5)   # Low correlation can be bullish (differentiation)
            confidence_components.append(regime_confidence)
        elif regime == 'high_volatility':
            signal_components.append(-0.4)  # High volatility generally bearish
            confidence_components.append(regime_confidence)
        
        # Signal analysis based signals
        signal_type = signal_results['signal_type']
        signal_strength = signal_results['overall_strength']
        
        if signal_type == 'bearish' and signal_strength > 0.3:
            signal_components.append(-signal_strength)
            confidence_components.append(signal_strength)
        elif signal_type == 'bullish' and signal_strength > 0.3:
            signal_components.append(signal_strength)
            confidence_components.append(signal_strength)
        
        # PCA-based signals
        first_pc_variance = pca_results['explained_variance_ratio'][0] if pca_results['explained_variance_ratio'] else 0
        condition_number = pca_results['condition_number']
        
        if first_pc_variance > 0.8:  # High concentration risk
            signal_components.append(-0.6)
            confidence_components.append(0.8)
        elif condition_number > 20:  # High instability
            signal_components.append(-0.5)
            confidence_components.append(0.7)
        
        # Correlation level signals
        avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        
        if avg_correlation > 0.8:
            signal_components.append(-0.8)  # Very high correlation - crisis signal
            confidence_components.append(0.9)
        elif avg_correlation < 0.1:
            signal_components.append(0.3)   # Very low correlation - might indicate opportunities
            confidence_components.append(0.6)
        
        # Calculate final signal
        if signal_components and confidence_components:
            weighted_signal = np.average(signal_components, weights=confidence_components)
            avg_confidence = np.mean(confidence_components)
        else:
            weighted_signal = 0.0
            avg_confidence = 0.0
        
        if weighted_signal > 0.6:
            signal = SignalType.STRONG_BUY if weighted_signal > 0.8 else SignalType.BUY
        elif weighted_signal < -0.6:
            signal = SignalType.STRONG_SELL if weighted_signal < -0.8 else SignalType.SELL
        else:
            signal = SignalType.NEUTRAL
        
        return signal, min(avg_confidence, 1.0)
    
    def _analyze_market_structure(self, correlation_matrix: np.ndarray, pca_results: Dict) -> Dict[str, Any]:
        """Analyze overall market structure from correlation patterns"""
        # Market concentration
        eigenvalues = pca_results['eigenvalues']
        concentration = eigenvalues[0] / sum(eigenvalues) if eigenvalues else 0
        
        # Market connectivity
        avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        
        # Market stability
        condition_number = pca_results['condition_number']
        stability = 1.0 / (1.0 + np.log(condition_number)) if condition_number > 1 else 1.0
        
        # Structure classification
        if concentration > 0.7 and avg_correlation > 0.6:
            structure = 'highly_integrated'
        elif concentration < 0.3 and avg_correlation < 0.2:
            structure = 'fragmented'
        elif stability < 0.3:
            structure = 'unstable'
        else:
            structure = 'normal'
        
        return {
            'structure_type': structure,
            'concentration': float(concentration),
            'connectivity': float(avg_correlation),
            'stability': float(stability),
            'integration_level': 'high' if avg_correlation > 0.5 else 'medium' if avg_correlation > 0.2 else 'low'
        }
    
    def _calculate_risk_metrics(self, correlation_matrix: np.ndarray, asset_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics from correlation analysis"""
        if not self.parameters['risk_adjustment']:
            return {'diversification_ratio': 1.0, 'concentration_risk': 0.0}
        
        # Diversification ratio
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        eigenvalues = np.real(eigenvalues[eigenvalues > 1e-8])
        
        if len(eigenvalues) > 0:
            effective_rank = np.exp(-np.sum(eigenvalues * np.log(eigenvalues + 1e-8)))
            diversification_ratio = effective_rank / len(eigenvalues)
        else:
            diversification_ratio = 1.0
        
        # Concentration risk
        max_eigenvalue = np.max(eigenvalues) if len(eigenvalues) > 0 else 1.0
        concentration_risk = max_eigenvalue / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 0.0
        
        # Systemic risk indicator
        avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        systemic_risk = avg_correlation ** 2  # Higher correlations indicate higher systemic risk
        
        return {
            'diversification_ratio': float(diversification_ratio),
            'concentration_risk': float(concentration_risk),
            'systemic_risk': float(systemic_risk),
            'effective_rank': float(effective_rank) if 'effective_rank' in locals() else 1.0
        }
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        if isinstance(value, dict) and 'signal' in value and 'confidence' in value:
            return value['signal'], value['confidence']
        return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        base_metadata = super()._get_metadata(data)
        base_metadata.update({
            'indicator_type': 'momentum',
            'indicator_category': 'correlation_matrix',
            'models_trained': self.models_trained,
            'correlation_window': self.parameters['correlation_window'],
            'pca_components': self.parameters['pca_components'],
            'clustering_enabled': self.parameters['clustering_enabled'],
            'regime_classification': self.parameters['regime_classification'],
            'adaptive_window': self.parameters['adaptive_window'],
            'volatility_weighted': self.parameters['volatility_weighted'],
            'data_type': 'ohlcv',
            'complexity': 'advanced'
        })
        return base_metadata