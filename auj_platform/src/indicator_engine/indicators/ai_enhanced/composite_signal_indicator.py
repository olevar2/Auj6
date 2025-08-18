"""
Composite Signal Indicator - AI Enhanced Category
================================================

Advanced AI-enhanced composite signal aggregator with multi-indicator fusion,
machine learning classification, and dynamic weighting algorithms.
"""

from typing import Dict, Any, Union, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType


class CompositeSignalIndicator(StandardIndicatorInterface):
    """
    Advanced Composite Signal Indicator using AI-enhanced signal fusion.
    
    Features:
    - Multi-indicator signal aggregation with dynamic weighting
    - Machine learning-based pattern recognition
    - Hierarchical clustering for signal grouping
    - PCA-based dimensionality reduction
    - Anomaly detection for signal validation
    - Adaptive confidence scoring
    - Real-time signal strength assessment
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'lookback_period': 21,
            'min_confidence': 0.6,
            'max_signals': 12,
            'pca_components': 6,
            'clustering_threshold': 0.7,
            'anomaly_contamination': 0.1,
            'ml_window': 200,
            'signal_decay': 0.95,
            'adaptive_weights': True,
            'use_clustering': True,
            'enable_anomaly_detection': True
        }
        if parameters:
            default_params.update(parameters)
        super().__init__("CompositeSignalIndicator", default_params)
        
        # Initialize ML models
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=8
        )
        self.isolation_forest = IsolationForest(
            contamination=self.parameters['anomaly_contamination'],
            random_state=42
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.parameters['pca_components'])
        
        # Signal tracking
        self.signal_history = []
        self.weight_history = []
        self.confidence_history = []
        self.is_fitted = False
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["open", "high", "low", "close", "volume"],
            min_periods=max(50, self.parameters['ml_window'])
        )
    
    def _calculate_technical_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various technical indicators for signal composition."""
        signals = pd.DataFrame(index=data.index)
        
        # Price-based signals
        signals['rsi'] = self._calculate_rsi(data['close'])
        signals['macd_signal'] = self._calculate_macd_signal(data['close'])
        signals['bb_position'] = self._calculate_bb_position(data['close'])
        signals['stoch'] = self._calculate_stochastic(data)
        
        # Momentum signals
        signals['momentum'] = data['close'].pct_change(self.parameters['lookback_period'])
        signals['roc'] = self._calculate_roc(data['close'])
        
        # Volume signals
        signals['obv'] = self._calculate_obv(data)
        signals['volume_sma_ratio'] = data['volume'] / data['volume'].rolling(
            self.parameters['lookback_period']
        ).mean()
        
        # Volatility signals
        signals['atr_position'] = self._calculate_atr_position(data)
        signals['volatility'] = data['close'].rolling(
            self.parameters['lookback_period']
        ).std() / data['close'].rolling(self.parameters['lookback_period']).mean()
        
        # Pattern signals
        signals['candlestick_pattern'] = self._detect_candlestick_patterns(data)
        signals['trend_strength'] = self._calculate_trend_strength(data['close'])
        
        return signals.fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd_signal(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD signal line."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return (macd - signal) / prices  # Normalized
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_bb = sma + (2 * std)
        lower_bb = sma - (2 * std)
        return (prices - lower_bb) / (upper_bb - lower_bb)
    
    def _calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic oscillator."""
        low_min = data['low'].rolling(period).min()
        high_max = data['high'].rolling(period).max()
        return (data['close'] - low_min) / (high_max - low_min) * 100
    
    def _calculate_roc(self, prices: pd.Series, period: int = 12) -> pd.Series:
        """Calculate Rate of Change."""
        return (prices / prices.shift(period) - 1) * 100
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        return obv / obv.rolling(self.parameters['lookback_period']).std()
    
    def _calculate_atr_position(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate position relative to Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        return (data['close'] - data['close'].shift()) / atr
    
    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> pd.Series:
        """Detect basic candlestick patterns."""
        body = data['close'] - data['open']
        body_size = np.abs(body)
        range_size = data['high'] - data['low']
        
        # Doji pattern (small body relative to range)
        doji = body_size / range_size < 0.1
        
        # Hammer pattern
        hammer = (
            (data['close'] > data['open']) &
            ((data['open'] - data['low']) > 2 * body_size) &
            ((data['high'] - data['close']) < 0.3 * body_size)
        )
        
        # Combine patterns with weights
        pattern_score = doji.astype(float) * 0.5 + hammer.astype(float) * 1.0
        return pattern_score
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression slope."""
        def calc_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            slope, _, r_value, _, _ = stats.linregress(x, series)
            return slope * r_value ** 2  # Weighted by R-squared
        
        return prices.rolling(period).apply(calc_slope)
    
    def _cluster_signals(self, signals: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, float]]:
        """Perform hierarchical clustering on signals."""
        if len(signals.columns) < 3:
            return np.zeros(len(signals.columns)), {0: 1.0}
        
        # Calculate correlation distance matrix
        corr_matrix = signals.corr().fillna(0)
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Perform hierarchical clustering
        condensed_distances = pdist(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        # Form clusters
        n_clusters = min(4, len(signals.columns) // 2 + 1)
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate cluster weights based on silhouette score
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(distance_matrix, cluster_labels)
            cluster_weights = {i: silhouette_avg for i in set(cluster_labels)}
        else:
            cluster_weights = {1: 1.0}
        
        return cluster_labels, cluster_weights
    
    def _apply_pca_reduction(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA dimensionality reduction to signals."""
        if signals.shape[1] <= self.parameters['pca_components']:
            return signals
        
        # Standardize signals
        signals_scaled = self.scaler.fit_transform(signals.fillna(0))
        
        # Apply PCA
        signals_pca = self.pca.fit_transform(signals_scaled)
        
        # Create DataFrame with PCA components
        pca_df = pd.DataFrame(
            signals_pca,
            index=signals.index,
            columns=[f'PC{i+1}' for i in range(signals_pca.shape[1])]
        )
        
        return pca_df
    
    def _detect_anomalies(self, signals: pd.DataFrame) -> pd.Series:
        """Detect anomalous signals using Isolation Forest."""
        if not self.parameters['enable_anomaly_detection']:
            return pd.Series(np.ones(len(signals)), index=signals.index)
        
        # Fit isolation forest
        signals_clean = signals.fillna(0).replace([np.inf, -np.inf], 0)
        anomaly_scores = self.isolation_forest.fit_predict(signals_clean)
        
        # Convert to confidence scores (1 = normal, 0 = anomaly)
        confidence = pd.Series(
            (anomaly_scores + 1) / 2,  # Convert {-1, 1} to {0, 1}
            index=signals.index
        )
        
        return confidence
    
    def _calculate_adaptive_weights(self, signals: pd.DataFrame) -> pd.Series:
        """Calculate adaptive weights based on signal performance."""
        if not self.parameters['adaptive_weights']:
            return pd.Series(np.ones(len(signals.columns)) / len(signals.columns))
        
        # Calculate signal volatility (lower volatility = higher weight)
        signal_volatility = signals.std()
        volatility_weights = 1 / (1 + signal_volatility)
        volatility_weights = volatility_weights / volatility_weights.sum()
        
        # Calculate signal correlation (lower average correlation = higher weight)
        corr_matrix = signals.corr().fillna(0)
        avg_correlation = corr_matrix.abs().mean()
        correlation_weights = 1 / (1 + avg_correlation)
        correlation_weights = correlation_weights / correlation_weights.sum()
        
        # Combine weights
        combined_weights = (volatility_weights + correlation_weights) / 2
        return combined_weights
    
    def _train_ml_classifier(self, signals: pd.DataFrame, target: pd.Series) -> None:
        """Train machine learning classifier for signal prediction."""
        if len(signals) < self.parameters['ml_window']:
            return
        
        # Prepare features
        features = signals.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Create target labels (1 for positive returns, 0 for negative)
        labels = (target > 0).astype(int)
        
        # Train classifier
        try:
            self.rf_classifier.fit(features, labels)
            self.is_fitted = True
        except Exception as e:
            self.logger.warning(f"Failed to train ML classifier: {e}")
    
    def calculate_raw(self, data: pd.DataFrame) -> Union[float, int, Dict[str, Any]]:
        """
        Calculate composite signal with advanced AI-enhanced algorithms.
        
        Returns:
            Dict containing:
            - composite_signal: Main signal value (-1 to 1)
            - confidence: Signal confidence (0 to 1)
            - component_signals: Individual signal contributions
            - anomaly_score: Anomaly detection confidence
            - cluster_info: Signal clustering information
        """
        try:
            if len(data) < self.parameters['lookback_period']:
                return {'composite_signal': 0.0, 'confidence': 0.0}
            
            # Calculate technical signals
            signals = self._calculate_technical_signals(data)
            
            if signals.empty:
                return {'composite_signal': 0.0, 'confidence': 0.0}
            
            # Apply PCA reduction if enabled
            if self.parameters['pca_components'] < len(signals.columns):
                signals_processed = self._apply_pca_reduction(signals)
            else:
                signals_processed = signals
            
            # Detect anomalies
            anomaly_confidence = self._detect_anomalies(signals_processed)
            
            # Perform signal clustering
            if self.parameters['use_clustering']:
                cluster_labels, cluster_weights = self._cluster_signals(signals_processed)
            else:
                cluster_labels = np.ones(len(signals_processed.columns))
                cluster_weights = {1: 1.0}
            
            # Calculate adaptive weights
            adaptive_weights = self._calculate_adaptive_weights(signals_processed)
            
            # Calculate weighted composite signal
            latest_signals = signals_processed.iloc[-1]
            weighted_signal = (latest_signals * adaptive_weights).sum()
            
            # Normalize to [-1, 1] range
            if signals_processed.std().mean() > 0:
                composite_signal = np.tanh(weighted_signal / signals_processed.std().mean())
            else:
                composite_signal = 0.0
            
            # Calculate confidence based on signal consensus
            signal_std = latest_signals.std()
            signal_mean = np.abs(latest_signals.mean())
            base_confidence = signal_mean / (signal_mean + signal_std + 1e-6)
            
            # Adjust confidence with anomaly detection
            anomaly_factor = anomaly_confidence.iloc[-1]
            final_confidence = base_confidence * anomaly_factor
            
            # Train ML classifier for future predictions
            if len(data) >= self.parameters['ml_window']:
                future_returns = data['close'].pct_change().shift(-1)
                self._train_ml_classifier(signals_processed, future_returns)
            
            # ML prediction if model is fitted
            ml_prediction = 0.0
            if self.is_fitted:
                try:
                    latest_features = signals_processed.iloc[-1:].fillna(0)
                    ml_proba = self.rf_classifier.predict_proba(latest_features)[0]
                    ml_prediction = ml_proba[1] - ml_proba[0]  # Convert to [-1, 1]
                except Exception as e:
                    self.logger.warning(f"ML prediction failed: {e}")
            
            # Combine signals
            final_signal = (composite_signal * 0.7 + ml_prediction * 0.3)
            
            # Update history
            self.signal_history.append(final_signal)
            self.confidence_history.append(final_confidence)
            self.weight_history.append(adaptive_weights.to_dict())
            
            # Trim history
            max_history = self.parameters['lookback_period'] * 2
            if len(self.signal_history) > max_history:
                self.signal_history = self.signal_history[-max_history:]
                self.confidence_history = self.confidence_history[-max_history:]
                self.weight_history = self.weight_history[-max_history:]
            
            return {
                'composite_signal': float(np.clip(final_signal, -1, 1)),
                'confidence': float(np.clip(final_confidence, 0, 1)),
                'component_signals': latest_signals.to_dict(),
                'anomaly_score': float(anomaly_factor),
                'cluster_info': {
                    'labels': cluster_labels.tolist(),
                    'weights': cluster_weights
                },
                'ml_prediction': float(ml_prediction),
                'adaptive_weights': adaptive_weights.to_dict(),
                'signal_strength': float(np.abs(final_signal)),
                'market_regime': self._detect_market_regime(data),
                'consensus_score': float(base_confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error in CompositeSignalIndicator calculation: {e}")
            return {
                'composite_signal': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime (trending, ranging, volatile)."""
        try:
            recent_data = data.tail(self.parameters['lookback_period'])
            
            # Calculate trend strength
            returns = recent_data['close'].pct_change().dropna()
            trend_strength = np.abs(returns.mean()) / (returns.std() + 1e-6)
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate range-bound behavior
            price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()
            
            if trend_strength > 1.5:
                return "trending"
            elif volatility > 0.3:
                return "volatile"
            elif price_range < 0.1:
                return "ranging"
            else:
                return "mixed"
                
        except Exception:
            return "unknown"
