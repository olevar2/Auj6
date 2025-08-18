"""
Block Trade Signal - Institutional Flow Detection System
=====================================================

Advanced block trade detection using volume clustering, flow analysis,
and machine learning to identify institutional order patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class BlockTradeSignal(StandardIndicatorInterface):
    """
    AI-Enhanced Block Trade Signal with institutional flow detection.
    
    Features:
    - Dynamic volume threshold detection
    - Institutional order pattern recognition
    - Smart money flow analysis
    - Block trade clustering and classification
    - Market impact assessment
    - Follow-through momentum prediction
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'volume_multiplier': 2.0,      # Volume threshold multiplier
            'block_size_percentile': 90,   # Percentile for block size definition
            'clustering_periods': 50,      # Periods for volume clustering
            'impact_window': 10,           # Window to measure market impact
            'momentum_window': 20,         # Window for momentum analysis
            'smart_money_threshold': 0.7,  # Threshold for smart money detection
            'pattern_recognition': True,
            'flow_analysis': True,
            'impact_prediction': True,
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("BlockTradeSignal", default_params)
        
        # Analysis components
        self.volume_clusters = None
        self.scaler = StandardScaler()
        self.block_history = []
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.TICK,
            required_columns=['price', 'volume', 'timestamp', 'side'],
            min_periods=self.parameters['clustering_periods'],
            lookback_periods=200
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive block trade analysis.
        """
        try:
            prices = data['price'].values
            volumes = data['volume'].values
            timestamps = data['timestamp'].values if 'timestamp' in data.columns else np.arange(len(data))
            sides = data['side'].values if 'side' in data.columns else np.ones(len(data))
            
            # Dynamic volume threshold detection
            volume_threshold = self._calculate_dynamic_threshold(volumes)
            
            # Identify block trades
            block_trades = self._identify_block_trades(volumes, prices, sides, volume_threshold)
            
            # Volume clustering analysis
            cluster_analysis = self._analyze_volume_clusters(volumes)
            
            # Institutional pattern recognition
            institutional_patterns = self._detect_institutional_patterns(block_trades, prices, volumes) if self.parameters['pattern_recognition'] else {}
            
            # Smart money flow analysis
            flow_analysis = self._analyze_smart_money_flow(block_trades, prices, sides) if self.parameters['flow_analysis'] else {}
            
            # Market impact assessment
            impact_analysis = self._assess_market_impact(block_trades, prices) if self.parameters['impact_prediction'] else {}
            
            # Block trade momentum
            momentum_analysis = self._analyze_block_momentum(block_trades, prices)
            
            # Signal generation
            signal_strength = self._calculate_signal_strength(
                block_trades, institutional_patterns, flow_analysis, 
                impact_analysis, momentum_analysis
            )
            
            return {
                'volume_threshold': volume_threshold,
                'block_trades': block_trades,
                'cluster_analysis': cluster_analysis,
                'institutional_patterns': institutional_patterns,
                'flow_analysis': flow_analysis,
                'impact_analysis': impact_analysis,
                'momentum_analysis': momentum_analysis,
                'signal_strength': signal_strength,
                'block_frequency': self._calculate_block_frequency(block_trades),
                'institutional_score': self._calculate_institutional_score(block_trades, flow_analysis)
            }
            
        except Exception as e:
            raise Exception(f"BlockTradeSignal calculation failed: {str(e)}")
    
    def _calculate_dynamic_threshold(self, volumes: np.ndarray) -> float:
        """Calculate dynamic volume threshold for block trade identification."""
        if len(volumes) < 20:
            return np.percentile(volumes, self.parameters['block_size_percentile']) if len(volumes) > 0 else 0
        
        # Use multiple methods to determine threshold
        
        # Method 1: Statistical percentile
        percentile_threshold = np.percentile(volumes, self.parameters['block_size_percentile'])
        
        # Method 2: Standard deviation method
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        std_threshold = mean_volume + (self.parameters['volume_multiplier'] * std_volume)
        
        # Method 3: Median absolute deviation (more robust)
        median_volume = np.median(volumes)
        mad = np.median(np.abs(volumes - median_volume))
        mad_threshold = median_volume + (self.parameters['volume_multiplier'] * mad * 1.4826)
        
        # Combine methods (weighted average)
        dynamic_threshold = (percentile_threshold * 0.5 + std_threshold * 0.3 + mad_threshold * 0.2)
        
        return float(dynamic_threshold)
    
    def _identify_block_trades(self, volumes: np.ndarray, prices: np.ndarray, 
                             sides: np.ndarray, threshold: float) -> Dict[str, Any]:
        """Identify and analyze block trades."""
        if len(volumes) == 0:
            return {}
        
        # Find block trades
        block_indices = np.where(volumes >= threshold)[0]
        
        if len(block_indices) == 0:
            return {
                'count': 0,
                'recent_blocks': [],
                'total_volume': 0,
                'avg_size': 0,
                'buy_sell_ratio': 0.5
            }
        
        # Analyze recent block trades (last 20)
        recent_blocks = []
        total_block_volume = 0
        buy_volume = 0
        sell_volume = 0
        
        for idx in block_indices[-20:]:  # Last 20 blocks
            block_info = {
                'index': int(idx),
                'volume': float(volumes[idx]),
                'price': float(prices[idx]),
                'side': int(sides[idx]) if idx < len(sides) else 1,
                'relative_size': float(volumes[idx] / threshold)
            }
            recent_blocks.append(block_info)
            total_block_volume += volumes[idx]
            
            if block_info['side'] > 0:
                buy_volume += volumes[idx]
            else:
                sell_volume += volumes[idx]
        
        # Calculate metrics
        avg_block_size = total_block_volume / len(recent_blocks) if recent_blocks else 0
        buy_sell_ratio = buy_volume / (buy_volume + sell_volume + 1e-8)
        
        return {
            'count': len(block_indices),
            'recent_count': len(recent_blocks),
            'recent_blocks': recent_blocks,
            'total_volume': float(total_block_volume),
            'avg_size': float(avg_block_size),
            'buy_sell_ratio': float(buy_sell_ratio),
            'largest_block': float(np.max(volumes[block_indices])),
            'frequency': len(block_indices) / len(volumes)
        }
    
    def _analyze_volume_clusters(self, volumes: np.ndarray) -> Dict[str, Any]:
        """Analyze volume clustering patterns."""
        if len(volumes) < self.parameters['clustering_periods']:
            return {}
        
        try:
            # Prepare volume data for clustering
            recent_volumes = volumes[-self.parameters['clustering_periods']:]
            volume_features = self._extract_volume_features(recent_volumes)
            
            if len(volume_features) < 5:
                return {}
            
            # Perform K-means clustering
            n_clusters = min(5, len(volume_features) // 3)
            if n_clusters < 2:
                return {}
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            scaled_features = self.scaler.fit_transform(volume_features)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Analyze clusters
            cluster_centers = self.scaler.inverse_transform(kmeans.cluster_centers_)
            
            clusters = []
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_volumes = np.array(volume_features)[cluster_mask]
                
                if len(cluster_volumes) > 0:
                    cluster_info = {
                        'id': i,
                        'size': len(cluster_volumes),
                        'avg_volume': float(np.mean(cluster_volumes[:, 0])),  # First feature is volume
                        'center': cluster_centers[i].tolist(),
                        'density': len(cluster_volumes) / len(volume_features)
                    }
                    clusters.append(cluster_info)
            
            # Identify dominant cluster (highest density)
            dominant_cluster = max(clusters, key=lambda x: x['density']) if clusters else None
            
            # Current volume cluster assignment
            if len(volume_features) > 0:
                current_features = volume_features[-1:].reshape(1, -1)
                current_scaled = self.scaler.transform(current_features)
                current_cluster = kmeans.predict(current_scaled)[0]
            else:
                current_cluster = 0
            
            return {
                'clusters': clusters,
                'dominant_cluster': dominant_cluster,
                'current_cluster': int(current_cluster),
                'cluster_diversity': len(clusters) / n_clusters if n_clusters > 0 else 0,
                'volume_regime': self._classify_volume_regime(clusters, current_cluster)
            }
            
        except Exception:
            return {}
    
    def _extract_volume_features(self, volumes: np.ndarray) -> np.ndarray:
        """Extract features from volume data for clustering."""
        features = []
        window = 5
        
        for i in range(window, len(volumes)):
            window_volumes = volumes[i-window:i+1]
            
            feature_vector = [
                np.mean(window_volumes),           # Average volume
                np.std(window_volumes),            # Volume volatility
                np.max(window_volumes),            # Peak volume
                window_volumes[-1],                # Current volume
                np.sum(window_volumes),            # Total volume
                len([v for v in window_volumes if v > np.mean(window_volumes)]),  # Above-average count
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _classify_volume_regime(self, clusters: list, current_cluster: int) -> str:
        """Classify current volume regime."""
        if not clusters:
            return 'normal'
        
        current_cluster_info = None
        for cluster in clusters:
            if cluster['id'] == current_cluster:
                current_cluster_info = cluster
                break
        
        if not current_cluster_info:
            return 'normal'
        
        avg_volume = current_cluster_info['avg_volume']
        density = current_cluster_info['density']
        
        # Classify based on volume level and cluster density
        if avg_volume > np.percentile([c['avg_volume'] for c in clusters], 80):
            if density > 0.3:
                return 'high_volume_sustained'
            else:
                return 'high_volume_spike'
        elif avg_volume < np.percentile([c['avg_volume'] for c in clusters], 20):
            return 'low_volume'
        else:
            return 'normal_volume'
    
    def _detect_institutional_patterns(self, block_trades: Dict, prices: np.ndarray, 
                                     volumes: np.ndarray) -> Dict[str, Any]:
        """Detect institutional trading patterns."""
        if not block_trades or block_trades['recent_count'] == 0:
            return {}
        
        recent_blocks = block_trades['recent_blocks']
        
        # Pattern 1: Volume-Weighted Average Price (VWAP) trading
        vwap_pattern = self._detect_vwap_pattern(recent_blocks, prices)
        
        # Pattern 2: Iceberg orders (repeated similar-sized blocks)
        iceberg_pattern = self._detect_iceberg_pattern(recent_blocks)
        
        # Pattern 3: Stealth trading (gradually increasing volume)
        stealth_pattern = self._detect_stealth_pattern(recent_blocks)
        
        # Pattern 4: Program trading (regular intervals)
        program_pattern = self._detect_program_pattern(recent_blocks)
        
        # Overall institutional score
        pattern_scores = [
            vwap_pattern.get('score', 0),
            iceberg_pattern.get('score', 0),
            stealth_pattern.get('score', 0),
            program_pattern.get('score', 0)
        ]
        
        institutional_score = np.mean(pattern_scores)
        
        return {
            'vwap_pattern': vwap_pattern,
            'iceberg_pattern': iceberg_pattern,
            'stealth_pattern': stealth_pattern,
            'program_pattern': program_pattern,
            'institutional_score': float(institutional_score),
            'dominant_pattern': self._identify_dominant_pattern(pattern_scores)
        }
    
    def _detect_vwap_pattern(self, blocks: list, prices: np.ndarray) -> Dict[str, Any]:
        """Detect VWAP-based institutional trading."""
        if len(blocks) < 3 or len(prices) < 20:
            return {'detected': False, 'score': 0}
        
        # Calculate VWAP
        recent_prices = prices[-20:]
        weights = np.linspace(0.5, 1.0, len(recent_prices))  # Simple weight approximation
        vwap = np.average(recent_prices, weights=weights)
        
        # Check if blocks are executed near VWAP
        vwap_trades = 0
        for block in blocks[-10:]:  # Last 10 blocks
            price_deviation = abs(block['price'] - vwap) / vwap
            if price_deviation < 0.005:  # Within 0.5% of VWAP
                vwap_trades += 1
        
        vwap_ratio = vwap_trades / min(len(blocks), 10)
        
        return {
            'detected': vwap_ratio > 0.6,
            'score': float(vwap_ratio),
            'vwap_level': float(vwap),
            'vwap_trades_ratio': float(vwap_ratio)
        }
    
    def _detect_iceberg_pattern(self, blocks: list) -> Dict[str, Any]:
        """Detect iceberg order patterns."""
        if len(blocks) < 5:
            return {'detected': False, 'score': 0}
        
        # Look for repeated similar-sized blocks
        block_sizes = [block['volume'] for block in blocks[-10:]]
        
        # Group similar sizes (within 20% variance)
        size_groups = {}
        for size in block_sizes:
            found_group = False
            for group_size in size_groups:
                if abs(size - group_size) / group_size < 0.2:
                    size_groups[group_size].append(size)
                    found_group = True
                    break
            if not found_group:
                size_groups[size] = [size]
        
        # Find largest group
        largest_group = max(size_groups.values(), key=len) if size_groups else []
        iceberg_score = len(largest_group) / len(block_sizes) if block_sizes else 0
        
        return {
            'detected': iceberg_score > 0.4 and len(largest_group) >= 3,
            'score': float(iceberg_score),
            'repeated_size': float(np.mean(largest_group)) if largest_group else 0,
            'repetition_count': len(largest_group)
        }
    
    def _detect_stealth_pattern(self, blocks: list) -> Dict[str, Any]:
        """Detect stealth trading patterns."""
        if len(blocks) < 5:
            return {'detected': False, 'score': 0}
        
        # Look for gradually increasing volume
        block_sizes = [block['volume'] for block in blocks[-10:]]
        
        if len(block_sizes) < 3:
            return {'detected': False, 'score': 0}
        
        # Calculate trend in block sizes
        x = np.arange(len(block_sizes))
        slope, _ = np.polyfit(x, block_sizes, 1)
        
        # Normalize slope
        avg_size = np.mean(block_sizes)
        normalized_slope = slope / (avg_size + 1e-8)
        
        # Check for consistent increase
        increases = sum(1 for i in range(1, len(block_sizes)) if block_sizes[i] > block_sizes[i-1])
        consistency = increases / (len(block_sizes) - 1) if len(block_sizes) > 1 else 0
        
        stealth_score = min(normalized_slope * 10, 1.0) * consistency
        
        return {
            'detected': stealth_score > 0.3 and consistency > 0.6,
            'score': float(max(0, stealth_score)),
            'volume_trend': float(normalized_slope),
            'consistency': float(consistency)
        }
    
    def _detect_program_pattern(self, blocks: list) -> Dict[str, Any]:
        """Detect algorithmic/program trading patterns."""
        if len(blocks) < 5:
            return {'detected': False, 'score': 0}
        
        # Look for regular timing intervals
        block_intervals = []
        for i in range(1, len(blocks)):
            interval = blocks[i]['index'] - blocks[i-1]['index']
            block_intervals.append(interval)
        
        if not block_intervals:
            return {'detected': False, 'score': 0}
        
        # Check for regularity in intervals
        interval_std = np.std(block_intervals)
        interval_mean = np.mean(block_intervals)
        
        # Coefficient of variation (lower is more regular)
        cv = interval_std / (interval_mean + 1e-8)
        
        # Regularity score (inverse of coefficient of variation)
        regularity_score = 1.0 / (1.0 + cv)
        
        return {
            'detected': regularity_score > 0.7 and len(block_intervals) >= 4,
            'score': float(regularity_score),
            'avg_interval': float(interval_mean),
            'interval_consistency': float(1 - cv) if cv < 1 else 0
        }
    
    def _identify_dominant_pattern(self, pattern_scores: list) -> str:
        """Identify the dominant institutional pattern."""
        patterns = ['vwap', 'iceberg', 'stealth', 'program']
        
        if not pattern_scores or max(pattern_scores) < 0.3:
            return 'none'
        
        max_idx = np.argmax(pattern_scores)
        return patterns[max_idx]
    
    def _analyze_smart_money_flow(self, block_trades: Dict, prices: np.ndarray, sides: np.ndarray) -> Dict[str, Any]:
        """Analyze smart money flow based on block trades."""
        if not block_trades or block_trades['recent_count'] == 0:
            return {}
        
        recent_blocks = block_trades['recent_blocks']
        
        # Smart money indicators
        
        # 1. Block trades against the trend (contrarian)
        if len(prices) >= 10:
            price_trend = np.polyfit(range(len(prices[-10:])), prices[-10:], 1)[0]
            contrarian_blocks = 0
            
            for block in recent_blocks[-5:]:
                block_side = block['side']
                if (price_trend > 0 and block_side < 0) or (price_trend < 0 and block_side > 0):
                    contrarian_blocks += 1
            
            contrarian_ratio = contrarian_blocks / min(len(recent_blocks), 5)
        else:
            contrarian_ratio = 0
        
        # 2. Block trades at price extremes
        if len(prices) >= 20:
            recent_high = np.max(prices[-20:])
            recent_low = np.min(prices[-20:])
            price_range = recent_high - recent_low
            
            extreme_blocks = 0
            for block in recent_blocks[-5:]:
                price_position = (block['price'] - recent_low) / (price_range + 1e-8)
                # Smart money often buys near lows, sells near highs
                if (block['side'] > 0 and price_position < 0.3) or (block['side'] < 0 and price_position > 0.7):
                    extreme_blocks += 1
            
            extreme_ratio = extreme_blocks / min(len(recent_blocks), 5)
        else:
            extreme_ratio = 0
        
        # 3. Volume persistence (smart money accumulation/distribution)
        buy_volume = sum(block['volume'] for block in recent_blocks if block['side'] > 0)
        sell_volume = sum(block['volume'] for block in recent_blocks if block['side'] < 0)
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            volume_imbalance = abs(buy_volume - sell_volume) / total_volume
            dominant_side = 'buy' if buy_volume > sell_volume else 'sell'
        else:
            volume_imbalance = 0
            dominant_side = 'neutral'
        
        # Overall smart money score
        smart_money_score = (contrarian_ratio * 0.4 + extreme_ratio * 0.4 + volume_imbalance * 0.2)
        
        return {
            'smart_money_score': float(smart_money_score),
            'contrarian_ratio': float(contrarian_ratio),
            'extreme_ratio': float(extreme_ratio),
            'volume_imbalance': float(volume_imbalance),
            'dominant_side': dominant_side,
            'activity_level': len(recent_blocks) / max(len(prices), 1),
            'smart_money_detected': smart_money_score > self.parameters['smart_money_threshold']
        }
    
    def _assess_market_impact(self, block_trades: Dict, prices: np.ndarray) -> Dict[str, Any]:
        """Assess market impact of block trades."""
        if not block_trades or block_trades['recent_count'] == 0 or len(prices) < self.parameters['impact_window']:
            return {}
        
        recent_blocks = block_trades['recent_blocks']
        impact_window = self.parameters['impact_window']
        
        impacts = []
        for block in recent_blocks[-5:]:  # Analyze last 5 blocks
            block_idx = block['index']
            
            # Calculate price impact after block trade
            if block_idx + impact_window < len(prices):
                pre_price = prices[block_idx]
                post_prices = prices[block_idx+1:block_idx+impact_window+1]
                
                if len(post_prices) > 0:
                    # Maximum impact
                    max_impact = np.max(np.abs(post_prices - pre_price)) / pre_price
                    
                    # Permanent impact (price at end of window)
                    permanent_impact = abs(post_prices[-1] - pre_price) / pre_price
                    
                    # Impact direction consistency
                    expected_direction = 1 if block['side'] > 0 else -1
                    actual_direction = np.sign(post_prices[-1] - pre_price)
                    direction_consistency = 1 if expected_direction == actual_direction else 0
                    
                    impacts.append({
                        'max_impact': max_impact,
                        'permanent_impact': permanent_impact,
                        'direction_consistency': direction_consistency,
                        'block_size': block['volume']
                    })
        
        if not impacts:
            return {}
        
        # Aggregate impact metrics
        avg_max_impact = np.mean([i['max_impact'] for i in impacts])
        avg_permanent_impact = np.mean([i['permanent_impact'] for i in impacts])
        avg_direction_consistency = np.mean([i['direction_consistency'] for i in impacts])
        
        # Impact efficiency (permanent / max)
        impact_efficiency = avg_permanent_impact / (avg_max_impact + 1e-8)
        
        return {
            'avg_max_impact': float(avg_max_impact),
            'avg_permanent_impact': float(avg_permanent_impact),
            'direction_consistency': float(avg_direction_consistency),
            'impact_efficiency': float(impact_efficiency),
            'impact_samples': len(impacts),
            'high_impact_detected': avg_max_impact > 0.01  # 1% threshold
        }
    
    def _analyze_block_momentum(self, block_trades: Dict, prices: np.ndarray) -> Dict[str, Any]:
        """Analyze momentum following block trades."""
        if not block_trades or block_trades['recent_count'] == 0:
            return {}
        
        recent_blocks = block_trades['recent_blocks']
        momentum_window = self.parameters['momentum_window']
        
        # Analyze momentum for recent blocks
        momentum_scores = []
        for block in recent_blocks[-3:]:  # Last 3 blocks
            block_idx = block['index']
            
            if block_idx + momentum_window < len(prices):
                pre_prices = prices[max(0, block_idx-5):block_idx]
                post_prices = prices[block_idx:block_idx+momentum_window]
                
                if len(pre_prices) > 0 and len(post_prices) > 0:
                    # Pre-block momentum
                    pre_momentum = (pre_prices[-1] - pre_prices[0]) / pre_prices[0] if len(pre_prices) > 1 else 0
                    
                    # Post-block momentum
                    post_momentum = (post_prices[-1] - post_prices[0]) / post_prices[0] if len(post_prices) > 1 else 0
                    
                    # Momentum acceleration
                    momentum_acceleration = post_momentum - pre_momentum
                    
                    # Direction consistency with block trade
                    block_direction = 1 if block['side'] > 0 else -1
                    momentum_alignment = 1 if np.sign(post_momentum) == block_direction else 0
                    
                    momentum_scores.append({
                        'pre_momentum': pre_momentum,
                        'post_momentum': post_momentum,
                        'acceleration': momentum_acceleration,
                        'alignment': momentum_alignment
                    })
        
        if not momentum_scores:
            return {}
        
        # Aggregate momentum metrics
        avg_acceleration = np.mean([m['acceleration'] for m in momentum_scores])
        avg_alignment = np.mean([m['alignment'] for m in momentum_scores])
        momentum_strength = np.mean([abs(m['post_momentum']) for m in momentum_scores])
        
        return {
            'momentum_acceleration': float(avg_acceleration),
            'momentum_alignment': float(avg_alignment),
            'momentum_strength': float(momentum_strength),
            'follow_through_quality': float(avg_alignment * momentum_strength),
            'momentum_samples': len(momentum_scores)
        }
    
    def _calculate_block_frequency(self, block_trades: Dict) -> float:
        """Calculate block trade frequency."""
        if not block_trades:
            return 0.0
        
        return float(block_trades.get('frequency', 0))
    
    def _calculate_institutional_score(self, block_trades: Dict, flow_analysis: Dict) -> float:
        """Calculate overall institutional activity score."""
        components = []
        
        # Block frequency component
        frequency = self._calculate_block_frequency(block_trades)
        components.append(min(frequency * 10, 1.0))  # Scale frequency
        
        # Smart money component
        if flow_analysis and 'smart_money_score' in flow_analysis:
            components.append(flow_analysis['smart_money_score'])
        
        # Block size component
        if block_trades and 'avg_size' in block_trades:
            avg_size = block_trades['avg_size']
            size_score = min(avg_size / 10000, 1.0)  # Normalize by typical size
            components.append(size_score)
        
        if components:
            return float(np.mean(components))
        
        return 0.0
    
    def _calculate_signal_strength(self, block_trades: Dict, institutional_patterns: Dict,
                                 flow_analysis: Dict, impact_analysis: Dict,
                                 momentum_analysis: Dict) -> float:
        """Calculate trading signal strength based on block trade analysis."""
        signal_components = []
        
        # Block trade volume signal
        if block_trades and block_trades.get('recent_count', 0) > 0:
            buy_sell_ratio = block_trades.get('buy_sell_ratio', 0.5)
            volume_signal = (buy_sell_ratio - 0.5) * 2  # Convert to -1 to 1 range
            signal_components.append(volume_signal * 0.3)
        
        # Institutional pattern signal
        if institutional_patterns:
            inst_score = institutional_patterns.get('institutional_score', 0)
            if inst_score > 0.7:
                signal_components.append(0.4)
        
        # Smart money flow signal
        if flow_analysis:
            if flow_analysis.get('smart_money_detected', False):
                dominant_side = flow_analysis.get('dominant_side', 'neutral')
                if dominant_side == 'buy':
                    signal_components.append(0.5)
                elif dominant_side == 'sell':
                    signal_components.append(-0.5)
        
        # Momentum follow-through signal
        if momentum_analysis:
            follow_through = momentum_analysis.get('follow_through_quality', 0)
            acceleration = momentum_analysis.get('momentum_acceleration', 0)
            momentum_signal = follow_through * np.sign(acceleration) * 0.3
            signal_components.append(momentum_signal)
        
        # Market impact signal (contrarian - high impact suggests reversal)
        if impact_analysis:
            if impact_analysis.get('high_impact_detected', False):
                avg_impact = impact_analysis.get('avg_max_impact', 0)
                impact_signal = -min(avg_impact * 20, 0.3)  # Negative signal for high impact
                signal_components.append(impact_signal)
        
        # Combine signals
        if signal_components:
            total_signal = np.sum(signal_components)
            return float(np.clip(total_signal, -1, 1))
        
        return 0.0
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on block trade analysis."""
        signal_strength = value['signal_strength']
        institutional_score = value['institutional_score']
        block_frequency = value['block_frequency']
        
        # Confidence based on institutional activity
        confidence = min(abs(signal_strength) * institutional_score * (1 + block_frequency), 1.0)
        
        # Require minimum institutional activity for strong signals
        if institutional_score < 0.3:
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
