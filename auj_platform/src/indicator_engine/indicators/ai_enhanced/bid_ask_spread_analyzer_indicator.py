"""
Bid-Ask Spread Analyzer - Advanced Market Microstructure Analysis
===============================================================

Sophisticated bid-ask spread analysis with liquidity detection, market maker behavior analysis,
and institutional flow identification using advanced statistical methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class BidAskSpreadAnalyzer(StandardIndicatorInterface):
    """
    AI-Enhanced Bid-Ask Spread Analyzer for market microstructure analysis.
    
    Features:
    - Real-time spread calculation and normalization
    - Liquidity regime detection and classification
    - Market maker behavior pattern recognition
    - Institutional order flow identification
    - Spread clustering analysis for anomaly detection
    - Market impact prediction using spread dynamics
    - Optimal execution timing recommendations
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'spread_window': 50,           # Window for spread analysis
            'liquidity_threshold': 0.002,  # 0.2% spread threshold for liquidity
            'clustering_eps': 0.001,       # DBSCAN epsilon for spread clustering
            'min_samples': 5,              # Minimum samples for clustering
            'institutional_threshold': 2.0, # Threshold for institutional flow detection
            'market_maker_detection': True,
            'anomaly_detection': True,
            'impact_prediction': True,
            'execution_timing': True,
            'volatility_adjustment': True,
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("BidAskSpreadAnalyzer", default_params)
        
        # Analysis components
        self.spread_history = []
        self.liquidity_regimes = []
        self.anomaly_detector = DBSCAN(
            eps=self.parameters['clustering_eps'],
            min_samples=self.parameters['min_samples']
        )
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.ORDER_BOOK,
            required_columns=['bid_price', 'ask_price', 'bid_size', 'ask_size', 'timestamp'],
            min_periods=self.parameters['spread_window'],
            lookback_periods=200
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive bid-ask spread analysis.
        """
        try:
            # Extract order book data
            bid_prices = data['bid_price'].values
            ask_prices = data['ask_price'].values
            bid_sizes = data['bid_size'].values
            ask_sizes = data['ask_size'].values
            
            # Calculate spreads
            spreads = self._calculate_spreads(bid_prices, ask_prices)
            
            # Normalize spreads
            normalized_spreads = self._normalize_spreads(spreads, bid_prices, ask_prices)
            
            # Liquidity analysis
            liquidity_analysis = self._analyze_liquidity(spreads, bid_sizes, ask_sizes, bid_prices, ask_prices)
            
            # Market maker behavior detection
            mm_analysis = self._detect_market_maker_behavior(spreads, bid_sizes, ask_sizes) if self.parameters['market_maker_detection'] else {}
            
            # Institutional flow detection
            institutional_flow = self._detect_institutional_flow(spreads, bid_sizes, ask_sizes, normalized_spreads) if self.parameters['institutional_threshold'] else {}
            
            # Anomaly detection in spread patterns
            anomaly_analysis = self._detect_spread_anomalies(normalized_spreads) if self.parameters['anomaly_detection'] else {}
            
            # Market impact prediction
            impact_prediction = self._predict_market_impact(spreads, bid_sizes, ask_sizes) if self.parameters['impact_prediction'] else {}
            
            # Execution timing analysis
            execution_timing = self._analyze_execution_timing(spreads, liquidity_analysis) if self.parameters['execution_timing'] else {}
            
            # Spread dynamics
            spread_dynamics = self._analyze_spread_dynamics(spreads, normalized_spreads)
            
            # Signal generation
            signal_strength = self._calculate_signal_strength(
                liquidity_analysis, mm_analysis, institutional_flow, 
                anomaly_analysis, spread_dynamics
            )
            
            return {
                'current_spread': spreads[-1] if len(spreads) > 0 else 0,
                'normalized_spread': normalized_spreads[-1] if len(normalized_spreads) > 0 else 0,
                'spread_percentile': self._calculate_spread_percentile(spreads),
                'liquidity_analysis': liquidity_analysis,
                'market_maker_analysis': mm_analysis,
                'institutional_flow': institutional_flow,
                'anomaly_analysis': anomaly_analysis,
                'impact_prediction': impact_prediction,
                'execution_timing': execution_timing,
                'spread_dynamics': spread_dynamics,
                'signal_strength': signal_strength,
                'liquidity_score': self._calculate_liquidity_score(spreads, bid_sizes, ask_sizes),
                'market_efficiency': self._calculate_market_efficiency(spreads, normalized_spreads)
            }
            
        except Exception as e:
            raise Exception(f"BidAskSpreadAnalyzer calculation failed: {str(e)}")
    
    def _calculate_spreads(self, bid_prices: np.ndarray, ask_prices: np.ndarray) -> np.ndarray:
        """Calculate absolute and relative spreads."""
        absolute_spreads = ask_prices - bid_prices
        return absolute_spreads
    
    def _normalize_spreads(self, spreads: np.ndarray, bid_prices: np.ndarray, ask_prices: np.ndarray) -> np.ndarray:
        """Normalize spreads by mid-price."""
        mid_prices = (bid_prices + ask_prices) / 2
        normalized_spreads = spreads / (mid_prices + 1e-8)
        return normalized_spreads
    
    def _analyze_liquidity(self, spreads: np.ndarray, bid_sizes: np.ndarray, 
                          ask_sizes: np.ndarray, bid_prices: np.ndarray, 
                          ask_prices: np.ndarray) -> Dict[str, Any]:
        """Comprehensive liquidity analysis."""
        if len(spreads) == 0:
            return {}
        
        # Current liquidity metrics
        current_spread = spreads[-1]
        current_bid_size = bid_sizes[-1] if len(bid_sizes) > 0 else 0
        current_ask_size = ask_sizes[-1] if len(ask_sizes) > 0 else 0
        
        # Liquidity measures
        total_depth = current_bid_size + current_ask_size
        depth_imbalance = (current_ask_size - current_bid_size) / (total_depth + 1e-8)
        
        # Historical liquidity percentile
        recent_spreads = spreads[-20:] if len(spreads) >= 20 else spreads
        spread_percentile = stats.percentileofscore(recent_spreads, current_spread) / 100
        
        # Liquidity regime classification
        if current_spread < np.percentile(recent_spreads, 25):
            liquidity_regime = 'high_liquidity'
        elif current_spread > np.percentile(recent_spreads, 75):
            liquidity_regime = 'low_liquidity'
        else:
            liquidity_regime = 'normal_liquidity'
        
        # Liquidity volatility
        spread_volatility = np.std(recent_spreads) if len(recent_spreads) > 1 else 0
        
        # Time-weighted spread
        if len(spreads) >= 10:
            weights = np.exp(np.linspace(-1, 0, len(spreads[-10:])))
            weights = weights / np.sum(weights)
            time_weighted_spread = np.average(spreads[-10:], weights=weights)
        else:
            time_weighted_spread = current_spread
        
        return {
            'liquidity_regime': liquidity_regime,
            'total_depth': float(total_depth),
            'depth_imbalance': float(depth_imbalance),
            'spread_percentile': float(spread_percentile),
            'spread_volatility': float(spread_volatility),
            'time_weighted_spread': float(time_weighted_spread),
            'liquidity_trend': self._calculate_liquidity_trend(spreads)
        }
    
    def _calculate_liquidity_trend(self, spreads: np.ndarray) -> str:
        """Calculate liquidity trend direction."""
        if len(spreads) < 10:
            return 'stable'
        
        recent_avg = np.mean(spreads[-5:])
        historical_avg = np.mean(spreads[-15:-5]) if len(spreads) >= 15 else np.mean(spreads[:-5])
        
        change_ratio = (recent_avg - historical_avg) / (historical_avg + 1e-8)
        
        if change_ratio > 0.1:  # 10% increase in spread
            return 'deteriorating'
        elif change_ratio < -0.1:  # 10% decrease in spread
            return 'improving'
        else:
            return 'stable'
    
    def _detect_market_maker_behavior(self, spreads: np.ndarray, bid_sizes: np.ndarray, 
                                    ask_sizes: np.ndarray) -> Dict[str, Any]:
        """Detect market maker behavior patterns."""
        if len(spreads) < 20:
            return {}
        
        # Market maker indicators
        
        # 1. Spread consistency (market makers maintain consistent spreads)
        spread_consistency = 1 - (np.std(spreads[-20:]) / (np.mean(spreads[-20:]) + 1e-8))
        
        # 2. Depth symmetry (market makers often provide balanced liquidity)
        depth_symmetry = []
        for i in range(min(20, len(bid_sizes))):
            if bid_sizes[-i-1] + ask_sizes[-i-1] > 0:
                symmetry = 1 - abs(bid_sizes[-i-1] - ask_sizes[-i-1]) / (bid_sizes[-i-1] + ask_sizes[-i-1])
                depth_symmetry.append(symmetry)
        
        avg_depth_symmetry = np.mean(depth_symmetry) if depth_symmetry else 0
        
        # 3. Spread mean reversion (market makers adjust spreads quickly)
        spread_autocorr = np.corrcoef(spreads[-19:], spreads[-20:-1])[0, 1] if len(spreads) >= 20 else 0
        mean_reversion = 1 - abs(spread_autocorr) if not np.isnan(spread_autocorr) else 0.5
        
        # 4. Size-to-spread ratio (market makers often increase depth with wider spreads)
        total_sizes = bid_sizes + ask_sizes
        if len(total_sizes) >= 10 and len(spreads) >= 10:
            size_spread_corr = np.corrcoef(total_sizes[-10:], spreads[-10:])[0, 1]
            if np.isnan(size_spread_corr):
                size_spread_corr = 0
        else:
            size_spread_corr = 0
        
        # Overall market maker score
        mm_score = (spread_consistency * 0.3 + avg_depth_symmetry * 0.3 + 
                   mean_reversion * 0.2 + abs(size_spread_corr) * 0.2)
        
        # Market maker presence classification
        if mm_score > 0.7:
            mm_presence = 'strong'
        elif mm_score > 0.5:
            mm_presence = 'moderate'
        else:
            mm_presence = 'weak'
        
        return {
            'market_maker_score': float(mm_score),
            'market_maker_presence': mm_presence,
            'spread_consistency': float(spread_consistency),
            'depth_symmetry': float(avg_depth_symmetry),
            'mean_reversion': float(mean_reversion),
            'size_spread_correlation': float(size_spread_corr)
        }
    
    def _detect_institutional_flow(self, spreads: np.ndarray, bid_sizes: np.ndarray,
                                 ask_sizes: np.ndarray, normalized_spreads: np.ndarray) -> Dict[str, Any]:
        """Detect institutional order flow patterns."""
        if len(spreads) < 10:
            return {}
        
        # Institutional flow indicators
        
        # 1. Large size combined with spread impact
        total_sizes = bid_sizes + ask_sizes
        size_threshold = np.percentile(total_sizes[-50:] if len(total_sizes) >= 50 else total_sizes, 90)
        
        recent_large_orders = 0
        spread_impact_events = 0
        
        for i in range(min(10, len(total_sizes))):
            if total_sizes[-i-1] > size_threshold:
                recent_large_orders += 1
                
                # Check if spread widened after large order
                if i < len(spreads) - 1 and spreads[-i] > spreads[-i-1] * 1.1:
                    spread_impact_events += 1
        
        institutional_pressure = recent_large_orders / 10
        impact_ratio = spread_impact_events / max(recent_large_orders, 1)
        
        # 2. Order flow imbalance
        recent_imbalances = []
        for i in range(min(20, len(bid_sizes))):
            total_size = bid_sizes[-i-1] + ask_sizes[-i-1]
            if total_size > 0:
                imbalance = (ask_sizes[-i-1] - bid_sizes[-i-1]) / total_size
                recent_imbalances.append(imbalance)
        
        avg_imbalance = np.mean(recent_imbalances) if recent_imbalances else 0
        imbalance_persistence = len([x for x in recent_imbalances if x * avg_imbalance > 0]) / len(recent_imbalances) if recent_imbalances else 0
        
        # 3. Spread volatility spikes
        if len(normalized_spreads) >= 20:
            spread_vol = np.std(normalized_spreads[-20:])
            historical_vol = np.std(normalized_spreads[-50:-20]) if len(normalized_spreads) >= 50 else spread_vol
            vol_spike = spread_vol / (historical_vol + 1e-8)
        else:
            vol_spike = 1.0
        
        # Overall institutional flow score
        inst_score = (institutional_pressure * 0.4 + abs(avg_imbalance) * 0.3 + 
                     (vol_spike - 1) * 0.3)
        
        # Flow direction
        if avg_imbalance > 0.1:
            flow_direction = 'selling_pressure'
        elif avg_imbalance < -0.1:
            flow_direction = 'buying_pressure'
        else:
            flow_direction = 'balanced'
        
        return {
            'institutional_score': float(np.clip(inst_score, 0, 1)),
            'flow_direction': flow_direction,
            'order_flow_imbalance': float(avg_imbalance),
            'imbalance_persistence': float(imbalance_persistence),
            'large_order_frequency': float(institutional_pressure),
            'spread_impact_ratio': float(impact_ratio),
            'volatility_spike': float(vol_spike)
        }
    
    def _detect_spread_anomalies(self, normalized_spreads: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in spread patterns using clustering."""
        if len(normalized_spreads) < 20:
            return {}
        
        try:
            # Prepare features for anomaly detection
            features = []
            window = 5
            
            for i in range(window, len(normalized_spreads)):
                window_data = normalized_spreads[i-window:i]
                feature_vector = [
                    np.mean(window_data),
                    np.std(window_data),
                    np.max(window_data),
                    np.min(window_data),
                    window_data[-1] - window_data[0]  # Change over window
                ]
                features.append(feature_vector)
            
            if len(features) < self.parameters['min_samples']:
                return {}
            
            features_array = np.array(features)
            
            # Perform clustering to identify anomalies
            labels = self.anomaly_detector.fit_predict(features_array)
            
            # Anomalies are points labeled as -1
            anomaly_count = np.sum(labels == -1)
            total_points = len(labels)
            anomaly_rate = anomaly_count / total_points if total_points > 0 else 0
            
            # Check if recent points are anomalies
            recent_anomalies = np.sum(labels[-5:] == -1) if len(labels) >= 5 else 0
            current_anomaly_score = recent_anomalies / min(5, len(labels))
            
            # Anomaly severity
            if len(features) > 0:
                current_features = features[-1]
                feature_distances = []
                
                for feature in features[-20:]:  # Compare with recent normal patterns
                    if len(feature) == len(current_features):
                        distance = np.linalg.norm(np.array(current_features) - np.array(feature))
                        feature_distances.append(distance)
                
                anomaly_severity = np.percentile(feature_distances, 95) if feature_distances else 0
            else:
                anomaly_severity = 0
            
            return {
                'anomaly_detected': current_anomaly_score > 0.4,
                'anomaly_score': float(current_anomaly_score),
                'anomaly_rate': float(anomaly_rate),
                'anomaly_severity': float(anomaly_severity),
                'cluster_count': len(set(labels)) - (1 if -1 in labels else 0)
            }
            
        except Exception:
            return {'anomaly_detected': False}
    
    def _predict_market_impact(self, spreads: np.ndarray, bid_sizes: np.ndarray, 
                             ask_sizes: np.ndarray) -> Dict[str, Any]:
        """Predict market impact based on current spread and liquidity conditions."""
        if len(spreads) == 0:
            return {}
        
        current_spread = spreads[-1]
        total_depth = (bid_sizes[-1] + ask_sizes[-1]) if len(bid_sizes) > 0 and len(ask_sizes) > 0 else 0
        
        # Impact prediction based on Kyle's lambda model (simplified)
        # Impact = lambda * (Order Size / Average Volume)
        
        # Estimate lambda from recent spread and size relationship
        if len(spreads) >= 10 and len(bid_sizes) >= 10:
            recent_spreads = spreads[-10:]
            recent_depths = (bid_sizes[-10:] + ask_sizes[-10:])
            
            # Kyle's lambda approximation
            lambda_estimate = np.mean(recent_spreads) / (np.mean(recent_depths) + 1e-8)
        else:
            lambda_estimate = current_spread / (total_depth + 1e-8)
        
        # Predict impact for different order sizes (as multiples of current depth)
        order_size_multiples = [0.1, 0.25, 0.5, 1.0, 2.0]
        impact_predictions = {}
        
        for multiplier in order_size_multiples:
            order_size = total_depth * multiplier
            predicted_impact = lambda_estimate * order_size
            impact_predictions[f'size_{multiplier}x'] = float(predicted_impact)
        
        # Market resilience (how quickly spreads return to normal)
        if len(spreads) >= 10:
            spread_autocorr = np.corrcoef(spreads[-9:], spreads[-10:-1])[0, 1]
            resilience = 1 - abs(spread_autocorr) if not np.isnan(spread_autocorr) else 0.5
        else:
            resilience = 0.5
        
        return {
            'lambda_estimate': float(lambda_estimate),
            'impact_predictions': impact_predictions,
            'market_resilience': float(resilience),
            'current_liquidity_depth': float(total_depth),
            'optimal_order_size': float(total_depth * 0.25)  # Conservative recommendation
        }
    
    def _analyze_execution_timing(self, spreads: np.ndarray, liquidity_analysis: Dict) -> Dict[str, Any]:
        """Analyze optimal execution timing based on spread patterns."""
        if len(spreads) < 10:
            return {}
        
        current_spread = spreads[-1]
        recent_spreads = spreads[-10:]
        
        # Timing score based on spread level
        spread_percentile = stats.percentileofscore(recent_spreads, current_spread) / 100
        
        # Optimal timing (lower spreads are better for execution)
        if spread_percentile < 0.2:
            timing_recommendation = 'excellent'
            timing_score = 0.9
        elif spread_percentile < 0.4:
            timing_recommendation = 'good'
            timing_score = 0.7
        elif spread_percentile < 0.6:
            timing_recommendation = 'fair'
            timing_score = 0.5
        elif spread_percentile < 0.8:
            timing_recommendation = 'poor'
            timing_score = 0.3
        else:
            timing_recommendation = 'very_poor'
            timing_score = 0.1
        
        # Trend consideration
        if len(spreads) >= 5:
            spread_trend = np.polyfit(range(5), spreads[-5:], 1)[0]
            if spread_trend < 0:  # Spreads tightening
                timing_score += 0.1
                trend_direction = 'improving'
            elif spread_trend > 0:  # Spreads widening
                timing_score -= 0.1
                trend_direction = 'deteriorating'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'stable'
        
        timing_score = np.clip(timing_score, 0, 1)
        
        # Expected wait time for better conditions
        liquidity_regime = liquidity_analysis.get('liquidity_regime', 'normal_liquidity')
        
        if liquidity_regime == 'high_liquidity':
            expected_wait_time = 'immediate'
        elif timing_recommendation in ['excellent', 'good']:
            expected_wait_time = 'immediate'
        elif timing_recommendation == 'fair':
            expected_wait_time = 'short'
        else:
            expected_wait_time = 'medium'
        
        return {
            'timing_recommendation': timing_recommendation,
            'timing_score': float(timing_score),
            'spread_percentile': float(spread_percentile),
            'trend_direction': trend_direction,
            'expected_wait_time': expected_wait_time,
            'urgency_adjustment': self._calculate_urgency_adjustment(spreads)
        }
    
    def _calculate_urgency_adjustment(self, spreads: np.ndarray) -> str:
        """Calculate urgency adjustment for execution timing."""
        if len(spreads) < 5:
            return 'normal'
        
        # Check for spread stability
        recent_volatility = np.std(spreads[-5:])
        historical_volatility = np.std(spreads[-15:-5]) if len(spreads) >= 15 else recent_volatility
        
        volatility_ratio = recent_volatility / (historical_volatility + 1e-8)
        
        if volatility_ratio > 2.0:
            return 'high_urgency'  # Spreads becoming more volatile
        elif volatility_ratio < 0.5:
            return 'low_urgency'   # Spreads becoming more stable
        else:
            return 'normal'
    
    def _analyze_spread_dynamics(self, spreads: np.ndarray, normalized_spreads: np.ndarray) -> Dict[str, Any]:
        """Analyze spread dynamics and patterns."""
        if len(spreads) < 10:
            return {}
        
        # Spread momentum
        if len(spreads) >= 5:
            momentum = spreads[-1] - spreads[-5]
            momentum_strength = abs(momentum) / (np.mean(spreads[-5:]) + 1e-8)
        else:
            momentum = 0
            momentum_strength = 0
        
        # Spread cycles (look for periodic patterns)
        if len(normalized_spreads) >= 20:
            # Simple cycle detection using autocorrelation
            autocorr_lags = range(1, min(10, len(normalized_spreads) // 2))
            autocorrs = []
            
            for lag in autocorr_lags:
                if len(normalized_spreads) > lag:
                    corr = np.corrcoef(normalized_spreads[:-lag], normalized_spreads[lag:])[0, 1]
                    autocorrs.append(corr if not np.isnan(corr) else 0)
            
            # Find dominant cycle
            if autocorrs:
                max_corr_idx = np.argmax(np.abs(autocorrs))
                dominant_cycle = autocorr_lags[max_corr_idx]
                cycle_strength = abs(autocorrs[max_corr_idx])
            else:
                dominant_cycle = 0
                cycle_strength = 0
        else:
            dominant_cycle = 0
            cycle_strength = 0
        
        # Mean reversion tendency
        if len(spreads) >= 10:
            mean_spread = np.mean(spreads[-20:]) if len(spreads) >= 20 else np.mean(spreads)
            current_deviation = (spreads[-1] - mean_spread) / (mean_spread + 1e-8)
            
            # Check how often spreads return to mean
            deviations = spreads[-10:] - mean_spread
            mean_reversions = 0
            for i in range(1, len(deviations)):
                if deviations[i-1] * deviations[i] < 0:  # Sign change indicates mean crossing
                    mean_reversions += 1
            
            reversion_frequency = mean_reversions / (len(deviations) - 1)
        else:
            current_deviation = 0
            reversion_frequency = 0.5
        
        return {
            'momentum': float(momentum),
            'momentum_strength': float(momentum_strength),
            'dominant_cycle': int(dominant_cycle),
            'cycle_strength': float(cycle_strength),
            'current_deviation': float(current_deviation),
            'mean_reversion_frequency': float(reversion_frequency),
            'spread_volatility': float(np.std(spreads[-10:]) if len(spreads) >= 10 else 0)
        }
    
    def _calculate_spread_percentile(self, spreads: np.ndarray) -> float:
        """Calculate current spread percentile."""
        if len(spreads) < 2:
            return 50.0
        
        current_spread = spreads[-1]
        historical_spreads = spreads[:-1]
        
        percentile = stats.percentileofscore(historical_spreads, current_spread)
        return float(percentile)
    
    def _calculate_liquidity_score(self, spreads: np.ndarray, bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> float:
        """Calculate overall liquidity score."""
        if len(spreads) == 0:
            return 0.5
        
        # Spread component (lower is better)
        current_spread = spreads[-1]
        avg_spread = np.mean(spreads[-20:]) if len(spreads) >= 20 else current_spread
        spread_score = 1.0 / (1.0 + current_spread / (avg_spread + 1e-8))
        
        # Depth component (higher is better)
        if len(bid_sizes) > 0 and len(ask_sizes) > 0:
            current_depth = bid_sizes[-1] + ask_sizes[-1]
            avg_depth = np.mean((bid_sizes + ask_sizes)[-20:]) if len(bid_sizes) >= 20 else current_depth
            depth_score = current_depth / (avg_depth + 1e-8)
            depth_score = min(depth_score, 2.0) / 2.0  # Normalize to 0-1
        else:
            depth_score = 0.5
        
        # Combined liquidity score
        liquidity_score = (spread_score * 0.6 + depth_score * 0.4)
        
        return float(np.clip(liquidity_score, 0, 1))
    
    def _calculate_market_efficiency(self, spreads: np.ndarray, normalized_spreads: np.ndarray) -> float:
        """Calculate market efficiency score."""
        if len(spreads) < 10:
            return 0.5
        
        # Efficiency based on spread tightness and stability
        avg_normalized_spread = np.mean(normalized_spreads[-10:])
        spread_stability = 1.0 - (np.std(normalized_spreads[-10:]) / (avg_normalized_spread + 1e-8))
        
        # Lower spreads and higher stability indicate better efficiency
        tightness_score = 1.0 / (1.0 + avg_normalized_spread * 1000)  # Scale for typical spreads
        efficiency = (tightness_score * 0.7 + spread_stability * 0.3)
        
        return float(np.clip(efficiency, 0, 1))
    
    def _calculate_signal_strength(self, liquidity_analysis: Dict, mm_analysis: Dict,
                                 institutional_flow: Dict, anomaly_analysis: Dict,
                                 spread_dynamics: Dict) -> float:
        """Calculate trading signal strength based on spread analysis."""
        signal_components = []
        
        # Liquidity signal
        if liquidity_analysis:
            regime = liquidity_analysis.get('liquidity_regime', 'normal_liquidity')
            if regime == 'high_liquidity':
                signal_components.append(0.3)  # Good for execution
            elif regime == 'low_liquidity':
                signal_components.append(-0.2)  # Poor for execution
        
        # Market maker signal
        if mm_analysis:
            mm_presence = mm_analysis.get('market_maker_presence', 'moderate')
            if mm_presence == 'strong':
                signal_components.append(0.2)  # Stable market making
            elif mm_presence == 'weak':
                signal_components.append(-0.1)  # Unstable liquidity
        
        # Institutional flow signal
        if institutional_flow:
            flow_direction = institutional_flow.get('flow_direction', 'balanced')
            inst_score = institutional_flow.get('institutional_score', 0)
            
            if flow_direction == 'buying_pressure' and inst_score > 0.5:
                signal_components.append(0.4)
            elif flow_direction == 'selling_pressure' and inst_score > 0.5:
                signal_components.append(-0.4)
        
        # Anomaly signal
        if anomaly_analysis and anomaly_analysis.get('anomaly_detected', False):
            anomaly_score = anomaly_analysis.get('anomaly_score', 0)
            signal_components.append(-anomaly_score * 0.3)  # Anomalies are negative
        
        # Spread dynamics signal
        if spread_dynamics:
            momentum_strength = spread_dynamics.get('momentum_strength', 0)
            current_deviation = spread_dynamics.get('current_deviation', 0)
            
            # Mean reversion signal
            if abs(current_deviation) > 0.1:
                reversion_signal = -np.sign(current_deviation) * min(abs(current_deviation), 0.5)
                signal_components.append(reversion_signal * 0.2)
        
        # Combine signals
        if signal_components:
            total_signal = np.sum(signal_components)
            return float(np.clip(total_signal, -1, 1))
        
        return 0.0
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on spread analysis."""
        signal_strength = value['signal_strength']
        liquidity_score = value['liquidity_score']
        market_efficiency = value['market_efficiency']
        
        # Confidence based on liquidity and efficiency
        confidence = min(abs(signal_strength) * liquidity_score * market_efficiency, 1.0)
        
        # Require minimum liquidity for strong signals
        if liquidity_score < 0.3:
            return SignalType.NEUTRAL, confidence
        
        if signal_strength > 0.5:
            return SignalType.STRONG_BUY, confidence
        elif signal_strength > 0.2:
            return SignalType.BUY, confidence
        elif signal_strength < -0.5:
            return SignalType.STRONG_SELL, confidence
        elif signal_strength < -0.2:
            return SignalType.SELL, confidence
        else:
            return SignalType.NEUTRAL, confidence
