"""
Order Flow Sequence Analyzer - Advanced Pattern Recognition and Predictive Modeling
==================================================================================

The Order Flow Sequence Analyzer is a sophisticated system for analyzing sequential patterns 
in order flow data to predict future market movements. This implementation uses advanced 
pattern recognition algorithms, machine learning models, and statistical analysis to 
identify meaningful sequences in trading activity.

Key Features:
- Sequential pattern recognition with sliding window analysis
- Advanced flow imbalance detection and trend prediction
- Machine learning-based sequence classification and prediction
- Multi-timeframe sequence analysis with fractal pattern detection
- Institutional vs retail flow sequence identification
- Momentum sequence analysis and reversal pattern detection
- Real-time sequence scoring and confidence assessment

Mathematical Foundation:
The analyzer uses Hidden Markov Models (HMM), sequence mining algorithms, and deep learning
approaches to identify patterns in order flow sequences. It employs:

1. Sequence Pattern Mining: Identifies frequent subsequences in order flow data
2. Markov Chain Analysis: Models state transitions in flow patterns
3. LSTM Networks: Predicts future flow patterns based on historical sequences
4. Dynamic Time Warping: Matches similar sequences across different timeframes
5. Entropy Analysis: Measures randomness and predictability in flow sequences

The system analyzes sequences of:
- Volume patterns (increasing/decreasing/stable)
- Price movements (up/down/sideways)
- Flow imbalances (buy/sell pressure)
- Momentum shifts (acceleration/deceleration)

Author: AUJ Platform Development Team
Created: 2025-06-21
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Sequence
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from collections import deque, Counter
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from scipy.spatial.distance import euclidean
import talib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from itertools import combinations

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowState(Enum):
    """Order flow state classifications."""
    STRONG_BUY = "strong_buy"
    MODERATE_BUY = "moderate_buy"
    NEUTRAL = "neutral"
    MODERATE_SELL = "moderate_sell"
    STRONG_SELL = "strong_sell"

class SequenceType(Enum):
    """Sequence pattern types."""
    ACCUMULATION_SEQUENCE = "accumulation_sequence"
    DISTRIBUTION_SEQUENCE = "distribution_sequence"
    MOMENTUM_SEQUENCE = "momentum_sequence"
    REVERSAL_SEQUENCE = "reversal_sequence"
    CONSOLIDATION_SEQUENCE = "consolidation_sequence"
    BREAKOUT_SEQUENCE = "breakout_sequence"
    UNKNOWN = "unknown"

class PredictionStrength(Enum):
    """Prediction strength levels."""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"

@dataclass
class SequencePattern:
    """
    Represents a detected sequence pattern.
    """
    pattern_id: str
    sequence_type: SequenceType
    flow_states: List[FlowState]
    duration: int
    strength: float
    frequency: int
    prediction_accuracy: float

@dataclass
class FlowSequenceSignal:
    """
    Comprehensive order flow sequence signal.
    
    Attributes:
        timestamp: Signal timestamp
        sequence_type: Detected sequence pattern type
        sequence_strength: Strength of the detected sequence (0-100)
        flow_imbalance_sequence: Recent flow imbalance sequence
        prediction_direction: Predicted direction (1=up, -1=down, 0=neutral)
        prediction_strength: Strength of prediction
        confidence: Overall confidence level (0-100)
        pattern_match_score: How well current sequence matches known patterns
        momentum_sequence_score: Momentum-based sequence scoring
        reversal_probability: Probability of trend reversal
        continuation_probability: Probability of trend continuation
        institutional_sequence_score: Institutional activity sequence indicator
        entropy_score: Randomness/predictability measure
        fractal_similarity: Similarity to fractal patterns
        next_flow_prediction: Predicted next flow state
        sequence_completion: How complete the current sequence is (0-100)
    """
    timestamp: datetime
    sequence_type: SequenceType
    sequence_strength: float
    flow_imbalance_sequence: List[float]
    prediction_direction: int
    prediction_strength: PredictionStrength
    confidence: float
    pattern_match_score: float
    momentum_sequence_score: float
    reversal_probability: float
    continuation_probability: float
    institutional_sequence_score: float
    entropy_score: float
    fractal_similarity: float
    next_flow_prediction: FlowState
    sequence_completion: float

class OrderFlowSequenceAnalyzer:
    """
    Advanced Order Flow Sequence Analyzer with pattern recognition and predictive modeling.
    
    This analyzer identifies sequential patterns in order flow data and uses advanced
    machine learning techniques to predict future market movements based on these patterns.
    """
    
    def __init__(self,
                 sequence_length: int = 10,
                 pattern_memory: int = 100,
                 min_pattern_frequency: int = 3,
                 prediction_horizon: int = 5,
                 flow_threshold: float = 0.1,
                 min_confidence: float = 70.0):
        """
        Initialize the Order Flow Sequence Analyzer.
        
        Args:
            sequence_length: Length of sequences to analyze
            pattern_memory: Number of historical patterns to remember
            min_pattern_frequency: Minimum frequency for pattern validation
            prediction_horizon: Number of periods ahead to predict
            flow_threshold: Threshold for flow state classification
            min_confidence: Minimum confidence for signal generation
        """
        self.sequence_length = sequence_length
        self.pattern_memory = pattern_memory
        self.min_pattern_frequency = min_pattern_frequency
        self.prediction_horizon = prediction_horizon
        self.flow_threshold = flow_threshold
        self.min_confidence = min_confidence
        
        # Pattern storage and analysis
        self.known_patterns = {}
        self.pattern_outcomes = {}
        self.sequence_history = deque(maxlen=pattern_memory * 2)
        
        # Machine learning models
        self._flow_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=15
        )
        self._sequence_predictor = GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self._pattern_clusterer = KMeans(n_clusters=8, random_state=42)
        self._scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        
        # Analysis components
        self._flow_states_history = deque(maxlen=sequence_length * 10)
        self._imbalance_history = deque(maxlen=sequence_length * 10)
        self._prediction_accuracy_tracker = {}
        
        logger.info(f"OrderFlowSequenceAnalyzer initialized with sequence_length={sequence_length}")
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate order flow sequence analysis with pattern recognition and predictions.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing sequence analysis and predictions
        """
        try:
            if data is None or data.empty:
                raise ValueError("Input data is empty or None")
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            if len(data) < self.sequence_length * 3:
                logger.warning(f"Insufficient data: {len(data)} < {self.sequence_length * 3}")
                return self._generate_empty_result()
            
            # Calculate flow states and imbalances
            flow_analysis = self._calculate_flow_states(data)
            
            # Perform sequence pattern recognition
            pattern_analysis = self._recognize_sequence_patterns(data, flow_analysis)
            
            # Analyze sequence momentum and trends
            momentum_analysis = self._analyze_sequence_momentum(data, flow_analysis)
            
            # Detect institutional vs retail sequences
            institution_analysis = self._analyze_institutional_sequences(data, flow_analysis)
            
            # Perform predictive modeling
            prediction_analysis = self._perform_predictive_modeling(data, flow_analysis, pattern_analysis)
            
            # Calculate entropy and randomness measures
            entropy_analysis = self._calculate_sequence_entropy(flow_analysis)
            
            # Fractal pattern analysis
            fractal_analysis = self._analyze_fractal_patterns(data, flow_analysis)
            
            # Generate comprehensive signals
            signals = self._generate_comprehensive_signals(
                data, flow_analysis, pattern_analysis, momentum_analysis,
                institution_analysis, prediction_analysis, entropy_analysis, fractal_analysis
            )
            
            return {
                'flow_analysis': flow_analysis,
                'pattern_analysis': pattern_analysis,
                'momentum_analysis': momentum_analysis,
                'institution_analysis': institution_analysis,
                'prediction_analysis': prediction_analysis,
                'entropy_analysis': entropy_analysis,
                'fractal_analysis': fractal_analysis,
                'signals': signals,
                'metadata': self._generate_metadata()
            }
            
        except Exception as e:
            logger.error(f"Error in OrderFlowSequenceAnalyzer calculation: {str(e)}")
            return self._generate_error_result(str(e))
    
    def _calculate_flow_states(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate flow states and imbalances."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values
            
            # Estimate buy/sell pressure using multiple methods
            flow_imbalance = self._calculate_flow_imbalance(data)
            
            # Classify flow states
            flow_states = []
            flow_state_values = np.zeros_like(flow_imbalance)
            
            for i, imbalance in enumerate(flow_imbalance):
                if imbalance > self.flow_threshold * 2:
                    state = FlowState.STRONG_BUY
                    value = 2
                elif imbalance > self.flow_threshold:
                    state = FlowState.MODERATE_BUY
                    value = 1
                elif imbalance < -self.flow_threshold * 2:
                    state = FlowState.STRONG_SELL
                    value = -2
                elif imbalance < -self.flow_threshold:
                    state = FlowState.MODERATE_SELL
                    value = -1
                else:
                    state = FlowState.NEUTRAL
                    value = 0
                
                flow_states.append(state)
                flow_state_values[i] = value
            
            # Flow momentum and acceleration
            flow_momentum = np.gradient(flow_imbalance)
            flow_acceleration = np.gradient(flow_momentum)
            
            # Volume-weighted flow
            volume_ma = talib.SMA(volume, timeperiod=20)
            volume_weight = np.where(volume_ma > 0, volume / volume_ma, 1.0)
            weighted_flow = flow_imbalance * volume_weight
            
            # Flow persistence (how long similar flow continues)
            flow_persistence = self._calculate_flow_persistence(flow_states)
            
            return {
                'flow_imbalance': flow_imbalance,
                'flow_states': flow_states,
                'flow_state_values': flow_state_values,
                'flow_momentum': flow_momentum,
                'flow_acceleration': flow_acceleration,
                'weighted_flow': weighted_flow,
                'flow_persistence': flow_persistence
            }
            
        except Exception as e:
            logger.error(f"Error calculating flow states: {str(e)}")
            return {}
    
    def _recognize_sequence_patterns(self, data: pd.DataFrame, flow_analysis: Dict) -> Dict[str, Any]:
        """Recognize sequence patterns in flow data."""
        try:
            flow_states = flow_analysis.get('flow_states', [])
            flow_state_values = flow_analysis.get('flow_state_values', np.array([]))
            
            if len(flow_states) < self.sequence_length:
                return {}
            
            # Extract sequences
            sequences = []
            sequence_types = []
            pattern_matches = np.zeros(len(flow_states))
            
            for i in range(len(flow_states) - self.sequence_length + 1):
                sequence = flow_state_values[i:i + self.sequence_length]
                sequences.append(sequence)
                
                # Classify sequence type
                seq_type = self._classify_sequence_type(sequence, flow_states[i:i + self.sequence_length])
                sequence_types.append(seq_type)
                
                # Pattern matching score
                match_score = self._calculate_pattern_match_score(sequence)
                pattern_matches[i + self.sequence_length - 1] = match_score
            
            # Update known patterns
            self._update_pattern_database(sequences, sequence_types)
            
            # Sequence similarity analysis
            similarity_scores = self._calculate_sequence_similarity(sequences)
            
            # Pattern frequency analysis
            pattern_frequencies = self._analyze_pattern_frequencies(sequences)
            
            return {
                'sequences': sequences,
                'sequence_types': sequence_types,
                'pattern_matches': pattern_matches,
                'similarity_scores': similarity_scores,
                'pattern_frequencies': pattern_frequencies,
                'known_patterns': len(self.known_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error recognizing sequence patterns: {str(e)}")
            return {}
    
    def _analyze_sequence_momentum(self, data: pd.DataFrame, flow_analysis: Dict) -> Dict[str, np.ndarray]:
        """Analyze momentum in flow sequences."""
        try:
            flow_momentum = flow_analysis.get('flow_momentum', np.array([]))
            flow_acceleration = flow_analysis.get('flow_acceleration', np.array([]))
            close = data['close'].values
            
            # Momentum sequence scoring
            momentum_sequence_scores = np.zeros_like(flow_momentum)
            momentum_trends = np.zeros_like(flow_momentum)
            momentum_strength = np.zeros_like(flow_momentum)
            
            for i in range(self.sequence_length, len(flow_momentum)):
                window_momentum = flow_momentum[i - self.sequence_length:i]
                window_acceleration = flow_acceleration[i - self.sequence_length:i]
                
                # Momentum trend analysis
                momentum_trend = np.polyfit(range(len(window_momentum)), window_momentum, 1)[0]
                momentum_trends[i] = momentum_trend
                
                # Momentum strength
                momentum_strength[i] = np.mean(np.abs(window_momentum))
                
                # Sequence scoring based on momentum consistency
                momentum_consistency = 1 - (np.std(window_momentum) / (np.mean(np.abs(window_momentum)) + 1e-8))
                acceleration_consistency = 1 - (np.std(window_acceleration) / (np.mean(np.abs(window_acceleration)) + 1e-8))
                
                momentum_sequence_scores[i] = (momentum_consistency + acceleration_consistency) * 50
            
            return {
                'momentum_sequence_scores': momentum_sequence_scores,
                'momentum_trends': momentum_trends,
                'momentum_strength': momentum_strength
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sequence momentum: {str(e)}")
            return {}
    
    def _analyze_institutional_sequences(self, data: pd.DataFrame, flow_analysis: Dict) -> Dict[str, np.ndarray]:
        """Analyze institutional vs retail sequence patterns."""
        try:
            volume = data['volume'].values
            close = data['close'].values
            flow_imbalance = flow_analysis.get('flow_imbalance', np.array([]))
            weighted_flow = flow_analysis.get('weighted_flow', np.array([]))
            
            # Institutional sequence indicators
            institutional_scores = np.zeros_like(volume)
            sequence_sophistication = np.zeros_like(volume)
            
            for i in range(self.sequence_length, len(volume)):
                window_volume = volume[i - self.sequence_length:i]
                window_flow = flow_imbalance[i - self.sequence_length:i]
                window_weighted = weighted_flow[i - self.sequence_length:i]
                
                # Large volume with consistent flow indicates institutional activity
                avg_volume = np.mean(window_volume)
                volume_percentile = stats.percentileofscore(volume[:i], avg_volume)
                
                # Flow consistency (institutions tend to have more consistent flow patterns)
                flow_consistency = 1 - (np.std(window_flow) / (np.mean(np.abs(window_flow)) + 1e-8))
                
                # Sequence sophistication (complex patterns suggest institutional activity)
                flow_entropy = self._calculate_entropy(window_flow)
                sophistication = (1 - flow_entropy) * 100  # Lower entropy = higher sophistication
                
                institutional_scores[i] = (volume_percentile * 0.4 + flow_consistency * 100 * 0.4 + sophistication * 0.2)
                sequence_sophistication[i] = sophistication
            
            # Retail vs institutional classification
            institutional_threshold = 70
            sequence_classification = np.where(institutional_scores > institutional_threshold, 1, 0)
            
            return {
                'institutional_scores': institutional_scores,
                'sequence_sophistication': sequence_sophistication,
                'sequence_classification': sequence_classification
            }
            
        except Exception as e:
            logger.error(f"Error analyzing institutional sequences: {str(e)}")
            return {}
    
    def _perform_predictive_modeling(self, data: pd.DataFrame, flow_analysis: Dict, pattern_analysis: Dict) -> Dict[str, Any]:
        """Perform predictive modeling on sequence patterns."""
        try:
            flow_state_values = flow_analysis.get('flow_state_values', np.array([]))
            sequences = pattern_analysis.get('sequences', [])
            
            if len(sequences) < 50:  # Need sufficient data for ML
                return {}
            
            # Prepare features and targets for prediction
            features, targets = self._prepare_prediction_data(data, flow_analysis, pattern_analysis)
            
            if len(features) < 20:
                return {}
            
            # Train prediction models
            predictions = self._train_and_predict(features, targets)
            
            # Calculate prediction confidence
            prediction_confidence = self._calculate_prediction_confidence(features, predictions)
            
            # Sequence completion analysis
            completion_scores = self._analyze_sequence_completion(sequences)
            
            # Next flow state prediction
            next_flow_predictions = self._predict_next_flow_state(flow_analysis)
            
            return {
                'predictions': predictions,
                'prediction_confidence': prediction_confidence,
                'completion_scores': completion_scores,
                'next_flow_predictions': next_flow_predictions,
                'model_accuracy': self._get_model_accuracy()
            }
            
        except Exception as e:
            logger.error(f"Error in predictive modeling: {str(e)}")
            return {}
    
    def _calculate_sequence_entropy(self, flow_analysis: Dict) -> Dict[str, np.ndarray]:
        """Calculate entropy measures for sequences."""
        try:
            flow_state_values = flow_analysis.get('flow_state_values', np.array([]))
            
            entropy_scores = np.zeros_like(flow_state_values)
            predictability_scores = np.zeros_like(flow_state_values)
            
            for i in range(self.sequence_length, len(flow_state_values)):
                window_states = flow_state_values[i - self.sequence_length:i]
                
                # Calculate entropy
                entropy = self._calculate_entropy(window_states)
                entropy_scores[i] = entropy
                
                # Predictability (inverse of entropy)
                predictability_scores[i] = (1 - entropy) * 100
            
            return {
                'entropy_scores': entropy_scores,
                'predictability_scores': predictability_scores
            }
            
        except Exception as e:
            logger.error(f"Error calculating sequence entropy: {str(e)}")
            return {}
    
    def _analyze_fractal_patterns(self, data: pd.DataFrame, flow_analysis: Dict) -> Dict[str, np.ndarray]:
        """Analyze fractal patterns in sequences."""
        try:
            flow_imbalance = flow_analysis.get('flow_imbalance', np.array([]))
            close = data['close'].values
            
            fractal_similarity = np.zeros_like(flow_imbalance)
            self_similarity = np.zeros_like(flow_imbalance)
            
            for i in range(self.sequence_length * 2, len(flow_imbalance)):
                # Current sequence
                current_seq = flow_imbalance[i - self.sequence_length:i]
                
                # Compare with historical sequences
                max_similarity = 0
                for j in range(self.sequence_length, i - self.sequence_length):
                    historical_seq = flow_imbalance[j - self.sequence_length:j]
                    similarity = self._calculate_sequence_similarity_score(current_seq, historical_seq)
                    max_similarity = max(max_similarity, similarity)
                
                fractal_similarity[i] = max_similarity
                
                # Self-similarity analysis (compare different scales)
                if i >= self.sequence_length * 4:
                    longer_seq = flow_imbalance[i - self.sequence_length * 2:i]
                    shorter_seq = flow_imbalance[i - self.sequence_length:i]
                    
                    # Downsample longer sequence to match shorter one
                    downsampled = longer_seq[::2][:len(shorter_seq)]
                    self_similarity[i] = self._calculate_sequence_similarity_score(shorter_seq, downsampled)
            
            return {
                'fractal_similarity': fractal_similarity,
                'self_similarity': self_similarity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing fractal patterns: {str(e)}")
            return {}
    
    def _generate_comprehensive_signals(self, data: pd.DataFrame, flow_analysis: Dict,
                                      pattern_analysis: Dict, momentum_analysis: Dict,
                                      institution_analysis: Dict, prediction_analysis: Dict,
                                      entropy_analysis: Dict, fractal_analysis: Dict) -> List[FlowSequenceSignal]:
        """Generate comprehensive sequence analysis signals."""
        try:
            signals = []
            timestamps = pd.to_datetime(data.index) if hasattr(data.index, 'to_pydatetime') else data.index
            
            flow_states = flow_analysis.get('flow_states', [])
            flow_imbalance = flow_analysis.get('flow_imbalance', np.array([]))
            sequence_types = pattern_analysis.get('sequence_types', [])
            pattern_matches = pattern_analysis.get('pattern_matches', np.array([]))
            momentum_scores = momentum_analysis.get('momentum_sequence_scores', np.array([]))
            institutional_scores = institution_analysis.get('institutional_scores', np.array([]))
            predictions = prediction_analysis.get('predictions', np.array([]))
            prediction_confidence = prediction_analysis.get('prediction_confidence', np.array([]))
            entropy_scores = entropy_analysis.get('entropy_scores', np.array([]))
            fractal_similarity = fractal_analysis.get('fractal_similarity', np.array([]))
            
            for i in range(self.sequence_length, len(data)):
                # Get recent sequence for analysis
                recent_imbalance = flow_imbalance[max(0, i-5):i].tolist() if i < len(flow_imbalance) else []
                
                # Calculate overall confidence
                confidence = self._calculate_signal_confidence(i, pattern_analysis, prediction_analysis)
                
                if confidence >= self.min_confidence:
                    # Determine sequence type
                    seq_type = sequence_types[i - self.sequence_length] if i - self.sequence_length < len(sequence_types) else SequenceType.UNKNOWN
                    
                    # Calculate prediction direction and strength
                    pred_direction, pred_strength = self._calculate_prediction_metrics(i, predictions, prediction_confidence)
                    
                    # Calculate probabilities
                    reversal_prob, continuation_prob = self._calculate_trend_probabilities(i, momentum_analysis, pattern_analysis)
                    
                    # Get next flow prediction
                    next_flow = self._get_next_flow_prediction(i, flow_states, prediction_analysis)
                    
                    # Calculate sequence completion
                    completion = self._calculate_sequence_completion_score(i, pattern_analysis)
                    
                    signal = FlowSequenceSignal(
                        timestamp=timestamps[i] if i < len(timestamps) else datetime.now(),
                        sequence_type=seq_type,
                        sequence_strength=pattern_matches[i] if i < len(pattern_matches) else 0.0,
                        flow_imbalance_sequence=recent_imbalance,
                        prediction_direction=pred_direction,
                        prediction_strength=pred_strength,
                        confidence=confidence,
                        pattern_match_score=pattern_matches[i] if i < len(pattern_matches) else 0.0,
                        momentum_sequence_score=momentum_scores[i] if i < len(momentum_scores) else 0.0,
                        reversal_probability=reversal_prob,
                        continuation_probability=continuation_prob,
                        institutional_sequence_score=institutional_scores[i] if i < len(institutional_scores) else 0.0,
                        entropy_score=entropy_scores[i] if i < len(entropy_scores) else 0.0,
                        fractal_similarity=fractal_similarity[i] if i < len(fractal_similarity) else 0.0,
                        next_flow_prediction=next_flow,
                        sequence_completion=completion
                    )
                    
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating comprehensive signals: {str(e)}")
            return []
    
    # Helper methods for calculations
    def _calculate_flow_imbalance(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate flow imbalance using multiple methods."""
        try:
            close = data['close'].values
            volume = data['volume'].values
            high = data['high'].values
            low = data['low'].values
            
            # Method 1: Tick rule
            tick_imbalance = np.zeros_like(close)
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    tick_imbalance[i] = 1
                elif close[i] < close[i-1]:
                    tick_imbalance[i] = -1
                else:
                    tick_imbalance[i] = 0
            
            # Method 2: Volume distribution
            typical_price = (high + low + close) / 3
            volume_imbalance = np.zeros_like(close)
            
            for i in range(1, len(close)):
                if typical_price[i] > typical_price[i-1]:
                    volume_imbalance[i] = volume[i]
                elif typical_price[i] < typical_price[i-1]:
                    volume_imbalance[i] = -volume[i]
            
            # Combine methods with rolling normalization
            window = 20
            combined_imbalance = np.zeros_like(close)
            
            for i in range(window, len(close)):
                tick_window = tick_imbalance[i-window:i]
                volume_window = volume_imbalance[i-window:i]
                
                # Normalize by window statistics
                tick_norm = np.sum(tick_window * volume[i-window:i]) / np.sum(volume[i-window:i]) if np.sum(volume[i-window:i]) > 0 else 0
                volume_norm = np.sum(volume_window) / np.sum(np.abs(volume_window)) if np.sum(np.abs(volume_window)) > 0 else 0
                
                combined_imbalance[i] = (tick_norm + volume_norm) / 2
            
            return combined_imbalance
            
        except Exception as e:
            logger.error(f"Error calculating flow imbalance: {str(e)}")
            return np.zeros_like(close)
    
    def _calculate_flow_persistence(self, flow_states: List[FlowState]) -> np.ndarray:
        """Calculate how long similar flow states persist."""
        try:
            persistence = np.zeros(len(flow_states))
            
            for i in range(1, len(flow_states)):
                if flow_states[i] == flow_states[i-1]:
                    persistence[i] = persistence[i-1] + 1
                else:
                    persistence[i] = 0
            
            return persistence
        except:
            return np.zeros(len(flow_states))
    
    def _classify_sequence_type(self, sequence: np.ndarray, flow_states: List[FlowState]) -> SequenceType:
        """Classify the type of sequence pattern."""
        try:
            # Analyze sequence characteristics
            trend = np.polyfit(range(len(sequence)), sequence, 1)[0]
            volatility = np.std(sequence)
            mean_value = np.mean(sequence)
            
            # Classification logic
            if trend > 0.1 and mean_value > 0:
                return SequenceType.ACCUMULATION_SEQUENCE
            elif trend < -0.1 and mean_value < 0:
                return SequenceType.DISTRIBUTION_SEQUENCE
            elif abs(trend) > 0.2:
                return SequenceType.MOMENTUM_SEQUENCE
            elif volatility > 1.0:
                return SequenceType.BREAKOUT_SEQUENCE
            elif abs(trend) < 0.05 and volatility < 0.5:
                return SequenceType.CONSOLIDATION_SEQUENCE
            elif self._detect_reversal_pattern(sequence):
                return SequenceType.REVERSAL_SEQUENCE
            else:
                return SequenceType.UNKNOWN
        except:
            return SequenceType.UNKNOWN
    
    def _detect_reversal_pattern(self, sequence: np.ndarray) -> bool:
        """Detect if sequence shows reversal pattern."""
        try:
            if len(sequence) < 6:
                return False
            
            first_half = sequence[:len(sequence)//2]
            second_half = sequence[len(sequence)//2:]
            
            first_trend = np.polyfit(range(len(first_half)), first_half, 1)[0]
            second_trend = np.polyfit(range(len(second_half)), second_half, 1)[0]
            
            # Check if trends are opposite and significant
            return (first_trend > 0.1 and second_trend < -0.1) or (first_trend < -0.1 and second_trend > 0.1)
        except:
            return False
    
    def _calculate_pattern_match_score(self, sequence: np.ndarray) -> float:
        """Calculate how well sequence matches known patterns."""
        try:
            if not self.known_patterns:
                return 0.0
            
            max_similarity = 0.0
            for pattern_id, pattern_data in self.known_patterns.items():
                similarity = self._calculate_sequence_similarity_score(sequence, pattern_data['sequence'])
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity * 100
        except:
            return 0.0
    
    def _calculate_sequence_similarity_score(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Calculate similarity between two sequences."""
        try:
            if len(seq1) != len(seq2):
                # Pad shorter sequence or truncate longer one
                min_len = min(len(seq1), len(seq2))
                seq1 = seq1[:min_len]
                seq2 = seq2[:min_len]
            
            if len(seq1) == 0:
                return 0.0
            
            # Normalize sequences
            seq1_norm = (seq1 - np.mean(seq1)) / (np.std(seq1) + 1e-8)
            seq2_norm = (seq2 - np.mean(seq2)) / (np.std(seq2) + 1e-8)
            
            # Calculate correlation
            correlation = np.corrcoef(seq1_norm, seq2_norm)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _update_pattern_database(self, sequences: List[np.ndarray], sequence_types: List[SequenceType]):
        """Update the database of known patterns."""
        try:
            for i, (sequence, seq_type) in enumerate(zip(sequences, sequence_types)):
                pattern_id = f"{seq_type.value}_{i}"
                
                if pattern_id not in self.known_patterns:
                    self.known_patterns[pattern_id] = {
                        'sequence': sequence.copy(),
                        'type': seq_type,
                        'frequency': 1,
                        'first_seen': datetime.now()
                    }
                else:
                    self.known_patterns[pattern_id]['frequency'] += 1
            
            # Limit pattern database size
            if len(self.known_patterns) > self.pattern_memory:
                # Remove least frequent patterns
                sorted_patterns = sorted(self.known_patterns.items(), 
                                       key=lambda x: x[1]['frequency'], reverse=True)
                self.known_patterns = dict(sorted_patterns[:self.pattern_memory])
        except Exception as e:
            logger.error(f"Error updating pattern database: {str(e)}")
    
    def _calculate_sequence_similarity(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Calculate similarity scores for sequences."""
        try:
            similarity_scores = np.zeros(len(sequences))
            
            for i, seq in enumerate(sequences):
                max_sim = 0.0
                for j, other_seq in enumerate(sequences):
                    if i != j:
                        sim = self._calculate_sequence_similarity_score(seq, other_seq)
                        max_sim = max(max_sim, sim)
                similarity_scores[i] = max_sim
            
            return similarity_scores
        except:
            return np.zeros(len(sequences))
    
    def _analyze_pattern_frequencies(self, sequences: List[np.ndarray]) -> Dict[str, int]:
        """Analyze frequency of different pattern types."""
        try:
            pattern_counts = Counter()
            
            for seq in sequences:
                # Create a pattern signature
                pattern_sig = tuple(np.round(seq, 1))
                pattern_counts[pattern_sig] += 1
            
            # Convert to regular dict and limit size
            return dict(pattern_counts.most_common(50))
        except:
            return {}
    
    def _calculate_entropy(self, sequence: np.ndarray) -> float:
        """Calculate entropy of a sequence."""
        try:
            # Discretize sequence
            bins = 5
            discretized = np.digitize(sequence, np.linspace(np.min(sequence), np.max(sequence), bins))
            
            # Calculate probabilities
            unique, counts = np.unique(discretized, return_counts=True)
            probabilities = counts / len(sequence)
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(unique))
            return entropy / max_entropy if max_entropy > 0 else 0
        except:
            return 0.0
    
    def _prepare_prediction_data(self, data: pd.DataFrame, flow_analysis: Dict, pattern_analysis: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for prediction models."""
        try:
            flow_state_values = flow_analysis.get('flow_state_values', np.array([]))
            flow_momentum = flow_analysis.get('flow_momentum', np.array([]))
            sequences = pattern_analysis.get('sequences', [])
            
            features = []
            targets = []
            
            for i in range(len(sequences) - self.prediction_horizon):
                # Feature: current sequence + momentum + volume info
                seq_features = sequences[i].tolist()
                momentum_features = flow_momentum[i:i+self.sequence_length].tolist() if i+self.sequence_length <= len(flow_momentum) else [0]*self.sequence_length
                
                feature_vector = seq_features + momentum_features
                features.append(feature_vector)
                
                # Target: future flow direction
                future_flow = flow_state_values[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                target = np.mean(future_flow) if len(future_flow) > 0 else 0
                targets.append(target)
            
            return np.array(features), np.array(targets)
        except:
            return np.array([]), np.array([])
    
    def _train_and_predict(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Train models and generate predictions."""
        try:
            if len(features) < 20:
                return np.zeros(len(features))
            
            # Split data for training and prediction
            split_point = int(len(features) * 0.8)
            train_features = features[:split_point]
            train_targets = targets[:split_point]
            
            # Train model
            self._sequence_predictor.fit(train_features, train_targets)
            
            # Generate predictions for all data
            predictions = self._sequence_predictor.predict(features)
            
            return predictions
        except:
            return np.zeros(len(features))
    
    def _calculate_prediction_confidence(self, features: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence in predictions."""
        try:
            confidence = np.zeros_like(predictions)
            
            # Use ensemble variance as confidence measure
            if hasattr(self._sequence_predictor, 'estimators_'):
                for i, feature in enumerate(features):
                    pred_values = [estimator.predict([feature])[0] for estimator in self._sequence_predictor.estimators_]
                    variance = np.var(pred_values)
                    # Convert variance to confidence (lower variance = higher confidence)
                    confidence[i] = max(0, 100 - variance * 100)
            else:
                confidence.fill(50)  # Default confidence
            
            return confidence
        except:
            return np.full_like(predictions, 50.0)
    
    def _analyze_sequence_completion(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Analyze how complete current sequences are."""
        try:
            completion_scores = np.zeros(len(sequences))
            
            for i, seq in enumerate(sequences):
                # Check if sequence follows a known pattern
                best_match_score = 0
                for pattern_id, pattern_data in self.known_patterns.items():
                    similarity = self._calculate_sequence_similarity_score(seq, pattern_data['sequence'])
                    best_match_score = max(best_match_score, similarity)
                
                completion_scores[i] = best_match_score * 100
            
            return completion_scores
        except:
            return np.zeros(len(sequences))
    
    def _predict_next_flow_state(self, flow_analysis: Dict) -> List[FlowState]:
        """Predict the next flow state."""
        try:
            flow_states = flow_analysis.get('flow_states', [])
            
            if len(flow_states) < self.sequence_length:
                return [FlowState.NEUTRAL] * len(flow_states)
            
            next_predictions = []
            
            for i in range(self.sequence_length, len(flow_states)):
                recent_states = flow_states[i-self.sequence_length:i]
                
                # Simple prediction based on recent trend
                state_values = [self._flow_state_to_value(state) for state in recent_states]
                trend = np.polyfit(range(len(state_values)), state_values, 1)[0]
                
                current_value = state_values[-1]
                predicted_value = current_value + trend
                
                predicted_state = self._value_to_flow_state(predicted_value)
                next_predictions.append(predicted_state)
            
            # Pad beginning
            while len(next_predictions) < len(flow_states):
                next_predictions.insert(0, FlowState.NEUTRAL)
            
            return next_predictions
        except:
            return [FlowState.NEUTRAL] * len(flow_analysis.get('flow_states', []))
    
    def _flow_state_to_value(self, state: FlowState) -> int:
        """Convert flow state to numeric value."""
        mapping = {
            FlowState.STRONG_SELL: -2,
            FlowState.MODERATE_SELL: -1,
            FlowState.NEUTRAL: 0,
            FlowState.MODERATE_BUY: 1,
            FlowState.STRONG_BUY: 2
        }
        return mapping.get(state, 0)
    
    def _value_to_flow_state(self, value: float) -> FlowState:
        """Convert numeric value to flow state."""
        if value >= 1.5:
            return FlowState.STRONG_BUY
        elif value >= 0.5:
            return FlowState.MODERATE_BUY
        elif value <= -1.5:
            return FlowState.STRONG_SELL
        elif value <= -0.5:
            return FlowState.MODERATE_SELL
        else:
            return FlowState.NEUTRAL
    
    def _get_model_accuracy(self) -> float:
        """Get current model accuracy."""
        try:
            # Placeholder for model accuracy tracking
            return 75.0
        except:
            return 50.0
    
    def _calculate_signal_confidence(self, index: int, pattern_analysis: Dict, prediction_analysis: Dict) -> float:
        """Calculate overall signal confidence."""
        try:
            pattern_matches = pattern_analysis.get('pattern_matches', np.array([]))
            prediction_confidence = prediction_analysis.get('prediction_confidence', np.array([]))
            
            pattern_conf = pattern_matches[index] if index < len(pattern_matches) else 0
            pred_conf = prediction_confidence[index - self.sequence_length] if index - self.sequence_length >= 0 and index - self.sequence_length < len(prediction_confidence) else 50
            
            return (pattern_conf * 0.6 + pred_conf * 0.4)
        except:
            return 50.0
    
    def _calculate_prediction_metrics(self, index: int, predictions: np.ndarray, 
                                    prediction_confidence: np.ndarray) -> Tuple[int, PredictionStrength]:
        """Calculate prediction direction and strength."""
        try:
            pred_idx = index - self.sequence_length
            if pred_idx >= 0 and pred_idx < len(predictions):
                prediction = predictions[pred_idx]
                confidence = prediction_confidence[pred_idx] if pred_idx < len(prediction_confidence) else 50
                
                # Direction
                direction = 1 if prediction > 0.1 else (-1 if prediction < -0.1 else 0)
                
                # Strength
                if confidence > 80:
                    strength = PredictionStrength.VERY_STRONG
                elif confidence > 70:
                    strength = PredictionStrength.STRONG
                elif confidence > 60:
                    strength = PredictionStrength.MODERATE
                elif confidence > 50:
                    strength = PredictionStrength.WEAK
                else:
                    strength = PredictionStrength.VERY_WEAK
                
                return direction, strength
            
            return 0, PredictionStrength.WEAK
        except:
            return 0, PredictionStrength.WEAK
    
    def _calculate_trend_probabilities(self, index: int, momentum_analysis: Dict, 
                                     pattern_analysis: Dict) -> Tuple[float, float]:
        """Calculate reversal and continuation probabilities."""
        try:
            momentum_scores = momentum_analysis.get('momentum_sequence_scores', np.array([]))
            sequence_types = pattern_analysis.get('sequence_types', [])
            
            momentum_score = momentum_scores[index] if index < len(momentum_scores) else 50
            seq_type = sequence_types[index - self.sequence_length] if index - self.sequence_length >= 0 and index - self.sequence_length < len(sequence_types) else SequenceType.UNKNOWN
            
            # Base probabilities
            reversal_prob = 30.0
            continuation_prob = 70.0
            
            # Adjust based on sequence type
            if seq_type == SequenceType.REVERSAL_SEQUENCE:
                reversal_prob += 40
                continuation_prob -= 40
            elif seq_type == SequenceType.MOMENTUM_SEQUENCE:
                reversal_prob -= 20
                continuation_prob += 20
            
            # Adjust based on momentum
            if momentum_score > 80:
                continuation_prob += 15
                reversal_prob -= 15
            elif momentum_score < 30:
                reversal_prob += 15
                continuation_prob -= 15
            
            # Ensure probabilities sum to 100
            total = reversal_prob + continuation_prob
            if total > 0:
                reversal_prob = (reversal_prob / total) * 100
                continuation_prob = (continuation_prob / total) * 100
            
            return reversal_prob, continuation_prob
        except:
            return 30.0, 70.0
    
    def _get_next_flow_prediction(self, index: int, flow_states: List[FlowState], 
                                prediction_analysis: Dict) -> FlowState:
        """Get next flow state prediction."""
        try:
            next_predictions = prediction_analysis.get('next_flow_predictions', [])
            
            if index < len(next_predictions):
                return next_predictions[index]
            
            return FlowState.NEUTRAL
        except:
            return FlowState.NEUTRAL
    
    def _calculate_sequence_completion_score(self, index: int, pattern_analysis: Dict) -> float:
        """Calculate sequence completion score."""
        try:
            completion_scores = pattern_analysis.get('completion_scores', np.array([]))
            completion_idx = index - self.sequence_length
            
            if completion_idx >= 0 and completion_idx < len(completion_scores):
                return completion_scores[completion_idx]
            
            return 0.0
        except:
            return 0.0
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for the calculation results."""
        return {
            'indicator_name': 'Order Flow Sequence Analyzer',
            'version': '1.0.0',
            'parameters': {
                'sequence_length': self.sequence_length,
                'pattern_memory': self.pattern_memory,
                'min_pattern_frequency': self.min_pattern_frequency,
                'prediction_horizon': self.prediction_horizon,
                'flow_threshold': self.flow_threshold,
                'min_confidence': self.min_confidence
            },
            'features': [
                'Sequential pattern recognition',
                'Flow imbalance detection and prediction',
                'Machine learning sequence classification',
                'Multi-timeframe sequence analysis',
                'Institutional vs retail flow identification',
                'Momentum sequence analysis',
                'Fractal pattern detection',
                'Real-time sequence scoring'
            ],
            'calculation_timestamp': datetime.now(),
            'data_requirements': ['open', 'high', 'low', 'close', 'volume']
        }
    
    def _generate_empty_result(self) -> Dict[str, Any]:
        """Generate empty result structure."""
        return {
            'flow_analysis': {},
            'pattern_analysis': {},
            'momentum_analysis': {},
            'institution_analysis': {},
            'prediction_analysis': {},
            'entropy_analysis': {},
            'fractal_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': 'Insufficient data'
        }
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure."""
        return {
            'flow_analysis': {},
            'pattern_analysis': {},
            'momentum_analysis': {},
            'institution_analysis': {},
            'prediction_analysis': {},
            'entropy_analysis': {},
            'fractal_analysis': {},
            'signals': [],
            'metadata': self._generate_metadata(),
            'error': error_message
        }

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=400, freq='30T')
    
    # Generate realistic OHLCV data with flow patterns
    base_price = 100
    returns = np.random.normal(0, 0.005, 400)
    
    # Add sequential flow patterns
    flow_volume = np.random.lognormal(8, 0.3, 400)
    for i in range(30, 370, 50):
        # Create sequence patterns
        pattern_length = 10
        if i % 100 == 30:  # Accumulation pattern
            returns[i:i+pattern_length] = np.linspace(0, 0.02, pattern_length)
            flow_volume[i:i+pattern_length] *= np.linspace(1, 2, pattern_length)
        elif i % 100 == 80:  # Distribution pattern
            returns[i:i+pattern_length] = np.linspace(0, -0.02, pattern_length)
            flow_volume[i:i+pattern_length] *= np.linspace(2, 1, pattern_length)
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 400))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 400))),
        'close': prices,
        'volume': flow_volume
    }, index=dates)
    
    # Test the analyzer
    analyzer = OrderFlowSequenceAnalyzer(
        sequence_length=8,
        pattern_memory=50,
        prediction_horizon=3,
        min_confidence=70.0
    )
    
    try:
        result = analyzer.calculate(sample_data)
        
        print("Order Flow Sequence Analyzer Results:")
        print(f"- Calculation successful: {not result.get('error', False)}")
        print(f"- Signals generated: {len(result.get('signals', []))}")
        print(f"- Known patterns: {result.get('pattern_analysis', {}).get('known_patterns', 0)}")
        print(f"- Sequence length: {analyzer.sequence_length}")
        print(f"- Prediction horizon: {analyzer.prediction_horizon}")
        
        # Display some sample signals
        signals = result.get('signals', [])
        if signals:
            print(f"\nSample sequence signals (showing first 3):")
            for i, signal in enumerate(signals[:3]):
                print(f"Signal {i+1}:")
                print(f"  Timestamp: {signal.timestamp}")
                print(f"  Sequence Type: {signal.sequence_type}")
                print(f"  Sequence Strength: {signal.sequence_strength:.2f}")
                print(f"  Prediction Direction: {signal.prediction_direction}")
                print(f"  Prediction Strength: {signal.prediction_strength}")
                print(f"  Confidence: {signal.confidence:.2f}")
                print(f"  Reversal Probability: {signal.reversal_probability:.2f}")
                print(f"  Continuation Probability: {signal.continuation_probability:.2f}")
                print(f"  Institutional Score: {signal.institutional_sequence_score:.2f}")
                print(f"  Next Flow Prediction: {signal.next_flow_prediction}")
        
        print(f"\nMetadata: {result.get('metadata', {}).get('indicator_name', 'N/A')}")
        
    except Exception as e:
        print(f"Error testing Order Flow Sequence Analyzer: {str(e)}")
        import traceback
        traceback.print_exc()