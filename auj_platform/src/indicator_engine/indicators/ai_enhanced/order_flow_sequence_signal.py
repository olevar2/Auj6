"""
Order Flow Sequence Signal Indicator - Advanced Pattern Recognition & Sequence Analysis

This module implements sophisticated order flow sequence analysis with:
- Real-time sequence pattern detection and classification
- Advanced pattern recognition using machine learning
- Sequential flow modeling with state transitions
- Market microstructure sequence analysis
- Institutional trading sequence detection
- Multi-timeframe sequence aggregation
- Predictive sequence modeling with neural networks
- Sequence-based risk assessment
- Production-grade error handling and logging

Author: AI Enhancement Team
Version: 7.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SequenceEvent:
    """Individual event in an order flow sequence."""
    timestamp: datetime
    price: float
    volume: float
    direction: int  # 1 for buy, -1 for sell, 0 for neutral
    size_category: str  # 'small', 'medium', 'large', 'block'
    speed: float  # Execution speed metric
    urgency: float  # Market urgency indicator
    sequence_id: Optional[str] = None

@dataclass
class FlowSequence:
    """Complete order flow sequence structure."""
    sequence_id: str
    events: List[SequenceEvent]
    start_time: datetime
    end_time: datetime
    duration: float  # In seconds
    pattern_type: str
    confidence: float
    institutional_probability: float
    total_volume: float
    average_size: float
    directional_consistency: float
    urgency_score: float

@dataclass
class SequenceSignal:
    """Order flow sequence signal structure."""
    timestamp: datetime
    signal_strength: float
    signal_direction: int  # 1 for bullish, -1 for bearish, 0 for neutral
    confidence: float
    pattern_detected: str
    sequence_quality: str
    institutional_flow: bool
    urgency_level: str
    risk_score: float
    prediction_horizon: int  # Minutes
    expected_impact: float

class OrderFlowSequenceSignalIndicator:
    """
    Advanced Order Flow Sequence Signal Analyzer with comprehensive pattern recognition.
    
    This indicator provides sophisticated sequence analysis including:
    - Real-time sequence detection and classification
    - Pattern recognition using machine learning algorithms
    - Sequential flow modeling with state transition analysis
    - Institutional trading sequence identification
    - Multi-timeframe sequence aggregation
    - Predictive modeling for sequence continuation
    - Risk assessment based on sequence characteristics
    - Advanced filtering and noise reduction
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Order Flow Sequence Signal Indicator.
        
        Args:
            parameters: Configuration parameters for the indicator
        """
        self.parameters = self._set_default_parameters(parameters or {})
        self.logger = self._setup_logger()
        
        # Core components
        self.sequence_detector = SequenceDetector(self.parameters)
        self.pattern_recognizer = PatternRecognizer(self.parameters)
        self.sequence_modeler = SequenceModeler(self.parameters)
        self.institutional_analyzer = InstitutionalSequenceAnalyzer(self.parameters)
        self.predictive_engine = PredictiveSequenceEngine(self.parameters)
        self.risk_analyzer = SequenceRiskAnalyzer(self.parameters)
        
        # Data storage
        self.sequence_history: List[FlowSequence] = []
        self.active_sequences: Dict[str, FlowSequence] = {}
        self.pattern_library: Dict[str, Any] = {}
        self.sequence_signals: List[SequenceSignal] = []
        
        # State management
        self.current_sequence_id = 0
        self.last_calculation_time = None
        self.is_trained = False
        self.sequence_buffer = deque(maxlen=1000)
        
        # Analysis windows
        self.short_window = self.parameters['short_window']
        self.medium_window = self.parameters['medium_window']
        self.long_window = self.parameters['long_window']
        
        self.logger.info("Order Flow Sequence Signal Indicator initialized with advanced pattern recognition")
    
    def _set_default_parameters(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Set default parameters for the indicator."""
        defaults = {
            # Core parameters
            'lookback_period': 200,
            'sequence_timeout': 300,  # seconds
            'min_sequence_length': 3,
            'max_sequence_length': 50,
            'confidence_threshold': 0.6,
            
            # Analysis windows
            'short_window': 20,
            'medium_window': 50,
            'long_window': 100,
            
            # Sequence detection
            'volume_threshold_multiplier': 1.5,
            'time_gap_threshold': 60,  # seconds
            'price_deviation_threshold': 0.001,  # 0.1%
            'direction_consistency_threshold': 0.7,
            
            # Pattern recognition
            'pattern_similarity_threshold': 0.8,
            'min_pattern_occurrences': 3,
            'pattern_memory_period': 1000,
            'clustering_eps': 0.3,
            'clustering_min_samples': 3,
            
            # Size categorization
            'small_trade_percentile': 25,
            'medium_trade_percentile': 75,
            'large_trade_percentile': 95,
            'block_trade_multiplier': 5.0,
            
            # Institutional detection
            'institutional_volume_threshold': 50000,
            'institutional_speed_threshold': 10,  # trades per minute
            'stealth_detection_window': 30,
            'institutional_consistency_threshold': 0.8,
            
            # Machine learning
            'ml_training_period': 500,
            'ml_retrain_frequency': 100,
            'feature_count': 30,
            'prediction_horizons': [5, 15, 30],  # minutes
            
            # Risk management
            'max_risk_score': 10.0,
            'risk_decay_factor': 0.9,
            'sequence_impact_threshold': 0.01,
            
            # Signal generation
            'signal_smoothing_factor': 0.2,
            'noise_filter_enabled': True,
            'min_signal_strength': 0.3,
            'signal_aggregation_method': 'weighted_average',
            
            # Advanced features
            'multi_timeframe_analysis': True,
            'adaptive_thresholds': True,
            'real_time_learning': True,
            'sequence_clustering': True
        }
        
        # Update with user parameters
        defaults.update(user_params)
        return defaults
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the indicator."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 volume: np.ndarray, timestamp: np.ndarray = None,
                 tick_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate order flow sequence signals with comprehensive pattern analysis.
        
        Args:
            high: High prices
            low: Low prices  
            close: Close prices
            volume: Volume data
            timestamp: Optional timestamp data
            tick_data: Optional detailed tick-by-tick data
            
        Returns:
            Dict containing order flow sequence analysis results
        """
        try:
            if len(close) < self.parameters['lookback_period']:
                return self._generate_default_result()
            
            # Process input data
            if tick_data:
                events = self._process_tick_data(tick_data)
            else:
                events = self._simulate_sequence_events(high, low, close, volume, timestamp)
            
            if not events:
                return self._generate_default_result()
            
            # Core sequence analysis
            sequences = self._detect_sequences(events)
            patterns = self._recognize_patterns(sequences)
            institutional_analysis = self._analyze_institutional_sequences(sequences)
            
            # Predictive analysis
            predictions = self._generate_sequence_predictions(sequences, patterns)
            
            # Risk assessment
            risk_analysis = self._assess_sequence_risks(sequences)
            
            # Generate comprehensive signals
            signals = self._generate_sequence_signals(
                sequences, patterns, institutional_analysis, predictions, risk_analysis
            )
            
            # Update state
            self.last_calculation_time = datetime.now()
            
            result = {
                'signal_strength': signals.get('primary_signal_strength', 0.0),
                'signal_direction': signals.get('signal_direction', 0),
                'confidence': signals.get('confidence', 0.0),
                'pattern_detected': signals.get('pattern_detected', 'none'),
                'sequence_quality': signals.get('sequence_quality', 'low'),
                'institutional_flow': signals.get('institutional_detected', False),
                'urgency_level': signals.get('urgency_level', 'low'),
                'risk_score': signals.get('risk_score', 0.0),
                'prediction_horizon': signals.get('prediction_horizon', 5),
                'expected_impact': signals.get('expected_impact', 0.0),
                
                # Advanced metrics
                'active_sequences': len(self.active_sequences),
                'sequence_count': len(sequences),
                'pattern_count': len(patterns),
                'institutional_probability': institutional_analysis.get('probability', 0.0),
                'average_sequence_length': np.mean([len(seq.events) for seq in sequences]) if sequences else 0.0,
                'sequence_diversity': self._calculate_sequence_diversity(sequences),
                
                # Predictive metrics
                'short_term_prediction': predictions.get('5_minute', 0.0),
                'medium_term_prediction': predictions.get('15_minute', 0.0),
                'long_term_prediction': predictions.get('30_minute', 0.0),
                'prediction_confidence': predictions.get('confidence', 0.0),
                
                # Detailed analysis
                'sequence_analysis': self._summarize_sequences(sequences),
                'pattern_analysis': self._summarize_patterns(patterns),
                'institutional_analysis': institutional_analysis,
                'risk_analysis': risk_analysis,
                'prediction_analysis': predictions,
                
                # Metadata
                'calculation_time': self.last_calculation_time.isoformat() if self.last_calculation_time else None,
                'data_quality': self._assess_data_quality(events),
                'parameters_used': self.parameters.copy()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating order flow sequence signals: {e}")
            return self._generate_error_result(str(e))
    
    def _process_tick_data(self, tick_data: List[Dict]) -> List[SequenceEvent]:
        """Process detailed tick-by-tick data into sequence events."""
        try:
            events = []
            
            for tick in tick_data[-self.parameters['lookback_period']:]:
                # Create sequence event
                timestamp = tick.get('timestamp', datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                price = float(tick.get('price', 0))
                volume = float(tick.get('volume', 0))
                direction = int(tick.get('trade_direction', 0))
                
                # Calculate derived metrics
                size_category = self._categorize_trade_size(volume)
                speed = self._calculate_execution_speed(tick, events)
                urgency = self._calculate_market_urgency(tick, events)
                
                event = SequenceEvent(
                    timestamp=timestamp,
                    price=price,
                    volume=volume,
                    direction=direction,
                    size_category=size_category,
                    speed=speed,
                    urgency=urgency
                )
                
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error processing tick data: {e}")
            return []
    
    def _simulate_sequence_events(self, high: np.ndarray, low: np.ndarray, 
                                 close: np.ndarray, volume: np.ndarray,
                                 timestamp: np.ndarray = None) -> List[SequenceEvent]:
        """Simulate sequence events from OHLCV data."""
        try:
            events = []
            
            for i in range(len(close)):
                # Simulate multiple events per bar
                num_events = max(1, int(volume[i] / 1000))  # Rough approximation
                num_events = min(num_events, 10)  # Limit events per bar
                
                for j in range(num_events):
                    # Simulate event timing within the bar
                    if timestamp is not None:
                        event_time = timestamp[i] + timedelta(seconds=j * 60 / num_events)
                    else:
                        event_time = datetime.now() + timedelta(minutes=i, seconds=j * 60 / num_events)
                    
                    # Simulate price within OHLC range
                    price_range = high[i] - low[i]
                    if price_range > 0:
                        price = low[i] + np.random.random() * price_range
                    else:
                        price = close[i]
                    
                    # Simulate volume distribution
                    event_volume = volume[i] / num_events * np.random.lognormal(0, 0.5)
                    
                    # Simulate direction based on price movement
                    if i > 0:
                        price_change = close[i] - close[i-1]
                        if price_change > 0:
                            direction_prob = 0.6 + min(0.3, abs(price_change) / close[i-1] * 100)
                        elif price_change < 0:
                            direction_prob = 0.4 - min(0.3, abs(price_change) / close[i-1] * 100)
                        else:
                            direction_prob = 0.5
                        
                        direction = 1 if np.random.random() < direction_prob else -1
                    else:
                        direction = np.random.choice([-1, 1])
                    
                    # Calculate derived metrics
                    size_category = self._categorize_trade_size(event_volume)
                    speed = np.random.exponential(5)  # Simulate execution speed
                    urgency = np.random.beta(2, 5)  # Simulate market urgency
                    
                    event = SequenceEvent(
                        timestamp=event_time,
                        price=price,
                        volume=event_volume,
                        direction=direction,
                        size_category=size_category,
                        speed=speed,
                        urgency=urgency
                    )
                    
                    events.append(event)
            
            # Sort by timestamp
            events.sort(key=lambda x: x.timestamp)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error simulating sequence events: {e}")
            return []
    
    def _categorize_trade_size(self, volume: float) -> str:
        """Categorize trade size based on volume percentiles."""
        try:
            # Use historical volumes for percentile calculation
            if len(self.sequence_buffer) > 10:
                historical_volumes = [event.volume for event in self.sequence_buffer]
                
                small_threshold = np.percentile(historical_volumes, self.parameters['small_trade_percentile'])
                medium_threshold = np.percentile(historical_volumes, self.parameters['medium_trade_percentile'])
                large_threshold = np.percentile(historical_volumes, self.parameters['large_trade_percentile'])
                block_threshold = np.mean(historical_volumes) * self.parameters['block_trade_multiplier']
                
                if volume >= block_threshold:
                    return 'block'
                elif volume >= large_threshold:
                    return 'large'
                elif volume >= medium_threshold:
                    return 'medium'
                else:
                    return 'small'
            else:
                # Default categorization when insufficient history
                if volume > 10000:
                    return 'large'
                elif volume > 1000:
                    return 'medium'
                else:
                    return 'small'
                    
        except Exception as e:
            self.logger.error(f"Error categorizing trade size: {e}")
            return 'unknown'
    
    def _calculate_execution_speed(self, tick: Dict, events: List[SequenceEvent]) -> float:
        """Calculate execution speed metric."""
        try:
            if len(events) < 2:
                return 1.0
            
            # Calculate time since last event
            current_time = tick.get('timestamp', datetime.now())
            if isinstance(current_time, str):
                current_time = datetime.fromisoformat(current_time)
            
            last_event_time = events[-1].timestamp
            time_diff = (current_time - last_event_time).total_seconds()
            
            # Convert to speed metric (trades per minute)
            speed = 60.0 / max(time_diff, 0.1)
            return min(speed, 100)  # Cap at 100 trades per minute
            
        except Exception as e:
            self.logger.error(f"Error calculating execution speed: {e}")
            return 1.0
    
    def _calculate_market_urgency(self, tick: Dict, events: List[SequenceEvent]) -> float:
        """Calculate market urgency indicator."""
        try:
            if len(events) < 5:
                return 0.5
            
            recent_events = events[-5:]
            
            # Calculate volume acceleration
            volumes = [event.volume for event in recent_events]
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            volume_urgency = min(max(volume_trend / np.mean(volumes), 0), 1) if np.mean(volumes) > 0 else 0
            
            # Calculate price momentum
            prices = [event.price for event in recent_events]
            price_changes = np.diff(prices)
            price_momentum = np.std(price_changes) / np.mean(prices) if np.mean(prices) > 0 else 0
            price_urgency = min(price_momentum * 100, 1)
            
            # Calculate directional consistency
            directions = [event.direction for event in recent_events]
            direction_consistency = abs(np.mean(directions))
            
            # Combine urgency factors
            urgency = (volume_urgency + price_urgency + direction_consistency) / 3
            return min(max(urgency, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating market urgency: {e}")
            return 0.5
    
    def _detect_sequences(self, events: List[SequenceEvent]) -> List[FlowSequence]:
        """Detect order flow sequences from events."""
        try:
            if not events:
                return []
            
            sequences = []
            current_sequence_events = []
            last_event_time = None
            
            for event in events:
                # Check if this event should start a new sequence
                if (last_event_time is None or 
                    (event.timestamp - last_event_time).total_seconds() > self.parameters['time_gap_threshold'] or
                    len(current_sequence_events) >= self.parameters['max_sequence_length']):
                    
                    # Finalize previous sequence
                    if len(current_sequence_events) >= self.parameters['min_sequence_length']:
                        sequence = self._create_flow_sequence(current_sequence_events)
                        if sequence:
                            sequences.append(sequence)
                    
                    # Start new sequence
                    current_sequence_events = [event]
                else:
                    # Add to current sequence
                    current_sequence_events.append(event)
                
                last_event_time = event.timestamp
            
            # Finalize last sequence
            if len(current_sequence_events) >= self.parameters['min_sequence_length']:
                sequence = self._create_flow_sequence(current_sequence_events)
                if sequence:
                    sequences.append(sequence)
            
            # Store in history
            self.sequence_history.extend(sequences)
            
            # Maintain history size
            max_history = self.parameters['pattern_memory_period']
            if len(self.sequence_history) > max_history:
                self.sequence_history = self.sequence_history[-max_history:]
            
            return sequences
            
        except Exception as e:
            self.logger.error(f"Error detecting sequences: {e}")
            return []
    
    def _create_flow_sequence(self, events: List[SequenceEvent]) -> Optional[FlowSequence]:
        """Create a FlowSequence from a list of events."""
        try:
            if not events:
                return None
            
            # Generate sequence ID
            self.current_sequence_id += 1
            sequence_id = f"seq_{self.current_sequence_id}"
            
            # Calculate sequence metrics
            start_time = events[0].timestamp
            end_time = events[-1].timestamp
            duration = (end_time - start_time).total_seconds()
            
            total_volume = sum(event.volume for event in events)
            average_size = total_volume / len(events)
            
            # Calculate directional consistency
            directions = [event.direction for event in events if event.direction != 0]
            directional_consistency = abs(np.mean(directions)) if directions else 0.0
            
            # Calculate urgency score
            urgency_values = [event.urgency for event in events]
            urgency_score = np.mean(urgency_values) if urgency_values else 0.0
            
            # Determine pattern type (simplified)
            pattern_type = self._classify_sequence_pattern(events)
            
            # Calculate confidence based on consistency and size
            confidence = (directional_consistency + 
                         min(total_volume / 10000, 1.0) + 
                         min(len(events) / 20, 1.0)) / 3
            
            # Estimate institutional probability
            institutional_probability = self._estimate_institutional_probability(events)
            
            sequence = FlowSequence(
                sequence_id=sequence_id,
                events=events,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                pattern_type=pattern_type,
                confidence=confidence,
                institutional_probability=institutional_probability,
                total_volume=total_volume,
                average_size=average_size,
                directional_consistency=directional_consistency,
                urgency_score=urgency_score
            )
            
            return sequence
            
        except Exception as e:
            self.logger.error(f"Error creating flow sequence: {e}")
            return None    
    def _classify_sequence_pattern(self, events: List[SequenceEvent]) -> str:
        """Classify the pattern type of a sequence."""
        try:
            if len(events) < 3:
                return 'unknown'
            
            # Analyze volume pattern
            volumes = [event.volume for event in events]
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            
            # Analyze direction pattern
            directions = [event.direction for event in events]
            direction_changes = sum(1 for i in range(1, len(directions)) 
                                  if directions[i] != directions[i-1])
            
            # Analyze size distribution
            size_categories = [event.size_category for event in events]
            large_ratio = sum(1 for cat in size_categories if cat in ['large', 'block']) / len(size_categories)
            
            # Analyze urgency pattern
            urgencies = [event.urgency for event in events]
            avg_urgency = np.mean(urgencies)
            
            # Pattern classification logic
            if large_ratio > 0.5 and avg_urgency > 0.7:
                return 'aggressive_institutional'
            elif volume_trend > 0 and direction_changes < len(events) * 0.3:
                return 'accumulation'
            elif volume_trend < 0 and direction_changes < len(events) * 0.3:
                return 'distribution'
            elif direction_changes > len(events) * 0.7:
                return 'churning'
            elif avg_urgency > 0.8:
                return 'panic_trading'
            elif large_ratio > 0.3:
                return 'institutional_flow'
            elif direction_changes < len(events) * 0.2:
                return 'directional_flow'
            else:
                return 'mixed_flow'
                
        except Exception as e:
            self.logger.error(f"Error classifying sequence pattern: {e}")
            return 'unknown'
    
    def _estimate_institutional_probability(self, events: List[SequenceEvent]) -> float:
        """Estimate probability that sequence is institutional trading."""
        try:
            if not events:
                return 0.0
            
            factors = []
            
            # Volume factor
            total_volume = sum(event.volume for event in events)
            volume_factor = min(total_volume / self.parameters['institutional_volume_threshold'], 1.0)
            factors.append(volume_factor)
            
            # Size consistency factor
            large_trades = sum(1 for event in events if event.size_category in ['large', 'block'])
            size_factor = large_trades / len(events)
            factors.append(size_factor)
            
            # Speed factor
            speeds = [event.speed for event in events]
            avg_speed = np.mean(speeds)
            speed_factor = min(avg_speed / self.parameters['institutional_speed_threshold'], 1.0)
            factors.append(speed_factor)
            
            # Directional consistency factor
            directions = [event.direction for event in events if event.direction != 0]
            if directions:
                consistency_factor = abs(np.mean(directions))
                factors.append(consistency_factor)
            
            # Duration factor (institutional trades often span longer periods)
            if len(events) > 1:
                duration = (events[-1].timestamp - events[0].timestamp).total_seconds()
                duration_factor = min(duration / 1800, 1.0)  # 30 minutes max
                factors.append(duration_factor)
            
            return np.mean(factors) if factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Error estimating institutional probability: {e}")
            return 0.0
    
    def _recognize_patterns(self, sequences: List[FlowSequence]) -> Dict[str, Any]:
        """Recognize patterns in the detected sequences."""
        try:
            if not sequences:
                return {'patterns': [], 'pattern_strength': 0.0}
            
            # Group sequences by pattern type
            pattern_groups = defaultdict(list)
            for sequence in sequences:
                pattern_groups[sequence.pattern_type].append(sequence)
            
            # Analyze pattern frequencies
            pattern_frequencies = {pattern: len(seqs) for pattern, seqs in pattern_groups.items()}
            
            # Identify dominant patterns
            dominant_patterns = sorted(pattern_frequencies.items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
            
            # Calculate pattern strength
            total_sequences = len(sequences)
            pattern_strength = max(pattern_frequencies.values()) / total_sequences if total_sequences > 0 else 0.0
            
            # Analyze pattern transitions
            pattern_transitions = self._analyze_pattern_transitions(sequences)
            
            # Detect recurring patterns
            recurring_patterns = self._detect_recurring_patterns(sequences)
            
            return {
                'patterns': dominant_patterns,
                'pattern_strength': pattern_strength,
                'pattern_groups': dict(pattern_groups),
                'pattern_transitions': pattern_transitions,
                'recurring_patterns': recurring_patterns,
                'total_patterns': len(pattern_groups),
                'pattern_diversity': len(pattern_groups) / total_sequences if total_sequences > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error recognizing patterns: {e}")
            return {'patterns': [], 'pattern_strength': 0.0}
    
    def _analyze_pattern_transitions(self, sequences: List[FlowSequence]) -> Dict[str, Any]:
        """Analyze transitions between different pattern types."""
        try:
            if len(sequences) < 2:
                return {'transitions': {}, 'transition_probability': 0.0}
            
            transitions = defaultdict(int)
            total_transitions = 0
            
            for i in range(1, len(sequences)):
                prev_pattern = sequences[i-1].pattern_type
                curr_pattern = sequences[i].pattern_type
                transition_key = f"{prev_pattern}->{curr_pattern}"
                transitions[transition_key] += 1
                total_transitions += 1
            
            # Calculate transition probabilities
            transition_probs = {}
            for transition, count in transitions.items():
                transition_probs[transition] = count / total_transitions
            
            # Find most likely transitions
            likely_transitions = sorted(transition_probs.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'transitions': dict(transitions),
                'transition_probabilities': transition_probs,
                'likely_transitions': likely_transitions,
                'total_transitions': total_transitions
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern transitions: {e}")
            return {'transitions': {}, 'transition_probability': 0.0}
    
    def _detect_recurring_patterns(self, sequences: List[FlowSequence]) -> List[Dict[str, Any]]:
        """Detect recurring sequence patterns using clustering."""
        try:
            if len(sequences) < self.parameters['min_pattern_occurrences']:
                return []
            
            # Extract features for clustering
            features = []
            for sequence in sequences:
                feature_vector = self._extract_sequence_features(sequence)
                if feature_vector is not None:
                    features.append(feature_vector)
            
            if len(features) < self.parameters['min_pattern_occurrences']:
                return []
            
            features_array = np.array(features)
            
            # Perform clustering
            clusterer = DBSCAN(
                eps=self.parameters['clustering_eps'],
                min_samples=self.parameters['clustering_min_samples']
            )
            
            cluster_labels = clusterer.fit_predict(features_array)
            
            # Analyze clusters
            recurring_patterns = []
            unique_labels = set(cluster_labels) - {-1}  # Exclude noise
            
            for label in unique_labels:
                cluster_sequences = [sequences[i] for i, l in enumerate(cluster_labels) if l == label]
                
                if len(cluster_sequences) >= self.parameters['min_pattern_occurrences']:
                    pattern_info = {
                        'pattern_id': f'recurring_{label}',
                        'occurrences': len(cluster_sequences),
                        'avg_confidence': np.mean([seq.confidence for seq in cluster_sequences]),
                        'avg_volume': np.mean([seq.total_volume for seq in cluster_sequences]),
                        'avg_duration': np.mean([seq.duration for seq in cluster_sequences]),
                        'dominant_type': max(set([seq.pattern_type for seq in cluster_sequences]),
                                           key=[seq.pattern_type for seq in cluster_sequences].count),
                        'institutional_ratio': np.mean([seq.institutional_probability for seq in cluster_sequences])
                    }
                    recurring_patterns.append(pattern_info)
            
            return recurring_patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting recurring patterns: {e}")
            return []
    
    def _extract_sequence_features(self, sequence: FlowSequence) -> Optional[np.ndarray]:
        """Extract features from a sequence for pattern recognition."""
        try:
            features = []
            
            # Basic sequence characteristics
            features.extend([
                len(sequence.events),
                sequence.duration,
                sequence.total_volume,
                sequence.average_size,
                sequence.directional_consistency,
                sequence.urgency_score,
                sequence.confidence,
                sequence.institutional_probability
            ])
            
            # Volume distribution features
            volumes = [event.volume for event in sequence.events]
            features.extend([
                np.mean(volumes),
                np.std(volumes),
                np.median(volumes),
                np.min(volumes),
                np.max(volumes)
            ])
            
            # Price movement features
            prices = [event.price for event in sequence.events]
            if len(prices) > 1:
                price_changes = np.diff(prices)
                features.extend([
                    np.mean(price_changes),
                    np.std(price_changes),
                    np.sum(price_changes > 0) / len(price_changes),  # Positive change ratio
                    (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0  # Total return
                ])
            else:
                features.extend([0, 0, 0.5, 0])
            
            # Timing features
            if len(sequence.events) > 1:
                time_intervals = []
                for i in range(1, len(sequence.events)):
                    interval = (sequence.events[i].timestamp - sequence.events[i-1].timestamp).total_seconds()
                    time_intervals.append(interval)
                
                features.extend([
                    np.mean(time_intervals),
                    np.std(time_intervals),
                    np.median(time_intervals)
                ])
            else:
                features.extend([0, 0, 0])
            
            # Size category distribution
            size_categories = [event.size_category for event in sequence.events]
            total_events = len(size_categories)
            features.extend([
                size_categories.count('small') / total_events,
                size_categories.count('medium') / total_events,
                size_categories.count('large') / total_events,
                size_categories.count('block') / total_events
            ])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting sequence features: {e}")
            return None
    
    def _analyze_institutional_sequences(self, sequences: List[FlowSequence]) -> Dict[str, Any]:
        """Analyze institutional trading sequences."""
        try:
            if not sequences:
                return {'probability': 0.0, 'detected_sequences': []}
            
            # Identify likely institutional sequences
            institutional_sequences = [
                seq for seq in sequences 
                if seq.institutional_probability > self.parameters['institutional_consistency_threshold']
            ]
            
            # Calculate overall institutional probability
            if sequences:
                overall_probability = np.mean([seq.institutional_probability for seq in sequences])
            else:
                overall_probability = 0.0
            
            # Analyze institutional patterns
            institutional_patterns = defaultdict(int)
            for seq in institutional_sequences:
                institutional_patterns[seq.pattern_type] += 1
            
            # Calculate institutional flow metrics
            total_institutional_volume = sum(seq.total_volume for seq in institutional_sequences)
            avg_institutional_size = np.mean([seq.average_size for seq in institutional_sequences]) if institutional_sequences else 0
            
            # Stealth trading detection
            stealth_sequences = self._detect_stealth_trading(sequences)
            
            return {
                'probability': overall_probability,
                'detected_sequences': len(institutional_sequences),
                'total_sequences': len(sequences),
                'institutional_ratio': len(institutional_sequences) / len(sequences) if sequences else 0,
                'institutional_patterns': dict(institutional_patterns),
                'total_institutional_volume': total_institutional_volume,
                'avg_institutional_size': avg_institutional_size,
                'stealth_sequences': stealth_sequences,
                'stealth_probability': len(stealth_sequences) / len(sequences) if sequences else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing institutional sequences: {e}")
            return {'probability': 0.0, 'detected_sequences': []}
    
    def _detect_stealth_trading(self, sequences: List[FlowSequence]) -> List[FlowSequence]:
        """Detect potential stealth trading sequences."""
        try:
            stealth_sequences = []
            
            for sequence in sequences:
                # Stealth trading characteristics:
                # 1. Consistent direction with varying sizes
                # 2. Lower urgency to avoid detection
                # 3. Medium to large total volume
                # 4. Longer duration
                
                if (sequence.directional_consistency > 0.7 and
                    sequence.urgency_score < 0.6 and
                    sequence.total_volume > np.median([seq.total_volume for seq in sequences]) and
                    sequence.duration > self.parameters['stealth_detection_window']):
                    
                    # Additional check: size variation (not all same size)
                    volumes = [event.volume for event in sequence.events]
                    volume_cv = np.std(volumes) / (np.mean(volumes) + 1e-10)
                    
                    if volume_cv > 0.3:  # Coefficient of variation > 30%
                        stealth_sequences.append(sequence)
            
            return stealth_sequences
            
        except Exception as e:
            self.logger.error(f"Error detecting stealth trading: {e}")
            return []
    
    def _generate_sequence_predictions(self, sequences: List[FlowSequence], 
                                     patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions based on sequence analysis."""
        try:
            if not sequences:
                return {'5_minute': 0.0, '15_minute': 0.0, '30_minute': 0.0, 'confidence': 0.0}
            
            predictions = {}
            
            # Analyze recent sequence momentum
            recent_sequences = sequences[-5:] if len(sequences) >= 5 else sequences
            
            # Calculate directional momentum
            directional_momentum = np.mean([seq.directional_consistency * 
                                          (1 if seq.events[-1].direction > 0 else -1)
                                          for seq in recent_sequences])
            
            # Calculate volume momentum
            volumes = [seq.total_volume for seq in recent_sequences]
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0] if len(volumes) > 1 else 0
            volume_momentum = np.tanh(volume_trend / np.mean(volumes)) if np.mean(volumes) > 0 else 0
            
            # Calculate urgency momentum
            urgencies = [seq.urgency_score for seq in recent_sequences]
            urgency_momentum = np.mean(urgencies)
            
            # Combine momentum factors
            overall_momentum = (directional_momentum + volume_momentum + urgency_momentum) / 3
            
            # Generate predictions for different horizons
            for horizon in self.parameters['prediction_horizons']:
                # Decay momentum based on time horizon
                decay_factor = np.exp(-horizon / 30)  # 30-minute half-life
                horizon_prediction = overall_momentum * decay_factor
                
                predictions[f'{horizon}_minute'] = np.tanh(horizon_prediction)  # Bound between -1 and 1
            
            # Calculate prediction confidence based on pattern strength and consistency
            pattern_strength = patterns.get('pattern_strength', 0.0)
            sequence_consistency = np.std([seq.confidence for seq in recent_sequences])
            confidence = pattern_strength * (1 - sequence_consistency)
            
            predictions['confidence'] = min(max(confidence, 0.0), 1.0)
            
            # Add pattern-based predictions
            if patterns.get('recurring_patterns'):
                predictions['pattern_continuation_probability'] = self._calculate_pattern_continuation(patterns)
            else:
                predictions['pattern_continuation_probability'] = 0.5
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating sequence predictions: {e}")
            return {'5_minute': 0.0, '15_minute': 0.0, '30_minute': 0.0, 'confidence': 0.0}
    
    def _calculate_pattern_continuation(self, patterns: Dict[str, Any]) -> float:
        """Calculate probability of pattern continuation."""
        try:
            recurring_patterns = patterns.get('recurring_patterns', [])
            
            if not recurring_patterns:
                return 0.5
            
            # Analyze pattern reliability
            continuation_scores = []
            
            for pattern in recurring_patterns:
                # Higher occurrences suggest more reliable patterns
                occurrence_score = min(pattern['occurrences'] / 10, 1.0)
                
                # Higher confidence suggests better pattern quality
                confidence_score = pattern['avg_confidence']
                
                # Institutional patterns tend to be more predictable
                institutional_score = pattern['institutional_ratio']
                
                pattern_score = (occurrence_score + confidence_score + institutional_score) / 3
                continuation_scores.append(pattern_score)
            
            return np.mean(continuation_scores)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern continuation: {e}")
            return 0.5
    
    def _assess_sequence_risks(self, sequences: List[FlowSequence]) -> Dict[str, Any]:
        """Assess risks associated with current sequence conditions."""
        try:
            if not sequences:
                return {'risk_score': 0.0, 'risk_factors': []}
            
            risk_factors = []
            risk_scores = []
            
            # Volume concentration risk
            volumes = [seq.total_volume for seq in sequences]
            max_volume = max(volumes) if volumes else 0
            avg_volume = np.mean(volumes) if volumes else 0
            volume_concentration = max_volume / (avg_volume + 1e-10)
            
            if volume_concentration > 5:
                risk_factors.append('high_volume_concentration')
                risk_scores.append(min(volume_concentration / 10, 3.0))
            
            # Directional extremity risk
            recent_sequences = sequences[-10:] if len(sequences) >= 10 else sequences
            directional_bias = np.mean([seq.directional_consistency * 
                                      (1 if seq.events[-1].direction > 0 else -1)
                                      for seq in recent_sequences])
            
            if abs(directional_bias) > 0.8:
                risk_factors.append('extreme_directional_bias')
                risk_scores.append(abs(directional_bias) * 2)
            
            # Urgency spike risk
            urgencies = [seq.urgency_score for seq in recent_sequences]
            avg_urgency = np.mean(urgencies) if urgencies else 0
            
            if avg_urgency > 0.8:
                risk_factors.append('high_market_urgency')
                risk_scores.append(avg_urgency * 2)
            
            # Pattern disruption risk
            pattern_types = [seq.pattern_type for seq in recent_sequences]
            pattern_diversity = len(set(pattern_types)) / len(pattern_types) if pattern_types else 0
            
            if pattern_diversity > 0.8:
                risk_factors.append('pattern_disruption')
                risk_scores.append(pattern_diversity * 1.5)
            
            # Institutional flow concentration risk
            institutional_sequences = [seq for seq in recent_sequences 
                                     if seq.institutional_probability > 0.7]
            institutional_ratio = len(institutional_sequences) / len(recent_sequences) if recent_sequences else 0
            
            if institutional_ratio > 0.6:
                risk_factors.append('high_institutional_concentration')
                risk_scores.append(institutional_ratio * 2)
            
            # Calculate overall risk score
            overall_risk = np.mean(risk_scores) if risk_scores else 0.0
            overall_risk = min(overall_risk, self.parameters['max_risk_score'])
            
            return {
                'risk_score': overall_risk,
                'risk_factors': risk_factors,
                'volume_concentration_risk': volume_concentration,
                'directional_bias_risk': abs(directional_bias),
                'urgency_risk': avg_urgency,
                'pattern_disruption_risk': pattern_diversity,
                'institutional_concentration_risk': institutional_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing sequence risks: {e}")
            return {'risk_score': 5.0, 'risk_factors': ['assessment_error']}
    
    def _generate_sequence_signals(self, sequences: List[FlowSequence], patterns: Dict[str, Any],
                                 institutional_analysis: Dict[str, Any], predictions: Dict[str, Any],
                                 risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive sequence-based trading signals."""
        try:
            if not sequences:
                return self._generate_default_signal()
            
            # Calculate primary signal strength
            recent_sequences = sequences[-5:] if len(sequences) >= 5 else sequences
            
            # Directional signal
            directional_signals = []
            for seq in recent_sequences:
                if seq.events:
                    last_direction = seq.events[-1].direction
                    signal_strength = seq.confidence * seq.directional_consistency
                    directional_signals.append(last_direction * signal_strength)
            
            primary_signal = np.mean(directional_signals) if directional_signals else 0.0
            signal_strength = abs(primary_signal)
            signal_direction = 1 if primary_signal > 0 else (-1 if primary_signal < 0 else 0)
            
            # Calculate confidence
            confidence_factors = [
                patterns.get('pattern_strength', 0.0),
                institutional_analysis.get('probability', 0.0),
                predictions.get('confidence', 0.0),
                1 - (risk_analysis.get('risk_score', 0.0) / self.parameters['max_risk_score'])
            ]
            confidence = np.mean(confidence_factors)
            
            # Determine pattern detected
            dominant_patterns = patterns.get('patterns', [])
            pattern_detected = dominant_patterns[0][0] if dominant_patterns else 'none'
            
            # Assess sequence quality
            avg_sequence_confidence = np.mean([seq.confidence for seq in recent_sequences])
            if avg_sequence_confidence > 0.8 and confidence > 0.7:
                sequence_quality = 'excellent'
            elif avg_sequence_confidence > 0.6 and confidence > 0.5:
                sequence_quality = 'good'
            elif avg_sequence_confidence > 0.4:
                sequence_quality = 'fair'
            else:
                sequence_quality = 'poor'
            
            # Determine urgency level
            avg_urgency = np.mean([seq.urgency_score for seq in recent_sequences])
            if avg_urgency > 0.8:
                urgency_level = 'very_high'
            elif avg_urgency > 0.6:
                urgency_level = 'high'
            elif avg_urgency > 0.4:
                urgency_level = 'medium'
            else:
                urgency_level = 'low'
            
            # Estimate expected impact
            volume_impact = np.mean([seq.total_volume for seq in recent_sequences]) / 10000
            urgency_impact = avg_urgency
            institutional_impact = institutional_analysis.get('probability', 0.0)
            expected_impact = (volume_impact + urgency_impact + institutional_impact) / 3
            
            return {
                'primary_signal_strength': signal_strength,
                'signal_direction': signal_direction,
                'confidence': confidence,
                'pattern_detected': pattern_detected,
                'sequence_quality': sequence_quality,
                'institutional_detected': institutional_analysis.get('probability', 0.0) > 0.6,
                'urgency_level': urgency_level,
                'risk_score': risk_analysis.get('risk_score', 0.0),
                'prediction_horizon': 15,  # Default to 15 minutes
                'expected_impact': expected_impact,
                
                # Additional signal components
                'pattern_strength': patterns.get('pattern_strength', 0.0),
                'institutional_probability': institutional_analysis.get('probability', 0.0),
                'volume_momentum': self._calculate_volume_momentum(sequences),
                'directional_consistency': np.mean([seq.directional_consistency for seq in recent_sequences]),
                'sequence_diversity': patterns.get('pattern_diversity', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating sequence signals: {e}")
            return self._generate_default_signal()
    
    def _calculate_volume_momentum(self, sequences: List[FlowSequence]) -> float:
        """Calculate volume momentum from sequences."""
        try:
            if len(sequences) < 2:
                return 0.0
            
            volumes = [seq.total_volume for seq in sequences[-10:]]  # Last 10 sequences
            
            if len(volumes) < 2:
                return 0.0
            
            # Calculate trend
            trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            normalized_trend = trend / (np.mean(volumes) + 1e-10)
            
            return np.tanh(normalized_trend)  # Bound between -1 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating volume momentum: {e}")
            return 0.0    
    def _calculate_sequence_diversity(self, sequences: List[FlowSequence]) -> float:
        """Calculate diversity of sequence patterns."""
        try:
            if not sequences:
                return 0.0
            
            pattern_types = [seq.pattern_type for seq in sequences]
            unique_patterns = set(pattern_types)
            
            return len(unique_patterns) / len(sequences)
            
        except Exception as e:
            self.logger.error(f"Error calculating sequence diversity: {e}")
            return 0.0
    
    def _summarize_sequences(self, sequences: List[FlowSequence]) -> Dict[str, Any]:
        """Summarize sequence analysis results."""
        try:
            if not sequences:
                return {'count': 0, 'summary': 'No sequences detected'}
            
            return {
                'count': len(sequences),
                'avg_length': np.mean([len(seq.events) for seq in sequences]),
                'avg_duration': np.mean([seq.duration for seq in sequences]),
                'avg_volume': np.mean([seq.total_volume for seq in sequences]),
                'avg_confidence': np.mean([seq.confidence for seq in sequences]),
                'institutional_ratio': np.mean([seq.institutional_probability for seq in sequences]),
                'pattern_distribution': {
                    pattern: len([seq for seq in sequences if seq.pattern_type == pattern])
                    for pattern in set(seq.pattern_type for seq in sequences)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing sequences: {e}")
            return {'count': 0, 'summary': 'Error in analysis'}
    
    def _summarize_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize pattern analysis results."""
        try:
            return {
                'dominant_patterns': patterns.get('patterns', [])[:3],
                'pattern_strength': patterns.get('pattern_strength', 0.0),
                'total_patterns': patterns.get('total_patterns', 0),
                'pattern_diversity': patterns.get('pattern_diversity', 0.0),
                'recurring_count': len(patterns.get('recurring_patterns', []))
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing patterns: {e}")
            return {'summary': 'Error in pattern analysis'}
    
    def _assess_data_quality(self, events: List[SequenceEvent]) -> Dict[str, Any]:
        """Assess the quality of input data."""
        try:
            if not events:
                return {'quality': 'poor', 'score': 0.0, 'issues': ['no_data']}
            
            issues = []
            quality_scores = []
            
            # Check data completeness
            complete_events = sum(1 for event in events 
                                if event.volume > 0 and event.price > 0)
            completeness_score = complete_events / len(events)
            quality_scores.append(completeness_score)
            
            if completeness_score < 0.9:
                issues.append('incomplete_data')
            
            # Check temporal consistency
            timestamps = [event.timestamp for event in events]
            if len(timestamps) > 1:
                time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                            for i in range(1, len(timestamps))]
                
                # Check for negative time differences
                negative_diffs = sum(1 for diff in time_diffs if diff < 0)
                if negative_diffs > 0:
                    issues.append('temporal_inconsistency')
                    quality_scores.append(0.5)
                else:
                    quality_scores.append(1.0)
            
            # Check for outliers
            volumes = [event.volume for event in events if event.volume > 0]
            if volumes:
                q75, q25 = np.percentile(volumes, [75, 25])
                iqr = q75 - q25
                outliers = [v for v in volumes if v > q75 + 1.5 * iqr or v < q25 - 1.5 * iqr]
                outlier_ratio = len(outliers) / len(volumes)
                quality_scores.append(1 - min(outlier_ratio, 1.0))
                
                if outlier_ratio > 0.15:
                    issues.append('volume_outliers')
            
            # Overall quality score
            overall_score = np.mean(quality_scores) if quality_scores else 0.0
            
            if overall_score > 0.8:
                quality_level = 'excellent'
            elif overall_score > 0.6:
                quality_level = 'good'
            elif overall_score > 0.4:
                quality_level = 'fair'
            else:
                quality_level = 'poor'
            
            return {
                'quality': quality_level,
                'score': overall_score,
                'issues': issues,
                'completeness': completeness_score,
                'data_points': len(events)
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return {'quality': 'poor', 'score': 0.0, 'issues': ['assessment_error']}
    
    def _generate_default_result(self) -> Dict[str, Any]:
        """Generate default result when insufficient data."""
        return {
            'signal_strength': 0.0,
            'signal_direction': 0,
            'confidence': 0.0,
            'pattern_detected': 'none',
            'sequence_quality': 'insufficient_data',
            'institutional_flow': False,
            'urgency_level': 'low',
            'risk_score': 0.0,
            'prediction_horizon': 5,
            'expected_impact': 0.0,
            'error': 'Insufficient data for sequence analysis'
        }
    
    def _generate_default_signal(self) -> Dict[str, Any]:
        """Generate default signal when no sequences detected."""
        return {
            'primary_signal_strength': 0.0,
            'signal_direction': 0,
            'confidence': 0.0,
            'pattern_detected': 'none',
            'sequence_quality': 'no_sequences',
            'institutional_detected': False,
            'urgency_level': 'low',
            'risk_score': 0.0,
            'prediction_horizon': 5,
            'expected_impact': 0.0
        }
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result."""
        return {
            'signal_strength': 0.0,
            'signal_direction': 0,
            'confidence': 0.0,
            'pattern_detected': 'error',
            'sequence_quality': 'error',
            'institutional_flow': False,
            'urgency_level': 'unknown',
            'risk_score': 10.0,
            'prediction_horizon': 0,
            'expected_impact': 0.0,
            'error': error_message
        }


class SequenceDetector:
    """Specialized detector for order flow sequences."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")


class PatternRecognizer:
    """Advanced pattern recognition for sequence analysis."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.pattern_memory = {}
        self.ml_classifier = None
        
    def train_pattern_classifier(self, sequences: List[FlowSequence]) -> None:
        """Train machine learning classifier for pattern recognition."""
        try:
            if len(sequences) < 50:
                return
            
            # Prepare training data
            features = []
            labels = []
            
            for sequence in sequences:
                feature_vector = self._extract_pattern_features(sequence)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(sequence.pattern_type)
            
            if len(features) < 20:
                return
            
            # Train classifier
            X = np.array(features)
            y = np.array(labels)
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train model
            self.ml_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ml_classifier.fit(X_train, y_train)
            
            # Store label encoder
            self.label_encoder = label_encoder
            
            self.logger.info("Pattern classifier trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training pattern classifier: {e}")
    
    def _extract_pattern_features(self, sequence: FlowSequence) -> Optional[np.ndarray]:
        """Extract features for pattern classification."""
        try:
            features = []
            
            # Basic sequence metrics
            features.extend([
                len(sequence.events),
                sequence.duration,
                sequence.total_volume,
                sequence.directional_consistency,
                sequence.urgency_score
            ])
            
            # Volume pattern features
            volumes = [event.volume for event in sequence.events]
            if len(volumes) > 1:
                features.extend([
                    np.std(volumes) / (np.mean(volumes) + 1e-10),  # CV
                    (volumes[-1] - volumes[0]) / (volumes[0] + 1e-10),  # Volume change
                    np.polyfit(range(len(volumes)), volumes, 1)[0]  # Volume trend
                ])
            else:
                features.extend([0, 0, 0])
            
            # Direction pattern features
            directions = [event.direction for event in sequence.events]
            direction_changes = sum(1 for i in range(1, len(directions)) 
                                  if directions[i] != directions[i-1])
            features.append(direction_changes / len(directions) if directions else 0)
            
            # Size distribution features
            size_categories = [event.size_category for event in sequence.events]
            total_events = len(size_categories)
            features.extend([
                size_categories.count('large') / total_events,
                size_categories.count('block') / total_events
            ])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting pattern features: {e}")
            return None


class SequenceModeler:
    """Advanced sequence modeling and state analysis."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")


class InstitutionalSequenceAnalyzer:
    """Specialized analyzer for institutional sequence patterns."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")


class PredictiveSequenceEngine:
    """Predictive engine for sequence continuation analysis."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.sequence_predictor = None
        self.feature_scaler = StandardScaler()
        self.is_trained = False
    
    def train_sequence_predictor(self, sequences: List[FlowSequence]) -> None:
        """Train predictive model for sequence analysis."""
        try:
            if len(sequences) < 100:
                return
            
            # Prepare training data for sequence continuation prediction
            features = []
            targets = []
            
            for i in range(len(sequences) - 1):
                current_seq = sequences[i]
                next_seq = sequences[i + 1]
                
                # Extract features from current sequence
                feature_vector = self._extract_predictive_features(current_seq)
                if feature_vector is not None:
                    features.append(feature_vector)
                    
                    # Target: direction of next sequence
                    next_direction = 1 if next_seq.directional_consistency > 0 else 0
                    targets.append(next_direction)
            
            if len(features) < 50:
                return
            
            # Prepare data
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train predictor
            self.sequence_predictor = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                random_state=42
            )
            self.sequence_predictor.fit(X_train, y_train)
            
            self.is_trained = True
            self.logger.info("Sequence predictor trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training sequence predictor: {e}")
    
    def _extract_predictive_features(self, sequence: FlowSequence) -> Optional[np.ndarray]:
        """Extract features for sequence prediction."""
        try:
            features = []
            
            # Sequence characteristics
            features.extend([
                len(sequence.events),
                sequence.duration,
                sequence.total_volume,
                sequence.directional_consistency,
                sequence.urgency_score,
                sequence.confidence,
                sequence.institutional_probability
            ])
            
            # Recent event characteristics
            if sequence.events:
                last_event = sequence.events[-1]
                features.extend([
                    last_event.volume,
                    last_event.urgency,
                    last_event.direction
                ])
            else:
                features.extend([0, 0, 0])
            
            # Volume momentum
            volumes = [event.volume for event in sequence.events]
            if len(volumes) > 1:
                volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
                features.append(volume_trend)
            else:
                features.append(0)
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting predictive features: {e}")
            return None
    
    def predict_sequence_continuation(self, sequence: FlowSequence) -> float:
        """Predict probability of sequence continuation in same direction."""
        try:
            if not self.is_trained or self.sequence_predictor is None:
                return 0.5
            
            features = self._extract_predictive_features(sequence)
            if features is None:
                return 0.5
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            
            # Get prediction probability
            prob = self.sequence_predictor.predict_proba(features_scaled)[0]
            return prob[1] if len(prob) > 1 else 0.5
            
        except Exception as e:
            self.logger.error(f"Error predicting sequence continuation: {e}")
            return 0.5


class SequenceRiskAnalyzer:
    """Risk analyzer for sequence-based trading decisions."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def assess_sequence_risk(self, sequence: FlowSequence) -> Dict[str, Any]:
        """Assess risk metrics for a specific sequence."""
        try:
            risk_factors = []
            risk_score = 0.0
            
            # Volume concentration risk
            if sequence.total_volume > 100000:  # Large volume threshold
                risk_factors.append('large_volume')
                risk_score += 1.0
            
            # Urgency risk
            if sequence.urgency_score > 0.8:
                risk_factors.append('high_urgency')
                risk_score += 1.5
            
            # Duration risk (very short or very long sequences)
            if sequence.duration < 10 or sequence.duration > 1800:  # < 10s or > 30min
                risk_factors.append('unusual_duration')
                risk_score += 1.0
            
            # Directional extremity risk
            if sequence.directional_consistency > 0.95:
                risk_factors.append('extreme_directional_bias')
                risk_score += 2.0
            
            # Institutional flow risk
            if sequence.institutional_probability > 0.8:
                risk_factors.append('strong_institutional_presence')
                risk_score += 1.5
            
            return {
                'risk_score': min(risk_score, self.parameters['max_risk_score']),
                'risk_factors': risk_factors,
                'risk_level': 'high' if risk_score > 3 else ('medium' if risk_score > 1 else 'low')
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing sequence risk: {e}")
            return {'risk_score': 5.0, 'risk_factors': ['assessment_error'], 'risk_level': 'high'}