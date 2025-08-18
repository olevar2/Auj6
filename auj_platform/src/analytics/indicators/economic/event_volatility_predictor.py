"""
Event Volatility Predictor
Predicts market volatility based on upcoming economic events and historical patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ....core.data_contracts import EconomicEvent
from ..base.base_indicator import EconomicIndicator

logger = logging.getLogger(__name__)

@dataclass
class VolatilityPrediction:
    """Volatility prediction result"""
    currency_pair: str
    predicted_volatility: float      # Expected volatility multiplier
    confidence_interval: Tuple[float, float]  # (lower, upper) bounds
    prediction_horizon_hours: int    # Hours ahead prediction is valid
    contributing_events: List[str]   # Events contributing to prediction
    risk_level: str                  # LOW, MEDIUM, HIGH, EXTREME

class EventVolatilityPredictor(EconomicIndicator):
    """
    Advanced machine learning-based indicator that predicts market volatility
    based on upcoming economic events, historical patterns, and market conditions.
    """
    
    def __init__(self,
                 prediction_horizon_hours: int = 48,
                 min_confidence: float = 0.6,
                 volatility_window: int = 20):
        """
        Initialize Event Volatility Predictor
        
        Args:
            prediction_horizon_hours: Hours ahead to predict volatility
            min_confidence: Minimum confidence for predictions
            volatility_window: Window for historical volatility calculation
        """
        super().__init__()
        self.prediction_horizon_hours = prediction_horizon_hours
        self.min_confidence = min_confidence
        self.volatility_window = volatility_window
        
        # Pre-trained models for different currency pairs
        self.volatility_models = {}
        self.feature_scalers = {}
        
        # Historical volatility patterns by event type
        self.event_volatility_patterns = {
            'NFP': {
                'base_multiplier': 2.8,
                'duration_hours': 8,
                'peak_offset_hours': 0.5,
                'decay_rate': 0.3
            },
            'CPI': {
                'base_multiplier': 2.2,
                'duration_hours': 6,
                'peak_offset_hours': 0.25,
                'decay_rate': 0.25
            },
            'Interest Rate Decision': {
                'base_multiplier': 3.5,
                'duration_hours': 12,
                'peak_offset_hours': 0,
                'decay_rate': 0.2
            },
            'FOMC': {
                'base_multiplier': 4.0,
                'duration_hours': 16,
                'peak_offset_hours': 0,
                'decay_rate': 0.15
            },
            'GDP': {
                'base_multiplier': 1.8,
                'duration_hours': 4,
                'peak_offset_hours': 0.5,
                'decay_rate': 0.35
            },
            'Retail Sales': {
                'base_multiplier': 1.5,
                'duration_hours': 3,
                'peak_offset_hours': 0.5,
                'decay_rate': 0.4
            }
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize machine learning models for volatility prediction"""
        # Model for major pairs
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
        
        for pair in major_pairs:
            # Random Forest model for volatility prediction
            self.volatility_models[pair] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Feature scaler
            self.feature_scalers[pair] = StandardScaler()
        
        # Train models with synthetic data (in real implementation, use historical data)
        self._train_models_with_synthetic_data()
    
    def _train_models_with_synthetic_data(self):
        """Train models with synthetic historical data"""
        try:
            for pair in self.volatility_models.keys():
                # Generate synthetic training data
                X_train, y_train = self._generate_synthetic_training_data(pair)
                
                # Scale features
                X_train_scaled = self.feature_scalers[pair].fit_transform(X_train)
                
                # Train model
                self.volatility_models[pair].fit(X_train_scaled, y_train)
                
        except Exception as e:
            logger.error(f"Error training volatility models: {e}")
    
    def _generate_synthetic_training_data(self, pair: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for model training"""
        n_samples = 1000
        
        # Features: [event_count, max_impact_score, time_to_event_hours, 
        #           current_volatility, market_session, day_of_week]
        X = np.random.rand(n_samples, 6)
        
        # Synthetic target volatility (would be real historical volatility in production)
        y = np.random.lognormal(0, 0.5, n_samples) + 1.0  # Base volatility of 1.0
        
        # Add some realistic patterns
        for i in range(n_samples):
            event_impact = X[i, 1]  # Max impact score
            if event_impact > 0.8:  # High impact event
                y[i] *= (1.5 + event_impact)
            elif event_impact > 0.5:  # Medium impact event
                y[i] *= (1.2 + event_impact * 0.5)
        
        return X, y
    
    async def calculate(self, data: pd.DataFrame,
                       economic_events: List[EconomicEvent] = None,
                       pair: str = 'EURUSD') -> Dict:
        """
        Predict volatility based on economic events and market conditions
        
        Args:
            data: Price data DataFrame
            economic_events: List of upcoming economic events
            pair: Currency pair
            
        Returns:
            Dictionary containing volatility predictions
        """
        try:
            if economic_events is None:
                economic_events = []
            
            current_time = datetime.now()
            
            # Filter relevant events
            relevant_events = self._filter_events_by_timeframe(
                economic_events, current_time
            )
            
            # Calculate current market volatility
            current_volatility = self._calculate_current_volatility(data)
            
            # Extract features for prediction
            features = self._extract_prediction_features(
                relevant_events, current_volatility, current_time, pair
            )
            
            # Make volatility prediction
            prediction = await self._predict_volatility(features, pair)
            
            # Calculate confidence intervals
            confidence_interval = self._calculate_confidence_interval(
                prediction, features, pair
            )
            
            # Identify contributing events
            contributing_events = [
                event.name for event in relevant_events
                if event.impact_level in ['HIGH', 'CRITICAL']
            ]
            
            # Determine risk level
            risk_level = self._classify_risk_level(prediction)
            
            # Create prediction object
            volatility_prediction = VolatilityPrediction(
                currency_pair=pair,
                predicted_volatility=prediction,
                confidence_interval=confidence_interval,
                prediction_horizon_hours=self.prediction_horizon_hours,
                contributing_events=contributing_events,
                risk_level=risk_level
            )
            
            # Generate detailed analysis
            detailed_analysis = await self._generate_detailed_analysis(
                relevant_events, prediction, current_volatility, pair
            )
            
            return {
                'prediction': volatility_prediction,
                'detailed_analysis': detailed_analysis,
                'current_volatility': current_volatility,
                'event_count': len(relevant_events),
                'high_impact_events': len([e for e in relevant_events 
                                         if e.impact_level in ['HIGH', 'CRITICAL']]),
                'calculation_time': current_time,
                'model_confidence': self._calculate_model_confidence(features, pair)
            }
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return self._generate_error_result(str(e))
    
    def _filter_events_by_timeframe(self, events: List[EconomicEvent], 
                                   current_time: datetime) -> List[EconomicEvent]:
        """Filter events within prediction timeframe"""
        end_time = current_time + timedelta(hours=self.prediction_horizon_hours)
        
        return [
            event for event in events
            if current_time <= event.time <= end_time
            and event.impact_level in ['MEDIUM', 'HIGH', 'CRITICAL']
        ]
    
    def _calculate_current_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current market volatility"""
        if len(data) < self.volatility_window:
            return 1.0  # Default volatility
        
        # Calculate True Range-based volatility
        high = data['high'].tail(self.volatility_window)
        low = data['low'].tail(self.volatility_window)
        close = data['close'].tail(self.volatility_window)
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        current_volatility = true_range.mean() / close.mean()
        
        return max(current_volatility, 0.001)  # Minimum volatility
    
    def _extract_prediction_features(self, events: List[EconomicEvent],
                                   current_volatility: float,
                                   current_time: datetime,
                                   pair: str) -> np.ndarray:
        """Extract features for volatility prediction"""
        # Count events by impact level
        high_impact_count = len([e for e in events if e.impact_level == 'HIGH'])
        critical_impact_count = len([e for e in events if e.impact_level == 'CRITICAL'])
        
        # Calculate maximum impact score
        impact_scores = []
        for event in events:
            score = {'CRITICAL': 1.0, 'HIGH': 0.8, 'MEDIUM': 0.5, 'LOW': 0.2}.get(
                event.impact_level, 0.1
            )
            impact_scores.append(score)
        
        max_impact_score = max(impact_scores) if impact_scores else 0.0
        
        # Time to nearest high-impact event
        high_impact_events = [e for e in events if e.impact_level in ['HIGH', 'CRITICAL']]
        if high_impact_events:
            nearest_event_time = min(high_impact_events, key=lambda x: x.time).time
            hours_to_event = (nearest_event_time - current_time).total_seconds() / 3600
        else:
            hours_to_event = self.prediction_horizon_hours
        
        # Market session factor (simplified)
        hour_of_day = current_time.hour
        session_factor = self._get_session_factor(hour_of_day)
        
        # Day of week factor
        day_of_week = current_time.weekday()  # 0 = Monday
        
        # Features array
        features = np.array([
            len(events),              # Total event count
            max_impact_score,         # Maximum impact score
            hours_to_event,           # Time to nearest high-impact event
            current_volatility,       # Current market volatility
            session_factor,           # Market session factor
            day_of_week              # Day of week
        ]).reshape(1, -1)
        
        return features
    
    def _get_session_factor(self, hour: int) -> float:
        """Get session activity factor based on hour of day (UTC)"""
        # London session (8-17 UTC): Higher activity
        if 8 <= hour <= 17:
            return 1.0
        # New York session (13-22 UTC): Highest activity
        elif 13 <= hour <= 22:
            return 1.2
        # Tokyo session (0-9 UTC): Medium activity
        elif 0 <= hour <= 9:
            return 0.8
        # Low activity periods
        else:
            return 0.6
    
    async def _predict_volatility(self, features: np.ndarray, pair: str) -> float:
        """Make volatility prediction using trained model"""
        try:
            # Use model for specific pair or fallback to EURUSD model
            model_pair = pair if pair in self.volatility_models else 'EURUSD'
            
            model = self.volatility_models[model_pair]
            scaler = self.feature_scalers[model_pair]
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Ensure prediction is within reasonable bounds
            prediction = max(0.5, min(prediction, 5.0))
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in volatility prediction: {e}")
            return 1.0  # Default prediction
    
    def _calculate_confidence_interval(self, prediction: float,
                                     features: np.ndarray,
                                     pair: str) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        try:
            # Use ensemble of trees for uncertainty estimation
            model_pair = pair if pair in self.volatility_models else 'EURUSD'
            model = self.volatility_models[model_pair]
            scaler = self.feature_scalers[model_pair]
            
            features_scaled = scaler.transform(features)
            
            # Get predictions from individual trees
            tree_predictions = []
            for tree in model.estimators_:
                tree_pred = tree.predict(features_scaled)[0]
                tree_predictions.append(tree_pred)
            
            # Calculate confidence interval
            std_pred = np.std(tree_predictions)
            lower_bound = max(0.1, prediction - 1.96 * std_pred)
            upper_bound = min(10.0, prediction + 1.96 * std_pred)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return (prediction * 0.8, prediction * 1.2)
    
    def _classify_risk_level(self, predicted_volatility: float) -> str:
        """Classify risk level based on predicted volatility"""
        if predicted_volatility >= 3.0:
            return 'EXTREME'
        elif predicted_volatility >= 2.0:
            return 'HIGH'
        elif predicted_volatility >= 1.5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    async def _generate_detailed_analysis(self, events: List[EconomicEvent],
                                        prediction: float,
                                        current_volatility: float,
                                        pair: str) -> Dict:
        """Generate detailed volatility analysis"""
        analysis = {
            'volatility_change': prediction / current_volatility if current_volatility > 0 else 1.0,
            'event_contributions': [],
            'time_series_forecast': [],
            'risk_factors': []
        }
        
        # Analyze individual event contributions
        for event in events:
            pattern = self.event_volatility_patterns.get(
                event.name,
                {'base_multiplier': 1.2, 'duration_hours': 2, 'peak_offset_hours': 0.5}
            )
            
            contribution = self._calculate_event_contribution(event, pattern)
            analysis['event_contributions'].append({
                'event_name': event.name,
                'currency': event.currency,
                'impact_level': event.impact_level,
                'contribution': contribution,
                'time': event.time
            })
        
        # Generate time series forecast
        analysis['time_series_forecast'] = self._generate_volatility_forecast(
            events, prediction, current_volatility
        )
        
        # Identify risk factors
        if prediction > current_volatility * 2:
            analysis['risk_factors'].append('High volatility spike expected')
        
        if len([e for e in events if e.impact_level == 'CRITICAL']) > 0:
            analysis['risk_factors'].append('Critical impact events present')
        
        if len(events) > 5:
            analysis['risk_factors'].append('Multiple events clustered')
        
        return analysis
    
    def _calculate_event_contribution(self, event: EconomicEvent, pattern: Dict) -> float:
        """Calculate individual event contribution to volatility"""
        base_impact = {
            'CRITICAL': 1.0,
            'HIGH': 0.8,
            'MEDIUM': 0.5,
            'LOW': 0.2
        }.get(event.impact_level, 0.1)
        
        multiplier = pattern.get('base_multiplier', 1.2)
        
        return base_impact * multiplier * 0.3  # Scale to contribution factor
    
    def _generate_volatility_forecast(self, events: List[EconomicEvent],
                                    base_prediction: float,
                                    current_volatility: float) -> List[Dict]:
        """Generate hourly volatility forecast"""
        forecast = []
        current_time = datetime.now()
        
        for hour in range(min(24, self.prediction_horizon_hours)):
            forecast_time = current_time + timedelta(hours=hour)
            volatility = current_volatility
            
            # Apply event impacts
            for event in events:
                time_diff = abs((forecast_time - event.time).total_seconds() / 3600)
                
                if time_diff <= 8:  # Within impact window
                    pattern = self.event_volatility_patterns.get(
                        event.name,
                        {'base_multiplier': 1.2, 'duration_hours': 2, 'decay_rate': 0.3}
                    )
                    
                    # Calculate impact based on time distance
                    impact_factor = np.exp(-pattern['decay_rate'] * time_diff)
                    event_multiplier = 1 + (pattern['base_multiplier'] - 1) * impact_factor
                    
                    volatility *= event_multiplier
            
            forecast.append({
                'time': forecast_time,
                'predicted_volatility': min(volatility, 5.0),  # Cap at 5x
                'hour_offset': hour
            })
        
        return forecast
    
    def _calculate_model_confidence(self, features: np.ndarray, pair: str) -> float:
        """Calculate model confidence based on feature values"""
        # Simple confidence calculation based on feature ranges
        try:
            feature_vector = features.flatten()
            
            # Check if features are within training range
            confidence = 1.0
            
            # Reduce confidence for extreme values
            if feature_vector[1] > 0.9:  # Very high impact score
                confidence *= 0.8
            
            if feature_vector[2] < 1:  # Very near-term event
                confidence *= 0.9
            
            if feature_vector[0] > 10:  # Too many events
                confidence *= 0.7
            
            return max(confidence, 0.3)
            
        except Exception:
            return 0.5  # Default confidence
    
    def _generate_error_result(self, error_message: str) -> Dict:
        """Generate error result"""
        return {
            'prediction': None,
            'detailed_analysis': None,
            'current_volatility': 1.0,
            'event_count': 0,
            'high_impact_events': 0,
            'calculation_time': datetime.now(),
            'model_confidence': 0.0,
            'error': error_message
        }