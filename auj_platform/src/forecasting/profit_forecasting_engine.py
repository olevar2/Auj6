"""
Profit Forecasting Engine for the AUJ Platform.

This module provides profit and loss forecasting capabilities using time-series
analysis and machine learning models to predict future performance of trading agents.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..analytics.performance_tracker import PerformanceTracker, ValidationPeriodType
from ..core.logging_setup import get_logger
from ..core.exceptions import ForecastingError

logger = get_logger(__name__)


class ForecastModel(str, Enum):
    """Available forecasting models."""
    LINEAR_REGRESSION = "LINEAR_REGRESSION"
    RANDOM_FOREST = "RANDOM_FOREST"
    SIMPLE_MOVING_AVERAGE = "SIMPLE_MOVING_AVERAGE"
    EXPONENTIAL_SMOOTHING = "EXPONENTIAL_SMOOTHING"


@dataclass
class ForecastResult:
    """Result of a profit forecasting operation."""
    agent_name: str
    forecast_horizon_days: int
    predicted_pnl: Decimal
    confidence_interval_lower: Decimal
    confidence_interval_upper: Decimal
    model_used: ForecastModel
    forecast_timestamp: datetime
    historical_data_points: int
    model_accuracy_score: float
    trend_direction: str  # "UPWARD", "DOWNWARD", "FLAT"
    risk_assessment: str  # "LOW", "MEDIUM", "HIGH"


class ProfitForecastingEngine:
    """
    Advanced profit forecasting engine using time-series analysis.
    
    Provides forecasting capabilities for agent performance prediction,
    risk assessment, and portfolio optimization support.
    """
    
    def __init__(self, performance_tracker: Optional[PerformanceTracker] = None):
        """
        Initialize the profit forecasting engine.
        
        Args:
            performance_tracker: Instance of PerformanceTracker for data access
        """
        self.performance_tracker = performance_tracker
        self.logger = logger
        
        # Model cache
        self.trained_models = {}
        self.last_training_time = {}
        
        # Configuration
        self.min_data_points = 10  # Minimum historical data points required
        self.default_confidence_level = 0.95
        self.model_retrain_interval = timedelta(hours=24)  # Retrain models daily
        
        self.logger.info("Profit Forecasting Engine initialized")
    
    async def forecast_future_pnl(self,
                                 agent_name: str,
                                 horizon_days: int = 7,
                                 model_type: ForecastModel = ForecastModel.LINEAR_REGRESSION,
                                 include_confidence_interval: bool = True) -> Optional[ForecastResult]:
        """
        Forecast future P&L for a specific agent.
        
        Args:
            agent_name: Name of the agent to forecast
            horizon_days: Number of days to forecast ahead
            model_type: Type of forecasting model to use
            include_confidence_interval: Whether to calculate confidence intervals
            
        Returns:
            ForecastResult with prediction and confidence intervals
        """
        try:
            if not self.performance_tracker:
                raise ForecastingError("Performance tracker not available")
            
            self.logger.info(f"Forecasting P&L for agent '{agent_name}' for {horizon_days} days")
            
            # Get historical performance data
            historical_data = await self._get_historical_pnl_data(agent_name)
            
            if historical_data is None or len(historical_data) < self.min_data_points:
                self.logger.warning(f"Insufficient data for agent '{agent_name}' - need at least {self.min_data_points} points")
                return None
            
            # Prepare time series data
            ts_data = self._prepare_time_series_data(historical_data)
            
            # Select and train model
            model = await self._get_or_train_model(agent_name, ts_data, model_type)
            
            # Make prediction
            prediction = self._make_prediction(model, ts_data, horizon_days, model_type)
            
            # Calculate confidence intervals if requested
            if include_confidence_interval:
                lower_bound, upper_bound = self._calculate_confidence_interval(
                    ts_data, prediction, model_type
                )
            else:
                lower_bound = prediction * Decimal('0.8')  # Simple fallback
                upper_bound = prediction * Decimal('1.2')
            
            # Assess trend and risk
            trend_direction = self._assess_trend(ts_data)
            risk_assessment = self._assess_risk(ts_data, prediction)
            
            # Calculate model accuracy
            accuracy_score = self._calculate_model_accuracy(model, ts_data, model_type)
            
            # Create forecast result
            result = ForecastResult(
                agent_name=agent_name,
                forecast_horizon_days=horizon_days,
                predicted_pnl=Decimal(str(prediction)),
                confidence_interval_lower=Decimal(str(lower_bound)),
                confidence_interval_upper=Decimal(str(upper_bound)),
                model_used=model_type,
                forecast_timestamp=datetime.utcnow(),
                historical_data_points=len(historical_data),
                model_accuracy_score=accuracy_score,
                trend_direction=trend_direction,
                risk_assessment=risk_assessment
            )
            
            self.logger.info(f"Forecast completed for '{agent_name}': Predicted P&L = {prediction:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error forecasting P&L for agent '{agent_name}': {e}")
            raise ForecastingError(f"Forecasting failed: {e}")
    
    async def forecast_portfolio_performance(self,
                                           agent_names: List[str],
                                           horizon_days: int = 7) -> Dict[str, ForecastResult]:
        """
        Forecast performance for multiple agents (portfolio view).
        
        Args:
            agent_names: List of agent names to forecast
            horizon_days: Number of days to forecast ahead
            
        Returns:
            Dictionary mapping agent names to their forecast results
        """
        results = {}
        
        for agent_name in agent_names:
            try:
                forecast = await self.forecast_future_pnl(agent_name, horizon_days)
                if forecast:
                    results[agent_name] = forecast
            except Exception as e:
                self.logger.error(f"Failed to forecast for agent '{agent_name}': {e}")
        
        self.logger.info(f"Portfolio forecast completed for {len(results)}/{len(agent_names)} agents")
        return results
    
    async def _get_historical_pnl_data(self, agent_name: str) -> Optional[pd.DataFrame]:
        """Get historical P&L data for an agent."""
        try:
            # Get last 90 days of performance data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=90)
            
            # This would typically call performance_tracker methods
            # For now, we'll simulate with a simple approach
            performance_data = await self.performance_tracker.get_agent_performance(
                agent_name, start_date, end_date
            )
            
            if not performance_data:
                return None
            
            # Convert to DataFrame with cumulative P&L
            df_data = []
            cumulative_pnl = 0
            
            for record in performance_data:
                if hasattr(record, 'pnl') and record.pnl is not None:
                    cumulative_pnl += float(record.pnl)
                    df_data.append({
                        'timestamp': record.exit_time or record.entry_time or datetime.utcnow(),
                        'pnl': float(record.pnl),
                        'cumulative_pnl': cumulative_pnl
                    })
            
            if not df_data:
                return None
            
            df = pd.DataFrame(df_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for '{agent_name}': {e}")
            return None
    
    def _prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data for modeling."""
        try:
            # Add time-based features
            df = df.copy()
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour'] = df['timestamp'].dt.hour
            df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
            
            # Add lag features
            df['pnl_lag1'] = df['cumulative_pnl'].shift(1)
            df['pnl_lag2'] = df['cumulative_pnl'].shift(2)
            df['pnl_lag3'] = df['cumulative_pnl'].shift(3)
            
            # Add moving averages
            df['ma_3'] = df['cumulative_pnl'].rolling(window=3).mean()
            df['ma_7'] = df['cumulative_pnl'].rolling(window=7).mean()
            
            # Add volatility measure
            df['pnl_volatility'] = df['pnl'].rolling(window=5).std()
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing time series data: {e}")
            raise
    
    async def _get_or_train_model(self, agent_name: str, data: pd.DataFrame, model_type: ForecastModel):
        """Get cached model or train a new one."""
        try:
            # Check if we have a cached model that's still valid
            cache_key = f"{agent_name}_{model_type}"
            
            if (cache_key in self.trained_models and 
                cache_key in self.last_training_time and
                datetime.utcnow() - self.last_training_time[cache_key] < self.model_retrain_interval):
                return self.trained_models[cache_key]
            
            # Train new model
            model = self._train_model(data, model_type)
            
            # Cache the model
            self.trained_models[cache_key] = model
            self.last_training_time[cache_key] = datetime.utcnow()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training model for '{agent_name}': {e}")
            raise
    
    def _train_model(self, data: pd.DataFrame, model_type: ForecastModel):
        """Train a forecasting model."""
        try:
            if not SKLEARN_AVAILABLE and model_type in [ForecastModel.LINEAR_REGRESSION, ForecastModel.RANDOM_FOREST]:
                self.logger.warning("Scikit-learn not available, falling back to simple moving average")
                model_type = ForecastModel.SIMPLE_MOVING_AVERAGE
            
            # Prepare features and target
            feature_cols = ['days_since_start', 'day_of_week', 'pnl_lag1', 'pnl_lag2', 'ma_3', 'ma_7']
            features = data[feature_cols].values
            target = data['cumulative_pnl'].values
            
            if model_type == ForecastModel.LINEAR_REGRESSION:
                model = LinearRegression()
                model.fit(features, target)
                
            elif model_type == ForecastModel.RANDOM_FOREST:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(features, target)
                
            elif model_type == ForecastModel.SIMPLE_MOVING_AVERAGE:
                # Simple moving average - no actual model to train
                model = {'type': 'moving_average', 'window': 7}
                
            else:  # EXPONENTIAL_SMOOTHING
                # Simple exponential smoothing
                alpha = 0.3
                model = {'type': 'exponential_smoothing', 'alpha': alpha}
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training {model_type} model: {e}")
            raise
    
    def _make_prediction(self, model, data: pd.DataFrame, horizon_days: int, model_type: ForecastModel) -> float:
        """Make a prediction using the trained model."""
        try:
            if model_type in [ForecastModel.LINEAR_REGRESSION, ForecastModel.RANDOM_FOREST]:
                # Use the last known values to predict
                last_row = data.iloc[-1]
                feature_cols = ['days_since_start', 'day_of_week', 'pnl_lag1', 'pnl_lag2', 'ma_3', 'ma_7']
                
                # Project forward
                future_features = [
                    last_row['days_since_start'] + horizon_days,
                    (last_row['day_of_week'] + horizon_days) % 7,
                    last_row['cumulative_pnl'],
                    last_row['pnl_lag1'],
                    last_row['ma_3'],
                    last_row['ma_7']
                ]
                
                prediction = model.predict([future_features])[0]
                
            elif model_type == ForecastModel.SIMPLE_MOVING_AVERAGE:
                # Simple moving average projection
                recent_pnl = data['cumulative_pnl'].tail(7).mean()
                daily_change = data['pnl'].tail(7).mean()
                prediction = recent_pnl + (daily_change * horizon_days)
                
            else:  # EXPONENTIAL_SMOOTHING
                # Simple exponential smoothing
                alpha = model['alpha']
                values = data['cumulative_pnl'].values
                
                # Apply exponential smoothing
                smoothed = values[0]
                for value in values[1:]:
                    smoothed = alpha * value + (1 - alpha) * smoothed
                
                # Simple projection
                daily_trend = (values[-1] - values[-7]) / 7 if len(values) >= 7 else 0
                prediction = smoothed + (daily_trend * horizon_days)
            
            return float(prediction)
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            # Fallback to simple average
            return float(data['cumulative_pnl'].tail(7).mean())
    
    def _calculate_confidence_interval(self, data: pd.DataFrame, prediction: float, model_type: ForecastModel) -> Tuple[float, float]:
        """Calculate confidence intervals for the prediction."""
        try:
            # Calculate historical volatility
            pnl_std = data['pnl'].std()
            
            # Confidence interval based on historical volatility
            margin = 1.96 * pnl_std  # 95% confidence interval
            
            lower_bound = prediction - margin
            upper_bound = prediction + margin
            
            return float(lower_bound), float(upper_bound)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence interval: {e}")
            # Fallback to simple percentage bounds
            return float(prediction * 0.8), float(prediction * 1.2)
    
    def _assess_trend(self, data: pd.DataFrame) -> str:
        """Assess the trend direction of the time series."""
        try:
            recent_data = data['cumulative_pnl'].tail(10)
            
            if len(recent_data) < 2:
                return "FLAT"
            
            # Simple trend analysis
            trend_slope = (recent_data.iloc[-1] - recent_data.iloc[0]) / len(recent_data)
            
            if trend_slope > 0.1:
                return "UPWARD"
            elif trend_slope < -0.1:
                return "DOWNWARD"
            else:
                return "FLAT"
                
        except Exception as e:
            self.logger.error(f"Error assessing trend: {e}")
            return "FLAT"
    
    def _assess_risk(self, data: pd.DataFrame, prediction: float) -> str:
        """Assess the risk level of the forecast."""
        try:
            # Calculate volatility metrics
            pnl_volatility = data['pnl'].std()
            recent_volatility = data['pnl'].tail(10).std()
            
            # Risk assessment based on volatility
            if recent_volatility > pnl_volatility * 1.5:
                return "HIGH"
            elif recent_volatility < pnl_volatility * 0.5:
                return "LOW"
            else:
                return "MEDIUM"
                
        except Exception as e:
            self.logger.error(f"Error assessing risk: {e}")
            return "MEDIUM"
    
    def _calculate_model_accuracy(self, model, data: pd.DataFrame, model_type: ForecastModel) -> float:
        """Calculate model accuracy score."""
        try:
            if len(data) < 10:
                return 0.5  # Default accuracy for insufficient data
            
            # Simple accuracy based on recent prediction vs actual
            recent_data = data.tail(10)
            actual_values = recent_data['cumulative_pnl'].values
            
            if len(actual_values) < 2:
                return 0.5
            
            # Calculate mean absolute percentage error (MAPE)
            errors = []
            for i in range(1, len(actual_values)):
                predicted = actual_values[i-1]  # Simple persistence model
                actual = actual_values[i]
                if actual != 0:
                    error = abs((actual - predicted) / actual)
                    errors.append(error)
            
            if not errors:
                return 0.5
            
            mape = np.mean(errors)
            accuracy = max(0.0, 1.0 - mape)  # Convert MAPE to accuracy score
            
            return float(min(1.0, accuracy))
            
        except Exception as e:
            self.logger.error(f"Error calculating model accuracy: {e}")
            return 0.5
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get the current status of the forecasting engine."""
        return {
            "engine": "ProfitForecastingEngine",
            "status": "ACTIVE",
            "sklearn_available": SKLEARN_AVAILABLE,
            "cached_models": len(self.trained_models),
            "min_data_points": self.min_data_points,
            "available_models": [model.value for model in ForecastModel],
            "last_updated": datetime.utcnow().isoformat()
        }