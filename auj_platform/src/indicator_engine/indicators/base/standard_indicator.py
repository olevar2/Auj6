"""
Standard Indicator Interface - Complete Base Class for All Indicators
=====================================================================

This module defines the standardized interface that all 230 indicators must implement.
Provides validation, caching, performance monitoring, and error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time
import hashlib
import json

from ....core.exceptions import IndicatorCalculationException, ValidationException


class DataType(Enum):
    """Supported data types for indicators"""
    OHLCV = "ohlcv"
    TICK = "tick"
    NEWS = "news"
    ORDER_BOOK = "order_book"
    MARKET_DEPTH = "market_depth"


class SignalType(Enum):
    """Types of signals an indicator can generate"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NEUTRAL = "neutral"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class IndicatorResult:
    """
    Standardized result structure for all indicator calculations
    """
    indicator_name: str
    timestamp: datetime
    value: Union[float, int, Dict[str, Any]]
    signal: Optional[SignalType] = None
    confidence: float = 0.0  # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None
    calculation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "indicator_name": self.indicator_name,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "signal": self.signal.value if self.signal else None,
            "confidence": self.confidence,
            "metadata": self.metadata or {},
            "calculation_time_ms": self.calculation_time_ms
        }


@dataclass
class DataRequirement:
    """
    Defines what data an indicator needs
    """
    data_type: DataType
    required_columns: List[str]
    min_periods: int = 1
    lookback_periods: int = 100
    preprocessing: Optional[str] = None
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data meets requirements"""
        if data is None or data.empty:
            return False
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            raise ValidationException(
                validator="DataRequirement",
                field="columns",
                message=f"Missing required columns: {missing_columns}"
            )
        
        # Check minimum periods
        if len(data) < self.min_periods:
            raise ValidationException(
                validator="DataRequirement",
                field="min_periods",
                message=f"Insufficient data: {len(data)} < {self.min_periods}"
            )
        
        return True


class StandardIndicatorInterface(ABC):
    """
    Abstract base class that all indicators must implement.
    Provides caching, validation, performance monitoring, and standardized interface.
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.parameters = parameters or {}
        self.cache = {}
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        self.last_calculation_time = None
        self.is_initialized = False
        
        # Initialize the indicator
        self._initialize()
    
    def _initialize(self):
        """Initialize indicator-specific settings"""
        try:
            self.validate_parameters()
            self.is_initialized = True
        except Exception as e:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="initialization",
                message=f"Failed to initialize indicator: {str(e)}",
                cause=e
            )
    
    @abstractmethod
    def get_data_requirements(self) -> DataRequirement:
        """
        Define what data this indicator needs.
        Must be implemented by each indicator.
        """
        raise NotImplementedError("Subclasses must implement get_data_requirements()")
    
    @abstractmethod
    def calculate_raw(self, data: pd.DataFrame) -> Union[float, int, Dict[str, Any]]:
        """
        Perform the actual indicator calculation.
        Must be implemented by each indicator.
        """
        raise NotImplementedError("Subclasses must implement calculate_raw()")
    
    def validate_parameters(self) -> bool:
        """
        Validate indicator parameters.
        Override in subclasses for custom validation.
        """
        return True
    
    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate a unique cache key for the data"""
        data_hash = hashlib.md5(
            str(data.values.tobytes() + str(self.parameters)).encode()
        ).hexdigest()
        return f"{self.name}_{data_hash}"
    
    def _should_use_cache(self, cache_key: str) -> bool:
        """Determine if cache should be used"""
        return cache_key in self.cache and self.cache[cache_key]["timestamp"] > datetime.utcnow().timestamp() - 60
    
    def calculate(self, data: pd.DataFrame, use_cache: bool = True) -> IndicatorResult:
        """
        Main calculation method with caching, validation, and performance monitoring.
        """
        if not self.is_initialized:
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="validation",
                message="Indicator not properly initialized"
            )
        
        start_time = time.time()
        
        try:
            # Validate data requirements
            requirements = self.get_data_requirements()
            requirements.validate_data(data)
            
            # Check cache
            cache_key = self._generate_cache_key(data)
            if use_cache and self._should_use_cache(cache_key):
                cached_result = self.cache[cache_key]["result"]
                return cached_result
            
            # Perform calculation
            raw_value = self.calculate_raw(data)
            
            # Generate signal and confidence
            signal, confidence = self._generate_signal(raw_value, data)
            
            # Create result
            calculation_time = (time.time() - start_time) * 1000
            result = IndicatorResult(
                indicator_name=self.name,
                timestamp=datetime.utcnow(),
                value=raw_value,
                signal=signal,
                confidence=confidence,
                metadata=self._get_metadata(data),
                calculation_time_ms=calculation_time
            )
            
            # Update cache
            if use_cache:
                self.cache[cache_key] = {
                    "result": result,
                    "timestamp": datetime.utcnow().timestamp()
                }
            
            # Update performance metrics
            self.calculation_count += 1
            self.total_calculation_time += calculation_time
            self.last_calculation_time = datetime.utcnow()
            
            return result
            
        except Exception as e:
            calculation_time = (time.time() - start_time) * 1000
            if isinstance(e, (IndicatorCalculationException, ValidationException)):
                raise
            else:
                raise IndicatorCalculationException(
                    indicator_name=self.name,
                    calculation_step="calculation",
                    message=str(e),
                    cause=e,
                    context={"calculation_time_ms": calculation_time}
                )
    
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signal based on indicator value.
        Override in subclasses for custom signal logic.
        """
        return None, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get additional metadata for the calculation.
        Override in subclasses for custom metadata.
        """
        return {
            "data_points": len(data),
            "parameters": self.parameters,
            "calculation_count": self.calculation_count,
            "avg_calculation_time_ms": self.total_calculation_time / max(self.calculation_count, 1)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this indicator"""
        return {
            "name": self.name,
            "calculation_count": self.calculation_count,
            "total_calculation_time_ms": self.total_calculation_time,
            "avg_calculation_time_ms": self.total_calculation_time / max(self.calculation_count, 1),
            "last_calculation_time": self.last_calculation_time.isoformat() if self.last_calculation_time else None,
            "cache_size": len(self.cache),
            "is_initialized": self.is_initialized
        }
    
    def clear_cache(self):
        """Clear the indicator cache"""
        self.cache.clear()
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        self.last_calculation_time = None
    
    def __str__(self) -> str:
        return f"{self.name}({self.parameters})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"