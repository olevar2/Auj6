"""
Base Indicator Class
===================

Provides the base class and configuration for indicators, bridging the gap
between legacy implementations and the StandardIndicatorInterface.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from .standard_indicator import StandardIndicatorInterface, SignalType

@dataclass
class IndicatorConfig:
    """Base configuration for indicators"""
    min_periods: int = 14
    
class BaseIndicator(StandardIndicatorInterface):
    """
    Base indicator class that provides common functionality used by
    many specific indicator implementations.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        """Initialize with configuration"""
        self.config = config or IndicatorConfig()
        name = self.__class__.__name__
        super().__init__(name)
        self.logger = logging.getLogger(name)
        
    def get_data_requirements(self):
        """
        Default data requirements. 
        Subclasses should override this if they need specific columns.
        """
        # This is a placeholder to satisfy abstract method if not overridden
        from .standard_indicator import DataRequirement, DataType
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['close', 'volume'],
            min_periods=self.config.min_periods
        )

    def calculate_raw(self, data: pd.DataFrame) -> Any:
        """
        Placeholder for raw calculation.
        Most subclasses override calculate() directly, bypassing this.
        """
        return 0.0

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data dictionary"""
        if not data:
            return False
        
        # Check for essential keys
        required = ['high', 'low', 'close', 'volume']
        if not all(key in data for key in required):
            # Try to handle if it's a DataFrame passed as dict
            if isinstance(data, pd.DataFrame):
                return all(col in data.columns for col in required)
            return False
            
        return True

    def _create_default_result(self) -> Dict[str, Any]:
        """Create a default/empty result dictionary"""
        return {
            "value": 0.0,
            "signal": SignalType.NEUTRAL,
            "confidence": 0.0,
            "metadata": {
                "status": "insufficient_data"
            }
        }

    def _create_error_result(self, message: str) -> Dict[str, Any]:
        """Create an error result dictionary"""
        return {
            "value": 0.0,
            "signal": SignalType.NEUTRAL,
            "confidence": 0.0,
            "error": message,
            "metadata": {
                "status": "error",
                "error_message": message
            }
        }

    def _format_result(self, result: Any, signal: Any) -> Dict[str, Any]:
        """Format the final result dictionary"""
        # Extract confidence if available in result object
        confidence = getattr(result, "confidence_score", 0.0)
        
        return {
            "value": result,
            "signal": signal,
            "confidence": confidence,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    def _update_history(self, data: Any, result: Any):
        """Update internal history (placeholder)"""
        pass
