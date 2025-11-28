from abc import abstractmethod
from typing import Dict, Any, List
import pandas as pd
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface, IndicatorResult, IndicatorCalculationError

class BasePatternIndicator(StandardIndicatorInterface):
    """Base class for pattern recognition indicators."""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        super().__init__(name=name)
        self.parameters = parameters or {}

    @abstractmethod
    def calculate(self, data: Dict[str, Any]) -> IndicatorResult:
        pass
