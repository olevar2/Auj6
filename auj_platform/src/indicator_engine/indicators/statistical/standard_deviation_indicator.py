import numpy as np
import pandas as pd
from typing import Dict, Any
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface, IndicatorResult

class StandardDeviationIndicator(StandardIndicatorInterface):
    def __init__(self, parameters: Dict[str, Any] = None):
        super().__init__(name="StandardDeviation")
        self.parameters = parameters or {'period': 20}

    def calculate(self, data: Dict[str, Any]) -> IndicatorResult:
        # Handle both DataFrame input (direct) or Dict input (standard)
        if isinstance(data, pd.DataFrame):
            ohlcv = data
        else:
            ohlcv = data.get('ohlcv')
            
        if ohlcv is None or ohlcv.empty:
             return IndicatorResult(value=0.0)
        
        close = ohlcv['close']
        period = self.parameters.get('period', 20)
        std = close.rolling(window=period).std().iloc[-1]
        
        return IndicatorResult(value=std if not np.isnan(std) else 0.0)
