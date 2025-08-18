"""
Trading Engine - Main trading engine interface for AUJ Platform
"""

from typing import Dict, List, Any, Optional
from .execution_handler import ExecutionHandler
from .dynamic_risk_manager import DynamicRiskManager

class TradingEngine:
    """
    Main trading engine that coordinates execution and risk management.
    """
    
    def __init__(self):
        self.execution_handler = None
        self.risk_manager = None
        self.running = False
        
    async def initialize(self):
        """Initialize the trading engine"""
        self.execution_handler = ExecutionHandler()
        self.risk_manager = DynamicRiskManager()
        
    async def start(self):
        """Start the trading engine"""
        if self.execution_handler:
            await self.execution_handler.start()
        if self.risk_manager:
            await self.risk_manager.start()
        self.running = True
        
    async def stop(self):
        """Stop the trading engine"""
        if self.execution_handler:
            await self.execution_handler.stop()
        if self.risk_manager:
            await self.risk_manager.stop()
        self.running = False