import asyncio
import logging
import sys
import os
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime

# Setup lightweight logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("LightSim")

# Add project root to path
sys.path.append(os.getcwd())

# --- Mocks for Heavy Components ---

class MockConfigManager:
    def get_int(self, key, default): 
        if key == 'risk.max_concurrent_positions': return 3
        return default
    def get_str(self, key, default): return default
    def get_float(self, key, default): return default
    def get_dict(self, key, default): return default
    def get(self, key, default): return default

class MockRiskManager:
    async def initialize(self): pass
    def check_risk(self, signal): return True

class MockMessagingService:
    async def send_message(self, msg): pass

class MockIndicatorEngine:
    async def initialize(self): pass

class MockDataManager:
    async def initialize(self): pass

class MockSmartIndicatorExecutor:
    pass

# --- Import Core Logic Classes (The ones we want to test) ---
# We try to import the actual classes to test their real logic.
# If imports fail due to missing dependencies in this lightweight env, we'll mock them too,
# but the goal is to test the wiring of THESE specific classes.

try:
    from auj_platform.src.trading_engine.deal_monitoring_teams import DealMonitoringTeams
    from auj_platform.src.trading_engine.execution_handler import ExecutionHandler
    from auj_platform.src.coordination.genius_agent_coordinator import GeniusAgentCoordinator
    from auj_platform.src.core.data_contracts import TradeSignal, SignalType, SignalSource
except ImportError as e:
    logger.error(f"❌ Critical Import Error: {e}")
    logger.error("Ensure you are running this from the project root (e:\\AUG6)")
    sys.exit(1)

# --- Simulation Logic ---

async def run_lightweight_simulation():
    logger.info("🚀 Starting Lightweight Live Simulation (Low CPU Mode)")
    
    # 1. Setup Environment
    config = MockConfigManager()
    risk_manager = MockRiskManager()
    messaging = MockMessagingService()
    hierarchy = MagicMock()
    
    # 2. Initialize Components
    logger.info("⚙️  Initializing Core Components...")
    
    # Monitoring Team
    deal_monitoring = DealMonitoringTeams(
        performance_tracker=MagicMock(),
        risk_manager=risk_manager,
        hierarchy_manager=hierarchy
    )
    
    # Execution Handler (The Bridge)
    execution_handler = ExecutionHandler(
        config_manager=config,
        risk_manager=risk_manager,
        deal_monitoring_teams=deal_monitoring,
        messaging_service=messaging
    )
    
    # Coordinator (The Brain)
    coordinator = GeniusAgentCoordinator(
        config_manager=config,
        hierarchy_manager=hierarchy,
        indicator_engine=MockIndicatorEngine(),
        data_manager=MockDataManager(),
        risk_manager=risk_manager,
        execution_handler=execution_handler,
        deal_monitoring_teams=deal_monitoring,
        smart_indicator_executor=MockSmartIndicatorExecutor(),
        messaging_service=messaging
    )
    
    # Mock the actual analysis part of the coordinator to just return a signal
    # We want to test the *workflow* around it (capacity check, execution), not the heavy analysis.
    coordinator._run_agent_analysis = AsyncMock(return_value=TradeSignal(
        symbol="EURUSD",
        signal_type=SignalType.BUY,
        strength=0.85,
        source=SignalSource.COORDINATOR,
        timestamp=datetime.utcnow(),
        metadata={"reason": "Simulation Test"}
    ))
    
    # Mock ExecutionHandler's internal execution to simulate success without a broker
    async def mock_execute(order):
        logger.info(f"⚡ ExecutionHandler: Executing {order.side} on {order.symbol}...")
        await asyncio.sleep(0.1) # Simulate network delay
        
        # Manually register the position in monitoring to simulate a fill
        position_id = f"POS_{datetime.utcnow().timestamp()}"
        deal_monitoring.active_positions[position_id] = {
            "symbol": order.symbol,
            "entry_price": 1.1050,
            "volume": order.quantity
        }
        logger.info(f"✅ Trade Executed! Position ID: {position_id}")
        return True

    execution_handler.execute_order = AsyncMock(side_effect=mock_execute)

    # 3. Run Simulation Loop
    logger.info("\n🎬 Starting Simulation Loop (3 Cycles)...")
    
    for i in range(1, 4):
        logger.info(f"\n--- Cycle {i}/3 ---")
        
        # Check Capacity (The logic we fixed)
        active_count = len(execution_handler.active_positions)
        logger.info(f"📊 Current Active Positions: {active_count}")
        
        # Trigger Coordinator
        logger.info("🧠 Coordinator: Analyzing market...")
        
        # We manually call the method that contains the capacity check logic
        # But since we mocked _run_agent_analysis, we need to ensure execute_analysis_cycle calls it
        # OR we just test execute_analysis_cycle directly.
        
        # Let's rely on the actual execute_analysis_cycle logic. 
        # It calls self.execution_handler.active_positions.
        
        signal = await coordinator.execute_analysis_cycle("EURUSD")
        
        if signal:
            logger.info(f"💡 Signal Generated: {signal.signal_type} {signal.symbol}")
            # Simulate passing to execution (Coordinator usually does this via orchestrator, 
            # but here we simulate the orchestrator passing it)
            
            # Create a mock order from signal
            mock_order = MagicMock()
            mock_order.symbol = signal.symbol
            mock_order.side = "BUY"
            mock_order.quantity = 1.0
            
            await execution_handler.execute_order(mock_order)
        else:
            logger.info("zzz Coordinator skipped or produced no signal")
            
        await asyncio.sleep(0.1) # Breathe

    # 4. Final Report
    logger.info("\n🏁 Simulation Complete")
    final_count = len(deal_monitoring.active_positions)
    logger.info(f"📈 Final Active Positions: {final_count}")
    
    if final_count == 3:
        logger.info("✅ SUCCESS: System successfully opened and tracked 3 positions.")
    else:
        logger.warning(f"⚠️  PARTIAL: Expected 3 positions, got {final_count}.")

if __name__ == "__main__":
    asyncio.run(run_lightweight_simulation())
