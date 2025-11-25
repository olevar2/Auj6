import asyncio
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.event_bus import event_bus, Event, EventType
from src.coordination.genius_agent_coordinator import GeniusAgentCoordinator
from src.core.unified_config import UnifiedConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_flow():
    logger.info("Starting Event-Driven Flow Verification")

    # 1. Mock Config Manager
    config_manager = UnifiedConfigManager()
    
    # 2. Mock Dependencies (we can pass None for most as we are testing the coordination layer)
    # We need a mock data manager to avoid actual data fetching
    class MockDataManager:
        async def get_data(self, *args, **kwargs):
            return None # Return None to simulate no data, but avoid crash
    
    class MockRiskManager:
        pass
        
    class MockExecutionHandler:
        pass
        
    class MockHierarchyManager:
        async def get_current_rankings(self):
            print("DEBUG: get_current_rankings called")
            from src.core.data_contracts import AgentRank
            return {
                'risk_genius': AgentRank.ALPHA,
                'momentum_agent': AgentRank.BETA,
                'trend_agent': AgentRank.BETA,
                'execution_expert': AgentRank.GAMMA,
                'microstructure_agent': AgentRank.GAMMA
            }
        
    class MockIndicatorEngine:
        pass

    # Mock Agent
    class MockAgent:
        def __init__(self, name):
            self.name = name
            
        def get_required_data_types(self):
            return []
            
        def get_assigned_indicators(self):
            return []
            
        def get_recent_performance(self, days=7):
            return {'win_rate': 0.6, 'profit_factor': 1.5}
            
        def is_ready_for_analysis(self):
            print(f"DEBUG: is_ready_for_analysis called for {self.name}")
            return True
            
        async def perform_analysis(self, *args, **kwargs):
            print(f"DEBUG: perform_analysis called for {self.name}")
            from src.agents.base_agent import AnalysisResult
            
            return AnalysisResult(
                agent_name=self.name,
                symbol="EURUSD",
                decision="BUY",
                confidence=0.95,
                reasoning="Mock reasoning",
                indicators_used=["MockIndicator"],
                technical_analysis={"trend": "UP"},
                risk_assessment={"risk_level": "LOW"}
            )

    # 3. Initialize Coordinator
    coordinator = GeniusAgentCoordinator(
        config_manager=config_manager,
        hierarchy_manager=MockHierarchyManager(),
        indicator_engine=MockIndicatorEngine(),
        data_manager=MockDataManager(),
        risk_manager=MockRiskManager(),
        execution_handler=MockExecutionHandler()
    )
    
    # Inject Mock Agents
    from datetime import datetime
    coordinator.agents = {
        'risk_genius': MockAgent('risk_genius'),
        'momentum_agent': MockAgent('momentum_agent'),
        'trend_agent': MockAgent('trend_agent'),
        'execution_expert': MockAgent('execution_expert'),
        'microstructure_agent': MockAgent('microstructure_agent')
    }
    
    # 4. Start Coordinator (subscribes to events)
    await coordinator.start()
    
    # 5. Simulate Market Data Event
    test_symbol = "EURUSD"
    event_payload = {"symbol": test_symbol, "price": 1.1050, "timestamp": "2023-10-27T10:00:00Z"}
    event = Event(type=EventType.MARKET_DATA_UPDATE, payload=event_payload, source="TestScript")
    
    logger.info(f"Publishing event: {event}")
    await event_bus.publish(event)
    
    # 6. Wait for processing (Coordinator runs analysis in background task)
    logger.info("Waiting for analysis cycle to start...")
    await asyncio.sleep(2) # Give it a moment to pick up
    
    # 7. Verify Cycle Started
    if coordinator.current_cycle:
        logger.info(f"✅ Cycle started! ID: {coordinator.current_cycle.cycle_id}")
        logger.info(f"Trigger Event: {coordinator.current_cycle.trigger_event}")
        
        if coordinator.current_cycle.symbol == test_symbol:
             logger.info(f"✅ Symbol matches: {test_symbol}")
        else:
             logger.error(f"❌ Symbol mismatch: {coordinator.current_cycle.symbol}")
             
    else:
        # It might have finished already if it failed fast (due to mock data manager returning None)
        # Check history
        if coordinator.cycle_history:
             logger.info(f"✅ Cycle completed and stored in history! Last ID: {coordinator.cycle_history[-1].cycle_id}")
        else:
             logger.error("❌ No cycle started or completed.")

    # 8. Cleanup
    await coordinator.stop()
    logger.info("Verification Complete")

if __name__ == "__main__":
    asyncio.run(verify_flow())
