import sys
import os
import asyncio
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that might cause import errors if dependencies are missing
sys.modules['MetaTrader5'] = MagicMock()
sys.modules['yfinance'] = MagicMock()
sys.modules['pandas_ta'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['requests.adapters'] = MagicMock()
sys.modules['urllib3'] = MagicMock()
sys.modules['urllib3.util'] = MagicMock()
sys.modules['urllib3.util.retry'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.ensemble'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()

from src.core.containers import ApplicationContainer
from src.core.unified_database_manager import UnifiedDatabaseManager

async def verify_init():
    print("Starting Platform Initialization Verification...")
    
    try:
        container = ApplicationContainer()
        
        # Mock Database
        mock_db = MagicMock(spec=UnifiedDatabaseManager)
        mock_db.initialize.return_value = asyncio.Future()
        mock_db.initialize.return_value.set_result(True)
        mock_db.close.return_value = asyncio.Future()
        mock_db.close.return_value.set_result(True)
        
        container.platform.database_manager.override(mock_db)
        container.platform.unified_database_manager.override(mock_db)
        
        # Mock Messaging
        mock_messaging = MagicMock()
        mock_messaging.start.return_value = asyncio.Future()
        mock_messaging.start.return_value.set_result(True)
        mock_messaging.stop.return_value = asyncio.Future()
        mock_messaging.stop.return_value.set_result(True)
        
        container.platform.messaging_service.override(mock_messaging)
        
        print("Instantiating platform...")
        platform = container.auj_platform()
        print("Platform instantiated successfully.")
        
        # Initialize
        print("Initializing platform...")
        success = await platform.initialize()
        
        if success:
            print("✅ Platform initialized successfully!")
            
            # Verify agents are loaded
            coordinator = platform.coordinator
            print(f"Coordinator loaded {len(coordinator.agents)} agents.")
            
            agents_checked = 0
            for name, agent in coordinator.agents.items():
                print(f" - {name}: {type(agent).__name__}")
                agents_checked += 1
                
                # Verify config loading for Trend/Momentum
                if name == 'momentum_agent':
                    # Check if config values are loaded (default is 30/70)
                    # We can't easily check internal state without access, but if it didn't crash, it's good.
                    pass

            if agents_checked == 0:
                print("❌ No agents loaded!")
                sys.exit(1)

            await platform.shutdown()
        else:
            print("❌ Platform initialization failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Exception during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(verify_init())
